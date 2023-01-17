import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as stats
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
# Create Data audit Report for continuous variables

def continuous_var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  

                      x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),

                          x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), 

                              x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 

                  index = ['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1', 

                               'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])
# Create Data audit Report for categorical variables

def categorical_var_summary(x):

    Mode = x.value_counts().sort_values(ascending = False)[0:1].reset_index()

    return pd.Series([x.count(), x.isnull().sum(), Mode.iloc[0, 0], Mode.iloc[0, 1], 

                          round(Mode.iloc[0, 1] * 100/x.count(), 2)], 

                  index = ['N', 'NMISS', 'MODE', 'FREQ', 'PERCENT'])
# Missing value imputation for categorical and continuous variables

def missing_imputation(x, stats = 'mean'):

    if (x.dtypes == 'float64') | (x.dtypes == 'int64'):

        x = x.fillna(x.mean()) if stats == 'mean' else x.fillna(x.median())

    else:

        x = x.fillna(x.mode())

    return x
# An utility function to create dummy variable

def create_dummies(df, colname):

    col_dummies = pd.get_dummies(df[colname], prefix = colname, drop_first = True)

    df = pd.concat([df, col_dummies], axis = 1)

    df.drop(colname, axis = 1, inplace = True )

    return df
df_campus = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df_campus.info()
df_campus.drop(columns='sl_no',inplace=True)
df_campus.info()
### Imputing 0 in salary column for candidates whose status is not_placed
df_campus.salary = np.where(df_campus.status=='Not Placed',0,df_campus.salary)
df_campus_cont=df_campus[['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary']]
df_campus_cat=df_campus.loc[:,df_campus.columns.difference(['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary'])]
df_campus_cont.apply(continuous_var_summary)
df_campus_cat.apply(categorical_var_summary)
def cat_count (x):

    sns.countplot(x)

    plt.show()
df_campus_cat.apply(cat_count)
def box_cont(x):

    sns.boxplot(x)

    plt.show()
df_campus_cont.apply(box_cont)
### Outlier Treatement
df_campus_cont = df_campus_cont.apply(lambda x: x.clip(lower = x.dropna().quantile(0.01), upper = x.dropna().quantile(0.99)))
df_campus_cont.apply(box_cont)
import seaborn as sns

plt.figure(figsize = (10,10))

sns.heatmap(df_campus_cont.corr())

plt.show()
# for c_feature in categorical_features

for c_feature in list(df_campus_cat.columns):

    df_campus_cat[c_feature] = df_campus_cat[c_feature].astype('category')

    df_campus_cat = create_dummies(df_campus_cat, c_feature)
df_campus_new=pd.concat([df_campus_cont,df_campus_cat],axis = 1)
df_campus_new.info()
df_campus_new.status_Placed.value_counts()
# F_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices
df_campus_new.rename(columns={'specialisation_Mkt&HR':'specialisation_Mkt_HR'},inplace=True)
df_campus_new.rename(columns={'degree_t_Sci&Tech':'degree_t_Sci_Tech'},inplace=True)
# get all the independant variables for model creation

model_param = 'status_Placed ~ ' + ' + '.join(list(df_campus_new.columns.difference(['status_Placed'])))
model_param
model_param='''status_Placed ~ degree_p + degree_t_Others + degree_t_Sci_Tech + etest_p + gender_M + 

                hsc_b_Others + hsc_p + hsc_s_Commerce + mba_p + 

            salary + specialisation_Mkt_HR + ssc_b_Others + ssc_p + workex_Yes'''
# separate the Y and X variables

y, X = dmatrices(model_param,df_campus_new, return_type = 'dataframe')
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif['Features'] = X.columns

vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



# display the output

vif.round(1)
# Train and Test split

train, test = train_test_split(df_campus_new, test_size = 0.3, random_state =  123)
model_param='''status_Placed ~ degree_p + degree_t_Others + degree_t_Sci_Tech + etest_p + gender_M + 

                hsc_b_Others + hsc_p + hsc_s_Commerce + mba_p + specialisation_Mkt_HR + ssc_b_Others + ssc_p + workex_Yes'''
train
m1 = smf.logit(formula = model_param, data = train).fit()
print(m1.summary())
p = m1.predict(train)
# empty dataframe

somersd_df = pd.DataFrame()



# iterate for each of the X - dependant variables and get the Somer's D value

for num_variable in train.columns.difference(['status_Placed','salary']):

    

    # execute teh logit model

    logreg = smf.logit(formula = str('status_Placed ~ ') + str(num_variable), data = train).fit()

    

    # get the predicted probabilities and create a dataframe with the predicted values

    y_score = pd.DataFrame(logreg.predict())

    

    # name the column of the df as score

    y_score.columns = ['Score']

    

    # calculate the Somer's D values

    somers_d = 2 * metrics.roc_auc_score(train['status_Placed'], y_score) - 1

    

    # place the variable name and the Somers' D score in a temp dataframe

    temp = pd.DataFrame([num_variable, somers_d]).T

    temp.columns = ['Variable Name', 'SomersD']

    

    # append the data in the dataframe for all the X variables

    somersd_df = pd.concat([somersd_df, temp], axis=0)
somersd_df['Var_Sig']=np.where(somersd_df.SomersD < 0.2 , 'Insig','Sig')
### Yes, the percentage matter in getting placed. Please check the below significant variables after Sommer's D
somersd_df
model_param='''status_Placed ~ degree_p +hsc_p+ specialisation_Mkt_HR + ssc_p + workex_Yes'''
m1 = smf.logit(formula = model_param, data = train).fit()
print(m1.summary())
train_predict = m1.predict(train)
test_predict = m1.predict(test)
df_campus_new.status_Placed.mean()
metrics.accuracy_score( train['status_Placed'],

                            train_predict>0.688)
# model performance | Method 1: AUC

train_auc = metrics.roc_auc_score(train['status_Placed'], train_predict)

test_auc = metrics.roc_auc_score(test['status_Placed'], test_predict)



print("The AUC for the model built on the Train Data is : ", train_auc)

print("The AUC for the model built on the Test Data is : ", test_auc)
feature_col=list(['degree_p','degree_t_Others','degree_t_Sci_Tech','etest_p','gender_M','hsc_b_Others','hsc_p','hsc_s_Commerce','mba_p','salary','specialisation_Mkt_HR','ssc_b_Others','ssc_p','workex_Yes'])
## Using Random Forest
from sklearn.model_selection import train_test_split





train_X, test_X, train_y, test_y = train_test_split( df_campus_new[feature_col],

                                                  df_campus_new['status_Placed'],

                                                  test_size = 0.3,

                                                  random_state = 555 )
#!pip install imblearn

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=123)



train_X_os, train_y_os = ros.fit_sample(train_X, train_y)



unique_elements, counts_elements = np.unique(train_y_os, return_counts=True)

print("Frequency of unique values of the said array:")

print(np.asarray((unique_elements, counts_elements)))
from sklearn.ensemble import RandomForestClassifier

import sklearn.tree as dt

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier, export_graphviz, export

from sklearn.model_selection import GridSearchCV
pargrid_rf = {'n_estimators': np.arange(50,60,70),

                  'max_features': np.arange(5,8)}



#from sklearn.grid_search import GridSearchCV

gscv_rf = GridSearchCV(estimator=RandomForestClassifier(), 

                        param_grid=pargrid_rf, 

                        cv=5,

                        verbose=True, n_jobs=-1)



gscv_results = gscv_rf.fit(train_X_os, train_y_os)
gscv_results.best_params_
gscv_rf.best_score_
radm_clf = RandomForestClassifier(oob_score=True,n_estimators=50, max_features=5, n_jobs=-1)

radm_clf.fit( train_X, train_y )
radm_train_pred = pd.DataFrame( { 'actual':  train_y,

                            'predicted': radm_clf.predict( train_X ) } )
radm_test_pred = pd.DataFrame( { 'actual':  test_y,

                            'predicted': radm_clf.predict( test_X ) } )
print(metrics.accuracy_score( radm_test_pred.actual, radm_test_pred.predicted ))
feat_imp=list(zip(train_X.columns,radm_clf.feature_importances_))
feat_imp
df_feat_imp=pd.DataFrame(feat_imp)
df_feat_imp.columns=['Features','Importance']
### Top 5 features are selected using Logistic Regression 

#degree_p 

#hsc_p 

#specialisation_Mkt_HR

#ssc_p

#workex_Yes

### Top 5 features are selected using Random Forest

#degree_p 

#hsc_p 

#salary

#ssc_p

#mba_p
#Top 5 features

df_feat_imp.sort_values(by='Importance',ascending = False).head(5)
### Specialisation in Marketing and Finance is in more demand in the Corporate
df_campus[['specialisation','status']]
df_campus.specialisation.value_counts()
df_campus.loc[(df_campus.specialisation == 'Mkt&Fin') & (df_campus.status=='Placed')].status.count()
df_campus.loc[(df_campus.specialisation == 'Mkt&HR') & (df_campus.status=='Placed')].status.count()
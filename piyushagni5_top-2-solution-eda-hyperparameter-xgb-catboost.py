# Import Dependencies
%matplotlib inline

# Start Python Imports
import math, time, random, datetime
from random import shuffle

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, MinMaxScaler

# Machine learning
from sklearn.metrics import roc_auc_score
import catboost
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, CatBoostRegressor,cv
import xgboost
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import SVR



# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/summeranalytics2020/train.csv')
df.shape
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

#df.isnull().sum()
test_df=pd.read_csv('../input/summeranalytics2020/test.csv')
print(test_df.shape)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

test_df.head()
sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False)
### concatenating training and testing dataset horizontally
df_combine=pd.concat([df,test_df],axis=0, sort = False,ignore_index = True)
df_combine.shape
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df_combine.head()
nominal_catg_col = list(df_combine.select_dtypes(['object']).columns)
print(nominal_catg_col)
ordinal_catg_col = ["Education", "EnvironmentSatisfaction", "JobInvolvement","JobSatisfaction",
                    "PerformanceRating","StockOptionLevel","CommunicationSkill", "Behaviour" ]
print(ordinal_catg_col)
# Since Gender and overtime feature have only 2 labels hence using "drop_first=True".

df_BusinessTravel_one_hot = pd.get_dummies(df_combine['BusinessTravel'], prefix='BusinessTravel')

df_Department_one_hot = pd.get_dummies(df_combine['Department'], prefix='Department')

df_EducationField_one_hot = pd.get_dummies(df_combine['EducationField'], prefix='EducationField')

df_Gender_one_hot = pd.get_dummies(df_combine['Gender'], prefix='Gender',drop_first=True)

df_JobRole_one_hot = pd.get_dummies(df_combine['JobRole'], prefix='JobRole')

df_MaritalStatus_one_hot = pd.get_dummies(df_combine['MaritalStatus'], prefix='MaritalStatus')

df_OverTime_one_hot = pd.get_dummies(df_combine['OverTime'], prefix='OverTime',drop_first=True)
# Combine the one hot encoded columns with df_con_enc
df_nominal_catg = pd.concat([df_BusinessTravel_one_hot, df_Department_one_hot, df_EducationField_one_hot,
                             df_Gender_one_hot, df_JobRole_one_hot, df_MaritalStatus_one_hot,
                             df_OverTime_one_hot], axis=1)

print(df_nominal_catg.shape)
df_nominal_catg.head()
## Creating ordinal dataframe

df_ordinal_catg = df_combine[ordinal_catg_col]
print(df_ordinal_catg.shape)
df_ordinal_catg.head()
final_catg_col = nominal_catg_col + ordinal_catg_col + ["Attrition"]

df_numeric = df_combine.drop( final_catg_col , axis=1)
df_numeric.shape
# As we separated categorical data from the numeric one, now we can do feature scaling

numeric_col = list(df_numeric.columns)

scaler = MinMaxScaler(feature_range=(0, 1))
df_numeric = scaler.fit_transform(df_numeric)
df_numeric = pd.DataFrame(df_numeric, columns= numeric_col)
print(df_numeric.shape)
df_numeric.head()
print(df_numeric.shape)
print(df_ordinal_catg.shape)
print(df_nominal_catg.shape)
df_pre_process = pd.concat([df_numeric, df_ordinal_catg.reset_index(drop=True),
                           df_nominal_catg.reset_index(drop=True)],axis=1,sort = False)

print(df_pre_process.shape)
df_pre_process.head()
X_train_pre = df_pre_process.iloc[:1628,:]
X_test_pre = df_pre_process.iloc[1628:,:]
y_train = df['Attrition']

print(X_train_pre.shape)
print(y_train.shape)
print(X_test_pre.shape)
def feat_to_binning(str, bin):

    df_bin = pd.DataFrame()
    df_bin[str] = df_combine[str]

    fig, ax = plt.subplots()
    df_bin[str].hist(bins = bin, color='#A9C5D3', edgecolor='black',  
                              grid=False)
    #ax.set_title('Whole dataset {} Histogram'.format{str}, fontsize=12)
    ax.set_xlabel(str, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    df_bin[str + '_bin_round'] = np.array(np.int64(
                              np.array(df_bin[str]) / 10.))
    return df_bin

df_age = feat_to_binning("Age",5)

print(df_age['Age_bin_round'].value_counts())

df_age.head()
X_train_pre['Age'] = df_age.loc[:1628,'Age_bin_round']
X_test_pre['Age'] = df_age.loc[1628:,'Age_bin_round']

df.describe()
X_train_pre.describe()
num_col = numeric_col
num_col.remove('Age')

plt.figure(figsize=(20,10))
sns.boxplot(data= X_train_pre.loc[:,num_col])

# Rotate x-labels
plt.xticks(rotation=-45)
plt.figure(figsize=(20,10))
sns.boxplot(data= X_test_pre.loc[:,num_col])

# Rotate x-labels
plt.xticks(rotation=-45)
ord_col = ordinal_catg_col + ['Age']

plt.figure(figsize=(20,10))
sns.boxplot(data= X_train_pre.loc[:,ord_col])

# Rotate x-labels
plt.xticks(rotation=-45)
plt.figure(figsize=(20,10))
sns.boxplot(data= X_test_pre.loc[:,ord_col])

# Rotate x-labels
plt.xticks(rotation=-45)
# Let's view the distribution of Attrition
plt.figure(figsize=(10, 2))
sns.countplot(y="Attrition", data=df);
def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df: 
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)                                      # 1 row, 2 col., 1 imamge
        sns.countplot(y=target_column, data=bin_df);         # here data is not the args one, its inbuilt countplot arg.
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Attrition-1"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Attrition-0"});
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Attrition-1"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Attrition-0"});
df_bin = pd.DataFrame() # for discretised continuous variables
df_con = pd.DataFrame() # for continuous variables
# Add Education to subset dataframes
df_bin['Education'] = df['Education']

# Visualise the counts of Education and the distribution of the values against Attrition class 
plot_count_dist(df, 
                bin_df=df_bin, 
                label_column='Attrition', 
                target_column='Education', 
                figsize=(12, 8))

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif

X = X_train_pre.iloc[:,:21]
y= y_train

print(X.shape)
print(y.shape)
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=21)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores                # higher is the score, more important the feature is 
print(featureScores.nlargest(21,'Score'))  
#apply SelectKBest class to extract top 10 best features
bestfeatures_f = SelectKBest(score_func=f_classif, k=39)
fit_f = bestfeatures.fit(X,y)
dfscores_f = pd.DataFrame(fit_f.scores_)
dfcolumns_f = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores_f = pd.concat([dfcolumns_f,dfscores_f],axis=1)
featureScores_f.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores_f.nlargest(39,'Score')) 
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
#model = ExtraTreesClassifier()
#model = xgboost.XGBClassifier()

X_imp = X_train_pre
y_imp = y_train

model = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1.0, gamma=0.5, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.03, max_delta_step=0, max_depth=100,
              min_child_weight=1, missing=None, monotone_constraints='()',
              n_estimators=500, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=0.8,
              tree_method='exact', validate_parameters=1, verbosity=None)
model.fit(X_imp,y_imp)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
plt.figure(figsize=(20,20))

feat_importances = pd.Series(model.feature_importances_, index=X_imp.columns)
feat_importances.nlargest(X_imp.shape[1]).plot(kind='barh')
plt.show()
import seaborn as sns

attr = y
df_corr = pd.concat([X, y], axis =1 )

corrmat = df_corr.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(30,30))
#plot heat map
g=sns.heatmap(df_corr[top_corr_features].corr(),annot=True,cmap="RdYlGn")
pearson_corr_col = nominal_catg_col + ['Attrition']

df_corr =df[pearson_corr_col]
corrmat = df[pearson_corr_col].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(corrmat,annot=True,cmap="RdYlGn")
X_train = X_train_pre.drop(['Id','Behaviour','PerformanceRating',  'PercentSalaryHike',
                            'YearsAtCompany','TotalWorkingYears', 'EmployeeNumber', 
                            'YearsSinceLastPromotion'], axis=1)
X_test = X_test_pre.drop(['Id','Behaviour','PerformanceRating',  'PercentSalaryHike',
                          'YearsAtCompany','TotalWorkingYears', 'EmployeeNumber', 
                          'YearsSinceLastPromotion'], axis=1)

print(X_train.shape)
print(X_test.shape)

X_train_pre_cat = df_combine.iloc[:1628,:]
X_train_pre_cat = X_train_pre_cat.drop(['Attrition'], axis =1 )


X_test_pre_cat = df_combine.iloc[1628:,:]
X_test_pre_cat = X_test_pre_cat.drop(['Attrition'], axis =1 )

X_train_pre_cat['Age'] = df_age.loc[:1628,'Age_bin_round']
X_test_pre_cat['Age'] = df_age.loc[1628:,'Age_bin_round']


print(X_train_pre_cat.shape)
print(X_test_pre_cat.shape)

X_train_cat = X_train_pre_cat.drop(['Id','Behaviour','PerformanceRating',  'PercentSalaryHike',
                            'YearsAtCompany','TotalWorkingYears', 'EmployeeNumber', 
                            'YearsSinceLastPromotion'], axis=1)
X_test_cat = X_test_pre_cat.drop(['Id','Behaviour','PerformanceRating',  'PercentSalaryHike',
                          'YearsAtCompany','TotalWorkingYears', 'EmployeeNumber', 
                          'YearsSinceLastPromotion'], axis=1)

print(X_train_cat.shape)
print(X_test_cat.shape)

all_catg_col = ['Age'] + ordinal_catg_col + nominal_catg_col 

indices_cat = []
for col in all_catg_col:
    if (col in list(X_train_cat.columns)):
        indices_cat.append(X_train_cat.columns.get_loc(col))

indices_cat.sort()
print(indices_cat)

train_pool_cat = Pool(X_train_cat, 
                  y_train,
                  indices_cat)
### RANDOMIZED SEARCH

model_cat = CatBoostClassifier(random_state = 51,eval_metric = 'AUC')

random_grid_cat = {'learning_rate': [0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
        'depth': [4, 6,10,15,20,30,40,50,60,70],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}

randomized_search_cat = model_cat.randomized_search(random_grid_cat, train_pool_cat, cv=5,plot=True)
                    
# it will print the optimize parameter you get after randomized search...
randomized_search_cat
""" Use the value of parameters you get after performing randomized search """

#model_train = CatBoostClassifier(iterations=100, depth =10, learning_rate=0.1, random_state = 51,
#                                 loss_function ='CrossEntropy', custom_loss = ['Accuracy'], eval_metrics = 'AUC')                              

# These fixed values of hyperparameters got overfitted in the private leaderboard.
model_train = CatBoostClassifier(iterations=170, learning_rate=0.1, random_state = 51,eval_metric = 'AUC',loss_function ='CrossEntropy')                              

model_train.fit(train_pool_cat) #, plot=True)

# CatBoost accuracy
acc_catboost = round(model_train.score(X_train_cat, y_train) * 100, 2)
# Set params for cross-validation as same as initial model
cv_params = model_train.get_params()

# Run the cross-validation for 10-folds (same as the other models)
cv_data = cv(train_pool_cat,
             cv_params,
             fold_count=5,
             plot=True)


# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score
acc_cv_catboost = round(np.max(cv_data['test-AUC-mean']) * 100, 2)

print('train accuracy: ' , acc_catboost)
print("CV Accuracy: " ,acc_cv_catboost)

cv_data.head()
y_pred=model_train.predict_proba(X_test_cat)

pred=pd.DataFrame(y_pred[:,1])
sub_df=pd.read_csv('../input/summeranalytics2020/Sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','Attrition']
datasets.to_csv('Catboost_submission_temp.csv',index=False)
xbg_classifier = xgboost.XGBClassifier(scoring = 'roc_auc', random_state = 51)

booster=['gbtree']    

## Hyper Parameter Optimization

hyperparameter_grid = {
        'n_estimators' : [100, 500, 900, 1100, 1500],      # no. of decision trees used
        'min_child_weight': [1, 2, 3, 5, 7, 9],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        'learning_rate' : [0.005,0.01,0.03, 0.05, 0.15,0.3, 0.45, 0.55]
        }

# Set up the random search with 4-fold cross validation
folds = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 51)

random_cv = RandomizedSearchCV(estimator=xbg_classifier,
            param_distributions=hyperparameter_grid,
            cv=skf.split(X_train,y_train), n_iter=50,
            scoring = 'roc_auc',n_jobs = 4,
            verbose = 5, 
            return_train_score = True)
random_cv.fit(X_train, y_train)
classifier_t = random_cv.best_estimator_
print(classifier_t)
classifier_t.fit(X_train,y_train)     

acc_xgb = round(classifier_t.score(X_train, y_train) * 100, 2)

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

train_pred = model_selection.cross_val_predict(classifier_t, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=5, 
                                                  n_jobs = -1)


# Cross-validation accuracy metric
acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

print(acc_xgb)
print(acc_cv)
y_pred=classifier_t.predict_proba(X_test)

pred=pd.DataFrame(y_pred[:,1])
sub_df=pd.read_csv('../input/summeranalytics2020/Sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','Attrition']
datasets.to_csv('XBGC_random_submission_temp.csv',index=False)
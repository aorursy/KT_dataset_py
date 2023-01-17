import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_excel('../input/flipr-hiring-challenge/Train_dataset.xlsx')

df_test = pd.read_excel('../input/flipr-hiring-challenge/Test_dataset.xlsx')

df_train.head(14)
df_train.describe()
df_train.info()
df_train.isna().sum()
for col in df_train.columns:

    print("col-name: ", col, " | no_of_unique_values: ", df_train[col].nunique(dropna=True))
df_train = df_train.drop(['Designation', 'Name'], axis = 1)

df_train.head(10)
# import pandas_profiling

# df_train.profile_report()
df_train.Infect_Prob[(df_train['Infect_Prob']>45.0) & (df_train['Infect_Prob']<55.0)].count()
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder



lb = LabelBinarizer()

df_train['Gender'] = lb.fit_transform(df_train.Gender)

df_train['Married'] = lb.fit_transform(df_train.Married)





df_train.head(5)
import missingno as msno

msno.matrix(df_train)
msno.heatmap(df_train)
df_train.Children.isna().sum()
df_train.Married[(df_train['Married']==0.0) & (df_train['Children'].isna()==1)].count()
def impute_children(cols):

    children = cols[0]

    married = cols[1]

    

    if np.isnan(children):

        if married == 0.0:

            return 0

        else:

            return children

    else:

        return children



df_train['Children'] = df_train[['Children', 'Married']].apply(impute_children, axis=1)          

    
df_train.Children.isna().sum()
df_train.salary[df_train['Occupation'].isna()].head()
sns.boxplot(x = "Occupation", y = "salary", hue='Gender', data = df_train)
sns.countplot(x = "Occupation", hue = "Gender", data = df_train)
sns.countplot(x = "Occupation", data = df_train)
sns.boxplot(x='Occupation', y='salary', data=df_train)
sns.scatterplot(x='Occupation', y='salary', data=df_train)
sns.boxplot(x='Occupation', y='Infect_Prob', data=df_train)
sns.boxplot(x='Occupation', y='Charlson Index',hue='Gender', data=df_train)
sns.boxplot(x='Occupation', y='HDL cholesterol',hue='Gender', data=df_train)
df_train[df_train.Mode_transport.isna()].head()
sns.countplot(x='Mode_transport', data=df_train)
df_train['Mode_transport'] = df_train.Mode_transport.fillna('Public')

df_train.Mode_transport.isna().sum()
sns.countplot(x='comorbidity', data=df_train)
# sns.catplot(x="comorbidity", y='salary', hue="Gender", kind='swarm', data=df_train)
df_train['comorbidity'] = df_train.comorbidity.fillna('None')

df_train.comorbidity.isna().sum()
sns.countplot(x='cardiological pressure', data=df_train)
sns.boxplot(x='cardiological pressure', y='HDL cholesterol',hue='Gender', data=df_train)
sns.boxenplot(x='Gender', y='Age',hue='cardiological pressure', data=df_train)
sns.boxplot(x='cardiological pressure', y='Infect_Prob', data=df_train)
df_train['cardiological pressure'] = df_train['cardiological pressure'].fillna('Normal')

df_train['cardiological pressure'].isna().sum()
df_train['Diuresis'] = df_train['Diuresis'].fillna(df_train['Diuresis'].mean())

df_train['Diuresis'].isna().sum()
df_train['Platelets'] = df_train['Platelets'].fillna(df_train['Platelets'].mean())

df_train['Platelets'].isna().sum()
df_train['HBB'] = df_train['HBB'].fillna(df_train['HBB'].mean())

df_train['HBB'].isna().sum()
df_train['d-dimer'] = df_train['d-dimer'].fillna(df_train['d-dimer'].mean())

df_train['d-dimer'].isna().sum()
df_train['Heart rate'] = df_train['Heart rate'].fillna(df_train['Heart rate'].mean())

df_train['Heart rate'].isna().sum()
df_train['HDL cholesterol'] = df_train['HDL cholesterol'].fillna(df_train['HDL cholesterol'].mean())

df_train['HDL cholesterol'].isna().sum()
df_train.head()
df_train[(df_train.Occupation.isna()) & (df_train.Insurance.isna())].count()
df_train[~(df_train.Occupation.isna()) & (df_train.Insurance.isna())].count()
means_ins = df_train.groupby('Occupation')['Insurance'].mean()

mean_ins = df_train.Insurance.mean()

print(mean_ins)

means_ins

means_ins['Cleaner']
def impute_insurance(cols):

    occ = cols[0]

    ins = cols[1]



    if pd.isnull(ins):

        if not pd.isnull(occ):

            ins = means_ins[str(occ)]

            return ins 

        else:

            return mean_ins

    else:

        return ins



    

df_train['Insurance'] = df_train[['Occupation', 'Insurance']].apply(impute_insurance, axis=1)

df_train.Insurance.isna().sum()
sns.boxenplot(x='FT/month', y='salary', data=df_train)
df_train['FT/month'] = df_train['FT/month'].fillna(df_train['FT/month'].median())

df_train['FT/month'].isna().sum()
sns.countplot(x='Region', data=df_train)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df_train['Region']= le.fit_transform(df_train['Region'])

# df_train['Occupation']= le.fit_transform(df_train['Occupation'])

df_train['Mode_transport']= le.fit_transform(df_train['Mode_transport'])

df_train['comorbidity']= le.fit_transform(df_train['comorbidity'])



df_train.head()
sns.boxenplot(x='Diuresis', y='Infect_Prob', hue='Gender', data=df_train)
sns.boxenplot(x='Region', y='Infect_Prob', data=df_train)
sns.boxenplot(x='Occupation', y='Mode_transport', data=df_train)
sns.countplot(x='Occupation', hue='Region', data=df_train)
sns.countplot(x='Occupation', hue='Mode_transport', data=df_train)
# sns.pairplot(df_train)
# for i, col in enumerate(df_train.columns):

#     if not col in ['Occupation', 'people_ID','Pulmonary score','cardiological pressure','Diuresis','Platelets','HBB','d-dimer','Heart rate','HDL cholesterol','Charlson Index','Blood Glucose', 'Infect_Prob']:

#         plt.figure(i)

#         sns.countplot(x=col, hue='Occupation', data=df_train)
df_train = df_train.drop(['Occupation'], axis=1)

df_train.head()
df_train['Pulmonary score'] = df_train['Pulmonary score'].str.replace('<', '')

df_train['Pulmonary score'].head()
df_train.head()
df_train.drop(['Region', 'Deaths/1M'],axis=1, inplace=True)
df_train = pd.concat([df_train,pd.get_dummies(df_train['Mode_transport'], prefix='Mode_transport')],axis=1)

df_train = pd.concat([df_train,pd.get_dummies(df_train['comorbidity'], prefix='comorbidity')],axis=1)

df_train.drop(['Mode_transport', 'comorbidity'],axis=1, inplace=True)

df_train.head()
col_list = df_train.columns

col_list = ['people_ID','Mode_transport_0','Mode_transport_1','Mode_transport_2', 'Gender', 'Married', 'Children',

       'cases/1M','comorbidity_0','comorbidity_1','comorbidity_2','comorbidity_3', 'Age',

       'Coma score', 'Pulmonary score', 'cardiological pressure', 'Diuresis',

       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',

       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary', 'FT/month',

       'Infect_Prob']

df_train = df_train[col_list]

df_train.head()
df_train = pd.concat([df_train,pd.get_dummies(df_train['cardiological pressure'], prefix='cardiological pressure')],axis=1)

df_train.drop(['cardiological pressure'],axis=1, inplace=True)

df_train.head()
col_list = df_train.columns

col_list = ['people_ID','Mode_transport_0','Mode_transport_1','Mode_transport_2', 'Gender', 'Married', 'Children',

       'cases/1M','comorbidity_0','comorbidity_1','comorbidity_2','comorbidity_3', 'Age',

       'Coma score', 'Pulmonary score', 'cardiological pressure_Elevated','cardiological pressure_Normal','cardiological pressure_Stage-01',

       'cardiological pressure_Stage-02', 'Diuresis',

       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',

       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary', 'FT/month',

       'Infect_Prob']

df_train = df_train[col_list]

df_train.head()
from sklearn.preprocessing import MinMaxScaler

column_names_to_normalize = ['Age','Coma score', 'Pulmonary score', 'Diuresis',

       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',

       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary']

x = df_train[column_names_to_normalize].values

min_max_scaler=MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df_train.index)

df_train[column_names_to_normalize] = df_temp



df_train.head()
df_train['Target_norm'] = df_train["Infect_Prob"]/100.0

df_train.head()
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor 
X = df_train.iloc[:, 1:-2]

Y = df_train.iloc[:, -2]

Y_norm = df_train.iloc[:, -1]

Y.count()
X.head()
Y_norm.dtype
sns.distplot(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.02, random_state = 42)
X_train, X_test, Y_n_train, Y_n_test = train_test_split(X, Y_norm, test_size = 0.02, random_state = 42)
rf = RandomForestRegressor(criterion='mse', 

                             n_estimators=500,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(X_train, Y_n_train)

print("%.4f" % rf.oob_score_)
pd.concat((pd.DataFrame(X_train.columns, columns = ['variable']), 

           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,Y_train)
regressor.predict(X_test)

regressor.score(X_test,Y_test)
import statsmodels.api as sm

X_opt = X.iloc[:,:]

regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()

regressor_OLS.summary()
X_train = df_train.iloc[:,1:-2]

X_train.head()
Y_n_train = df_train.iloc[:,-1]

Y_n_train.head()
# def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 

#                        model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',

#                        do_probabilities = False):

#     gs = GridSearchCV(

#         estimator=model,

#         param_grid=param_grid, 

#         cv=cv, 

#         n_jobs=-1, 

#         scoring=scoring_fit,

#         verbose=2

#     )

#     fitted_model = gs.fit(X_train_data, y_train_data)

    

#     if do_probabilities:

#       pred = fitted_model.predict_proba(X_test_data)

#     else:

#       pred = fitted_model.predict(X_test_data)

    

#     return fitted_model, pred
import xgboost

from sklearn.model_selection import GridSearchCV



# # Let's try XGboost algorithm to see if we can get better results

# xgb = xgboost.XGBRegressor()

# param_grid = {

#     'n_estimators': [400, 500, 600,700,800],

#     'learning_rate': [0.002,0.008, 0.02, 0.4, 0.2],

#     'colsample_bytree': [0.3, 0.4,0.5,0.6,0.7],

#     'max_depth': [30,40,50,70,100],

#     'reg_alpha': [1.1, 1.2, 1.3],

#     'reg_lambda': [1.1, 1.2, 1.3],

#     'subsample': [0.7, 0.8, 0.9]

# }
xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.02, gamma=0, subsample=0.75,alpha=10,

                           colsample_bytree=0.3, max_depth=100)
# xgb, pred = algorithm_pipeline(X_train, X_test, Y_n_train, Y_n_test, xgb, 

#                                  param_grid, cv=5)
# print(np.sqrt(-xgb.best_score_))

# print(xgb.best_params_)
xgb.fit(X_train,Y_n_train,eval_metric='rmsle')
from sklearn.metrics import mean_squared_error

predictions = xgb.predict(X_test)

mse = mean_squared_error(predictions,Y_n_test)

print(np.sqrt(mse))
predictions = predictions*100.0
sns.distplot(predictions)
predictions
df_test = pd.read_excel('../input/flipr-hiring-challenge/Test_dataset.xlsx')

df_test.head(14)
df_test.describe()
df_test.info()
df_test.isna().sum()
for col in df_test.columns:

    print("col-name: ", col, " | no_of_unique_values: ", df_test[col].nunique(dropna=True))
df_test = df_test.drop(['Designation', 'Name'], axis = 1)

df_test.head(10)
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder



lb = LabelBinarizer()

df_test['Gender'] = lb.fit_transform(df_test.Gender)

df_test['Married'] = lb.fit_transform(df_test.Married)



df_test.head(5)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df_test['Region']= le.fit_transform(df_test['Region'])

# df_train['Occupation']= le.fit_transform(df_train['Occupation'])

df_test['Mode_transport']= le.fit_transform(df_test['Mode_transport'])

df_test['comorbidity']= le.fit_transform(df_test['comorbidity'])



df_test.head()
df_test = df_test.drop(['Occupation'], axis=1)

df_test.head()
df_test['Pulmonary score'] = df_test['Pulmonary score'].str.replace('<', '')

df_test['Pulmonary score'].head()
df_test.drop(['Region', 'Deaths/1M'],axis=1, inplace=True)
df_test = pd.concat([df_test,pd.get_dummies(df_test['Mode_transport'], prefix='Mode_transport')],axis=1)

df_test = pd.concat([df_test,pd.get_dummies(df_test['comorbidity'], prefix='comorbidity')],axis=1)

df_test.drop(['Mode_transport', 'comorbidity'],axis=1, inplace=True)

df_test.head()
col_list = df_test.columns

col_list = ['people_ID','Mode_transport_0','Mode_transport_1','Mode_transport_2', 'Gender', 'Married', 'Children',

       'cases/1M','comorbidity_0','comorbidity_1','comorbidity_2','comorbidity_3', 'Age',

       'Coma score', 'Pulmonary score', 'cardiological pressure', 'Diuresis',

       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',

       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary', 'FT/month']

df_test = df_test[col_list]

df_test.head()
df_test = pd.concat([df_test,pd.get_dummies(df_test['cardiological pressure'], prefix='cardiological pressure')],axis=1)

df_test.drop(['cardiological pressure'],axis=1, inplace=True)

df_test.head()
col_list = df_test.columns

col_list = ['people_ID','Mode_transport_0','Mode_transport_1','Mode_transport_2', 'Gender', 'Married', 'Children',

       'cases/1M','comorbidity_0','comorbidity_1','comorbidity_2','comorbidity_3', 'Age',

       'Coma score', 'Pulmonary score', 'cardiological pressure_Elevated','cardiological pressure_Normal','cardiological pressure_Stage-01',

       'cardiological pressure_Stage-02', 'Diuresis',

       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',

       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary', 'FT/month']

df_test = df_test[col_list]

df_test.head()
from sklearn.preprocessing import MinMaxScaler

column_names_to_normalize = ['Age','Coma score', 'Pulmonary score', 'Diuresis',

       'Platelets', 'HBB', 'd-dimer', 'Heart rate', 'HDL cholesterol',

       'Charlson Index', 'Blood Glucose', 'Insurance', 'salary']

x = df_test[column_names_to_normalize].values

min_max_scaler=MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df_test.index)

df_test[column_names_to_normalize] = df_temp



df_test.head()
test = df_test.iloc[:,1:]

test.head()
pred = xgb.predict(test)
pred = pred*100.0
pred
sns.distplot(pred)
submission = pd.read_excel('../input/flipr-hiring-challenge/Test_dataset.xlsx')

submission['infect_prob_20'] = pd.Series(pred)

submission.head()
pd.DataFrame(submission, columns=['people_ID', 'infect_prob_20']).to_csv('submission.csv', index = False)

sns.scatterplot(x='Diuresis', y='Infect_Prob', data=df_train)
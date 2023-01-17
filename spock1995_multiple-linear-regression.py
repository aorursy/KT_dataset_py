import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import os
from sklearn import metrics


%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.drop('sl_no', axis=1, inplace=True);
df.rename(columns={'ssc_p':'grade_X', 'ssc_b':'X_board', 'hsc_p':'grade_XII', 'hsc_b':'XII_board', 'hsc_s':'stream', 'degree_p':'grade_UG', 'degree_t':'field_UG', 'etest_p':'grade_PP', 'mba_p':'grade_MBA' }, inplace=True)
df[['Central_X','drop']] = pd.get_dummies(df.X_board)
df.drop('drop',axis=1,inplace=True)
df[['Central_XII','drop']] = pd.get_dummies(df.XII_board)
df.drop('drop',axis=1,inplace=True)
df[['Arts' ,'Commerce','Science']] = pd.get_dummies(df.stream)
df.drop('Arts',axis=1,inplace=True)
df[['Comm&Mgmt' ,'Others','Sci&Tech']] = pd.get_dummies(df.field_UG)
df.drop('Others',axis=1, inplace=True)
df[['drop','Experience']] = pd.get_dummies(df.workex)
df.drop('drop',axis=1,inplace=True)
df[['Mkt&Fin','Mkt&HR']] = pd.get_dummies(df.specialisation)
df.drop('Mkt&HR',axis=1,inplace=True)
df[['Female','Male']] = pd.get_dummies(df.gender)
df.drop('Female',inplace=True,axis=1)
df.drop(['gender', 'X_board', 'XII_board', 'stream', 'field_UG', 'workex', 'specialisation'], axis=1, inplace = True)
df.rename(columns={'Comm&Mgmt':'Comm_Mgmt', 'Sci&Tech':'Sci_Tech', 'Mkt&Fin':'Mkt_Fin'}, inplace = True);
df.head()
df['intercept'] = 1
df.head()
X = ['intercept', 'grade_X', 'grade_XII']
Y = df.grade_MBA
y, x = dmatrices('grade_MBA ~ grade_X + grade_XII', df, return_type='dataframe')

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
print(vif)
ln = sm.OLS(Y, df[X])
result = ln.fit()
print(result.summary())
X = ['intercept', 'grade_X', 'grade_UG']
Y = df.grade_MBA
y, x = dmatrices('grade_MBA ~ grade_X + grade_UG', df, return_type='dataframe')

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
print(vif)
ln = sm.OLS(Y, df[X])
result = ln.fit()
print(result.summary())
X = ['intercept', 'grade_XII', 'grade_UG']
Y = df.grade_MBA
y, x = dmatrices('grade_MBA ~ grade_XII + grade_UG', df, return_type='dataframe')

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
print(vif)
ln = sm.OLS(Y, df[X])
result = ln.fit()
print(result.summary())
X = ['intercept', 'grade_X' , 'grade_XII', 'grade_UG']
Y = df.grade_MBA
y, x = dmatrices('grade_MBA ~ grade_X + grade_XII + grade_UG', df, return_type='dataframe')

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
print(vif)
ln = sm.OLS(Y, df[X])
result = ln.fit()
print(result.summary())
X = ['intercept', 'grade_X' , 'grade_UG']
Y = df.grade_MBA
ln = sm.OLS(Y, df[X])
result = ln.fit()
print(result.summary())
X_train, X_test, y_train, y_test = train_test_split(df[X], y, test_size=0.2, random_state=1001)
ln = LinearRegression()
ln.fit(X_train, y_train)
y_pred = ln.predict(X_test)
print('Accuracy of linear regression classifier on test set: {:.2f}'.format(ln.score(X_test, y_test)))
df_new = y_test.copy()
df_new['Prediction'] = y_pred
df_new.rename(columns={'grade_MBA':'Actual'},inplace=True)
df_new.head()

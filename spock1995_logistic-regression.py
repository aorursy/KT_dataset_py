import pandas as pd

import numpy as np

import seaborn as sns

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

import os

import matplotlib.pyplot as plt



%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')



df.drop('sl_no', axis=1, inplace=True);



df.rename(columns={'ssc_p':'grade_X', 'ssc_b':'X_board', 'hsc_p':'grade_XII', 'hsc_b':'XII_board', 'hsc_s':'stream', 'degree_p':'grade_UG', 'degree_t':'field_UG', 'etest_p':'grade_PP', 'mba_p':'grade_MBA' }, inplace=True)



df.drop([24,42,49,120,134,169,177,206,197],axis=0, inplace = True)
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
df.head()
df.rename(columns={'Comm&Mgmt':'Comm_Mgmt', 'Sci&Tech':'Sci_Tech', 'Mkt&Fin':'Mkt_Fin'}, inplace = True);
df.head()
X = ['grade_X', 'grade_XII','grade_UG', 'grade_PP', 'grade_MBA', 'Central_X', 'Central_XII', 'Commerce', 'Science', 'Comm_Mgmt', 'Sci_Tech', 'Experience', 'Mkt_Fin', 'Male']

Y = df.status


logreg = LogisticRegression()

rfe = RFE(logreg)

rfe = rfe.fit(df[X], Y.values.ravel())

print(rfe.support_)

print(rfe.ranking_)
df.head()
df.status.replace('Placed',1,inplace=True);

df.status.replace('Not Placed',0,inplace=True);
y, x = dmatrices('status ~ grade_X + Central_XII + Commerce + Science + Comm_Mgmt + Experience + Mkt_Fin', df, return_type='dataframe')



vif = pd.DataFrame()

vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

vif['features'] = x.columns
vif
df['intercept'] = 1
Z = ["grade_X" , "Comm_Mgmt",  "Experience",   "intercept"]
logit_model=sm.Logit(df['status'], df[Z])

result=logit_model.fit()

print(result.summary2())
np.exp(0.2040), np.exp(1.3632),  np.exp(1.7775)
df.head()
X = df[Z]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_MIT= pd.read_csv('../input/course-study/appendix.csv')

df_MIT.head()
df_MIT.info()
df_MIT=df_MIT.drop(['% Certified','Course Number','Course Title','% Grade Higher Than Zero'],axis=1)

df_MIT
df_MIT['Course Subject'].value_counts() #DUMMIES
df_MIT['Instructors'].value_counts() #DELETE
plt.figure(figsize=(15,10))

sns.heatmap(df_MIT.isnull(),cmap="YlGnBu")
figure= plt.figure(figsize=(10,10))

sns.heatmap(df_MIT.corr(), annot=True,cmap="YlGnBu")

#To show the correlation between variables
figure= plt.figure(figsize=(20,10))

sns.boxenplot(x='Course Subject',y='% Certified of > 50% Course Content Accessed',data=df_MIT,palette="Blues")
figure= plt.figure(figsize=(20,10))

sns.boxenplot('Participants (Course Content Accessed)','Institution',data=df_MIT)
df_pairplot_cols=df_MIT[['Institution','Audited (> 50% Course Content Accessed)','Year','% Certified of > 50% Course Content Accessed','Total Course Hours (Thousands)','% Female','% Male','Median Age']]

plt.figure(figsize=(20,20))

sns.pairplot(df_pairplot_cols,hue='Institution',palette='rainbow')
sns.lmplot(x='Participants (Course Content Accessed)',y='Audited (> 50% Course Content Accessed)',data=df_MIT,col='Course Subject',hue='Institution',palette='coolwarm',

          aspect=0.6,size=8)
sns.lmplot(x='Median Age',y='% Female',data=df_MIT,col='Course Subject',hue='Institution',palette='coolwarm',

          aspect=0.6,size=8)
sns.lmplot(x='Median Age',y='% Male',data=df_MIT,col='Course Subject',hue='Institution',palette='coolwarm',

          aspect=0.6,size=8)
x= df_MIT['Median Hours for Certification']

y= df_MIT['% Certified of > 50% Course Content Accessed']

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

plt.figure(figsize=(10,10))

sns.kdeplot(x, y, cmap=cmap, shade=True);
x= df_MIT['Median Age']

y= df_MIT['% Certified of > 50% Course Content Accessed']

plt.figure(figsize=(10,10))

sns.kdeplot(x, y, shade=True);
df_MIT.info()
df_XGB = df_MIT.drop(['Total Course Hours (Thousands)','Certified','Audited (> 50% Course Content Accessed)','Instructors','Launch Date','% Played Video'],axis=1)
df_XGB.info()
Institution = pd.get_dummies(df_XGB['Institution'],drop_first=True)

CourseSubject = pd.get_dummies(df_XGB['Course Subject'],drop_first=True)

df_XGB.drop(['Institution','Course Subject'],axis=1,inplace=True)

df_XGB = pd.concat([df_XGB,Institution,CourseSubject],axis=1)

df_XGB
plt.figure(figsize=(15,10))

sns.heatmap(df_XGB.isnull(),cmap="YlGnBu")
from sklearn.model_selection import train_test_split

x= df_XGB

y=df_XGB['% Certified of > 50% Course Content Accessed']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=109)
import xgboost as xgb

train= xgb.DMatrix(x_train,label=y_train)

test = xgb.DMatrix(x_test, label= y_test)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.2,

                max_depth = 7, alpha = 10, n_estimators = 75)
xg_reg.fit(x_train,y_train)

preds = xg_reg.predict(x_test)

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))
import matplotlib.pyplot as plt



xgb.plot_tree(xg_reg,num_trees=0)

plt.rcParams['figure.figsize'] = [20, 15]

plt.show()
xgb.plot_importance(xg_reg)

plt.rcParams['figure.figsize'] = [15,15]

plt.show()
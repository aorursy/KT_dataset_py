# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df.describe(include='all')
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt



plt.style.use('ggplot')

fig,ax=plt.subplots(1,2,figsize=(15,5))

sns.countplot(x='gender',data=df,hue='gender',ax=ax[0])

sns.countplot(x='gender',data=df,hue='status',ax=ax[1])

ax[0].set_title('Total no of female and male students')

ax[1].set_title('Total no of female and male students who have been placed and not placed')



fig.tight_layout(pad=5.0)

# lets plot 

sns.set(style="dark")

fig,ax=plt.subplots(1,3,figsize=(25,5))

sns.countplot(x='degree_t',data=df,hue='status',palette="Set2",ax=ax[0])

sns.countplot(x='hsc_b',data=df,hue='status',ax=ax[1],palette="GnBu")

sns.countplot(x='workex',data=df,hue='status',ax=ax[2],palette="deep")



ax[0].set_title('Degree on placement')

ax[1].set_title('Board on placement')

ax[2].set_title('Previous work experience on placement')





#plotting pair plot bw diff data

plt.figure(figsize=(25,20))

sns.pairplot(df[[column for column in df.columns if column not in ['sl_no']]])

plt.tight_layout(pad=2.0)
df.head()
# first of all we will drop salary and sl_no as they are not the requi9red feature for our model creations

df.drop(['sl_no','salary'],axis=1,inplace=True)

df.head(2)
# lets seperate independent feature with dependent feature

ind_df=df.drop('status',axis=1)

dep_df=df[['status']]

#creating dummy variable and dropping first row

ind_df=pd.get_dummies(ind_df,drop_first=True)

ind_df.head(2)
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
# splitting data into train part and test part

X_train,X_test,y_train,y_test=train_test_split(ind_df,dep_df,test_size=0.2,random_state=0)
#model creation SVC

model=SVC()

model.fit(X_train,y_train)
# predicting score 

y_pred=model.predict(X_test)

model.score(X_test,y_test)
cm=confusion_matrix(y_test,y_pred)

# lets plot heat map to visulize this

sns.heatmap(cm)
# lets import library for doing hyperparameter tuning there are two ways 

# 1. Grid search cv

# 2. Randomized search cv

from sklearn.model_selection import GridSearchCV

# lets set our params

parameter=[{'C':[50,100,150,200],'kernel':['linear']}

          ,{'C':[100,200,300,400],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_srch=GridSearchCV(estimator=model,param_grid=parameter,scoring='accuracy',cv=10,n_jobs=-1)

grid_srch.fit(X_train,y_train)
# now check our best accuracy

grid_srch.best_score_
# these parameters are the best fit for our model

grid_srch.best_params_
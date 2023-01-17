# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
df=pd.read_csv('../input/train.csv')

df.set_index('PassengerId',inplace=True)

df=df.drop(['Ticket','Name'],1)

df.info()
df.isnull().sum()
#Replacing null values of age as mean age value

#Encoding Sex:Male and Female into 0 and 1

df['Age'].fillna(df.Age.mean(),inplace=True)

df['Sex'].replace({'male':int(0),'female':int(1)},inplace=True)

df.head(5)
#Let's check the unique values of Cabin in the dataset:



#Most of the cabins  have common first letter ,lets extract that:

df['Cabin'].fillna('N',inplace=True)



df['Cabin'].value_counts()



def clean_cabin(x):

    try:

        return(x[0])

    except:

        return(None)

    

df['Cabin']=df.Cabin.apply(clean_cabin)

df['Cabin'].unique()
ls_non_numeric=list(df.dtypes[df.dtypes == 'object'].index)

for i in ls_non_numeric:

    dummies=pd.get_dummies(df[i],prefix=i)

    df=pd.concat([df,dummies],axis=1)

    df.drop([i],axis=1,inplace=True)

df.drop(['Cabin_N'],axis=1,inplace=True)

df.head(5)
df.dtypes['Sex']

df.dtypes[0]



df.dtypes[:]
#Spliting up the data:

X=df.drop('Survived',1)

y=df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
%%timeit

model=RandomForestRegressor(n_estimators=100, oob_score=True,n_jobs=-1,random_state=42)

model.fit(X,y)

print('C-stat :',roc_auc_score(y,model.oob_prediction_))
#All the methods with trailing underscore are available only after training the model

model.oob_score_
%%timeit

#n_jobs:Makes easier to train the model

#sets how many processors are allowed to use:

#'-1':No restriction,'1':1 processor





model=RandomForestRegressor(n_estimators=100, oob_score=True,n_jobs=1,random_state=42)

model.fit(X,y)



print('C-stat :',roc_auc_score(y,model.oob_prediction_))
feature_importances=pd.Series(model.feature_importances_,index=X.columns)

feature_importances.sort(inplace=True)

#plt.hist(feature_importance)

#feature_importance

feature_importances.plot(kind='barh',figsize=(7,6));
#Cummulative graph
#
results=[]

n = [30,50,100,200,500,1000,2000]



for trees in n:

    model=RandomForestRegressor(trees,oob_score=True,n_jobs=-1,random_state=42)

    model.fit(X,y)

    print(trees,'trees')

    roc=roc_auc_score(y,model.oob_prediction_)

    print('C-stat :',roc)

    results.append(roc)

    print('')

pd.Series(results,n_estimators_options).plot()
##**max features**##
results=[]

max_features_options=['auto',None,'sqrt','log2',0.9,0.2]



for max_features in max_features_options:

    model=RandomForestRegressor(n_estimators=1000,oob_score=True,n_jobs=-1,random_state=42,max_features=max_features)

    model.fit(X,y)

    print(max_features,'options')

    roc=roc_auc_score(y,model.oob_prediction_)

    results.append(roc)

    print('')

pd.Series(results,max_features_options).plot(kind='barh',xlim=(0.88,0.80))  
results=[]

min_sample_in_leaf_option=[1,2,3,4,5,6,7,8,9,10]



for min_samples in min_sample_in_leaf_option:

    model=RandomForestRegressor(n_estimators=1000,oob_score=True,n_jobs=-1,random_state=42,max_features='auto',min_samples_leaf=min_samples)

    model.fit(X,y)

    print(min_samples,'min sample')

    roc=roc_auc_score(y,model.oob_prediction_)

    results.append(roc)

    print('')

pd.Series(results,min_sample_in_leaf_option).plot()  
RandomForestRegressor(n_estimators=1000,

                      oob_score=True,

                      n_jobs=-1,

                      random_state=42,

                      max_features='auto',

                      min_samples_leaf=5)

model.fit(X,y)



roc=roc_auc_score(y,model.oob_prediction_)

print('C-Stat :',roc)
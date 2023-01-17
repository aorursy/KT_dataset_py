# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV,LinearRegression

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.utils import resample

from sklearn.decomposition import PCA, NMF

import seaborn as sns

from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC

# from sklearn.tree.DecisionTreeClassicv_results['test_score']Rfier

import math

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import matplotlib.pyplot as plt

import re

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_input=pd.read_csv('../input/train.csv')

test_input=pd.read_csv('../input/test.csv')

np.random.seed(3)



train_input2=train_input.copy(deep=True)

test_input2=test_input.copy(deep=True)

train_output=train_input['Survived']

train_input.drop(['Survived'],axis=1, inplace=True)

train_input.sample(5)
for i in train_input.columns:

    if train_input[i].isnull().values.mean()>0 or test_input[i].isnull().values.mean()>0:

        total=train_input[i].isnull().values.mean()*train_input.shape[0] + test_input[i].isnull().values.mean()*test_input.shape[0]

        print(f"{i} contains {total} null values")       
datasets=[train_input,test_input]



for dataset in datasets:

    dataset['Age'].fillna(24, inplace=True)

    dataset['Cabin'].fillna('N',inplace=True)

    dataset['Embarked'].fillna('S',inplace=True)

    dataset['Fare'].fillna(dataset['Fare'].mean(),inplace=True)

    display(dataset.sample(5))
def get_title(name):

    title_name=re.search('([A-Za-z]+)\.',name)

    

    if title_name:

        return title_name.group(1)

    return "Other"



datasets=[train_input,test_input]

test_data_passenger_id=test_input2['PassengerId']

for index,dataset in enumerate(datasets):

    dataset['SibSp'].astype(int)

    dataset['IsAlone']=0

    dataset['Family']=dataset['SibSp']+dataset['Parch']

    dataset.loc[dataset['SibSp']+dataset['Parch']==0,'IsAlone']=1

    dataset['Title']=dataset['Name'].apply(get_title)

    

    dataset.replace(to_replace=['male','female'],value=[1,0],inplace=True)

    

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 5,labels=[i for i in range(5)])

    dataset['AgeBin'] = pd.cut(dataset['Age'], 4,labels=[i for i in range(4)])

    

    datasets[index]=pd.get_dummies(dataset,columns=['Embarked','Pclass'])

    dataset=datasets[index]

    

    dataset.drop(['SibSp','Parch','Cabin','Name','Ticket','PassengerId'],axis=1,inplace=True)

    display(dataset.sample(5))

    

train_input=datasets[0]

test_input=datasets[1]
print(train_input.Title.value_counts())

print(test_input.Title.value_counts())


for index,dataset in enumerate(datasets):

    dataset.loc[(dataset['Title']!='Mr') & (dataset['Title'] !='Miss') & (dataset['Title'] !='Mrs'),'Title']='Other'

    datasets[index]=pd.get_dummies(dataset, columns=['Title'])

    dataset=datasets[index]

    print(dataset.sample(5))    

    

train_input, test_input=datasets[0],datasets[1]
print(id(train_input) == id(datasets[0]))

print(train_input.head())

print(datasets[0].head())
features=train_input.columns

for feature in features:

    if train_input[feature].dtype == 'int64' :

        print("Relation b/w {} and survival".format(str(feature)))

        print(pd.DataFrame({feature:train_input[feature],'output':train_output}).

                            groupby(feature).mean())

        print('\n'*3)
_, axes=plt.subplots(10,2 , figsize=(16,40))

for feature ,ax in zip(features, axes.ravel()):

    sns.barplot(train_input[feature],train_output,ax=ax)
#v for virtual

train_input_v, test_input_v,train_output_v,test_output_v=train_test_split(

    train_input,train_output,test_size=0.2)

print(train_input_v.shape, test_input_v.shape,train_output_v.shape,test_output_v.shape)
train_input_layer1,train_input_layer2, train_output_layer1,train_output_layer2=train_test_split(

train_input_v,train_output_v)

pipe1=make_pipeline(StandardScaler(),SVC(C=0.1,gamma=0.5,probability=True))

pipe2=make_pipeline(StandardScaler(),KNeighborsClassifier(1))

pipe3=make_pipeline(StandardScaler(),RandomForestClassifier(max_depth=5, min_samples_split=5))

pipe4=make_pipeline(StandardScaler(), GaussianProcessClassifier())



train_input_layer1_copy=train_input_layer1.copy(deep=True)

train_input_layer1_copy['Survived']=train_output_layer1

train_inputoutput_layer1_1=resample(train_input_layer1_copy,replace=True,n_samples=400)

train_inputoutput_layer1_2=resample(train_input_layer1_copy,replace=True,n_samples=400)

train_inputoutput_layer1_3=resample(train_input_layer1_copy,replace=True,n_samples=400)

train_inputoutput_layer1_4=resample(train_input_layer1_copy,replace=True,n_samples=400)





pipe1.fit(train_inputoutput_layer1_1.drop(['Survived'], axis=1),train_inputoutput_layer1_1['Survived'])



pipe2.fit(train_inputoutput_layer1_2.drop(['Survived'], axis=1),train_inputoutput_layer1_2['Survived'])



pipe3.fit(train_inputoutput_layer1_3.drop(['Survived'], axis=1),train_inputoutput_layer1_3['Survived'])



pipe4.fit(train_inputoutput_layer1_4.drop(['Survived'], axis=1),train_inputoutput_layer1_4['Survived'])



print(pipe1.score(test_input_v,test_output_v))

print(pipe2.score(test_input_v,test_output_v))



print(pipe3.score(test_input_v,test_output_v))



print(pipe4.score(test_input_v,test_output_v))
input_for_l2=pd.DataFrame({'i1':np.array(pipe1.predict_proba(train_input_layer2))[:,0],'i2':np.array(pipe2.predict_proba(train_input_layer2))[:,0],

                          'i3':np.array(pipe3.predict_proba(train_input_layer2))[:,0],'i4':np.array(pipe4.predict_proba(train_input_layer2))[:,0]})

print(input_for_l2.head())



reg=LogisticRegression()

reg.fit(input_for_l2,np.array(train_output_layer2))
input_for_l2_cv=pd.DataFrame({'i1':np.array(pipe1.predict_proba(test_input_v))[:,0],'i2':np.array(pipe2.predict_proba(test_input_v))[:,0],

                          'i3':np.array(pipe3.predict_proba(test_input_v))[:,0],'i4':np.array(pipe4.predict_proba(test_input_v))[:,0]})

final_ans=reg.predict(input_for_l2_cv)

ans=0

for index,des in zip(final_ans,test_output_v):

    if(des==index):

        ans+=1

        

print(ans/len(final_ans))
input_for_l2_test=pd.DataFrame({'i1':np.array(pipe1.predict_proba(test_input))[:,0],'i2':np.array(pipe2.predict_proba(test_input))[:,0],

                          'i3':np.array(pipe3.predict_proba(test_input))[:,0],'i4':np.array(pipe4.predict_proba(test_input))[:,0]})

output=reg.predict(input_for_l2_test)

subm3=pd.DataFrame({"PassengerId":np.array(test_data_passenger_id),"Survived":output})

subm3.to_csv("subm3.csv",index=False)

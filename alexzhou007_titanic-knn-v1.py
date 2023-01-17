# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Import data



df = pd.read_csv('../input/train.csv')

df.head(10)
#Observe the Influence of Sexuality



#Male Survival Rate

male_surv=(df[(df['Survived']==1) & (df['Sex'] =='male')].shape[0])/(df[df['Sex']=='male'].shape[0])



#Female Survival Rate

female_surv=(df[(df['Survived']==1) & (df['Sex'] =='female')].shape[0])/(df[df['Sex']=='female'].shape[0])



#General Survive

general_surv = df[df['Survived']==1].shape[0]/df.shape[0]

print('Male surv rate %s Female surv rate %s general rate %s' %(male_surv,female_surv,general_surv))
female= df['Survived'][df['Sex']=='female'].value_counts()

male = df['Survived'][df['Sex']=='male'].value_counts()

df_sex = pd.DataFrame({'female':female,'male':male})

# df_sex['Total'] = df_sex['female']+df_sex['male']

# df_sex.plot(kind='bar')
#Observe the influence of aga



#Survived Hisgram

plt.subplot(211)

df['Age'][df['Survived']==1].hist()

plt.title('Survived Age Distribution')



plt.subplot(212)

df['Age'].hist()

plt.title('Overal Age Distribution')
#Observe the influence of Cabin Class

ind=df['Pclass'].value_counts().index

p1 = plt.bar(ind,df['Pclass'][df['Survived']==1].value_counts(),color='orange',bottom=df['Pclass'][df['Survived']==0].value_counts())

p2 = plt.bar(ind,df['Pclass'][df['Survived']==0].value_counts(),color='blue')

plt.legend((p1[0], p2[0]), ('Survived', 'Not_Survived'))

plt.show()
#Observe the influence of Fare

df['Fare'].hist()

df['Fare'][df['Survived']==1].hist(alpha=0.3,color='r')
#Observe the Embarked Station

ind=df['Embarked'][df['Survived']==0].value_counts().index

p1 = plt.bar(ind,df['Embarked'][df['Survived']==1].value_counts(),color='orange',bottom=df['Embarked'][df['Survived']==0].value_counts())

p2 = plt.bar(ind,df['Embarked'][df['Survived']==0].value_counts(),color='blue')

plt.legend((p1[0], p2[0]), ('Survived', 'Not_Survived'))

plt.show()
#Separate_Label

label = df.loc[:,'Survived']

data = df.loc[:,['Pclass', 'Sex', 'Age','Fare', 'Embarked']]

data
#Data Cleaning



def fill_null(data):

    data_copy = data.copy(deep=True)

    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())

    data_copy.loc[:,'Fare'] = data['Fare'].fillna(data['Fare'].median())

    data_copy.loc[:,'Pclass']=data['Pclass'].fillna(data['Pclass'].median())

    data_copy.loc[:,'Embarked']=data['Embarked'].fillna('S')

    return data_copy



data_filled = fill_null(data)

data_filled.isnull().values.any()
#Convert string to numerical value

def sex_to_num(data):

    data_copy = data.copy(deep=True)

    data_copy.loc[data_copy['Sex']=='female','Sex'] =0

    data_copy.loc[data_copy['Sex']=='male','Sex']=1

    return data_copy



def embark_to_num(data):

    data_copy = data.copy(deep=True)

    data_copy.loc[data_copy['Embarked']=='S','Embarked'] =0

    data_copy.loc[data_copy['Embarked']=='C','Embarked'] =1

    data_copy.loc[data_copy['Embarked']=='Q','Embarked'] =2

    return data_copy



data_convert_sex = sex_to_num(data_filled)

data_convert_embark = embark_to_num(data_convert_sex)



data_convert_embark 
# Split Data

data_final = data_convert_embark 



from sklearn.model_selection import train_test_split



train_data, val_data, train_labels, val_labels = train_test_split(data_final,label,random_state=0,test_size=0.2)



print(train_data.shape, val_data.shape, train_labels.shape, val_labels.shape)

# KNN Learn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



scores=[]



for k in range(1,50):

    knn_model = KNeighborsClassifier(n_neighbors=k).fit(train_data, train_labels)

    predictions = knn_model.predict(val_data)

    accuracy = accuracy_score(val_labels,predictions)

    scores.append(accuracy)

    print('k = %s, accuracy = %s' %(k, accuracy))

 

print('the best k %d' %(np.array(scores).argsort()[-1]+1))
#Convert real test data to template format

df_test_raw = pd.read_csv('../input/test.csv')

df_test = df_test_raw.loc[:,['Pclass', 'Sex', 'Age','Fare', 'Embarked']]

df_test_filled = fill_null(df_test)

df_test_convert_sex = sex_to_num(df_test_filled)

df_test_convert_embark = embark_to_num(df_test_convert_sex)

df_test_final = df_test_convert_embark



#Fit model with all training data and best k

knn_model = KNeighborsClassifier(n_neighbors=33).fit(data_final, label)

results = knn_model.predict(df_test_final)

print(results)
#Print

df_result = pd.DataFrame({"PassengerId": df_test_raw['PassengerId'],"Survived": results})

df_result.to_csv('submission.csv',header=True, index=False)
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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
data_list=[train,test]

def remove_col(data_list):
    trans_data=[]
    col_list=['PassengerId','Name','Ticket','Cabin']
    for data in data_list:
        Data=data.drop(col_list,axis=1)
        trans_data.append(Data)
    Data1=trans_data[0]
    Data2=trans_data[1]
    return Data1,Data2

Train,Test=remove_col(data_list)
data_list2=[Train,Test]
def sex(data_list):
    trans_data=[]
    for data in data_list:
        data['Sex']=data['Sex'].map({'male':0,'female':1})
        trans_data.append(data)
    data1=trans_data[0]
    data2=trans_data[1]
        
    return data1,data2

Train,Test=sex(data_list2)
sns.countplot(train['Survived'])
sns.countplot(train['Survived'],hue=train['Pclass'])
sns.countplot(train['Sex'],hue=train['Survived'])
sns.distplot(train['Age'])
sns.catplot(x='Survived',y='Age',data=train,kind='box',hue='Pclass',col='Sex')
sns.catplot(x='Survived',kind='count',data=train,height=5,hue='Pclass',col='Sex')
sns.countplot(train['Pclass'])
sns.boxplot(x=train['Pclass'],y=train['Age'])
a=train.groupby('Pclass')['Age']
b=train.groupby('Pclass')['Fare']
print('Median Age of people in Pclass 1 is : {} years \t Mean Fare of people in Pclass 1 is : {:.2f} '.format(a.get_group(1).median(),b.get_group(1).mean()))
print('Median Age of people in Pclass 2 is : {} years \t Mean Fare of people in Pclass 2 is : {:.2f} '.format(a.get_group(2).median(),b.get_group(2).mean()))
print('Median Age of people in Pclass 3 is : {} years \t Mean Fare of people in Pclass 3 is : {:.2f} '.format(a.get_group(3).median(),b.get_group(3).mean()))
def impute(cols):
    age=cols[0]
    pclass=cols[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 24
    else:
        return age
    
Train['Age'] = Train[['Age','Pclass']].apply(impute,axis=1)
Test['Age'] = Test[['Age','Pclass']].apply(impute,axis=1)
sns.countplot(train['Embarked'])
Train['Embarked']= Train['Embarked'].fillna('S')
print(Test[Test['Fare'].isnull()])
print('\nFor Pclass =3 ,The mean fare was 13.68')
Test['Fare']=Test['Fare'].fillna(13.68)
train_with_dummies=pd.get_dummies(Train,drop_first=True)
test_with_dummies=pd.get_dummies(Test,drop_first=True)
x_train=train_with_dummies.drop('Survived',axis=1)
y_train=train_with_dummies['Survived']
x_test=test_with_dummies
lda=LDA()
x_train_lda=lda.fit_transform(x_train,y_train)
x_test_lda=lda.transform(x_test)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train_lda,y_train)
prediction=classifier.predict(x_test_lda)

results=pd.read_csv('../input/titanic/gender_submission.csv')
y_test=results['Survived']
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,prediction)
sns.heatmap(cm,annot=True)
print('Accurcay score is : {:.2f}%'.format(accuracy_score(y_test,prediction)*100))

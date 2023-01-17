# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
rawdata=pd.read_csv('/kaggle/input/titanic/train.csv')
rawdata.describe(include='all')
pd.crosstab(rawdata['Survived'],rawdata['Parch'],margins=True)
plt.scatter(rawdata['Pclass'],rawdata['Fare'])
plt.show()
data=rawdata.copy()
data
data=data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
data
data['Sex']=data['Sex'].map({'female':1,'male':0})
#data.loc[data['Fare'] >100, 'High_Fare'] = 1 
#data.loc[data['Fare'] <= 100, 'High_Fare'] = 0
#data=data.drop(['Fare'],axis=1)
#data['SibSp']=data['SibSp'].map({0:0,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1})
#data['Parch']=data['Parch'].map({0:0,1:1,2:2,3:2,4:2,5:2,6:2})
data
classes=pd.get_dummies(data['Pclass'])
classes=classes.rename(columns={1:'Class 1',2:'Class 2',3:'Class 3'})

sexx=pd.get_dummies(data['Sex'])
sexx=sexx.rename(columns={1:'Female',0:'Male'})

embark=pd.get_dummies(data['Embarked'])
embark=embark.rename(columns={'S':'Embark S','Q':'Embark Q','C':'Embark C'})
frames=[data,classes,sexx,embark]
data2=pd.concat(frames,axis=1)
data2=data2.drop(['Pclass','Sex','Embarked','Class 3','Male','Embark S','Fare','Parch'],axis=1)
data2
x_train=data2.iloc[:,1:9]
y_train=data2.iloc[:,0]
x_train
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values = my_imputer.fit_transform(x_train)
x_train2=pd.DataFrame(data_with_imputed_values)
x_train2.columns=['Age', 'SibSp', 'Class 1', 'Class 2', 'Female', 'Embark C',
       'Embark Q']
x_train2
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train2,y_train)
reg.score(x_train2,y_train)
model_outputs=reg.predict(x_train2)
model_outputs==y_train
name=x_train2.columns.values
summary_table=pd.DataFrame(columns=['Name'],data=name)
summary_table['Coefficient']=np.transpose(reg.coef_)
summary_table.loc[7]=['Intercept',reg.intercept_[0]]
summary_table['Odds Ratio']=np.exp(summary_table.Coefficient)
summary_table
#Test outputs
rawtestdata=pd.read_csv('/kaggle/input/titanic/test.csv')
testdata=rawtestdata.copy()
testdata.head()
testdata=testdata.drop(['Name','Ticket','Cabin'],axis=1)

testdata['Sex']=testdata['Sex'].map({'female':1,'male':0})

classes=pd.get_dummies(testdata['Pclass'])
classes=classes.rename(columns={1:'Class 1',2:'Class 2',3:'Class 3'})

sexx=pd.get_dummies(testdata['Sex'])
sexx=sexx.rename(columns={1:'Female',0:'Male'})

embark=pd.get_dummies(testdata['Embarked'])
embark=embark.rename(columns={'S':'Embark S','Q':'Embark Q','C':'Embark C'})

frames=[testdata,classes,sexx,embark]
testdata2=pd.concat(frames,axis=1)

testdata2=testdata2.drop(['Pclass','Sex','Embarked','Class 3','Male','Embark S','Fare','Parch'],axis=1)
testdata2.head()
from sklearn.impute import SimpleImputer
my_imputer2 = SimpleImputer()
data_with_imputed_values2 = my_imputer2.fit_transform(testdata2)
testdata3=pd.DataFrame(data_with_imputed_values2)
testdata3
testdata3.columns=testdata2.columns
testdata3
x_test=testdata3.iloc[:,1:]
submission=testdata3.iloc[:,0]
test_outputs=reg.predict(x_test)
test_outputs2=pd.DataFrame(test_outputs)
test_outputs2.columns=['Survived']
test_outputs2
frames2=[submission,test_outputs2]
outputs=pd.concat(frames2,axis=1)
outputs['PassengerId']=outputs['PassengerId'].astype(int)
outputs
outputs.to_csv('first_submission.csv')

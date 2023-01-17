# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
%matplotlib inline
def data_input(file):
    data= pd.read_csv(file)
    return data
df=data_input('/kaggle/input/titanic/train.csv')
df.head()
df.describe()
print('Shape:',df.shape)
print(df.info())
#Finding Null Values In eni=tire dataframe
sns.heatmap(df.isnull(),cbar=False)
df['Survived'].value_counts(normalize=True).plot(kind='pie',autopct='%1.1f%%')
pd.crosstab(df['Pclass'],df['Survived'])
sns.heatmap(df.corr(),annot=True)
def data_dummy(data,col):
    return pd.get_dummies(data,columns=col)
def drop_column(data,col):
    return data.drop(col,axis=1)

df = data_dummy(df,['Sex','Embarked'])
df=drop_column(df,['PassengerId','Name','Ticket','Cabin','Sex_male'])
#Removing Null Rows
df.dropna(inplace=True)
print('Skewness :',stats.skew(df['Age']))
sns.distplot(df['Age'].dropna())
print('Skewness :',stats.skew(df['Fare']))
sns.distplot(df['Fare'].dropna())
#df= df.sort_values(by='Fare',ascending=False)[10:]  #Skewness Reduction
df = df.sample(frac=1).reset_index(drop=True)
print('Data Shape :', df.shape)
X=df.drop('Survived',axis=1)
y=df['Survived']
z=[]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
def data_modelling(ml_model,X_train,X_test,y_train,y_test):
    model = ml_model
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    ml_model=str(ml_model).split('(')[0]
    print('Accuracy for {}'.format(ml_model))
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    train_model= round(model.score(X_train, y_train)*100,2)
    test_model = round(accuracy_score(y_pred, y_test,normalize=True) * 100, 2)
    print('Training Model:',train_model)
    print('Predicted Model:', test_model)
    
    return z.append([ml_model,train_model,test_model])
data_modelling(SGDClassifier(max_iter=10000),X_train,X_test,y_train,y_test)
data_modelling(KNeighborsClassifier(),X_train,X_test,y_train,y_test)
data_modelling(LogisticRegression(penalty='l2',C=1.0,solver='liblinear',max_iter=10000),X_train,X_test,y_train,y_test)
data_modelling(LinearSVC(max_iter=1000),X_train,X_test,y_train,y_test)
data_modelling(DecisionTreeClassifier(),X_train,X_test,y_train,y_test)
data_modelling(GaussianNB(),X_train,X_test,y_train,y_test)
data_modelling(MLPClassifier(solver='lbfgs'),X_train,X_test,y_train,y_test)
outcomes=pd.DataFrame(z, columns=['Model', 'Training Model', 'Predicted Model'])
sns.barplot(data=outcomes, x='Predicted Model',y='Model')
test_df=data_input('/kaggle/input/titanic/test.csv')
print(test_df.info())
#Feature Engg.
test_df = data_dummy(test_df,['Sex','Embarked'])
test_df = drop_column(test_df,['Name','Ticket','Cabin','Sex_male'])
test_df.isnull().sum()
#test_df[test_df['Fare'].isnull()]
test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'][test_df['Pclass']== 3].mean())
def data_filling(median):
    Age=median[0]
    Pclass=median[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 60
        elif Pclass == 2:
            return 16
        elif Pclass == 3:
            return 8
    else:
        return Age

test_df['Age']=test_df[['Age','Pclass']].apply(data_filling,axis=1)
#Modelling
model = GaussianNB()
model.fit(X_train,y_train)
test_df['Survived'] = model.predict(test_df.drop('PassengerId',axis=1))
result=test_df[['PassengerId','Survived']]
result.to_csv('Submission.csv',index=False)

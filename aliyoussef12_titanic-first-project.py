import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

df = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
df.dropna(axis=0, how='any', thresh=None, subset=['Embarked'])



print(df.describe())
print(df.info())
df.head()
list(df.columns)
numeric = df[['Age','SibSp','Parch','Fare']]
categorical = df[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
for i in numeric.columns:
    plt.hist(numeric[i])
    plt.title(i)
    plt.show()
sns.heatmap(numeric.corr())
pd.pivot_table(df, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])
print(pd.pivot_table(df, index = 'Survived', columns = 'Pclass', values = "Embarked", aggfunc ='count' ))
print()
print(pd.pivot_table(df, index = 'Survived', columns = 'Sex', values = "Embarked", aggfunc ='count' ))
print()
#print(pd.pivot_table(df, index = 'Survived', columns = 'Ticket', values = "Embarked", aggfunc ='count' ))
#print()
#print(pd.pivot_table(df, index = 'Survived', columns = 'Cabin', values = "Embarked", aggfunc ='count' ))
#print()
print(pd.pivot_table(df, index = 'Survived', columns = 'Embarked', values = "Ticket", aggfunc ='count' ))
for i in categorical.columns:
    sns.barplot(categorical[i].value_counts().index, categorical[i].value_counts()).set_title(i)
    plt.show()
df_test.head()
df = df.drop(columns = ['Cabin', 'Ticket'])
from sklearn.preprocessing import OneHotEncoder
encode = OneHotEncoder(handle_unknown='ignore')

categorical = categorical.drop(columns= ['Cabin', 'Ticket'])

categorical.info()
categorical = categorical.dropna(subset = ['Embarked'])
df = df.dropna(subset = ['Embarked'])
df_01= pd.get_dummies(df, columns = ['Pclass','Sex', 'Embarked'])

df_01 = pd.DataFrame(df_01)
df_01.Age = df_01.Age.fillna(df.Age.median())


df_01 = df_01.drop(columns = ['Name'])
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df_02 = df_01.copy()

df_02[['Age', 'SibSp', 'Parch', 'Fare']] = scale.fit_transform(df_02[['Age', 'SibSp', 'Parch', 'Fare']])

x_train = df_02.drop(columns = ['Survived'])
y_train = df_02['Survived']
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
lr = LogisticRegression(max_iter = 2000) 
cv = cross_val_score(lr, x_train, y_train, cv = 5)
print(cv)
print(cv.mean())
knn = KNeighborsClassifier()
cv = cross_val_score(knn, x_train, y_train, cv = 5)
print(cv)
print(cv.mean())
svm = SVC(probability = True)
cv = cross_val_score(svm, x_train, y_train, cv =5)
print(cv)
print(cv.mean())
#df_test = df_test.drop(columns =['Cabin', 'Ticket'])
df_test.head()

df_test = df_test.dropna(subset = ['Embarked'])
df_03= pd.get_dummies(df_test, columns = ['Pclass','Sex', 'Embarked'])

df_03 = pd.DataFrame(df_02)
#df_03 = df_02.drop(columns = ['Name'])
df_03.Age = df_03.Age.fillna(df.Age.median())
df_03.Fare = df_03.Fare.fillna(df.Fare.median())

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df_04 = df_03.copy()

df_04[['Age', 'SibSp', 'Parch', 'Fare']] = scale.fit_transform(df_04[['Age', 'SibSp', 'Parch', 'Fare']])

x_test = df_04
x_test.info()
lr.fit(x_train, y_train)
logistic = lr.predict(x_test).astype(int)
final_data = {'PassengerId': x_test.PassengerId, 'Survived': logistic}
submission = pd.DataFrame(data=final_data)
submission.to_csv('submission', index = False)
submission.head()

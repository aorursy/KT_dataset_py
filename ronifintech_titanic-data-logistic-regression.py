import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.head()
data.describe()
data.info()
data.isnull().head(20)

#Age has a few Nulls and Cabin has a lot of Nulls
#heatmap to visualize the missing values
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=data)
sns.countplot(x='Survived', data=data, hue='Sex', palette='PRGn')
sns.countplot(x='Survived', data=data, hue='Pclass')

sns.distplot(data['Age'].dropna(), kde=False, bins=30)
sns.countplot(x='SibSp', data=data)
data['Fare'].hist(bins=40, figsize=(10,4))
plt.figure(figsize=(10,10))
sns.boxplot(x='Pclass', y='Age', data=data)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 24
    return Age
data['Age'] = data[['Age','Pclass']].apply(impute_age, axis=1)
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# too many values are missing... can't really do anything with this column
data.drop('Cabin', axis=1, inplace=True)
#drop rows with any NAs if any are left
data.dropna(inplace=True)
#change categorical columns into 
sex = pd.get_dummies(data['Sex'],drop_first=True)
embark = pd.get_dummies(data['Embarked'],drop_first=True)
data=pd.concat([data, sex, embark], axis=1)
#drop all non numerical columns
data.drop(['Name', 'Sex','Ticket','Embarked'], axis=1, inplace=True)

#this column is essentailly the index of the DataFrame
data.drop(['PassengerId'], axis=1, inplace=True)
data.head()
X = data.drop(['Survived'], axis=1)
Y = data['Survived']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

pred= pd.DataFrame(predictions, columns=['Survived'])
pred['model']='Logistic Reg'
y_test_df= pd.DataFrame(y_test)
y_test_df.reset_index(inplace=True)
y_test_df.drop(['index'], axis=1, inplace=True)
y_test_df['model']='Test Data'
pred_test = pred.append(y_test_df)
pred_test.tail()
plt.figure(figsize=(10,8))
sns.countplot(x='Survived', hue='model', data=pred_test)





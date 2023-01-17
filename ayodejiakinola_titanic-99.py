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
titanic_train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_train_data.head()
titanic_train_data.dtypes
titanic_train_data.isnull().values.any()
titanic_train_data.apply(lambda x : sum(x.isnull()),axis=0)
titanic_train_data.shape
import matplotlib.pyplot as plt
titanic_train_data['Pclass'].value_counts().plot(kind='bar')

plt.xlabel("Pclass")

plt.ylabel("Frequency")

plt.show()
table=pd.crosstab(titanic_train_data['Pclass'], titanic_train_data['Survived'])

table.head()

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
titanic_train_data['Sex'].value_counts().plot(kind='bar')

plt.xlabel("Sex")

plt.ylabel("Frequency")

plt.show()
table=pd.crosstab(titanic_train_data['Sex'], titanic_train_data['Survived'])

table.head()

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
titanic_train_data['Age'].hist(bins=10)

plt.show()
titanic_train_data['AGE_BINNED']=pd.cut(titanic_train_data['Age'],10)
table=pd.crosstab(titanic_train_data['AGE_BINNED'], titanic_train_data['Survived'])

table.head()

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
titanic_train_data['SibSp'].value_counts().plot(kind='bar')

plt.show()
table=pd.crosstab(titanic_train_data['SibSp'], titanic_train_data['Survived'])

table.head()

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
titanic_train_data['Parch'].value_counts().plot(kind='bar')

plt.show()
table=pd.crosstab(titanic_train_data['Parch'], titanic_train_data['Survived'])

table.head()

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
plt.scatter(titanic_train_data['Parch'],titanic_train_data['Fare'])

plt.show()
titanic_train_data['Fare'].hist(bins=10)

plt.show()
titanic_train_data['Fare_BINNED']=pd.cut(titanic_train_data['Fare'],10)
table=pd.crosstab(titanic_train_data['Fare_BINNED'], titanic_train_data['Survived'])

table.head()

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()
titanic_train_data['Embarked'].value_counts().plot(kind='bar')

plt.show()
table=pd.crosstab(titanic_train_data['Embarked'], titanic_train_data['Survived'])

table.head()

table.plot(kind='bar',stacked=True,figsize=(6,6))

plt.show()



DF_classification = titanic_train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]

DF_classification.head()
type(DF_classification)
import seaborn as sns



DF_classificationC = titanic_train_data[['Age','Fare']].corr()

# plot the heatmap and annotation on it

sns.heatmap(DF_classificationC, xticklabels=DF_classificationC.columns, yticklabels=DF_classificationC.columns, annot=True)
target_feature = pd.DataFrame(DF_classification['Survived'])

target_feature.head()
DF_classification = DF_classification.drop(['Survived'],axis=1)

DF_classification.head()
def M_F(value):

    if value == "male":

        return 1

    else:

        return 0
DF_classification['Sex'] = DF_classification['Sex'].apply(M_F)

DF_classification = pd.get_dummies(DF_classification)

#DF_classification = pd.DataFrame(DF_classification)

DF_classification.head()
from sklearn import preprocessing

DF_classification_column_names = DF_classification.columns.values

DF_classification_np = preprocessing.minmax_scale(DF_classification)

DF_classificationHH = pd.DataFrame(DF_classification_np, columns=DF_classification_column_names)



DF_classificationHH.head()
#from sklearn.model_selection import train_test_split



#X_train,X_test,Y_train,Y_test = train_test_split(DF_classificationHH,target_feature,test_size=0.1, random_state=42)
from sklearn.impute import SimpleImputer



X_train = SimpleImputer().fit_transform(DF_classificationHH)

Y_train = target_feature
from sklearn.tree import DecisionTreeClassifier

clf_Tree = DecisionTreeClassifier(criterion='entropy',random_state=0)

clf_Tree

clf_Tree.fit(X_train,Y_train)
titanic_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

titanic_test_data.head()
DF_classification = titanic_test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]





def M_F(value):

    if value == "male":

        return 1

    else:

        return 0



DF_classification['Sex'] = DF_classification['Sex'].apply(M_F)

DF_classification = pd.get_dummies(DF_classification)

#DF_classification = pd.DataFrame(DF_classification)

DF_classification.head()



from sklearn import preprocessing

DF_classification_column_names = DF_classification.columns.values

DF_classification_np = preprocessing.minmax_scale(DF_classification)

DF_classificationHH = pd.DataFrame(DF_classification_np, columns=DF_classification_column_names)



DF_classificationHH.head()



from sklearn.impute import SimpleImputer



X_test = SimpleImputer().fit_transform(DF_classificationHH)

#Y_train = target_feature
predicted = clf_Tree.predict(X_test)
titanic_subm_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

titanic_subm_data.head()
output = pd.DataFrame({'PassengerId': titanic_test_data['PassengerId'], 'Survived': predicted})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
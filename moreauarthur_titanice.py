import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tr_data = pd.read_csv('/kaggle/input/titanic/train.csv')

tr_data.head()
te_data = pd.read_csv('/kaggle/input/titanic/test.csv')

te_data.head()
cols = tr_data.columns

print(cols)
tr_data.info()
tr_data['Age'].describe()
mean_age = tr_data["Age"].mean()

print(mean_age)

tr_data['Age'] = tr_data['Age'].fillna(mean_age) 

te_data['Age'] = te_data['Age'].fillna(mean_age) 


tr_data.loc[ tr_data['Age'] <= 16, "Age"] = 1

tr_data.loc[(tr_data['Age'] > 16) & (tr_data['Age'] <= 32), "Age"] = 2

tr_data.loc[(tr_data['Age'] > 32) & (tr_data['Age'] <= 48), "Age"] = 3

tr_data.loc[(tr_data['Age'] > 48) & (tr_data['Age'] <= 64), "Age"] = 4

tr_data.loc[ tr_data['Age'] > 64, "Age"] = 5



te_data.loc[ te_data['Age'] <= 16, "Age"] = 1

te_data.loc[(te_data['Age'] > 16) & (te_data['Age'] <= 32), "Age"] = 2

te_data.loc[(te_data['Age'] > 32) & (te_data['Age'] <= 48), "Age"] = 3

te_data.loc[(te_data['Age'] > 48) & (te_data['Age'] <= 64), "Age"] = 4

te_data.loc[ te_data['Age'] > 64, "Age"] = 5

    

tr_data["Age"].isnull().sum()

te_data["Age"].isnull().sum()

women = tr_data.loc[tr_data['Sex']=='female']["Survived"]

rate_women = sum(women)/len(women)



print('% of women who survived:', rate_women)
men = tr_data.loc[tr_data['Sex']=='male']['Survived']

rate_men = sum(men)/len(men)



print('% of men who survived:', rate_men)
name = tr_data['Name']



#for tr_data

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in tr_data["Name"]]

tr_data["Title"] = pd.Series(dataset_title)

tr_data["Title"] = tr_data["Title"].replace(["Master", "Dr", "Rev", "Col", "Major", "the Countess", "Capt", "Jonkheer", "Lady", "Sir","Don", "Dona", "Mlle", "Ms", "Mme"], 'Rare')

tr_data["Title"] = tr_data["Title"].replace(["Mr", "Miss", "Mrs"], 'Common')



#for te_data

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in te_data["Name"]]

te_data["Title"] = pd.Series(dataset_title)

te_data["Title"] = te_data["Title"].replace(["Master", "Dr", "Rev", "Col", "Major", "the Countess", "Capt", "Jonkheer", "Lady", "Sir", "Don", "Dona", "Mlle", "Ms", "Mme"], 'Rare')

te_data["Title"] = te_data["Title"].replace(["Mr", "Miss", "Mrs"], 'Common')

     

print(tr_data["Title"].value_counts())

print(te_data["Title"].value_counts())
# #for train set

# tr_data['FamilySize'] = tr_data['SibSp'] + tr_data['Parch'] +1



# #for test set

# te_data['FamilySize'] = tr_data['SibSp'] + tr_data['Parch'] +1



# tr_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# list = []



# for i in tr_data['FamilySize']:

#     if i in [4, 3, 2, 7]:

#         list.append("ideal")

#     elif i ==1:

#         list.append("alone")

#     else:

#         list.append("non_ideal")

        

# tr_data["IdealFamilySize"] = list



# #for te_data

# list2 = []



# for i in te_data['FamilySize']:

#     if i in [4, 3, 2, 7]:

#         list2.append("ideal")

#     elif i ==1:

#         list2.append("alone")

#     else:

#         list2.append("non_ideal")

        

# te_data["IdealFamilySize"] = list2

        

# print(te_data['FamilySize'].head(10))

# print(te_data['IdealFamilySize'].head(10))
#for tr_data

cabin_list = []



tr_data['Cabin'] = tr_data['Cabin'].fillna('None')



for i in tr_data['Cabin']:

    if i == 'None':

        cabin_list.append('None')

    else:

        cabin_list.append('Cabin')



tr_data['Cabin'] = cabin_list



#for te_data

cabin_list2 = []



te_data['Cabin'] = te_data['Cabin'].fillna('None')



for i in te_data['Cabin']:

    if i == 'None':

        cabin_list2.append('None')

    else:

        cabin_list2.append('Cabin')



te_data['Cabin'] = cabin_list2



te_data['Cabin'].value_counts()
#tr_data['Fare'] = pd.qcut(tr_data['Fare'], 4)

#tr_data[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True)



map(float(), tr_data['Fare'])



print(tr_data['Fare'].head())
#for tr_data

fare_list = []



for i in tr_data['Fare']:

    i = float(i)

    if i <= 7.91:

        fare_list.append(1)

    elif i <= 14.454:

        fare_list.append(2)

    elif i <= 31.0:

        fare_list.append(3)

    else:

        fare_list.append(4)

        

tr_data['Fare'] = fare_list



#for te_data

fare_list2 = []



for i in te_data['Fare']:

    i = float(i)

    if i <= 7.91:

        fare_list2.append(1)

    elif i <= 14.454:

        fare_list2.append(2)

    elif i <= 31.0:

        fare_list2.append(3)

    else:

        fare_list2.append(4)

        

te_data['Fare'] = fare_list2



print(tr_data['Fare'].head())

print(te_data['Fare'].head())
tr_data = tr_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

te_data = te_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
features = ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked', "Title", "Fare"]



for i in features:

    tr_data [i] = pd.get_dummies(tr_data[i])

    te_data [i] = pd.get_dummies(te_data[i])



print(tr_data.head())
df = pd.DataFrame(tr_data,columns=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked', "Title"])

corrMatrix = df.corr()

print (corrMatrix)
# tr_data['IntAgeParch'] = tr_data['Age'] * tr_data['Parch']

# tr_data['IntSibParch'] = tr_data['SibSp'] * tr_data['Parch']

tr_data['IntSibFare'] = tr_data['Age'] * tr_data['Parch']

te_data['IntSibFare'] = te_data['Age'] * te_data['Parch']
print(tr_data['Survived'].head(10))
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC



y = tr_data['Survived']



features = ['Pclass', 'Sex', 'Age', 'SibSp','Parch','Embarked', "Title"]

X = tr_data[features]

X_test = te_data[features]



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#rfc=RandomForestClassifier(random_state=42)
# param_grid = { 

#     'n_estimators': [200, 500],

#     'max_features': ['auto', 'sqrt', 'log2'],

#     'max_depth' : [4,5,6,7,8],

#     'criterion' :['gini', 'entropy']

# }
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

# CV_rfc.fit(x_train, y_train)
# CV_rfc.best_params_
# model = RandomForestClassifier(max_depth= 4, max_features = 'auto', n_estimators = 500, criterion= 'entropy')

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': te_data.PassengerId, 'Survived':predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully save! :D")
import tensorflow as tf

import numpy as np

from tensorflow import keras
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 

                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 

                                    tf.keras.layers.Dense(2, activation=tf.nn.softmax)])
model.compile(optimizer='sgd', loss='mean_squared_error')
x_train_list = x_train.values.tolist()

y_train_list = y_train.values.tolist()

x_test_list = x_test.values.tolist()

y_test_list = y_test.values.tolist()
X_list = x_train.values.tolist()

y_list = y_train.values.tolist()
model.fit(X_list, y_list, epochs=100)
X_test_list = X_test.values.tolist()

print(len(X_test_list))
predictions = model.predict(X_test_list)
df_pred = pd.DataFrame(predictions)



pred_final = []



for i, j in df_pred.iterrows():

    if j[0] > j[1]:

         pred_final.append(1)

    else:

         pred_final.append(0)

        

print(pred_final[1:10])
output = pd.DataFrame({'PassengerId': te_data.PassengerId, 'Survived':pred_final})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully save! :D")
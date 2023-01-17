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
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import Perceptron
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

# 読み込むデータが格納されたディレクトリのパス，必要に応じて変更の必要あり

path = "/kaggle/input/titanic/"

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
train["train"] = 1
test["train"] = 0
combine = pd.concat([train, test])
combine["Sex"] = combine["Sex"].replace("male", "0").replace("female", "1")
combine["Sex"] = combine["Sex"].astype(int)

combine["Age"].fillna(combine.Age.mean(), inplace=True) 

combine['honorific'] = combine['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
combine['honorific'] = combine['honorific'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combine['honorific'] = combine['honorific'].replace('Mlle', 'Miss')
combine['honorific'] = combine['honorific'].replace('Ms', 'Miss')
combine['honorific'] = combine['honorific'].replace('Mme', 'Mrs')
Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 
combine['honorific'] = combine['honorific'].map(Salutation_mapping) 
combine['honorific'] = combine['honorific'].fillna(0) 

combine["FamilySize"] = combine["SibSp"] + combine["Parch"] + 1

combine['IsAlone'] = 0
combine.loc[combine['FamilySize'] == 1, 'IsAlone'] = 1

combine['Ticket_Alphabet'] = combine['Ticket'].apply(lambda x: str(x)[0])
combine['Ticket_Alphabet'] = combine['Ticket_Alphabet'].apply(lambda x: str(x)) 
combine['Ticket_Alphabet'] = np.where((combine['Ticket_Alphabet']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), combine['Ticket_Alphabet'], np.where((combine['Ticket_Alphabet']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 
combine['Ticket_Alphabet']=combine['Ticket_Alphabet'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3) 
combine['Ticket_Len'] = combine['Ticket'].apply(lambda x: len(x)) 
    
combine['Cabin_Alphabet'] = combine['Cabin'].apply(lambda x: str(x)[0]) 
combine['Cabin_Alphabet'] = combine['Cabin_Alphabet'].apply(lambda x: str(x)) 
combine['Cabin_Alphabet'] = np.where((combine['Cabin_Alphabet']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),combine['Cabin_Alphabet'], np.where((combine['Cabin_Alphabet']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))
combine['Cabin_Alphabet']=combine['Cabin_Alphabet'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1) 

combine['Fare'].fillna(combine['Fare'].median(), inplace = True)

combine['Embarked'] = combine['Embarked'].fillna(combine["Embarked"].mode()[0])
combine['Embarked'] = combine['Embarked'].replace('S', "0").replace( 'C', "1").replace('Q', "2")
combine['Embarked'] = combine['Embarked'].astype(int)
combine["Embarked"].fillna(combine.Embarked.mean(), inplace=True) 

train = combine[combine["train"]==1]
test = combine[combine["train"]==0]
train.drop(["PassengerId", "Name", "Ticket", "Cabin", "train"], axis=1, inplace=True)
test.drop(['PassengerId', "Name", "Ticket", "Cabin", "train"], axis=1, inplace=True)
y_train  = train["Survived"]  
X_train = train.drop(["Survived"], axis=1)
X_test = test.drop(["Survived"], axis=1)
random_forest=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=51, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

random_forest.fit(X_train, y_train)
Y_pred_rf = random_forest.predict(X_test)
path = "/kaggle/input/titanic/"

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

train["train"] = 1
test["train"] = 0
data = pd.concat([train, test])
data["Name"] = data["Name"].apply(lambda x: str(x)[x.find(",")+2 : x.find(".")])
data["Female_boy"] = 0
data.loc[(data["Sex"]=="female")|(data["Name"]=="Master"), "Female_boy"] = 1

data = data[["PassengerId", 'Survived', 'Ticket', 'Female_boy', "train"]]
data = data[data['Female_boy']==1]
index = data.index
num_Ticket = data.Ticket.value_counts()
num_Ticket = pd.DataFrame(num_Ticket)
num_Ticket.reset_index(inplace=True)
num_Ticket.rename({'index': 'Ticket', 'Ticket':'num_Ticket'}, inplace=True, axis=1)
data = pd.merge(data, num_Ticket, on=['Ticket'], how='left')
data['Ticket_sv'] = data.groupby('Ticket')['Survived'].transform('mean')
data.index = index
test_data = data[data["train"]==0]
test_data = test_data[["PassengerId", 'Ticket_sv']]

submission_rf = pd.read_csv(path + "gender_submission.csv")
submission_rf["Survived"] = Y_pred_rf
submission_rf = submission_rf.merge(test_data, on=["PassengerId"], how="left")
submission_rf.loc[submission_rf['Ticket_sv']==1.0, "Survived"] = 1
submission_rf.loc[submission_rf['Ticket_sv']==0.0, "Survived"] = 0
del submission_rf['Ticket_sv']
submission_rf
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
train["train"] = 1
test["train"] = 0
combine = pd.concat([train, test])
combine["Sex"] = combine["Sex"].replace("male", "0").replace("female", "1")
combine["Sex"] = combine["Sex"].astype(int)

combine['honorific'] = combine['Name']
for name_string in combine['Name']:
    combine['honorific'] = combine['Name'].str.extract('([A-Za-z]+)\.', expand=True)

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss','Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
combine.replace({'honorific': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = combine.groupby('honorific')['Age'].median()[titles.index(title)]
    combine.loc[(combine['Age'].isnull()) & (combine['honorific'] == title), 'Age'] = age_to_impute
combine.drop('honorific', axis = 1, inplace = True)

combine['Family_Size'] = combine['Parch'] + combine['SibSp']
combine['Family_Name'] = combine['Name'].apply(lambda x: str.split(x, ",")[0])

pre = 0.5
combine['Family_Survival'] = pre
for grp, grp_df in combine[['Survived','Name', 'Family_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Family_Name', 'Fare']):
    if (len(grp_df) != 1):
        # len(grp_df) != 1なら家族がいるって事
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                combine.loc[combine['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                combine.loc[combine['PassengerId'] == passID, 'Family_Survival'] = 0
                
for _, grp_df in combine.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    combine.loc[combine['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    combine.loc[combine['PassengerId'] == passID, 'Family_Survival'] = 0


label = LabelEncoder()
combine['Fare'].fillna(combine['Fare'].median(), inplace = True)
combine['Farelabal'] = pd.qcut(combine['Fare'], 5)
combine['Farelabal'] = label.fit_transform(combine['Farelabal'])

combine['Agelabel'] = pd.qcut(combine['Age'], 4)
combine['Agelabel'] = label.fit_transform(combine['Agelabel'])


train = combine[combine["train"]==1]
test = combine[combine["train"]==0]

train.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'train', 'Family_Name'], axis = 1, inplace = True)
test.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin','Embarked', 'train', 'Family_Name'], axis = 1, inplace = True)

X = train.drop('Survived',  axis = 1)
y = train['Survived']
X_test = test.drop('Survived',  axis = 1)
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_test = std_scaler.transform(X_test)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                           weights='uniform')
knn.fit(X, y)
y_pred_knn = knn.predict(X_test)
path = "/kaggle/input/titanic/"

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

train["train"] = 1
test["train"] = 0
data = pd.concat([train, test])
data["Name"] = data["Name"].apply(lambda x: str(x)[x.find(",")+2 : x.find(".")])
data["Female_boy"] = 0
data.loc[(data["Sex"]=="female")|(data["Name"]=="Master"), "Female_boy"] = 1

data = data[["PassengerId", 'Survived', 'Ticket', 'Female_boy', "train"]]
data = data[data['Female_boy']==1]
index = data.index
num_Ticket = data.Ticket.value_counts()
num_Ticket = pd.DataFrame(num_Ticket)
num_Ticket.reset_index(inplace=True)
num_Ticket.rename({'index': 'Ticket', 'Ticket':'num_Ticket'}, inplace=True, axis=1)
data = pd.merge(data, num_Ticket, on=['Ticket'], how='left')
data['Ticket_sv'] = data.groupby('Ticket')['Survived'].transform('mean')
data.index = index
test_data = data[data["train"]==0]
test_data = test_data[["PassengerId", 'Ticket_sv']]

submission_knn = pd.read_csv(path + "gender_submission.csv")
submission_knn["Survived"] = y_pred_knn
submission_knn = submission_knn.merge(test_data, on=["PassengerId"], how="left")
submission_knn.loc[submission_knn['Ticket_sv']==1.0, "Survived"] = 1
submission_knn.loc[submission_knn['Ticket_sv']==0.0, "Survived"] = 0
del submission_knn['Ticket_sv']
submission_knn
sum_pred = submission_rf.merge(submission_knn, on=["PassengerId"], how="left")
sum_pred["Survived_x"] = sum_pred["Survived_x"].astype(int)
sum_pred["Survived_sum"] = (sum_pred["Survived_x"] + sum_pred["Survived_y"])/2
sum_pred["Survived_sum"].value_counts()
for i in range(418):
    if sum_pred["Survived_sum"][i]<=0.5:
        sum_pred["Survived_sum"][i]=0
    else:
        sum_pred["Survived_sum"][i]=1
sum_pred["Survived_sum"] = sum_pred["Survived_sum"].astype(int)        
sum_pred["Survived_sum"].value_counts()
submission = pd.DataFrame({
        "PassengerId": pre_test["PassengerId"],
        "Survived":pre_test["Survived_sum"]
    })

submission.to_csv('submission.Titanic_Best1.csv', index=False)
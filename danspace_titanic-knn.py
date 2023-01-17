# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train_len = len(train)
test_copy = test.copy()
# combine train and test set
total = train.append(test)
total.isnull().sum()
total[total.Fare.isnull()]
# fill the missing value for Fare column with median
total['Fare'].fillna(value = total[total.Pclass==3]['Fare'].median(), inplace = True)
import matplotlib.pyplot as plt
import seaborn as sns

# extract title from name, fill the missing value for Age column according to title's median
total['Title'] = total['Name'].str.extract('([A-Za-z]+)\.', expand=True)
# check distribution of title
plt.figure(figsize=(8,6))
sns.countplot(x= "Title",data = total)
plt.xticks(rotation='45')
plt.show()
# Replacing rare titles with more common ones
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
total.replace({'Title': mapping}, inplace=True)
# fill the missing value for Age column with median of its title
titles = list(total.Title.unique())
for title in titles:
    age = total.groupby('Title')['Age'].median().loc[title]
    total.loc[(total.Age.isnull()) & (total.Title == title),'Age'] = age
# add family size as a feature
total['Family_Size'] = total['Parch'] + total['SibSp']
# This feature is from S.Xu, https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever
total['Last_Name'] = total['Name'].apply(lambda x: str.split(x, ",")[0])
total['Fare'].fillna(total['Fare'].mean(), inplace=True)

default_survival_rate = 0.5
total['Family_Survival'] = default_survival_rate

for grp, grp_df in total[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                total.loc[total['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                total.loc[total['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      total.loc[total['Family_Survival']!=0.5].shape[0])
for _, grp_df in total.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    total.loc[total['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    total.loc[total['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(total[total['Family_Survival']!=0.5].shape[0]))

# add fare bins
total['Fare_Bin'] = pd.qcut(total['Fare'], 5,labels=False)
# add age bins
total['Age_Bin'] = pd.qcut(total['Age'], 4,labels=False)
# convert Sex to catergorical value
total.Sex.replace({'male':0, 'female':1}, inplace = True)

# only select the features we want
features = ['Survived','Pclass','Sex','Family_Size','Family_Survival','Fare_Bin','Age_Bin']
total = total[features]
# split total to train and test set
train = total[:train_len]
# set Survied column as int
x_train = train.drop(columns = ['Survived'])
y_train = train['Survived'].astype(int)

x_test = total[train_len:].drop(columns = ['Survived'])
# Scaling features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
params = {'n_neighbors':[6,8,10,12,14,16,18,20],
         'leaf_size':list(range(1,50,5))}

# Using ROC_AUC as metric has a better result than using accuracy. 
gs = GridSearchCV(clf, param_grid= params, cv = 5,scoring = "roc_auc",verbose=1)
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_estimator_)
preds = gs.predict(x_test)
result = pd.DataFrame({'PassengerId': test_copy['PassengerId'], 'Survived': preds})
result.to_csv('result.csv', index = False)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#Imports for learning
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
# Any results you write to the current directory are saved as output.
#Read data('../input/train.csv')
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_data_read = df_train.append(df_test,sort=False) # The entire data: train + test.
#Check for nulls
print(pd.isnull(df_data_read).sum())
#impute Embarked and Fare with most common value - not manny missing values for these two
#We are going to build a model to impute age
df_data_read['Embarked'].fillna(df_data_read['Embarked'].mode()[0], inplace = True)

#complete missing fare with median
df_data_read['Fare'].fillna(df_data_read['Fare'].median(), inplace = True)
print(pd.isnull(df_data_read).sum())
df_data = df_data_read
#%%ENGINEER SOME FEATURES
df_data["Title"] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#Unify common titles. 
df_data["Title"] = df_data["Title"].replace('Mlle', 'Mrs')
df_data["Title"] = df_data["Title"].replace('Master', 'Master')
df_data["Title"] = df_data["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')
df_data["Title"] = df_data["Title"].replace(['Jonkheer','Don'],'Mr')
df_data["Title"] = df_data["Title"].replace(['Capt','Major', 'Col','Rev','Dr'], 'Rare')
df_data["Title"] = df_data["Title"].replace(['Lady', 'Countess','Sir'], 'Rare')

#Family Size 
df_data['FamilySize'] = df_data['SibSp'] + df_data['Parch'] + 1
"""
Impute values for age using a linear model - identified Title, Sex and class as the most important variables
"""
lm_test = df_data[np.isnan(df_data.Age)]
lm_train = df_data.dropna(subset = ['Age'])
lm_train_X = lm_train[['Title','Sex','Pclass']]
lm_train_Y = lm_train[['Age']]
lm_train_X =  pd.get_dummies(lm_train_X)

X_train, X_test, y_train, y_test = train_test_split( lm_train_X, lm_train_Y, test_size=0.3, random_state=42)

lm = linear_model.LinearRegression()

# Train the model using the training sets
lm.fit(X_train, y_train)
print("Mean Square Error: {}".format(mean_squared_error(y_test,lm.predict(X_test))))

df_data['Age'] = lm.predict(pd.get_dummies(df_data[['Title','Sex','Pclass']]))

print(pd.isnull(df_data).sum())
"""
    Feature engineering
"""

# Name Length variables
df_data['Special_Name'] = df_data['Name'].str.contains('\(',na=False) # (), "" persons of note
df_data['Name_length'] = df_data['Name'].apply(len) # (), "" persons of note

#has cabin or not
df_data['Has_Cabin'] = df_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
# Extract Deck Letter - position on titantic
df_data["Deck"] = df_data.Cabin.str.extract('([A-Za-z])', expand=False)
df_data["Deck"] =df_data["Deck"].fillna("Lower")
df_data["Deck"] =df_data["Deck"].replace(['A','B','C'],'Upper')
df_data["Deck"] =df_data["Deck"].replace(['D','E','F','G'],'Middle')
df_data["Deck"] =df_data["Deck"].replace(['T'],'Lower')

#early or late buyer 0f a ticket... preparedness..?
def extract_number(str):
    ticket = [int(s) for s in str.split() if s.isdigit()]
    try:
        ticket = ticket[-1]
    except:
        ticket = "260000"
    return ticket

df_data['Ticket_Time']= df_data['Ticket'].map(lambda x : extract_number(x)).astype(int)
df_data['Ticket_Time'] = pd.qcut(df_data['Ticket_Time'], 3, labels=['Early','Middle','Late'])

#Lone travler
df_data['IsAlone'] = 0
df_data.loc[df_data['FamilySize'] == 1, 'IsAlone'] = 1

#How much per person was paid for the ticket?
df_data['Fare_per_family'] = df_data['Fare']/df_data['FamilySize']
df_data['Fare_per_family'] = df_data['Fare_per_family'].astype(int)
#df_data['Fare_per_family'] = pd.cut(df_data['Fare_per_family'],4)
          
#BIN FARE into 4 buckets
df_data.loc[ df_data['Fare'] <= 7.91, 'Fare'] 						        = 0
df_data.loc[(df_data['Fare'] > 7.91) & (df_data['Fare'] <= 14.454), 'Fare'] = 1
df_data.loc[(df_data['Fare'] > 14.454) & (df_data['Fare'] <= 31), 'Fare']   = 2
df_data.loc[ df_data['Fare'] > 31, 'Fare'] 							        = 3
df_data['Fare'] = df_data['Fare'].astype(int)

 # Mapping Age
df_data.loc[ df_data['Age'] <= 16, 'Age'] 					       = 0
df_data.loc[(df_data['Age'] > 16) & (df_data['Age'] <= 32), 'Age'] = 1
df_data.loc[(df_data['Age'] > 32) & (df_data['Age'] <= 48), 'Age'] = 2
df_data.loc[(df_data['Age'] > 48) & (df_data['Age'] <= 64), 'Age'] = 3
df_data.loc[ df_data['Age'] > 64, 'Age'] = 4 ;

df_data['AgeBin_Code'] = df_data['Age'].astype('category')
df_data['FareBin_Code'] = df_data['Fare'].astype('category')
df_data['Pclass_Code'] = df_data['Pclass'].astype('category')

df_data.head()
#Prepare data for modelling - one hot encoding of categorical variables
drop_columns = ['Name', 'Ticket', 'Cabin', 'SibSp']
df_data_tmp = df_data.drop(drop_columns,axis =1)

df_data_tmp = pd.get_dummies(df_data_tmp)

# Age in df_train and df_test:
df_train = df_data_tmp[:891].copy(deep=True)
df_test = df_data_tmp[891:].copy(deep=True)

#SET index to passenger ID
df_test.set_index('PassengerId', inplace=True)
df_test = df_test.drop('Survived', axis=1)

print(df_test.head())

print("possible cols", list(df_test))
# g = sns.pairplot(df_train[sel_cols,'Survived'], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
# g.set(xticklabels=[])

X_tmp = df_train.drop('Survived', axis=1)
X_tmp = X_tmp.drop('PassengerId', axis=1)

#keep playing aroun with n_est and states until you have a test accuracy > 83
#need to do KFLOLD
sel_cols =['Sex_female', 'Sex_male', 'Title_Mr', 'Name_length', 'Fare_per_family', 'Pclass', 'Title_Miss', 'Pclass_Code_3', 'Special_Name', 'Deck_Lower', 'Has_Cabin', 'FamilySize', 'Title_Mrs', 'Age']
X=X_tmp#[sel_cols]
Y = df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=41)
#max depth of 4 for each tree to stop overfitting
model = RandomForestClassifier(n_estimators=1500,max_depth=4,oob_score=True)

model.fit(X_train, y_train)
print('Evaluation complete')
# Print the accuracy# Print 
print("Accuracy Train: {}".format(model.score(X_train, y_train)))
print("Accuracy Test: {}".format(model.score(X_test, y_test)))

#Feature importance
importances = model.feature_importances_

sort_import = np.argsort(importances)[::-1]

cols_to_use = []
print("\nRanked Importance of Columns:")
for i in sort_import:
    print(list(X)[i],importances[i])
    if importances[i] > 0.021:
       cols_to_use.append(list(X)[i]) 
print(cols_to_use)
Submission = df_test.copy(deep = True)
Submission.head()
Submission['Survived']= model.predict(Submission[list(X)].copy())
Submission['Survived'].to_csv('submission.csv', index=True)

# with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
#     print(Submission['Survived'])
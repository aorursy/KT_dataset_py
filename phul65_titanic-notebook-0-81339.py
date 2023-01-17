# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, normalize

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
ROOT_DIR = "../input/"
SUBINT_DIR = "./"
print ("Lecture des fichiers train.csv")
train = pd.read_csv(ROOT_DIR+"train.csv",sep=',')
print(train.shape)
print ("Lecture des fichiers test.csv")
test = pd.read_csv(ROOT_DIR+"test.csv",sep=',')
print(test.shape)
print ("Lecture des fichiers gender_submission.csv")
submit = pd.read_csv(ROOT_DIR+"gender_submission.csv",sep=',')
print(submit.shape)
print("Concatenation train + test")
train['X'] = 'X'
test['X'] = 'Y'
big = pd.concat([train,test])
# Nom exemple : Hirvonen, Mrs. Alexander (Helga E Lindqvist)
# XName : Hirvonen
# TName : Mrs.
big['XName'] = big['Name'].apply(lambda x: str(x)[0:str(x).find(',')] if str(x).find(',') != -1 else x)
big['TName'] = big['Name'].apply(lambda x: str(x)[str(x).find(',')+2:str(x).find('. ')+1:] if str(x).find('. ') != -1 else x)

big['XCabin'] = big['Cabin'].apply(lambda x: 'U' if (x is np.nan or x != x) else str(x)[0])

big['LTick'] = big['Ticket'].apply(lambda x: str(x)[0:str(x).find(' ')] if str(x).find(' ') != -1 else ' ')
K = big.groupby(['Ticket']).groups
#display(K)
for name,group in K.items():
    if len(group) > 1:
        CN = list(set([ str(x)[0] for x in big['Cabin'].iloc[group] ]) - set(['n']))
        if (len(CN) == 0):
            big['XCabin'].iloc[group] = 'U'
        else:
            big['XCabin'].iloc[group] = CN[0]
big['XFam'] = big['SibSp'] + big['Parch'] + 1
big['XFam'] = np.log1p((big['XFam'] - big['XFam'].mean()) / big['XFam'].std())
big['Last_Name'] = big['Name'].apply(lambda x: str.split(x, ",")[0])
big['Fare'].fillna(big['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
big['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in big[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                        'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                big.loc[big['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                big.loc[big['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", big.loc[big['Family_Survival']!=0.5].shape[0])
for _, grp_df in big.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    big.loc[big['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    big.loc[big['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: "+str(big[big['Family_Survival']!=0.5].shape[0]))
del big['Ticket'], big['Cabin'], big['Name'], big['XName'], big['Last_Name']
print(big['TName'].value_counts())

big['XWho'] = big['TName']

for i in [ 'Master.', 'Sir.', 'Don.', 'Lady.', 'Dona.', 'the Countess.', 'Mme.' ]:
    big['XWho'][big['TName'] == i] = "High."
    
for i in [ 'Col.', 'Major.', 'Capt.' ]:
    big['XWho'][big['TName'] == i] = "Mil."
    
for i in [ 'Mr.', 'Dr.', 'Rev.' ]:
    big['XWho'][big['TName'] == i] = "Mr."
    
for i in [ 'Mrs.', 'Ms.', 'Mlle.', 'Miss.' ]:
    big['XWho'][big['TName'] == i] = "Miss."

big['XWho'][~big['TName'].isin([ 'Sir.', 'Don.', 'Lady.', 'Dona.', 'the Countess.', 'Col.', 
                                'Major.', 'Capt.', 'Mr.', 'Master.', 'Dr.', 'Rev.', 'Mrs.', 
                                'Ms.', 'Mlle.', 'Mme.', 'Miss.' ])] = "Oth."
            
print(big['XWho'].value_counts())
for col in [ 'Sex', 'Pclass', 'XWho', 'Embarked', 'LTick', 'XCabin', 'TName' ]:
    dummy = pd.get_dummies(big[col],prefix=str(col),prefix_sep="__")
    big = pd.concat([big, dummy], axis=1)
    big.drop(col, inplace=True, axis=1) 
    
for col in [ 'XFam' ]:
    lbl = LabelEncoder()
    lbl.fit(list(big[col].values))
    big[col] = lbl.transform(list(big[col].values))
CNULL = big.isnull().sum()
print(CNULL[CNULL != 0])
big['Fare'] = big['Fare'].fillna(big['Fare'].mean())
big['Age'] = big['Age'].fillna(big['Age'].mean())
train = big[big['X'] == 'X']
test = big[big['X'] == 'Y']

del test['Survived']
del train['X'], test['X']

print(train.shape)
print(test.shape)
train['Age'] = (train['Age'] - train['Age'].mean()) / train['Age'].std()
test['Age'] = (test['Age'] - test['Age'].mean()) / test['Age'].std()
train['Fare'] = np.log1p((train['Fare'] - train['Fare'].mean()) / train['Fare'].std())
test['Fare'] = np.log1p((test['Fare'] - test['Fare'].mean()) / test['Fare'].std())
# Add colonnes statistiques
train['c_mean'] = pd.Series(train.mean(axis=1), index=train.index)
c_mean_max = train['c_mean'].max()
c_mean_min = train['c_mean'].min()
c_mean_scaled = (train.c_mean-c_mean_min) / c_mean_max
train['c_mean_s'] = pd.Series(c_mean_scaled, index=train.index)
del train['c_mean']

train['c_std'] = pd.Series(train.std(axis=1), index=train.index)
c_std_max = train['c_std'].max()
c_std_min = train['c_std'].min()
c_std_scaled = (train.c_std-c_std_min) / c_std_max
train['c_std_s'] = np.log1p(pd.Series(c_std_scaled, index=train.index))
del train['c_std']

test['c_mean'] = pd.Series(test.mean(axis=1), index=test.index)
c_mean_max = test['c_mean'].max()
c_mean_min = test['c_mean'].min()
c_mean_scaled = (test.c_mean-c_mean_min) / c_mean_max
test['c_mean_s'] = np.log1p(pd.Series(c_mean_scaled, index=test.index))
del test['c_mean']

test['c_std'] = pd.Series(test.std(axis=1), index=test.index)
c_std_max = test['c_std'].max()
c_std_min = test['c_std'].min()
c_std_scaled = (test.c_std-c_std_min) / c_std_max
test['c_std_s'] = pd.Series(c_std_scaled, index=test.index)
del test['c_std']

print(train.shape, test.shape)
print(train.shape)
train.drop_duplicates(inplace=True)
print(train.shape)
import time
import datetime

print (" <*> Debut")

kDate = time.strftime('%d%m%y_%H%M%S',time.localtime())

start = time.time()

y_train = train['Survived'].values.astype(np.float64)
x_train = train.drop(['PassengerId', 'Survived'], axis=1).values.astype(np.float64)
x_test  = test.drop(['PassengerId'], axis=1).values.astype(np.float64)

print('Shape train: {}\nShape test: {}\nShape Y: {}'.format(x_train.shape, x_test.shape, y_train.shape))

NSplit = 5
SliceTrain = 0.75
SliceTest  = 0.25
models = []
NIter = 0
TScore = 0

print("Entrainement")
rs = StratifiedShuffleSplit(n_splits=NSplit, random_state=99, test_size=SliceTest) 
for train_index, test_index in rs.split(x_train, y_train):
    
    X_train = x_train[train_index]
    Y_train = y_train[train_index]
    X_valid = x_train[test_index]
    Y_valid = y_train[test_index]

    rfc_params = {}
    rfc_params['n_estimators'] = 200  
    rfc_params['learning_rate'] = 0.015
    rfc_params['max_depth'] = 250   
    rfc_params['max_features'] = "auto"
    rfc_params['min_samples_split'] = 0.7
    rfc_params['min_samples_leaf'] = 0.01    
    rfc_params['random_state'] = 0
    rfc_params['verbose'] = 0   
    
    sum_score = 0
    score     = 0
    
    clf = GradientBoostingClassifier(**rfc_params)
    clf.fit(X_train, Y_train)
    models.append(clf)

    score = clf.score(X_valid, Y_valid)
    print (" <*> Entrainement ",NIter," avec ", SliceTrain, " pour train et ",SliceTest," pour test - Score : ", score)
    TScore += score
    NIter += 1
    
TScore /= NSplit

print(" <*> ---------------- Resultats CV ------------------ ")
print(" <*> params : ",rfc_params)
print(" <*> Score Moyenne training : ", TScore)
    
print("Verification avec le train")
score = 0
SCLOG = 0
NIter = 0
for clf in models:

    PTrain = clf.predict(x_train)
    score  = clf.score(x_train, y_train)
    SCLOG += score
    NIter += 1

SCLOG /= NSplit
print(" <*> Score Moyenne Train    : ", SCLOG)
       
print("Predictions")
NIter = 0
ctb_pred1 = []
for clf in models:

    PTest = clf.predict(x_test)
    ctb_pred1.append(PTest)

    NIter += 1

PTest = [0] * len(ctb_pred1[0])
for i in range(NSplit):
    PTest += ctb_pred1[i]
PTest /= NSplit

print( pd.DataFrame(PTest).head() )        
    
end = time.time()
print (" <*> Duree : ",end - start)
    
print (" <*> Fin")
# Submit resultats
print( " Mise a jour des colonnes submit" )
submit['Survived'] = np.clip(PTest, 0, 1).astype(int) 
localtime = time.localtime(time.time())
WDate = str(localtime.tm_mday).rjust(2, '0')+str(localtime.tm_mon).rjust(2, '0')+str(localtime.tm_year)

SUBFIC = SUBINT_DIR+"Titanic_GBR_"+str(kDate)+".csv"
print (" <*> Ecriture deb CSV/7z : ", SUBFIC)
submit.to_csv(SUBFIC, index=False) 
print (" <*> Ecriture fin CSV/7z : ", SUBFIC)
# read csv files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/train.csv')

df.head()
test_df = pd.read_csv('../input/test.csv')
test_df.head()
surv_col = df.iloc[:,1]
surv_col.head()
train_df = df.iloc[:,2:]
train_df.head()
pessengerId = test_df.iloc[:,0]
pessengerId.head()
test_df.drop(['PassengerId'],axis=1,inplace=True)

test_df.head()
train_df.head()
concated_df = pd.concat([train_df,test_df])

concated_df.head()
concated_df.info()
from sklearn import preprocessing as prep
le = prep.LabelEncoder()

concated_df.Sex =le.fit_transform(concated_df.Sex)

df.Sex[0:10]
concated_df.head()
concated_df.info()
embarked = concated_df['Embarked'].fillna('0')

embarked.unique()
concated_df.Embarked = le.fit_transform(embarked)

concated_df.Embarked.unique()
concated_df.head()
concated_df.tail()
concated_df.dtypes
print( 'Pclass:' ,concated_df.Pclass.unique())
print( 'Sex:' ,concated_df.Sex.unique())
print( 'SibSp:' ,concated_df.SibSp.unique())
print( 'Parch:' ,concated_df.Parch.unique())
print( 'Embarked:' ,concated_df.Embarked.unique())
concated_df.drop(['Cabin'],axis=1,inplace=True)

concated_df.head()
NameSplit = concated_df.Name.str.split('[,.]')

NameSplit.head()
titles = [str.strip(name[1]) for name in NameSplit.values]
titles[:10]
# new feature

concated_df['Title'] = titles

concated_df.head()
concated_df.Title.unique()
# useless words: I will combine Mademoiselle and Madame into a single type

concated_df.Title.values[concated_df.Title.isin(['Mme', 'Mmle'])] = 'Mmle'
# keep reducing

concated_df.Title.values[concated_df.Title.isin(['Capt', 'Don', 'Major', 'Sir'])] = 'Sir'
concated_df.Title.values[concated_df.Title.isin(['Dona', 'Lady', 'the Countess', 'Jonkheer'])] = 'Lady'
concated_df.Title.unique()
# label encode new feature too

concated_df.Title = le.fit_transform(concated_df.Title)
concated_df.head()
# new feature is family size
# number of spouses and siblings and oneself is family size

concated_df['FamilySize'] = concated_df.SibSp.values + concated_df.Parch.values + 1
concated_df.head()
surnames = [str.strip(name[0]) for name in NameSplit]
surnames[:10]
concated_df['Surname'] = surnames
concated_df['FamilyID'] = concated_df.Surname.str.cat(concated_df.FamilySize.astype(str),sep='')
concated_df.head()
# I will mark if any family id as small if family size is less than or equal to 2

concated_df.FamilyID.values[concated_df.FamilySize.values <= 2] = 'Small'

concated_df.head()
# check up the frequency of family ids
concated_df.FamilyID.value_counts()
freq = list(dict(zip(concated_df.FamilyID.value_counts().index.tolist(), concated_df.FamilyID.value_counts().values)).items())

type(freq)
freq = np.array(freq)

freq[:10]
freq.shape
# select the family ids with frequency of 2 or less
freq[freq[:,1].astype(int) <= 2].shape
freq = freq[freq[:,1].astype(int) <= 2]
# I'll assign 'Small' for those
concated_df.FamilyID.values[concated_df.FamilyID.isin(freq[:,0])] = 'Small'
concated_df.FamilyID.value_counts()
# label encoding for family id

concated_df.FamilyID = le.fit_transform(concated_df.FamilyID)
concated_df.FamilyID.unique()
# I will choose usefull features
concated_reduce = concated_df[[
    'Pclass', 'Sex', 'Age', 'SibSp',
    'Parch', 'Fare', 'Title', 'Embarked', 'FamilySize',
    'FamilyID']]

concated_reduce.head()
concated_reduce.Age.unique()
concated_reduce.info()
concated_reduce['Age'].fillna(concated_reduce['Age'].median(), inplace=True)
concated_reduce['Fare'].fillna(concated_reduce['Fare'].median(), inplace=True)
concated_reduce.info()
train_final = concated_reduce.iloc[:891].copy()
test_final = concated_reduce.iloc[891:].copy()
train_final.head()
test_final.head()
X = train_final.values

X
y = surv_col.values

y
X.shape
y.shape
test_data = test_final.values

test_data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential()

model.add(Dense(32, init = 'uniform', activation='relu', input_dim = 10))
model.add(Dense(64, init = 'uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, init = 'uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12, init = 'uniform', activation='relu'))
model.add(Dense(1, init = 'uniform', activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X,y, epochs=500, batch_size = 64, verbose = 1)
pred = model.predict(test_data)
pred
# convert to integer
outputBin = np.zeros(0)
for i in pred:
    
    if i <= .5:
        
        outputBin = np.append(outputBin, 0)
    else:
        
        outputBin = np.append(outputBin, 1)
output = np.array(outputBin).astype(int)
output
d = {'PassengerId':pessengerId, 'Survived':output}
final_df = pd.DataFrame(data=d)
final_df.head()
final = final_df.to_csv('new_result.csv',index=False) #convert to csv file

final
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=350, max_depth=15, random_state=42)

print("train accuracy: {} ".format(rf.fit(X, y).score(X, y)))

rf_pred = rf.predict(test_data)
rf_pred
r = {'PassengerId':pessengerId, 'Survived':rf_pred}
final_rf = pd.DataFrame(data=r)
final_rf.head(13)
final_rf = final_df.to_csv('random_forest_result.csv',index=False) #convert to csv file

final_rf

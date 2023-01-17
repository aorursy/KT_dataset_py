import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



drop_column = ['PassengerId','Survived', 'Ticket', 'Pclass', 'Name', 'SibSp', 'Parch', ]#drop a few so it is easier to visualize

train.drop(drop_column, axis=1, inplace = True)
train.info()
train.head()
print(train.info()) # before

# In python listwise

train.dropna(inplace=True)

print(train.info()) # after



#pairwise info is a bit scarce to implement in python, hence I will keep it at here. Since most of the time, listwise is used.

#if you found any resources to link to please let me know :) 

#either way, deletion is best not done.



#In python drop

#del mydata.column_name (alternative)

print(train.head()) #before

train.drop('Embarked', axis=1, inplace=True)

print(train.head()) #after
#need to reload the dataset so it is not deleted

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

drop_column = ['PassengerId','Survived', 'Ticket', 'Pclass', 'Name', 'SibSp', 'Parch', ]#drop a few so it is easier to visualize

train.drop(drop_column, axis=1, inplace = True)



print(train.tail()) #before

train['Age'].fillna(train['Age'].mean(), inplace = True) #fill with mean

#train['Age'].fillna(train['Age'].median(), inplace = True) #fill with median

#had to split due to 2 modes, but normally u wont need to as it will only have one

train['Cabin'].fillna(train['Cabin'].mode()[0].split(" ")[0], inplace = True) 

#train['Cabin'].fillna(train['Cabin'].mode()[0], inplace = True) #without split

print(train.tail()) #after



## alternative: Fill missing values in Age feature with each sex’s mean value of ## Age 

##train['Age'].fillna(train.groupby('Sex')['Age'].transform("mean"), inplace=True)

#need to reload the dataset so it is not deleted

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

drop_column = ['PassengerId','Survived', 'Ticket', 'Pclass', 'Name', 'SibSp', 'Parch', ]#drop a few so it is easier to visualize

train.drop(drop_column, axis=1, inplace = True)



print(train.tail()) #before

train['age_was_missing'] = train['Age'].isnull()

train['Age'].fillna(train['Age'].mean(), inplace = True) #fill with mean, can fill with 0's or 1's too.

print(train.tail()) #after
#need to reload the dataset so it is not deleted for demonstration

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

drop_column = ['PassengerId','Survived', 'Ticket', 'Pclass', 'Name', 'SibSp', 'Parch', ]#drop a few so it is easier to visualize

train.drop(drop_column, axis=1, inplace = True)



#convert Sex to numerical for regression

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



#linear regression

print(train.tail()) #before

train['Age']= train.apply(

    lambda row: 

            0.25743277 * row.Fare + 0.50958711*row.Sex

            if np.isnan(row.Age) else row.Age, axis=1)

print(train.tail()) #after

#prep data before doing imputation



from sklearn.compose import ColumnTransformer



#need to reload the dataset so it is not deleted for demonstration

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

drop_column = ['PassengerId','Survived', 'Ticket', 'Pclass', 'Name', 'SibSp', 'Parch', 'Cabin']#drop a few so it is easier to visualize

train.drop(drop_column, axis=1, inplace = True)



print(pd.DataFrame(train)) #before)

#convert Sex to numerical 

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#need to fill embarked before one hotting it, so here i will use just the mode to fill

#alternative: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python to one hot

#but dont forget to to the dummy trap prevention line



train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True) #without split

#convert to one hot #prevent dummy variable trap by passing drop ="first"

ct = ColumnTransformer(

   [('oh_enc', OneHotEncoder(sparse=False, drop='first'), [3]),],  # the column numbers I want to apply this to

   remainder='passthrough'  # This leaves the rest of my columns in place

)

train = ct.fit_transform(train)





print(pd.DataFrame(train)) # after



train_mice = train

train_knn = train
#implementation

from fancyimpute import IterativeImputer

MICE_imputer = IterativeImputer()

# train = train.copy()

# train.iloc[:, :] = MICE_imputer.fit_transform(train)

train_mice=MICE_imputer.fit_transform(train_mice)

print(pd.DataFrame(train_mice)) #after



#double check whethere there are still nulls

pd.DataFrame(train).isna().sum()
#implementation

from fancyimpute import KNN    



# Use 3 nearest rows which have a feature to fill in each row's missing features

train_knn = KNN(k=3).fit_transform(train_knn)

print(pd.DataFrame(train_knn)) #after
from sklearn.ensemble import RandomForestRegressor



#prep data before doing imputation

from sklearn.compose import ColumnTransformer

#need to reload the dataset so it is not deleted for demonstration

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

drop_column = ['PassengerId','Survived', 'Ticket', 'Pclass', 'Name', 'SibSp', 'Parch', 'Cabin']#drop a few so it is easier to visualize

train.drop(drop_column, axis=1, inplace = True)



print(pd.DataFrame(train)) #before)

#convert Sex to numerical 

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



#alternative: https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python to one hot

#but dont forget to to the dummy trap prevention line



#fill Nan in embarked with the mode

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)



#split the data based on missing value in the Age column

trainWithAge = train[pd.isnull(train['Age']) == False]

trainWithoutAge = train[pd.isnull(train['Age'])]



#another method to one hot a column, and drop first column to prevent dummy variable trap

trainWithoutAge_one_hot_encoded_embarked = pd.get_dummies(trainWithoutAge['Embarked'], drop_first=True)

trainWithAge_one_hot_encoded_embarked = pd.get_dummies(trainWithAge['Embarked'], drop_first= True)

trainWithAge = pd.concat([trainWithAge, trainWithAge_one_hot_encoded_embarked], axis = 1)

trainWithoutAge = pd.concat([trainWithoutAge, trainWithoutAge_one_hot_encoded_embarked], axis = 1)



#remove the original column and also preventing the dummy variable trap

trainWithAge.drop(['Embarked'], axis=1, inplace = True)

trainWithoutAge.drop(['Embarked'], axis=1, inplace = True)



print(trainWithAge)

print(trainWithoutAge)
#implement random forest

independentVariables = ['Sex', 'Fare', 'Q', 'S']

print(trainWithoutAge) #before



rf = RandomForestRegressor()

#train the model with the available values

rf.fit(trainWithAge[independentVariables], trainWithAge['Age'])



#predict

generatedAgeValues = rf.predict(X = trainWithoutAge[independentVariables])



#store the results into the column, the astype int converts it to interger, u can keep it as float if you want to

trainWithoutAge['Age'] = generatedAgeValues.astype(int)



print(trainWithoutAge) #after



#combine them back, reset the index, and drop the extra index column

train = trainWithAge.append(trainWithoutAge)

train.reset_index(inplace=True)

train.drop('index',inplace=True,axis=1)

print(train)
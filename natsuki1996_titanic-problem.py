import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
#load dataset

train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')

full_data = [train, test]
train.head()
#Check null

print(train.isnull().sum())

print('_'*40)

print(test.isnull().sum())
#Data Preprocessing

PassengerId = test['PassengerId']



for dataset in full_data:

    #Cabin 

    dataset['Has_Cabin'] = dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    

    #Family Size

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    #Alone? Family Size is 1 → Alone

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    

    #Sex

    dataset['Sex'] = dataset['Sex'].replace(['male', 'female'], [0, 1])

    

    #Age

    master_ave = dataset.loc[dataset.Name.str.contains('Master'), 'Age'].mean()

    mr_ave = dataset.loc[dataset.Name.str.contains('Mr'), 'Age'].mean() 

    miss_ave = dataset.loc[dataset.Name.str.contains('Miss'), 'Age'].mean() 

    mrs_ave = dataset.loc[dataset.Name.str.contains('Mrs'), 'Age'].mean() 

    dataset.loc[dataset.Name.str.contains('Mraster') & dataset.Age.isnull(), 'Age'] = master_ave

    dataset.loc[dataset.Name.str.contains('Mr') & dataset.Age.isnull(), 'Age'] = mr_ave

    dataset.loc[dataset.Name.str.contains('Miss') & dataset.Age.isnull(), 'Age'] = miss_ave

    dataset.loc[dataset.Name.str.contains('Mrs') & dataset.Age.isnull(), 'Age'] = mrs_ave

    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

    

    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4

    dataset['Age'] = dataset['Age'].astype(int)

    

    #Fare

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    #Embarked

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    #Drop

    dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train.head()
#No nulls

print(train.isnull().sum())

print('_'*40)

print(test.isnull().sum())
trn_x = train.drop(['Survived'], axis=1, inplace=False)

trn_y = train['Survived']
#Train & Predict

#Hyperparams decided by Grid Search

pred = np.zeros((test.shape[0], 10))



for i in range(10):

    model = RandomForestClassifier(

        max_depth = 10,

        max_features =10,

        min_samples_split = 15,

        n_estimators = 10,

        n_jobs = -1,

        random_state = i)



    model.fit(trn_x, trn_y)



    pred[:,i] = model.predict_proba(test)[:,1].reshape(test.shape[0],)



output = pred.mean(axis=1)

output[output >= 0.5] = 1

output[output <  0.5] = 0

output = output.astype(int)
# Submit

df_out = pd.DataFrame({ 'PassengerId': PassengerId,

                        'Survived':    output })

df_out.to_csv("submission.csv", index=False)
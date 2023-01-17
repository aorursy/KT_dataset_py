import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix,f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
train_dataset = pd.read_csv("../input/titanic/train.csv")

test_dataset = pd.read_csv('../input/titanic/test.csv')



#get idea how our data looks

train_dataset.head(4)
#extracting the survived column data

train_dataset_survival = train_dataset['Survived']
# find empty entries in the dataset so that it can be handles while defining the model

pd.DataFrame(train_dataset.isna().sum() + test_dataset.isna().sum())
def feature_scale(dataset):

    data = pd.DataFrame()



    data['Name'] = dataset['Name'].str.extract('([A-Za-z]+\.)',expand=False)

    data['Family_Size'] = dataset['Parch'] + dataset['SibSp']

    

    # fill the empty entries of the feature Age

    data['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

    # fill the empty entries of the feature Fare

    data['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())

    # fill the empty entries of the feature Embarked

    data['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])



    # divide the Age into 6 categories:

    # Category 1 : Child B/w (0-10)

    # Category 2 : Teenagers B/w (10-20)

    # Category 3 : Young B/w (20-32)

    # Category 4 : Adult B/w (32-48)

    # Category 5 : Aged B/w (48-64)

    # Category 6 : Senior B/w (64-80)

    data['Age'] = pd.cut(data['Age'],[0,10,20,32,48,64,80],precision=0)

    

    

    # divide the fare into 4 categories based on pd.qcut(data['Fare'])

    # Category 1 : B/w (0-8)

    # Category 2 : B/w (8-10)

    # Category 3 : B/w (10-22)

    # Category 4 : B/w (22-40)

    # Category 5 : B/w (40-513)

    data['Fare'] = pd.cut(data['Fare'],[0,8,10,22,40,513],include_lowest=1,precision=0)

   

    # Mlle, Ms. and Miss are same 

    # Mrs. and  Mme are same

    data['Name'] = data['Name'].replace({'Mlle.':"Miss.",      

                                         'Ms.':'Miss.',

                                         'Mme.': 'Mrs.'})

    #Group all the non-common title and assign them as Misc.

    data['Name'] = data['Name'].replace([x for x,y in data['Name'].value_counts().items() if y<=10],'Misc.')

    

    #One Hot Encoding

    data = data.join(pd.get_dummies(dataset['Sex']))

    data = data.join(pd.get_dummies(dataset['Pclass'],prefix='Pclass'))

    data = data.join(pd.get_dummies(dataset['Embarked'],prefix='Embarked'))    

    data = data.join(pd.get_dummies(data['Age'],prefix = 'Age'))

    data = data.join(pd.get_dummies(data['Fare'],prefix = 'Fare'))

    data = data.join(pd.get_dummies(data['Name'],prefix = 'Name'))

    

    #Label Encoding

    label_encoder = LabelEncoder()

    data['Name'] = label_encoder.fit_transform(data['Name'])

    data['Fare'] = label_encoder.fit_transform(data['Fare'].map(str))

    data['Age'] = label_encoder.fit_transform(data['Age'])

    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

    data['Pclass'] = dataset['Pclass']

    

    #Though Cabin contains most of the entries thus can be used in this way as

    # data will be of those passengers who could afford the Cabib

    data['Cabin'] = dataset['Cabin'].fillna(0).map(lambda x: 1 if x!=0 else 0)

    

    return data
train_data = feature_scale(train_dataset)
model = RandomForestClassifier()

model.fit(train_data,train_dataset_survival)

score = model.score(train_data,train_dataset_survival)

y_pre = model.predict(train_data)

c = confusion_matrix(train_dataset_survival,y_pre)

print(f'Score : {score}')

print(f'F1 Score : {f1_score(train_dataset_survival,y_pre)}')

print(f'Confusion Matrix \n {c}')    
test_data = feature_scale(test_dataset)
test_pred = model.predict(test_data)
result = pd.DataFrame({ 'PassengerId' : test_dataset['PassengerId'], 'Survived': test_pred })

result.to_csv('submission.csv',index=False)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train = pd.read_csv('../input/train.csv')



test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.info()
test.info()
def clean_data(data):

    data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())

    data['Age'] =  data['Age'].fillna(data['Age'].dropna().median())

    

    data.loc[data['Sex'] == 'male', 'Sex'] = 0

    data.loc[data['Sex'] =='female',  'Sex'] = 1

    

    data['Embarked'] = data['Embarked'].fillna('S')

    data.loc[data["Embarked"] == 'S', 'Embarked'] = 0

    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1

    data.loc[data['Embarked'] == 'Q', 'Embarked'] =2
def write_prediction(prediction, name):

    PassengerId = np.array(test['PassengerId']).astype(int)

    solution = pd.DataFrame(prediction, PassengerId, columns = ['Survived'])

    solution.to_csv(name, index_label = ['PassengerId'])
clean_data(train)

clean_data(test)
train.head()
print('check the nan value in train data')

print(train.isnull().sum())
print('check the nan value in test data')

print(test.isnull().sum())
drop_column = ['Cabin']

train.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column,axis=1,inplace=True)
train.head()
test.head()
g = sns.pairplot(data=train, hue='Survived', palette = 'seismic',

                 size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
## combine test and train as single to apply some function and applying the feature scaling

all_data=[train,test]
# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in all_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Define function to extract titles from passenger names

import re



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in all_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in all_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 

                                                 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
## create Range for age features

for dataset in all_data:

    dataset['Age_Range'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
## create RAnge for fare features

for dataset in all_data:

    dataset['Fare_Range'] = pd.cut(dataset['Fare'], bins=[0,7.91,14.45,31,120], labels=['Low_fare','median_fare',

                                                                                      'Average_fare','high_fare'])
#Avoiding dataloss making a copy of both DataSet start working for copy of dataset

traindf=train

testdf=test
all_dat=[traindf,testdf]
for dataset in all_dat:

    drop_column = ['Age','Fare','Name','Ticket']

    dataset.drop(drop_column, axis=1, inplace = True)
#Removing the passenger id from trainning set 

drop_column = ['PassengerId']

traindf.drop(drop_column, axis=1, inplace = True)
all_dat
testdf.head(5)

#Adding the extra feataure in Train data set

traindf = pd.get_dummies(traindf, columns = ["Sex","Title","Age_Range","Embarked","Fare_Range"],

                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])

traindf.head()
#Adding the extra feature in test data set

testdf = pd.get_dummies(testdf, columns = ["Sex","Title","Age_Range","Embarked","Fare_Range"],

                             prefix=["Sex","Title","Age_type","Em_type","Fare_type"])
testdf.head()
#For precaution let final check the training set...

print(traindf.isnull().sum())
sns.heatmap(traindf.corr(),annot=True,linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(20,12)

plt.show()
target = traindf['Survived'].values

features = traindf[['Pclass','SibSp','Parch','FamilySize','Sex_0','Sex_1','Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Rare','Age_type_Children','Age_type_Teenage','Age_type_Adult','Age_type_Elder','Em_type_0','Em_type_1','Em_type_2','Fare_type_Low_fare','Fare_type_median_fare','Fare_type_Average_fare','Fare_type_high_fare']].values
import keras
from keras.models import Sequential

from keras.layers import Dense,Dropout



classifier = Sequential()

classifier.add(Dense(activation="relu", input_dim=22, units=11, kernel_initializer="uniform"))

classifier.add(Dense(activation="relu", units=11, kernel_initializer="uniform"))

classifier.add(Dropout(0.5))

classifier.add(Dense(activation="relu", units=11, kernel_initializer="uniform"))

classifier.add(Dropout(0.5))

classifier.add(Dense(activation="relu", units=5, kernel_initializer="uniform"))

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()
history=classifier.fit(features, target, batch_size = 10, nb_epoch = 100,

    validation_split=0.1,verbose = 1,shuffle=True)
drop_column = ['PassengerId']

testdf.drop(drop_column, axis=1, inplace = True)

testdf.head()
#predicting the results

Y_pred = classifier.predict(testdf)
Y_pred.dtype
#Round off the result for submission

Y_pred=Y_pred.round()

Y_pred
#Call above write function to write the out for submission

write_prediction(Y_pred, "My_output.csv")
# predictions = classifier.predict(testdf)

# predictions = pd.DataFrame(predictions, columns=['Survived'])

# test = pd.read_csv(os.path.join('../input', 'test.csv'))

# predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)

# predictions.to_csv('my_output.csv', sep=",", index = False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

%matplotlib inline  
train_x = pd.read_csv("../input/train.csv")

train_y = train_x.pop("Survived")

test_x = pd.read_csv("../input/test.csv")

train_x['Source'] = 'train'

test_x['Source'] = 'test'
passengerids = test_x['PassengerId']



combined_x = train_x.append(test_x)
combined_x['Party'] = combined_x['Parch'] + combined_x['SibSp']

#Replace NAs in Cabin to U and strip Cabin to first letter

combined_x.Cabin.fillna('U',inplace=True)

combined_x['Cabin'] = combined_x['Cabin'].str[:1]

#Get title from Name

name_df = combined_x['Name'].apply(lambda x: pd.Series(x.split(',')))

name_df = name_df[1].apply(lambda x: pd.Series(x.split('.')))

name_df.head()

combined_x['Title'] = name_df[0]
combined_x['Age'] = combined_x['Age'].fillna(-1)

combined_x['Cabin'] = combined_x['Cabin'].fillna('N')

combined_x['Fare'] = combined_x['Fare'].fillna(-1)
combined_x['Age'].fillna(-0.5)

bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

categories = pd.cut(combined_x.Age, bins, labels=group_names)

combined_x['Age'] = categories
#Delete unused columns

del combined_x['PassengerId']

del combined_x['Ticket']

del combined_x['Name']

del combined_x['Sex']

del combined_x['SibSp']

del combined_x['Parch']
combined_x = pd.get_dummies(combined_x)

combined_x.head()
#Split back data

train_x = combined_x[combined_x['Source_train']==1]

del train_x['Source_train']

del train_x['Source_test']

test_x = combined_x[combined_x['Source_test']==1]

del test_x['Source_train']

del test_x['Source_test']
#Split train set 

from sklearn.cross_validation import train_test_split

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.33, random_state=42)
print(len(train_x))

print(len(train_y))

print(len(val_x))

print(len(val_y))
#train_y = train_y[np.isfinite(train_x['Age'])]

#train_x = train_x[np.isfinite(train_x['Age'])]

#val_y = val_y[np.isfinite(val_x['Age'])]

#val_x = val_x[np.isfinite(val_x['Age'])]

#train_y = train_y[np.isfinite(train_x['Fare'])]

#train_x = train_x[np.isfinite(train_x['Fare'])]

#val_y = val_y[np.isfinite(val_x['Fare'])]

#val_x = val_x[np.isfinite(val_x['Fare'])]
train_x.isnull().any().any()
val_x.isnull().any().any()
test_x.isnull().any().any()
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



accuracy_score = []

depth_array = []

for depth in range(1,50):

    depth_array.append(depth)

    rf = RandomForestClassifier(n_estimators=50, criterion='gini', n_jobs=4, max_depth=depth)

    rf.fit(train_x, train_y)

    accuracy =(metrics.accuracy_score(val_y, rf.predict(val_x)))

    accuracy_score.append(accuracy)

    #print("Depth: " + str(depth) + " Accuracy: " + str(accuracy))

    

plt.scatter(depth_array, accuracy_score)

plt.title('Changing depth')

plt.show()
accuracy_score = []

estimator_array = [] 

for i in range(1,20):

    estimators = i*50

    estimator_array.append(estimators)

    rf = RandomForestClassifier(n_estimators=estimators, criterion='gini', n_jobs=4, max_depth=10)

    rf.fit(train_x, train_y)

    accuracy =(metrics.accuracy_score(val_y, rf.predict(val_x)))

    accuracy_score.append(accuracy)

    #print("Estimators: " + str(estimators) + " Accuracy: " + str(accuracy))

    

plt.scatter(estimator_array, accuracy_score)

plt.title('Changing estimator number')

plt.show()
accuracy_score = []

feature_array =[]

for i in range(1,20):

    features = i

    feature_array.append(features)

    rf = RandomForestClassifier(max_features=features, n_estimators=200, criterion='entropy', n_jobs=4, max_depth=10)

    rf.fit(train_x, train_y)

    accuracy =(metrics.accuracy_score(val_y, rf.predict(val_x)))

    accuracy_score.append(accuracy)

    #print("Features: " + str(features) + " Accuracy: " + str(accuracy))

    

plt.scatter(feature_array, accuracy_score)

plt.title('Changing no. of features')

plt.show()
accuracy_score = []

split_array =[]

for i in range(1,50):

    split = i*5

    split_array.append(split)

    rf = RandomForestClassifier(max_features=10, n_estimators=200,max_depth=10, criterion='entropy', n_jobs=4, min_samples_split=split)

    rf.fit(train_x, train_y)

    accuracy =(metrics.accuracy_score(val_y, rf.predict(val_x)))

    accuracy_score.append(accuracy)

    #print("Features: " + str(features) + " Accuracy: " + str(accuracy))

    

plt.scatter(split_array, accuracy_score)

plt.title('Changing Min Split Samples')

plt.show()
max_features = 10

max_depth = 10

n_estimators = 500

min_samples_split = 50



rf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators, criterion='gini', n_jobs=4, min_samples_split=min_samples_split,max_depth=max_depth)

rf.fit(train_x, train_y)

accuracy =(metrics.accuracy_score(val_y, rf.predict(val_x)))

print(accuracy)
sub = rf.predict(test_x)
output = pd.DataFrame()
output = pd.DataFrame({ 'PassengerId' : passengerids, 'Survived': sub })
output = output.set_index('PassengerId')
output.to_csv('sub.csv')
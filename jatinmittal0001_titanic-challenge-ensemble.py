# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

test_data_copy = pd.read_csv('../input/test.csv')

ntrain = train_data.shape[0]

y = train_data.iloc[:,1]

train_data = train_data.drop(['PassengerId','Survived'],axis=1)

test_data = test_data.drop(['PassengerId'],axis=1)

total_data = train_data.append(test_data, sort=False)



train_data.head()
# missing value treatment

total_missing_values = total_data.isnull().sum().sort_values(ascending=False)

percentage_missing_values = (100*total_data.isnull().sum()/total_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total_missing_values,percentage_missing_values], axis=1 , keys=['#missing values', "missing percentage"])

print(missing_data)
# cabin column treatment

'''

we can see that 77% values in that col. are missing, we can do 2 things

1. Drop that variable

2. Since it is categorical variable we can replace those 77% values with 'U' which will represent

unknown class

We'll try both methods and finally keep that whih gives best accuracy

'''

total_data["Cabin"].unique()

total_data["Cabin"] = total_data["Cabin"].fillna('U')

total_data["Cabin"] = total_data["Cabin"].apply(lambda x: x[0])
'''

Now we have to treat age variable, for that we will first extarct Mr., mrs. etc titles and replace 

msising values with mean of that particuar title group

'''

total_data["Name"] = total_data["Name"].map(lambda x: x.split(",")[1].split(".")[0].strip())



total_data.head()
pd.crosstab(total_data["Name"], total_data["Sex"])
#We can replace many titles with a more common name or classify them as Rare.

total_data['Name'] = total_data['Name'].replace(['Lady', 'the Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

total_data['Name'] = total_data['Name'].replace('Mlle', 'Miss')

total_data['Name'] = total_data['Name'].replace('Ms', 'Miss')

total_data['Name'] = total_data['Name'].replace('Mme', 'Mrs')

    
unique_titles = total_data["Name"].unique()

print(unique_titles)
# now treating 'age' variable

total_data["Age"] = total_data.groupby("Name")["Age"].transform(lambda x: x.fillna(x.mean()))
# now checking #missing values

total_missing_values = total_data.isnull().sum().sort_values(ascending=False)

percentage_missing_values = (100*total_data.isnull().sum()/total_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total_missing_values,percentage_missing_values], axis=1 , keys=['#missing values', "missing percentage"])

print(missing_data)
# filling embark missing values with values which occured most

total_data['Embarked'] = total_data['Embarked'].fillna(total_data['Embarked'].mode()[0])

#now for fare it could be decided by the station embarked from and thecabin alloted

#

total_data["Fare"] = total_data.groupby(['Embarked','Pclass'])["Fare"].transform(lambda x: x.fillna(x.mean()))
# also ticket number won't provide aby valueable information so we can drop that

total_data = total_data.drop(['Ticket'],axis=1)



#creating feature

total_data["Family_size"] = total_data['SibSp'] + total_data['Parch'] + 1 

total_data['IsAlone'] = 1

total_data['IsAlone'].loc[total_data['Family_size']>1] =0



total_data['AgeBin'] = pd.cut(total_data['Age'].astype(int), 5)

total_data['FareBin'] = pd.cut(total_data['Fare'].astype(int), 4)

#label encoder

label = LabelEncoder()

total_data['AgeBin_Code'] = label.fit_transform(total_data['AgeBin'])

total_data['FareBin_Code'] = label.fit_transform(total_data['FareBin'])
total_data.head()


# ONE HOT ENCODING OF CATEGORICAL DATA

total_data = total_data.drop(['Cabin'], axis=1)

total_data_onehot = total_data.copy()

total_data_onehot = pd.get_dummies(total_data_onehot, columns=['Sex','Embarked','Name'], prefix = ['Sex','Embarked','Name'])
total_data_onehot.head()

#total_data_onehot = total_data_onehot.drop(['Age','Fare'],axis=1)
total_data_onehot = total_data_onehot.drop(['AgeBin','FareBin'],axis=1)
total_data_onehot.head()


final_train_data = total_data_onehot.iloc[:ntrain,:]

final_test_data = total_data_onehot.iloc[ntrain:,:]



#creating train, test split

x_train, x_test,y_train, y_test = train_test_split(final_train_data, y, shuffle=True,test_size=0.3)



# scaling data

from sklearn.preprocessing import StandardScaler



# fit only to training data i.e. find mean and dev for training data

scale = StandardScaler()

scale.fit(x_train)



# apply those transformtions to x_train and x_test data set

x_train = scale.transform(x_train)

x_test = scale.transform(x_test)
from sklearn.metrics import accuracy_score

def find_accuracy(y_test,y_pred):

    return accuracy_score(y_test,y_pred)
numebr_of_class0_counts = y[y==0].count()

# it is almost 60%, therefore right now we are not upsampling or downsampling it
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=70, learning_rate=1)

ada.fit(x_train, y_train)

y_pred = ada.predict(x_test)

print(find_accuracy(y_pred, y_test))
from sklearn.svm import SVC

svc = SVC(kernel = 'poly',probability=True)

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

print(find_accuracy(y_pred, y_test))
# applying model: XGBOOST

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier



xgb_model = XGBClassifier()

xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)



print(find_accuracy(y_test,y_pred))
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10,6,5,3),activation='relu',alpha = 0.0001,max_iter = 1000,solver='lbfgs')

mlp.fit(x_train, y_train)

y_pred = mlp.predict(x_test)

print(find_accuracy(y_test, y_pred))
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.0005)

lasso.fit(x_train, y_train)

y_pred = lasso.predict(x_test)

y_pred = (y_pred>0.6)

print(find_accuracy(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

print(find_accuracy(y_test, y_pred))


from sklearn.linear_model import ElasticNet

elastic_net_model = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

elastic_net_model.fit(x_train, y_train)

y_pred = elastic_net_model.predict(x_test)

y_pred = (y_pred>0.6)

print(find_accuracy(y_test, y_pred))
from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators=[('xgb', xgb_model),('svc',svc)], voting='soft')

model = model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(find_accuracy(y_pred, y_test))
#NOTE: we have trained our model on x_train, and tested on x_test

# since the size of data is small, we'll now train on final_train_data, and directly predict test_data

#and we'll not test it. This is bcoz we know that accuracy will be around 84%, so increasing size of 

# trainging data may give more accuracy

'''

scale.fit(final_train_data)



# apply those transformtions to x_train and x_test data set

train = scale.transform(final_train_data)

test = scale.transform(final_test_data)



model = model.fit(train, y)



#note that we first have to transform test_data

y_pred = model.predict(test)

'''



test = scale.transform(final_test_data)

model = model.fit(x_train, y_train)

y_pred = model.predict(test)
solution = pd.DataFrame({"PassengerId": test_data_copy["PassengerId"],

        "Survived": y_pred})

solution.to_csv("titanic_final.csv", index = False)
'''

from sklearn.model_selection import KFold

kf = KFold(n_splits=5)

for train_index, test_index in kf.split(final_train_data):

    #print("TRAIN:", train_index, "TEST:", test_index)

    x_train, x_test = final_train_data.iloc[train_index], final_train_data.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    scale = StandardScaler()

    scale.fit(x_train)



    # apply those transformtions to x_train and x_test data set

    x_train = scale.transform(x_train)

    x_test = scale.transform(x_test)

    model = VotingClassifier(estimators=[('xgb', xgb_model),('svc',svc)], voting='soft')

    model = model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    print(find_accuracy(y_pred, y_test))

'''
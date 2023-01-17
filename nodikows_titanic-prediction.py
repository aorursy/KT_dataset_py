# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder, Imputer

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

labels = train.iloc[:,1].values

test_id = test.iloc[:,0].values

train.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1,inplace=True)

test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

embarked = pd.get_dummies(train.Embarked)

test_embarked = pd.get_dummies(test.Embarked)

train.drop(['Embarked'],axis=1,inplace=True)

test.drop(['Embarked'],axis=1,inplace=True)

train = train.join(pd.DataFrame(embarked, index=train.index))

test = test.join(pd.DataFrame(test_embarked, index=test.index))



mapping = {'female':0,'male':1}



train = train.replace({'Sex':mapping})

test = test.replace({'Sex':mapping})



train.dropna(how='all')
train.isna().any()


train.isna().any()

age_avgs = test.groupby(['Sex','Pclass','SibSp','Parch']).Age.mean().reset_index()



for index, age in enumerate(test.Age):

    if np.isnan(age):

        test.Age[index]  = age_avgs[(age_avgs.Sex == test.Sex[index]) & (age_avgs.Pclass == test.Pclass[index]) & (age_avgs.SibSp == test.SibSp[index]) & (age_avgs.Parch == test.Parch[index])].Age



train = train.values

test = test.values
labelEncoder = LabelEncoder()

train[:,1] = labelEncoder.fit_transform(train[:,1])

test[:,1] = labelEncoder.fit_transform(test[:,1])

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(train[:,2:3])

train[:,2:3] = imputer.transform(train[:,2:3])

imputer_test = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer_test = imputer_test.fit(test[:,[2,5]])

test[:,[2,5]] = imputer_test.transform(test[:,[2,5]])

from sklearn.model_selection import train_test_split

x, x_test, y, y_test = train_test_split(train, labels,test_size=0.20)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)

x_test = sc.transform(x_test)

test = sc.transform(test)



from sklearn.decomposition import PCA

pca = PCA(n_components=9)

x = pca.fit_transform(x)

x_test = pca.transform(x_test)

test = pca.transform(test)

explained_variance = pca.explained_variance_ratio_

explained_variance
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_gaussian_quantiles

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),

                         algorithm="SAMME",

                         n_estimators=200).fit(x,y)



# checking the accuracy on validation set

count = 0

for i in range(len(x_test)):

    pre = bdt.predict([x_test[i]])

    if pre == y_test[i]:

        count += 1

print("Accuracy: " , count/len(x_test)*100)
#writing the prediction on test set on a csv file

file = open('submission.csv','w')

file.write('PassengerId'+','+'Survived'+'\n')

for i in range(len(test)):

    prediction = bdt.predict([test[i]])

    file.write(str(test_id[i]) + ','+str(prediction).replace("[","").replace("]","") + '\n')

file.close()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

#a new type of import

import pandas_profiling as pp



sns.set(rc={'figure.figsize':(11.7,8.27)})
#bring thte data to the dataframe

data = pd.read_csv('/kaggle/input/learn-together/train.csv')



#describe the data

data.describe()
#as id is not a feature or something, we can drop it 

data.drop(['Id'], axis = 1 , inplace = True)



data.describe()
print(data.columns)
#using Pandas_profiling for eda

report = pp.ProfileReport(data)

report.to_file("report.html")



report
data = data.rename(columns={"Horizontal_Distance_To_Roadways": "HDR", "Horizontal_Distance_To_Fire_Points": "HDF"  })

data = data.rename(columns={"Horizontal_Distance_To_Hydrology": "HDH", "Vertical_Distance_To_Hydrology": "VDH" })
data.columns
sns.heatmap(data.corr() , cmap = 'gist_ncar')
#appending all the positive correlated data

lst = []

column = []



for col in data.columns:

    if(data.corr()['Cover_Type'][col] > 0.0):

        column.append(col)

        print(col +" : " +  str(data.corr()['Cover_Type'][col]))

        lst.append(float(data.corr()['Cover_Type'][col]))

#print(lst.sort()[:5])
print(type(column[1]))

print(type(lst[0]))
print(column)

column = column[:-1]
column
selected = data[column]



test_df = pd.read_csv('/kaggle/input/learn-together/test.csv')

test_df = test_df.rename(columns={"Horizontal_Distance_To_Roadways": "HDR", "Horizontal_Distance_To_Fire_Points": "HDF"  })

test_df = test_df.rename(columns={"Horizontal_Distance_To_Hydrology": "HDH", "Vertical_Distance_To_Hydrology": "VDH" })

test_sel = test_df[["Id"] + column]
test_sel.columns
#selected.describe()

from sklearn.model_selection import train_test_split , cross_val_score

from sklearn.metrics import accuracy_score



y = data['Cover_Type'][:]



X_train , X_test , y_train , y_test = train_test_split(selected , y , stratify = y , random_state = 7)
X_test.shape

y_test = y_test.to_numpy()
from sklearn.ensemble import AdaBoostClassifier , BaggingClassifier , RandomForestClassifier , ExtraTreesClassifier , GradientBoostingClassifier

from sklearn.tree import ExtraTreeClassifier
#AdaBoostClassifier



l = [10 , 50 , 25 , 75 , 100 , 150 , 200]



for n in range(5 , 100 ,5):

    clf = AdaBoostClassifier(n_estimators=n , random_state = 0)

    clf.fit(X_train , y_train)

    

    y_pred = clf.predict(X_test)

      

    print("For l=" + str(n) , end = ' ')

    print(accuracy_score(y_pred , y_test))
#ExtraTreesClassifier

for n in range(300 , 500 ,10):

    clf = ExtraTreesClassifier(n_estimators=n , random_state = 0)

    clf.fit(X_train , y_train)

    

    y_pred = clf.predict(X_test)

      

    print("For Max Depth " + str(n) , end = ' ')

    print(accuracy_score(y_pred , y_test))
%%time

#GradientBoostingClassifier

for n in range(300 , 500 ,50):

    clf = GradientBoostingClassifier(learning_rate=0.2 , n_estimators=n , random_state = 0)

    clf.fit(X_train , y_train)

    

    y_pred = clf.predict(X_test)

      

    print("For Max Depth " + str(n) , end = ' ')

    print(accuracy_score(y_pred , y_test))
%%time

#RandomForestClassifier

for n in range(100 , 1500 ,100):

    clf = RandomForestClassifier(n_estimators=n , random_state = 0)

    clf.fit(X_train , y_train)

    

    y_pred = clf.predict(X_test)

      

    print("For Max Depth " + str(n) , end = ' ')

    print(accuracy_score(y_pred , y_test))
#BagggingClasssifier with ExtraTreeClassifier as base estimator



base = ExtraTreeClassifier(max_depth = 500)



for n in range(100 , 1500 ,100):

    clf = BaggingClassifier(base_estimator = base , n_estimators=n , random_state = 0)

    clf.fit(X_train , y_train)

    

    y_pred = clf.predict(X_test)

      

    print("For Max Depth " + str(n) , end = ' ')

    print(accuracy_score(y_pred , y_test))
train_X = selected.to_numpy()

train_y = y.to_numpy()
model = BaggingClassifier(base_estimator = base , n_estimators=200 , random_state = 0)



id = test_sel['Id']



test_sel = test_sel.drop(['Id'] , axis = 1)

print("Training")

model.fit(train_X , train_y)

print("Finished!!!")

print("Predicting")

pred = model.predict(test_sel)

submission = pd.DataFrame({ 'Id': id,

                            'Cover_Type': pred })

submission.to_csv("submission_example.csv", index=False)
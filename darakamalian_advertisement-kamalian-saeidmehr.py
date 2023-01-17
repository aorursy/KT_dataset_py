# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test = pd.read_csv("../input/advertsuccess/Test.csv")

train = pd.read_csv("../input/advertsuccess/Train.csv")
# check for null values in dataset

train.isnull().sum()

# no null values so no column needs to be ommitted
train.head()
le1 = preprocessing.LabelEncoder()

le1.fit(train['realtionship_status'])

list(le1.classes_)

train['realtionship_status'] = le1.transform(train['realtionship_status'])

train.head()
le2 = preprocessing.LabelEncoder()

le2.fit(train['industry'])

list(le2.classes_)

train['industry'] = le2.transform(train['industry']) 

train.head()
le3 = preprocessing.LabelEncoder()

le3.fit(train['genre'])

list(le3.classes_)

train['genre'] = le3.transform(train['genre']) 

train.head()
le4 = preprocessing.LabelEncoder()

le4.fit(train['targeted_sex'])

list(le4.classes_)

train['targeted_sex'] = le4.transform(train['targeted_sex']) 

train.head()
le5 = preprocessing.LabelEncoder()

le5.fit(train['airtime'])

list(le5.classes_)

train['airtime'] = le5.transform(train['airtime']) 

train.head()
le6 = preprocessing.LabelEncoder()

le6.fit(train['airlocation'])

list(le6.classes_)

train['airlocation'] = le6.transform(train['airlocation']) 

train.head()
le7 = preprocessing.LabelEncoder()

le7.fit(train['expensive'])

list(le7.classes_)

train['expensive'] = le7.transform(train['expensive'])

train.head()
le8 = preprocessing.LabelEncoder()

le8.fit(train['money_back_guarantee'])

list(le8.classes_)

train['money_back_guarantee'] = le8.transform(train['money_back_guarantee'])

train.head()
train.head()
le9 = preprocessing.LabelEncoder()

le9.fit(train['netgain'])

list(le9.classes_)

train['netgain'] = le9.transform(train['netgain'])

train.head()
#Considering all available features for decision tree classifier

features = ['realtionship_status','industry','genre','targeted_sex','average_runtime(minutes_per_week)','airtime','airlocation','ratings','expensive','money_back_guarantee']

X = train[features]

y = train['netgain']
# Decision tree classifier and model evaluation using kFold cross validation



from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold



kfold_mae_train=0

kfold_mae_test=0

kfold_f_imp_dic = 0



no_of_folds = 5



kf = KFold(no_of_folds,True,1)



for train_index, test_index in kf.split(X):

    

    X_train,X_test = X.iloc[train_index],X.iloc[test_index]

    y_train,y_test = y.iloc[train_index],y.iloc[test_index]

    

    dt_classifier = DecisionTreeClassifier(random_state=1)

    dt_classifier.fit(X_train,y_train)

    

    mae_train = mean_absolute_error(dt_classifier.predict(X_train),y_train)

    kfold_mae_train=(kfold_mae_train+mae_train)

    

    mae_test = mean_absolute_error(dt_classifier.predict(X_test),y_test)

    kfold_dt_mae_test = (kfold_mae_test+mae_test)

    

    kfold_f_imp_dic = kfold_f_imp_dic + dt_classifier.feature_importances_

    

print('Decision Tree Regressor train set mean absolute error =',kfold_mae_train/no_of_folds)

print('Decision Tree Regressor test set mean absolute error  =',kfold_dt_mae_test/no_of_folds)
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



f_importance_dic = dict(zip(features,kfold_f_imp_dic/no_of_folds))

df_imp_features = pd.DataFrame(list(f_importance_dic.items()),columns=['feature','score'])



plt.figure(figsize=(30,10))

plt.bar(df_imp_features['feature'], df_imp_features['score'],color='green',align='center', alpha=0.5)

plt.xlabel('Mobile features', fontsize=20)

plt.ylabel('Relative feature score',fontsize=20)

plt.title('Relative Feature importance in determining price',fontsize=30)
df = pd.DataFrame(test)
le1 = preprocessing.LabelEncoder()

le1.fit(test['realtionship_status'])

list(le1.classes_)

test['realtionship_status'] = le1.transform(test['realtionship_status'])

# df['realtionship_status'] = list(le.inverse_transform())

test.head()
le2 = preprocessing.LabelEncoder()

le2.fit(test['industry'])

list(le2.classes_)

test['industry'] = le2.transform(test['industry'])

test.head()
le3 = preprocessing.LabelEncoder()

le3.fit(test['genre'])

list(le3.classes_)

test['genre'] = le3.transform(test['genre'])

test.head()
le4 = preprocessing.LabelEncoder()

le4.fit(test['targeted_sex'])

list(le4.classes_)

test['targeted_sex'] = le4.transform(test['targeted_sex'])

test.head()
le5.fit(test['airtime'])

list(le5.classes_)

test['airtime'] = le5.transform(test['airtime'])

test.head()
le6.fit(test['airlocation'])

list(le6.classes_)

test['airlocation'] = le6.transform(test['airlocation'])

test.head()
le7.fit(test['expensive'])

list(le7.classes_)

test['expensive'] = le7.transform(test['expensive'])

test.head()
le8.fit(test['money_back_guarantee'])

list(le8.classes_)

test['money_back_guarantee'] = le8.transform(test['money_back_guarantee'])

test.head()
#Prediction on test data

test['netgain'] = dt_classifier.predict(test[features])

test.head(5)
test['netgain'] = le9.inverse_transform(test['netgain'])

test.head(5)
test['money_back_guarantee'] = le8.inverse_transform(test['money_back_guarantee'])

test.head(5)
test['expensive'] = le7.inverse_transform(test['expensive'])

test.head(5)
test['airlocation'] = le6.inverse_transform(test['airlocation'])

test.head(5)
test['airtime'] = le5.inverse_transform(test['airtime'])

test.head(5)
test['targeted_sex'] = le4.inverse_transform(test['targeted_sex'])

test.head(5)
test['genre'] = le3.inverse_transform(test['genre'])

test.head(5)
test['industry'] = le2.inverse_transform(test['industry'])

test.head(5)
test['realtionship_status'] = le1.inverse_transform(test['realtionship_status'])

test.head(5)
test.head(40)
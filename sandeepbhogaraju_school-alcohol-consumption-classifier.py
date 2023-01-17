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
import pandas as pd

test = pd.read_csv("../input/School_test_user.csv")

data = pd.read_csv("../input/School_train_data.csv")

data = data[data.school =='MS']
data.head()
Y = data.Result
#Label Encode the Y

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(Y)

le.classes_

Y = le.transform(Y)

print(Y[:5])
categorical_cols=data.select_dtypes(include=['object']).columns

print(categorical_cols)

int_cols = data.select_dtypes(include=['int64']).columns

print(int_cols)
# def Mapper(df):

    
# data.apply(Mapper)
#Scale the Int6 cols

from sklearn.preprocessing import MinMaxScaler,Normalizer



scaler = MinMaxScaler()

nor = Normalizer(norm='l2')

for col in int_cols:

    if col in ['age','absences','failures']:

        scaler.fit((data[col].to_numpy().reshape(-1, 1)))

        data[col] = scaler.transform((data[col].to_numpy().reshape(-1, 1)))

        test[col] = scaler.transform((test[col].to_numpy().reshape(-1, 1)))

    else:

#         nor.fit((data[col].to_numpy().reshape(-1, 1)))

#         data[col] = nor.transform((data[col].to_numpy().reshape(-1, 1)))

#         test[col] = nor.transform((test[col].to_numpy().reshape(-1, 1)))

        pass

        

data.head()
#Label Encode for Categories

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in categorical_cols[:-1]:

    le.fit(data[col].to_numpy().reshape(-1,1))

    data[col] = le.transform(data[col].to_numpy().reshape(-1,1))

    test[col] = le.transform(test[col].to_numpy().reshape(-1,1))

    

   

data.Result = Y

data.head()
data.corr().Result.sort_values()
from sklearn.model_selection import train_test_split



X = data.drop(['Result','id','school'],axis=1)

Y = Y.reshape(-1,1)

X_predict = test.drop(['id','school'],axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape,Y_train.shape)
# from sklearn.model_selection import train_test_split

# #['goout','Dalc','absences','failures','Medu','Fedu','schoolsup','studytime','internet','famrel','reason']  ==>72/66

# #['goout','Dalc','absences','failures','Medu','Fedu','schoolsup','studytime','internet','famrel','reason','sex'] ==>72/66



# X = data[['goout','Dalc','absences','failures','Medu','Fedu','schoolsup','studytime','internet','famrel','reason','sex']]

# Y = Y.reshape(-1,1)

# X_predict = test[['goout','Dalc','absences','failures','Medu','Fedu','schoolsup','studytime','internet','famrel','reason','sex']]

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# print(X_train.shape,Y_train.shape)
#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



clf = RandomForestClassifier(max_depth=3, random_state=0)

clf.fit(X_train, Y_train)

print(clf.score(X_test,Y_test))

Y_pred = clf.predict(X_test)

print(classification_report(Y_test, Y_pred))
# from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression().fit(X_train, Y_train)

# print(clf.score(X_test,Y_test))

# Y_pred = clf.predict(X_test)

# print(classification_report(Y_test, Y_pred))
# from sklearn import linear_model

# clf = linear_model.SGDClassifier(max_iter=100)

# clf.fit(X_train, Y_train)

# print(clf.score(X_test,Y_test))

# Y_pred = clf.predict(X_test)

# print(classification_report(Y_test, Y_pred))
# from sklearn import svm

# clf = svm.SVC(kernel='linear',gamma='auto')

# clf.fit(X_train, Y_train)

# print(clf.score(X,Y))

# Y_pred = clf.predict(X_test)

# print(classification_report(Y_test, Y_pred))
# from sklearn.model_selection import StratifiedKFold,cross_val_score

# import xgboost as xgb

# from xgboost import XGBClassifier



# clf = XGBClassifier()

# clf=XGBClassifier(learning_rate=0.1,n_estimators=50)

# clf.fit(X_train,Y_train)

# print(clf.score(X_test,Y_test))

# Y_pred = clf.predict(X_test)

# print(classification_report(Y_test, Y_pred))
# from sklearn.svm import LinearSVC



# clf = LinearSVC()

# clf.fit(X_train, Y_train)

# print(clf.score(X_test,Y_test))

# Y_pred = clf.predict(X_test)

# print(classification_report(Y_test, Y_pred))
# from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier(min_samples_leaf=30)

# clf.fit(X_train, Y_train)

# print(clf.score(X_test,Y_test))

# Y_pred = clf.predict(X_test)

# print(classification_report(Y_test, Y_pred))
# from sklearn.neighbors import KNeighborsClassifier



# neighbors = np.arange(1,9)

# train_accuracy =np.empty(len(neighbors))

# test_accuracy = np.empty(len(neighbors))



# for i,k in enumerate(neighbors):

#     #Setup a knn classifier with k neighbors

#     knn = KNeighborsClassifier(n_neighbors=k)

    

#     #Fit the model

#     knn.fit(X_train, Y_train)

    

#     #Compute accuracy on the training set

#     train_accuracy[i] = knn.score(X_train, Y_train)

    

#     #Compute accuracy on the test set

#     test_accuracy[i] = knn.score(X_test, Y_test) 
# print(train_accuracy)

# print(test_accuracy)
# from sklearn.pipeline import Pipeline

# from sklearn.feature_selection import SelectFromModel

# clf = Pipeline([

#   ('feature_selection', SelectFromModel(LinearSVC(penalty="l2"))),

#   ('classification', RandomForestClassifier())

# ])

# clf.fit(X_train, Y_train)

# print(clf.score(X_test,Y_test))

# Y_pred = clf.predict(X_test)

# print(classification_report(Y_test, Y_pred))
Y_predict = clf.predict(X_predict)

Y_final = []

for result in Y_predict:

    if(result ==1):

        Y_final.append('PASS')

    else:

        Y_final.append('FAIL')
Y_final
df = pd.DataFrame()

df['id'] = pd.Series(np.arange(1,len(Y_final)+1))

df['result'] = pd.Series(Y_final)
df.to_csv('y_predict.csv',index=False)
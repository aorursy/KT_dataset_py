# importing libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O 



print("list of files under the input directory:\n")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# reading the train data

train_data = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

print("\ntrain data info looks like:\n")

train_data.info()

train_data.head(10)
# reading the test data

test_data = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")

print("test data info looks like:\n")

test_data.info()

test_data.head(5)
# defining the target

X = train_data.drop(['target'],axis=1)

print('the shape of X is {}'.format(X.shape))

y = train_data['target']

print('the shape of y is {}'.format(y.shape))

X.head()
from sklearn.preprocessing import LabelEncoder



my_encoder = LabelEncoder()



X_encoded = X.copy()

for c in X.columns:

    if (X[c].dtype == 'object'):

        X_encoded[c] = my_encoder.fit_transform(X[c])



X_encoded.head()
# dividing the train data to 75% train set and 25% evaluation set

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X_encoded,y,random_state=1,test_size=0.25)

print('the shape of X_train is {}'.format(X_train.shape))

print('the shape of X_test is {}'.format(X_test.shape))

print('the shape of y_train is {}'.format(y_train.shape))

print('the shape of y_test is {}'.format(y_test.shape))
from sklearn.linear_model import  LogisticRegression



model_LogR = LogisticRegression(max_iter=500, C=0.10)

model_LogR.fit(X_train,y_train)

y_pre_LogR = model_LogR.predict(X_test)



from sklearn.metrics import accuracy_score

print('Accuracy : ',accuracy_score(y_test,y_pre_LogR))
from sklearn.neighbors import  KNeighborsClassifier



model_KNN = KNeighborsClassifier(n_neighbors=1)

model_KNN.fit(X_train,y_train)

y_pre_KNN = model_KNN.predict(X_test)



from sklearn.metrics import accuracy_score

print('Accuracy : ',accuracy_score(y_test,y_pre_KNN))
from sklearn.ensemble import RandomForestClassifier



model_RF = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1)

model_RF.fit(X_train,y_train)

y_pre_RF = model_RF.predict(X_test)



from sklearn.metrics import accuracy_score

print('Accuracy : ',accuracy_score(y_test,y_pre_RF))
from sklearn.preprocessing import OneHotEncoder



my_encoder_OH = OneHotEncoder()

my_encoder_OH.fit(X)



X_encoded_OH = my_encoder_OH.transform(X)



print('the shape of X_encoded_OH is {}'.format(X_encoded_OH.shape))



# dividing the train data to 75% train set and 25% evaluation set

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X_encoded_OH,y,random_state=1,test_size=0.25)

print('the shape of X_train is {}'.format(X_train.shape))

print('the shape of X_test is {}'.format(X_test.shape))

print('the shape of y_train is {}'.format(y_train.shape))

print('the shape of y_test is {}'.format(y_test.shape))
from sklearn.linear_model import  LogisticRegression



model_LogR = LogisticRegression(max_iter=5000, C=0.01)

model_LogR.fit(X_train,y_train)

y_pre_LogR = model_LogR.predict(X_test)



from sklearn.metrics import accuracy_score

print('Accuracy : ',accuracy_score(y_test,y_pre_LogR))
from sklearn.neighbors import  KNeighborsClassifier



model_KNN = KNeighborsClassifier(n_neighbors=1)

model_KNN.fit(X_train,y_train)

y_pre_KNN = model_KNN.predict(X_test)



from sklearn.metrics import accuracy_score

print('Accuracy : ',accuracy_score(y_test,y_pre_KNN))
from sklearn.ensemble import RandomForestClassifier



model_RF = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1)

model_RF.fit(X_train,y_train)

y_pre_RF = model_RF.predict(X_test)



from sklearn.metrics import accuracy_score

print('Accuracy : ',accuracy_score(y_test,y_pre_RF))
# creating a dataframe of all samples

X_test_actual=test_data.copy()

print('the shape of X_test_actual is {}'.format(X_test_actual.shape))

X_all = pd.concat([X, X_test_actual])

print('the shape of X_all is {}'.format(X_all.shape))



# encoding the dataframes

from sklearn.preprocessing import OneHotEncoder



my_encoder_OH_all = OneHotEncoder()

my_encoder_OH_all.fit(X_all)

X_test_actual_OH = my_encoder_OH_all.transform(X_test_actual) 

print('the shape of X_test_actual_OH is {}'.format(X_test_actual_OH.shape))

X_OH = my_encoder_OH_all.transform(X)

print('the shape of X_OH is {}'.format(X_OH.shape))

print('the shape of y is {}'.format(y.shape))



# fitting logistic regression

from sklearn.linear_model import  LogisticRegression



model_LogR = LogisticRegression(max_iter=5000, C=0.1)

model_LogR.fit(X_OH,y)

y_pre_LogR = model_LogR.predict(X_test_actual_OH)

print('the shape of y_pre_LogR is {}'.format(y_pre_LogR.shape))



output = pd.DataFrame({'id': X_test_actual.id, 'target': y_pre_LogR})

output.to_csv('my_submission_v1.csv', index=False)

print("Your submission was successfully saved!")
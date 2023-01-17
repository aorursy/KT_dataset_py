# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv")

test_data=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv")
train_data.info()
test_data.info()
y=train_data['price_range']

y.unique()
x=train_data.drop(columns=['price_range','id'])

x
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)

print(x_train.shape)
print(x_test.shape)
#lr

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100,multi_class='multinomial', random_state=0,max_iter=1000)

lr = lr.fit(x_train, y_train)
pred=lr.predict(x_test)
print(lr.score(x_train, y_train))

print(lr.score(x_test, y_test))
#KNN

from sklearn.neighbors import KNeighborsClassifier  

knn = KNeighborsClassifier(n_neighbors=3)  

knn.fit(x_train, y_train)



knnPickle = open('/kaggle/working/knnpickle_file', 'wb') 



# source, destination 

pickle.dump(knn, knnPickle)                      

loaded_model = pickle.load(open('knnpickle_file', 'rb'))

knn_pred = loaded_model.predict(x_test)

print(knn.score(x_train, y_train))

print(knn.score(x_test, y_test))
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb = nb.fit(x_train, y_train)
print(nb.score(x_train, y_train))

print(nb.score(x_test, y_test))
test_data=test_data.drop('id',axis=1)
test_data
final_pred=loaded_model.predict(test_data)
final_pred
sample_submission=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/sample_submission.csv')

data={'id':sample_submission['id'],'price_range':final_pred}

result=pd.DataFrame(data)

result.to_csv('/kaggle/working/result_knn.csv',index=False)

output=pd.read_csv('/kaggle/working/result_knn.csv')
#svm

from sklearn.svm import SVC

svm = SVC(C=1.0, kernel='linear', random_state=0)

svm = svm.fit(x_train, y_train)
print(svm.score(x_train, y_train))

print(svm.score(x_test, y_test))
final_pred1=svm.predict(test_data)

final_pred1
data1={'id':sample_submission['id'],'price_range':final_pred1}

result1=pd.DataFrame(data1)

result1.to_csv('/kaggle/working/result_svm.csv',index=False)

output1=pd.read_csv('/kaggle/working/result_svm.csv')
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini',

                              max_depth=4, 

                              random_state=0)

dt = dt.fit(x_train, y_train)

print(dt.score(x_train, y_train))

print(dt.score(x_test, y_test))
t=test_data.iloc[2:3,:]

final_pred2=svm.predict(t)

final_pred2
data2={'id':sample_submission['id'],'price_range':final_pred2}

result2=pd.DataFrame(data2)

result2.to_csv('/kaggle/working/result_dt.csv',index=False)

output2=pd.read_csv('/kaggle/working/result_dt.csv')
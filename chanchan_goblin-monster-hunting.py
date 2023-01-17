# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from xgboost import XGBClassifier



# Any results you write to the current directory are saved as output.
##########

data_train=pd.read_csv("../input/train.csv",index_col=0)

data_test=pd.read_csv("../input/test.csv",index_col=0)
features=["bone_length","rotting_flesh","hair_length","has_soul"]

X=data_train[features]

y=data_train["type"]
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(2)

X_poly_2=poly.fit_transform(X)

X_poly_2=pd.DataFrame(X_poly_2)

X_poly_2_new=X_poly_2.iloc[:,1:16]
from sklearn.preprocessing import LabelEncoder as LE

letype = LE()

y = letype.fit_transform(y)
from sklearn.model_selection import train_test_split

# current test size = 0 to permit the usage of whole training data

X_train, X_test, y_train, y_test = train_test_split(X_poly_2_new,y, test_size=0.75)
from tensorflow.contrib import learn

x=tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

tf_clf_dnn = learn.DNNClassifier(hidden_units=[16], n_classes=3, feature_columns=x, activation_fn=tf.sigmoid)

tf_clf_dnn.fit(X_train, y_train,max_steps=5000)



from sklearn.metrics import accuracy_score as acc_s



print(acc_s(y_train,tf_clf_dnn.predict(X_train)))
print(acc_s(y_test,tf_clf_dnn.predict(X_test)))
clf = tf.contrib.learn.LinearClassifier(

        feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(X_train),

        n_classes=3,

        #optimizer=tf.train.FtrlOptimizer(

        #    learning_rate=0.1,

        #    l2_regularization_strength=0.001,

        optimizer=tf.train.AdagradOptimizer(

            learning_rate=0.5,

        ))

clf.fit(X_train, y_train, steps=500)
print(acc_s(y_test,clf.predict(X_test)))
xgb_predict=XGBClassifier(objective="multi:softprob")
xgb_predict.fit(X_train,y_train)

xgb_predict.score(X_test,y_test)
features=["bone_length","rotting_flesh","hair_length","has_soul"]

X_1=data_test[features]

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(2)

X_poly_2_pre=poly.fit_transform(X_1)

X_poly_2_pre=pd.DataFrame(X_poly_2_pre)

X_poly_2_new_pre=X_poly_2_pre.iloc[:,1:16]
a=tf_clf_dnn.predict(X_poly_2_new_pre)

a=letype.inverse_transform(a)
data_test.shape
a_new=pd.DataFrame(a,columns=["type"])

a_new["id"]=data_test.index

a_new.set_index("id",drop=True,inplace=True)
a_new.to_csv("tf_predict.csv")
data_1=pd.read_csv("tf_predict.csv")
data_1.head()
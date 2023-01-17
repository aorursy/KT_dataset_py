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
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

df.head()



import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline





def plot_corr(df,size = 11):

    f = plt.figure(figsize=(19, 15))

    corr = df.corr()

    plt.matshow(corr, fignum=f.number)

   

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)

    

    #plt.show()

    

plot_corr(df)

df.corr()
df.isnull().values.any()
df.columns
from sklearn.model_selection import train_test_split



feature_col_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

predicted_class_names = ['Class']



X = df[feature_col_names].values

Y = df[predicted_class_names].values

split_test_size = 0.3



X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=split_test_size, random_state = 42)

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())





nb_predict_train = nb_model.predict(X_train)

nb_predict_test = nb_model.predict(X_test)



from sklearn import metrics



print("Accuracy training : {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train )))

print("Accuracy testing : {0:.4f}".format(metrics.accuracy_score(y_test,nb_predict_test )))

print('**********Confusion metrics**************')

print("{}".format(metrics.confusion_matrix(y_test, nb_predict_test)))

print('**********Classsification report**************')

print("{}".format(metrics.classification_report(y_test, nb_predict_test)))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state = 42)

rf_model.fit(X_train, y_train.ravel())



rf_predict_train = rf_model.predict(X_train)

rf_predict_test = rf_model.predict(X_test)



from sklearn import metrics



print("RF Accuracy training : {0:.4f}".format(metrics.accuracy_score(y_train,rf_predict_train )))

print("RF Accuracy testing : {0:.4f}".format(metrics.accuracy_score(y_test,rf_predict_test )))

print('********** RF Confusion metrics**************')

print("{}".format(metrics.confusion_matrix(y_test, rf_predict_test)))

print('**********RF Classsification report**************')

print("{}".format(metrics.classification_report(y_test, rf_predict_test)))
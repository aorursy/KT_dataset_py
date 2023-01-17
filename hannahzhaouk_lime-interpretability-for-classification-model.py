# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit

from imblearn.over_sampling import ADASYN



from sklearn.svm import SVC, LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import cross_val_predict,cross_validate

from sklearn import metrics

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score

from sklearn.metrics import make_scorer



from sklearn.pipeline import make_pipeline



# Input raw data

telcom=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#checking missing data

telcom['TotalCharges']=telcom['TotalCharges'].convert_objects(convert_numeric=True)

telcom["TotalCharges"].dtypes

#drop missing data

telcom.dropna(inplace=True)
#find features and the values of target variable

feature_names=telcom.iloc[:,1:].columns

#select values of target variable - 2-class categorical data

labels=telcom.iloc[:,20]



#encoding target variable

le=LabelEncoder()

le.fit(labels)

labels=le.transform(labels)

class_names=le.classes_

telcom=telcom.iloc[:,1:-1]

#check independent features which data value in categorical type

obj_features=telcom.select_dtypes(['object']).columns

categorical_features=[telcom.columns.get_loc(c) for c in obj_features]
#encoding categorical data of those features

categorical_names = {}

for feature in categorical_features:

    le = LabelEncoder()

    le.fit(telcom.iloc[:, feature])

    telcom.iloc[:, feature] = le.transform(telcom.iloc[:, feature])

    categorical_names[feature] = le.classes_
#one-hot-encoding on categorical_features

encoder = OneHotEncoder(categorical_features=categorical_features)

encoder.fit(telcom)
#set all data type to float

telcom=telcom.astype(float)
X=telcom

y=labels



#Split train/test sets of X and y

np.random.seed(1)

sss=StratifiedShuffleSplit(n_splits=5, test_size=0.2,random_state=0)

sss.get_n_splits(X,y)



#Training/Testing sets in 5 folds

for train_index, test_index in sss.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)
ada= ADASYN()
accuracy_scores = []

precision_scores = []

recall_scores = []

f1_scores = []



for train_index, test_index in sss.split(X, y):

    X_train,X_test=X.iloc[train_index], X.iloc[test_index]

    y_train,y_test=y[train_index], y[test_index]



    ### One-hot-encoding training/testing data

    encoded_train, encoded_test = encoder.transform(X_train), encoder.transform(X_test)

    

    ### Oversampling training sets

    X_resample,y_resample=ada.fit_sample(encoded_train,y_train)

    

    ### Model training

    lr=LogisticRegression()

    lr.fit(X_resample, y_resample)

    

    ### Make Prediction

    y_pred = lr.predict(encoded_test)

    

    ### Performance evaluation

    accuracy_scores.append(accuracy_score(y_test, y_pred))

    precision_scores.append(precision_score(y_test, y_pred))

    recall_scores.append(recall_score(y_test, y_pred))

    f1_scores.append(f1_score(y_test, y_pred))

    

print("----------------- Performance Evaluation -----------------")

print('Accuracy', np.mean(accuracy_scores))

print('Precision', np.mean(precision_scores))

print('Recall', np.mean(recall_scores))

print('F1-measure', np.mean(f1_scores)) 
predict_fn = lambda x: lr.predict_proba(encoder.transform(x)).astype(float)
import lime

import lime.lime_tabular

from __future__ import print_function
#implement LIME interpretation

explainer = lime.lime_tabular.LimeTabularExplainer(

    X_train.values,feature_names = feature_names,

    class_names=class_names,categorical_features=categorical_features, 

    categorical_names=categorical_names, kernel_width=3

)
#visualise LIME interpretation

np.random.seed(1)

i = int(np.random.randint(0,1407,size=1))

exp = explainer.explain_instance(X_test.values[i], predict_fn, num_features=5)

exp.show_in_notebook(show_all=False)
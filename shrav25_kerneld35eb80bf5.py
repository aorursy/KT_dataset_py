# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss,RandomUnderSampler

from imblearn.over_sampling import ADASYN,SMOTENC,BorderlineSMOTE,SVMSMOTE,RandomOverSampler

import seaborn as sns

import matplotlib as plt

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from keras.layers import Input, Dense

from keras.models import Model

from keras.models import Sequential

from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score,classification_report, recall_score, precision_score,confusion_matrix,mean_absolute_error,mean_squared_error

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV



#Learning curve

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import validation_curve
train_dd = pd.read_excel("../input/Train.xlsx")
train_dd.head(5)
train_dd.tail(5)
train_dd.describe(include='all')
train_dd['Price_Per_unit'] = train_dd['TotalSalesValue']/train_dd['Quantity']

train_dd.head()
sns.scatterplot(x='ProductID',y='Price_Per_unit',data=train_dd)

sns.scatterplot(x='Suspicious',y='Quantity',data=train_dd)

sns.barplot(x='Suspicious',y='TotalSalesValue',data=train_dd)
train_dd.Suspicious.value_counts()
train_dd.info()
for col in['ReportID','SalesPersonID','ProductID','Suspicious']:

    train_dd[col] = train_dd[col].astype('category')
y = train_dd['Suspicious']

X=train_dd.drop('Suspicious',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
y_train.value_counts()
num_col = X.select_dtypes(include=['float64','int64']).columns

num_col
cat_col = X.select_dtypes(include=['category']).columns

cat_col
scaler = StandardScaler()

scaler.fit(X[num_col])

X[num_col] = scaler.transform(X[num_col])
le1 = preprocessing.LabelEncoder()

for i in cat_col:

    le1.fit(X[i])

    X[i] = le1.fit_transform(X[i])
catr_col_train = X_train.select_dtypes(include='category').columns

catr_col_train
numr_col_train = X_train.select_dtypes(include=['float64','int64']).columns

numr_col_train
cat_col_test = X_test.select_dtypes(include=['category']).columns

cat_col_test
num_col_test = X_test.select_dtypes(include=['float64','int64']).columns

num_col_test
le1 = preprocessing.LabelEncoder()

for i in cat_col_test:

    le1.fit(X_test[i])

    X_test[i] = le1.fit_transform(X_test[i])
le2 = preprocessing.LabelEncoder()

for i in catr_col_train:

    le2.fit(X_train[i])

    X_train[i] = le2.fit_transform(X_train[i])
scaler = StandardScaler()

scaler.fit(X_train[numr_col_train])

X_train[numr_col_train] = scaler.transform(X_train[numr_col_train])
scaler = StandardScaler()

scaler.fit(X_test[num_col_test])

X_test[num_col_test] = scaler.transform(X_test[num_col_test])
X_train.head()
X_test.head()
sme = SMOTE()

(X_train,y_train) = sme.fit_sample(X_train,y_train)
np.bincount(y_train)
encoding_dim=80

actual_dim = X_train.shape[1]
input_img = Input(shape=(actual_dim,))

encoded = Dense(encoding_dim,activation='relu')(input_img)

decoded = Dense(actual_dim,activation='softmax')(encoded)
autoencoder = Model(input_img,decoded)

print(autoencoder.summary())
autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
autoencoder.fit(X_train,X_train,epochs=30,batch_size=32)
encoder = Model(input_img,encoded)

print(encoder.summary())
#New non-linear features
X_train_nonlinear = encoder.predict(X_train)

X_test_nonlinear = encoder.predict(X_test)
X_train = np.concatenate((X_train,X_train_nonlinear),axis=1)

X_test = np.concatenate((X_test,X_test_nonlinear),axis=1)
X_train = pd.DataFrame(data=X_train)
X_train.head()
X_test = pd.DataFrame(data=X_test)
X_test.head()
X_train.shape
X_test.shape
RF =RandomForestClassifier(n_estimators=100,random_state=123,class_weight='balanced')

RF.fit(X_train,y_train)

y_pred = RF.predict(X_test)



print(accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred,digits=4))
print(confusion_matrix(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
rfc_grid = RandomForestClassifier(n_jobs=-1,max_features='sqrt')

param_grid = {

    'n_estimators': [9,18,36,72,142,200],

    'max_depth': [5,10,15],

    'min_samples_leaf': [2,4]

}



rfc_cv_grid = GridSearchCV(estimator=rfc_grid,param_grid=param_grid,cv=15)
%%time

rfc_cv_grid.fit(X_train1,y_train)
model = rfc_cv_grid.best_estimator_

print(rfc_cv_grid.best_score_,rfc_cv_grid.best_params_)
y_pred_train = model.predict(X_train1)
print(accuracy_score(y_train,y_pred_train))

print(classification_report(y_train,y_pred_train,digits=4))
y_pred_test = model.predict(X_test)
print(accuracy_score(y_test,y_pred_test))

print(classification_report(y_test,y_pred_test,digits=4))
#ROC Curve for Random Forest
#Learning Cruve
lr = LogisticRegression()

lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)
y_pred_lr_train = lr.predict(X_train)
print(accuracy_score(y_train,y_pred_lr_train))

print(classification_report(y_train,y_pred_lr_train,digits=4))
print(accuracy_score(y_test,y_pred_lr))

print(classification_report(y_test,y_pred_lr,digits=4))
KNN = KNeighborsClassifier()

KNN.fit(X_train,y_train)
y_pred_Knn_test = KNN.predict(X_test)

y_pred_Knn_train = KNN.predict(X_train)
print(accuracy_score(y_test,y_pred_Knn_test))

print(classification_report(y_test,y_pred_Knn_test,digits=4))
print(accuracy_score(y_train,y_pred_Knn_train))

print(classification_report(y_train,y_pred_Knn_train,digits=4))
#GBM Model
GBM_Model = GradientBoostingClassifier(n_estimators=50,learning_rate=0.03,subsample=0.3)
%time GBM_Model.fit(X_train,y_train)
y_pred_GBM = GBM_Model.predict(X_test)
y_pred_train_gbm=GBM_Model.predict(X_train)
print(accuracy_score(y_test,y_pred_GBM))

print(classification_report(y_test,y_pred_GBM,digits=4))
print(accuracy_score(y_train,y_pred_train_gbm))

print(classification_report(y_train,y_pred_train_gbm,digits=4))
recall_score(y_test,y_pred_Knn_test,average=None)
recall_score(y_test,y_pred_GBM,average=None)
recall_score(y_test,y_pred_lr,average=None)
#Decision Tree Model
DT = DecisionTreeClassifier(max_depth=6,random_state=123)

DT.fit(X_train,y_train)
y_pred_DT_train = DT.predict(X_train)

y_pred_DT_test = DT.predict(X_test)
print(accuracy_score(y_train,y_pred_DT_train))

print(classification_report(y_train,y_pred_DT_train,digits=4))

print("\n")



print(accuracy_score(y_test,y_pred_DT_test))

print(classification_report(y_test,y_pred_DT_test,digits=4))
recall_score(y_test,y_pred_DT_test,average=None)
%time

XGB_mod = XGBClassifier(n_estimators=300,learning_rate=0.5,gamma=0.5)

XGB_mod.fit(X_train,y_train)
y_pred_XGB_test = XGB_mod.predict(X_test)
print(accuracy_score(y_test,y_pred_XGB_test))

print(classification_report(y_test,y_pred_XGB_test,digits=4))
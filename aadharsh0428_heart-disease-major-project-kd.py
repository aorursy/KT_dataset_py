import pandas as pd

import numpy as np

heart = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart.columns
heart.head
heart.target.value_counts()
y=heart.target

heart=heart.drop('target',axis=1)

a=heart.columns
from sklearn.preprocessing import RobustScaler

rob_scaler = RobustScaler()



heart['scaled_trestbps'] = rob_scaler.fit_transform(heart['trestbps'].values.reshape(-1,1))

heart['scaled_chol'] = rob_scaler.fit_transform(heart['chol'].values.reshape(-1,1))

heart['scaled_thalach'] = rob_scaler.fit_transform(heart['thalach'].values.reshape(-1,1))

heart['scaled_age'] = rob_scaler.fit_transform(heart['age'].values.reshape(-1,1))

heart.drop(['trestbps','chol','thalach', 'age'], axis=1, inplace=True)
scaled_trestbps = heart['scaled_trestbps']

scaled_chol = heart['scaled_chol']

scaled_thalach = heart['scaled_thalach']

scaled_age = heart['scaled_age']

heart.drop(['scaled_trestbps', 'scaled_chol', 'scaled_thalach', 'scaled_age'], axis=1, inplace=True)

heart.insert(0, 'scaled_trestbps', scaled_trestbps)

heart.insert(1, 'scaled_chol', scaled_chol)

heart.insert(2, 'scaled_thalach', scaled_thalach)

heart.insert(3, 'scaled_age', scaled_age)

heart.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(heart, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import *

model=LogisticRegression()

model.fit(x_train,y_train)

pred=model.predict(x_test)

target_names=['class 0','class 1']

print(classification_report(y_test,pred,target_names=target_names))
import xgboost as xgb

D_train = xgb.DMatrix(x_train, label=y_train)

D_test = xgb.DMatrix(x_test, label=y_test)

param = {

    'eta': 0.3, 

    'max_depth': 3,  

    'objective': 'multi:softprob',  

    'num_class': 3} 



steps = 20



model = xgb.train(param, D_train, steps)



preds2 = model.predict(D_test)

best_preds = np.asarray([np.argmax(line) for line in preds2])



target_names=['class 0','class 1']

print(classification_report(y_test,best_preds,target_names=target_names))
from sklearn.ensemble import RandomForestClassifier



regressor = RandomForestClassifier(n_estimators=20, random_state=0)

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print(classification_report(y_test,y_pred))
from sklearn.svm import SVC

svclassifier = SVC(kernel='linear')

svclassifier.fit(x_train, y_train)

y_pred = svclassifier.predict(x_test)

print(classification_report(y_test,y_pred))

preds = pd.DataFrame(y_pred,x_test)

preds.to_csv('submission.csv')
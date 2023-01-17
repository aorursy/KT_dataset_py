import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')
data.head()
data.drop(['Date' , 'Evaporation' , 'Sunshine' , 'Cloud9am' , 'Cloud3pm' , 'RISK_MM'] , axis=1 , inplace=True)
# One hot Encoding

data = pd.concat([data,pd.get_dummies(data['Location'], drop_first=True , dummy_na=True)],axis=1).drop(['Location'] , axis=1)

data = pd.concat([data,pd.get_dummies(data['WindGustDir'], drop_first=True , dummy_na=True)],axis=1).drop(['WindGustDir'] , axis=1)

data = pd.concat([data,pd.get_dummies(data['WindDir9am'], drop_first=True , dummy_na=True)],axis=1).drop(['WindDir9am'] , axis=1)

data = pd.concat([data,pd.get_dummies(data['WindDir3pm'], drop_first=True , dummy_na=True)],axis=1).drop(['WindDir3pm'] , axis=1)

data = pd.concat([data,pd.get_dummies(data['RainToday'], drop_first=True , dummy_na=True)],axis=1).drop(['RainToday'] , axis=1)
data.head()
data.dropna(axis=0 , inplace=True)



y = data.RainTomorrow

y = y.map({'No' : 0 , 'Yes' : 1})

y = np.array(y)



x = data.drop(['RainTomorrow'] , axis=1)
# Scaling

from sklearn import preprocessing

x = preprocessing.scale(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)
# Naive Bayes [We can understand from the accuracy why that it is called 'naive' :)) ]

from sklearn.naive_bayes import GaussianNB

Classifier = GaussianNB()

Classifier.fit(x_train, y_train)

# make predictions for test data

y_pred = Classifier.predict(x_test)

predictions = [round(value) for value in y_pred]

cm = confusion_matrix(y_test, predictions)

print(cm)

accuracy=accuracy_score(y_test,predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Logistic Regression

from sklearn import linear_model

Classifier = linear_model.LogisticRegression()



Classifier.fit(x_train, y_train)

# make predictions for test data

y_pred = Classifier.predict(x_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

cm = confusion_matrix(y_test, predictions)

print(cm)

accuracy=accuracy_score(y_test,predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# XGBOOST

from xgboost import XGBClassifier

Classifier = XGBClassifier()

Classifier.fit(x_train, y_train)

# make predictions for test data

y_pred = Classifier.predict(x_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy=accuracy_score(y_test,predictions)

cm = confusion_matrix(y_test, predictions)

print(cm)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# LightGBM



import lightgbm as lgb



lgb_train = lgb.Dataset(data = x_train , label = y_train , free_raw_data=False)

lgb_eval = lgb.Dataset(data = x_test , label = y_test , free_raw_data=False)



params = {

    'task' : 'train',

    'boosting_type':'gbdt',

    'objective':'binary',

    'metric':'acc',

    'num_leaves':300,

    'learning_rate':0.09,

    'verbose':-1,

    'max_bin':150

}



evals_result = {}

gbm = lgb.train(params ,

                train_set = lgb_train,

                valid_sets = lgb_eval,

                num_boost_round = 250,

                evals_result = evals_result

)



y_pred = gbm.predict(x_test , num_iteration = gbm.best_iteration)

print('Accuracy :' , accuracy_score(y_test , (y_pred >=0.5)*1))





#Feature Importance

ax = lgb.plot_importance(gbm , max_num_features = 10)

plt.show()



#Print Confusion Matrix

plt.figure()

cm = confusion_matrix(y_test, (y_pred >=0.5)*1)

labels = ['No Rain', 'Rain']

plt.figure(figsize=(8,6))

sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')

plt.show()
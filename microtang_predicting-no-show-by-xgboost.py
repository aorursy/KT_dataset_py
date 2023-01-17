import numpy as np 

import pandas as pd 

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import confusion_matrix

from termcolor import colored
df = pd.read_csv('../input/KaggleV2-May-2016.csv')

df.head()
# Choosing the dependent and independent variables

X = df[['Gender', 'ScheduledDay',

       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension',

       'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]

y=df['No-show']
# Transforming the date to int type

X['AppointmentDay'] = X['AppointmentDay'].apply(lambda x: int(x[:4]+x[5:7]+x[8:10]+x[11:13]+x[14:16]+x[17:19]))

X['ScheduledDay'] = X['ScheduledDay'].apply(lambda x: int(x[:4]+x[5:7]+x[8:10]+x[11:13]+x[14:16]+x[17:19]))

# Encoding the categorical data

lc1 = LabelEncoder()

X['Gender'] = lc1.fit_transform(X['Gender'])

lc2 = LabelEncoder()

X['Neighbourhood'] = lc2.fit_transform(X['Neighbourhood'])

labelencoder_y = LabelEncoder()

y = pd.Series(labelencoder_y.fit_transform(y))

# Splitting the dataset into the Training set and Validation set

Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.25, random_state = 0)

dt = xgb.DMatrix(Xt.as_matrix(),label=yt.as_matrix())

dv = xgb.DMatrix(Xv.as_matrix(),label=yv.as_matrix())

# Splitting the dataset into the Training set and Validation set

Xt, Xv, yt, yv = train_test_split(X, y, test_size = 0.25, random_state = 0)

dt = xgb.DMatrix(Xt.as_matrix(),label=yt.as_matrix())

dv = xgb.DMatrix(Xv.as_matrix(),label=yv.as_matrix())
# Building the model

params = {

    "eta": 0.2,

    "max_depth": 4,

    "objective": "binary:logistic",

    "silent": 1,

    "base_score": np.mean(yt),

    'n_estimators': 1000,

    "eval_metric": "logloss"

}

model = xgb.train(params, dt, 500, [(dt, "train"),(dv, "valid")], verbose_eval=500)
#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(yv, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
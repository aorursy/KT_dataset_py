import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools
train_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')

test_df = pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/test.csv')

train_df.head()

test_df.head()
X_train=train_df.drop(columns=['Severity', 'Accident_ID'])

X_test=test_df.drop(columns=['Accident_ID'])

Y_train=train_df['Severity']
import pandas as pd

from sklearn import preprocessing

x = X_train.values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

X_train_final = pd.DataFrame(x_scaled)

X_test_final=pd.DataFrame(min_max_scaler.fit_transform(X_test.values))
labels = {

          "Minor_Damage_And_Injuries": 0,

          "Significant_Damage_And_Fatalities": 1,

          "Significant_Damage_And_Serious_Injuries": 2,

          "Highly_Fatal_And_Damaging":3

}

inv_labels={

       0:"Minor_Damage_And_Injuries",

       1:"Significant_Damage_And_Fatalities",

       2:"Significant_Damage_And_Serious_Injuries",

       3:"Highly_Fatal_And_Damaging"

}

Y_train_final=[]

for d in Y_train:

    Y_train_final.append(labels[d])

    

Y_train_final=np.array(Y_train_final)
clf=LogisticRegression(solver='sag', max_iter=10000, multi_class='auto')
clf.fit(X_train_final, Y_train_final)
X=X_train_final

clf.predict(X)

clf.score(X_train_final, Y_train_final, sample_weight=None)
from sklearn.svm import SVC

clf = SVC(gamma='auto')

clf.fit(X_train_final, Y_train_final)

X=X_train_final

print(clf.predict(X))
clf.score(X_train_final, Y_train_final, sample_weight=None)
clf.predict(X_test_final)

submission = pd.DataFrame([test_df['Accident_ID'], np.vectorize(inv_labels.get)(clf.predict(X_train_final))], index=['Accident_ID', 'Severity']).T

submission.to_csv('submission_svc.csv', index=False)

submission.head()
from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

# integer encode

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(Y_train_final)

# binary encode

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print(onehot_encoded)

from keras.models import Sequential

from keras.layers import Dense

X_train_final.shape
model = Sequential()

model.add(Dense(12, input_dim=10, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(4, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X=X_train_final

y=onehot_encoded

print(X.shape,y.shape)
model.fit(X, y, epochs=150, batch_size=10)
pred_train=np.argmax(model.predict(X),axis=1)

print(pred_train)
pred_test=np.argmax(model.predict(X_test_final),axis=1)

print(pred_test)
submission = pd.DataFrame([test_df['Accident_ID'], np.vectorize(inv_labels.get)(pred_test)], index=['Accident_ID', 'Severity']).T

submission.to_csv('submission_nn.csv', index=False)

submission.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(n_estimators=1250, random_state=666, oob_score=True)



# 0.8589427

param_grid = { 

    'n_estimators': [1000],

    'max_features': [None],

    'min_samples_split': [3],

    'max_depth': [50]

    

}



CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=6, verbose=100, n_jobs=-1)

CV_rf.fit(X_train_final, Y_train_final)

print (f'Best Parameters: {CV_rf.best_params_}')
submission = pd.DataFrame([test_df['Accident_ID'], np.vectorize(inv_labels.get)(CV_rf.predict(X_test_final))], index=['Accident_ID', 'Severity']).T

submission.to_csv('submission_rf.csv', index=False)

submission.head()
Y_train_final
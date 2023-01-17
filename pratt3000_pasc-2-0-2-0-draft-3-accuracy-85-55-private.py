# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn import neighbors
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from numpy import concatenate
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import to_datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot

from keras import Sequential
from keras.layers import Dense, Dropout
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
X = pd.read_csv("/kaggle/input/pasc-data-quest-20-20/doctor_train.csv")
testF = pd.read_csv("/kaggle/input/pasc-data-quest-20-20/doctor_test.csv")
soln = pd.read_csv("/kaggle/input/derivedfff/Y_test.csv")

soln["Y"] = soln["Y"].replace(['yes'],1 )
soln["Y"] = soln["Y"].replace(['no'], 0 )
soln = soln["Y"].to_numpy()

testF['age'].fillna((testF['age'].mean()), inplace=True)
testF['Money'].fillna((testF['Money'].mean()), inplace=True)
testF["edu"].fillna("unknown", inplace = True) 
testF["prev_diagnosed"].fillna("None", inplace = True) 
testF["residence"].fillna("None", inplace = True) 
X['age'].fillna((X['age'].mean()), inplace=True)
X['Money'].fillna((X['Money'].mean()), inplace=True)
X["edu"].fillna("unknown", inplace = True) 
X["prev_diagnosed"].fillna("None", inplace = True) 
X["residence"].fillna("None", inplace = True) 
del testF["ID"]
del X["ID"]

#X['age_Range'] = pd.cut(X['age'], bins=[0, 33, 40, 48, 100], labels=[1,2,3,4])

#X['Money_Range'] = pd.cut(X['Money'], bins=[-10000, 83, 487, 1400, 202127], labels=[1,2,3,4])
from sklearn import preprocessing
x = X[['Money']].values.astype(float)
minmaxScaler = preprocessing.MinMaxScaler()
X['MoneyTran'] = minmaxScaler.fit_transform(x)
x = testF[['Money']].values.astype(float)
testF['MoneyTran'] = minmaxScaler.fit_transform(x)
testF['MoneyTran'] = testF['MoneyTran'] + 0.00001
X['MoneyTran'] = X['MoneyTran'] + 0.00001

#X['day_Range'] = pd.cut(X['day'], bins=[-1, 8, 16, 21, 35], labels=[1,2,3,4])
#X['Time_Range'] = pd.cut(X['Time'], bins=[-1, 103, 179, 320, 40000], labels=[1,2,3,4])
X['DoctorVisits_Range'] = pd.cut(X['Doctor_visits'], bins=[0, 1, 2, 3, 58], labels=[1,2,3,4])
#del lastvisit, cured_in probably # check its valuse count below
X['curedIn_Range'] = pd.cut(X['cured_in'], bins=[-1, 0, 1, 3, 275], labels=[1,2,3,4])
X['lastVisit_Range'] = pd.cut(X['last_visit'], bins=[-20, 0, 150, 270, 1000], labels=[1,2,3,4])



#testF['age_Range'] = pd.cut(testF['age'], bins=[0, 33, 40, 48, 100], labels=[1,2,3,4])
#testF['Money_Range'] = pd.cut(testF['Money'], bins=[-10000, 83, 487, 1400, 202127], labels=[1,2,3,4])
#testF['day_Range'] = pd.cut(testF['day'], bins=[-1, 8, 16, 21, 35], labels=[1,2,3,4])
#testF['Time_Range'] = pd.cut(testF['Time'], bins=[-1, 103, 179, 320, 40000], labels=[1,2,3,4])
testF['DoctorVisits_Range'] = pd.cut(testF['Doctor_visits'], bins=[0, 1, 2, 3, 58], labels=[1,2,3,4])
#del lastvisit, cured_in probably # check its valuse count below
testF['curedIn_Range'] = pd.cut(testF['cured_in'], bins=[-1, 0, 1, 3, 275], labels=[1,2,3,4])
testF['lastVisit_Range'] = pd.cut(testF['last_visit'], bins=[-20, 0, 150, 270, 1000], labels=[1,2,3,4])


#del the old columns
#del X['age']
del X['Money']
#del X['day']
#del X['Time']
del X['Doctor_visits']
del X['cured_in']
del X['last_visit']

#del testF['age']
del testF['Money']
#del testF['day']
#del testF['Time']
del testF['Doctor_visits']
del testF['cured_in']
del testF['last_visit']

#X['age_Range'] = X['age_Range'].astype(int)
#X['Money_Range'] = X['Money_Range'].astype(int)
#X['day_Range'] = X['day_Range'].astype(int)
#X['Time_Range'] = X['Time_Range'].astype(int)
X['DoctorVisits_Range'] = X['DoctorVisits_Range'].astype(int)
X['curedIn_Range'] = X['curedIn_Range'].astype(int)
X['lastVisit_Range'] = X['lastVisit_Range'].astype(int)


#testF['age_Range'] = testF['age_Range'].astype(int)
#testF['Money_Range'] = testF['Money_Range'].astype(int)
#testF['day_Range'] = testF['day_Range'].astype(int)
#testF['Time_Range'] = testF['Time_Range'].astype(int)
testF['DoctorVisits_Range'] = testF['DoctorVisits_Range'].astype(int)
testF['curedIn_Range'] = testF['curedIn_Range'].astype(int)
testF['lastVisit_Range'] = testF['lastVisit_Range'].astype(int)

X.Profession = pd.Categorical(X.Profession)
X['Profession'] = X.Profession.cat.codes
X.Month = pd.Categorical(X.Month)
X['Month'] = X.Month.cat.codes
X.Status = pd.Categorical(X.Status)
X['Status'] = X.Status.cat.codes
X.edu = pd.Categorical(X.edu)
X['edu'] = X.edu.cat.codes
X.Irregular = pd.Categorical(X.Irregular)
X['Irregular'] = X.Irregular.cat.codes
#X.residence = pd.Categorical(X.residence)
#X['residence'] = X.residence.cat.codes
X['resi'] = 5
X.loc[X['residence'] == 'no', 'resi'] = 1
del X['residence']
testF['resi'] = 5
testF.loc[testF['residence'] == 'no', 'resi'] = 1
del testF['residence']

#X.prev_diagnosed = pd.Categorical(X.prev_diagnosed)
#X['prev_diagnosed'] = X.prev_diagnosed.cat.codes
X['pd'] = 1
X.loc[X['prev_diagnosed'] == 'no', 'pd'] = 5
del X['prev_diagnosed']
testF['pd'] = 1
testF.loc[testF['prev_diagnosed'] == 'no', 'pd'] = 5
del testF['prev_diagnosed']

X['assetIndex'] = X['MoneyTran'] * X['pd'] * X['resi']
testF['assetIndex'] = testF['MoneyTran'] * testF['pd'] * testF['resi']
X['assetIndex_Range'] = pd.cut(X['assetIndex'], bins=[-1, 0.3, 0.57, 1.9, 100], labels=[1,2,3,4])
testF['assetIndex_Range'] = pd.cut(testF['assetIndex'], bins=[-1, 0.3, 0.57, 1.9, 100], labels=[1,2,3,4])
testF['assetIndex_Range'] = testF['assetIndex_Range'].astype(int)
X['assetIndex_Range'] = X['assetIndex_Range'].astype(int)
del testF['assetIndex']
del X['assetIndex']


X.communication = pd.Categorical(X.communication)
X['communication'] = X.communication.cat.codes
X.side_effects = pd.Categorical(X.side_effects)
X['side_effects'] = X.side_effects.cat.codes


testF.Profession = pd.Categorical(testF.Profession)
testF['Profession'] = testF.Profession.cat.codes
testF.Month = pd.Categorical(testF.Month)
testF['Month'] = testF.Month.cat.codes
testF.Status = pd.Categorical(testF.Status)
testF['Status'] = testF.Status.cat.codes
testF.edu = pd.Categorical(testF.edu)
testF['edu'] = testF.edu.cat.codes
testF.Irregular = pd.Categorical(testF.Irregular)
testF['Irregular'] = testF.Irregular.cat.codes
#testF.residence = pd.Categorical(testF.residence)
#testF['residence'] = testF.residence.cat.codes
#testF.prev_diagnosed = pd.Categorical(testF.prev_diagnosed)
#testF['prev_diagnosed'] = testF.prev_diagnosed.cat.codes
testF.communication = pd.Categorical(testF.communication)
testF['communication'] = testF.communication.cat.codes
testF.side_effects = pd.Categorical(testF.side_effects)
testF['side_effects'] = testF.side_effects.cat.codes

X['Y'] = X['Y'].replace(['yes'], 1)
X['Y'] = X['Y'].replace(['no'], 0)

ageT1 = X[['age']].values.astype(float)
dayT1 = X[['day']].values.astype(float)
durT1 = X[['Time']].values.astype(float)
X['ageTran'] = minmaxScaler.fit_transform(ageT1)
X['dayTran'] = minmaxScaler.fit_transform(dayT1)
X['TimeTran'] = minmaxScaler.fit_transform(durT1)
del X['age']
del X['day']
del X['Time']

ageT1 = testF[['age']].values.astype(float)
dayT1 = testF[['day']].values.astype(float)
durT1 = testF[['Time']].values.astype(float)
testF['ageTran'] = minmaxScaler.fit_transform(ageT1)
testF['dayTran'] = minmaxScaler.fit_transform(dayT1)
testF['TimeTran'] = minmaxScaler.fit_transform(durT1)
del testF['age']
del testF['day']
del testF['Time']


Y = X["Y"]
del X["Y"]

X = pd.get_dummies(X, columns = ["Profession","Status","edu","Irregular","communication","Month","side_effects","DoctorVisits_Range","curedIn_Range","lastVisit_Range","resi","pd","assetIndex_Range"],
                             prefix=[ "Prof","Status","edu","Irr","comm","Month","SideE","DocV","CIR","LVR","resi","pd","AIR"])
testF = pd.get_dummies(testF, columns = ["Profession","Status","edu","Irregular","communication","Month","side_effects","DoctorVisits_Range","curedIn_Range","lastVisit_Range","resi","pd","assetIndex_Range"],
                             prefix=[ "Prof","Status","edu","Irr","comm","Month","SideE","DocV","CIR","LVR","resi","pd","AIR"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1, random_state=24)

X_train.head()
X_train.describe()
from imblearn.over_sampling import SMOTE
from imblearn import under_sampling, over_sampling

def resamplingDataPrep(X_train, y_train, target_var): 
    # concatenate our training data back together
    resampling = X_train.copy()
    resampling[target_var] = y_train.values
    # separate minority and majority classes
    majority_class = resampling[resampling[target_var]==0]
    minority_class = resampling[resampling[target_var]==1]
    # Get a class count to understand the class imbalance.
    print('majority_class: '+ str(len(majority_class)))
    print('minority_class: '+ str(len(minority_class)))
    return majority_class, minority_class

def upsample_SMOTE(X_train, y_train, ratio):
    """Upsamples minority class using SMOTE.
    Ratio argument is the percentage of the upsampled minority class in relation
    to the majority class. Default is 1.0
    """
    sm = SMOTE(random_state=23, sampling_strategy=ratio)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print('X-size: '+ str(len(X_train_sm)))
    return X_train_sm, y_train_sm

a, b = resamplingDataPrep(X_train, y_train, "Y")
X_train, y_train = upsample_SMOTE(X_train, y_train, ratio = 0.15)
y_train.value_counts()

from sklearn.ensemble import BaggingClassifier

#Create a svm Classifier
model1 = SVC(kernel='rbf') # Linear Kernel

final_bc = BaggingClassifier(base_estimator = model1, n_estimators=40, random_state=1, oob_score=True)
final_bc.fit(X_train, y_train)
#{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}...
#gamma : it decides linearity of classificaction/dividing curve , directyl proportional...
#C : It controls the trade off between smooth decision boundary and classifying the training points correctly...
#degree : degree is a parameter used when kernel is set to ‘poly’.degree of the polynomial used to find the hyperplane to split the data...

preds = final_bc.predict(testF)
preds
X_train1 = X_train.iloc[:9000, :]
X_train2 = X_train.iloc[9000:18000, :]
X_train3 = X_train.iloc[18000:, :]

y_train

y_train1 = y_train[:9000]
y_train2 = y_train[9000:18000]
y_train3 = y_train[18000:]

model1 = Sequential()

model1.add(Dense(16, activation='relu', input_dim=64))
model1.add(Dense(20, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(20, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(20, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(20, activation='relu'))

#Output Layer
model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer ='Adadelta',loss='binary_crossentropy', metrics =['accuracy'])

model1.fit(X_train1,y_train1, batch_size=40, epochs=50)

model2 = Sequential()

model2.add(Dense(16, activation='relu', input_dim=64))
model2.add(Dense(20, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(20, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(20, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(20, activation='relu'))

#Output Layer
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer ='Adadelta',loss='binary_crossentropy', metrics =['accuracy'])

model2.fit(X_train2,y_train2, batch_size=40, epochs=50)


model = Sequential()

model.add(Dense(16, activation='relu', input_dim=64))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))

#Output Layer
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer ='Adadelta',loss='binary_crossentropy', metrics =['accuracy'])

model.fit(X_train3,y_train3, batch_size=40, epochs=50)

y_pred1 = model.predict(testF)
y_pred2 = model1.predict(testF)
y_pred3 = model2.predict(testF)


y_pred = y_pred1/3 + y_pred2/3 + y_pred3/3
y_pred = y_pred.T
#answer
y_pred = model.predict(testF)
#predictions = [round(value) for value in y_pred]
#predictions = np.round(np.array(y_pred))
predictions =(y_pred>0.13)
#print(np.shape(predictions))

sample_submission = pd.read_csv("/kaggle/input/pasc-data-quest-20-20/sample_submission.csv")
sample_submission["Y"] = predictions

sample_submission["Y"] = sample_submission["Y"].replace([True], "yes")
sample_submission["Y"] = sample_submission["Y"].replace([False], "no")


sample_submission.to_csv('submissions.csv', header=True, index=False)
sample_submission.head()


accuracy = accuracy_score(soln, predictions)
print("FINAL Accuracy: %.2f%%" % (accuracy * 100.0))

print(sample_submission["Y"].value_counts())
print(sample_submission)

from sklearn.metrics import accuracy_score
from sklearn import metrics

y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
#y_pred =(y_pred>0.5)
predictions =(y_pred>0.19)
predictions = predictions.replace([False], "no")
predictions = predictions.replace([True], "yes")


accuracy = accuracy_score(soln, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(soln, predictions))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(soln, predictions))
soln=[soln]

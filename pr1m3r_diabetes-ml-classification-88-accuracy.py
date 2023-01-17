# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
np.random.seed(0)



file_path = '/kaggle/input/pima-indians-diabetes-database/diabetes.csv'

data = pd.read_csv(file_path)



data.head(5)
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
data.loc[(data['Outcome'] == 0 ) & (data['Insulin'].isnull()), 'Insulin'] = 102.5

data.loc[(data['Outcome'] == 1 ) & (data['Insulin'].isnull()), 'Insulin'] = 169.5
data.loc[(data['Outcome'] == 0 ) & (data['Glucose'].isnull()), 'Glucose'] = 107

data.loc[(data['Outcome'] == 1 ) & (data['Glucose'].isnull()), 'Glucose'] = 140
data.loc[(data['Outcome'] == 0 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 27

data.loc[(data['Outcome'] == 1 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 32
data.loc[(data['Outcome'] == 0 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 70

data.loc[(data['Outcome'] == 1 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
data.loc[(data['Outcome'] == 0 ) & (data['BMI'].isnull()), 'BMI'] = 30.1

data.loc[(data['Outcome'] == 1 ) & (data['BMI'].isnull()), 'BMI'] = 34.3
(data == 0).sum(axis=0)
data.loc[:, 'N1'] = 0

data.loc[(data['Age'] <= 30) & (data['Glucose'] <= 120), 'N1'] = 1
data.loc[:, 'N2'] = 0

data.loc[(data['BMI'] <= 30), 'N2'] = 1
data.loc[:, 'N3'] = 0

data.loc[(data['Age'] <= 30) & (data['Pregnancies'] <= 6), 'N3'] = 1
data.loc[:,'N4']=0

data.loc[(data['Glucose']<=105) & (data['BloodPressure']<=80),'N4']=1
data.loc[:,'N5']=0

data.loc[(data['SkinThickness']<=20) ,'N5']=1
data.loc[:,'N6']=0

data.loc[(data['BMI']<30) & (data['SkinThickness']<=20),'N6']=1
data.loc[:,'N7']=0

data.loc[(data['Glucose']<=105) & (data['BMI']<=30),'N7']=1
data.loc[:,'N9']=0

data.loc[(data['Insulin']<200),'N9']=1
data.loc[:,'N10']=0

data.loc[(data['BloodPressure']<80),'N10']=1
data.loc[:,'N11']=0

data.loc[(data['Pregnancies']<4) & (data['Pregnancies']!=0) ,'N11']=1
data['N0'] = data['BMI'] * data['SkinThickness']



data['N8'] =  data['Pregnancies'] / data['Age']



data['N13'] = data['Glucose'] / data['DiabetesPedigreeFunction']



data['N12'] = data['Age'] * data['DiabetesPedigreeFunction']



data['N14'] = data['Age'] / data['Insulin']
target_name = 'Outcome'

train = data.drop([target_name], axis=1)

train.head()
target = data.Outcome

target.head()
from sklearn.model_selection import train_test_split



np.random.seed(0)

train, test_train, target, test_target = train_test_split(train, target, test_size=0.2)

print(train.info())

print(test_train.info())
(train == 0).sum(axis=0)
mean = np.mean(train)

std = np.std(train)



train = (train-mean)/(std+1e-7)

test_train = (test_train-mean)/(std+1e-7)
train, val_train, target, val_target = train_test_split(train, target, test_size=0.2)

print(train.info())

print(val_train.info())
from keras.models import Sequential

from keras.layers import Dense



model = Sequential()



model.add(Dense(32, input_dim=train.shape[1], activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.summary()
from keras import optimizers



opt = optimizers.Adam(lr=0.001)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
from keras.callbacks import EarlyStopping



es = EarlyStopping(monitor='val_loss', patience=20, mode='max')



history = model.fit(train, target, epochs=100, validation_data=(val_train, val_target), batch_size=10)
plt.plot(history.history['accuracy'], label='acc')

plt.plot(history.history['val_accuracy'], label='val_acc')

plt.ylim(0, 1)

plt.legend()
from sklearn import metrics



prediction = model.predict(train) > 0.5

prediction = (prediction > 0.5) * 1



accuracy_nn = round(metrics.accuracy_score(target, prediction) * 100, 2)
test_prediction = model.predict(test_train) > 0.5

test_prediction = (test_prediction > 0.5) * 1



test_accuracy_nn = round(metrics.accuracy_score(test_target, test_prediction) * 100, 2)

print(test_accuracy_nn)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()



logreg.fit(train, target)



accuracy_logreg = round(logreg.score(train, target) * 100, 2)

print(accuracy_logreg)
test_accuracy_logreg = round(logreg.score(test_train, test_target) * 100, 2)

print(test_accuracy_logreg)
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.tree import DecisionTreeRegressor



dec_tree_model = DecisionTreeRegressor(max_leaf_nodes=3000)



dec_tree_model.fit(train, target)



dec_tree_prediction = dec_tree_model.predict(train)

dec_tree_prediction = (dec_tree_prediction > 0.5) * 1

accuracy_dec_tree = round(metrics.accuracy_score(target, dec_tree_prediction) * 100, 2)

print(accuracy_dec_tree)
dec_tree_test_prediction = dec_tree_model.predict(test_train)

dec_tree_test_prediction = (dec_tree_test_prediction > 0.5) * 1

test_accuracy_dec_tree = round(metrics.accuracy_score(test_target, dec_tree_test_prediction) * 100, 2)

print(test_accuracy_dec_tree)
from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor()



rf_model.fit(train, target)



rf_prediction = rf_model.predict(train)

rf_prediction = (rf_prediction > 0.5) * 1

accuracy_rf = round(metrics.accuracy_score(target, rf_prediction) * 100, 2)

print(accuracy_rf)
rf_test_prediction = rf_model.predict(test_train)

rf_test_prediction = (rf_test_prediction > 0.5) * 1

test_accuracy_rf = round(metrics.accuracy_score(test_target, rf_test_prediction) * 100, 2)

print(test_accuracy_rf)
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier(n_neighbors=100)   # Tried some values in the range of 0-300, 90-100 seems to work well



knn_model.fit(train, target)



knn_test_prediction = knn_model.predict(test_train)   # Doesn't need to be turned into 1s and 0s because KNN already is a classifier

test_accuracy_knn = round(metrics.accuracy_score(test_target, knn_test_prediction) * 100, 2)

print(test_accuracy_knn)
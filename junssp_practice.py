import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from sklearn import preprocessing
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
os.getcwd()
raw_data = pd.read_csv("../input/my-dataset/credit_train.csv")
raw_data.info()
raw_data.describe()
data = raw_data.drop(['Loan ID', 'Customer ID'], axis=1)
# Loan Status dropna

data.dropna(subset=['Loan Status'], inplace = True)
data.info()
# Credit Score 850 이상 데이터 처리 

data['Credit Score'][data['Credit Score'] > 850] = data['Credit Score'][data['Credit Score'] > 850] / 10
data['Credit Score'].fillna(0, inplace = True)
data['Annual Income'].fillna(0, inplace = True)
data['Monthly Debt'].fillna(0, inplace = True)
data['Number of Open Accounts'].fillna(0, inplace = True)
data['Current Credit Balance'].fillna(0, inplace = True)
data['Maximum Open Credit'].fillna(0, inplace = True)

data['Number of Credit Problems'].fillna(0, inplace = True)
data['Bankruptcies'].fillna(0, inplace = True)
data['Tax Liens'].fillna(0, inplace = True)
data['Years in current job'].fillna(0, inplace = True)
data['Months since last delinquent'].max()
data['Months since last delinquent'] = data['Months since last delinquent'] - 177
data['Months since last delinquent'].fillna(0, inplace = True)
plt.figure(10*10)
for i in range(len(data.columns)):
  try:
    sns.distplot(data.loc[data['Loan Status']=='Fully Paid', data.columns[i]], label='Fully Paid')
    sns.distplot(data.loc[data['Loan Status']=='Charged Off', data.columns[i]], label = 'Charged Off')
    plt.legend(loc='best')
    plt.show()

  except:
    pass
data.info()
data.describe()
# labeling
data['Loan Status'] = data['Loan Status'].replace(['Fully Paid', 'Charged Off'], [1, 0])
data['Loan Status'].unique()
data['Term'].replace(("Short Term","Long Term"),(0,1), inplace=True)
data.head()
# data = data.join(pd.get_dummies(data['Purpose'],drop_first = True))
data = data.drop(['Purpose'], axis=1)
data = data.join(pd.get_dummies(data['Home Ownership'],drop_first = True))
data = data.drop(['Home Ownership'], axis=1)
# Years in current job
data['Years in current job'] = data['Years in current job'].str.lower()
data['Years in current job'] = data['Years in current job'].str.extract(r"(\d+)")
data['Years in current job'] = data['Years in current job'].astype(float)
data['Years in current job'].fillna(0, inplace = True)
data

data.columns
#독립변수(Purpose 분류)
debt_invest= ['renewable_energy', 'small_business', 'Business Loan']
debt_consume = ['Buy House', 'Take a Trip', 'Home Improvements', 'Buy a Car', 'Medical Bills', 'wedding', 'major_purchase', 'vacation', 'Educational Expenses', 'moving', 'Other', 'other']
debt_change = ['Debt Consolidation']
data
# con_short = list()
# con_long = list()
# cha_short = list()
# cha_long = list()
# inv_short = list()
# inv_long = list()

# # 목적별 기간별 분류
# for index, row in data.iterrows():

#   if row['Purpose'] in debt_change:
#     if row['Term'] == 0:
#       cha_short.append(row.values)
#     elif row['Term'] == 1:
#       cha_long.append(row.values)

#   elif row['Purpose'] in debt_consume:
#     if row['Term'] == 0:
#       con_short.append(row.values)
#     elif row['Term'] == 1:
#       con_long.append(row.values)

#   elif row['Purpose'] in debt_invest:
#     if row['Term'] == 0:
#       inv_short.append(row.values)
#     elif row['Term'] == 1:
#       inv_long.append(row.values)
# all_list = [con_short,
#             con_long,
#             cha_short,
#             cha_long,
#             inv_short,
#             inv_long]
# n = 1
# for li in all_list:
#   globals()["df_{}".format(n)] = pd.DataFrame(li, columns=data.columns)

#   # Home Ownership
#   globals()["df_{}".format(n)]['Home Ownership'] = globals()["df_{}".format(n)]['Home Ownership'].str.lower()
#   globals()["df_{}".format(n)]['Home Ownership'] = globals()["df_{}".format(n)]['Home Ownership'].replace([val for val in globals()["df_{}".format(n)]['Home Ownership'].unique()],
#                                                                                                           [num for num in range(len(globals()["df_{}".format(n)]['Home Ownership'].unique()))])

#   # Purpose
#   globals()["df_{}".format(n)]['Purpose'] = globals()["df_{}".format(n)]['Purpose'].str.lower()
#   globals()["df_{}".format(n)]['Purpose'] = globals()["df_{}".format(n)]['Purpose'].replace([val for val in globals()["df_{}".format(n)]['Purpose'].unique()],
#                                                                                             [num for num in range(len(globals()["df_{}".format(n)]['Purpose'].unique()))])
  
#   # Years in current job
#   globals()["df_{}".format(n)]['Years in current job'] = globals()["df_{}".format(n)]['Years in current job'].str.lower()
#   globals()["df_{}".format(n)]['Years in current job'] = globals()["df_{}".format(n)]['Years in current job'].str.extract(r"(\d+)")
#   globals()["df_{}".format(n)]['Years in current job'] = globals()["df_{}".format(n)]['Years in current job'].astype(float)
#   globals()["df_{}".format(n)]['Years in current job'].fillna(globals()["df_{}".format(n)]['Years in current job'].mean(), inplace = True)

  
#   n += 1
# from sklearn.preprocessing import MinMaxScaler
# mmscaler = MinMaxScaler()

# for i in range(1,7):
#   globals()["df_{}".format(i)]
#   globals()["x_{}".format(i)] = globals()["df_{}".format(i)].drop('Loan Status', axis=1)
#   globals()["y_{}".format(i)] = globals()["df_{}".format(i)]['Loan Status']

#   mmscaler.fit(globals()["x_{}".format(i)])
#   globals()["mmScaled_{}".format(i)] =pd.DataFrame(mmscaler.transform(globals()["x_{}".format(i)]), columns= globals()["x_{}".format(i)].columns)


# from keras.models import Sequential
# from keras.layers import Activation, Dense
# from keras import optimizers
# from sklearn.model_selection import train_test_split
# for i in range(1, 7):
#   (x_train, x_test, y_train, y_test) = train_test_split(globals()["mmScaled_{}".format(i)], globals()["y_{}".format(i)], train_size=0.8, random_state=1)
#   globals()['model_{}'.format(i)] = Sequential()
#   globals()['model_{}'.format(i)].add(Dense(128, input_shape = (x_train.shape[1],), activation = 'relu'))
#   globals()['model_{}'.format(i)].add(Dense(1, activation = 'sigmoid'))
#   globals()['model_{}'.format(i)].compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
#   globals()['model_{}'.format(i)].fit(x_train, y_train, epochs=10, verbose=1, batch_size=32)
#   print("model Score: ",globals()['model_{}'.format(i)].evaluate(x_test, y_test, batch_size=32))
ke = data.dropna()
data
# ke = ke.drop(['Years in current job', 'Home Ownership', 'Purpose'], axis=1)
ke['Term'] = ke['Term'].replace([i for i in ke['Term'].unique()], [i for i in range(len(ke['Term'].unique()))])
ke['Loan Status'] = ke['Loan Status'].replace([i for i in ke['Loan Status'].unique()], [1,0])
ke
x = ke.drop(['Loan Status'], axis=1)
y = ke['Loan Status']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1 = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

(x_train, x_test, y_train, y_test) = train_test_split(x1, y, train_size=0.8, random_state=1)
log = LogisticRegression()
log.fit(x_train, y_train)
log.score(x_test, y_test)

from sklearn.metrics import confusion_matrix
pre = log.predict(x_test)
confusion_matrix(y_true=y_test,y_pred=pre)
from sklearn.ensemble import GradientBoostingClassifier

gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(x_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(x_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(x_test, y_test)))
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics    
forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
print(y_pred)
# 정확도 확인
print('정확도 :', metrics.accuracy_score(y_test, y_pred))
x_train.shape
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Our vectorized labels
# y_train = np.asarray(train_labels).astype('float32').reshape((-1,1))
# y_test = np.asarray(test_labels).astype('float32').reshape((-1,1))

(x_train, x_test, y_train, y_test) = train_test_split(x1, y, train_size=0.8, random_state=1)

model = tf.keras.Sequential()

model.add(layers.Input(shape=x_train.shape[1]))
model.add(layers.Dense(128, activation='sigmoid')) 
model.add(layers.Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=30, verbose=1, batch_size=30)
history.history.keys() # 출력 가능 값 확인
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])

plt.plot(history.history['binary_accuracy'])
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
predictions = model.predict(x_test)
predictions
y_pred = np.argmax(predictions, axis=1)
for i, j in enumerate(predictions):
    if j > 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0
from sklearn.metrics import confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(y_test, predictions))
predictions.shape
predictions
predictions = predictions.reshape(1,7285)
ser_test = pd.Series(predictions[0])
ser_test
import seaborn as sns
sns.kdeplot(ser_test, cumulative=True, bw=1.5)
predictions
predictions

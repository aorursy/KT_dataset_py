# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# READING THE INPUT CSV FILE
hotel_review = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
hotel_review.head()
hotel_review.tail()
#Getting information regarding the data types and number of missing values in the dataset
hotel_review.info()
#seperating numerical and categorical columns
categorical_columns=[]
numerical_columns=[]
for col in hotel_review.columns:
    if hotel_review[col].dtype!='object':
        numerical_columns.append(col)
    else:
        categorical_columns.append(col)
hotel_review.describe() # for numerical values
hotel_review[categorical_columns].describe()# Statistical relations for categorical values
#CHECKING FOR MISSING VALUE
#As we saw earlier in the info method number of missing values in few of columns.
#Finding missing values in all columns
hotel_review.isna().sum()
#Getting a closer look on the 3 parameter having Missing values
#Checking for corelation in missing data columns
check_for_corelation = hotel_review[['is_canceled','agent','company']]
check_for_corelation.corr()
# dropping Company column
hotel_review.drop(columns=['agent', 'company'],inplace=True)
hotel_review.dropna(axis=0,inplace=True)
hotel_review.shape
# removing the empty observation for country column
hotel_review.country.dropna()
hotel_review.country.isna().sum()
# Lets copy data to check the correlation between variables. 
from sklearn.preprocessing import LabelEncoder, StandardScaler
corelation_of_data = hotel_review.copy()
le = LabelEncoder()
# for variables and thier correlation with other variables.
corelation_of_data['meal'] = le.fit_transform(corelation_of_data['meal'])
corelation_of_data['distribution_channel'] = le.fit_transform(corelation_of_data['distribution_channel'])
corelation_of_data['reserved_room_type'] = le.fit_transform(corelation_of_data['reserved_room_type'])
corelation_of_data['assigned_room_type'] = le.fit_transform(corelation_of_data['assigned_room_type'])
corelation_of_data['customer_type'] = le.fit_transform(corelation_of_data['customer_type'])
corelation_of_data['reservation_status'] = le.fit_transform(corelation_of_data['reservation_status'])
corelation_of_data['market_segment'] = le.fit_transform(corelation_of_data['market_segment'])
corelation_of_data['deposit_type'] = le.fit_transform(corelation_of_data['deposit_type'])
corelation_of_data['reservation_status_date'] = le.fit_transform(corelation_of_data['deposit_type'])
corelation_of_data['is_canceled'] = le.fit_transform(corelation_of_data['deposit_type'])
plt.figure(figsize=(20,10))
sns.heatmap(corelation_of_data.corr(),annot=True,cmap='viridis')
corelation_of_data.corr().is_canceled.sort_values(ascending = False)
#graphical potray of the correlation values
corelation_of_data.corr()['is_canceled'][:-1].sort_values().plot(kind='bar')
#Having a closer look at the type of values inside different attributes
hotel_review.reservation_status.unique()
hotel_review.customer_type.unique()
hotel_review.customer_type.value_counts()
plt.figure(figsize=(12,8))
plt.title(label='Cancellation by ADR & Hotel Type')
sns.barplot(x='hotel',y='adr',hue='is_canceled',data=hotel_review)
plt.show()
plt.figure(figsize=(12,8))
plt.title(label='Cancellation by Market Segments')
plt.xticks(rotation=45) 
sns.countplot(x='market_segment',hue='is_canceled',data=hotel_review)
plt.show()
hotel_review.arrival_date_month.value_counts()
plt.figure(figsize=(12,8))
sns.barplot(data = hotel_review, x= 'arrival_date_month',y='adr',hue='hotel')
most_occupied_month_price = hotel_review.groupby(['arrival_date_month','hotel']).sum().adr
most_occupied_month_price
# next we can look for the number of people and diffenrt variates of people come in
# combining the adults and children into one category as the expense is relatively the same and excluidng the babies
hotel_review['Family'] = hotel_review.adults + hotel_review.children 
# droping the existing columns
hotel_review.drop(columns=['adults','children','babies'],inplace=True)
hotel_review['Family'] = hotel_review['Family'].astype(int)
# now checking for which type of Hotel have more number of cancelations
# % of cancellations in City Hotel
hotel_review[hotel_review['hotel']=='City Hotel']['is_canceled'].value_counts(normalize=True)
# cancelation with respect to time
plt.figure(figsize=(12,8))
plt.title(label='Cancellation by Lead Time')
sns.barplot(x='hotel',y='lead_time',hue='is_canceled',data=hotel_review)
plt.show()
# converting hotel and months into numerical value and mapping them
hotel_review['hotel'] = hotel_review['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
hotel_review['arrival_date_month'] = hotel_review['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
hotel_review.country.nunique()
hotel_review.Family.value_counts()
hotel_review.deposit_type.value_counts()
#As discussed earlier due to high correlation with these factors we will highly inaccurate results therefore we drop these columns
hotel_review.columns
hotel_review.drop(columns="reservation_status_date", inplace=True, axis=1)
hotel_review.reservation_status.value_counts()
hotel_review.drop(columns=['reservation_status'], inplace=True, axis=1)
hotel_review['country'] = le.fit_transform(hotel_review['country'])
hotel_review['deposit_type'] = le.fit_transform(hotel_review['deposit_type'])
hotel_review['adr'] = le.fit_transform(hotel_review['adr'])
hotel_review['market_segment'] = le.fit_transform(hotel_review['market_segment'])
hotel_review['meal'] = le.fit_transform(hotel_review['meal'])
hotel_review['distribution_channel'] = le.fit_transform(hotel_review['distribution_channel'])
hotel_review['reserved_room_type'] = le.fit_transform(hotel_review['reserved_room_type'])
hotel_review['assigned_room_type'] = le.fit_transform(hotel_review['assigned_room_type'])
hotel_review['customer_type'] = le.fit_transform(hotel_review['customer_type'])
hotel_review.shape
# APPLYING MACHINE LEARNING MODELS
import statsmodels.formula.api as smf

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from warnings import filterwarnings
filterwarnings('ignore')
y = hotel_review["is_canceled"]
X = hotel_review.drop(["is_canceled"], axis=1)

# SPLITTING THE DATA INTO 30 PERCENT TEST AND 70 PERCENT TRAINING DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
tree = DecisionTreeClassifier(max_depth = 10)
tree_model = tree.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
print('Decision Tree Model')

print('Accuracy Score: {}\n\nConfusion Matrix:\n {}'
      .format(accuracy_score(y_test,y_pred), confusion_matrix(y_test,y_pred)))
# APPLYING RANDOM FORREST
rf_model = RandomForestClassifier(min_samples_leaf = 6, min_samples_split=6,
                                  n_estimators = 100)

# fitting of the model
estimator= rf_model.fit(X_train, y_train)
#Prediction of the Model
predict_rf = rf_model.predict(X_test)
RF_matrix = confusion_matrix(y_test, predict_rf)
RF_matrix = confusion_matrix(y_test, predict_rf)
ax = plt.plot()
sns.heatmap(RF_matrix,annot=True, fmt="d", cbar=False, cmap="Pastel2")
rf_model.feature_importances_
for name, importance in zip(X.columns, rf_model.feature_importances_):
    print(name, "=", importance)
#MODELLING WITH EXTREME GRADIENT BOOST
D_train = xgb.DMatrix(X_train, label=y_train)
D_test = xgb.DMatrix(X_test, label=y_test)
param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 

steps = 20  # The number of training iterations
model = xgb.train(param, D_train, steps)
preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
# first neural network with keras 
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(25,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=5, batch_size=1, verbose=1)
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test,verbose=1)

print(score)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split



#Reading the data

X_full = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing

import matplotlib.pyplot as plt
features = ['hotel', 'lead_time', 'arrival_date_year',

       'arrival_date_month','arrival_date_day_of_month', 'stays_in_weekend_nights',

       'stays_in_week_nights', 'adults', 'children','meal',

       'distribution_channel', 'deposit_type',

       'is_repeated_guest', 'previous_cancellations','booking_changes',

       'customer_type','adr',

        'total_of_special_requests','is_canceled']
def numeric_class(X,threshold):

    temp = []

    for i in X:

        if i <= threshold:

            temp.append(i)

        else:

            temp.append(threshold+1)

    return pd.DataFrame(temp)



def categoric_class(X,threshold):

    for i in X.index:

        if X[i] in threshold:

            continue

        else:

            X[i] = threshold[len(threshold)-1]

    return X

            
def data_cleaning(X,rows):

    #dropping the unneccesary columns 

    X_1 = X[rows]

    

    #Checking to see if the data has missing value and drop if it has

    X_1.dropna(axis=0,inplace = True)

    X_1.drop(X_1[X_1['meal'] == 'Undefined'].index, inplace =True)

    X_1.drop(X_1[X_1['distribution_channel'] == 'Undefined'].index, inplace =True)

    X_1.drop(X_1[X_1.adults == 0].index, inplace = True)

        

    #Categorizing few columns

    X_2 = X_1.copy()

    X_2.deposit_type = categoric_class(X_2.deposit_type,['No Deposit','Deposit'])

    X_2.distribution_channel = categoric_class(X_2.distribution_channel,['Direct', 'TA/TO','Other'])

    X_2.customer_type = categoric_class(X_2.customer_type,['Transient', 'Transient-Party', 'Other'])

    X_2.children = numeric_class(X_2.children,0)

    X_2.stays_in_week_nights = numeric_class(X_2.stays_in_week_nights,3)

    X_2.previous_cancellations = numeric_class(X_2.previous_cancellations,0)

    X_2.booking_changes = numeric_class(X_2.booking_changes, 0)

    X_2.adults = numeric_class(X_2.adults,2)

    X_2.total_of_special_requests = numeric_class(X_2.total_of_special_requests,0)

    

    return X_2

    

    

    

    
X_clean = data_cleaning(X_full,features)





#Columns to be encoded

col_label_encode = ['arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','previous_cancellations','customer_type',

                    'booking_changes','is_repeated_guest','total_of_special_requests','meal','children','arrival_date_year',

                   'arrival_date_month','distribution_channel','hotel','deposit_type']



#Applying Label Encoding

enc = preprocessing.LabelEncoder()



label_data = X_clean.copy()

for col in col_label_encode:

    label_data[col] = enc.fit_transform(X_clean[col])



y = label_data.is_canceled

X_cleaned = label_data.iloc[:,:-1]



#Separating training and test data

X_train_full,X_test,y_train_full,y_test = train_test_split(X_cleaned,y,train_size = 0.9, test_size = 0.1, random_state = 0)



#Separating validation data 

X_train,X_validation,y_train,y_validation = train_test_split(X_train_full,y_train_full,train_size = 0.9, test_size = 0.1, random_state = 0)

std = preprocessing.StandardScaler()



X_train_processed = X_train.copy()

X_valid_processed = X_validation.copy()

X_test_processed = X_test.copy()

#standardizing the remaining columns



X_train_processed[['lead_time','adr']] = std.fit_transform(X_train[['lead_time','adr']])

X_valid_processed[['lead_time','adr']] = std.transform(X_validation[['lead_time','adr']])

X_test_processed[['lead_time','adr']] = std.transform(X_test[['lead_time','adr']])



        
pred
from sklearn.model_selection import GridSearchCV



model = MLPClassifier(hidden_layer_sizes = 100,batch_size = 150,early_stopping = True, validation_fraction = 0.1,learning_rate_init = 0.001 )

model.fit(X_train_processed,y_train)
from sklearn.metrics import accuracy_score

pred = model.predict(X_valid_processed)

acc = accuracy_score(pred,y_validation)

print(acc*100,'%')

pred_final = model.predict(X_test_processed)

print(accuracy_score(pred_final,y_test)*100,'%')
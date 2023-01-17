import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use('fivethirtyeight')
data = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
data.head()
data.info()
# get overview of numeric columns



data.describe().T
data['hotel'].value_counts()/len(data)*100
data['is_canceled'].value_counts()/len(data)*100  
data.isna().sum()/len(data)*100
# drop columns with too many missing values



data.drop(['agent', 'company'], axis=1, inplace=True)
# transform target variable into categorical



data['is_canceled'] = pd.Categorical(data['is_canceled'])
data['distribution_channel'].value_counts()/len(data)
print(len(data[(data['distribution_channel'] == 'TA/TO') | (data['distribution_channel'] == 'Direct') | (data['distribution_channel'] == 'Corporate')]))

print(len(data))
data = data[(data['distribution_channel'] == 'TA/TO') |

            (data['distribution_channel'] == 'Direct') |

            (data['distribution_channel'] == 'Corporate')] 
data['distribution_channel'].value_counts()
print(len(data))

print(len(data.dropna()))
data = data.dropna()
# Distribution of variable



data['reserved_room_type'].value_counts()
data = data[(data['reserved_room_type'] != 'P') & (data['reserved_room_type'] != 'L')]

data = data[(data['assigned_room_type'] != 'P') & (data['assigned_room_type'] != 'L')]
room_cancel=data.groupby('reserved_room_type')['is_canceled'].value_counts().unstack() 
room_cancel=data.groupby('reserved_room_type')['is_canceled'].value_counts().unstack()

room_cancel['total']=room_cancel[0]+room_cancel[1]

room_cancel['percentage_canceled']=round(room_cancel[1]/room_cancel["total"],2)

room_cancel
plt.figure(figsize=(10,5))

sns.countplot(x='reserved_room_type',hue="is_canceled",data=data,palette='viridis')
plt.figure(figsize=(4,4))

sns.countplot(x='hotel',hue="is_canceled",data=data,palette='viridis')
hotel_cancel=data.groupby('hotel')['is_canceled'].value_counts().unstack()

hotel_cancel['total']=hotel_cancel[0]+hotel_cancel[1]

hotel_cancel['percentage_canceled']=round(hotel_cancel[1]/hotel_cancel["total"],2)

hotel_cancel
# strong positive skew



data["lead_time"].plot.hist(alpha=0.5,bins=10)

data["lead_time"].describe()
data['far_in_advance'] = pd.Categorical(np.where(data['lead_time'] >= 160, 1, 0))

data['recent_booking'] = pd.Categorical(np.where(data['lead_time'] <= 14, 1, 0))

data = data.drop('lead_time', axis=1) # drop initial column
plt.figure(figsize=(15,5))

sns.countplot(x='arrival_date_month',data=data,hue='is_canceled',palette='viridis')
display(data["stays_in_weekend_nights"].value_counts())



plt.figure(figsize=(10,5))

sns.countplot(x='stays_in_weekend_nights',data=data,hue='is_canceled',palette='viridis')
display(data["stays_in_week_nights"].value_counts())



plt.figure(figsize=(10,5))

sns.countplot(x='stays_in_week_nights',data=data,hue='is_canceled',palette='viridis')
data = data[data.stays_in_week_nights <= 5]

data = data[data.stays_in_weekend_nights <= 2]
plt.figure(figsize = (10,5))

sns.countplot(x='adults',data=data,hue='is_canceled',palette='viridis')
data = data[data.adults <= 3]
plt.figure(figsize = (4,4))

sns.countplot(x='is_repeated_guest',data=data,hue='is_canceled',palette='viridis')
data['is_repeated_guest'].value_counts()/len(data)*100
data = data.drop('is_repeated_guest', axis=1)
data["booking_changes"].describe() 
pd.Categorical(data['booking_changes']).value_counts()
data['changed_booking'] = pd.Series([0 if x == 0 else 1 for x in data['booking_changes']])



# make it categorical

data['changed_booking'] = pd.Categorical(data['changed_booking'])
data["children"].value_counts()
plt.figure(figsize = (10,5))

sns.countplot(x='children',data=data,hue='is_canceled',palette='viridis')
data = data[data.children <= 2]
data["customer_type"].value_counts()
plt.figure(figsize = (10,4))

sns.countplot(x='customer_type',data=data,hue='is_canceled',palette='viridis')
data["deposit_type"].value_counts()
plt.figure(figsize = (6,4))

sns.countplot(x='deposit_type',data=data,hue='is_canceled',palette='viridis')
data[data["deposit_type"]=="Non Refund"]["is_canceled"].value_counts()
data = data[data.deposit_type != 'Refundable']
data["meal"].value_counts()
plt.figure(figsize=(10,4))

sns.countplot(x='meal',data=data,hue='is_canceled',palette='viridis')
data["previous_cancellations"].value_counts()
plt.figure(figsize=(10,4))

sns.countplot(x='previous_cancellations',

              data=data[data["previous_cancellations"]>=1],

              hue='is_canceled',

              palette='viridis')
data['previous_cancel'] = pd.Series([0 if x == 0 else 1 for x in data['previous_cancellations']]) # binary for previous_cancelations

data['previous_cancel'] = pd.Categorical(data['previous_cancel'])
sum(data["reserved_room_type"]==data["assigned_room_type"])
sum(data["reserved_room_type"]!=data["assigned_room_type"])
data[data["reserved_room_type"]!=data["assigned_room_type"]]["is_canceled"].value_counts()
# same room type assigned



bol = data['assigned_room_type'] == data['reserved_room_type']

data['right_room'] = bol.astype(int)

data['right_room'] = pd.Categorical(data['right_room'])
data["adr"].describe()
sns.distplot(data["adr"], kde = False)
data = data[data.adr >=0 ]

data = data[data.adr <= 300]
sns.distplot(data["adr"], kde = False)
data["total_of_special_requests"].value_counts()
plt.figure(figsize = (10,4))

sns.countplot(x='total_of_special_requests',

              data=data[data["total_of_special_requests"]>=1],hue='is_canceled',palette='viridis')

data['special_requests'] = pd.Series([0 if x == 0 else 1 for x in data['total_of_special_requests']]) 

# binary for special requests

data['special_requests'] = pd.Categorical(data['special_requests'])

data["reservation_status"].value_counts()
data["is_canceled"].value_counts()
data["required_car_parking_spaces"].value_counts()
plt.figure(figsize = (10,4))

sns.countplot(x='required_car_parking_spaces',

              data=data,hue='is_canceled',palette='viridis')
data['required_car_parking_spaces'] = pd.Series([0 if x == 0 else 1 for x in data['required_car_parking_spaces']]) 

# binary for parking spots

data['required_car_parking_spaces'] = pd.Categorical(data['required_car_parking_spaces'])

data = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
# encode dependent variable for classification

data['is_canceled'] = pd.Categorical(data['is_canceled']) 



#------------------ Clean the data and transform ---------------------------#



# cut under represented factor levels or split into binary



## Distribution channel



# TA/TO        0.819750

# Direct       0.122665

# Corporate    0.055926

# GDS          0.001617

# Undefined    0.000042



data = data[(data['distribution_channel'] == 'TA/TO') |

            (data['distribution_channel'] == 'Direct') |

            (data['distribution_channel'] == 'Corporate')] 



#---------------------------------#

## Rooms



# A    85446

# D    19161

# E     6470

# F     2890

# G     2083

# B     1114

# C      931

# H      601

# L        6

# P        2



# we clean room types P and L

data = data[(data['reserved_room_type'] != 'P') & (data['reserved_room_type'] != 'L')]

data = data[(data['assigned_room_type'] != 'P') & (data['assigned_room_type'] != 'L')]





#---------------------------------#

## Duration of stay & Kids



# cleaned for now: or create binary with long / short stays

data = data[data.stays_in_weekend_nights <= 2]

data = data[data.stays_in_week_nights <= 5]



# all very unequal distributed

data = data[data.children <= 2]

data = data[data.adults <= 3]



#---------------------------------#

## Deposit



# No Deposit    104641

# Non Refund     14587

# Refundable       162



# also cannot logical combine with other levels

data = data[data.deposit_type != 'Refundable']



#---------------------------------#



# agent                             13.686238 maybe we can keep agent -> only 13 % missing

# company                           94.306893





data = data[data.adr >=0 ]

data = data[data.adr <= 300]



#------------------------Creating new variables-----------------------------------------#





# correct room type assigned

bol = data['assigned_room_type'] == data['reserved_room_type']

data['right_room'] = bol.astype(int)

data['right_room'] = pd.Categorical(data['right_room'])



# lead time

# lead time severly skewed -> binary and we will drop initial variable

data['far_in_advance'] = pd.Categorical(np.where(data['lead_time'] >= 160, 1, 0))

data['recent_booking'] = pd.Categorical(np.where(data['lead_time'] <= 14, 1, 0))





data['changed_booking'] = pd.Series([0 if x == 0 else 1 for x in data['booking_changes']])# binary for changed booking at least once

data['changed_booking'] = pd.Categorical(data['changed_booking'])



data['previous_cancel'] = pd.Series([0 if x == 0 else 1 for x in data['previous_cancellations']]) # binary for previous_cancelations

data['previous_cancel'] = pd.Categorical(data['previous_cancel'])



data['special_requests'] = pd.Series([0 if x == 0 else 1 for x in data['total_of_special_requests']]) # binary for special requests

data['special_requests'] = pd.Categorical(data['special_requests'])



data['required_car_parking_spaces'] = pd.Series([0 if x == 0 else 1 for x in data['required_car_parking_spaces']]) # binary for parking spots

data['required_car_parking_spaces'] = pd.Categorical(data['required_car_parking_spaces'])





#--------- drop initial variables for those where we create binaries------------



data = data.drop(['total_of_special_requests',

                   'reservation_status',

                   'previous_cancellations',

                   'booking_changes',

                   'lead_time',

                  'babies'

                    ], axis=1)



# --------- drop columns which we don't want to use --------------



data = data.drop(['company','agent','is_repeated_guest',

                  'reservation_status_date', 'previous_bookings_not_canceled'

                 ],axis=1)



# ---------- drop remaining NAs from the data---------

data=data.dropna()
from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten, Dropout

from keras.layers.embeddings import Embedding

from keras.layers import LSTM, Bidirectional

from keras import optimizers

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping

from keras.layers import Layer



from collections import Counter



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics



from scipy.stats import zscore
# drop unused columns

data = data.drop(['arrival_date_week_number', 'arrival_date_day_of_month'], axis=1 )
def encode_columns(column, data):

    

    data = pd.concat([data,pd.get_dummies(data[column],prefix=column)],axis=1)

    data.drop(column, axis=1, inplace=True)

    

    return data
### ------------- encode categorical columns ----------------



categorical_columns = ["required_car_parking_spaces",

                       "right_room",

                       "far_in_advance",

                       "recent_booking",

                       "changed_booking",

                       "previous_cancel",

                       "special_requests",

    

    

                       "hotel", 

                       "arrival_date_year",

                       "arrival_date_month",

                       "meal",

                       "country",

                       "market_segment",

                       "distribution_channel",

                       "deposit_type",

                       "customer_type",

                       "reserved_room_type",

                       "assigned_room_type"

                      ]

    

for col in categorical_columns:

    data=encode_columns(col,data)
data['adr'] = zscore(data['adr'])
data = data.dropna()
x = data.drop('is_canceled', axis=1)

y = data['is_canceled']
x = np.asarray(x)

y = np.asarray(y)
# Split into train/test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)



model = Sequential()

model.add(Dense(100, input_dim=x.shape[1], activation='relu', kernel_initializer='random_normal'))

model.add(Dropout(0.5))

model.add(Dense(50,activation='relu',kernel_initializer='random_normal'))

model.add(Dropout(0.2))

model.add(Dense(25,activation='relu',kernel_initializer='random_normal'))

model.add(Dense(1,activation='sigmoid', kernel_initializer='random_normal'))





# compile the model

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())





monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=12, 

                        verbose=1, mode='auto', restore_best_weights=True)





history = model.fit(x_train, y_train, validation_split=0.2, callbacks=[monitor], verbose=1, epochs=100)



loss, accuracy = model.evaluate(x_test, y_test, verbose=1)

print('Accuracy: %f' % (accuracy*100))

print('\n')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model train vs validation loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper right')

plt.show()
from sklearn.metrics import roc_curve, auc





# Plot an ROC. pred - the predictions, y - the expected output.

def plot_roc(pred,y):

    fpr, tpr, _ = roc_curve(y, pred)

    roc_auc = auc(fpr, tpr)



    plt.figure(figsize=(7,7))

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC)')

    plt.legend(loc="lower right")

    plt.show()
prediction_proba = model.predict(x_test)
plot_roc(prediction_proba,y_test)
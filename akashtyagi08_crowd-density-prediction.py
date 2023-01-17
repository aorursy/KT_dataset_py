import numpy as np

import pandas as pd

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/trainn/Train.csv")

train.head()
# Checking rows and columns of dataset

train.shape
missing_data = pd.DataFrame({'total_missing': train.isnull().sum(), 'perc_missing': (train.isnull().sum()/82790)*100})

print(missing_data)
train[train['latitude_destination'].isnull()].index
train[0:30][['country_code_destination','longitude_destination','latitude_destination','mean_halt_times_destination']]
train[0:40]
train.drop(train[train['latitude_destination'].isnull()].index,inplace=True)
# Checking again for missing values

missing_data = pd.DataFrame({'total_missing': train.isnull().sum(), 'perc_missing': (train.isnull().sum()/82790)*100})

print(missing_data)
train[train['latitude_source'].isnull()]
train.drop(train[train['latitude_source'].isnull()].index,inplace=True)
# Final check for missing values

missing_data = pd.DataFrame({'total_missing': train.isnull().sum(), 'perc_missing': (train.isnull().sum()/82790)*100})

print(missing_data)
# Lets see how many rows of data are we left with now.

print("Dataset rows: ",train.shape[0])
train = train.iloc[:,1:]

train.head()
def convert24(str1): 

    # Checking if last two elements of time 

    # is AM and first two elements are 12 

    if str1[-2:] == "AM" and str1[:2] == "12": 

        return "00" + str1[2:-2] 

          

    # remove the AM     

    elif str1[-2:] == "AM": 

        return str1[:-2] 

      

    # Checking if last two elements of time 

    # is PM and first two elements are 12    

    elif str1[-2:] == "PM" and str1[:2] == "12": 

        return str1[:-2] 

          

    else: 

        # add 12 to hours and remove PM 

        return str(int(str1[:2]) + 12) + str1[2:8] 
import math    

def deg2rad(deg):

    return deg * (math.pi/180)



def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):

    R = 6371 # Radius of the earth in km

    dLat = deg2rad(lat2-lat1) # deg2rad below

    dLon = deg2rad(lon2-lon1) 

    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)

     

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) 

    d = R * c #Distance in km

    return d
# Finding distance from Latitude, Longitude.

train['distance'] = [0]*train.shape[0]

# Converting Lat,Log to Distance

for x in train['source_name'].index:   

        train['distance'][x] = round(getDistanceFromLatLonInKm(train['latitude_source'][x],

             train['longitude_source'][x],train['latitude_destination'][x],train['longitude_destination'][x]))



train.drop(['latitude_source','longitude_source','latitude_destination','longitude_destination'],axis=1,inplace=True)

train.head()
plt.plot(train['distance'])
train.drop(train[train['distance']>700].index,inplace=True)

train.drop(train[train['distance'] == 0].index,inplace=True)     # We also have trains where distance=0, remove them
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelX = LabelEncoder()

onehotX = OneHotEncoder(sparse=False)



numeric_cols = train._get_numeric_data().columns

print(numeric_cols)
print("List of columns: ",train.columns)
print("Total categorical columns: ",len(train.columns)-len(numeric_cols))

print("Columns: ")

for x in train.columns:

    if x not in numeric_cols:

        print("\t",x)
train[['source_name','destination_name']].head()
print("Unique Source Name: ",len(train['source_name'].unique()))

print("Unique Destination Name: ",len(train['destination_name'].unique()))
import re

def station_number(name):

    return ''.join(re.findall(r'[\d]',name))



train['source_name'] = pd.to_numeric(train['source_name'].apply(station_number)) 

train['destination_name'] = pd.to_numeric(train['destination_name'].apply(station_number))
sns.distplot(train['source_name'])

sns.distplot(train['destination_name'])
train[['current_date']].head()
# We will extract the date and month into a seperate column.

train['month'] = pd.to_numeric(train.current_date.str[5:7])

train['date']= pd.to_numeric(train.current_date.str[8:])

print(train['month'].head())

train.drop(['current_date'],axis=1,inplace=True) # As we have breaked date into month and date 
current_month = train.groupby(['month'])['target'].sum()

print(current_month)

plt.plot(train['month'],train['target'])
# Converting time into 24hr format

for x in train['current_time'].index:

        train['current_time'][x] = convert24(train['current_time'][x])[:5]

        

print(train['current_time'].head())
 # Converting hour values to sin, cos value to achieve better relation between them.

train['hourfloat']=pd.to_numeric(train.current_time.str[:2])+pd.to_numeric(train.current_time.str[3:5])/60.0

train['hourx']=np.sin(2.*np.pi*train.hourfloat/24.)

train['houry']=np.cos(2.*np.pi*train.hourfloat/24.)

train.drop(['current_time'],axis=1,inplace=True)

print(train['hourx'].head())
''' Lets count which trains were most active in our dataset'''

print("Total trains: ",len(train['train_name'].unique()))

train_name = pd.DataFrame({'count':train.groupby(['train_name']).count().iloc[:, 1]})

morethan5trains = train_name[train_name['count']>5]

print(morethan5trains) # Trains with more than 5 entries

print(f"%d trains has more than 5 entries"%(len(morethan5trains)))
morethan5trains_name = morethan5trains.index

#     Replacing train name using frequnent trains .

train['frequent_train'] = 0*train.shape[0]



# Converting train-name

xe = []

for x in morethan5trains_name:

    xe.extend(train[train['train_name']==x].index)



for x in xe:

    train['frequent_train'][x]=1

train.drop(['train_name'],axis=1,inplace=True)

print(train['frequent_train'].head())

train['is_weekend'].head()
train['is_weekend'] = labelX.fit_transform(train['is_weekend']) # LabelEncode weekend

train['is_weekend'].head()
numeric_cols = train._get_numeric_data().columns

print("Total categorical columns: ",len(train.columns)-len(numeric_cols))

print("Categorical Columns remaining: ")

for x in train.columns:

    if x not in numeric_cols:

        print("\t",x)
print(train['current_day'].head())
onehot_days = pd.get_dummies(train.current_day)

train = train.join(onehot_days)
print(train.columns)
train.drop(['current_day','current_year'],axis=1,inplace=True) # Current year is same through out the year.
print(train['country_code_source'].unique())

print(train['country_code_destination'].unique())
onehot_sourcecode = pd.get_dummies(train.country_code_source)

train = train.join(onehot_sourcecode)

train.head()
train.drop(['country_code_source','country_code_destination'],axis=1,inplace=True) 
numeric_cols = train._get_numeric_data().columns

print("Total categorical columns: ",len(train.columns)-len(numeric_cols))

print("Categorical Columns remaining: ")

for x in train.columns:

    if x not in numeric_cols:

        print("\t",x)
train.head()
from sklearn.linear_model import SGDClassifier,SGDRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.calibration import CalibratedClassifierCV

#import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score
y = train['target']

target_label = LabelEncoder() 

y = target_label.fit_transform(y)

train.drop(['target'],axis=1,inplace=True)
Xtrain, Xtest, ytrain, ytest = train_test_split(train, y, test_size=0.25)
scalerX = StandardScaler()

scalerX.fit(Xtrain)

Xtrain = scalerX.transform(Xtrain)

Xtest = scalerX.transform(Xtest)    



from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees

model = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')
# Fit on training data

model.fit(Xtrain,ytrain)

rf_predictions = model.predict(Xtest)



plt.plot(ytest,color='g')

plt.plot(rf_predictions,color='r')



from sklearn import metrics

pscore = metrics.accuracy_score(ytest, rf_predictions)



feature_imp = pd.Series(model.feature_importances_,index=train.columns).sort_values(ascending=False)

feature_imp



sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()
print(Xtest[0],rf_predictions[0])
x_new_inverse = scalerX.inverse_transform(Xtest[0])

print(x_new_inverse)
model.score(Xtest, ytest)
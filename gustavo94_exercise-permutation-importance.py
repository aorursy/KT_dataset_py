# Loading data, dividing, modeling and EDA below
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)

data.head()
sns.boxplot(x="passenger_count",y="fare_amount",data=data[data["fare_amount"]<40])
sns.barplot(x="passenger_count",y="fare_amount",data=data)
sns.pairplot(data,hue="passenger_count")

# Remove data with extreme outlier coordinates or negative fares
data = data.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                  'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                  'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                  'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                  'fare_amount > 0'
                  )

y = data.fare_amount

base_features = ['pickup_longitude',
                 'pickup_latitude',
                 'dropoff_longitude',
                 'dropoff_latitude',
                 'passenger_count']

X = data[base_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_X, train_y)

# Environment Set-Up for feedback system.
import sys
sys.path.append('../input/ml-insights-tools')
from learntools.core import binder
binder.bind(globals())
from ex2 import *
print("Setup Complete")

# show data
print("Data sample:")
data.head()
train_X.describe()
train_y.describe()
q_1.solution()

# Import the libraries
import eli5
from eli5.sklearn import PermutationImportance

# Make a small change to the code below to use in this problem. 
perm = PermutationImportance(first_model, random_state=1).fit(val_X, val_y)

# uncomment the following line to check your results
q_2.check()

# uncomment the following line to visualize your results
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
q_2.solution()
q_3.solution()
# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
# Use a random_state of 1 for reproducible results that match the expected solution.
perm2 = PermutationImportance(second_model,random_state=1).fit(new_val_X,new_val_y) 

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm2, feature_names = new_val_X.columns.tolist())

# uncomment and run the following cell to check your answer
# q_4.check()
q_4.solution()
# create new features
data['abs_lon_change'] = abs(data.dropoff_longitude - data.pickup_longitude)*100
data['abs_lat_change'] = abs(data.dropoff_latitude - data.pickup_latitude)*100

features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = data[features_2]
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_X, new_train_y)

# Create a PermutationImportance object on second_model and fit it to new_val_X and new_val_y
# Use a random_state of 1 for reproducible results that match the expected solution.
perm2 = PermutationImportance(second_model,random_state=1).fit(new_val_X,new_val_y) 

# show the weights for the permutation importance you just calculated
eli5.show_weights(perm2, feature_names = new_val_X.columns.tolist())

q_5.solution()
q_6.solution()
dt = pd.read_csv('../input/hospital-readmissions/train.csv')
dt.shape
dt.info()
dt.columns
# Separate the target variable from the original dataset dt
y1 = dt.readmitted

# I've just keep some pf the features to practice 
base_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'race_Caucasian',
       'race_AfricanAmerican', 'gender_Female', 'age_[70-80)', 'age_[60-70)',
       'age_[50-60)', 'age_[80-90)', 'age_[40-50)',
       'medical_specialty_InternalMedicine',
       'medical_specialty_Emergency/Trauma',
       'medical_specialty_Family/GeneralPractice',
       'medical_specialty_Cardiology','change_No',
       'diabetesMed_Yes']

X1 = dt[base_features]

tr_X, test_X, tr_y, test_y = train_test_split(X1, y1, random_state=1)
rf_model = RandomForestRegressor(n_estimators=20, random_state=1).fit(tr_X, tr_y)
import eli5
from eli5.sklearn import PermutationImportance

# Make a small change to the code below to use in this problem. 
perm = PermutationImportance(rf_model, random_state=1).fit(test_X, test_y)

# uncomment the following line to visualize your results
eli5.show_weights(perm, feature_names = test_X.columns.tolist())
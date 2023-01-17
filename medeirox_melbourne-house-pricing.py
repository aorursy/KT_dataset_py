import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#path = '../input/melbourne-housing-market/Melbourne_housing_FULL.csv' 
# Not working for the above dataset at the moment. Not all the preprocessing needed have been done yet.

path = '../input/melbourne-housing-snapshot/melb_data.csv'
raw_data = pd.read_csv(path)
raw_data.head(5)
columns_of_interest = ['Rooms', 'Type', 'Distance', 'Bedroom2', 
                        'Bathroom', 'Car', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude', 'Regionname', 'Price']
df = raw_data.copy()
df.info()
df = df[columns_of_interest]
df.head()
column_names = ['Landsize', 'BuildingArea']
df_ls_ba = df[column_names]
df_ls_ba = df_ls_ba.dropna(axis = 0)
df_ls_ba.info()
landsize_buildingarea_ratio = df_ls_ba['Landsize'].median()/df_ls_ba['BuildingArea'].median()
landsize_buildingarea_ratio
df_buildingarea = df.copy()
def remove_nan_buildingarea(row):
    return row['Landsize']/landsize_buildingarea_ratio if \
                pd.isnull(row['BuildingArea']) else row['BuildingArea']
df_buildingarea['BuildingArea'] = df_buildingarea.apply(remove_nan_buildingarea, axis = 1)
df_buildingarea.head()
df_buildingarea.info()
for col in df.columns:
    print('Column: {}  |  Type: {}'.format(col, type(df[col][0])))

df_region = df_buildingarea.copy()
df_region['Regionname'].unique()
df_region['Regionname'].isnull().sum()
for row in df_region['Regionname']:
    row = 'Unknown' if row is None else row
df_region['Regionname'].isnull().sum()
def remove_nan_str(value):
    if isinstance(value, str):
        return value
    else:
        return 'Unknown' if pd.np.isnan(value) else value
df_region['Regionname'] = df_region['Regionname'].apply(remove_nan_str)
df_region['Regionname'].isnull().sum()
df_region['Regionname'].unique()
df_region['Regionname'].value_counts()
regionname_dummies = pd.get_dummies(df_region['Regionname'])
regionname_dummies.head()
#Removes the Unknown column and avoid multicollinearity

if regionname_dummies.columns.contains('Unknown'):
    regionname_dummies = regionname_dummies.drop(['Unknown'], axis = 1)
regionname_dummies.head()
df_region = pd.concat([df_region, regionname_dummies], axis = 1)
df_region = df_region.drop(['Regionname'], axis = 1)
df_region.head()
df['Type'].isnull().sum()
df['Type'].unique()
type_dummies = pd.get_dummies(df['Type'])
type_dummies
type_dummies.sum()
# Since 'h' is the most seen value, we'll remove it and consider as the default value, again
# avoiding multicollinearity
type_dummies = type_dummies.drop(['h'], axis = 1)
df_type = pd.concat([df_region, type_dummies], axis = 1)
df_type = df_type.drop(['Type'], axis = 1)
df_type.head()
df_YearBuilt = df_type.copy()
df_YearBuilt['YearBuilt'].median()
df_YearBuilt.info()
# Checking the possible ages of the data

print(df_YearBuilt['YearBuilt'].value_counts())
# Age Groups in Decades
# Up to:
# 0, 1, 2, 3, 5, 100, 9999 (Unknown)
import datetime

age_groups = {0, 1, 2, 3, 5, 10, 20, 100}

def divide_data_by_age_groups(year_built):
    if pd.isnull(year_built):
        return 9999
    age = datetime.datetime.now().year - year_built
    if (age % 10) >= 5:
        age_decades = ((age // 10) + 1)
    else:
        age_decades = (age // 10)

    for group in age_groups:
        if age_decades <= group:
            age_decades = group
            break
        
    #print('AGE {} | DECADES {}'.format(age, age_decades))
    return age_decades
df_YearBuilt['AgeInDecades'] = df_YearBuilt['YearBuilt'].apply(divide_data_by_age_groups)
df_YearBuilt.head(5)
decades_dummies = pd.get_dummies(df_YearBuilt['AgeInDecades'])
decades_dummies.head()
column_names = ['0_Decades_Old', '1_Decades_Old', '2_Decades_Old', '3_Decades_Old', '100_Decades_Old', 'Unknown_Decades_Old']
decades_dummies.columns = column_names
decades_dummies.head()
decades_dummies = decades_dummies.drop(['Unknown_Decades_Old'], axis = 1)
decades_dummies.head()
df_YearBuilt = df_YearBuilt.drop(['AgeInDecades', 'YearBuilt'], axis = 1)
df_YearBuilt.head()
df_YearBuilt = pd.concat([df_YearBuilt, decades_dummies], axis = 1)
df_YearBuilt.head()
df_Car = df_YearBuilt.copy()
df_Car['Car'].isnull().sum()
# The 62 null values must be filled, let's use the median
car_median = df_YearBuilt['Car'].median()

def remove_nan_car(value):
    if pd.isnull(value):
        return car_median
    else:
        return value
print(df_Car['Car'].isnull().sum())
df_Car['Car'] = df_Car['Car'].apply(remove_nan_car)
print(df_Car['Car'].isnull().sum())
df_Car.head()
df_Car.isnull().sum()
df_Car.columns.values
column_names = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
       'BuildingArea', 'Lattitude', 'Longtitude',
       'Eastern Metropolitan', 'Eastern Victoria',
       'Northern Metropolitan', 'Northern Victoria',
       'South-Eastern Metropolitan', 'Southern Metropolitan',
       'Western Metropolitan', 'Western Victoria', 't', 'u',
       '0_Decades_Old', '1_Decades_Old', '2_Decades_Old', '3_Decades_Old',
       '100_Decades_Old', 'Price']

df_Car = df_Car[column_names]
df_Car.head()
df_preprocessed = df_Car.copy()
df_preprocessed.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
raw_data = df_preprocessed.copy()

# Since BuildingArea has been filtered by a factor of the ratio of Landsize vs BuildingArea
# a nice check is to run the Kernel without this column, and check it's accuracy without it
raw_data = raw_data.drop('BuildingArea', axis = 1)

raw_data.head()
y = raw_data['Price']
raw_data.drop(['Price'], axis = 1)

train_X, test_X, train_y, test_y = train_test_split(df_preprocessed, y,train_size=0.8, 
                                                    test_size=0.2, 
                                                    random_state=0)
test_X.head(100)
model = RandomForestRegressor(random_state=900)
model.fit(train_X, train_y)
predictions = model.predict(test_X)
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=900)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
score_dataset(train_X,test_X, train_y, test_y)
def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()
get_mae(test_X, test_y)
## Plotting Predicted vs Real
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
diff = test_y - predictions
plt.plot(diff, 'rd')
plt.ylabel('Absolute Price Error ($)')
plt.show()
print(diff.max())
print(diff.min())
diff_percent = ((test_y - predictions)/test_y)*100
plt.plot(diff_percent, 'rd')
plt.ylabel('Relative Price Error (%)')
plt.show()
print("Minimum error: {}".format(diff_percent.min()))
print("Maximum error: {}".format(diff_percent.max()))
error_mean = diff_percent.mean()
print("Error Mean: {}".format(diff_percent.mean()))
accuracy = (1 - diff_percent.mean())*100
print("Accuracy: {}".format(accuracy))
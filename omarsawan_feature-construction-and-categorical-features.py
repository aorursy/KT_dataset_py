#import panda library to be able to load the dataset
import pandas as pd
#read the dataset
data=pd.read_csv('https://raw.githubusercontent.com/Omarsawan/Feature-construction-and-Categorical-features-tutorial/master/data/ks-projects-201801.csv',parse_dates=['deadline', 'launched'], encoding='latin-1')
#show the first 7 rows
data.head(7)
#make a data series with index equal to the column name and the value of the index is whether this column is of type object
indicesObjects=(data.dtypes=='object')
#make a data series with index equal to the column name and the value of the index is whether this column is of type datetime
indicesDate=(data.dtypes=='datetime64[ns]')

#make a data series with columns that have true only
objectColumns=indicesObjects[indicesObjects]
dateColumns=indicesDate[indicesDate]

#show only the index (which is the column name of categorical features)
objectsList=list(objectColumns.index)
datetimeList=list(dateColumns.index)

categoricalFeatures=(objectsList+datetimeList)

print('Columns with data type object: ',objectsList)
print('Columns with data type datetime: ',datetimeList)
print('Categorical features are',categoricalFeatures)
print('Count of Categorical features is',len(categoricalFeatures))
dropCategorical=data.select_dtypes(exclude='object').select_dtypes(exclude='datetime64[ns]')
dropCategorical.head(7)
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_data = data.copy()

# Apply label encoder to column 'name'
label_encoder = LabelEncoder()
label_data['name'] = label_encoder.fit_transform(label_data['name'])
label_data.head(7)
#lets count the number of unique values in each column
uniqueCount ={}
for col in categoricalFeatures:
    curCol=data.filter([col]).iloc[:,0]
    uniqueCount[col]=len(curCol.unique())
print(uniqueCount)
from sklearn.preprocessing import OneHotEncoder

cols=['state','currency']

# Make copy to avoid changing original data 
OH_data = data.copy()

# Apply one-hot encoder to the columns we have choosen
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(OH_data[cols]))

# One-hot encoding removed index; put it back
OH_cols.index = OH_data.index

# Remove categorical columns (will be replaced with one-hot encoding)
num_X = OH_data.drop(cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X = pd.concat([num_X, OH_cols], axis=1)

OH_X.head(7)
# Make copy to avoid changing original data 
data_copy = data.copy()

#add the four columns
data_copy=data_copy.assign(hour=data_copy.launched.dt.hour,
                           day=data_copy.launched.dt.day,
                           month=data_copy.launched.dt.month,
                           year=data_copy.launched.dt.year)
#remove the launched column
data_copy=data_copy.drop(['launched'], axis=1)

data_copy.head(7)
# Make copy to avoid changing original data 
data_copy = data.copy()
#make new feature
interactions = data['category'] + "_" + data['country']
#give this new column a name
interactions.name='category-country'
#add the column to the data
dataInteraction=pd.concat([interactions, data_copy], axis=1)

#remove the category and country columns
dataInteraction=dataInteraction.drop(['category','country'], axis=1)

dataInteraction.head(7)
# First, create a Series with a timestamp index and the values in the series are the original index of rows
# then sort it by the timestamp index
launched = pd.Series(data.index, index=data.launched, name="count_last_week").sort_index()
launched.head(20)
count_last_week = launched.rolling('7d').count() - 1
count_last_week.head(20)
#now that we have the counts, we need to adjust the index so we can join it with the other training data.
count_last_week.index = launched.values
count_last_week = count_last_week.reindex(data.index)
count_last_week.head(20)
#now join the new feature with the other data again using .join since we've matched the index.
data.join(count_last_week).head(10)
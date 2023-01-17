import pandas as pd

exercise_data = pd.read_csv( '../input/exercise.csv' )

calories_data = pd.read_csv( '../input/calories.csv' )
# join both CSV files using User_ID as key and left outer join (preserve exercise_data even if there are no calories)

joined_data = exercise_data.join( calories_data.set_index( 'User_ID' ), on='User_ID', how='left')

joined_data.head()
# in the scatter plot of duration vs calories and heart rate vs calories the relationship

# was curved upward (not linear)

# feature engineering:  add squared duration and heart rate to try a better fit with calories

joined_data = joined_data.assign( squared_duration = joined_data[ 'Duration' ] ** 2 )

joined_data = joined_data.assign( squared_heart_rate = lambda x: x[ 'Heart_Rate' ] ** 2 )



joined_data.head()
# since we don't want the prediction to be negative calories, 

# convert calories to natural logarithm to always get a positive number

import numpy as np

joined_data = joined_data.assign( log_Calories = lambda x: 

                                 np.log( x[ 'Calories' ] ) )

joined_data.head()
# scale numbers with normal distribution using z-score

from scipy.stats import zscore



joined_data = joined_data.assign( zscore_body_temp = zscore( joined_data[ 'Body_Temp' ] ) )

joined_data = joined_data.assign( zscore_height = zscore( joined_data[ 'Height' ] ) )

joined_data = joined_data.assign( zscore_weight = zscore( joined_data[ 'Weight' ] ) )

joined_data = joined_data.assign( zscore_squared_heart_rate = zscore( joined_data[ 'squared_heart_rate' ] ) )



joined_data.head()
# scale non-normal columns (age, squared_duration) using Min-Max 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



# NOTE:  joined_data[ ['Age', 'squared_duration'] ] produces a copy, loc doesn't

minMaxData = pd.DataFrame( scaler.fit_transform( joined_data.loc[ :, ['Age','squared_duration'] ] )

                         , columns = [ 'minMaxAge', 'minMaxSquaredDuration' ] )

joined_data = pd.concat( [ joined_data, minMaxData ], axis = 1, join = 'inner' )

# what to do with Gender (string binary categorical variable)?

# convert to zero (male) and one (female)

# trick:  first convert to boolean (Gender==female) , then to int by adding 0

joined_data = joined_data.assign( numeric_gender = 0 + ( joined_data[ 'Gender' ] == 'female' ) )

# exclude User_ID and log_Calories from the prediction model (they're not features)

del joined_data[ 'User_ID' ]
ageDF = joined_data[ 'Age' ]

heartRateDF = joined_data[ 'Heart_Rate' ]



# remove unneeded columns



# remove Duration and Heart_Rate

del joined_data[ 'Duration' ]

del joined_data[ 'Heart_Rate' ]

del joined_data[ 'Calories' ]





joined_data.pop( 'Body_Temp' )

joined_data.pop( 'Height' )

joined_data.pop( 'Weight' )

joined_data.pop( 'squared_heart_rate' )

joined_data.pop( 'Age' )

joined_data.pop( 'squared_duration' )

joined_data.pop( 'Gender' )

joined_data.head()
# split data into test and training



from sklearn.model_selection import train_test_split



train, test = train_test_split( joined_data, test_size = 0.3 )
# separate features from what we want to predict

train_target = train[ 'log_Calories' ]

train.pop( 'log_Calories' )



# create linear regression object and train the model (ordinary least squares linear regression)

from sklearn import linear_model

regr = linear_model.LinearRegression( fit_intercept = True )

regr.fit( train, train_target )
# separate features from what we want to test

test_target = test[ 'log_Calories' ]

del test[ 'log_Calories' ]
test_prediction = regr.predict( test )

print(test_prediction)

# evaluate model against test data 

# coeficient of determination (r-squared is better near 1)

from sklearn.metrics import r2_score

rSquared = r2_score( test_target, test_prediction )
# relative absolute/squared error (it is better near zero)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error( test_target, test_prediction )
# get ages and heart rates whose index match the test dataframe

ageDF = ageDF[ ageDF.index.isin( test.index ) ]

heartRateDF = heartRateDF[ heartRateDF.index.isin( test.index ) ]
# join log_Calories back into the main test dataframe

test = pd.concat( [ test, test_target ], axis = 1, join = 'inner' )

print(test)
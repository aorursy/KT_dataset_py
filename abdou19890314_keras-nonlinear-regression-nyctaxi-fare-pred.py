import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import numpy
%matplotlib inline
size = 10_000_000
percent = 25
sizeSplit = int(size * (100 - percent) / 100)
print('sizeSplit : ' + str(sizeSplit))
print(os.listdir('../input'))
#Red data from csv file for training and validation data
train_df =  pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows = size)
print(train_df.dtypes)
# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
add_travel_vector_features(train_df)
print(train_df.isnull().sum())
train_df = train_df.dropna(how = 'any', axis = 'rows')
plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print(plot)
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
print(plot)
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))
train_X = get_input_matrix(train_df)
train_y = np.array(train_df['fare_amount'])

print(train_X.shape)
print(train_y.shape)
X1 = train_X[0:sizeSplit]
X2 = train_X[sizeSplit:size]
Y1 = train_y[0:sizeSplit]
Y2 = train_y[sizeSplit:size]
#numpy.savetxt("testSplit.csv", X2, delimiter=",")
#numpy.savetxt("valTtestSplit.csv", Y2, delimiter=",")
# Fit the model
model = keras.models.load_model("../input/modelsfile/predictedModel.model")

# Calculate predictions
PredTestSet = model.predict(X1)
PredValSet = model.predict(X2)

# Save predictions
#numpy.savetxt("trainresults.csv", PredTestSet, delimiter=",")
#numpy.savetxt("valresults.csv", PredValSet, delimiter=",")

##### Plot actual vs predition for training set
#TestResults = numpy.genfromtxt("trainresults.csv", delimiter=",")
#plt.plot(Y1,TestResults,'ro')
plt.plot(Y1,PredTestSet,'ro')
plt.title('Training Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')

#Compute R-Square value for training set
TestR2Value = r2_score(Y1,PredTestSet)
print("Training Set R-Square=", TestR2Value)
#Compute explained_variance_score value for training set
TestR2Value = explained_variance_score(Y1,PredTestSet)
print("Training Set explained_variance_score=", TestR2Value)
#Plot actual vs predition for validation set
#ValResults = numpy.genfromtxt("valresults.csv", delimiter=",")
plt.plot(Y2,PredValSet,'ro')
plt.title('Validation Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')

#Compute R-Square value for validation set
ValR2Value = r2_score(Y2,PredValSet)
print("Validation Set R-Square=",ValR2Value)
#Compute explained_variance_score value for validation set
ValR2Value = explained_variance_score(Y2,PredValSet)
print("Validation Set explained_variance_score=",ValR2Value)
import decimal
realValues = pd.read_csv('../input/new-york-city-taxi-fare-prediction/sample_submission.csv')
test_df = pd.read_csv('../input/new-york-city-taxi-fare-prediction/test.csv')
keys = test_df.key
add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)
PredTestX = model.predict(test_X)
PredTestX = PredTestX.round(decimals=2)
PredTestXList = map(lambda x: x[0], PredTestX)
serPredTest = pd.Series(PredTestXList)
realValues.fare_amount.update(serPredTest)
# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(
    {'key': keys, 'fare_amount': serPredTest},
    columns = ['key', 'fare_amount'])
submission.to_csv('sample_submission_prediction.csv', index = False)
#realValues.to_csv('sample_submission_prediction.csv')
#numpy.savetxt("sample_submission_prediction.csv", PredTestX, delimiter=",")
import numpy as np # linear algebra

import pandas as pd # data processing

import os

import matplotlib.pyplot as plt # charts and graphs

import seaborn as sns # styling & pretty colors

import shutil

import tensorflow as tf



dataset = pd.read_csv('../input/PGA_Data_Historical.csv')

dataset.info()

dataset.head()

print(tf.__version__)
#show lots of columns and rows in pandas

pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 3000)

pd.set_option('display.width', 3000)
# Transpose based on key value pairs

df = dataset.set_index(['Player Name', 'Variable', 'Season'])['Value'].unstack('Variable').reset_index()



print("original column count:\t" + str(len(dataset.columns)))

print("     new column count:\t" + str(len(df.columns)))

df.head()
#Narrow down to the more interesting X columns. You could add others.

Keep_Columns = ['Player Name','Season','Total Money (Official and Unofficial) - (MONEY)','3-Putt Avoidance - (%)',

                'Average Distance to Hole After Tee Shot - (AVG)','Scrambling from the Sand - (%)',

                'Scrambling from the Fringe - (%)','Scrambling from the Rough - (%)',

                'Driving Accuracy Percentage - (%)','Total Distance Efficiency - (AVERAGE DISTANCE (YARDS))',

                'Ball Speed - (AVG.)','Birdie or Better Conversion Percentage - (%)'

               ]

Keep_Columns 

#Drop non-numeric

df=df[Keep_Columns].dropna()

#Rename the columns to something shorter

df.rename(columns = {'Total Money (Official and Unofficial) - (MONEY)':'Money'}, inplace = True)

df.rename(columns = {'3-Putt Avoidance - (%)':'ThreePutt'}, inplace = True)

df.rename(columns = {'Average Distance to Hole After Tee Shot - (AVG)':'AverageDistance'}, inplace = True)

df.rename(columns = {'Scrambling from the Sand - (%)':'ScramblingSand'}, inplace = True)

df.rename(columns = {'Scrambling from the Fringe - (%)':'ScramblingFringe'}, inplace=True)

df.rename(columns = {'Scrambling from the Rough - (%)':'ScramblingRough'}, inplace=True)

df.rename(columns = {'Driving Accuracy Percentage - (%)':'DrivingAccuracy'}, inplace=True)

df.rename(columns = {'Total Distance Efficiency - (AVERAGE DISTANCE (YARDS))':'Distance'}, inplace=True)

df.rename(columns = {'Ball Speed - (AVG.)':'BallSpeed'}, inplace=True)

df.rename(columns = {'Birdie or Better Conversion Percentage - (%)':'BirdieConversion'}, inplace=True)

df.head()
df.columns
#Remove $ and commas from Money

df['Money']= df['Money'].str.replace('$','')

df['Money']= df['Money'].str.replace(',','')
#Make all variables into numbers

for col in  df.columns[2:]:

   df[col] = df[col].astype(float)

df
#Scale the data so that all features are of a comparable magnitude

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

Feature_Columns=['ThreePutt', 'AverageDistance', 'ScramblingSand', 'ScramblingFringe', 'ScramblingRough', 'DrivingAccuracy', 'Distance', 'BallSpeed', 'BirdieConversion']

df[Feature_Columns]=scaler.fit_transform(df[Feature_Columns])

df
#check correlations with Money, to see which features are useful.  Driving accuracy seems to be not so useful. 

#The others look useful.

df.corr(method ='pearson') 
sns.regplot(x="BirdieConversion", y="Money", data=df);
#Three putts are negatively correlated with Money

sns.regplot(x="ThreePutt", y="Money", data=df);
sns.regplot(x="ScramblingFringe", y="Money", data=df);
sns.regplot(x="ScramblingSand", y="Money", data=df);
sns.regplot(x="ScramblingRough", y="Money", data=df);
sns.regplot(x="DrivingAccuracy", y="Money", data=df);
sns.regplot(x="BallSpeed", y="Money", data=df);
sns.regplot(x="Distance", y="Money", data=df);
# Imports for Linear Regression

import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
# Create a LinearRegression Object

lreg = LinearRegression()
# Drop a few columns to get the X (feature) columns

X=df.drop(['Money','Player Name', 'Season'], axis=1)

# Target

Y=df.Money
#I'm going to add some squared features and crossed features

#This did improve the regression results vs only linear features 

poly = PolynomialFeatures(2)

X=poly.fit_transform(X)  
X
X.shape
#Split into training and test sets

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X,Y)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
lreg.fit(X_train,Y_train)
lreg.coef_
lreg.intercept_
# Calculate predictions on training and test data sets

pred_train = lreg.predict(X_train)

pred_test = lreg.predict(X_test)

pred_train
print("RMSE with Y_train: %.2f"  % np.sqrt(np.mean((Y_train - pred_train) ** 2)))

print("RMSE with Y_test: %.2f"  % np.sqrt(np.mean((Y_test - pred_test) ** 2)))
# Scatter plot the training data

train = plt.scatter(pred_train,(Y_train-pred_train),c='b',alpha=0.5)



# Scatter plot the testing data

test = plt.scatter(pred_test,(Y_test-pred_test),c='r',alpha=0.5)



# Plot a horizontal axis line at 0

plt.hlines(y=0,xmin=-10,xmax=50)



#Labels

plt.legend((train,test),('Training','Test'),loc='lower left')

plt.title('Residual Plot')
Jason_Day=df[df['Player Name'].str.contains('Day')]

Jason_Day.Money.mean()
Adam_Hadwin=df[df['Player Name'].str.contains('Hadwin')]

Adam_Hadwin.Money.mean()
X_predict=Jason_Day.drop(['Money','Player Name', 'Season'], axis=1)

lreg.predict(poly.fit_transform(X_predict)).mean()
X_predict=Adam_Hadwin.drop(['Money','Player Name', 'Season'], axis=1)

lreg.predict(poly.fit_transform(X_predict)).mean()
Dustin_Johnson=df[df['Player Name'].str.contains('Dustin Johnson')]

Dustin_Johnson.Money.mean()
X_predict=Dustin_Johnson.drop(['Money','Player Name', 'Season'], axis=1)

lreg.predict(poly.fit_transform(X_predict)).mean()
FEATURE_NAMES=list(df.columns.drop(['Money','Player Name', 'Season']))

FEATURE_NAMES
LABEL_NAME = 'Money'
# Drop a few columns to get a dataframe with the target (Money) as well as the X variables

DNNdata=df.drop(['Player Name', 'Season'], axis=1)
DNNdata
DNNdata.shape
#I got this code from Google's Github on Tensorflow training: 

#https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/03_tensorflow/b_estimator.ipynb



def make_train_input_fn(d, num_epochs):

  return tf.estimator.inputs.pandas_input_fn(

    x = d,

    y = d[LABEL_NAME],

    batch_size = 64,

    num_epochs = num_epochs,

    shuffle = True,

    queue_capacity = 1000

  )



def make_eval_input_fn(d):

  return tf.estimator.inputs.pandas_input_fn(

    x = d,

    y = d[LABEL_NAME],

    batch_size = 64,

    shuffle = False,

    queue_capacity = 1000

  )



def make_prediction_input_fn(d):

  return tf.estimator.inputs.pandas_input_fn(

    x = d,

    y = None,

    batch_size = 128,

    shuffle = False,

    queue_capacity = 1000

  )



def make_feature_cols():

  input_columns = [tf.feature_column.numeric_column(k) for k in FEATURE_NAMES]

  return input_columns



DATASET_SIZE = DNNdata.shape[0]

DATASET_SIZE
#Divide into train and test sets

DATASET_SIZE = DNNdata.shape[0]



train_df=DNNdata.sample(frac=0.8,random_state=200)

test_df=DNNdata.drop(train_df.index)
train_df.shape[0], test_df.shape[0]
OUTDIR = "Golf_Training"



tf.logging.set_verbosity(tf.logging.INFO)



shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time



model = tf.estimator.DNNRegressor(hidden_units = [16,8],

      feature_columns = make_feature_cols(), model_dir = OUTDIR)
model.train(input_fn = make_train_input_fn(train_df, num_epochs = 500))
def print_rmse(model, d):

  metrics = model.evaluate(input_fn = make_eval_input_fn(df))

  print('RMSE on dataset = {}'.format(np.sqrt(metrics['average_loss'])))

print_rmse(model, test_df)
# Try a prediction for Jason Day

predictions = model.predict(input_fn = make_prediction_input_fn(Jason_Day))

for items in predictions:

  print(items)
predictions = model.predict(input_fn = make_prediction_input_fn(Adam_Hadwin))

for items in predictions:

  print(items)
df.mean()
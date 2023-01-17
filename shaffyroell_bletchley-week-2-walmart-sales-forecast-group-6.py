import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
len(train) # Get number of training examples
len(test) # Get number of test examples
df = pd.concat([train,test],axis=0) # Join train and test
df.head() # Get an overview of the data
df.describe()

df.isnull().sum()
df.isnull().sum()
df.fillna(0, inplace=True)
df.isnull().sum()

df.dtypes
df['Week'] = pd.to_datetime(df.Date).dt.week
df['Year'] = pd.to_datetime(df.Date).dt.year
# Make sure we can later recognize what a dummy once belonged to
df['Type'] = 'Type_' + df['Type'].map(str)
df['Store'] = 'Store_' + df['Store'].map(str)
df['Dept'] = 'Dept_' + df['Dept'].map(str)
df['Week'] = 'Week_' + df['Week'].map(str)
df['Year'] = 'Year_' + df['Year'].map(str)
# Create dummies
type_dummies = pd.get_dummies(df['Type'])
store_dummies = pd.get_dummies(df['Store'])
dept_dummies = pd.get_dummies(df['Dept'])
week_dummies = pd.get_dummies(df['Week'])
year_dummies = pd.get_dummies(df['Year'])
# Add dummies
df = pd.concat([df,type_dummies,store_dummies,dept_dummies, week_dummies, year_dummies],axis=1)
# Remove originals
del df['Type']
del df['Store']
del df['Dept']
del df['Week']
del df['Year']
del df['Date']
#del df['CPI']
#del df['Fuel_Price']
#del df['MarkDown1']
#del df['MarkDown2']
#del df['MarkDown3']
#del df['MarkDown4']
#del df['MarkDown5']
#del df['Size']
del df['Temperature']
#del df['Unemployment']
#del df['IsHoliday']
df.head()
df.shape
# smaller training set just to test out different models
train_fake = df.iloc[:15000]
train = df.iloc[:282451]

test_fake = df.iloc[15000:20000]
test = df.iloc[282451:]
test = test.drop('Weekly_Sales',axis=1) # We should remove the nonsense values from test
y = train['Weekly_Sales'].values
y.shape
X = train.drop('Weekly_Sales',axis=1).values
X.shape
from keras.layers import Dense, Activation
from keras.models import Sequential
model = Sequential()

# Linear regression is a single layer network
model.add(Dense(1, input_dim=194, activation='linear'))

model.compile(optimizer='adam',
              loss='mae')
model.fit(X, y, epochs=5, batch_size= 2048)
model.evaluate(x=X,y=y)
y_pred = model.predict(test.values, batch_size = X.shape[0])
y_pred[:10]
X_test = test.values
y_pred = model.predict(X_test,batch_size=2048)
testfile = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                      'Weekly_Sales':y_pred.flatten()})
submission.to_csv('submission_batchmode_1.csv',index=False)

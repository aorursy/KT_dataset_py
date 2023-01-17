
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

len(train) # Get number of training examples

len(test) # Get number of test examples

df = pd.concat([train,test],axis=0) # Join train and test
df.head() # Get an overview of the data

#df.isnull().sum()

df.fillna(0, inplace=True)

df['Week'] = pd.to_datetime(df.Date).dt.week

# Make sure we can later recognize what a dummy once belonged to
df['Type'] = 'Type_' + df['Type'].map(str)
df['Store'] = 'Store_' + df['Store'].map(str)
df['Dept'] = 'Dept_' + df['Dept'].map(str)
df['Week'] = 'Week_' + df['Week'].map(str)

# Create dummies
type_dummies = pd.get_dummies(df['Type'])
store_dummies = pd.get_dummies(df['Store'])
dept_dummies = pd.get_dummies(df['Dept'])
week_dummies = pd.get_dummies(df['Week'])

# Add dummies
df = pd.concat([df,type_dummies,store_dummies,dept_dummies, week_dummies],axis=1)

# Remove originals
del df['Type']
del df['Store']
del df['Dept']
del df['Week']
del df['Date']

# Remove variables that are not useful
del df['CPI']
del df['MarkDown2']
del df['MarkDown3']
del df['MarkDown4']
del df['MarkDown5']
del df['Unemployment']

df.dtypes
#Split the sets
#train = df.iloc[:282451]
#test = df.iloc[282451:]

# smaller training set just to test out different models
#train_fake = df.iloc[:15000]
train = df.iloc[:282451]

#test_fake = df.iloc[15000:20000]
test = df.iloc[282451:]

#test = test.drop('Weekly_Sales',axis=1) # We should remove the nonsense values from test
test = test.drop('Weekly_Sales',axis=1) # We should remove the nonsense values from test


#numpy array out of the panda data frame FOR FAKES
y = train['Weekly_Sales'].values
X = train.drop('Weekly_Sales',axis=1).values

#numpy array out of the panda data frame
#y = train['Weekly_Sales'].values
#X = train.drop('Weekly_Sales',axis=1).values
X.shape
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import metrics
from keras import regularizers
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.optimizers import adam
import matplotlib.pyplot as plt
# Sequential model
model = Sequential()

# Logistic regresison is a single layer network
model.add(Dense(50,activation='relu',input_dim=186))

model.add(Dense(50,activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dense(1,activation='linear'))

# Compile the model
model.compile(optimizer='adam',loss='mae',metrics=['mae'])

#model.fit(X, y, # Train on training set
                            # epochs=1000, # We will train over 1,000 epochs
                            # batch_size=X.shape[0], # Batch size = training set size
                            # verbose=0) # Suppress Keras output


# Train
history = model.fit(X, y, epochs=5, batch_size = 100) 
plt.plot(history.history['mean_absolute_error'])
plt.xlabel('Epochs')
plt.ylabel('mae')
plt.legend()
plt.show()
test.head()
ypred = model.predict(test)
testfile = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                          'Weekly_Sales':ypred.flatten()})
submission.to_csv('submission.csv',index=False)
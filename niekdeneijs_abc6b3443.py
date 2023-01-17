import pandas as pd
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
len(train) # Get number of training examples
len(test) # Get number of test examples
df = pd.concat([train,test],axis=0) # Join train and test
df.head() # Get an overview of the data
df.describe()
df.isnull().sum()
df = df.assign(md1_present = df.MarkDown1.notnull())
df = df.assign(md2_present = df.MarkDown2.notnull())
df = df.assign(md3_present = df.MarkDown3.notnull())
df = df.assign(md4_present = df.MarkDown4.notnull())
df = df.assign(md5_present = df.MarkDown5.notnull())


df.isnull().sum()
df.fillna(0, inplace=True)
df['MarkDown1'] = (df['MarkDown1'] - df['MarkDown1'].mean())/(df['MarkDown1'].std())
df['MarkDown2'] = (df['MarkDown2'] - df['MarkDown2'].mean())/(df['MarkDown2'].std())
df['MarkDown3'] = (df['MarkDown3'] - df['MarkDown3'].mean())/(df['MarkDown3'].std())
df['MarkDown4'] = (df['MarkDown4'] - df['MarkDown4'].mean())/(df['MarkDown4'].std())
df['MarkDown5'] = (df['MarkDown5'] - df['MarkDown5'].mean())/(df['MarkDown5'].std())



df['Unemployment'] = (df['Unemployment'] - df['Unemployment'].mean())/(df['Unemployment'].std())
df['CPI'] = (df['CPI'] - df['CPI'].mean())/(df['CPI'].std())
df['Fuel_Price'] = (df['Fuel_Price'] - df['Fuel_Price'].mean())/(df['Fuel_Price'].std())

df['Size'] = (df['Size'] - df['Size'].mean())/(df['Size'].std())





sns.distplot(df['Size'])
df['SSize'] = np.where(df["Size"] < -1.0, 1,0)
df['MSize'] = np.where((df["Size"] >= -1.0) & (df["Size"] < 0.5) , 1,0)
df['LSize'] = np.where(df["Size"] > 0.5, 1,0)

#df['LSize'] =  np.where(df['Size'] >= 0.5, 1,0)

sns.distplot(df['Fuel_Price'])
df['SFuel_Price'] = np.where(df["Fuel_Price"] < -0.2, 1,0)
df['LFuel_Price'] = np.where(df["Fuel_Price"] > -0.2, 1,0)
sns.distplot(df['CPI'])
df['SCPI'] = np.where(df["CPI"] < -0.5, 1,0)
df['MCPI'] = np.where((df["CPI"] >= -0.5) & (df["Size"] < 0.75) , 1,0)
df['LCPI'] = np.where(df["CPI"] > 0.75, 1,0)
sns.distplot(df['MarkDown1'])
sns.distplot(df['Unemployment'])
#df.fillna(0, inplace=True)

df.dtypes
# Make sure we can later recognize what a dummy once belonged to
df['Type'] = 'Type_' + df['Type'].map(str)
df['Store'] = 'Store_' + df['Store'].map(str)
df['Dept'] = 'Dept_' + df['Dept'].map(str)
# Create dummies
type_dummies = pd.get_dummies(df['Type'])
store_dummies = pd.get_dummies(df['Store'])
dept_dummies = pd.get_dummies(df['Dept'])
# Add dummies
df = pd.concat([df,type_dummies,store_dummies,dept_dummies],axis=1)
ChristmasDays = ['2010-12-31','2011-12-30','2012-12-28','2013-12-27']
ThanksgivingDays = ['2010-11-26','2011-11-25','2012-11-23','2013-11-29']
SpecialDays = ['2010-12-31','2011-12-30','2012-12-28','2013-12-27','2010-11-26','2011-11-25','2012-11-23','2013-11-29']





# Make christmas
df['Christmas'] = np.where(df['Date'].isin(ChristmasDays), 1,0)
df['Thanksgiving'] = np.where(df['Date'].isin(ThanksgivingDays), 1,0)
df['NotSpecial'] = np.where(~df['Date'].isin(SpecialDays), 1,0)
df['January'] = np.where(pd.to_datetime(df['Date']).dt.month == 1, 1,0)

# Remove originals
del df['Type']
del df['Store']
del df['Dept']
del df['Date']
df.dtypes
train = df.iloc[:282451]
test = df.iloc[282451:]
test = test.drop('Weekly_Sales',axis=1) # We should remove the nonsense values from test
y = train['Weekly_Sales'].values
X = train.drop('Weekly_Sales',axis=1).values
X.shape
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import BatchNormalization
model = Sequential()

model.add(Dense(74,input_dim=157))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Dense(36,input_dim=74))
model.add(Activation('relu'))


model.add(Dense(1,input_dim=36))



model.compile(optimizer='adam', loss='mae')
model.fit(X,y,batch_size=2048,epochs=100)
X_test = test.values
y_pred = model.predict(X_test,batch_size=2048)
testfile = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                          'Weekly_Sales':y_pred.flatten()})
submission.to_csv('submission.csv',index=False)

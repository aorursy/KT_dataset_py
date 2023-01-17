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
def get_holiday_feature(date):
    super_bowl = ['2010-02-12','2011-02-11','2012-02-10','2013-02-08']
    labor = ['2010-09-10','2011-09-09','2012-09-07','2013-09-06']
    thanksgiving = ['2010-11-26','2011-11-25','2012-11-23','2013-11-29']
    christmas = ['2010-12-31','2011-12-30','2012-12-28','2013-12-27']
    if date in super_bowl:
        return [0,0,0,1]
    elif date in labor:
        return [0,0,1,0]
    elif date in thanksgiving:
        return [0,1,0,0]
    elif date in christmas:
        return [1,0,0,0]
    else:
        return [0,0,0,0]
def dates(datelist):
    x = []
    for date in datelist:
        temp = 0
        temp = get_holiday_feature(date)
        x.append(temp)
    return x
x = dates(df['Date'])
x[:100]
df['Week'] = pd.to_datetime(df.Date).dt.week
df['Year'] = pd.to_datetime(df.Date).dt.year
lastweek = df.sort_values(by = ['Store', 'Dept', 'Date'])
sales = lastweek['Weekly_Sales'].values
avg = df['Weekly_Sales'].mean()
for i in range(1,len(sales)):
    avg.append((z[i-1]))
for j in range(len(avg)):
    if avg[j] == 0:
        avg[j] = Prev[j-1]
lastweek = lastweek.assign(np.array(avg))
df = lastweek
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
#del df['Temperature']
#del df['Unemployment']
#del df['IsHoliday']
df.head()
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
from keras import regularizers
model = Sequential()
model.add(Dense(1,input_dim=196,
                activation ='relu',
                kernel_regularizer= regularizers.l2(0.01)))
model.compile(optimizer='adam', loss='mae')
model.fit(X, y, epochs=5, batch_size= 2048)
model.evaluate(x=X,y=y)
y_pred = model.predict(test.values, batch_size = X.shape[0])
y_pred[:10]
X_test = test.values
y_pred = model.predict(X_test,batch_size=2048)
testfile = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                         'Weekly_Sales':y_pred.flatten()})
submission.to_csv('submission.csv',index=False)

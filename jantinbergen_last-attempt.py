import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
len(train) # Get number of training examples
len(test) # Get number of test examples
df = pd.concat([train,test],axis=0) # Join train and test
a1 = df.iloc[:282451]
a2 = df.iloc[282451:]
w = len(df)
p = [1]*282451
p = np.array(p)
q = np.arange(282451)
a1 = a1.assign(p=p)
p = [0]*(w-282451)
p = np.array(p)
q = np.arange(w-282451)
a2 = a2.assign(p = p)
df = pd.concat([a1,a2], axis= 0)
q = np.arange(len(df))
df = df.assign(q=q)
df.iloc[282451:]
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
C = df.sort_values(by = ['Store', 'Dept', 'Date'])
z = C['Weekly_Sales'].values
Prev = [df['Weekly_Sales'].mean()]
for i in range(1,len(z)):
    Prev.append((z[i-1]))
for j in range(len(Prev)):
    if Prev[j] == 0:
        Prev[j] = Prev[j-1]
C = C.assign(Prev = np.array(Prev))
df = C
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
# Remove originals
del df['Type']
del df['Store']
del df['Dept']
del df['Date']
del df['CPI']
del df['Temperature']
df.dtypes
train = df[df['p'] == 1]
test = df[df['p'] == 0]
del train['p']
del test['p']
train = train.sort_values(by = ['q'])
test = test.sort_values(by = ['q'])
del train['q']
del test['q']
test = test.drop('Weekly_Sales',axis=1) # We should remove the nonsense values from test
y = train['Weekly_Sales'].values
X = train.drop('Weekly_Sales',axis=1).values
X.shape
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import regularizers
model = Sequential()
model.add(Dense(1,input_dim=144,activation ='relu',kernel_regularizer= regularizers.l2(0.01)))
model.compile(optimizer='adam', loss='mae')
y[10]
model.fit(X,y,batch_size=2048,epochs=20)
model.evaluate(X,y)
X_test = test.values
test.head()
y_pred = model.predict(X_test,batch_size=2048)
y_pred[:10]
testfile = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                          'Weekly_Sales':y_pred.flatten()})
submission.to_csv('submission.csv',index=False)

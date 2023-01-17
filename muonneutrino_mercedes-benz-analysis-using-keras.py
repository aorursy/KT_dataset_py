import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt



%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv',index_col=0)

test = pd.read_csv('../input/test.csv',index_col=0)

train = train[['X0','y']]

train, val = train_test_split(train,test_size=0.3,random_state=1234)

test = test['X0']



groups = train.groupby('X0')

ymap = {}

for name,group in groups:

    ymap[name] = group.y.mean()



train['ypred'] = train.X0.map(ymap)

val['ypred'] = val.X0.map(ymap)    

val['ypred'] = val.ypred.fillna(train.y.mean())



train_score = r2_score(train.y,train.ypred)

val_score = r2_score(val.y,val.ypred)



print('Training score: ' + str(train_score))

print('Validate score: ' + str(val_score))



ytest = test.map(ymap)

ytest = ytest.fillna(train.y.mean())

        

        

ytest = pd.DataFrame({'y':ytest})

ytest.to_csv('submission_X0.csv')
from sklearn.model_selection import train_test_split



###

validate = False

###



train = pd.read_csv('../input/train.csv',index_col=0)

train.head()

test = pd.read_csv('../input/test.csv',index_col=0)



xtrain = pd.get_dummies(train.X0)

ytrain = train.y

xtest = pd.get_dummies(test.X0)



# Get list of columns

for col in xtrain.columns:

    if col not in xtest.columns:

        xtest[col] = 0

        

for col in xtest.columns:

    if col not in xtrain.columns:

        xtest = xtest.drop(col,axis=1)

        



if validate is True: 

    xtrain, xval, ytrain, yval = train_test_split(xtrain,ytrain,test_size=0.3,random_state=1234)

xtest = xtest.sort_index(axis=1)

print(xtrain.head())

print(xtest.head())
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.layers.core import Dropout

from keras import optimizers

from keras import regularizers
print('Building Model')

model = Sequential()

model.add(Dense(units=1,input_dim=xtrain.shape[1]))

model.add(Activation('linear')) # Linear to get fit

print('Compiling Model')

sgd = optimizers.SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error',optimizer=sgd)

print('Fit Model')

model.fit(xtrain.as_matrix(), ytrain.as_matrix(), epochs=1000, batch_size=512)

print('Evaluating and predicting')

#loss = model.evaluate(xtrain.as_matrix(),ytrain.as_matrix(),batch_size=128)

train_vals = model.predict(xtrain.as_matrix(),batch_size=128)

if validate is True:

    val_vals = model.predict(xval.as_matrix(), batch_size=128)

test_vals = model.predict(xtest.as_matrix(),batch_size=128)
from sklearn.metrics import r2_score

if validate is True:

    val_score = r2_score(yval,val_vals)

    print('Validation score: '+ str(val_score))



train_score = r2_score(ytrain,train_vals)

print('Training score: ' + str(train_score))

#model.get_weights()[0] + model.get_weights()[1][0]
test['y'] = test_vals

test_out = test[['y']]

test_out.to_csv('submission_linearfit.csv')
train = pd.read_csv('../input/train.csv',index_col=0)

train.head()

test = pd.read_csv('../input/test.csv',index_col=0)

dum = pd.get_dummies(train.X0,drop_first = True)

train = pd.merge(train,dum,left_index=True,right_index=True,suffixes=('','_x0'))

dum = pd.get_dummies(test.X0,drop_first = True)

test = pd.merge(test,dum,left_index=True,right_index=True,suffixes=('','_x0'))

train.head()



# Get list of columns

for col in train.columns:

    if col not in test.columns:

        test[col] = 0

        

for col in test.columns:

    if col not in train.columns:

        test = test.drop(col,axis=1)

        

groups = train.groupby('X0')

ymap = {}

for name,group in groups:

    ymap[name] = group.y.mean()



train['yX0'] = train.X0.map(ymap)

train['ydiff'] = train.y - train.yX0

test['yX0'] = test.X0.map(ymap)

test['yX0'] = test['yX0'].fillna(train.y.mean())

        

print(test.shape)

print(train.shape)
validate = True



from sklearn.model_selection import train_test_split

if validate is True:

    train, val = train_test_split(train,test_size=0.3,random_state=1234)

fig = plt.figure(1,figsize=(10,10))

ax = fig.add_subplot(221)

plt.hist(train.y,bins=80)

plt.xlabel('y')

plt.ylabel('Number of Entries')

ax = fig.add_subplot(222)

plt.hist(np.log(train.y),bins=80)

plt.xlabel('log(y)')

plt.ylabel('Number of Entries')

ax = fig.add_subplot(223)

plt.hist(train.ydiff,bins=80)

plt.xlabel('y-y(X0)')

plt.show()
dupl_cols = []

for i in range(10,386):

    for j in range(i+1,386):

        try:

            label1 = 'X%i'%(i)

            label2 = 'X%i'%(j)

            vals = (train[label1]==train[label2])

            if vals.std()<0.02:

                dupl_cols.append(label2)

        except:

            pass

#print(dupl_cols)

dupl_cols = {x for x in dupl_cols} # unique set

print('# of duplicate columns: ' +str(len(dupl_cols)))



train = train.drop(dupl_cols,axis=1)

if validate is True:

    val = val.drop(dupl_cols,axis=1)

test = test.drop(dupl_cols,axis=1)
diff = []

name = []

c0 = []

c1 = []

mean0 = []

mean1 = []

std0 = []

std1 = []

for i in range(10,386):

    try:

        yy = train.groupby('X%i'%(i)).ydiff

        diff0 = np.abs(yy.mean()[1] - yy.mean()[0])/np.sqrt(yy.var()[1]+yy.var()[0])

        #c0 = yy.count()[0]

        #c1 = yy.count()[1]

        c0.append(yy.count()[0])

        c1.append(yy.count()[1])

        mean0.append(yy.mean()[0])

        mean1.append(yy.mean()[1])



        std0.append(yy.std()[0])

        std1.append(yy.std()[1])

        diff.append(diff0)

        name.append('X%i'%(i))



    except:

        pass

df = pd.DataFrame({'c0':c0,'c1':c1,'diff':diff,'mean0':mean0,'std0':std0,'mean1':mean1,'std1':std1},index=name)

indices = df[((df.c0<=10) | (df.c1<=10) | (df['diff']<=0.05))].index

#df = df[((df.c0>50) & (df.c1>50) & (df['diff']>0.2))].sort_values(by='diff',ascending=False)

df = df[(df.c0>10) & (df.c1>10) & (df['diff']>0.05)].sort_values(by='diff',ascending=False)

df.head(100)
ytrain = train.loc[:,['y','yX0','ydiff']]

cols = [x for x in df.index]

if validate is True:

    xval = val[cols]

    yval = val.loc[:,['y','yX0','ydiff']]

ytest = test.loc[:,['yX0']]

xtrain = train[cols]

xtest = test[cols]
xtrain.head()
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.layers.core import Dropout

from keras import optimizers

from keras import regularizers
print('Building Model')

model = Sequential()

l2reg = 0.0

model.add(Dropout(0.2,input_shape=(xtrain.shape[1],)))

model.add(Dense(units=10,kernel_regularizer=regularizers.l2(l2reg)))

model.add(Activation('relu')) # These are all categorical so probably doesn't matter

model.add(Dropout(0.2))



model.add(Dense(units=1,kernel_regularizer=regularizers.l2(l2reg)))

#model.add(Dropout(0.5))

model.add(Activation('linear')) # Linear to get fit

print('Compiling Model')

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)

model.compile(loss='mean_squared_error',optimizer=sgd)

print('Fit Model')

model.fit(xtrain.as_matrix(), ytrain.ydiff.as_matrix(), epochs=1000, batch_size=64)

print('Evaluating and predicting')

#loss = model.evaluate(xtrain.as_matrix(),ytrain.as_matrix(),batch_size=128)

train_vals = model.predict(xtrain.as_matrix(),batch_size=128)

test_vals = model.predict(xtest.as_matrix(),batch_size=128)



ytrain['ypred'] = train_vals

if validate is True:

    val_vals = model.predict(xval.as_matrix(), batch_size=128)

    yval['ypred'] = val_vals

    yval['ypred'] = yval.ypred+yval.yX0



ytest['ypred'] = test_vals

ytrain['ypred'] = ytrain.ypred+ytrain.yX0

ytest['ypred'] = ytest.ypred+ytest.yX0
from sklearn.metrics import r2_score

train_score = r2_score(ytrain.y,ytrain.ypred)

train_score2 = r2_score(ytrain.ydiff,ytrain.ypred-ytrain.yX0)

print('Training score (X0 diff): ' + str(train_score2))

print('Training score (full): ' + str(train_score))



if validate is True:

    val_score2 = r2_score(yval.ydiff,yval.ypred-yval.yX0)

    val_score = r2_score(yval.y,yval.ypred)

    print('Validation score (X0 diff): '+ str(val_score2))

    print('Validation score (full): '+ str(val_score))
test_out = ytest[['ypred']]

test_out = test_out.sort_index(ascending=True)

test_out['y'] = test_out.ypred

test_out = test_out[['y']]

test_out.to_csv('submission.csv')
train.ydiff.std()

quantiles = train.ydiff.quantile([0.16,0.84])

0.25*(quantiles.iloc[1]- quantiles.iloc[0])**2 / train.ydiff.var()
# here are the available input files for this competition
!ls -l ../input
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
# read the raw training data
df = pd.read_csv('../input/training_set.csv')

# I have found from other analysis that the measurement for passband 0 
# is only non-null when ALL the other passband measurements in the 
# same ostensible "observation" ARE null; we might be able to fix this 
# by adjusting the date binning, but for now let's just drop PB 0
df = df[df['passband'] != 0]

# digitize the dates so that the different passband measurements 
# that are close together in time can hang together as one observation
bins = np.array(np.arange(59000.0, 61000.0, 1.0, dtype=np.float64))
df['date_bin'] = np.digitize(df['mjd'], bins)

# take just the columns needed for a first very simple attempt
df = df[['object_id', 'date_bin', 'passband', 'flux']]

# pivot the passband readings into columns; we are working toward 
# input for a Keras LSTM model; the date digitizing has not worked 
# perfectly, but there are less than 2000 duplicate indexes in almost 
# a million rows, so just delete them
df = df.set_index(['object_id', 'date_bin', 'passband'])
df = df[~df.index.duplicated(keep='first')]
df = df.unstack(level=-1)

# flatten the structure of the table
df = df.reset_index()
df.columns = ['object_id', 'date_bin', 'pb1', 'pb2', 'pb3', 'pb4', 'pb5']

# we can drop all rows with any nulls and still have 100K rows, 
# so for now let's do it; we may have to revisit this is we don't 
# have a proper mix of target values left
df.dropna(inplace=True)

# pull in the target values
dfm = pd.read_csv('../input/training_set_metadata.csv')
df_tgt = dfm[['object_id', 'target']]
df = df.merge(df_tgt, on='object_id', how='left')

# for the 1st attempt I would like to just try binary classification 
# between the most numerous class (90) and all the others, so make new target col
df['tgt90'] = (df['target'] == 90).astype(int)

# check what we have
print('{} rows, {} IDs, {} tgts'.format(len(df), df['object_id'].nunique(), df['target'].nunique()))
df.head()

# roll up to object ID level; some of the aggregate values are 
# just for sanity checking
aggs = {'target':['min', 'max', 'size'], 'date_bin':['min', 'max'], 'tgt90':['min', 'max']}
new_cols = ['tgt-min', 'tgt-max', 'tgt-size', 'dbin-min', 'dbin-max', 'tgt90-min', 'tgt90-max']
df_agg = df.groupby('object_id').agg(aggs)
df_agg.columns = new_cols
df_agg['tgt-diff'] = df_agg['tgt-max'] - df_agg['tgt-min']
df_agg['dbin-diff'] = df_agg['dbin-max'] - df_agg['dbin-min']

# about 25% of the objects have less than 50 timesteps so 
# lets discard them
df_agg = df_agg[df_agg['tgt-size'] >= 50]

print(df_agg.shape)
print(df_agg['tgt90-min'].value_counts())
df_agg.head(10)
# make lists of object IDs to use for extracting sets of observations

df_train, df_test = train_test_split(df_agg.sample(200), test_size=0.3, random_state=42)
print('train: {}, test: {}'.format(len(df_train), len(df_test)))
print(df_train['tgt90-min'].value_counts())
print(df_test['tgt90-min'].value_counts())
idlist_train = df_train.index.values
idlist_test = df_test.index.values
print('{:5d} training IDs: {}...'.format(len(idlist_train), idlist_train[:5]))
print('{:5d} testing  IDs: {}...'.format(len(idlist_test), idlist_test[:5]))
# use ID lists to create numpy arrays for input to Keras model
# uses DF created in tidying section above

timesteps = 50
features = 5

def make_input_arrays_from_list(l):
    X = np.zeros((len(l), timesteps, features))
    y = np.zeros((len(l)))
    dfx = df[df['object_id'].isin(l)]
    for i, (oid, dfg) in enumerate(dfx.groupby('object_id')):
        X[i] = (dfg.iloc[:50]).values[:, 2:7]
        y[i] = dfg.iloc[0,-1]
        
    print('\nX,y shapes: {} {}'.format(X.shape, y.shape))
    print('y: {}, sum={}'.format(y[:10], sum(y)))
    print('X[0, :2]:')
    print(X[0, :2])
    print('X[0, -2:]:')
    print(X[0, -2:])
    print('X[-1, :2]:')
    print(X[-1, :2])
    print('X[-1, -2:]:')
    print(X[-1, -2:])
    return X, y

train_X, train_Y = make_input_arrays_from_list(idlist_train)
test_X, test_Y = make_input_arrays_from_list(idlist_test)

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing import sequence

epochs = 100
batch_size = 10
time_steps = 50
features = 5

# build LSTM layers
model = Sequential()
model.add(LSTM(100, dropout=0.2, input_shape=(time_steps, features)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=epochs, batch_size=batch_size)

# score model and log accuracy and parameters
scores = model.evaluate(test_X, test_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

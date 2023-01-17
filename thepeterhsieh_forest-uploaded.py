# import 
from IPython import get_ipython
from IPython.display import display
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')

# check the data source 
from subprocess import check_output
print(check_output(["ls", "../input/aia-st4-forest-type-prediction-dl"]).decode("utf8"))

# load the dataset as test & train
train = pd.read_csv("../input/aia-st4-forest-type-prediction-dl/train.csv")
test = pd.read_csv("../input/aia-st4-forest-type-prediction-dl/test.csv")
train.head(5)

# raw shape and type
display(train.shape, test.shape)
display(type(train), type(test))

# store the shapes
shapes = {'train':[], 'test':[], 'prep':[]}
shapes['train'].append(train.shape)
shapes['test'].append(test.shape)
shapes['prep'].append('raw')
display(shapes)

# pop the columns = 'Id'
display(train.columns[0] == 'Id', test.columns[0] == 'Id')

## save the 'Id' columns for further use 
train_id = train.pop('Id') # might be useless
test_id = test.pop('Id') # this is for the submission 
## pop the 'Id'
display('Id' in train.columns) # check if the popping is executed
# since the prediction is not hinged on it
display('Id' in test.columns)

# shape and type after popping 'Id'
shapes['train'].append(train.shape)
shapes['test'].append(test.shape)
shapes['prep'].append('pop_Id')
display(shapes)

## check correlation
corrmat = train.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corrmat, vmax=0.9, square=True)
display(corrmat['Cover_Type'])

## inspect the target: 'Cover_Type'
train['Cover_Type'].shape[0] == train.shape[0]
train_target = train.pop('Cover_Type')

# check if any na
display(len(train_target[train_target.isna()])) # it shows no na

# check dtype
train_target.dtype # it shows int64

# count the unique numbers
target_values = pd.DataFrame(train_target.value_counts(ascending=True).values, columns=['number']) 
target_values['tree']=target_values.index
display(target_values)

# vis
# the bar plot
fig_target0 = plt.figure(figsize=(8,8))
sns.barplot(data=target_values, x='tree', y='number')
fig_target0.legend(target_values.index)
# the pie chart
fig_target1 = plt.figure(figsize=(5,5))
plt.pie(target_values['number'], labels=target_values.index, autopct='%1.1f%%')

# shape and type after poping 'Cover_Type'
shapes['train'].append(train.shape)
shapes['test'].append(test.shape)
shapes['prep'].append('pop_target')
display(shapes)

# concat into a all data set
allData = pd.concat((train, test), axis=0, ignore_index=True)
display(allData)

# All dataset
# allData.info() # it shows the data type 
             # they are all int64
             # without any na

# check with df.dtypes
nonInt = []
for i in range(len(allData.columns)):
    if allData.dtypes[i] != "int":
        nonint.append(allData.dtypes.index[i])
print(f"NonInteger types in {str(nonInt)}")


# check with isna()
na = []
for i in range(len(allData.columns)):
    if allData.isna().sum()[i] != 0:
        na.append(allData.isna().sum().index[i])
print(f"NAN in {str(na)}")

shapes['train'].append(train.shape)
shapes['test'].append(test.shape)
shapes['prep'].append('concat')
display(shapes)

# fill in missing values
## no NAN

# encode string
## all is int dtype

# vis
fig_all, ax = plt.subplots(ncols=6, nrows=10, figsize=(15,15))
ax = ax.ravel()
plt.tight_layout()
for ind, ax in enumerate(ax):
    if ind < allData.shape[1]:
        ax.hist(allData.iloc[:, ind].values)
## from the hist plots, we could conclude...
print(f"continuous data are:\n {allData.columns[:10]}")
print('-'*40)
print(f"binary data are:\n {allData.columns[10:]}")

# feature normalization
for i in range(10):
    u = allData.iloc[:, i]
    u = (u - u.min(axis=0))/(u.max(axis=0) - u.min(axis=0) + 1e-7)
    allData.iloc[:, i] = u

fig_after, ax = plt.subplots(ncols=5, nrows=2, figsize=(15,10))
ax = ax.ravel()
plt.tight_layout()
for ind, ax in enumerate(ax):
    if ind < 10:
        ax.hist(allData.iloc[:, ind].values)

# pop features
# which are extreme data in train set
popped = pd.DataFrame() # stored the popped feature data

edge_train = []
for i in range(10, allData.shape[1]):
    if (train.iloc[:, i].value_counts()[0]) == train.shape[0]:
        edge_train.append(allData.columns[i])
print(f"The extreme data are:\n {edge_train}")

for i in edge_train:
    popped[i] = allData.pop(i)

display(allData.shape)

## which are extreme data in whole dataset
edge_all = []
for i in range(10, allData.shape[1]):
    if (allData.iloc[:, i].value_counts()[0]) == allData.shape[0]:
        edge_all.append(allData.columns[i])
print(f"The extreme data are:\n {edge_all}")

for i in edge_all:
    if i not in edge_train:
        popped[i] = allData.pop(i)

display(allData.shape)

## more feature engineering
corr_target = corrmat['Cover_Type'].drop(labels=['Cover_Type', 'Soil_Type7', 'Soil_Type15'])
display(corr_target)
corr_target = corr_target / corr_target.abs().max()
display(corr_target.sort_values())
# pop feature: Soil_Type30
popped['Soil_Type30'] = allData.pop('Soil_Type30')
display(allData.shape)

# target encoding 
## label encode first (1,2,3,4,5,6,7) to (0,1,2,3,4,5,6)
# create a sklearn encoder instance
le = preprocessing.LabelEncoder()
# train the encoder with target values
le.fit(train_target)
display(le.classes_) # show the raw categories
# transform the raw values
train_target = pd.DataFrame(le.transform(train_target))

# after label encoding, do the one-hot encoding
ohe = preprocessing.OneHotEncoder(sparse=False)
ohe.fit(train_target)
display(ohe.categories_)
train_target = pd.DataFrame(ohe.transform(train_target), columns=[1,2,3,4,5,6,7])
# train_target.rename(columns={0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7})

# restore the concatenated dataset
train = allData[:len(train)]
test = allData[len(train):]

shapes['train'].append(train.shape)
shapes['test'].append(test.shape)
shapes['prep'].append('restore concat')
display(shapes)

## train_test_split, split train data into train and valid
x_train, x_valid, y_train, y_valid = train_test_split(train.values,
                                                      train_target.values,
                                                      test_size=0.1,
                                                      random_state=40)

print('x_train shape:', x_train.shape, '\ny_train shape:', y_train.shape)
print('x_valid shape:', x_valid.shape, '\ny_valid shape:', y_valid.shape)

assert(x_train.dtype == 'float')
assert(x_valid.dtype == 'float')
assert(y_train.dtype == 'float')
assert(y_valid.dtype == 'float')

display(type(x_train), type(x_valid), type(y_train), type(y_valid))

# pd.DataFrame(data=x_train, columns=train.columns).to_csv(path_or_buf='./x_train.csv', index=False)
# pd.DataFrame(data=y_train, columns=train_target.columns).to_csv(path_or_buf='./y_train.csv', index=False)
# pd.DataFrame(data=x_valid, columns=train.columns).to_csv(path_or_buf='./x_valid.csv', index=False)
# pd.DataFrame(data=y_valid, columns=train_target.columns).to_csv(path_or_buf='./y_valid.csv', index=False)
# pd.DataFrame(data=test, columns=train.columns).to_csv(path_or_buf='./test.csv', index=False)


## define a model frame
'''1. code here for a loop inside the def
   2. code the layer def by hand'''
### subclass
class modelDL_0(tf.keras.Model):
    # instance with attributes
    def __init__(self):
        super(modelDL_0, self).__init__() # same as super().__init__()
        # build keras placeholders attributes
        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')
        self.dense2 = tf.keras.layers.Dense(units=32, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')
        self.dense3 = tf.keras.layers.Dense(units=16, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')
        self.dense4 = tf.keras.layers.Dense(units=4, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')
        self.dense5 = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax, kernel_initializer='glorot_uniform')
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, keras_input, training=False): # call the function with arg
        x = self.dense1(keras_input) # the keras_input should be a keras.Input() instance (keras placeholder)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense3(x)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense4(x)
        if training:
            x = self.dropout(x, training=training)
        return self.dense5(x) # the return should be a keras placeholder (a model placeholder)

### function API
def modelDL_1(feature_num): # number of the features
    keras_input = keras.Input(shape=(feature_num,))
    x = layers.Dense(units=64, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')(keras_input)
    x = layers.Dense(units=40, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')(x)
    x = layers.Dense(units=40, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')(x)
    x = layers.Dense(units=24, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')(x)
    x = layers.Dense(units=24, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')(x)
    x = layers.Dense(units=16, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')(x)
    x = layers.Dense(units=16, activation=tf.nn.leaky_relu, kernel_initializer='glorot_uniform')(x)
    x_output = layers.Dense(units=7, activation=tf.nn.softmax, kernel_initializer='glorot_uniform')(x)
    # return a model placeholder
    model = keras.Model(inputs=keras_input, outputs=x_output) # args both are keras placeholders

    display(model.summary())
    return model 


### sequential model frame 
def modelSeq(feature_num):
    model = keras.Sequential()
    model.add(keras.Input(shape=(feature_num,)))
    model.add(layers.Dense(64, activation=tf.nn.leaky_relu))
    model.add(layers.Dense(40, activation=tf.nn.leaky_relu))
    model.add(layers.Dense(40, activation=tf.nn.leaky_relu))
    model.add(layers.Dense(32, activation=tf.nn.leaky_relu))
    model.add(layers.Dense(32, activation=tf.nn.leaky_relu))
    model.add(layers.Dense(16, activation=tf.nn.leaky_relu))
    model.add(layers.Dense(4, activation=tf.nn.leaky_relu))
    model.add(layers.Dense(7, activation=tf.nn.softmax))

    display(model.summary())
    return model
# build the model
#keras_input = keras.Input(shape=(x_train.shape[-1],))
#model = modelDL_0()
#model(keras_input, training=True)

model = modelSeq(x_train.shape[-1])

# plot the model frame
# plot_model(model, to_file=<Str_Path>, show_shapes=<Bool>)
plot_model(model, to_file='./model.png', show_shapes=True)

# display the weights and biases
# model.get_weights()
# or 
# model.weights
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=50,
                    validation_data=(x_valid, y_valid))

train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']

# plot the losses
plt.plot(train_loss, 'b', label='train')
plt.plot(valid_loss, 'r', label='test')
plt.legend()
plt.title("Loss")
plt.show()

# plot the metrices, acc
plt.plot(train_acc, 'b', label='train')
plt.plot(valid_acc, 'r', label='test')
plt.legend()
plt.title("Accuracy")
plt.show()
pred = model.predict(test.values)
pred.shape, type(pred)
pred = ohe.inverse_transform(pred)
pred = pred.astype(int)
pred = np.squeeze(pred)
pred = le.inverse_transform(pred)
pred.shape, type(pred)
sub = pd.DataFrame() # an empty df
sub['Id'] = test_id
sub['Class'] = pred
sub.to_csv('submission.csv',index=False) # use the set 'Id'
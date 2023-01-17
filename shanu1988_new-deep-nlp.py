import pandas as pd

import numpy as np



# imports

%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns

from scipy import signal

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, plot_confusion_matrix

from keras.models import Model

import keras.layers as L

from keras.utils import to_categorical, plot_model
train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

test = pd.read_csv('../input/liverpool-ion-switching/test.csv')

print('The shape of train data:', train.shape)

print('The shape of test data:', test.shape)      
#Check the heads of data

train.head()
%%time

plt.figure(figsize=(15,6))

#scatter with positive examples

plt.scatter(train.time[train.open_channels==1],

           train.signal[train.open_channels==1],

           c='salmon')



plt.scatter(train.time[train.open_channels==2],

           train.signal[train.open_channels==2],

           c='red')



plt.scatter(train.time[train.open_channels==3],

           train.signal[train.open_channels==3],

           c='green')



plt.scatter(train.time[train.open_channels==4],

           train.signal[train.open_channels==4],

           c='blue')



plt.scatter(train.time[train.open_channels==5],

           train.signal[train.open_channels==5],

           c='black')



plt.scatter(train.time[train.open_channels==6],

           train.signal[train.open_channels==6],

           c='pink')



plt.scatter(train.time[train.open_channels==7],

           train.signal[train.open_channels==7],

           c='lightgreen')



plt.scatter(train.time[train.open_channels==8],

           train.signal[train.open_channels==8],

           c='yellow')



plt.scatter(train.time[train.open_channels==9],

           train.signal[train.open_channels==9],

           c='orange')



plt.scatter(train.time[train.open_channels==10],

           train.signal[train.open_channels==10],

           c='brown')







# scatter with nagative examples

plt.scatter(train.time[train.open_channels==0],

           train.signal[train.open_channels==0],

           c='lightblue')





# add some usefull information

plt.title('Time in function of Signal')

plt.xlabel('Time')

plt.ylabel("Signal")

plt.legend();
# create the function

def calc_gradients(s, n_grads=4):

    '''

    Calculate the gradients for pandas series. Returns the same number of sampels

    '''

    grads = pd.DataFrame()

    g = s.values

    for i in range(n_grads):

        g = np.gradient(g)

        grads['grad_'+str(i+1)] = g

        

    return grads

def calc_low_pass(s, n_filters=10):

    '''

    Applies low pass filters to the signal. Left delayed and no delayed

    '''

    wns = np.logspace(-2,-0.3, n_filters)

    

    low_pass = pd.DataFrame()

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='low')

        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, s.values)

        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, s.values)

        

    return low_pass

def calc_high_pass(s, n_filters=10):

    '''

    Applies high pass filters to the signal. Left delayed and delayed

    '''

    wns = np.logspace(-2, -0.1, n_filters)

    

    high_pass = pd.DataFrame()

    for wn in wns:

        b, a = signal.butter(1, Wn=wn, btype='high')

        high_pass['highpasslf_' + str('%4f' %wn)] = signal.lfilter(b, a, s.values)

        high_pass['highpassff_' + str('%4f' %wn)] = signal.filtfilt(b, a, s.values)

    return high_pass

def calc_roll_stats(s, windows=[10, 50, 100, 500, 1000]):

    '''

    Calculating rolling stats like mean std, min, max

    '''

    

    roll_stats = pd.DataFrame()

    for window in windows:

        roll_stats['roll_mean_' + str(window)] = s.rolling(window=window, min_periods=1).mean()

        roll_stats['roll_std_'+ str(window)] = s.rolling(window=window, min_periods=1).std()

        roll_stats['roll_min_'+ str(window)] = s.rolling(window=window, min_periods=1).min()

        roll_stats['roll_max_'+ str(window)] = s.rolling(window=window, min_periods=1).max()

        roll_stats['roll_range_'+ str(window)] = roll_stats['roll_max_'+ str(window)] - roll_stats['roll_min_'+ str(window)]

        roll_stats['roll_q10_'+ str(window)] = s.rolling(window=window, min_periods=1).quantile(0.10)

        roll_stats['roll_q25_'+ str(window)] = s.rolling(window=window, min_periods=1).quantile(0.25)

        roll_stats['roll_q50_'+ str(window)] = s.rolling(window=window, min_periods=1).quantile(0.50)

        roll_stats['roll_q75_'+ str(window)] = s.rolling(window=window, min_periods=1).quantile(0.75)

        roll_stats['roll_q90_'+ str(window)] = s.rolling(window=window, min_periods=1).quantile(0.90)

        

    # Let's add zero when creates na value (std)

    roll_stats = roll_stats.fillna(value=0)

    return roll_stats

def calc_ewm(s, windows=[10, 50, 100, 500, 1000]):

    '''

    calculates exponential weight function

    '''

    ewm = pd.DataFrame()

    for window in windows:

        ewm['ewm_mean_'+ str(window)] = s.ewm(span=window, min_periods=1).mean()

        ewm['ewm_std_'+ str(window)] = s.ewm(span=window, min_periods=1).std()

        

        # Add zeros when creates na value(std)

    ewm = ewm.fillna(value=0)

    return ewm

def add_featurs(s):

    '''

    Keep all calculation together

    '''

    gradients = calc_gradients(s)

    low_pass = calc_low_pass(s)

    high_pass = calc_high_pass(s)

    roll_stats = calc_roll_stats(s)

    ewm = calc_ewm(s)

    

    return pd.concat([s, gradients, low_pass, high_pass, roll_stats, ewm], axis=1)





def divide_and_add_featurs(s, signal_size = 500000):

    '''

    Divide the signal in the bags of 'signal_size'

    Normalize the data dividing it by 15.0    

    

    '''

    # normalize

    s = s/15.0

    

    ls = []

    for i in tqdm(range(int(s.shape[0]/signal_size))):

        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)

        sig_featured = add_featurs(sig)

        ls.append(sig_featured)

        

    return pd.concat(ls, axis=0)    



%%time

# Let's apply every feature to the train data

df = divide_and_add_featurs(train['signal'])

df.shape
df.head()
# X_train, X_valid, y_train, y_valid = train_test_split(df.values, train['open_channels'].values, test_size=0.2)



# X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
"""def create_mlp(shape):

    

    

    X_input = L.Input(shape)

    X = L.Dense(350, activation='relu')(X_input)

    X = L.Dense(300, activation='relu')(X)

    X = L.Dense(200, activation='relu')(X)

    X = L.Dense(100, activation='relu')(X)

    X = L.Dense(125, activation='relu')(X)

    X = L.Dense(75, activation='relu')(X)

    X = L.Dense(25, activation='relu')(X)

    X = L.Dense(11, activation='softmax')(X)

    

    

    model = Model(inputs=X_input, outputs=X)

    

    return model

    

    

mlp = create_mlp(X_train[0].shape)

mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

print(mlp.summary())"""
"""def get_class_weight(classes):

    '''

    Weight of the class is inversely proportional to the population of the class

    '''

    hist, _ = np.histogram(classes, bins=np.arange(12)-0.5)

    class_weight = hist.sum()/hist

    

    return class_weight



class_weight = get_class_weight(y_train)"""
# %%time

# # fit the model on train data

# mlp.fit(x = X_train, y=y_train, epochs=30, batch_size=800, class_weight=class_weight)
"""figsize=(10, 15)

plt.figure(1)

plt.plot(mlp.history.history['loss'], 'b', label='loss')

plt.xlabel('epochs')

plt.legend()

plt.figure(2)

plt.plot(mlp.history.history['sparse_categorical_accuracy'], 'g', label='sparse_categorical_accuracy')

plt.xlabel('epochs')

plt.legend();"""
"""# prediction on validation set

y_pred = mlp.predict(X_valid)

y_pred = np.argmax(y_pred, axis=-1)

pd.Series(y_pred).value_counts()"""
"""# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

def plot_cm(y_true, y_pred, title):

    figsize=(25,15)

    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual open channels'

    cm.columns.name = 'Predicted open channels'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap='YlGnBu', annot=annot, fmt='', ax=ax)



# f1 score

f1 = f1_score(y_valid, y_pred, average='macro')



# plot confusion matrix

plot_cm(y_valid, y_pred, 'MLP f1_score=' + str('%.4f' %f1))"""
"""# create test DataFrame as X_test

X_test = divide_and_add_featurs(test['signal'])

print('X_test.shape=', X_test.shape)

X_test.head()"""

"""#Let's predict test dataset

y_test = mlp.predict(X_test)

y_test = np.argmax(y_test, axis=-1)

pd.Series(y_test).value_counts()"""
"""# create submission

submission = pd.DataFrame()

submission['time'] = test['time']

submission['open_channels'] = y_test



# write file

submission.to_csv('final_if_imp.csv', index=False, float_format='%.4f')"""
import tensorflow as tf
X_train, X_valid, y_train, y_valid = train_test_split(df.values, train['open_channels'].values, test_size=0.2)



X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
def create_mlp(shape):

    

    

    X_input = L.Input(shape)

    X = L.Dense(1536, activation='relu')(X_input)

    X = L.Dense(1024, activation='relu')(X)

    X = L.Dense(512, activation='relu')(X)

    X = L.Dense(11, activation='softmax')(X)

        

    model = Model(inputs=X_input, outputs=X)

    

    return model

    

mlp = create_mlp(X_train[0].shape)

# mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

# print(mlp.summary())    





optimizer=tf.keras.optimizers.Adam(lr=0.001)

mlp.compile(optimizer='adam',

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

mlp.summary()    
def get_class_weight(classes):

    '''

    Weight of the class is inversely proportional to the population of the class

    '''

    hist, _ = np.histogram(classes, bins=np.arange(12)-0.5)

    class_weight = hist.sum()/hist

    

    return class_weight



class_weight = get_class_weight(y_train)
%%time

# fit the model on train data

mlp.fit(x = X_train, y=y_train, epochs=20, batch_size=1536,validation_data = (X_valid, y_valid), class_weight=class_weight,callbacks = [reduce_lr])
plt.plot(mlp.history.history['accuracy'])

plt.plot(mlp.history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['Train','Valid'])

plt.show();
plt.plot(mlp.history.history['loss'])

plt.plot(mlp.history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epochs')

plt.ylabel('loss')

plt.legend(['Train', 'Valid'])

plt.show();
# prediction on validation set

y_pred = mlp.predict(X_valid)

y_pred = np.argmax(y_pred, axis=-1)

pd.Series(y_pred).value_counts()
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

def plot_cm(y_true, y_pred, title):

    figsize=(25,15)

    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual open channels'

    cm.columns.name = 'Predicted open channels'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap='YlGnBu', annot=annot, fmt='', ax=ax)



# f1 score

f1 = f1_score(y_valid, y_pred, average='macro')



# plot confusion matrix

plot_cm(y_valid, y_pred, 'MLP f1_score=' + str('%.4f' %f1))
# create test DataFrame as X_test

X_test = divide_and_add_featurs(test['signal'])

print('X_test.shape=', X_test.shape)

X_test.head()

#Let's predict test dataset

y_test = mlp.predict(X_test)

y_test = np.argmax(y_test, axis=-1)

pd.Series(y_test).value_counts()
# create submission

submission = pd.DataFrame()

submission['time'] = test['time']

submission['open_channels'] = y_test



# write file

submission.to_csv('final3_if_imp.csv', index=False, float_format='%.4f')
# Began from the following starter kernel, https://www.kaggle.com/johnnyd113/baseline-with-explanations-how-to-get-started

#    processed the data a little differently and utilized a different benchmark

# Added some additional metrics from the following kernel, https://www.kaggle.com/harshel7/earthquake-prediction-ensemble-nn

#    used a deep NN for a benchmark

# optimizers = https://towardsdatascience.com/preventing-deep-neural-network-from-overfitting-953458db800a

# grid search method = https://blogs.oracle.com/meena/simple-neural-network-model-using-keras-and-grid-search-hyperparameterstuning



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
traindata = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv',

                    dtype={'acoustic_data': np.int16,

                           'time_to_failure': np.float64}) 

print(traindata.shape)
# Smaller sample set of training data 



#traindata = pd.read_csv('../input/test-sample/train_sample.csv',

#                    dtype={'acoustic_data': np.int16,

#                           'time_to_failure': np.float64}) 

#print(traindata.shape)
# Loop to find all earthquake events and print the index of the beginning of the next data segment



#prev = 0

#index = 0

#event_index = []

#for index in range(traindata.shape[0]):

#    row = traindata.iloc[index]

#    if (row[1] - prev) >= 0.1: # Check for jump in the values to know where the earthquake event stops

#        event_index.append(index)

#        print('')

#        print(index)

#    if (index % 100000) == 0:

#        print('*', end='') # Each * represents 100,000 data points, print this to know the loop is running

#    prev = row[1] 

#Visualize the data

import matplotlib.pyplot as plt



#

# Figure 1 - plot all data

##########################

x = pd.Series(traindata['acoustic_data'].values[::50])

y = pd.Series(traindata['time_to_failure'].values[::50])



# source for plotting, https://matplotlib.org/gallery/api/two_scales.html

plt.figure(1)

fig, ax1 = plt.subplots(figsize=(14, 7))



color = 'tab:red'

ax1.set_xlabel('data (#)')

ax1.set_ylabel('input (acoustic data)', color=color)

ax1.plot(x, color=color)

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylabel('output time (s)', color=color)  # we already handled the x-label with ax1

ax2.plot(y, color=color)

ax2.tick_params(axis='y', labelcolor=color)



plt.title("All Data")

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show() #all data



#

# Figure 2 - zoom in to 4th event

######################################

x = pd.Series(traindata['acoustic_data'].values[104677356:139772453:20])

y = pd.Series(traindata['time_to_failure'].values[104677356:139772453:20])



plt.figure(2)

fig, ax1 = plt.subplots(figsize=(14, 7))



color = 'tab:red'

ax1.set_xlabel('data (#)')

ax1.set_ylabel('input (acoustic data)', color=color)

ax1.plot(x, color=color)

#ax1.plot(np.abs(x), color=color)

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylabel('output time (s)', color=color)  # we already handled the x-label with ax1

ax2.plot(y, color=color)

ax2.tick_params(axis='y', labelcolor=color)



plt.title("Zoom to 4th event")

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show() #all data
#store and print out all of the earthquake events from the training data

#Data splits will be created from these event points

seg_size = 150000



event_index=[5656574,

50085878,

104677356,

138772453,

187641820,

218652630,

245829585,

307838917,

338276287,

375377848,

419368880,

461811623,

495800225,

528777115,

585568144,

621985673]



index = 0



# update the event_index with the earthquake events instead of the beginnings

for eq in event_index:

    event_index[index] = eq - 1

    index = index + 1

# append the final earthquake event

event_index.append(traindata.shape[0])

# event_index now contains all earthquake events

print(event_index)



# process the data into equal segments starting from each earthquake

index = 0

event_index_segments = []

for eq in event_index:

    if index == 0:

        event_index_segments.append(int(eq / seg_size))

    else :

        event_index_segments.append(int((eq - event_index[index-1]) / seg_size))

    index = index + 1



print(event_index_segments)
# Feature for training data

cols = ['mean','median','std','max','skew',

        'min','sum','var','ptp','mean_change_abs','max_to_min','max_to_min_diff',

        'abs_avg','abs_std','abs_max','abs_min','abs_med','abs_skew','abs_sum','abs_var','abs_ptp',

        '10p','25p','50p','75p','90p','abs_1p','abs_5p','abs_30p','abs_60p','abs_95p','abs_99p',

#        'mean_first_10000','mean_last_10000','mean_first_50000','mean_last_50000',

#        'std_first_10000','std_last_10000','std_first_50000','std_last_50000'

       ]



# function for condensing segments into statistical data

def condense_segment(segment_, x_condensed, y_label, loc_):        

    #Converts to numpy array

    x = pd.Series(segment_['acoustic_data'].values)



    #Grabs the final 'time_to_failure' value

    y = segment_['time_to_failure'].values[-1]

    y_label.loc[loc_, 'time_to_failure'] = y



    #For every 150,000 rows, we make these calculations

    x_condensed.loc[loc_, 'mean'] = np.mean(x)

    x_condensed.loc[loc_, 'median'] = np.median(x)

    x_condensed.loc[loc_, 'std'] = np.std(x)

    x_condensed.loc[loc_, 'max'] = np.max(x)

    x_condensed.loc[loc_, 'skew'] = x.skew()

    x_condensed.loc[loc_, 'min'] = np.min(x)

    x_condensed.loc[loc_, 'sum'] = np.sum(x)

    x_condensed.loc[loc_, 'var'] = np.var(x)

    x_condensed.loc[loc_, 'ptp'] = np.ptp(x) #Peak-to-peak is like range

    x_condensed.loc[loc_, 'mean_change_abs'] = np.mean(np.diff(x))

    x_condensed.loc[loc_, 'max_to_min'] = np.max(x) / np.abs(np.min(x))

    x_condensed.loc[loc_, 'max_to_min_diff'] = np.max(x) - np.abs(np.min(x))



    x_condensed.loc[loc_, 'abs_avg'] = np.abs(x).mean()

    x_condensed.loc[loc_, 'abs_std'] = np.abs(x).std()

    x_condensed.loc[loc_, 'abs_max'] = np.abs(x).max()

    x_condensed.loc[loc_, 'abs_min'] = np.abs(x).min()

    x_condensed.loc[loc_, 'abs_med'] = np.abs(x).median()

    x_condensed.loc[loc_, 'abs_skew'] = np.abs(x).skew()

    x_condensed.loc[loc_, 'abs_sum'] = np.abs(x).sum()

    x_condensed.loc[loc_, 'abs_var'] = np.abs(x).var()

    x_condensed.loc[loc_, 'abs_ptp'] = np.abs(x).ptp() #Peak-to-peak is like range



    x_condensed.loc[loc_, '10p'] = np.percentile(x,q=10) 

    x_condensed.loc[loc_, '25p'] = np.percentile(x,q=25) #We can also grab percentiles

    x_condensed.loc[loc_, '50p'] = np.percentile(x,q=50)

    x_condensed.loc[loc_, '75p'] = np.percentile(x,q=75)

    x_condensed.loc[loc_, '90p'] = np.percentile(x,q=90)   



    x_condensed.loc[loc_, 'abs_1p'] = np.percentile(x, np.abs(0.01))

    x_condensed.loc[loc_, 'abs_5p'] = np.percentile(x, np.abs(0.05))

    x_condensed.loc[loc_, 'abs_30p'] = np.percentile(x, np.abs(0.30))

    x_condensed.loc[loc_, 'abs_60p'] = np.percentile(x, np.abs(0.60))

    x_condensed.loc[loc_, 'abs_95p'] = np.percentile(x, np.abs(0.95))

    x_condensed.loc[loc_, 'abs_99p'] = np.percentile(x, np.abs(0.99))



#    x_condensed.loc[loc_, 'mean_first_10000'] = x[:10000].mean()

#    x_condensed.loc[loc_, 'mean_last_10000']  =  x[-10000:].mean()

#    x_condensed.loc[loc_, 'mean_first_50000'] = x[:50000].mean()

#    x_condensed.loc[loc_, 'mean_last_50000'] = x[-50000:].mean()



#    x_condensed.loc[loc_, 'std_first_10000'] = x[:10000].std()

#    x_condensed.loc[loc_, 'std_last_10000']  =  x[-10000:].std()

#    x_condensed.loc[loc_, 'std_first_50000'] = x[:50000].std()

#    x_condensed.loc[loc_, 'std_last_50000'] = x[-50000:].std()
# Dataframe for condensed training data

total_segments = sum(event_index_segments)

print("Total Segments = " + str(total_segments))

X_train = pd.DataFrame(index=range(total_segments), dtype=np.float64, columns=cols) # Feature list

y_train = pd.DataFrame(index=range(total_segments), dtype=np.float64, columns=['time_to_failure']) #Our target variable



# Condense the segments into the training dataframe

index = 0

i_total = 0 # i counter restarts at 0 for each event_index_segment iteration, keep track of total segments

p_done = 0

print("Data condensing...")

for segments in event_index_segments: # Load total segments available for this event

    event_end = event_index[index]  # Earthquake index in the training data

    index = index + 1



    for i in range(segments):

        # Creating segments backwards from the earthquake events

        # Start of segment = event_end - ((i+1)*seg_size)

        # End of segment = event_end - (i*seg_size)        

        sample_segment = traindata.iloc[event_end - ((i+1)*seg_size):event_end - (i*seg_size)]

        condense_segment(sample_segment, X_train, y_train, i_total)

        # Print status %

        i_total = i_total + 1

        if (i_total % int(total_segments/10)) == 0:

            p_done = p_done + 1

            print(str(p_done) + "0%\r", end='') # Each * represents 400 data segments



print(i_total)
# Shift and create addiitional alternate training sets, subtracting 1 segment from each event because of shifting

total_segments = total_segments - len(event_index_segments)

print("Alt Total Segments = " + str(total_segments))

X_train_alt1 = pd.DataFrame(index=range(total_segments), dtype=np.float64, columns=cols) # Feature list

y_train_alt1 = pd.DataFrame(index=range(total_segments), dtype=np.float64, columns=['time_to_failure']) #Our target variable

X_train_alt2 = pd.DataFrame(index=range(total_segments), dtype=np.float64, columns=cols) # Feature list

y_train_alt2 = pd.DataFrame(index=range(total_segments), dtype=np.float64, columns=['time_to_failure']) #Our target variable

X_train_alt3 = pd.DataFrame(index=range(total_segments), dtype=np.float64, columns=cols) # Feature list

y_train_alt3 = pd.DataFrame(index=range(total_segments), dtype=np.float64, columns=['time_to_failure']) #Our target variable



#

# Condense alt 1 dataframe (offset 50,0000)

##########

index = 0

i_total = 0 # i counter restarts at 0 for each event_index_segment iteration, keep track of total segments

p_done = 0

offset = 37500

print("Alt1 condensing...  offset=" + str(offset))

for segments in event_index_segments: # Load total segments available for this event

    event_end = event_index[index] - offset  # Earthquake index in the training data

    index = index + 1

    segments = segments - 1 # decrement 1 since we are shifting into the last segment



    for i in range(segments):   

        sample_segment = traindata.iloc[event_end - ((i+1)*seg_size):event_end - (i*seg_size)]

        condense_segment(sample_segment, X_train_alt1, y_train_alt1, i_total)

        # Print status %

        i_total = i_total + 1

        if (i_total % int(total_segments/10)) == 0:

            p_done = p_done + 1

            print(str(p_done) + "0%\r", end='') # Each * represents 400 data segments



print(i_total)



#

# Condense alt 2 dataframe (offset 100,0000)

##########

index = 0

i_total = 0 # i counter restarts at 0 for each event_index_segment iteration, keep track of total segments

p_done = 0

offset = 37500 * 2

print("Alt2 condensing...  offset=" + str(offset))

for segments in event_index_segments: # Load total segments available for this event

    event_end = event_index[index] - offset  # Earthquake index in the training data

    index = index + 1

    segments = segments - 1



    for i in range(segments):    

        sample_segment = traindata.iloc[event_end - ((i+1)*seg_size):event_end - (i*seg_size)]

        condense_segment(sample_segment, X_train_alt2, y_train_alt2, i_total)

        # Print status %

        i_total = i_total + 1

        if (i_total % int(total_segments/10)) == 0:

            p_done = p_done + 1

            print(str(p_done) + "0%\r", end='') # Each * represents 400 data segments



print(i_total)



#

# Condense alt 3 dataframe (offset 100,0000)

##########

index = 0

i_total = 0 # i counter restarts at 0 for each event_index_segment iteration, keep track of total segments

p_done = 0

offset = 37500 * 3

print("Alt3 condensing...  offset=" + str(offset))

for segments in event_index_segments: # Load total segments available for this event

    event_end = event_index[index] - offset  # Earthquake index in the training data

    index = index + 1

    segments = segments - 1



    for i in range(segments):    

        sample_segment = traindata.iloc[event_end - ((i+1)*seg_size):event_end - (i*seg_size)]

        condense_segment(sample_segment, X_train_alt3, y_train_alt3, i_total)

        # Print status %

        i_total = i_total + 1

        if (i_total % int(total_segments/10)) == 0:

            p_done = p_done + 1

            print(str(p_done) + "0%\r", end='') # Each * represents 400 data segments



print(i_total)
X_train.head(5)
X_train_alt1.head(5)
X_train_alt2.head(5)
X_train_alt3.head(5)
#Visualize the data

import matplotlib.pyplot as plt



#

# Figure 4 - condensed data visualization

######################################



plt.figure(4)

fig, ax1 = plt.subplots(figsize=(14, 7))



color = 'tab:red'

ax1.set_xlabel('data (#)')

ax1.set_ylabel('input (acoustic abs mean)', color=color)

ax1.plot(X_train['abs_avg'].values[0:2000], color=color)

ax1.plot(X_train_alt1['abs_avg'].values[0:2000], color='green')

ax1.plot(X_train_alt2['abs_avg'].values[0:2000], color='yellow')

ax1.plot(X_train_alt3['abs_avg'].values[0:2000], color='orange')

#ax1.plot(np.abs(x), color=color)

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylabel('output time (s)', color=color)  # we already handled the x-label with ax1

ax2.plot(y_train.values[0:2000], color=color)

ax2.plot(y_train_alt1.values[0:2000], color='cyan')

ax2.plot(y_train_alt2.values[0:2000], color='magenta')

ax2.plot(y_train_alt3.values[0:2000], color='black')

ax2.tick_params(axis='y', labelcolor=color)



plt.title("1ST 2000 of Condensed Data - ABS Mean (all 3 data structures)")

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show() #all data

# Split the training data for additional tests to tune performance

from sklearn.model_selection import train_test_split



X_train_s, X_test, y_train_s, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

X_train_s, X_val, y_train_s, y_val = train_test_split(X_train_s, y_train_s, test_size=0.2, random_state=1)



#Alt1 split

X_train_alt1_s, X_test_alt1, y_train_alt1_s, y_test_alt1 = train_test_split(X_train_alt1, y_train_alt1, test_size=0.2, random_state=1)

X_train_alt1_s, X_val_alt1, y_train_alt1_s, y_val_alt1 = train_test_split(X_train_alt1_s, y_train_alt1_s, test_size=0.2, random_state=1)

#Alt2 split

X_train_alt2_s, X_test_alt2, y_train_alt2_s, y_test_alt2 = train_test_split(X_train_alt2, y_train_alt2, test_size=0.2, random_state=1)

X_train_alt2_s, X_val_alt2, y_train_alt2_s, y_val_alt2 = train_test_split(X_train_alt2_s, y_train_alt2_s, test_size=0.2, random_state=1)

#Alt3 split

X_train_alt3_s, X_test_alt3, y_train_alt3_s, y_test_alt3 = train_test_split(X_train_alt3, y_train_alt3, test_size=0.2, random_state=1)

X_train_alt3_s, X_val_alt3, y_train_alt3_s, y_val_alt3 = train_test_split(X_train_alt3_s, y_train_alt3_s, test_size=0.2, random_state=1)
### Example Benchmark from the Starter Kernel ###

#Fit a random forest

#from sklearn.ensemble import RandomForestRegressor



#This creates the Randomforest with the given parameters

#rf = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)

#                          max_features=0.5, #Max number of features each tree can use 

#                          min_samples_leaf=30, #Min amount of samples in each leaf

#                          random_state=11)



#This trains the random forest on our training data

#rf.fit(X_train,y_train)



### Score the Example Benchmark

#from sklearn.metrics import mean_absolute_error



#mean_absolute_error(y_val, rf.predict(X_val))

# Begin creation of benchmark Simple Deep NN

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation



nn_bm_model = Sequential()

nn_bm_model.add(Dense(128, activation='relu', input_shape=(X_train_s.shape[1],)))

nn_bm_model.add(Dropout(.2))

nn_bm_model.add(Dense(64, activation='relu'))

nn_bm_model.add(Dropout(.1))

nn_bm_model.add(Dense(1, activation='linear'))



# TODO: Compile the model using a loss function and an optimizer.

nn_bm_model.compile(loss = 'mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

nn_bm_model.summary()
nn_bm_model.fit(X_train_s, y_train_s, epochs=100, batch_size=20, verbose=1)
score = nn_bm_model.evaluate(X_test, y_test, verbose=0)

print("        " + nn_bm_model.metrics_names[0] + "                " + nn_bm_model.metrics_names[1])

print("Test: ", score)

score = nn_bm_model.evaluate(X_val, y_val, verbose=0)

print("Val : ", score)

score = nn_bm_model.evaluate(X_test_alt1, y_test_alt1, verbose=0)

print("Test_alt1 : ", score)

score = nn_bm_model.evaluate(X_val_alt1, y_val_alt1, verbose=0)

print("Val_alt1  : ", score)

score = nn_bm_model.evaluate(X_test_alt2, y_test_alt2, verbose=0)

print("Test_alt2 : ", score)

score = nn_bm_model.evaluate(X_val_alt2, y_val_alt2, verbose=0)

print("Val_alt2  : ", score)

score = nn_bm_model.evaluate(X_test_alt3, y_test_alt3, verbose=0)

print("Test_alt3 : ", score)

score = nn_bm_model.evaluate(X_val_alt3, y_val_alt3, verbose=0)

print("Val_alt3  : ", score)
# Utilize Grid Search to attempt to tune paramters...

from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, BatchNormalization

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV



# Create Tunable Model

def nn_model_(dropout_rate=0.2,optimizer='adam',activation='relu',regularizer=0.0):

    #default parameters

    

    #kernel_regularizer=regularizers.l1(0.)

    model = Sequential()

    model.add(Dense(128, activation=activation, input_shape=(X_train_s.shape[1],)))

    nn_model.add(BatchNormalization()) # added batch normalization to normalize data between layers

    model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation=activation))

    nn_model.add(BatchNormalization())

    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])

    return model



model = KerasClassifier(build_fn=nn_model_,epochs=100, batch_size=20)

# parameters

optimizer_=['adam','adamax'] ### best - adam

activation_=['relu','tanh'] ### best - relu

epochs_=[80,100,120]

dropout_rate_=[0.2,0.3,0.4,0.5,0.6] 

batch_size_=[20,32,48] 

regularizer=[0.0,0.0001,0.001,0.01,0.1]

param_grid1 = dict(batch_size=batch_size_,dropout_rate=dropout_rate_)

param_grid2 = dict(activation=activation_,optimizer=optimizer_)

param_grid3 = dict(epochs=epochs_)

#grid = GridSearchCV(estimator=model,param_grid=param_grid3,scoring='neg_mean_absolute_error',n_jobs=-1)

#grid_result = grid.fit(X_train_s, y_train_s)
# Print results of grid search

#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



### 128-64 nodes ###

# param_grid1 results 

# Best: -5.664138 using {'batch_size': 20, 'dropout_rate': 0.2}

##

# param_grid2 results (batch_size, dropout_rate set from previous run)

# Best: -5.664138 using {'activation': 'relu', 'optimizer': 'adam'}

##

# param_grid3 results (L1 regularizer)

# Best: -5.664138 using {'regularizer': 0.0}

##

# param_grid4 results (L2 regularizer)

# Best: -5.664138 using {'regularizer': 0.0}



### 512-256 nodes ###

# param_grid1 results 

# Best: -5.635036 using {'batch_size': 20, 'dropout_rate': 0.2}

##

# param_grid2 results (batch_size, dropout_rate set from previous run)

# Best: -5.635036 using {'activation': 'relu', 'optimizer': 'adam'}

##

# param_grid3 results (epochs)

# Best: -5.635036 using {'epochs': 80}
# Final network with tuned hyperparameters...

from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, BatchNormalization

from numpy.random import seed

seed(7)

from tensorflow import set_random_seed

set_random_seed(77)



#

# Model Hyperparamters

#===================

dropout_rate = 0.20    # 0.2, 0.15, 0.1

activation = 'relu'

optimizer = 'adam'

epochs = 80      # 80, 100, 250

batch_size = 20  # 20, 15

l1r = 0.1        # 0.1, 0.15, 0.20



# at default, 2 fits seems to do the trick

# dropout=0.15, 1 fit gave 2.4, 3rd fit 2.5, looks very good

# dropout=0.1, 3rd fit dropped to 2.4



# 3-layer Deep NN with batch normalization and dropout

nn_model = Sequential()

nn_model.add(Dense(64, activation=activation, input_shape=(X_train_s.shape[1],),kernel_regularizer=regularizers.l2(l1r)))

nn_model.add(BatchNormalization())

nn_model.add(Dropout(dropout_rate))

nn_model.add(Dense(32, activation=activation))

nn_model.add(BatchNormalization())

nn_model.add(Dropout(dropout_rate))

nn_model.add(Dense(1, activation='linear'))



# TODO: Compile the model using a loss function and an optimizer.

nn_model.compile(loss = 'mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])

nn_model.summary()
nn_model.fit(X_train_s, y_train_s, epochs=epochs, batch_size=batch_size, verbose=1)
# Visualize predictions

y_pred = nn_model.predict(X_train)

#y_pred_bm = nn_bm_model.predict(X_train)



#

# Figure 6 - Condensed Training Data Final Prediction

######################################



plt.figure(6)

fig, ax1 = plt.subplots(figsize=(14, 7))



color = 'tab:red'

ax1.set_ylim(0,10)

ax1.set_xlabel('data (#)')

ax1.set_ylabel('pred output (s)', color=color)

ax1.plot(y_pred, color=color)

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylim(0,10)

ax2.set_ylabel('output time (s)', color=color)  # we already handled the x-label with ax1

ax2.plot(y_train.values, color=color)

ax2.tick_params(axis='y', labelcolor=color)



plt.title("Predicted Values for Final Model")

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show() #all data

print("              " + nn_model.metrics_names[0] + "                " + nn_model.metrics_names[1])

score = nn_model.evaluate(X_test, y_test, verbose=0)

print("Test_s      : ", score)

score = nn_model.evaluate(X_val, y_val, verbose=0)

print("Val_s       : ", score)

score = nn_model.evaluate(X_test_alt1, y_test_alt1, verbose=0)

print("Test_alt1 : ", score)

score = nn_model.evaluate(X_val_alt1, y_val_alt1, verbose=0)

print("Val_alt1  : ", score)

score = nn_model.evaluate(X_test_alt2, y_test_alt2, verbose=0)

print("Test_alt2 : ", score)

score = nn_model.evaluate(X_val_alt2, y_val_alt2, verbose=0)

print("Val_alt2  : ", score)

score = nn_model.evaluate(X_test_alt3, y_test_alt3, verbose=0)

print("Test_alt3 : ", score)

score = nn_model.evaluate(X_val_alt3, y_val_alt3, verbose=0)

print("Val_alt3  : ", score)
nn_model.fit(X_train_alt1_s, y_train_alt1_s, epochs=epochs, batch_size=batch_size, verbose=1)
# Visualize predictions

y_pred = nn_model.predict(X_train)

#y_pred_bm = nn_bm_model.predict(X_train)



#

# Figure 6 - Condensed Training Data Final Prediction

######################################



plt.figure(6)

fig, ax1 = plt.subplots(figsize=(14, 7))



color = 'tab:red'

ax1.set_ylim(0,10)

ax1.set_xlabel('data (#)')

ax1.set_ylabel('pred output (s)', color=color)

ax1.plot(y_pred, color=color)

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylim(0,10)

ax2.set_ylabel('output time (s)', color=color)  # we already handled the x-label with ax1

ax2.plot(y_train.values, color=color)

ax2.tick_params(axis='y', labelcolor=color)



plt.title("Predicted Values for Final Model - 2nd fit")

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show() #all data#
print("              " + nn_model.metrics_names[0] + "                " + nn_model.metrics_names[1])

score = nn_model.evaluate(X_test, y_test, verbose=0)

print("Test_s      : ", score)

score = nn_model.evaluate(X_val, y_val, verbose=0)

print("Val_s       : ", score)

score = nn_model.evaluate(X_test_alt1, y_test_alt1, verbose=0)

print("Test_alt1 : ", score)

score = nn_model.evaluate(X_val_alt1, y_val_alt1, verbose=0)

print("Val_alt1  : ", score)

score = nn_model.evaluate(X_test_alt2, y_test_alt2, verbose=0)

print("Test_alt2 : ", score)

score = nn_model.evaluate(X_val_alt2, y_val_alt2, verbose=0)

print("Val_alt2  : ", score)

score = nn_model.evaluate(X_test_alt3, y_test_alt3, verbose=0)

print("Test_alt3 : ", score)

score = nn_model.evaluate(X_val_alt3, y_val_alt3, verbose=0)

print("Val_alt3  : ", score)
nn_model.fit(X_train_alt2_s, y_train_alt2_s, epochs=epochs, batch_size=batch_size, verbose=1)
# Visualize predictions

y_pred = nn_model.predict(X_train)

#y_pred_bm = nn_bm_model.predict(X_train)



#

# Figure 6 - Condensed Training Data Final Prediction

######################################



plt.figure(6)

fig, ax1 = plt.subplots(figsize=(14, 7))



color = 'tab:red'

ax1.set_ylim(0,10)

ax1.set_xlabel('data (#)')

ax1.set_ylabel('pred output (s)', color=color)

ax1.plot(y_pred, color=color)

ax1.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylim(0,10)

ax2.set_ylabel('output time (s)', color=color)  # we already handled the x-label with ax1

ax2.plot(y_train.values, color=color)

ax2.tick_params(axis='y', labelcolor=color)



plt.title("Predicted Values for Final Model - 3rd fit")

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show() #all data
print("              " + nn_model.metrics_names[0] + "                " + nn_model.metrics_names[1])

score = nn_model.evaluate(X_test, y_test, verbose=0)

print("Test_s      : ", score)

score = nn_model.evaluate(X_val, y_val, verbose=0)

print("Val_s       : ", score)

score = nn_model.evaluate(X_test_alt1, y_test_alt1, verbose=0)

print("Test_alt1 : ", score)

score = nn_model.evaluate(X_val_alt1, y_val_alt1, verbose=0)

print("Val_alt1  : ", score)

score = nn_model.evaluate(X_test_alt2, y_test_alt2, verbose=0)

print("Test_alt2 : ", score)

score = nn_model.evaluate(X_val_alt2, y_val_alt2, verbose=0)

print("Val_alt2  : ", score)

score = nn_model.evaluate(X_test_alt3, y_test_alt3, verbose=0)

print("Test_alt3 : ", score)

score = nn_model.evaluate(X_val_alt3, y_val_alt3, verbose=0)

print("Val_alt3  : ", score)

#nn_model.fit(X_train_alt3_s, y_train_alt3_s, epochs=epochs, batch_size=batch_size, verbose=1)
# Gather and process the testing data to create the submission results...



submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col = 'seg_id')

X_test_sub = pd.DataFrame(columns = X_train.columns, dtype = np.float64, index = submission.index)



for id in X_test_sub.index:

    seg = pd.read_csv('../input/LANL-Earthquake-Prediction/test/' + id + '.csv')

    x = pd.Series(seg['acoustic_data'].values)



    #For every segment (150,000), make these calculations

    X_test_sub.loc[id, 'mean'] = np.mean(x)

    X_test_sub.loc[id, 'median'] = np.median(x)

    X_test_sub.loc[id, 'std'] = np.std(x)

    X_test_sub.loc[id, 'max'] = np.max(x)

    X_test_sub.loc[id, 'skew'] = x.skew()

    X_test_sub.loc[id, 'min'] = np.min(x)

    X_test_sub.loc[id, 'sum'] = np.sum(x)

    X_test_sub.loc[id, 'var'] = np.var(x)

    X_test_sub.loc[id, 'ptp'] = np.ptp(x) #Peak-to-peak is like range

    X_test_sub.loc[id, 'mean_change_abs'] = np.mean(np.diff(x))

    X_test_sub.loc[id, 'max_to_min'] = np.max(x) / np.abs(np.min(x))

    X_test_sub.loc[id, 'max_to_min_diff'] = np.max(x) - np.abs(np.min(x))



    X_test_sub.loc[id, 'abs_avg'] = np.abs(x).mean()

    X_test_sub.loc[id, 'abs_std'] = np.abs(x).std()

    X_test_sub.loc[id, 'abs_max'] = np.abs(x).max()

    X_test_sub.loc[id, 'abs_min'] = np.abs(x).min()

    X_test_sub.loc[id, 'abs_med'] = np.abs(x).median()

    X_test_sub.loc[id, 'abs_skew'] = np.abs(x).skew()

    X_test_sub.loc[id, 'abs_sum'] = np.abs(x).sum()

    X_test_sub.loc[id, 'abs_var'] = np.abs(x).var()

    X_test_sub.loc[id, 'abs_ptp'] = np.abs(x).ptp() #Peak-to-peak is like range



    X_test_sub.loc[id, '10p'] = np.percentile(x,q=10) 

    X_test_sub.loc[id, '25p'] = np.percentile(x,q=25) #We can also grab percentiles

    X_test_sub.loc[id, '50p'] = np.percentile(x,q=50)

    X_test_sub.loc[id, '75p'] = np.percentile(x,q=75)

    X_test_sub.loc[id, '90p'] = np.percentile(x,q=90)   



    X_test_sub.loc[id, 'abs_1p'] = np.percentile(x, np.abs(0.01))

    X_test_sub.loc[id, 'abs_5p'] = np.percentile(x, np.abs(0.05))

    X_test_sub.loc[id, 'abs_30p'] = np.percentile(x, np.abs(0.30))

    X_test_sub.loc[id, 'abs_60p'] = np.percentile(x, np.abs(0.60))

    X_test_sub.loc[id, 'abs_95p'] = np.percentile(x, np.abs(0.95))

    X_test_sub.loc[id, 'abs_99p'] = np.percentile(x, np.abs(0.99))



#    X_test_sub.loc[id, 'mean_first_10000'] = x[:10000].mean()

#    X_test_sub.loc[id, 'mean_last_10000']  =  x[-10000:].mean()

#    X_test_sub.loc[id, 'mean_first_50000'] = x[:50000].mean()

#    X_test_sub.loc[id, 'mean_last_50000'] = x[-50000:].mean()



#    X_test_sub.loc[id, 'std_first_10000'] = x[:10000].std()

#    X_test_sub.loc[id, 'std_last_10000']  =  x[-10000:].std()

#    X_test_sub.loc[id, 'std_first_50000'] = x[:50000].std()

#    X_test_sub.loc[id, 'std_last_50000'] = x[-50000:].std()

# Predict the test submission data

nn_predictions = nn_model.predict(X_test_sub)

nn_bm_prediction = nn_bm_model.predict(X_test_sub)



# Generate Benchmark Prediction

submission['time_to_failure'] = nn_bm_prediction

print(submission.head(5))

submission.to_csv('submission_bm.csv')



# Generate Final Prediction

submission['time_to_failure'] = nn_predictions

print(submission.head(5))

submission.to_csv('submission_nn.csv')
# At this point, i've submitted both my benchmark and NN for comparison...

# Frustratingly so, the benchmark did better...  It obtained a score of 2.659

# I plan to compute MAE between the NN predicted and the benhcmark to hopefully improve the NN

# Kaggle limits submissions to 2 per day so hoping this helps as a comparison for checking my work



# pull in previously scored benchmark (keep in mind only 20% is scored so this may not work)

comp_sub = pd.read_csv('../input/submission-216/submission_nn_216.csv', index_col = 'seg_id')

total_error = 0

for id in comp_sub.index:

    # calculate MAE

    total_error = total_error + np.abs(comp_sub.loc[id].values[0] - submission.loc[id].values[0])

# print MAE between NN and benchmark    

print(total_error/comp_sub.shape[0])



# initial MAE was 17 prior to adding L1 regularization, the scoring made me think it was overfitting test data

# L1 with 0.1 gave MAE of 2.01 so 

# L1 with 0.01 gave MAE of 1.055 and the output comparison chart above looked best fit
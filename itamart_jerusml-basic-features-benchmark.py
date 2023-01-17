## JerusML meetup #12: Hands on Kaggle for everyone ##

## Credit for 'inversion' who's Kernel was used as a skeleton ##



import numpy as np # Python package that effiecently performs computations on n-dimentional data types.

import pandas as pd # Python package useful to hold and persent data. 



import matplotlib.pyplot as plt # Used to plot graphs of the data.

from tqdm import tqdm # Show progess bar for example it can show the progess of a loop.



#Sklearn is a pyhton packages that provides tools for data mining and analysis.

from sklearn.preprocessing import StandardScaler 

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error
# Read the provides sesmic data 

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}) #train is of type DataFrame.

# **Short explantion about imports**
#Now we can perform some simple operations to view the data.

#Train holds the data as a DataFrame variable which is in the form of a table.  

train.head() #Shows the first 5 entries of the table. 

#train.head(10) #Shows the first 10 entries

# pandas doesn't show us all the decimals

pd.options.display.precision = 15
# much better! (Note another way to show the first 10 entries).

train[0:10]

#train[0:20:2] #Or the first 10 even entries.
train[0:10]['acoustic_data']

#train[0:10]['time_to_failure']
#Now, lets try to understand the data. 

#First lets see what are the dimentions of our train data.

train.shape
#function for plotting based on both features

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure"):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color='r')

    ax1.set_ylabel('acoustic data', color='r')

    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color='b')

    ax2.set_ylabel('time to failure', color='b')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    plt.grid(True)
#We can sample the data set to be able to plot it and maybe see patterns.

train_ad_sample_df = train['acoustic_data'].values[::100]

train_ttf_sample_df = train['time_to_failure'].values[::100]



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

del train_ad_sample_df

del train_ttf_sample_df
#Lets zoom into one 'quake'

test = train[0:6_000_000]

train_ad_sample_df = test['acoustic_data'].values[::100]

train_ttf_sample_df = test['time_to_failure'].values[::100]



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

del train_ad_sample_df

del train_ttf_sample_df
#We choose to divide the data into segments of 150,000 since the test data is provided in segements of this size.

rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



print(segments) #We get 4194 segments.
#Now we would like to extract useful insights on each segments - those insights will be called features.



#We now defined two new data types, each holds the extracted features of each segments.



# Will holds information about the acoustic data of each seagment.

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min']) 

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



#For each consequent 150,000 rows, extract features and save them a new row in X_train and y_train.

for segment in tqdm(range(segments)):

    #Extract consquent rows. 

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values

    

    #Extract simple statistical features on the data. 

    #Could we extract other interesting features? Would it help?

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()



    #For each segment we choose to take the last time to failure as the segment's time to failure as specified.

    y_train.loc[segment, 'time_to_failure'] = y[-1]

    

X_train.head(10)
y_train.head(10)
# StandardScaler is used to normalize/standardize (mean = 0 and standard deviation = 1) your features

# before applying the machine learning techniques

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
# Learn the data.

# We are interested in predicting a value from a segment of samples.

# Therefore we are looking for a regression model, which can use the extracted features from the segment and make a prediction.

# We will apply Support Vector Regression in order to make predictions.

svm = NuSVR()

#Learns the data.

svm.fit(X_train_scaled, y_train.values.flatten())

# Generate the predictions.

y_pred = svm.predict(X_train_scaled)
#Plot the average features of data points vs. 

plt.plot(X_train_scaled[:,[0]], y_train.values.flatten(), 'ro')

# Show the correlation between the predicted and the actual time to failure.

plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), y_pred)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
score = mean_absolute_error(y_train.values.flatten(), y_pred)

print(f'Score: {score:0.3f}')
# Read the output file for submission. 

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission.head()
# Create a data frame to hold the test results.

X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
X_test.head()
# Read each test segment and extract features from each.

for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()
# Predict the result and save to submission file.

X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = svm.predict(X_test_scaled)

submission.to_csv('submission.csv')
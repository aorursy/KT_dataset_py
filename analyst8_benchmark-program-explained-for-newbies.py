# Load package to display pictures and videos

from IPython.display import Audio, Image, YouTubeVideo
# Display a video using the id which is the string between the = and & (if there is an &)

YouTubeVideo(id='m_dBwwDJ4uo', width=600, height=400)
# Display a video using the id which is the string between the = and & (if there is an &)

YouTubeVideo(id='P7h0JfQ-oHg', width=600, height=400)
# Import specific packages and modules

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error



import os

print(os.listdir("../input"))
# Gives number of rows without loading the file first

! wc -l "../input/train.csv"
# Load training data

# I used float32 and not float64 (like the example) since I did not have the memory for float64.

train = pd.read_csv("../input/train.csv", dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
# Display time_to_failure with more units of precision

pd.options.display.precision = 15
train.head()
# Break approximately 600 million rows up into 150K segments.

rows = 150_000



# Determine how many EQUAL segments to divide total dataset into.

# Following code takes the first [0] (or zero) column of the train data. Then divides by rows (or 150,000).

# Then rounds the value down (or floor). Then sets the value as an integer. 

segments = int(np.floor(train.shape[0] / rows))

print(segments)
# Create an empty dataframe (X_train) which is the size of the dataset/ 150000 or 4194 with ave, std, max, min columns.

# Create another empty dataframe (y_train) with just the time to failure.

# For both dataframes, index is set as range of the segments.

# range() function returns a sequence of numbers with format range(start, stop, step) with start=1 and step=1 if not given

X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])
# Empty dataframe for X_train

X_train.head()
# Empty dataframe for y_train

y_train.head()
# Calculate all the values for each of the 4194 segments in the X_train dataframe (mean, std, max, min).



# tqdm make your loops show a smart progress meter



# iloc is integer-location based indexing for selection by position

# [segment*rows:segment*rows+rows] points to which data to apply calculation to

# loops through all segments starting at 0 to 150k then 150k to 300k, etc.

# For each segment, the mean, standard deviation, maximum and min are computed.



# Fill in the time values for the y_train dataframe. .values[-1] will point to the last time value in the 150,000 segment



for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()
# View the X_train dataframe after the engineered features added

X_train.head()
# View the y_train dataframe after calculation applied to it.

y_train.head()
# Create a StandardScaler model called scaler.

# Normalize/standardize (mean = 0 and standard deviation = 1) features before applying machine learning techniques.

scaler = StandardScaler()



# fit will apply the scaler model to the dataframe (X_train in this case)

scaler.fit(X_train)



# transform will look at the dataframe columns one by one and return back a series (or group of series) 'made' of scalars

X_train_scaled = scaler.transform(X_train)

print(X_train_scaled)
# Create a model Nu Support Vector Regression

# In simple regression we try to minimise the error rate. While in SVR we try to fit the error within a certain threshold.

# NuSVR allows you to limit the number of support vectors used.

# Create svm model and fit to the data

# .flatten will return a copy of the array collapsed into one dimension

svm = NuSVR()

svm.fit(X_train_scaled, y_train.values.flatten())



# Predict y (acoustic data) with the model

y_pred = svm.predict(X_train_scaled)

print(y_pred)# Create a model Nu Support Vector Regression

# In simple regression we try to minimise the error rate. While in SVR we try to fit the error within a certain threshold.

# NuSVR allows you to limit the number of support vectors used.

# Create svm model and fit to the data

# .flatten will return a copy of the array collapsed into one dimension

svm = NuSVR()

svm.fit(X_train_scaled, y_train.values.flatten())



# Predict y (acoustic data) with the model

y_pred = svm.predict(X_train_scaled)

print(y_pred)
# Plot y actual vs y predicted

# Orange diagonal line is point of reference to compare acutal (on x-axis) vs predicted (on y-axis)

plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), y_pred)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
# Determine the score of the model

score = mean_absolute_error(y_train.values.flatten(), y_pred)

print(f'Score: {score:0.3f}')
# Read in the blank sample submission file

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

submission.head()
# Prepare the dataframe for the test segments so can apply svm model

X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)

X_test.head()
# Read in each test segment, apply engineered features and put in dataframe.

for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()



X_test.head()
# Apply the model to the X_test dataframe by scaling then transforming it.

X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = svm.predict(X_test_scaled)



# Prepare submission for Kaggle

# Best to use date and submssion

submission.to_csv('submission_one_04_15.csv')
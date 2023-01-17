# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import CSV into Pandas dataframe and test shape of file

train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
print ("Train data shape:", train_df.shape)

print ("Test data shape:", test_df.shape)

print ("Submission shape:", submission_df.shape)
# Create variable names to use



names = ['label']

for i in range(1, 785):

    names.append("var" + str(i).zfill(3))

names
%%time

#Split training data



from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV



X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['label'], axis=1), train_df["label"], shuffle=True,

train_size=.25, random_state=1)
# Check the shape of the trainig data set array



print('Shape of X_train_data:', X_train.shape)

print('Shape of y_train_data:', y_train.shape)

print('Shape of X_test_data:', X_test.shape)

print('Shape of y_test_data:', y_test.shape)
%%time

#import libraries & regressor

from sklearn.ensemble import RandomForestClassifier



seed = 427



rf = RandomForestClassifier(max_features='sqrt', random_state=seed)



#conduct random forest and generate test data predictions



rf_no_pca_prediction = rf.fit(X_train, y_train).predict(X_test)

#Create names function and metrics to be captured in the summary

names = ['F1 Score', 'Time', 'Number of Variables', 'time']
%%time

#Set up metrics dataframe to compare results



import time

from sklearn.metrics import f1_score



comparison = ['F1 Score', 'Time', 'Number of Variables', 'time']

metrics = {}

begin = time.clock()





#rf_no_pca_prediction results

score = f1_score(y_test, rf_no_pca_prediction, average='micro')

duration = time.clock() - begin

metrics['RF without PCA'] = [score, duration, X_train.shape[1], begin]
%%time

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



begin2 = time.clock()



X = np.concatenate([X_train, X_test]).astype(float)

X = StandardScaler().fit_transform(X)

X = PCA(n_components = 0.95, random_state=seed).fit_transform(X)



X_train_pca = X[0:60000]

X_test_pca = X[-10000:]



duration_pca_combined = time.clock() - begin2

metrics['PCA'] = [float('NaN'), duration, X_train.shape[1], begin2]
%%time

begin3 = time.clock()



# Transform testing and training data and scale data

std_scaler = StandardScaler().fit(X_train.astype(float))

X_train_std = std_scaler.transform(X_train.astype(float))

X_test_std = std_scaler.transform(X_test.astype(float))



# Define PCA to explain at least 95% of variance of training data

rf_pca = PCA(n_components=0.95, random_state=seed).fit(X_train_std)



# Generate components on training and testing data

# Assumption is that this should go with PCA timing

X_train_rf_pca = rf_pca.transform(X_train_std)

X_test_rf_pca = rf_pca.transform(X_test_std)



# Record the clock time it takes

duration = time.clock() - begin3



# Capture metrics (f1 score does not apply for PCA)

metrics['PCA - Fixed'] = [float('NaN'), duration, X_train.shape[1],

                          begin3]
%%time

# PCA BASED ON TRAINING



# Start timer

begin4 = time.clock()



# Scale training data

std_scaler = StandardScaler().fit(X_train.astype(float))



# Transform training and testing data based on scaler

X_train_std = std_scaler.transform(X_train.astype(float))

X_test_std = std_scaler.transform(X_test.astype(float))



# Define PCA to explain at least 95% of variance of training data

pca = PCA(n_components=0.95, random_state=seed).fit(X_train_std)



# Generate components on training and testing data

# Assumption is that this should go with PCA timing

X_train_pca = pca.transform(X_train_std)

X_test_pca = pca.transform(X_test_std)



# Record the clock time it takes

duration = time.clock() - begin4



# Capture metrics (f1 score does not apply for PCA)

metrics['PCA - Fixed'] = [float('NaN'), duration, X_train.shape[1],

                          time.clock()]

%%time

# RANDOM FOREST WITH PCA (FIXED PCA)



# Start timer

begin5 = time.clock()



# Fit the random forest and generate test data predictions

rf_pca_predictions = rf.fit(X_train_pca, y_train).predict(X_test_pca)



# Calcuate the f1 score

score = f1_score(y_test, rf_pca_predictions, average='micro')



# Record the clock time it takes

duration = time.clock() - begin5



# Store the results in a dictionary

metrics['RF - PCA (Fixed)'] = [score, duration, X_train_pca.shape[1],

                               time.clock()]
#Generate 



summary = pd.DataFrame.from_dict(metrics, orient='index')

summary.columns = names

summary.sort_values(by=['time'], inplace=True)

summary.drop(['time'], axis=1, inplace=True)

summary.to_csv('results_summary.csv')
summary
results = rf.predict(X_test_pca)



np.savetxt('results.csv', 

           np.c_[range(1,len(X_test_pca)+1),results], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '', 

           fmt='%d')
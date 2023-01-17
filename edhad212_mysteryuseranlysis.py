# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read the files
usrData = pd.read_csv("../input/mysteryusers/users.csv")
viewData = pd.read_csv("../input/viewings/viewings.csv")


usrData.head(n = 10)
viewData.head(n = 10)
usrData.info()
usrData.shape
#Filter the data based on the 2016 - 2017 school year
usr2017 = usrData[(usrData['Sign up date'] > '2016-07-01') & (usrData['Sign up date'] < '2017-06-30')]
# Define a function that would take in a data frame and return the number of schools that signed up at least once 
def countOrgs(df):
    '''
    inputs:
    - A pandas dataFrame that contains the users info
    returns:
    - The number of organisations that signed up at least once
    
    '''
    #Declare a dictionary to log the number of sign ups
    signUps = {}
    #Loop over the data frame and log the number of signups per school
    counter = 0
    for index, row in df.iterrows():
     if row["Organization ID"] not in signUps.keys():
        if row["Sign up date"] != "NaN":
            signUps[row["Organization ID"]] = 1
            counter += 1
    return(counter)
#Compare the number of unique organizatin in the data frame to the number of unique signups 
print(countOrgs(usr2017),len(usr2017["Organization ID"].unique()))
#Print the total number of unique schools in the original (useData) data frame
print(len(usrData["Organization ID"].unique()))
#To verify count the number of NaN in the signup date column for the 2017 users data frame
sum(usr2017["Sign up date"].isna()) # If 0 is retuned, boolean vevtor contains no NaN (Null values)
#Define a function that takes in the data framw of the school year 2016 - 2017 and finds ou the percntage that created a quote
def countOrgs(df):
    '''
    inputs:
    - A pandas dataFrame that contains users info 
    returns:
    - The percentage of organizations that created a quote 
    
    '''
    #Loop over all the unique ids in the data frame then subset 
    counter1 = 0 
    counter2 = 0
    for orgNum in df["Organization ID"].unique():
        tempDF = df[df["Organization ID"] == orgNum]
        #Sum the boolen vector to calclate the number of quotes created for that unique organization.
        if sum(tempDF['Quote created?']) > 0:
            counter1 += 1
            counter2 +=  sum(tempDF['Quote created?'])
    return(counter1,counter2) 

countOrgs(usr2017)
countOrgs(usr2017)[0]/len(usr2017["Organization ID"].unique())
viewData.info()
viewData.shape
sum(usrData["Sign up date"].isna())
# join both datasets to match an Orgnization's ID to the viewing behavior of its users
joinDf17 = pd.merge(usr2017,viewData, left_on = "User ID", right_on = "user_id", sort = False)
joinDf17.head(5)
joinDf17.info()
#Dimensions of the the new dataset
joinDf17.shape
#Identify the number of Null vales per column 
joinDf17.isna().sum()
len(joinDf17 ) - joinDf17.count()

joinDf17.head(6)
#Explore the distribution across different organization types
joinDf17.groupby("Organization Type")["Organization Type"].count()
#Explore countries that appear in the dataset
joinDf17["Country"].unique()
# Type cast potenitially taught as boolean
joinDf17["potentially_taught"] = joinDf17["potentially_taught"].astype("bool")
#Number of potentially taught lessons per organization
joinDf17[["Organization ID","potentially_taught"]].groupby("Organization ID")["potentially_taught"].sum()
# Number of quotes created by country
joinDf17[joinDf17["Quote created?"] == True].groupby("Country")["Quote created?"].count()
#Number of quotes created per lesson ID
quotesPerLess = joinDf17[joinDf17["Quote created?"] == True].groupby("lesson_id")["Quote created?"].count()
quotesPerLess
max(quotesPerLess)
# Number of quotes created per NGSS status

joinDf17[joinDf17["Quote created?"] == True].groupby("NGSS Status")["Quote created?"].count()
#Total number of unique users per ORg
unqUser = joinDf17.groupby("Organization ID")["User ID"].nunique()
unqUser = pd.Series.to_frame(unqUser)
#Quality assurance
joinDf17[ joinDf17["Organization ID"] == 398465 ]["User ID"].unique()
#total duration per organiztion
totDuration = joinDf17.groupby("Organization ID")["duration"].sum()
totDuration = pd.Series.to_frame(totDuration)
#Join the 2 dataframes
joinAll = totDuration.join(unqUser)
# Number of potenitally taught lessons per orgnization
potTaught = joinDf17.groupby("Organization ID")["potentially_taught"].sum()
potTaught = pd.Series.to_frame(potTaught)
joinAll = joinAll.join(potTaught)
totVidDur = joinDf17.groupby("Organization ID")["video_duration"].sum()
totVidDur = pd.Series.to_frame(totVidDur)
joinAll = joinAll.join(totVidDur)
prevTot = joinDf17.groupby("Organization ID")["previewed"].sum()
prevTot = pd.Series.to_frame(prevTot)
joinAll = joinAll.join(prevTot)
#Convert Quotes created to a 0,1 target variable for prediction
numQuotes = joinDf17.groupby("Organization ID")["Quote created?"].sum()
numQuotes = pd.Series.to_frame(numQuotes)
lambdaFunc = lambda x: [0,1][x>0]
numQuotes["Quote created?"] = numQuotes["Quote created?"].apply(lambdaFunc)
numQuotes["Quote created?"].sum()/586
joinAll.info()
joinAll = joinAll.join(numQuotes)
joinAll["Quote created?"].value_counts()
# Import resampling from sci-kit learn
from sklearn.utils import resample
# Separate majority and minority classes
df_majority = joinAll[ joinAll["Quote created?"] == 0]
df_minority = joinAll[ joinAll["Quote created?"] == 1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=534,    
                                 random_state=123) 
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled["Quote created?"].value_counts()
df_upsampled.info()
df_upsampled
#Shuffle the dataset
to_Train = df_upsampled.sample(frac = 1)
sns.pairplot(df_upsampled)
# Graph correlations as a heat map
plt.figure(figsize=(15,11))
sns.heatmap(df_upsampled.corr(), annot=True)
sns.distplot(df_upsampled["video_duration"])
sns.distplot(df_upsampled["previewed"])
sns.distplot(df_upsampled["potentially_taught"])
sns.distplot(df_upsampled["User ID"])
sns.distplot(df_upsampled["duration"])
to_Train.head()
data = to_Train
#Split the data into features and taeget sets 
Quoted = data["Quote created?"]
features_raw = data[["duration","User ID","potentially_taught","video_duration","previewed"]]  
features_raw = data[["duration","User ID","potentially_taught","video_duration","previewed"]]  
#Log trnsform skewed features
skewed = ["duration","User ID","potentially_taught","video_duration","previewed"]
features_raw[skewed] = data[skewed].apply(lambda x:np.log(x+1))
#Scale features
from sklearn.preprocessing import MinMaxScaler
#Initialize scaler, then apply it to numerical features
scaler = MinMaxScaler()
numerical =  ["duration","User ID","potentially_taught","video_duration","previewed"]
features_raw[numerical] = scaler.fit_transform(features_raw[numerical])
#OneHotEncode categorical features
#features = pd.get_dummies(features_raw)
#import train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_raw, Quoted, test_size = .2, random_state = 42) 
features_raw.info()
#Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: retained training set
       - X_test: features testing set
       - y_test: retained testing set
    ''' 
    results = {}
    #Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    #Calculate the training time
    results['train_time'] = end - start 
    #Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    #Calculate the total prediction time
    results['pred_time'] = end - start
    #Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(predictions_train,y_train[:300])
    #Compute accuracy on test set
    results['acc_test'] = accuracy_score(predictions_test,y_test)
    #Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(predictions_train,y_train[:300],.5)
    #Compute F-score on the test set
    results['f_test'] = fbeta_score(predictions_test,y_test,.5)
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    # Return the results
    return results
#import the needed classifiers    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from time import time
from sklearn.metrics import fbeta_score, accuracy_score
#Initialize the three models
clf_A = KNeighborsClassifier()
clf_B = RandomForestClassifier()
clf_C = svm.SVC()

#Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(round(.01 * X_train.shape[0]))
samples_10 = int(round(.1 * X_train.shape[0]))
samples_100 = X_train.shape[0]

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)  
results
clfRF = RandomForestClassifier()
#Train the model
model = clfRF.fit(X_train,y_train)
#Extract important features
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in clfRF.estimators_],
         axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(8,8))
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
   color="r", yerr=std[indices], align="center")
feature_names = X_train.columns
plt.xticks(range(X_train.shape[1]), feature_names)
plt.xticks(rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
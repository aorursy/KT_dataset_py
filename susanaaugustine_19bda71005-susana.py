import os #to read the directory

import pandas as pd #to read the csv file as a dataframe

import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split #to split the train into test and train sets

from sklearn.ensemble import RandomForestClassifier #to build my model

from sklearn.metrics import f1_score #to check the f1 score 

from sklearn.model_selection import cross_val_score #to check the accuracy
test_data = pd.read_csv('../input/bda-2019-ml-test/Test_Mask_Dataset.csv') #The given test data for which the predictions are suposed to be made

train_data = pd.read_csv('../input/bda-2019-ml-test/Train_Mask.csv') #The given Train data to prepare your model
#Gives you the number of columns and rows present in the given train and test data

print(train_data.shape , test_data.shape) 
#Gives you the features along with their data type available in the training data

train_data.info() 
#Checking if any missing value present in the training data

train_data.isnull().sum() 
#Gives you the summary statistics

train_data.describe() 
#Checking for unique values in the particular feature since they are cateogrical

train_data['motorTempBack'].unique()
train_data['motorTempFront'].unique()
#Checking the correlation between columns

corr = train_data.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
#Droping the column timeindex

train_data.drop('timeindex', axis=1, inplace=True)
#Spliting the train into X variable

X_train = train_data.drop("flag",axis=1)

#Spliting the train data into Y variable

y_train = train_data['flag']
#Spliting the train data into 70% as training data and 30% as testing data for checking the accuracy of the model 

train_X, test_X, train_y, test_y = train_test_split(X_train, y_train, test_size=0.30,random_state=0)
#Initializition of the algorithm with its parameters

clf = RandomForestClassifier(n_estimators=500,criterion='entropy',max_depth=30, random_state=0)

#Fitting the model

clf.fit(train_X, train_y)

#Predicting the y values for the trainging data

Predictions_rand = clf.predict(test_X)

#Checking the f1 score

print(f1_score(Predictions_rand,test_y))
#Checking for accuracy of the model by splitting it into 5 folds

scores = cross_val_score(clf, X_train, y_train, cv=5)

scores.sort()

accuracy = scores.mean()

print(scores)

print(accuracy)
#Droping the column timeindex

test_data.drop('timeindex', axis=1, inplace=True)
#Predicting the y values for the given test data

Sample = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
Sample['Anomaly_flag'] = clf.predict(test_data)

Sample['Anomaly_flag'].value_counts().unique()
Sample.to_csv('Submittion_clf.csv',index=False)
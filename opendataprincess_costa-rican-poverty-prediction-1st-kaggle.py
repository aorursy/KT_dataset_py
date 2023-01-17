#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a few plotting defaults
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'
#Read in Data and look at Summary Information
pd.options.display.max_columns = 150

# Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.shape
test.shape
test
Ids = test['Id'].unique()
Ids
test.groupby('SQBescolari').describe()
train.groupby('SQBescolari').describe()
#Let's look at the distribution of the 'Target' variable from the training data
fig=plt.figure(figsize=(10,10))
train.hist(column='Target')
plt.xlabel('Poverty Level')
#Now let's compare a few variables in the training and test datasets.

x = test['SQBescolari']
y = train['SQBescolari']

from matplotlib import pyplot
pyplot.hist(x, label='Test')
pyplot.hist(y, label='Train',color='purple')
pyplot.legend(loc='upper right')
pyplot.title('SQBescolari')
pyplot.show()
#To explore SQBovercrowding variable, a histogram won't show enough info so let's do a line plot
x = test['SQBovercrowding']
y = train['SQBovercrowding']

from matplotlib import pyplot as plt

plt.plot(x, label='Test', marker='o')
plt.plot(y, label='Train',color='purple')
plt.legend(loc='upper left')
plt.title('SQBovercrowding')
plt.show()

#Well that isn't extremely helpful since there don't appear to be any patterns. 

#Let's look at the outlier data points a bit more in each of the Test and Training data sets.
#Let's zoom in on the test outliers by changing the x and y axis limits
plt.plot(x, label='Test', marker='o')
plt.xlim(16000,20100)
plt.ylim(80,180)
plt.title('SQBovercrowding Test')
plt.show()

#The outlier values in the test dataset seem to be 100 and 170.
#Most values in the training dataset are between 0-40; and 0-50 in the test dataset.
#Let's look more closely at the variable 'SQBescolari' (square years of education) that has values 0, 1, 4 and 9
#And where the Target is 1 or 2 (extreme or moderate poverty)

SQBescolari_train = train.query('SQBescolari <=9' and 'Target <=2')
SQBescolari_train

#There are 755 with Target of 1, 1597 with Target of 2, and 2352 with either 1 or 2 
#out of the total train sample size of 9553.

#We still don't know if the 'SQBage' or 'SQBhogar_total' are significant variables so let's keep for now.
#Explore relationship between 'SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBovercrowding' and 'Target'
#Output feature in training dataset

import seaborn as sns

#Calculate the correlation matrix
corr = SQBescolari_train.corr()

#Plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

#From the heatmap, we see 'SQBage' and 'SQBhogar_total' have 0 correlation so these variables can be deleted.

#SQBescolari is positively correlated (0.4) and 'SQBovercrowding' is highly negatively correlated (-0.25)
trainclean = SQBescolari_train[['Id','SQBescolari', 'SQBovercrowding']]
trainclean
test.groupby('SQBovercrowding').describe()
#From looking at the training dataset above, we see Target of 1 or 2 when 'SQBovercrowding' has values between 
#0.111 and 12.25 so let's query with this in mind.

SQBescolari_test= test.query('SQBescolari <=9')
SQBescolari_test

#3162 have 0 years SQBescolari, 3879 1 years, 769 have 4 years, 1051 have 9 years out of 23856 test sample size.
##(about 37% of total sample size)
#Now let's find unique values for SQBovercrowding
SQBescolari_test.SQBovercrowding.unique()
#Clean up SQBescolari_test dataframe to get rid of unnecessary columsn
testclean = SQBescolari_test[['Id','SQBescolari', 'SQBovercrowding']]
testclean
#Let's see if we can visualize the relationship (if any) between SQBescolari & SQBovercrowding

import seaborn as sns

#Calculate the correlation matrix
corr = testclean.corr()

#Plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

#It appears that the two variables are highly linearly correlated.

#Build a quick baseline Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

#Define input and output features
ytrain = train.iloc[:,-1] #Define target variable as last column of data frame (see https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/)
Xtrain = train.drop('Target', axis=1)

#Fill NAs
Xtrain = Xtrain.fillna(-999)

#label encoder
for c in train.columns[train.dtypes == 'object']:
    Xtrain[c] = Xtrain[c].factorize()[0]

rf = RandomForestClassifier()
rf.fit(Xtrain,ytrain)
#Test the model

#Create a copy to work with
Xtest = test.copy()

#Save and drop labels
ytest = Xtrain
Xtest = Xtrain.iloc[0:141]

#Fill NAs
Xtest = Xtest.fillna(-999)

#Make the prediction
ypredictions = rf.predict(ytest)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print("=== Confusion Matrix ===")
print(confusion_matrix(ytest, ypredictions))
print('\n')
print("=== Classification Report ===")
print(classification_report(ytest, ypredictions))
print('\n')
#Random Forest Classifier Model Metrics

# Use the forest's predict method on the test data
predictions = rf.predict(Xtest)
# Calculate the absolute errors
errors = abs(predictions - ytest)

#Macro F1 Score is Model Evaluation Metric
from sklearn.metrics import f1_score

print("=== Macro F1 Score ===")
f1_score (ytrain, ypredictions, average='macro')
#Now add these tuned parameters to the model to see if we can improve results
rfc = RandomForestClassifier(n_estimators=1000, max_depth=300, max_features='auto')
rfc.fit(Xfeatures_train,yfeatures_train)
rfc_predict = rfc.predict(Xfeatures_test)

print("=== Macro F1 Score ===")
f1_score (ytrain, ypredictions, average='macro')

#Notice there is no change in the macro F1 score with hypertuned parameters.
#Plot Feature Importance
plt.figure(figsize=(10,10)) #Increased figure size to see which features are most interesting
plt.plot(rf.feature_importances_, 'bo') #change to points to see individual feature points.
plt.xticks(np.arange(Xtrain.shape[1], Xtrain.columns.tolist, rotation=vertical))
plt.xlabel('Features')
plt.xlim(90,140)
plt.show()
#TLet's take a closer look at the outliers to see which features might affect the model the most.
import numpy as np
np.set_printoptions(threshold=np.inf)  #https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array

print("-Here are the predicted Poverty Level Targets-")
ypredictions
#Read in Results Data
submission = pd.read_csv('CR_Kaggle_LKahn.csv')
submission.head(5)
submission.to_csv('./Submission_log_RF.csv')
#Next, let's try a NN to see if we can improve F1 macro score
from sklearn.neural_network import MLPClassifier

#Create a copy to work with
Xtrain = train.copy()

#Save and drop labels
ytrain = Xtrain.iloc[:,-1] #Define target variable as last column of data frame (see https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/)
Xtrain = Xtrain.drop('Target', axis=1)

#Fill NAs
Xtrain = Xtrain.fillna(-999)

#label encoder
for c in train.columns[train.dtypes == 'object']:
    Xtrain[c] = Xtrain[c].factorize()[0]

MLP= MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2),random_state=1)
MLP.fit(Xtrain,ytrain)

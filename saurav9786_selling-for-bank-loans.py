# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import model_selection

from sklearn import metrics

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bank_df = pd.read_excel('/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')


#Displaying the Dataset



bank_df.head(5)
#Columns in the datset



bank_df.columns
#Data types for each attributes

bank_df.dtypes
#Five point summary for the dataset



bank_df.describe().T
# Column descriptions



##	Data Description:								

##									

##	ID	Customer ID							

##	Age	Customer's age in completed years							

##	Experience	#years of professional experience							

##	Income	Annual income of the customer ($000)							

##	ZIPCode	Home Address ZIP code.							

##	Family	Family size of the customer							

##	CCAvg	Avg. spending on credit cards per month ($000)							

##	Education	Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional							

##	Mortgage	Value of house mortgage if any. ($000)							

##	Personal Loan	Did this customer accept the personal loan offered in the last campaign?							

##	Securities Account	Does the customer have a securities account with the bank?							

##	CD Account	Does the customer have a certificate of deposit (CD) account with the bank?							

##	Online	Does the customer use internet banking facilities?							

##	CreditCard	Does the customer use a credit card issued by UniversalBank?							
#importing seaborn for statistical plots

import seaborn as sns

bank_df_attr = bank_df.iloc[:,0:12] #compare all attributes visually to check for relationships that can be exploited



sns.pairplot(bank_df_attr)
# Age and experience have strong positive relation

# Age and income have no linear relationship

# Age and experience have no linear relationship
bank_df.groupby(["Personal Loan"]).count()  #The data set is skewed in terms of target column.

# There are far few records in class 1 i.e. people who took the personal loan last time. But that 

# is the class of interst to us! We have to identify potential customers and do not want any potential

# customer to be missed. 
# Separate the independent attributes i.e. columns 0 to 8 and store them in X array

# Store the target column (column 8) into Y array



X_df = bank_df.loc[:, bank_df.columns != 'Personal Loan']

y_df = bank_df.loc[:, bank_df.columns == 'Personal Loan']
####### --------------------------- Logistic Model --------------------------------------
# Create the training and test data set in the ratio of 70:30 respectively. Can be any other ratio...

# Prepare data for logistic regression



features=X_df.iloc[:, 0:10]

features_array = features.values   #logistic modeling algorithm requires feature array not dataframe

target_labels = y_df.values





test_size = 0.40 # taking 60:40 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code when using random functions to 

# generate the training and test data



X_train, X_test, y_train, y_test = model_selection.train_test_split(features_array, target_labels, test_size=test_size, random_state=seed)

y_train = np.ravel(y_train)   # to convert 1 d vector into 1 d array
# Let us first try logistic regression to predict the personal loan affinity - 

# We removed the binned columns for this (considered only 11 columns 1:10) which are numeric



model = LogisticRegression()

model.fit(X_train, y_train)

model_score = model.score(X_test, y_test)

y_predict = model.predict(X_test)

print(model_score)

print(metrics.confusion_matrix(y_test, y_predict))
# The accuracy score of .954 looks impressive but do not forget, it is unreliable as it is a score at

# model level. Let us look at class level, especially the class 1.



# summarize the fit of the model

print(metrics.classification_report(y_test, y_predict))
#The precision and recall for class 1 is low...



#Precision: Within a given set of positively-labeled results, the fraction that were 

#true positives = tp/(tp + fp) , this has to be calculated for each class i.e. 0 and 1 and should be

# high for the class less represented, class 1 in our example



#Recall: Given a set of positively-labeled results, the fraction of all positives that were 

#retrieved = tp/(tp + fn)



# Accuracy: tp + tn / (tp + tn + fp +fn) But this measure can be dominated by larger class. 

# Suppose 10, 90 and 80 of 90 is correctly predicted while only 2 of 0 is predicted correctly. 

# Accuracy is 80+2 / 100 i.e. 82%



# F is harmonic mean of precision and recal given by ((B^2 +1) PR) / (B^2P +R)

#When B is set to 1 we get F1 = 2PR / (P+R)
###### ----------------------- Naive Bayes Model ---------------------------------

# Create the training and test data set in the ratio of 70:30 respectively. Can be any other ratio...

# Prepare data for logistic regression



features=X_df.iloc[:, 0:10]



target_labels = bank_df.loc[:, bank_df.columns == 'Personal Loan']





X_array = features.values

y_array = target_labels.values





test_size = 0.40 # taking 60:40 training and test set

seed = 7  # Random numbmer seeding for reapeatability of the code



X_train, X_test, y_train, y_test = model_selection.train_test_split(X_array, y_array, test_size=test_size, random_state=seed)

y_train = np.ravel(y_train)   # to convert 1 d vector into 1 d array
# Invoking the NB Gaussian function to create the model

# fitting the model in the training data set

model = GaussianNB()

model.fit(X_train, y_train)



predictions=model.predict(X_test)



#Assess the accuracy of the model on test data

print(metrics.confusion_matrix(y_test,predictions))
# make predictions

expected = y_test

predicted = model.predict(X_test)

# summarize the fit of the model

print(metrics.classification_report(expected, predicted))
## Naive Bayes also is not giving the class 1 metrics in the acceptal range (80% and above). Let us try

## K Nearest Neighbours
#####  ------------------------------ K Nearest Neighbours ------------------------------------
from sklearn.neighbors import KNeighborsClassifier

NNH = KNeighborsClassifier(n_neighbors= 3 , weights = 'distance')

NNH.fit(X_train, y_train)
predicted_labels = NNH.predict(X_test)
predicted_labels = NNH.predict(X_test)
print(metrics.confusion_matrix(y_test, predicted_labels))
# summarize the fit of the model

print(metrics.classification_report(y_test, predicted_labels))
### Recall (true positives / (true positives + false negatives)) for class 1 is the least. That is 

### because majority of data points belong to class 0 and in KNN, probability of finding data points

### from class 0 closer to a test point than a data point from class 1 is high.
### let us check the effect of scaling (convert all dimensions to z scores)

from sklearn import preprocessing

X_train_scaled = preprocessing.scale(X_train)

X_test_scaled = preprocessing.scale(X_test)

NNH.fit(X_train_scaled, y_train)
predicted_labels = NNH.predict(X_test_scaled)
print(metrics.confusion_matrix(y_test, predicted_labels))
# summarize the fit of the model

print(metrics.classification_report(y_test, predicted_labels))
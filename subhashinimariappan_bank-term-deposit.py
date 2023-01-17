# Importing necessary packages

import pandas as pd
#importing data

file_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter03/bank-full.csv'

bankData = pd.read_csv(file_url, sep=";")
# Getting the total counts under each job category

jobTot = bankData.groupby('job')['y'].agg(jobTot='count').reset_index()

jobTot
# Getting all the details in one place

jobProp = bankData.groupby(['job', 'y'])['y'].agg(jobCat='count').reset_index()
# Merging both the data frames

jobComb = pd.merge(jobProp, jobTot, on=['job'])

jobComb['catProp'] = (jobComb.jobCat/jobComb.jobTot)*100



jobComb.head()
import matplotlib.pyplot as plt

import numpy as np



# Create seperate data frames for Yes and No

jobcombYes = jobComb[jobComb['y'] == 'yes']

jobcombNo = jobComb[jobComb['y'] == 'no']



# Get the length of the xaxis labels 

xlabels = jobTot['job'].nunique()



# Get the proportion values 

jobYes = jobcombYes['catProp'].unique()

jobNo = jobcombNo['catProp'].unique()



# Arrange the indexes of x asix

ind = np.arange(xlabels)



# Get the width of each bar

width = 0.35  



# Getting the plots

p1 = plt.bar(ind, jobYes, width)

p2 = plt.bar(ind, jobNo, width,bottom=jobYes)



plt.ylabel('Propensity Proportion')

plt.title('Propensity of purchase by Job')



# Defining the x label indexes and y label indexes

plt.xticks(ind, jobTot['job'].unique())

plt.yticks(np.arange(0, 100, 10))



# Defining the legends

plt.legend((p1[0], p2[0]), ('Yes', 'No'))



# To rotate the axis labels 

plt.xticks(rotation=90)

plt.show()
# Normalising data

from sklearn import preprocessing

x = bankData[['balance']].values.astype(float)

# Creating the scaling function

minmaxScaler = preprocessing.MinMaxScaler()

# Transforming the balance data by normalising it with minmaxScalre

bankData['balanceTran'] = minmaxScaler.fit_transform(x)

# Printing the head of the data

bankData.head()
# Adding a small numerical constant to eliminate 0 values



bankData['balanceTran'] = bankData['balanceTran'] + 0.00001
# Let us transform values for loan data

bankData['loanTran'] = 1

# Giving a weight of 5 if there is no loan

bankData.loc[bankData['loan'] == 'no', 'loanTran'] = 5

bankData.head()
# Let us transform values for Housing data

bankData['houseTran'] = 5

# Giving a weight of 1 if the customer has a house

bankData.loc[bankData['housing'] == 'no', 'houseTran'] = 1



bankData.head()
# Let us now create the new variable which is a product of all these

bankData['assetIndex'] = bankData['balanceTran'] * bankData['loanTran'] * bankData['houseTran']

bankData.head()
# Finding the quantile

np.quantile(bankData['assetIndex'],[0.25,0.5,0.75])
# Creating quantiles from the assetindex data

bankData['assetClass'] = 'Quant1'



bankData.loc[(bankData['assetIndex'] > 0.38) & (bankData['assetIndex'] < 0.57), 'assetClass'] = 'Quant2'



bankData.loc[(bankData['assetIndex'] > 0.57) & (bankData['assetIndex'] < 1.9), 'assetClass'] = 'Quant3'



bankData.loc[bankData['assetIndex'] > 1.9, 'assetClass'] = 'Quant4'



bankData.head()
# Calculating total of each asset class

assetTot = bankData.groupby('assetClass')['y'].agg(assetTot='count').reset_index()

# Calculating the category wise counts

assetProp = bankData.groupby(['assetClass', 'y'])['y'].agg(assetCat='count').reset_index()
# Merging both the data frames

assetComb = pd.merge(assetProp, assetTot, on=['assetClass'])

assetComb['catProp'] = (assetComb.assetCat / assetComb.assetTot)*100

assetComb
# Categorical variables, removing loan and housing

bankCat1 = pd.get_dummies(bankData[['job','marital','education','default','contact','month','poutcome']])
bankNum1 = bankData[['age','day','duration','campaign','pdays','previous','assetIndex']]

bankNum1.head()
# Normalise some of the numerical variables

from sklearn import preprocessing
# Creating the scaling function

minmaxScaler = preprocessing.MinMaxScaler()
# Creating the transformation variables

ageT1 = bankNum1[['age']].values.astype(float)

dayT1 = bankNum1[['day']].values.astype(float)

durT1 = bankNum1[['duration']].values.astype(float)
# Transforming the balance data by normalising it with minmaxScalre

bankNum1['ageTran'] = minmaxScaler.fit_transform(ageT1)

bankNum1['dayTran'] = minmaxScaler.fit_transform(dayT1)

bankNum1['durTran'] = minmaxScaler.fit_transform(durT1)
# Let us create a new numerical variable by selecting the transformed variables

bankNum2 = bankNum1[['ageTran','dayTran','durTran','campaign','pdays','previous','assetIndex']]



# Printing the head of the data

bankNum2.head()
# Preparing the X variables

X = pd.concat([bankCat1, bankNum2], axis=1)

print(X.shape)

# Preparing the Y variable

Y = bankData['y']

print(Y.shape)

X.head()
from sklearn.model_selection import train_test_split

# Splitting the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=123)

from sklearn.linear_model import LogisticRegression

# Defining the LogisticRegression function

bankModel = LogisticRegression()

bankModel.fit(X_train, y_train)
pred = bankModel.predict(X_test)

print('Accuracy of Logisticr regression model prediction on test set: {:.2f}'.format(bankModel.score(X_test, y_test)))
# Confusion Matrix for the model

from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, pred)

print(confusionMatrix)
#accuracy on model

from sklearn.metrics import classification_report

print(classification_report(y_test, pred))
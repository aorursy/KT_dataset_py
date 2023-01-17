import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/fatality-analysis-reporting-system-data/fars_train.csv")

test = pd.read_csv("../input/fatality-analysis-reporting-system-data/fars_test.csv")
train.info()

train[0:10]
train1 = train[['AGE','INJURY_SEVERITY']]

train1 = train[pd.notnull(train['AGE'])]

train1.rename(columns = {'AGE':'AGE'}, inplace = True)

train1.head(10)
# Distribution of INJURY_SEVERITY

train.INJURY_SEVERITY.value_counts()[0:30].plot(kind='bar')

plt.show()
train1 = train[['CASE_STATE','AGE','SEX','PERSON_TYPE','SEATING_POSITION','RESTRAINT_SYSTEM-USE','AIR_BAG_AVAILABILITY/DEPLOYMENT','EJECTION','EJECTION_PATH','EXTRICATION','NON_MOTORIST_LOCATION','NON_MOTORIST_LOCATION','POLICE_REPORTED_ALCOHOL_INVOLVEMENT','METHOD_ALCOHOL_DETERMINATION',

          'ALCOHOL_TEST_TYPE','ALCOHOL_TEST_RESULT','POLICE-REPORTED_DRUG_INVOLVEMENT','METHOD_OF_DRUG_DETERMINATION','DRUG_TEST_RESULTS_(1_of_3)','DRUG_TEST_TYPE_(2_of_3)','DRUG_TEST_RESULTS_(2_of_3)','DRUG_TEST_TYPE_(3_of_3)','DRUG_TEST_RESULTS_(3_of_3)',

          'HISPANIC_ORIGIN','TAKEN_TO_HOSPITAL','RELATED_FACTOR_(1)-PERSON_LEVEL','RELATED_FACTOR_(2)-PERSON_LEVEL','RELATED_FACTOR_(3)-PERSON_LEVEL','RACE','INJURY_SEVERITY']] #Subsetting the data

cor = train1.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=train["INJURY_SEVERITY"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def FunLabelEncoder(df):

    for c in df.columns:

        if df.dtypes[c] == object:

            le.fit(df[c].astype(str))

            df[c] = le.transform(df[c].astype(str))

    return df
train = FunLabelEncoder(train)

train.info()

train.iloc[235:300,:]
test.info()

test[0:10]
test = FunLabelEncoder(train)

test.info()

test.iloc[235:300,:]
print("Any missing sample in training set:",train.isnull().values.any())

print("Any missing sample in test set:",test.isnull().values.any(), "\n")


features=['CASE_STATE','AGE','SEX','PERSON_TYPE','SEATING_POSITION','RESTRAINT_SYSTEM-USE','AIR_BAG_AVAILABILITY/DEPLOYMENT','EJECTION','EJECTION_PATH','EXTRICATION','NON_MOTORIST_LOCATION','NON_MOTORIST_LOCATION','POLICE_REPORTED_ALCOHOL_INVOLVEMENT','METHOD_ALCOHOL_DETERMINATION',

          'ALCOHOL_TEST_TYPE','ALCOHOL_TEST_RESULT','POLICE-REPORTED_DRUG_INVOLVEMENT','METHOD_OF_DRUG_DETERMINATION','DRUG_TEST_RESULTS_(1_of_3)','DRUG_TEST_TYPE_(2_of_3)','DRUG_TEST_RESULTS_(2_of_3)','DRUG_TEST_TYPE_(3_of_3)','DRUG_TEST_RESULTS_(3_of_3)',

          'HISPANIC_ORIGIN','TAKEN_TO_HOSPITAL','RELATED_FACTOR_(1)-PERSON_LEVEL','RELATED_FACTOR_(2)-PERSON_LEVEL','RELATED_FACTOR_(3)-PERSON_LEVEL','RACE']

target = 'INJURY_SEVERITY'
#This is input which our classifier will use as an input.

train[features].head(10)
#Display first 10 target variables

train[target].head(10).values
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000, random_state=42)



# We train model

mlp.fit(train[features],train[target]) 





#Make predictions using the features from the test data set

predictions = mlp.predict(test[features])



#Display our predictions

predictions
submission = pd.DataFrame({'AGE':test['AGE'],'INJURY_SEVERITY':predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
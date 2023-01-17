import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#importing the relevant libraries necessary
#Basic settings

datatrain = pd.read_csv("../input/modeltrap/train.csv", low_memory = False, skipinitialspace=True)
datatest = pd.read_csv("../input/modeltrap/test.csv", low_memory = False, skipinitialspace=True)
pd.options.display.float_format = '{:,.2f}'.format #to standardize the formatting

datatrain.head()
#returns the first five rows of the training dataset - to check the data, features etc

datatest.head()
#returns the first five rows of the test dataset - to check the data, features etc
default_count = datatrain.default.sum()
#calculates the total number of defaults in the training set

default_percentage = default_count/len(datatrain)*100
#calculates the overall percentage of defaults within the entire training set 

print(f'The share of defaults in the training set is {default_percentage:.2f}%')
#prints the percentage value calculated previously, to 2 decimal places
max_default_zip = datatrain.groupby(by="ZIP").default.mean().idxmax()

print(f'The postcode that has the highest default rate is {max_default_zip}')
#Finds the postcode that has the highest number of defaults
default_train = datatrain.default[datatrain.year== 0].mean()

print(f'The Year 1 training set default rate is {default_train*100:.2f}%')
#Calculates the mean default rate within the first year of the training set
corr_age_income = datatrain.age.corr(datatrain.income)

print(f'The correlation between age and income in the training set is {corr_age_income *100:.2f}%')
#Calculates the correlation between age and income in the training dataset to two decimal places
predictors = ['ZIP', 'rent', 'education',  'income', 'loan_size', 'payment_timing', 'job_stability','occupation']
target = ['default']
#specifying which features to use within our predictive model to call upon in future use to simplify the code

X_train = pd.get_dummies(datatrain[predictors])
Y_train = datatrain[target]
#creating the dependent and independent variables, all of which come from the training dataset 

clf = RandomForestClassifier(n_jobs=-1,n_estimators=100,oob_score=True,random_state=42,max_depth=4)
#RandomForestClassifier is pulled from the sklearn library

clf.fit(X_train,Y_train)
#Estimating the coefficients of the Random Forest ie fitting the model

in_sample_score = accuracy_score(y_pred = clf.predict(X_train),y_true=Y_train)
#calculate the in sample accuracy score

print(f'The in sample accuracy is {in_sample_score*100:.2f}%')
oob_score = (clf.oob_score_)*100

print(f'The out of bag score of the model is {oob_score:.4f}%')
#calculats the OOB score to four decimal places 
X_test = pd.get_dummies(datatest[predictors])
Y_test = datatest[target].values

oos_score = clf.score(X_test, Y_test)*100
#Creating new variables for the test model 

print(f'The out of sample accuracy score is {oos_score:.2f}%')

#OOS score is much less that OOB and in-sample score
maj_dist = clf.predict(X_test[datatest.minority == 0])*100
#using the RandomForestClassifier within the clf variable to predict average default probability for non-minorities ie when the minority feature equals 0.

print(f'The predicted average default probability for non-minorities is {maj_dist.mean():.2f}%')
min_dist = clf.predict(X_test[datatest.minority == 1])
#performing a similar action as Q8 but for minorities ie when minority is equal to 1 

print(f'The predicted average default probability for minorities is {min_dist.mean()*100:.2f}%')
pred = clf.predict(X_test)

print(f'Share of successful applicants amongst: minority members: {(len(datatest[(~pred) & (datatest.minority==1)]) / len(datatest[datatest.minority==1]))*100:.2f}%, non-minority members: {(len(datatest[(~pred) & (datatest.minority==0)]) / len(datatest[datatest.minority==0]))*100:.2f}%')

print(f'Share of successful applicants amongst: females: {(len(datatest[(~pred) & (datatest.sex==1)]) / len(datatest[datatest.sex==1]))*100:.2f}%, males: {(len(datatest[(~pred) & (datatest.sex==0)]) / len(datatest[datatest.sex==0]))*100:.2f}%')
#Rejected applicants by sex: Male - 18.5%, Female - 18.9%. 
#There is therefore gender parity as the same proportions of M & F applicants received loans.
#However, this was not the case with minorities and non-minorities; minorities received a 6% approval rate, while non-minorities received a 31% approval rate. 
#Generating the array of predicted default stats
Y_oos_pred = clf.predict(X_test)
out = pd.Series(data=Y_oos_pred)
out = out.rename('defaults')

#Collecting the predictions & regressors & sex w/ minority in 1 dataset
data_q10 = pd.concat([out, X_test, datatest.sex, datatest.minority],axis=1)

#Looking for any interaction between minority status and gender:
data_q10.groupby(by=['sex','minority']).mean().defaults

#We can see that the model does not predict any additional increase in default rates as a result of the interaction between "minority" and "sex" i.e. no further discrimination
pred = pd.DataFrame(data = [clf.predict(X_test)]).T
#generating a dataframe for the prediction variable

y_test_df = pd.DataFrame(data = [datatest.default.tolist()]).T
#generate a dataframe for the observed defaults

x_test_df = pd.DataFrame(data = [datatest.minority.tolist(),datatest.sex.tolist()]).T
#generate a dataframe for the minority and gender status

aa = pd.concat([pred, y_test_df, x_test_df], ignore_index=True, axis=1)
#concatenating all three matrices together

aa.columns = ['pred','test','minority','sex']
#specify the column names

print(f'Share of granted loans amongst those who would repay: minority members: {(1 - aa["pred"][(aa["minority"] == 1) & (aa["test"]==0)].mean())*100:.2f}% \\\
non-minority: {(1 - aa["pred"][(aa["minority"] == 0) & (aa["test"]==0)].mean())*100:.2f}%')

print(f'Share of granted loans amongst those who would repay: female: {(1 - aa["pred"][(aa["sex"] == 1) & (aa["test"]==0)].mean())*100:.2f}% \
male: {(1 - aa["pred"][(aa["sex"] == 0) & (aa["test"]==0)].mean())*100:.2f}%')
#What do these shares indicate about the likelihood of default in different population groups that secured loans?

data_q12 = pd.concat([out, datatest.default, datatest.sex, datatest.minority],axis=1)

#We then create a sub-sample "paid" - ie the applicants that repaid their loan. 
#We'll then check whether the model correctly predicted that they would pay.

paid = data_q12[(data_q12.default==0)]

#Looking at predicted default rates among those applicants who did not default
paid.groupby(by='minority').mean().defaults

#The model has incorrectly rejected 12% of loan-worthy within the non-minority demographic and 30.5% loan-worthy applicants from a minority background.
#This violates equal opportunity.

#Looking at the same, but split by gender
paid.groupby(by='sex').mean().defaults

#Same effect present but of smaller magnitude - 20% of loan-worthy males were rejected vs 23% of loan-worthy females

approved = data_q12[(data_q12.defaults==0)]
approved.groupby(by='minority').mean().default
#using matplotlib to create a chart showing the varying importance of the features within the model

features = X_test.columns.values
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='c', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.show()
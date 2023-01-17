# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing Pandas and NumPy

import pandas as pd, numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Importing IBM HR datasets

HRdata = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

HRdata.head(5).transpose()
# Let's check the dimensions of the dataframe

HRdata.shape
# Let's see the type of each column

HRdata.info()
# summarising number of missing values in each column

HRdata.isnull().sum()
# percentage of missing values in each column

round(HRdata.isnull().sum()/len(HRdata.index), 2)*100
# missing values in rows

HRdata.isnull().sum(axis=1)
# checking whether some rows have more than 5 missing values

len(HRdata[HRdata.isnull().sum(axis=1) > 5].index)
#checking for redundant duplicate rows

print(sum(HRdata.duplicated()))

#Dropping Duplicate Rows

HRdata.drop_duplicates(keep=False,inplace=True)

print(sum(HRdata.duplicated()))
# Get the value counts of all the columns



for column in HRdata:

    print(HRdata[column].astype('category').value_counts())

    print('___________________________________________________')
#dropping columns having single value - EmployeeCount, StandardHours, Over18 

HRdata.drop(['EmployeeCount','Over18','StandardHours'], axis = 1, inplace = True)
# let's look at the statistical aspects of the numeric features in dataframe

HRdata.describe()
# let's look at the outliers for numeric features in dataframe

HRdata.describe(percentiles=[.25,.5,.75,.90,.95,.99]).transpose()
# Outlier treatment for Age 

Q1 = HRdata.Age.quantile(0.25)

Q3 = HRdata.Age.quantile(0.75)

IQR = Q3 - Q1

HRdata = HRdata[(HRdata.Age >= Q1 - 1.5*IQR) & (HRdata.Age <= Q3 + 1.5*IQR)]

# Outlier treatment for DailyRate 

Q1 = HRdata.DailyRate.quantile(0.25)

Q3 = HRdata.DailyRate.quantile(0.75)

IQR = Q3 - Q1

HRdata = HRdata[(HRdata.DailyRate >= Q1 - 1.5*IQR) & (HRdata.DailyRate <= Q3 + 1.5*IQR)]

# Outlier treatment for HourlyRate 

Q1 = HRdata.HourlyRate.quantile(0.25)

Q3 = HRdata.HourlyRate.quantile(0.75)

IQR = Q3 - Q1

HRdata = HRdata[(HRdata.HourlyRate >= Q1 - 1.5*IQR) & (HRdata.HourlyRate <= Q3 + 1.5*IQR)]

# Outlier treatment for TotalWorkingYears

Q1 = HRdata.TotalWorkingYears.quantile(0.25)

Q3 = HRdata.TotalWorkingYears.quantile(0.75)

IQR = Q3 - Q1

HRdata = HRdata[(HRdata.TotalWorkingYears >= Q1 - 1.5*IQR) & (HRdata.TotalWorkingYears <= Q3 + 1.5*IQR)]

# Outlier treatment for YearsAtCompany 

Q1 = HRdata.YearsAtCompany.quantile(0.25)

Q3 = HRdata.YearsAtCompany.quantile(0.75)

IQR = Q3 - Q1

HRdata = HRdata[(HRdata.YearsAtCompany >= Q1 - 1.5*IQR) & (HRdata.YearsAtCompany <= Q3 + 1.5*IQR)]

# Outlier treatment for YearsInCurrentRole 

Q1 = HRdata.YearsInCurrentRole.quantile(0.25)

Q3 = HRdata.YearsInCurrentRole.quantile(0.75)

IQR = Q3 - Q1

HRdata = HRdata[(HRdata.YearsInCurrentRole >= Q1 - 1.5*IQR) & (HRdata.YearsInCurrentRole <= Q3 + 1.5*IQR)]

# Outlier treatment for YearsSinceLastPromotion 

Q1 = HRdata.YearsSinceLastPromotion.quantile(0.25)

Q3 = HRdata.YearsSinceLastPromotion.quantile(0.75)

IQR = Q3 - Q1

HRdata = HRdata[(HRdata.YearsSinceLastPromotion >= Q1 - 1.5*IQR) & (HRdata.YearsSinceLastPromotion <= Q3 + 1.5*IQR)]

# Outlier treatment for YearsWithCurrManager 

Q1 = HRdata.YearsWithCurrManager.quantile(0.25)

Q3 = HRdata.YearsWithCurrManager.quantile(0.75)

IQR = Q3 - Q1

HRdata = HRdata[(HRdata.YearsWithCurrManager >= Q1 - 1.5*IQR) & (HRdata.YearsWithCurrManager <= Q3 + 1.5*IQR)]
# correlation matrix

cor = HRdata.corr()

cor
# Plotting correlations on a heatmap post outlier treatment

# figure size

plt.figure(figsize=(20,15))

# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()
# Plotting count of EducationField vs. Gender

sns.countplot(x = "EducationField", hue = "Gender", data = HRdata)
HRdata.shape
# List of binary variables with Yes/No values using map converting these to 1/0

varlist =  ['OverTime','Attrition']



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, 'No': 0})



# Applying the function to the leads score list

HRdata[varlist] = HRdata[varlist].apply(binary_map)
# Creating a dummy variable for the multilevel categorical variables and dropping the first one to remove redundancy.

dummy1 = pd.get_dummies(HRdata[['Gender','JobRole', 'EducationField', 'Department','MaritalStatus','BusinessTravel']], drop_first=True)



# Adding the results to the master dataframe

HRdata = pd.concat([HRdata, dummy1], axis=1)
# Dropping the repeated variables as we have created dummies for the below variables

HRdata = HRdata.drop(['Gender','JobRole', 'EducationField', 'Department','MaritalStatus','BusinessTravel'], 1)
print(HRdata.shape)

HRdata.head()
# Rechecking if all the levels of all the columns are OK (without any NaN or error)

for column in HRdata:

    print(HRdata[column].astype('category').value_counts())

    print('___________________________________________________')
from sklearn.model_selection import train_test_split

# Putting feature variables to X by first dropping y (Attrition) from HRdata

X = HRdata.drop(['Attrition'], axis=1)

# Putting response variable to y

y = HRdata['Attrition']

print(y.head())
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
X.columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',

       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',

       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',

       'OverTime', 'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',

       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',

       'Gender_Male', 'JobRole_Human Resources',

       'JobRole_Laboratory Technician', 'JobRole_Manager',

       'JobRole_Manufacturing Director', 'JobRole_Research Director',

       'JobRole_Research Scientist', 'JobRole_Sales Executive',

       'JobRole_Sales Representative', 'EducationField_Life Sciences',

       'EducationField_Marketing', 'EducationField_Medical',

       'EducationField_Other', 'EducationField_Technical Degree',

       'Department_Research & Development', 'Department_Sales',

       'MaritalStatus_Married', 'MaritalStatus_Single',

       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely']] = scaler.fit_transform(X_train[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',

       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',

       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',

       'OverTime', 'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',

       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',

       'Gender_Male', 'JobRole_Human Resources',

       'JobRole_Laboratory Technician', 'JobRole_Manager',

       'JobRole_Manufacturing Director', 'JobRole_Research Director',

       'JobRole_Research Scientist', 'JobRole_Sales Executive',

       'JobRole_Sales Representative', 'EducationField_Life Sciences',

       'EducationField_Marketing', 'EducationField_Medical',

       'EducationField_Other', 'EducationField_Technical Degree',

       'Department_Research & Development', 'Department_Sales',

       'MaritalStatus_Married', 'MaritalStatus_Single',

       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely']])



#verifying the scaled data in X_train dataframe

X_train.describe()
### Before we build the Logistic regression model, we need to know how much percent of attrition is seen in the original data

### Calculating the Attrition Rate

AttritionRate = round((sum(HRdata['Attrition'])/len(HRdata['Attrition'].index))*100,2)

AttritionRate
import statsmodels.api as sm

# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

from sklearn.feature_selection import RFE

rfe = RFE(logreg,20)             # running RFE with 20 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
#### Creating a dataframe with the actual Attrition flag and the predicted probabilities
y_train_pred_final = pd.DataFrame({'Attrition':y_train.values, 'Attrition_Probability':y_train_pred})

y_train_pred_final['EmployeeID'] = y_train.index

y_train_pred_final.head()
#### Creating new column 'Predicted' with 1 if Convert_Prob > 0.8 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Attrition_Probability.map(lambda x: 1 if x > 0.8 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Attrition, y_train_pred_final.predicted )

print(confusion)
# Predicted     not_attrition    attrition

# Actual

# not_attrition        675       1

# attrition            105      29 
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Attrition, y_train_pred_final.predicted))
#### Check for the VIF values of the feature variables.

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
##"YearsAtCompany" is high in VIF exceeding value of 5, so lets remove it

col = col.drop('YearsAtCompany', 1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)

y_train_pred[:10]
y_train_pred_final['Attrition_Probability'] = y_train_pred

# Creating new column 'predicted' with 1 if Convert_Prob > 0.8 else 0

y_train_pred_final['Predicted'] = y_train_pred_final.Attrition_Probability.map(lambda x: 1 if x > 0.8 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Attrition, y_train_pred_final.predicted))
## VIF AGAIN

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
## All VIF values are less than 5, so we are going to review p-values and drop features having highest p-values

col = col.drop(['JobRole_Manager','JobRole_Research Director','YearsInCurrentRole'], 1)

col

# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
## VIF AGAIN

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = res.predict(X_train_sm).values.reshape(-1)

y_train_pred[:10]
y_train_pred_final['Attrition_Probability'] = y_train_pred

# Creating new column 'predicted' with 1 if Convert_Prob > 0.8 else 0

y_train_pred_final['Predicted'] = y_train_pred_final.Attrition_Probability.map(lambda x: 1 if x > 0.8 else 0)

y_train_pred_final.head()
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Attrition, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Attrition, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

print("Sensitivity is:")

TP / float(TP+FN)
# Let us calculate specificity

print("Specificity is:")

TN / float(TN+FP)
# Calculate false postive rate - predicting Conversion when customer does not Convert

print("False Positive Rate is:")

print(FP/ float(TN+FP))
# positive predictive value 

print("Positive Predictive value is:")

print (TP / float(TP+FP))
# Negative predictive value

print("Negative Predictive value is:")

print (TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Attrition, y_train_pred_final.Attrition_Probability, drop_intermediate = False )
draw_roc(y_train_pred_final.Attrition, y_train_pred_final.Attrition_Probability)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Attrition_Probability.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Attrition, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Attrition_Probability.map( lambda x: 1 if x > 0.35 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Attrition, y_train_pred_final.final_predicted)
#### Accuracy of 86.9 ~ 87% indicates the model can 
confusion2 = metrics.confusion_matrix(y_train_pred_final.Attrition, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity - High specificity indicates the model can identify those who will not have attrition will have a negative test result.

TN / float(TN+FP)
# Calculate false postive rate - predicting Attrition when Employee is not Attrition

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Attrition, y_train_pred_final.predicted )

print(confusion)

#Precision

confusion[1,1]/(confusion[0,1]+confusion[1,1])
#Recall

confusion[1,1]/(confusion[1,0]+confusion[1,1])
## Using sklearn to calculate above

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_pred_final.Attrition, y_train_pred_final.predicted)
recall_score(y_train_pred_final.Attrition, y_train_pred_final.predicted)
from sklearn.metrics import precision_recall_curve

y_train_pred_final.Attrition, y_train_pred_final.predicted

p, r, thresholds = precision_recall_curve(y_train_pred_final.Attrition, y_train_pred_final.Attrition_Probability)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',

       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',

       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',

       'OverTime', 'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',

       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',

       'Gender_Male', 'JobRole_Human Resources',

       'JobRole_Laboratory Technician', 'JobRole_Manager',

       'JobRole_Manufacturing Director', 'JobRole_Research Director',

       'JobRole_Research Scientist', 'JobRole_Sales Executive',

       'JobRole_Sales Representative', 'EducationField_Life Sciences',

       'EducationField_Marketing', 'EducationField_Medical',

       'EducationField_Other', 'EducationField_Technical Degree',

       'Department_Research & Development', 'Department_Sales',

       'MaritalStatus_Married', 'MaritalStatus_Single',

       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely']] = scaler.transform(X_test[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber',

       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',

       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',

       'OverTime', 'PercentSalaryHike', 'PerformanceRating',

       'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',

       'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',

       'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',

       'Gender_Male', 'JobRole_Human Resources',

       'JobRole_Laboratory Technician', 'JobRole_Manager',

       'JobRole_Manufacturing Director', 'JobRole_Research Director',

       'JobRole_Research Scientist', 'JobRole_Sales Executive',

       'JobRole_Sales Representative', 'EducationField_Life Sciences',

       'EducationField_Marketing', 'EducationField_Medical',

       'EducationField_Other', 'EducationField_Technical Degree',

       'Department_Research & Development', 'Department_Sales',

       'MaritalStatus_Married', 'MaritalStatus_Single',

       'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely']])

X_test = X_test[col]

X_test.head()
X_test.columns
X_test_sm = sm.add_constant(X_test)

# Making predictions on the test set

y_test_pred = res.predict(X_test_sm)

y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)

# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Putting LeadID to index

y_test_df['LeadID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Attrition_Probability'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex_axis(['LeadID','Attrition','Attrition_Probability'], axis=1)
# Let's see the head of y_pred_final

y_pred_final
y_pred_final['final_predicted'] = y_pred_final.Attrition_Probability.map(lambda x: 1 if x > 0.4 else 0) #0.4 taken from cross over point in plot
y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Attrition, y_pred_final.final_predicted)
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
#calculating Lead Score for the predicted data

y_pred_final['Lead_Score'] = round((y_pred_final.Attrition_Probability*100),0)
y_pred_final.head()
#Finding all the Attrition leads from lead Score

AttritionLeads=y_pred_final[y_pred_final.final_predicted==1]

AttritionLeads
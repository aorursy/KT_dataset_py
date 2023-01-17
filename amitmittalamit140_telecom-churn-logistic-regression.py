#Importing all libraries

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler



#building model

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression



#model evaluation

from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import metrics

from sklearn.metrics import precision_recall_curve



#model validation

from sklearn.model_selection import train_test_split



#visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')



# Data display coustomization

pd.set_option('display.max_columns', 100)
telecom = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Now we have one data frame consisting all data. Now we will see first five rows of the new data frame

telecom.head()
#prinitng shape of the dataset

r,c = telecom.shape

print(f"Shape of telecom dataset: {telecom.shape}")

print(f"Number of rows: {r}")

print(f"Number of columns: {c}")
# let's look at the some statistics of the dataframe

telecom.describe()
# Let's look at the data type of each feature

telecom.info()
# Defining method to convert them

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



bin_var =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']



# Applying the method to the data frame

telecom[bin_var] = telecom[bin_var].apply(binary_map)
#creating dummies for gender and dropping first column because single column can capture the whole data

gender = pd.get_dummies(telecom['gender'], drop_first=True)



# Merging the above results with telecom data frame 

telecom = pd.concat([telecom, gender], axis=1)
#printing first 5 rows of the data frame after converting Binary variables

telecom.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(telecom[['Contract', 'PaymentMethod', 'InternetService']], drop_first=True)



# Adding the results to the telecom dataframe

telecom = pd.concat([telecom, dummy1], axis=1)
telecom.head()
# Creating dummy variables for the remaining categorical variables and dropping the level with big names.



# Creating dummy variables for the variable 'MultipleLines'

ml = pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')

# Dropping MultipleLines_No phone service column

ml1 = ml.drop(['MultipleLines_No phone service'], 1)

# Adding the results to the telecom dataframe

telecom = pd.concat([telecom,ml1], axis=1)



# Creating dummy variables for the variable 'OnlineSecurity'.

os = pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')

os1 = os.drop(['OnlineSecurity_No internet service'], 1)

# Adding the results to the telecom dataframe

telecom = pd.concat([telecom,os1], axis=1)



# Creating dummy variables for the variable 'OnlineBackup'.

ob = pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')

ob1 = ob.drop(['OnlineBackup_No internet service'], 1)

# Adding the results to the telecom dataframe

telecom = pd.concat([telecom,ob1], axis=1)



# Creating dummy variables for the variable 'DeviceProtection'. 

dp = pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')

dp1 = dp.drop(['DeviceProtection_No internet service'], 1)

# Adding the results to the telecom dataframe

telecom = pd.concat([telecom,dp1], axis=1)



# Creating dummy variables for the variable 'TechSupport'. 

ts = pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')

ts1 = ts.drop(['TechSupport_No internet service'], 1)

# Adding the results to the telecom dataframe

telecom = pd.concat([telecom,ts1], axis=1)



# Creating dummy variables for the variable 'StreamingTV'.

st =pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')

st1 = st.drop(['StreamingTV_No internet service'], 1)

# Adding the results to the telecom dataframe

telecom = pd.concat([telecom,st1], axis=1)



# Creating dummy variables for the variable 'StreamingMovies'. 

ssm = pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')

ssm1 = ssm.drop(['StreamingMovies_No internet service'], 1)

# Adding the results to the telecom dataframe

telecom = pd.concat([telecom,ssm1], axis=1)
telecom.InternetService.value_counts()
telecom.PhoneService.value_counts()
telecom.head()
# We have created dummies for the below variables, so we can drop them

telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)
#The varaible TotalCharges is of String data type so converting it into float type

telecom['TotalCharges'] = pd.to_numeric(telecom["TotalCharges"].replace(" ",""),downcast="float")
#checking data types of variables 

telecom.info()
#Plot Box Plot for all there continuous variables 



plt.figure(figsize=(15,3))

plt.subplot(1,3,1)

sns.boxplot(telecom[["tenure"]])

plt.title("Tenure",size=15)



plt.subplot(1,3,2)

sns.boxplot(telecom[["MonthlyCharges"]])

plt.title("MonthlyCharges",size=15)



plt.subplot(1,3,3)

sns.boxplot(telecom[["TotalCharges"]])

plt.title("TotalCharges",size=15)
# Adding up the missing values (column-wise)

telecom.isnull().sum()
# Removing NaN TotalCharges rows

telecom = telecom[~np.isnan(telecom['TotalCharges'])]
# Checking again for missing values (column-wise)

telecom.isnull().sum()
# Correlation Matrix

plt.figure(figsize = (20,10))

sns.heatmap(round(telecom.corr(),1),annot = True)

plt.show()
#dropping the highly correlated variables



telecom.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',

                       'StreamingTV_No','StreamingMovies_No'],axis=1,inplace=True)
#customerID column is of no use for the model so we drop that column also

X = telecom.drop(['Churn','customerID'], axis=1)

X.head()
y = telecom['Churn']

y.head()
# Splitting the dataset into training set and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
#we will standardize continous variables only and not categorical variables

sc = StandardScaler()

sc.fit(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train[['tenure','MonthlyCharges','TotalCharges']] = sc.transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()
### Checking the 

p = (sum(y_train)/len(y_train))

print(f"p: {p}")



k = X_train.shape[1]

print(f"k: {k}")



N = 10 * k / p

print(f"N: {int(N)}")
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
logreg = LogisticRegression()

rfe = RFE(logreg, 15)

rfe = rfe.fit(X_train, y_train)
#top 15 columns returned by RFE

col = X_train.columns[rfe.support_]

col
#building the model with top 15 features which we got from RFE

X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set and showing first 10 predictions in terms of probabilities

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
#reshaping the predicted array

y_train_pred = y_train_pred.values.reshape(-1)
#creating data frame with actual churn and predicted probablilities

y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})

y_train_pred_final['CustID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
#Checking Accuracy of the model

print("Accuracy (Training Set): ",round(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted),4))
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#We will drop variables one by one, droping MonthlyCharges column

col = col.drop('MonthlyCharges',1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)

y_train_pred_final['Churn_Prob'] = y_train_pred

y_train_pred[:10]
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print("Accuracy (Training Set): ",round(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted),4))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#We are droping MultipleLines_Yes variable as it is insignificant

#We give priority to p-value than VIF

col = col.drop('MultipleLines_Yes',1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)

y_train_pred_final['Churn_Prob'] = y_train_pred

y_train_pred[:10]
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print("Accuracy (Training Set): ",round(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted),4))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Let's drop TotalCharges since it has a high VIF

col = col.drop('TotalCharges')

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)

y_train_pred_final['Churn_Prob'] = y_train_pred

y_train_pred[:10]
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print("Accuracy (Training Set): ",round(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted),4))
vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Let's take a look at the confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )

confusion
print("Predicted     |  Not Churn (0)  |  Churn (1)")

print("Actual        |                 | ")

print("--------------|-----------------|----------------")

print("Not Churn (0) |     3270        |     365")

print("--------------|-----------------|----------------")

print("Churn     (1) |      604        |     683")
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
#Accuracy of the final model

accuracy = (TN + TP)/float(TN+FN+TP+FP)

print("Accuracy of the model: ",round(accuracy,3))



# Sensitivity of the final model

sensitivity = TP / float(TP+FN)

print("Sensitivity of the model: ",round(sensitivity,3))



# Specificity of the final model

specificity = TN / float(TN+FP)

print("Specificity of the model: ",round(specificity,3))
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['Probability','Accuracy','Sensitivity','Specificity'])



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])

plt.xlabel("Thresh-hold")

plt.ylabel("Scores")

plt.title("Sensitivity and Specificity Trade-off",size=15)

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.3 else 0)



y_train_pred_final.head()
confusion2 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.final_predicted )

confusion2
print("Predicted     |  Not Churn (0)  |  Churn (1)")

print("Actual        |                 | ")

print("--------------|-----------------|----------------")

print("Not Churn (0) |     2787        |     848")

print("--------------|-----------------|----------------")

print("Churn     (1) |      288        |     999")
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
#Accuracy of the final model

accuracy = (TN + TP)/float(TN+FN+TP+FP)

print("Accuracy of the model: ",round(accuracy,3))



# Sensitivity of the final model

sensitivity = TP / float(TP+FN)

print("Sensitivity of the model: ",round(sensitivity,3))



# Specificity of the final model

specificity = TN / float(TN+FP)

print("Specificity of the model: ",round(specificity,3))
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

    #plt.savefig("E:/1. NITW/Project 4th Sem/ROC Curve.jpg")

    plt.show()

    

    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Churn_Prob, drop_intermediate = False )



draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)
# Precision of the final model

precision = TP / float(TP+FP)

print("Precision of the model: ",round(precision,3))



# Recall of the final model

recall = TP / float(TP+FN)

print("Recall of the model: ",round(recall,3))
p, r, thresholds = precision_recall_curve(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)
plt.plot(thresholds, p[:-1], "g-",label="Precision")

plt.plot(thresholds, r[:-1], "r-",label="Recall")

plt.xlabel("Thresh-hold")

plt.ylabel("Scores")

plt.title("Precision and Recall Trade-off",size=15)

plt.legend()

plt.show()
X_test[['tenure','MonthlyCharges','TotalCharges']] = sc.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])
X_test = X_test[col]

X_test.head()
X_test_sm = sm.add_constant(X_test)

y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Putting CustID to index

y_test_df['CustID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Churn_Prob'})
# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.3 else 0)
y_pred_final.head()
confusion2 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )

confusion2
print("Predicted     |  Not Churn (0)  |  Churn (1)")

print("Actual        |                 | ")

print("--------------|-----------------|----------------")

print("Not Churn (0) |     1144        |     384")

print("--------------|-----------------|----------------")

print("Churn     (1) |      163        |     419")
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
#Accuracy of the final model

accuracy = (TN + TP)/float(TN+FN+TP+FP)

print("Accuracy of the model: ",round(accuracy,3))



# Sensitivity of the final model

sensitivity = TP / float(TP+FN)

print("Sensitivity of the model: ",round(sensitivity,3))



# Specificity of the final model

specificity = TN / float(TN+FP)

print("Specificity of the model: ",round(specificity,3))
model  = pd.DataFrame({"Features": X_train_sm.columns,"Coefficient":res.params.values})

model["Odds_Ratio"] = model["Coefficient"].apply(lambda x: np.exp(x))

model[["Coefficient","Odds_Ratio"]] = model[["Coefficient","Odds_Ratio"]].apply(lambda x: round(x,2))

model["Perc_Impact"] = model["Odds_Ratio"].apply(lambda x: (x-1)*100)

model
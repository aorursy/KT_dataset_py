# Importing Pandas and NumPy

import pandas as pd

import numpy as np
# Importing all datasets

churn_data = pd.read_csv("../input/churn_data.csv")

customer_data = pd.read_csv("../input/customer_data.csv")

internet_data = pd.read_csv("../input/internet_data.csv")
#Merging on 'customerID'

df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')
#Final dataframe with all predictor variables

telecom = pd.merge(df_1, internet_data, how='inner', on='customerID')
# Let's see the head of our master dataset

telecom.head()
telecom
telecom.describe()
# Let's see the type of each column

telecom.info()
# Converting Yes to 1 and No to 0

telecom['PhoneService'] = telecom['PhoneService'].map({'Yes': 1, 'No': 0})

telecom['PaperlessBilling'] = telecom['PaperlessBilling'].map({'Yes': 1, 'No': 0})

telecom['Churn'] = telecom['Churn'].map({'Yes': 1, 'No': 0})

telecom['Partner'] = telecom['Partner'].map({'Yes': 1, 'No': 0})

telecom['Dependents'] = telecom['Dependents'].map({'Yes': 1, 'No': 0})
# Creating a dummy variable for the variable 'Contract' and dropping the first one.

cont = pd.get_dummies(telecom['Contract'],prefix='Contract',drop_first=True)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,cont],axis=1)



# Creating a dummy variable for the variable 'PaymentMethod' and dropping the first one.

pm = pd.get_dummies(telecom['PaymentMethod'],prefix='PaymentMethod',drop_first=True)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,pm],axis=1)



# Creating a dummy variable for the variable 'gender' and dropping the first one.

gen = pd.get_dummies(telecom['gender'],prefix='gender',drop_first=True)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,gen],axis=1)



# Creating a dummy variable for the variable 'MultipleLines' and dropping the first one.

ml = pd.get_dummies(telecom['MultipleLines'],prefix='MultipleLines')

#  dropping MultipleLines_No phone service column

ml1 = ml.drop(['MultipleLines_No phone service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,ml1],axis=1)



# Creating a dummy variable for the variable 'InternetService' and dropping the first one.

iser = pd.get_dummies(telecom['InternetService'],prefix='InternetService',drop_first=True)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,iser],axis=1)



# Creating a dummy variable for the variable 'OnlineSecurity'.

os = pd.get_dummies(telecom['OnlineSecurity'],prefix='OnlineSecurity')

os1= os.drop(['OnlineSecurity_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,os1],axis=1)



# Creating a dummy variable for the variable 'OnlineBackup'.

ob =pd.get_dummies(telecom['OnlineBackup'],prefix='OnlineBackup')

ob1 =ob.drop(['OnlineBackup_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,ob1],axis=1)



# Creating a dummy variable for the variable 'DeviceProtection'. 

dp =pd.get_dummies(telecom['DeviceProtection'],prefix='DeviceProtection')

dp1 = dp.drop(['DeviceProtection_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,dp1],axis=1)



# Creating a dummy variable for the variable 'TechSupport'. 

ts =pd.get_dummies(telecom['TechSupport'],prefix='TechSupport')

ts1 = ts.drop(['TechSupport_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,ts1],axis=1)



# Creating a dummy variable for the variable 'StreamingTV'.

st =pd.get_dummies(telecom['StreamingTV'],prefix='StreamingTV')

st1 = st.drop(['StreamingTV_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,st1],axis=1)



# Creating a dummy variable for the variable 'StreamingMovies'. 

sm =pd.get_dummies(telecom['StreamingMovies'],prefix='StreamingMovies')

sm1 = sm.drop(['StreamingMovies_No internet service'],1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,sm1],axis=1)
#telecom['MultipleLines'].value_counts()
# We have created dummies for the below variables, so we can drop them

telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)
#The varaible was imported as a string we need to convert it to float

telecom['TotalCharges'] =telecom['TotalCharges'].convert_objects(convert_numeric=True)

#telecom['tenure'] = telecom['tenure'].astype(int).astype(float)
telecom.info()
# Checking for outliers in the continuous variables

num_telecom = telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]
# Checking outliers at 25%,50%,75%,90%,95% and 99%

num_telecom.describe(percentiles=[.25,.5,.75,.90,.95,.99])
# Adding up the missing values (column-wise)

telecom.isnull().sum()
# Checking the percentage of missing values

round(100*(telecom.isnull().sum()/len(telecom.index)), 2)
# Removing NaN TotalCharges rows

telecom = telecom[~np.isnan(telecom['TotalCharges'])]
# Checking percentage of missing values after removing the missing values

round(100*(telecom.isnull().sum()/len(telecom.index)), 2)
# Normalising continuous features

df = telecom[['tenure','MonthlyCharges','TotalCharges']]
normalized_df=(df-df.mean())/df.std()
telecom = telecom.drop(['tenure','MonthlyCharges','TotalCharges'], 1)
telecom = pd.concat([telecom,normalized_df],axis=1)
telecom
churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100
churn
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = telecom.drop(['Churn','customerID'],axis=1)



# Putting response variable to y

y = telecom['Churn']
y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Let's see the correlation matrix 

plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(telecom.corr(),annot = True)
X_test2 = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No'],1)

X_train2 = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No','StreamingTV_No','StreamingMovies_No'],1)
plt.figure(figsize = (20,10))

sns.heatmap(X_train2.corr(),annot = True)
logm2 = sm.GLM(y_train,(sm.add_constant(X_train2)), family = sm.families.Binomial())

logm2.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

from sklearn.feature_selection import RFE

rfe = RFE(logreg, 13)             # running RFE with 13 variables as output

rfe = rfe.fit(X,y)

print(rfe.support_)           # Printing the boolean results

print(rfe.ranking_)           # Printing the ranking
# Variables selected by RFE 

col = ['PhoneService', 'PaperlessBilling', 'Contract_One year', 'Contract_Two year',

       'PaymentMethod_Electronic check','MultipleLines_No','InternetService_Fiber optic', 'InternetService_No',

       'OnlineSecurity_Yes','TechSupport_Yes','StreamingMovies_No','tenure','TotalCharges']
# Let's run the model using the selected variables

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logsk = LogisticRegression()

logsk.fit(X_train[col], y_train)
#Comparing the model with StatsModels

logm4 = sm.GLM(y_train,(sm.add_constant(X_train[col])), family = sm.families.Binomial())

logm4.fit().summary()
# UDF for calculating vif value

def vif_cal(input_data, dependent_col):

    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])

    x_vars=input_data.drop([dependent_col], axis=1)

    xvar_names=x_vars.columns

    for i in range(0,xvar_names.shape[0]):

        y=x_vars[xvar_names[i]] 

        x=x_vars[xvar_names.drop(xvar_names[i])]

        rsq=sm.OLS(y,x).fit().rsquared  

        vif=round(1/(1-rsq),2)

        vif_df.loc[i] = [xvar_names[i], vif]

    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)
telecom.columns

['PhoneService', 'PaperlessBilling', 'Contract_One year', 'Contract_Two year',

       'PaymentMethod_Electronic check','MultipleLines_No','InternetService_Fiber optic', 'InternetService_No',

       'OnlineSecurity_Yes','TechSupport_Yes','StreamingMovies_No','tenure','TotalCharges']
# Calculating Vif value

vif_cal(input_data=telecom.drop(['customerID','SeniorCitizen', 'Partner', 'Dependents',

                                 'PaymentMethod_Credit card (automatic)','PaymentMethod_Mailed check',

                                 'gender_Male','MultipleLines_Yes','OnlineSecurity_No','OnlineBackup_No',

                                 'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_Yes',

                                 'TechSupport_No','StreamingTV_No','StreamingTV_Yes','StreamingMovies_Yes',

                                 'MonthlyCharges'], axis=1), dependent_col='Churn')
col = ['PaperlessBilling', 'Contract_One year', 'Contract_Two year',

       'PaymentMethod_Electronic check','MultipleLines_No','InternetService_Fiber optic', 'InternetService_No',

       'OnlineSecurity_Yes','TechSupport_Yes','StreamingMovies_No','tenure','TotalCharges']
logm5 = sm.GLM(y_train,(sm.add_constant(X_train[col])), family = sm.families.Binomial())

logm5.fit().summary()
# Calculating Vif value

vif_cal(input_data=telecom.drop(['customerID','PhoneService','SeniorCitizen', 'Partner', 'Dependents',

                                 'PaymentMethod_Credit card (automatic)','PaymentMethod_Mailed check',

                                 'gender_Male','MultipleLines_Yes','OnlineSecurity_No','OnlineBackup_No',

                                 'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_Yes',

                                 'TechSupport_No','StreamingTV_No','StreamingTV_Yes','StreamingMovies_Yes',

                                 'MonthlyCharges'], axis=1), dependent_col='Churn')
# Let's run the model using the selected variables

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logsk = LogisticRegression()

logsk.fit(X_train[col], y_train)
# Predicted probabilities

y_pred = logsk.predict_proba(X_test[col])
# Converting y_pred to a dataframe which is an array

y_pred_df = pd.DataFrame(y_pred)
# Converting to column dataframe

y_pred_1 = y_pred_df.iloc[:,[1]]
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

y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 1 : 'Churn_Prob'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex_axis(['CustID','Churn','Churn_Prob'], axis=1)
# Let's see the head of y_pred_final

y_pred_final.head()
# Creating new column 'predicted' with 1 if Churn_Prob>0.5 else 0

y_pred_final['predicted'] = y_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.5 else 0)
# Let's see the head

y_pred_final.head()
from sklearn import metrics
help(metrics.confusion_matrix)
# Confusion matrix 

confusion = metrics.confusion_matrix( y_pred_final.Churn, y_pred_final.predicted )

confusion
# Predicted     not_churn    churn

# Actual

# not_churn        1326      166

# churn            249       333  
#Let's check the overall accuracy.

metrics.accuracy_score( y_pred_final.Churn, y_pred_final.predicted)
TP = confusion[0,0] # true positive 

TN = confusion[1,1] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(6, 4))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return fpr, tpr, thresholds
draw_roc(y_pred_final.Churn, y_pred_final.predicted)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_pred_final[i]= y_pred_final.Churn_Prob.map( lambda x: 1 if x > i else 0)

y_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix( y_pred_final.Churn, y_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    sensi = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    speci = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.3 else 0)
y_pred_final.head()
#Let's check the overall accuracy.

metrics.accuracy_score( y_pred_final.Churn, y_pred_final.final_predicted)
metrics.confusion_matrix( y_pred_final.Churn, y_pred_final.final_predicted )
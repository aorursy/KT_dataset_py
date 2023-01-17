# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
# Importing Pandas and NumPy

import pandas as pd, numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Importing Breast Cancer datasets

BCdata = pd.read_csv('../input/breastcancerdata.csv')

BCdata.head(5).transpose()
# Let's check the dimensions of the dataframe

print(BCdata.shape)

# Let's see the type of each column

print(BCdata.info())
# summarising number of missing values in each column

BCdata.isnull().sum()
# summarising number of missing values in each row

BCdata.isnull().sum(axis=1)
#checking for redundant duplicate rows

print(sum(BCdata.duplicated()))

#Dropping Duplicate Rows

BCdata.drop_duplicates(keep=False,inplace=True)

print(sum(BCdata.duplicated()))
#dropping columns having null value "Unnamed:32"

BCdata.drop(['Unnamed: 32'], axis = 1, inplace = True)
# let's look at the outliers for numeric features in dataframe

BCdata.describe(percentiles=[.25,.5,.75,.90,.95,.99]).transpose()
# correlation matrix

cor = BCdata.corr()

cor
# Plotting correlations on a heatmap post outlier treatment

# figure size

plt.figure(figsize=(20,15))

# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#pairplots for numerical data frames

plt.figure(figsize=(20,12))

sns.pairplot(BCdata)

plt.show()
# List of binary variables with M/B values using map converting these to 1/0

varlist =  ['diagnosis']



# Defining the map function

def binary_map(x):

    return x.map({'M': 1, 'B': 0})



# Applying the function to the leads score list

BCdata[varlist] = BCdata[varlist].apply(binary_map)
from sklearn.model_selection import train_test_split

# Putting feature variables to X by first dropping y (Attrition) from HRdata

X = BCdata.drop(['diagnosis'], axis=1)

# Putting response variable to y

y = BCdata['diagnosis']

print(y.head())
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
X.columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train[['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',

       'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']] = scaler.fit_transform(X_train[['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',

       'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']])



#verifying the scaled data in X_train dataframe

X_train.describe()
### Before we build the Logistic regression model, we need to know how much percent of Diagnosis as Malign is seen in the original data

### Calculating the Diagnosis Rate

DiagnosisRate = round((sum(BCdata['diagnosis'])/len(BCdata['diagnosis'].index))*100,2)

DiagnosisRate
import statsmodels.api as sm

# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

from sklearn.feature_selection import RFE

rfe = RFE(logreg,10)             # running RFE with 20 variables as output

rfe = rfe.fit(X_train, y_train)
rfe.support_

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train.columns[~rfe.support_]
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
#### Check for the VIF values of the feature variables.

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
##Removing all features showing high value in VIF exceeding value of 5, as this indicates high multi collinearity

col = col.drop('radius_worst',1)

col
#,'perimeter_mean','perimeter_worst','area_mean','area_worst','radius_se','perimeter_se'
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
##Removing all features showing high value in VIF exceeding value of 5, as this indicates high multi collinearity

col = col.drop('perimeter_worst',1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
## VIF AGAIN

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
##Removing all features showing high value in VIF exceeding value of 5, as this indicates high multi collinearity

col = col.drop('area_se',1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
## VIF AGAIN

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
##Removing all features showing high value in VIF exceeding value of 5, as this indicates high multi collinearity

col = col.drop('concave points_worst',1)

col
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train[col])

logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm6.fit()

res.summary()
## VIF AGAIN

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Diagnosis':y_train.values, 'Diagnosis_Probability':y_train_pred})

y_train_pred_final['PatientID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Diagnosis_Probability.map(lambda x: 1 if x > 0.8 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Diagnosis, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Diagnosis, y_train_pred_final.predicted))
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Diagnosis, y_train_pred_final.Diagnosis_Probability, drop_intermediate = False )
draw_roc(y_train_pred_final.Diagnosis, y_train_pred_final.Diagnosis_Probability)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Diagnosis_Probability.map(lambda x: 1 if x > i else 0)

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

    cm1 = metrics.confusion_matrix(y_train_pred_final.Diagnosis, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Diagnosis_Probability.map( lambda x: 1 if x > 0.35 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Diagnosis, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Diagnosis, y_train_pred_final.final_predicted )

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
#Precision

confusion[1,1]/(confusion[0,1]+confusion[1,1])
#Recall

confusion[1,1]/(confusion[1,0]+confusion[1,1])
## Using sklearn to calculate above

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_pred_final.Diagnosis, y_train_pred_final.predicted)
recall_score(y_train_pred_final.Diagnosis, y_train_pred_final.predicted)
from sklearn.metrics import precision_recall_curve

y_train_pred_final.Diagnosis, y_train_pred_final.predicted

p, r, thresholds = precision_recall_curve(y_train_pred_final.Diagnosis, y_train_pred_final.Diagnosis_Probability)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_test[['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',

       'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']] = scaler.fit_transform(X_test[['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',

       'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']])



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

y_test_df['PatientID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Diagnosis_Probability'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex_axis(['PatientID','diagnosis','Diagnosis_Probability'], axis=1)
# Let's see the head of y_pred_final

y_pred_final
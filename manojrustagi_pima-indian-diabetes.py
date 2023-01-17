# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/diabetes.csv")
df.shape
df.dtypes
df.isnull().sum()
df.head()
df['Pregnancies'].value_counts().plot.bar()
plt.figure(1)



plt.subplot(121,title='Glucose Distribution')



sns.distplot(df['Glucose'])



print("Skewness: %f" % df['Glucose'].skew())



plt.subplot(122, title='Glucose - Box Plot ')

df['Glucose'].plot.box(figsize=(16,5))



plt.show()
plt.figure(1)



plt.subplot(121,title='BloodPressure Distribution')



sns.distplot(df['BloodPressure'])



print("Skewness: %f" % df['BloodPressure'].skew())



plt.subplot(122, title='BloodPressure - Box Plot ')

df['BloodPressure'].plot.box(figsize=(16,5))



plt.show()
plt.figure(1)



plt.subplot(121,title='SkinThickness Distribution')



sns.distplot(df['SkinThickness'])



print("Skewness: %f" % df['SkinThickness'].skew())



plt.subplot(122, title='SkinThickness - Box Plot ')

df['SkinThickness'].plot.box(figsize=(16,5))



plt.show()
plt.figure(1)



plt.subplot(121,title='Insulin Distribution')



sns.distplot(df['Insulin'])



print("Skewness: %f" % df['Insulin'].skew())



plt.subplot(122, title='Insulin - Box Plot ')

df['Insulin'].plot.box(figsize=(16,5))



plt.show()
plt.figure(1)



plt.subplot(121,title='BMI Distribution')



sns.distplot(df['BMI'])



print("Skewness: %f" % df['BMI'].skew())



plt.subplot(122, title='BMI - Box Plot ')

df['BMI'].plot.box(figsize=(16,5))



plt.show()
plt.figure(1)



plt.subplot(121,title='DiabetesPedigreeFunction Distribution')



sns.distplot(df['DiabetesPedigreeFunction'])



print("Skewness: %f" % df['DiabetesPedigreeFunction'].skew())



plt.subplot(122, title='DiabetesPedigreeFunction - Box Plot ')

df['DiabetesPedigreeFunction'].plot.box(figsize=(16,5))



plt.show()
df['Age'].value_counts().plot.pie()
def plot_bar(df,stack=False,displayVal=True):

    ax = df.plot(kind='bar',figsize=(16,5),stacked=stack) 

    if displayVal:

        for p in ax.patches:

            h=round(p.get_height(),2)

            x=round(p.get_x(),2)

            ax.annotate(str(h), (x, h))
#creating bins for the field

bins=[15,30,45,60,75,90]

group=['0-15','15-30','30-45', '60-75','>75']

df['age_bin']=pd.cut(df['Age'],bins,labels=group)

#print(income_df_95)
#Checking the how age of the person is impacting the diabetes status.

x=pd.crosstab(df['age_bin'],df['Outcome'])

x=x.div(x.sum(1).astype(float), axis=0)

plot_bar(x,stack=False)
#Checking the how Pregnancies of the person is impacting the diabetes status.

x=pd.crosstab(df['Pregnancies'],df['Outcome'])

x=x.div(x.sum(1).astype(float), axis=0)

plot_bar(x,stack=False)
corr=df.corr()

fig, ax = plt.subplots(figsize=(15,15)) 

sns.heatmap(corr,annot=True,linewidths=.5, ax=ax,fmt='.2f')
df.head(10)
#Lets remove the column age-bin

df.drop('age_bin',axis=1,inplace=True)
df.head(10)
df.describe()
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X=df.drop('Outcome',axis=1)

X.head(10)
# Putting response variable to y

y=df['Outcome']

y.head(5)
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.70,test_size=0.20,random_state=100)
X_train.columns
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()

X_train[X_train.columns]=scaler.fit_transform(X_train[X_train.columns])
X_train.head()
import statsmodels.api as sm
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 6)             # running RFE with 13 variables as output

rfe = rfe.fit(X_train, y_train)
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
y_train_pred_final = pd.DataFrame({'Outcome':y_train.values, 'Outcome_Prob':y_train_pred})

y_train_pred_final['Row_id'] = y_train.index

y_train_pred_final.head()
##### Creating new column 'predicted' with 1 if Outcome_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Outcome_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Outcome, y_train_pred_final.predicted )

print(confusion)
### Predicted     Diabetes(-)    Diabetes(+)

### Actual

### Diabetes(-)       307        44

### Diabetes(+)        75        111
TN=307  #True -Ve

FN=44   #False -ve

FP=75   #False +ve

TP=111  #True +ve



accuracy=((TP+TN)/(TP+TN+FP+FN))

recall=((TP)/(TP+FP))

precision = ((TP)/(FN+TP))

print(accuracy)

print(recall)

print(precision)
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Outcome, y_train_pred_final.predicted)
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_pred_final.Outcome, y_train_pred_final.predicted)
recall_score(y_train_pred_final.Outcome, y_train_pred_final.predicted)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Outcome, y_train_pred_final.Outcome_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Outcome, y_train_pred_final.Outcome_Prob)
# Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Outcome_Prob.map(lambda x: 1 if x > i else 0)

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

    cm1 = metrics.confusion_matrix(y_train_pred_final.Outcome, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Outcome_Prob.map( lambda x: 1 if x > 0.3 else 0)



y_train_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Outcome, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Outcome, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_train_pred_final.Outcome, y_train_pred_final.Outcome_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
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

y_test_df['row_id'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Outcome_Prob'})
# Rearranging the columns

y_pred_final = y_pred_final.reindex_axis(['row_id','Outcome','Outcome_Prob'], axis=1)
# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final.Outcome_Prob.map(lambda x: 1 if x > 0.40 else 0)
y_pred_final.head()
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Outcome, y_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Outcome, y_train_pred_final.final_predicted )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
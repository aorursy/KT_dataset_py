# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')



#import pandas 

import pandas as pd



#import numpy

import numpy as np



#import seaborn for visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px



#import statsmodel 

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor



#import sklearn libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_curve,roc_auc_score
# Changing to show more rows for visual analysis

pd.set_option('display.max_rows', 500)

# Changing display format to not show scientific notation

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Displaying all columns

pd.set_option('display.max_columns', 500)
#import data

data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.shape
data.describe()
#Function to calculate the missing value percent in DataFrame columns - As we are goining to do it frequently

def missingValues(df):

   missingcontent=round(df.isnull().sum()/len(df) *100,2)

   print("Total Missing Value Percentage in dataframe: ",round(missingcontent.mean(),2))

   print(missingcontent[missingcontent>0].sort_values(ascending=False))
#null value percentage in data

missingValues(data)
data.drop(columns=['id','Unnamed: 32'],inplace=True)
#null value percentage in data

missingValues(data)
# 'Diagnosis' count details

data['diagnosis'].value_counts()
sns.barplot(y=data['diagnosis'].value_counts(),x=data['diagnosis'].unique(),palette="pastel")
plt.figure(figsize=(20,20))

sns.heatmap(data.corr(),cmap='YlGnBu',annot=True)

plt.show()
# generate a pair plot with the "mean" columns alone

cols = ['diagnosis',

        'radius_mean', 

        'texture_mean', 

        'perimeter_mean', 

        'area_mean', 

        'smoothness_mean', 

        'compactness_mean', 

        'concavity_mean',

        'concave points_mean', 

        'symmetry_mean', 

        'fractal_dimension_mean']



sns.pairplot(data=data[cols], hue='diagnosis', palette='RdBu')
#cols to be dropped inorder to handle the multicollinearity between the variables

cols= ['perimeter_se', 'area_se',

       'perimeter_mean','area_mean',

       'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst',

       'concavity_mean',

       'concave points_mean',

       'concavity_se', 'concave points_se']
data.drop(columns=cols,axis=1,inplace=True)
#Our final columns for our model

data.columns
#Binary map of 'M' & 'B' values in the diagnosis column 

data['diagnosis'] = data['diagnosis'].map({'B':0,'M':1})
data.describe()
y = data.pop('diagnosis')

X= data
X.head()
# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)



X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 100)
print('X_Train Dataset: ',X_train.shape)

print('y_Train Dataset: ',y_train.shape)

print('X_Test Dataset: ',X_test.shape)

print('y_Test Dataset: ',y_test.shape)
#data normalization using sklearn MinMaxScaler

scaler = MinMaxScaler()
cols = X_train.columns
X_train[cols] = scaler.fit_transform(X_train[cols])

X_test[cols] = scaler.transform(X_test[cols])
X_train.describe()
# Logistic regression model using statsmodel

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
#Function to calculate VIF values

def VIF_values(X_train):

    vif = pd.DataFrame()

    X= X_train

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    print(vif)
VIF_values(X_train)
#dropping the 'smoothness_mean' from the model

X_train = X_train.drop(columns='smoothness_mean',axis=1)
X_train_sm = sm.add_constant(X_train)

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
VIF_values(X_train)
#dropping the 'compactness_mean' from the model

X_train = X_train.drop(columns='compactness_mean',axis=1)
X_train_sm = sm.add_constant(X_train)

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
# feature variables and their corresponding VIFs

VIF_values(X_train)
#dropping the 'symmetry_mean' from the model

X_train = X_train.drop(columns='symmetry_mean',axis=1)
X_train_sm = sm.add_constant(X_train)

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

VIF_values(X_train)
#dropping the 'compactness_se' from the model

X_train = X_train.drop(columns='compactness_se',axis=1)
X_train_sm = sm.add_constant(X_train)

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
#dropping the 'symmetry_se' from the model

X_train = X_train.drop(columns='symmetry_se',axis=1)
X_train_sm = sm.add_constant(X_train)

logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm6.fit()

res.summary()
#dropping the 'smoothness_se' from the model

X_train = X_train.drop(columns='smoothness_se',axis=1)
X_train_sm = sm.add_constant(X_train)

logm7 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm7.fit()

res.summary()
#dropping the 'texture_se' from the model

X_train = X_train.drop(columns='texture_se',axis=1)
X_train_sm = sm.add_constant(X_train)

logm8 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm8.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

VIF_values(X_train)
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred
y_train_pred_final = pd.DataFrame({'Diagnosis':y_train.values, 'Diagnosis_Prob':y_train_pred})
#predicted values above 0.5 is considered to be Malignant i.e 1

y_train_pred_final.Diagnosis_Prob = y_train_pred_final.Diagnosis_Prob.map(lambda x: 1 if x>0.5 else 0)
y_train_pred_final.head()
#build confusion matrix using confusion_matrix from sklearn.metrics

confusion = confusion_matrix(y_train_pred_final.Diagnosis, y_train_pred_final.Diagnosis_Prob)
plt.figure(figsize=(10,5))

categories = ['Beingn', 'Malignant']

sns.heatmap(confusion,annot=True,fmt='d', cmap='Blues',linewidths=1,xticklabels=categories,yticklabels=categories,cbar=False)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
print('Accuracy Score : ',accuracy_score(y_train_pred_final.Diagnosis, y_train_pred_final.Diagnosis_Prob))

print('f1 Score : ',f1_score(y_train_pred_final.Diagnosis, y_train_pred_final.Diagnosis_Prob))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives



print('Sensitivity : ',TP / float(TP+FN))

print('Specificity : ',TN / float(TN+FP))
# Calculate false postive rate - predicting Malignant when patient does have beingn

print(FP/ float(TN+FP))
# positive predictive value  and  Negative predictive value



print('positive predictive value: ',TP / float(TP+FP))

print('Negative predictive value: ',TN / float(TN+ FN))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = roc_auc_score( actual, probs )

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
draw_roc(y_train_pred_final.Diagnosis, y_train_pred_final.Diagnosis_Prob)
#filtering out the columns based on our final model 



X_test = X_test[X_train.columns]
#add constant to the X_test data



X_test_sm = sm.add_constant(X_test)
#predict the y_test values



y_test_pred = res.predict(X_test_sm)
# forming new dataframe holding y_test and y_test_pred values



y_pred_final = pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_pred)],axis=1)
y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'diagnosis_Prob'})
y_pred_final['diagnosis_Prob'] = y_pred_final['diagnosis_Prob'].map(lambda x: 1 if x>0.5 else 0)
print('Accuracy Score : ',accuracy_score(y_pred_final.diagnosis, y_pred_final.diagnosis_Prob))

print('f1 Score : ',f1_score(y_pred_final.diagnosis, y_pred_final.diagnosis_Prob))
#build confusion matrix using confusion_matrix from sklearn.metrics

confusion = confusion_matrix(y_pred_final.diagnosis, y_pred_final.diagnosis_Prob)
plt.figure(figsize=(10,5))

categories = ['Beingn', 'Malignant']

sns.heatmap(confusion,annot=True,fmt='d', cmap='Blues',linewidths=1,xticklabels=categories,yticklabels=categories,cbar=False)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
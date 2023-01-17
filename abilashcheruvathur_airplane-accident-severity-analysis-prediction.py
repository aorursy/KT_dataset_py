import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import StandardScaler 

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import multilabel_confusion_matrix

import scikitplot as skplt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/airplane-accidents-severity-dataset/train.csv')
data.head()
data.dtypes
data.info()
data.shape
na_values=data.isna().sum()

print(na_values)
data.drop(['Accident_ID'],axis=1,inplace=True)

data.describe().T
# Distribution of continous data



# Safety_Score, Control_Metric,Turbulence_In_gforce





plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('Safety_Score')

sns.distplot(data['Safety_Score'],color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Control_Metric')

sns.distplot(data['Control_Metric'],color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Turbulence_In_gforces')

sns.distplot(data['Turbulence_In_gforces'],color='green')







plt.figure(figsize=(30,6))



#Subplot 1- Boxplot

plt.subplot(1,3,1)

plt.title('Safety_Score')

sns.boxplot(data['Safety_Score'],orient='horizondal',color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Control_Metric')

sns.boxplot(data['Control_Metric'],orient='horizondal',color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Turbulence_In_gforces')

sns.boxplot(data['Turbulence_In_gforces'],orient='horizondal',color='green')

# Distribution of continous data

# Cabin_Temprature, Max_Elevation, Adverse_Weather_Metric



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('Cabin_Temperature')

sns.distplot(data['Cabin_Temperature'],color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Max_Elevation')

sns.distplot(data['Max_Elevation'],color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Adverse_Weather_Metric')

sns.distplot(data['Adverse_Weather_Metric'],color='green')







plt.figure(figsize=(30,6))



#Subplot 1- Boxplot

plt.subplot(1,3,1)

plt.title('Cabin_Temperature')

sns.boxplot(data['Cabin_Temperature'],orient='horizondal',color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Max_Elevation')

sns.boxplot(data['Max_Elevation'],orient='horizondal',color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Adverse_Weather_Metric')

sns.boxplot(data['Adverse_Weather_Metric'],orient='horizondal',color='green')

##### Days_Since_Inspection, Total_Safety_Compliant

plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,2,1)

plt.title('Days_Since_Inspection')

sns.countplot(data['Days_Since_Inspection'],color='red')



#Subplot 2

plt.subplot(1,2,2)

plt.title('Total_Safety_Complaints')

sns.countplot(data['Total_Safety_Complaints'],color='blue')

# Accident_Type_Code and Violations



plt.figure(figsize=(30,6))





#Subplot 1

plt.subplot(1,2,1)

plt.title('Accident_Type_Code')

sns.countplot(data['Accident_Type_Code'],color='red')



#Subplot 2

plt.subplot(1,2,2)

plt.title('Violations')

sns.countplot(data['Violations'],color='blue')
plt.figure(figsize=(30,6))



plt.title('Severity')

sns.countplot(data['Severity'],color='red')
sns.pairplot(data,palette="Set2", diag_kind="kde", height=2.5)
correlation=data.corr()

correlation.style.background_gradient(cmap='coolwarm')
data.corr()>0.5
data.corr()<-0.5
df=data.drop(['Severity'],axis=1)

def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=20):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations")

print(get_top_abs_correlations(df, 3))
data.info()
dataNumericals = pd.DataFrame(data, columns =data.columns[data.dtypes == 'float64']) 

dataNumericals.head()
dataNumericals=dataNumericals.apply(zscore)
dataNumericals.head()
floats = dataNumericals.columns[dataNumericals.dtypes == 'float64']

for columns in floats:

    indexNames_larger = dataNumericals[dataNumericals[columns]>3].index

    indexNames_lesser = dataNumericals[dataNumericals[columns]<-3].index

    # Delete these row indexes from dataFrame

    dataNumericals.drop(indexNames_larger , inplace=True)

    dataNumericals.drop(indexNames_lesser , inplace=True)

    data.drop(indexNames_larger , inplace=True)

    data.drop(indexNames_lesser , inplace=True)
dataNumericals.info()
data.info()
data.drop(data.columns[data.dtypes == 'float64'],axis=1,inplace=True)
data.head()
for column in dataNumericals.columns:

    data[column]=dataNumericals[column]
data.head()
data.info()
data['Severity'].unique()
encoder=LabelEncoder()

data['Severity']=encoder.fit_transform(data['Severity'])
data.head()
data['Total_Safety_Complaints'] = np.power(2, data['Total_Safety_Complaints'])

data['Days_Since_Inspection'] = np.power(2, data['Days_Since_Inspection'])

data['Safety_Score'] = np.power(2, data['Safety_Score'])
X=data.drop(['Severity'],axis=1)
Y=data['Severity']
Xtrain_val,X_test,ytrain_val,Y_test=train_test_split(X,Y,test_size=0.2,random_state=22)
kf = KFold(n_splits=10,random_state=2,shuffle=True)

kf.get_n_splits(Xtrain_val)

print(kf)





for train_index, val_index in kf.split(Xtrain_val):

    print("TRAIN:", train_index, "VALIDATION:", val_index)

    X_train, X_val = Xtrain_val.iloc[train_index], Xtrain_val.iloc[val_index]

    y_train, y_val = ytrain_val.iloc[train_index], ytrain_val.iloc[val_index]
#Pipeline

pipe_GBR = Pipeline([('GBR', GradientBoostingClassifier())]) 



#Parameter-grid

param_grid = {'GBR__n_estimators': [50,100,150],'GBR__learning_rate':[0.1,0.2,0.5]} 

 

#Using RandomSearchCV

Random_GBR = RandomizedSearchCV( pipe_GBR , param_distributions=param_grid, cv= 5, n_iter=3) 



#Fitting the data in the model

Random_GBR.fit(X_train, y_train) 



print(" Best cross-validation score obtained is: {:.2f}". format( Random_GBR.best_score_)) 

print(" Best parameters as part of Gridsearch is: ", Random_GBR.best_params_) 

print(" Train set score obtained is: {:.2f}". format( Random_GBR.score( X_train, y_train)))

print(" Validation set score obtained is: {:.2f}". format( Random_GBR.score( X_val, y_val)))

print(" Test set score obtained is: {:.2f}". format( Random_GBR.score( X_test, Y_test)))
y_pred=Random_GBR.predict(X_test)
accuracy_score=metrics.accuracy_score(Y_test,y_pred)

percision_score=metrics.precision_score(Y_test,y_pred,average='macro')

recall_score=metrics.recall_score(Y_test,y_pred,average='macro')

f1_score=metrics.f1_score(Y_test,y_pred,average='macro')

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percision of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))

print("The f1 score of this model is {0:.2f}%".format(f1_score*100))
Random_GBR.cv_results_
classification_report=metrics.classification_report(Y_test,y_pred)
print(classification_report)
#Pipeline

pipe_XGB = Pipeline([('XGB', XGBClassifier())]) 



#Parameter-grid

param_grid = {'XGB__learning_rate':[0.1,0.2,0.3],'XGB__max_depth' :[10,50,100], 'XGB__gamma':[0.1,0.3,0.5]} 

 

#Using RandomSearchCV

Random_XGB = RandomizedSearchCV( pipe_XGB , param_distributions=param_grid, cv= 5, n_iter=3) 

#Fitting the data in the model

Random_XGB.fit(X_train, y_train)



print(" Best cross-validation score obtained is: {:.2f}". format( Random_XGB.best_score_)) 

print(" Best parameters as part of Gridsearch is: ", Random_XGB.best_params_) 

print(" Train set score obtained is: {:.2f}". format( Random_XGB.score( X_train, y_train)))

print(" Validation set score obtained is: {:.2f}". format( Random_XGB.score( X_val, y_val)))

print(" Test set score obtained is: {:.2f}". format( Random_XGB.score( X_test, Y_test)))
y_pred=Random_XGB.predict(X_test)
accuracy_score=metrics.accuracy_score(Y_test,y_pred)

percision_score=metrics.precision_score(Y_test,y_pred,average='macro')

recall_score=metrics.recall_score(Y_test,y_pred,average='macro')

f1_score=metrics.f1_score(Y_test,y_pred,average='macro')

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percision of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))

print("The f1 score of this model is {0:.2f}%".format(f1_score*100))
Random_XGB.cv_results_
classification_report=metrics.classification_report(Y_test,y_pred)
print(classification_report)
skplt.metrics.plot_confusion_matrix(Y_test,y_pred,figsize=(12,12))
testData=pd.read_csv("/kaggle/input/airplane-accidents-severity-dataset/test.csv")
testData.drop(['Accident_ID'],axis=1,inplace=True)

testData.head()
testData.info()
testDataNumericals = pd.DataFrame(testData, columns =testData.columns[testData.dtypes == 'float64']) 

testDataNumericals.head()
testDataNumericals=testDataNumericals.apply(zscore)
testData.drop(testData.columns[testData.dtypes == 'float64'],axis=1,inplace=True)

testData.head()
for column in testDataNumericals.columns:

    testData[column]=testDataNumericals[column]
testData.head()
testData['Total_Safety_Complaints'] = np.power(2, testData['Total_Safety_Complaints'])

testData['Days_Since_Inspection'] = np.power(2, testData['Days_Since_Inspection'])

testData['Safety_Score'] = np.power(2, testData['Safety_Score'])
testPredictions=Random_XGB.predict(testData)
testData['Severity']=encoder.inverse_transform(testPredictions)
testData.head()
finalData=pd.read_csv("/kaggle/input/airplane-accidents-severity-dataset/test.csv")
finalData['Severity']=testData['Severity']
finalData.head()
finalData.to_csv('test.csv')
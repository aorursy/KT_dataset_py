import pandas as pd
import numpy as np
wineRawData = pd.read_csv('../input/winequality-red.csv')
display(wineRawData.head())
display(wineRawData.describe())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
plt.figure(figsize=(20,20))

for i in range(0,len(wineRawData.columns)):
    plt.subplot(5,3,i+1)
    parameterName = wineRawData.columns[i]
    sns.distplot(wineRawData[parameterName]);
plt.show()

wineGB_ori = wineRawData['quality'].apply(lambda x: 1 if x>=7 else 0)
print(wineGB_ori.value_counts())
plt.figure(figsize=(20,20))

for i in range(0,len(wineRawData.columns)):
    plt.subplot(5,3,i+1)
    parameterName = wineRawData.columns[i]
    sns.boxplot(wineRawData[parameterName]);
plt.show()
outliersListArray = []

for feature in wineRawData.keys():
    
    Q1 = np.percentile(wineRawData[feature],25)
    Q3 = np.percentile(wineRawData[feature],75)

    step = (Q3-Q1)*1.5
    
    # Display the outliers
#     print("Data points considered outliers for the feature '{}':".format(feature))
    tmpOutlierList = wineRawData[~((wineRawData[feature] >= Q1 - step) & (wineRawData[feature] <= Q3 + step))]
    outliersListArray = outliersListArray + tmpOutlierList.index.tolist()

outlierDf =  pd.DataFrame(outliersListArray)
outlierDf.columns = ['feq']
outlierCountingDf = pd.value_counts(outlierDf['feq'])
outlierRemoveIndexList = outlierCountingDf[outlierCountingDf>1].index.tolist()
# print(outlierCountingDf[outlierCountingDf>1])
outliers  = outlierRemoveIndexList


# # Remove the outliers, if any were specified
wineRawData_rmOutlier = wineRawData.drop(wineRawData.index[outliers]).reset_index(drop = True)

plt.figure(figsize=(20,20))

for i in range(0,len(wineRawData_rmOutlier.columns)):
    plt.subplot(5,3,i+1)
    parameterName = wineRawData_rmOutlier.columns[i]
    sns.distplot(wineRawData_rmOutlier[parameterName]);
plt.show()
sns.countplot(x='quality', data=wineRawData_rmOutlier)
wineGB = wineRawData_rmOutlier['quality'].apply(lambda x: 1 if x>=7 else 0)
print(wineGB.value_counts())
TP = float(np.sum(wineGB))
FP = float(wineGB.count() - TP)

accuracy = TP / (TP+ FP)
recall = TP / TP
precision = TP / (TP + FP)

f1score = 2 * (precision * recall) / (precision + recall)

# Print the results 
print ("Naive Predictor: [Accuracy score: {:.4f}, F1-score: {:.4f}]".format(accuracy, f1score))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report

XData = wineRawData_rmOutlier.drop(columns=['quality'])
X_train, X_test, y_train, y_test = train_test_split(XData, wineGB, test_size = 0.2)
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train,y_train)
y_test_pred = rfc.predict(X_test)
print("[Training] Random Forest Classifier Accuracy is: " + str(rfc.score(X_train,y_train)))
print("[Testing] Random Forest Classifier Accuracy is: " + str(rfc.score(X_test,y_test)))
print("[Testing] Random Forest Classifier F1-Score is: " + str(f1_score(y_test, y_test_pred)))

print("------------------")

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_test_pred = lr.predict(X_test)
print("[Training] Logistic Regression Accuracy is: " + str(lr.score(X_train,y_train)))
print("[Testing] Logistic Regression Accuracy is: " + str(lr.score(X_test,y_test)))
print("[Testing] Logistic Regression F1-Score is: " + str(f1_score(y_test, y_test_pred)))
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':list(range(10,250,10)), 'max_depth':list(range(5,15,2))}
rfc = RandomForestClassifier(n_jobs=-1)
rfc = GridSearchCV(rfc, parameters, cv=10)
rfc.fit(X_train,y_train)
rfc_cvResult = pd.DataFrame(rfc.cv_results_)
rfc_mdp05 = rfc_cvResult[rfc_cvResult['param_max_depth']==5].reset_index()
rfc_mdp07 = rfc_cvResult[rfc_cvResult['param_max_depth']==7].reset_index()
rfc_mdp09 = rfc_cvResult[rfc_cvResult['param_max_depth']==9].reset_index()
rfc_mdp11 = rfc_cvResult[rfc_cvResult['param_max_depth']==11].reset_index()
rfc_mdp13 = rfc_cvResult[rfc_cvResult['param_max_depth']==13].reset_index()

# plt.plot(x=rfc_mdp05.index.values,y=rfc_mdp05['mean_train_score'],)
# plt.ylim((0.5,1.0))

plt.figure(figsize=(10,5))
plt.title('mean_test_score')
plt.plot(rfc_mdp05['param_n_estimators'], rfc_mdp05['mean_test_score'], '.r-')
plt.ylim((0.84,0.93))

plt.plot(rfc_mdp07['param_n_estimators'], rfc_mdp07['mean_test_score'], '.b-')
plt.plot(rfc_mdp09['param_n_estimators'], rfc_mdp09['mean_test_score'], '.g-')
plt.plot(rfc_mdp11['param_n_estimators'], rfc_mdp11['mean_test_score'], '.y-')
plt.plot(rfc_mdp13['param_n_estimators'], rfc_mdp13['mean_test_score'], '.m-')
plt.show()
# plt.plot(rfc_mdp05.index.values, rfc_mdp05['std_train_score'], '.b-')
# std_test_score	std_train_score
# mean_test_score	mean_train_score
plt.figure(figsize=(10,5))
plt.title('std_test_score')
plt.plot(rfc_mdp05['param_n_estimators'], rfc_mdp05['std_test_score'], '.r-')
plt.ylim((0,0.06))

plt.plot(rfc_mdp07['param_n_estimators'], rfc_mdp07['std_test_score'], '.b-')
plt.plot(rfc_mdp09['param_n_estimators'], rfc_mdp09['std_test_score'], '.g-')
plt.plot(rfc_mdp11['param_n_estimators'], rfc_mdp11['std_test_score'], '.y-')
plt.plot(rfc_mdp13['param_n_estimators'], rfc_mdp13['std_test_score'], '.m-')
plt.show()
rfc_best = rfc.best_estimator_
print(rfc_best)
rfc_best.score(X_test,y_test)
y_test_pred = rfc_best.predict(X_test)

print("[Testing] Random Forest Accuracy is: " + str(rfc_best.score(X_test,y_test)))
print("[Testing] Random Forest Classifier F1-Score is: " + str(f1_score(y_test, y_test_pred)))

import numpy as np
parameters_lr = {'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

lr = LogisticRegression()
lr = GridSearchCV(lr, parameters_lr, cv=10)
lr.fit(X_train,y_train)
lr_best = lr.best_estimator_
print(lr_best)
lr_best.score(X_test,y_test)
y_test_pred = lr_best.predict(X_test)

print("[Testing] Logistic Regression Accuracy is: " + str(lr_best.score(X_test,y_test)))
print("[Testing] Logistic Regression F1-Score is: " + str(f1_score(y_test, y_test_pred)))
print('Optimized Result of Random Forest Classification\n')
print(classification_report(y_test, rfc_best.predict(X_test)))
print('Optimized Result of Logistic Regression\n')
print(classification_report(y_test, lr_best.predict(X_test)))
importanceTable = pd.DataFrame(rfc_best.feature_importances_)
featureNameList = pd.DataFrame(XData.columns)
importanceTable.columns = ['Importance']
featureNameList.columns = ['WineFeature']
featureImportance = featureNameList.join(importanceTable)
featureImportance = featureImportance.sort_values(['Importance'],ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=featureImportance['Importance'],y=featureImportance['WineFeature'])
plt.show()
plt.scatter(x=wineRawData_rmOutlier['alcohol'],y=wineRawData_rmOutlier['quality'])
plt.ylabel('quality')
plt.xlabel('alcohol %')

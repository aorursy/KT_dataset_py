import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Load data

rain_data = pd.read_csv('../input/weatherAUS.csv')

rain_data.head()
# Visualising missing data:

sns.heatmap(rain_data.isnull(),yticklabels=False,cbar=False,cmap='Reds_r')
# High percentage of missing data for Evaporation, Sunshine, Cloud9am and Cloud3pm features.

# Date, Location and RISK_MM will be removed.

# Lastly, remove any observations/rows with missing data

rain_data.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm','RISK_MM','Date','Location'],axis=1,inplace=True)

rain_data.dropna(inplace=True)

rain_data[['RainTomorrow','RainToday']] = rain_data[['RainTomorrow','RainToday']].replace({'No':0,'Yes':1})
# Frequency of Rainy and No Rain:

mpl.style.use('ggplot')

plt.figure(figsize=(6,4))

plt.hist(rain_data['RainTomorrow'],bins=2,rwidth=0.8)

plt.xticks([0.25,0.75],['No Rain','Rain'])

plt.title('Frequency of No Rain and Rainy days\n')

print(rain_data['RainTomorrow'].value_counts())
# Segregating our numerical features from the categorical

rain_data_num = rain_data[['MinTemp','MaxTemp','Rainfall','WindSpeed9am','WindSpeed3pm',

                           'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',

                           'Temp9am','Temp3pm','RainToday','RainTomorrow']]



# Histogram of each numerical feature

mpl.rcParams['patch.force_edgecolor'] = True

ax_list = rain_data_num.drop(['RainTomorrow'],axis=1).hist(figsize=(20,15),bins=20)

ax_list[2,1].set_xlim((0,100))
plt.figure(figsize=(12,8))

sns.heatmap(rain_data_num.corr(),annot=True,cmap='bone',linewidths=0.25)
# Creating dummy variables for the categorical features:

WindGustDir_data = pd.get_dummies(rain_data['WindGustDir'])

WindDir9am_data = pd.get_dummies(rain_data['WindDir9am'])

WindDir3pm_data = pd.get_dummies(rain_data['WindDir3pm'])



# Dataframe of the categorical features

rain_data_cat = pd.concat([WindGustDir_data,WindDir9am_data,WindDir3pm_data],

                          axis=1,keys=['WindGustDir','WindDir9am','WindDir3pm'])



# Combining the Numerical and Categorical/Dummy Variables

rain_data = pd.concat([rain_data_num,rain_data_cat],axis=1)
from sklearn.model_selection import train_test_split



X = rain_data.drop(['RainTomorrow'],axis=1)

y = rain_data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=88)
from sklearn.ensemble import RandomForestClassifier

# Out of Bag (oob) set to True. We will compare the oob_score with accuracy to see if they differ by much

# n_estimators, or number of decision trees set to 100

rf = RandomForestClassifier(n_estimators=100,oob_score=True,random_state=88)

rf.fit(X_train,y_train)

y_rf_pred = rf.predict(X_test)
# No Rain and Rain frequency in test set

print(y_test.value_counts())

null_accuracy = float(y_test.value_counts().head(1) / len(y_test))

print('Null Accuracy Score: {:.2%}'.format(null_accuracy))
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

print('Accuracy Score: {:.2%}'.format(accuracy_score(y_test,y_rf_pred),'\n'))

print('Out of Bag Accuracy Score: {:.2%}'.format(rf.oob_score_),'\n')

print('Confusion Matrix:\n',confusion_matrix(y_test,y_rf_pred))
# Using feature_importance_ for feature selection

feature_importance_rf = pd.DataFrame(rf.feature_importances_,index=X_train.columns,columns=['Importance']).sort_values(['Importance'],ascending=False)

feature_importance_rf.head(5)
# Plot feature_importance

feature_importance_rf.plot(kind='bar',legend=False,figsize=(15,8))
# Our Top 5 Features

features_top_5 = list(feature_importance_rf.index[0:6])



# X dataframe - with only the top 5 features

subset_1 = [X.columns.get_loc(x) for x in features_top_5]



# Split, Train, Predict

X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,subset_1], y, test_size=0.30, random_state=88)

rf.fit(X_train,y_train)

y_rf_pred = rf.predict(X_test)



print('Accuracy Score: {:.2%}'.format(accuracy_score(y_test,y_rf_pred)))

print('Out of Bag Score {:.2%}:'.format(rf.oob_score_),'\n')

print('Confusion Matrix:\n',confusion_matrix(y_test,y_rf_pred))
# X dataframe - with top 5 features and the categorical variables

subset_2 = subset_1 + list(range(12,len(X.columns)))



# Split, Train, Predict

X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,subset_2], y, test_size=0.30, random_state=88)

rf.fit(X_train,y_train)

y_rf_pred = rf.predict(X_test)



print('Accuracy Score: {:.2%}'.format(accuracy_score(y_test,y_rf_pred)))

print('Out of Bag Score {:.2%}:'.format(rf.oob_score_),'\n')

print('Confusion Matrix:\n',confusion_matrix(y_test,y_rf_pred))
%%time 



# Up to what number of features to plot

index = np.array(list(range(2,9)) + [15, 30, 60])



# creating list of index location

features = list(feature_importance_rf.index)

features = [X.columns.get_loc(x) for x in features]



# instantiate classifier

rf = RandomForestClassifier(n_estimators=100,random_state=88)



accuracy_rate = []



# append the accuracy rate

for i in index:

    X_train, X_test, y_train, y_test = train_test_split(X.iloc[:,features[0:i]], y, test_size=0.30, random_state=88)

    rf.fit(X_train,y_train)

    y_rf_pred = rf.predict(X_test)    

    accuracy_rate.append(accuracy_score(y_test,y_rf_pred))
# Plot accuracy vs. number of features

plt.figure(figsize=(7,5))

plt.scatter(x=index-1,y=accuracy_rate)

plt.ylabel('Accuracy Rate',fontsize=12)

plt.xlabel('Number of Features',fontsize=12)

plt.xlim(-0.2,60)

plt.title('Random Forest \nAccuracy Rate vs. Number of Features', fontsize = 14)
# Split, Train, Predict on The 7 Features

X_train, X_test, y_train, y_test = train_test_split(X[feature_importance_rf.head(7).index], y, test_size=0.30, random_state=88)

rf.fit(X_train,y_train)

y_rf_pred = rf.predict(X_test)

cm = pd.DataFrame(confusion_matrix(y_test,y_rf_pred), index=['NO RAIN','RAIN'],columns=['NO RAIN','RAIN'])
print('Accuracy Score (Top 7 Features): {:.2%}'.format(accuracy_score(y_test,y_rf_pred)),'\n')



# plot confusion matrix

fig = plt.figure(figsize=(8,6))

ax = sns.heatmap(cm,annot=True,cbar=False, cmap='CMRmap_r',linewidths=0.5,fmt='.0f')

ax.set_title('Random Forest Confusion Matrix',fontsize=16,y=1.25)

ax.set_ylabel('ACTUAL',fontsize=14)

ax.set_xlabel('PREDICTED',fontsize=14)

ax.xaxis.set_ticks_position('top')

ax.xaxis.set_label_position('top')

ax.tick_params(labelsize=12)
TP = cm.iloc[1,1] # True Positive - Predicted Rain Correctly

TN = cm.iloc[0,0] # True Negative - Predicted No Rain Incorrectly

FP = cm.iloc[0,1] # False Positive - Predicted Rain when it didn't rain

FN = cm.iloc[1,0] # False Negative - Predicted No Rain when it did rain
print('Sensitivity: {:.2%}'.format(TP/(FN+TP)))

print('Specificity: {:.2%}'.format(TN/(FP+TN)))
# proves np.array of the probability scores

y_prob_rain = rf.predict_proba(X_test)



# To convert x-axis to a percentage

from matplotlib.ticker import PercentFormatter



# Plot histogram of predicted probabilities

fig,ax = plt.subplots(figsize=(10,6))

plt.hist(y_prob_rain[:,1],bins=50,alpha=0.5,color='teal',label='Rain')

plt.hist(y_prob_rain[:,0],bins=50,alpha=0.5,color='orange',label='No Rain')

plt.xlim(0,1)

plt.title('Histogram of Predicted Probabilities')

plt.xlabel('Predicted Probability (%)')

plt.ylabel('Frequency')



ax.xaxis.set_major_formatter(PercentFormatter(1))

ax.text(0.025,0.83,'n = 33,878',transform=ax.transAxes)



plt.legend()
#ROC Curve

fpr, tpr, thresholds = roc_curve(y_test,y_prob_rain[:,1])



fig,ax1 = plt.subplots(figsize=(9,6))

ax1.plot(fpr, tpr,color='orange')

ax1.legend(['ROC Curve'],loc=1)

ax1.set_xlim([-0.005, 1.0])

ax1.set_ylim([0,1])

ax1.set_ylabel('True Positive Rate (Sensitivity)')

ax1.set_xlabel('False Positive Rate \n(1 - Specificity)\n FP / (TN + FP)')

ax1.set_title('ROC Curve for RainTomorrow Random Forest Classifier\n')



plt.plot([0,1],[0,1],linestyle='--',color='teal')

plt.plot([0,1],[0.5,0.5],linestyle='--',color='red',linewidth=0.25)



#Threshold Curve

ax2 = plt.gca().twinx()

ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='black')

ax2.legend(['Threshold'],loc=4)

ax2.set_ylabel('Threshold',color='black')

ax2.set_ylim([0,1])

ax2.grid(False)
# Function to calc sensitivity and specificity rate for a given threshold

def evaluate_threshold(threshold):

    print('Sensitivity: {:.2%}'.format(tpr[thresholds > threshold][-1]))

    print('Specificity: {:.2%}'.format(1 - fpr[thresholds > threshold][-1]))

    

evaluate_threshold(0.25)
from sklearn.preprocessing import binarize

# change the predicted class with 25% threshold

y_pred_class = binarize(y_prob_rain,0.25)[:,1]



cm = pd.DataFrame(confusion_matrix(y_test,y_pred_class), index=['NO RAIN','RAIN'],columns=['NO RAIN','RAIN'])



print('Accuracy Score (Top 7 Features with 25% Threshold): {:.2%}'.format(accuracy_score(y_test,y_pred_class)),'\n')



# Plot Confusion Matrix

fig = plt.figure(figsize=(8,6))

ax = sns.heatmap(cm,annot=True,cbar=False, cmap='CMRmap_r',linewidths=0.5,fmt='.0f')

ax.set_title('Random Forest Confusion Matrix',fontsize=16,y=1.25)

ax.set_ylabel('ACTUAL',fontsize=14)

ax.set_xlabel('PREDICTED',fontsize=14)

ax.xaxis.set_ticks_position('top')

ax.xaxis.set_label_position('top')

ax.tick_params(labelsize=12)
TP = cm.iloc[1,1] # True Positive - Predicted Rain Correctly

TN = cm.iloc[0,0] # True Negative - Predicted No Rain Incorrectly

FP = cm.iloc[0,1] # False Positive - Predicted Rain when it didn't rain

FN = cm.iloc[1,0] # False Negative - Predicted No Rain when it did rain



sens_rf = TP/(FN+TP)

spec_rf = TN/(FP+TN)



print('Sensitivity: {:.2%}'.format(sens_rf))

print('Specificity: {:.2%}'.format(spec_rf))
rf_auc = roc_auc_score(y_test,y_prob_rain[:,1])

print('AUC Score: {:.2%}'.format(rf_auc))
# Libraries for the Logistic Regression 

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



X = rain_data.drop(['RainTomorrow'],axis=1)

y = rain_data['RainTomorrow']

# Remove the Categorical (Dummy) Variables, as we have identified earlier that they do not add much value

X = X.iloc[:,0:12] 





# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=88)



# Logistic Regression train

lr = LogisticRegression(random_state=88, solver='liblinear')

lr.fit(X_train,y_train)



# predict

y_lr_pred = lr.predict(X_test)
# The 10-Fold Cross Validation method is used to calculate the accuracy score of the Logistic Regression model.

print('Accuracy Score with 10-KFolds: {:.2%}'.format(cross_val_score(lr,X,y,cv=10,scoring='accuracy').mean()),'\n')

print('Confusion Matrix:\n',confusion_matrix(y_test,y_lr_pred))
%%time

#Feature Selection Method: Recursive Feature Elimination 

rfe = RFE(estimator=lr, n_features_to_select=7)

rfe = rfe.fit(X_train,y_train)



print("Number of Features: {}".format(rfe.n_features_)) 

print("Selected Features: {}".format(rfe.support_))

print("Feature Ranking: {}".format(rfe.ranking_))
pd.DataFrame(X.iloc[:,rfe.support_].columns,columns=['Importance'])
X_rfe = X.iloc[:,rfe.support_]

# Train Test split with subset of X features

X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.30, random_state=88)

# Train and Predict

lr.fit(X_train,y_train)

y_lr_pred = lr.predict(X_test)
#accuracy rate using 10-Fold CV

accuracy_kfold = cross_val_score(lr,X_rfe,y,cv=10,scoring='accuracy').mean()

print('Accuracy Score with 7 Features and 10-KFolds: {:.2%}'.format(accuracy_kfold),'\n')

print('Confusion Matrix:\n',confusion_matrix(y_test,y_lr_pred))
pd.concat([pd.DataFrame(lr.coef_,index=['coefficient'],columns=X_train.columns).T, 

                         X_train.aggregate([np.mean,np.std,np.min,np.max]).T],axis=1)
cm = pd.DataFrame(confusion_matrix(y_test,y_lr_pred), index=['NO RAIN','RAIN'],columns=['NO RAIN','RAIN'])



print('Accuracy Score with 7 Features and 10-KFolds: {:.2%}'.format(accuracy_kfold),'\n')



# Plot CM

fig = plt.figure(figsize=(8,6))

ax = sns.heatmap(cm,annot=True,cbar=False, cmap='CMRmap_r',linewidths=0.5,fmt='.0f')

ax.set_title('Logistic Regression Confusion Matrix',fontsize=16,y=1.25)

ax.set_ylabel('ACTUAL',fontsize=14)

ax.set_xlabel('PREDICTED',fontsize=14)

ax.xaxis.set_ticks_position('top')

ax.xaxis.set_label_position('top')

ax.tick_params(labelsize=12)
TP = cm.iloc[1,1] # True Positive - Predicted Rain Correctly

TN = cm.iloc[0,0] # True Negative - Predicted No Rain Incorrectly

FP = cm.iloc[0,1] # False Positive - Predicted Rain when it didn't rain

FN = cm.iloc[1,0] # False Negative - Predicted No Rain when it did rain



print('Sensitivity: {:.2%}'.format(TP/(FN+TP)))

print('Specificity: {:.2%}'.format(TN/(FP+TN)))
# Probability of Rain for X_test

y_prob_rain = lr.predict_proba(X_test)



fpr, tpr, thresholds = roc_curve(y_test,y_prob_rain[:,1])



#ROC Curve

fig,ax1 = plt.subplots(figsize=(9,6))

ax1.plot(fpr, tpr,color='orange')

ax1.legend(['ROC Curve'],loc=1)

ax1.set_xlim([-0.005, 1.0])

ax1.set_ylim([0,1])

ax1.set_ylabel('True Positive Rate (Sensitivity)')

ax1.set_xlabel('False Positive Rate \n(1 - Specificity)\n FP / (TN + FP)')

ax1.set_title('ROC Curve for RainTomorrow Logistic Regression Classifier\n')



plt.plot([0,1],[0,1],linestyle='--',color='teal')

plt.plot([0,1],[0.5,0.5],linestyle='--',color='red',linewidth=0.25)



#Threshold Curve

ax2 = plt.gca().twinx()

ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='black')

ax2.legend(['Threshold'],loc=4)

ax2.set_ylabel('Threshold',color='black')

ax2.set_ylim([0,1])

ax2.grid(False)
# Changing predictions using threshold of 25%

y_pred_class = binarize(y_prob_rain,0.25)[:,1]



cm = pd.DataFrame(confusion_matrix(y_test,y_pred_class), index=['NO RAIN','RAIN'],columns=['NO RAIN','RAIN'])



print('Accuracy Score (Top 7 Features with 25% Threshold): {:.2%}'.format(accuracy_score(y_test,y_pred_class)),'\n')



fig = plt.figure(figsize=(8,6))

ax = sns.heatmap(cm,annot=True,cbar=False, cmap='CMRmap_r',linewidths=0.5,fmt='.0f')

ax.set_title('Logistic Regression Confusion Matrix',fontsize=16,y=1.25)

ax.set_ylabel('ACTUAL',fontsize=14)

ax.set_xlabel('PREDICTED',fontsize=14)

ax.xaxis.set_ticks_position('top')

ax.xaxis.set_label_position('top')

ax.tick_params(labelsize=12)
TP = cm.iloc[1,1] # True Positive - Predicted Rain Correctly

TN = cm.iloc[0,0] # True Negative - Predicted No Rain Incorrectly

FP = cm.iloc[0,1] # False Positive - Predicted Rain when it didn't rain

FN = cm.iloc[1,0] # False Negative - Predicted No Rain when it did rain



sens_lr = TP/(FN+TP)

spec_lr = TN/(FP+TN)



print('Sensitivity: {:.2%}'.format(sens_lr))

print('Specificity: {:.2%}'.format(spec_lr))
lr_auc = cross_val_score(lr,X,y,cv=10,scoring='roc_auc').mean()



print('Null Accuracy Score: {:.2%}\n'.format(null_accuracy))

print('{:>30} {:>26}'.format('Random Forest','Logistic Regression'))

print('{} {:>17.2%} {:>22.2%}'.format('AUC Score',rf_auc,lr_auc))

print('{} {:>14.2%} {:>22.2%}'.format('Sensitivity*',sens_rf,sens_lr))

print('{} {:>14.2%} {:>22.2%}'.format('Specificity*',spec_rf,spec_lr))

print('\n*25% Threshold')
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load into dataframe and look at the first few rows 

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.head()
# print dataframe information, particularly data type and count, if there are any missing entries which turns out to be none

data.info()
# Check the target incidence

print(data.Class.value_counts(normalize=True).round(3))



# Plot it out 

data.Class.value_counts().sort_index().plot(kind='bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency");
# Segregate features and labels into separate variables, drop Loan ID (index 0) 

X = data.iloc[:, 0:30].values

y = data.iloc[:, 30].values
# split data into train and test sets with sklearn

from sklearn.model_selection import train_test_split



# split into train/test sets with same class ratio

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)



# summarize dataset

print('Dataset: Class0=%d, Class1=%d' % (len(y[y==0]), len(y[y==1])))

print('Train: Class0=%d, Class1=%d' % (len(trainy[trainy==0]), len(trainy[trainy==1])))

print('Test: Class0=%d, Class1=%d' % (len(testy[testy==0]), len(testy[testy==1])))
# data normalization with sklearn

from sklearn.preprocessing import MinMaxScaler



# instantiate MinMaxScaler and use it to rescale X_train and test 

scaler = MinMaxScaler(feature_range=(0, 1))

trainX_norm = scaler.fit_transform(trainX)

testX_norm = scaler.fit_transform(testX)
trainX_norm[1]
# Load libraries for modeling

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
# plot no skill and model roc curves

def plot_roc_curve(test_y, naive_probs, model_probs):

	# plot naive skill roc curve

	fpr, tpr, _ = roc_curve(test_y, naive_probs)

	plt.plot(fpr, tpr, linestyle='--', label='No Skill')

	# plot model roc curve

	fpr, tpr, _ = roc_curve(test_y, model_probs)

	plt.plot(fpr, tpr, marker='.', label='Logistic')

	# axis labels

	plt.xlabel('False Positive Rate')

	plt.ylabel('True Positive Rate')

	# show the legend

	plt.legend()

	# show the plot

	plt.show()
# no skill model, stratified random class predictions

model = DummyClassifier(strategy='stratified')

model.fit(trainX_norm, trainy)



# predict probabilities for each class of the label y 

yhat = model.predict_proba(testX_norm)



# retrieve just the probabilities for the positive class

naive_probs = yhat[:, 1]



# calculate roc auc

roc_auc = roc_auc_score(testy, naive_probs)

print('No Skill ROC AUC %.3f' % roc_auc)
# train and fit a skilled model with Logistic Regression

model = LogisticRegression(solver='lbfgs')

model.fit(trainX_norm, trainy)



yhat = model.predict_proba(testX_norm)

model_probs = yhat[:, 1]



# calculate roc auc

roc_auc = roc_auc_score(testy, model_probs)

print('Logistic ROC AUC %.3f' % roc_auc)



# plot roc curves

plot_roc_curve(testy, naive_probs, model_probs)
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc
def plot_pr_curve(test_y, model_probs):

	# calculate the no skill line as the proportion of the positive class

	no_skill = len(test_y[test_y==1]) / len(test_y)

	# plot the no skill precision-recall curve

	plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

	# plot model precision-recall curve

	precision, recall, _ = precision_recall_curve(testy, model_probs)

	plt.plot(recall, precision, marker='.', label='Logistic')

	# axis labels

	plt.xlabel('Recall')

	plt.ylabel('Precision')

	# show the legend

	plt.legend()

	# show the plot

	plt.show()
# calculate the precision-recall auc

precision, recall, _ = precision_recall_curve(testy, naive_probs)

auc_score = auc(recall, precision)

print('No Skill PR AUC: %.3f' % auc_score)
# calculate the precision-recall auc

precision, recall, _ = precision_recall_curve(testy, model_probs)

auc_score = auc(recall, precision)

print('Logistic PR AUC using normalized data: %.3f' % auc_score)



# plot precision-recall curves

plot_pr_curve(testy, model_probs)
# check the distribution of the first 6 features



from scipy.stats import norm



f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 6))



v1_fraud_dist = data['V1'].loc[data['Class'] == 1].values

sns.distplot(v1_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V1 Distribution', fontsize=14)



v2_fraud_dist = data['V2'].loc[data['Class'] == 1].values

sns.distplot(v2_fraud_dist,ax=ax2, fit=norm, color='#FB8861')

ax2.set_title('V2 Distribution', fontsize=14)



v3_fraud_dist = data['V3'].loc[data['Class'] == 1].values

sns.distplot(v3_fraud_dist,ax=ax3, fit=norm, color='#FB8861')

ax3.set_title('V3 Distribution', fontsize=14)



v4_fraud_dist = data['V4'].loc[data['Class'] == 1].values

sns.distplot(v4_fraud_dist,ax=ax4, fit=norm, color='#FB8861')

ax4.set_title('V4 Distribution', fontsize=14)



v5_fraud_dist = data['V5'].loc[data['Class'] == 1].values

sns.distplot(v5_fraud_dist,ax=ax5, fit=norm, color='#FB8861')

ax5.set_title('V5 Distribution', fontsize=14)



v6_fraud_dist = data['V6'].loc[data['Class'] == 1].values

sns.distplot(v6_fraud_dist,ax=ax6, fit=norm, color='#FB8861')

ax6.set_title('V6 Distribution', fontsize=14)



plt.show()
# check if there are outliers in the features 



list = [1,2,3,4,5,6]



for i in list:

    ax = sns.boxplot(x=data.iloc[i])

    plt.show();
# data standardization with sklearn

from sklearn.preprocessing import StandardScaler



# Instantiate a StandardScaler 

scale = StandardScaler()

trainX_stand = scale.fit_transform(trainX)

testX_stand = scale.fit_transform(testX)
# train and fit a skilled model with Logistic Regression, this time using trainX_stand

model1 = LogisticRegression(solver='lbfgs')

model1.fit(trainX_stand, trainy)



yhat1 = model1.predict_proba(testX_stand)

model1_probs = yhat1[:, 1]



# calculate the precision-recall auc

precision, recall, _ = precision_recall_curve(testy, model1_probs)

auc_score = auc(recall, precision)

print('Logistic PR AUC using standardized data: %.3f' % auc_score)



# plot precision-recall curves

plot_pr_curve(testy, model1_probs)
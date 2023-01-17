import pandas as pd, numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

import warnings 

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/wine-quality-classification/wine_quality_classification.csv', engine='python')
data.head()
data.info()
data.isnull().sum()
data['quality'] = data['quality'].replace({'good' : 1, 'bad' : 0})
data.describe()
data.quantile(np.linspace(0.90,1,12))
x_vars = data.columns[data.columns != 'quality']

fig,ax = plt.subplots(len(x_vars))

fig.set_figheight(24)

fig.set_figwidth(12)

for num,i in enumerate(x_vars) : 

    ax[num].set_title(i)

    ax[num].set_xlabel('')

    sns.boxplot(data[i],ax=ax[num])
# removing outliers : 

x_vars = data.columns[data.columns != 'quality']

for i in x_vars :

    q1 = data[i].quantile(0.25)

    q3 = data[i].quantile(0.75)

    upper_extreme = data[i].quantile(0.75) + 1.5*(q3-q1) # q3-q1 is IQR

    lower_extreme = data[i].quantile(0.75) - 1.5*(q3-q1)

    mask =  (data[i] > lower_extreme) & (data[i] < upper_extreme)  # sans outliers

    outliers = data[mask].index

    data.drop(index=outliers)

from sklearn.model_selection import train_test_split

y = data.pop('quality')

X = data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)
# In our case, all the independent variables are continuous

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])



# Scaling test set for later use

X_test[X_train.columns] = scaler.transform(X_test[X_train.columns])
plt.figure(figsize=[20,10])

sns.heatmap(X_train.corr(),annot=True)

plt.title('Visualizing Correlations')

plt.show()
import statsmodels.api as sm
# Logistic Regression Model 

logm1 = sm.GLM(y_train, sm.add_constant(X_train),family=sm.families.Binomial())

logm1.fit().summary()
from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()
from sklearn.feature_selection import RFE 

rfe = RFE(logReg,10)

rfe = rfe.fit(X_train,y_train)
## RFE results

rfe_results = list(zip(X_train.columns,rfe.support_,rfe.ranking_))

sorted(rfe_results,key=lambda x : (x[2]))
X_train.drop(columns=['pH'],inplace=True)

X_test.drop(columns=['pH'],inplace=True)

X_train.columns = X_train.columns[X_train.columns !='pH']

logm1 = sm.GLM(y_train, sm.add_constant(X_train),family=sm.families.Binomial())

logm1.fit().summary()
X = X_train.loc[:,X_train.columns != 'free sulfur dioxide']

logm2 = sm.GLM(y_train, sm.add_constant(X),family=sm.families.Binomial())

logm2.fit().summary()
X = X.loc[:,X.columns != 'free sulfur dioxide']

logm3 = sm.GLM(y_train, sm.add_constant(X),family=sm.families.Binomial())

logm3.fit().summary()
X = X.loc[:,X.columns != 'density']

logm4 = sm.GLM(y_train, sm.add_constant(X),family=sm.families.Binomial())

logm4.fit().summary()
X = X.loc[:,X.columns != 'chlorides']

logm5 = sm.GLM(y_train, sm.add_constant(X),family=sm.families.Binomial())

logm5.fit().summary()
X = X.loc[:,X.columns != 'residual sugar']

logm6 = sm.GLM(y_train, sm.add_constant(X),family=sm.families.Binomial())

logm6.fit().summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(X) : 

    df = sm.add_constant(X)

    vif = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]

    vif_frame = pd.DataFrame({'vif' : vif[0:]},index = df.columns).reset_index()

    print(vif_frame.sort_values(by='vif',ascending=False))
vif(X)
print('Selected columns :' , X.columns)
logm_final = sm.GLM(y_train, sm.add_constant(X_train[X.columns]),family=sm.families.Binomial())

res = logm_final.fit()

res.summary()
selected_vars = X.columns

y_train_pred = res.predict(sm.add_constant(X_train[X.columns]))
print(y_train_pred.head())
predictions = pd.DataFrame({'Quality' : y_train.values,'class_probability' : y_train_pred.values.reshape(-1)}, index=X_train.index)

print(predictions.head())
predictions['Predicted_Quality'] = predictions['class_probability'].apply(lambda x : 1 if x > 0.5 else 0)

print(predictions.head())
from sklearn import metrics
confusion = metrics.confusion_matrix(predictions['Quality'],predictions['Predicted_Quality'])

print(confusion)
# Accuracy of the model

print(metrics.accuracy_score(predictions['Quality'],predictions['Predicted_Quality']))
TP = confusion[1,1]

TN = confusion[0,0]

FP = confusion[0,1]

FN = confusion[1,0]
#### Metrics

import math

def model_metrics(TP,TN,FP,FN) : 

    print('Accuracy :' , round((TP + TN)/float(TP+TN+FP+FN),3))

    print('Misclassification Rate / Error Rate :', round((FP + FN)/float(TP+TN+FP+FN),3))

    print('Sensitivity / True Positive Rate / Recall :', round(TP/float(FN + TP),3))

    sensitivity = round(TP/float(FN + TP),3)

    print('Specificity / True Negative Rate : ', round(TN/float(TN + FP),3))

    specificity = round(TN/float(TN + FP),3)

    print('False Positive Rate :',round(FP/float(TN + FP),3))

    print('Precision / Positive Predictive Value :', round(TP/float(TP + FP),3))

    precision = round(TP/float(TP + FP),3)

    print('Prevalance :',round((FN + TP)/float(TP+TN+FP+FN),3))

    print('Negative Predictive Value', round(TN/float(TN + FN),3))

    print('Likelihood Ratio : Sensitivity / 1-Specificity :', round(sensitivity/float(1-specificity) ,3))

    print('F1-score :', round(2*precision*sensitivity/(precision + sensitivity),3))
model_metrics(TP,TN,FP,FN)
print(predictions.head())
# generating predictions for cutoffs between 0 and 1

cutoffs = pd.DataFrame()

for i in np.arange(0,1,0.1) : 

    cutoffs[i] = predictions['class_probability'].map(lambda x : 1 if x > i else 0)
tpr = []

fpr = []

for column in cutoffs.columns : 

    confusion = metrics.confusion_matrix(predictions['Quality'],cutoffs[column])

    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives

    tpr.append(TP/float(TP + FN))

    fpr.append(FP/float(FP + TN))

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

sns.scatterplot(fpr,tpr);



sensitivity = []

specificity = []

accuracy = []

coffs = []

for column in cutoffs.columns : 

    confusion = metrics.confusion_matrix(predictions['Quality'],cutoffs[column])

    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives

    sensitivity.append(TP/float(TP + FN))

    specificity.append(1 - FP/float(FP + TN))

    accuracy.append((TP + TN)/(TP + TN + FP + FN))

fig,ax = plt.subplots()

ax.set_xlabel('Cutoffs')

ax.plot(cutoffs.columns,sensitivity,label='sensitivity')

ax.plot(cutoffs.columns,specificity,label='specificity')

ax.plot(cutoffs.columns,accuracy,label='accuracy')

ax.legend(('sensitivity','specificity','accuracy'))

plt.show()
predictions['Final_Predictions'] = predictions['Predicted_Quality'].map(lambda x : 1 if x > 0.5 else 0)
confusion_final = metrics.confusion_matrix(predictions['Quality'],predictions['Final_Predictions'])

TP = confusion_final[1,1]

TN = confusion_final[0,0]

FP = confusion_final[0,1]

FN = confusion_final[1,0]
#### Metrics

model_metrics(TP,TN,FP,FN)
precision = [] # positive predictive power - TP / TP + FP

recall = []   ## same as sensitivity



for column in cutoffs.columns : 

    confusion = metrics.confusion_matrix(predictions['Quality'],cutoffs[column])

    TP = confusion[1,1] # true positive 

    TN = confusion[0,0] # true negatives

    FP = confusion[0,1] # false positives

    FN = confusion[1,0] # false negatives

    precision.append(TP/float(TP + FP))

    recall.append(TP/float(FN + TP))



fig,ax = plt.subplots()

ax.set_xlabel('Cutoffs')

ax.plot(cutoffs.columns,precision,label='precision')

ax.plot(cutoffs.columns,recall,label='recall')

ax.legend(('precision','recall'))

plt.show()
# using sklearn utilities 

from sklearn.metrics import precision_score, recall_score

print('Precision',precision_score(predictions['Quality'],predictions['Predicted_Quality']))

print('Recall', recall_score(predictions['Quality'],predictions['Predicted_Quality']))
print(X_test[X.columns].head())
test_predictions = pd.DataFrame()

X_test_ = X_test[X.columns]

test_predictions['Class_Probabilities'] = res.predict(sm.add_constant(X_test_))
test_predictions['Original'] = y_test

test_predictions.index = y_test.index
# Predictions are made using 0.5 as the threshold

test_predictions['Predicted'] = test_predictions['Class_Probabilities'].map(lambda x : 1 if x > 0.5 else 0)
#### Metrics

TN,FP,FN,TP = metrics.confusion_matrix(test_predictions['Original'],test_predictions['Predicted']).reshape(-1)

model_metrics(TP,TN,FP,FN)

    
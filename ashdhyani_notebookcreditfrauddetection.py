import numpy as np

from numpy import random

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

from sklearn import metrics

from sklearn.metrics import confusion_matrix



pd.set_option('display.max_columns', 200)



creditDataFrame = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv", encoding = 'unicode_escape')

print(creditDataFrame.shape)

print(creditDataFrame.size)
creditDataFrame.info()
creditDataFrame.head()
creditDataFrame.drop(columns= ['Time'], inplace=True)
stdScaler = StandardScaler()

creditDataFrame[['Amount']] = stdScaler.fit_transform(creditDataFrame[['Amount']])

creditDataFrame.head()
print(creditDataFrame['Class'].value_counts()/len(creditDataFrame)*100)

print(creditDataFrame['Class'].value_counts())



sns.countplot(creditDataFrame['Class'])
def getRandomNonFraudData(df):

    random.seed(100)

    nonFraudDF = df[df['Class'] == 0]

    numbers = random.choice(len(nonFraudDF), size=492, replace=False)

    return nonFraudDF.iloc[numbers]
nonFraudDF = getRandomNonFraudData(creditDataFrame)

fraudDF = creditDataFrame[creditDataFrame['Class'] == 1]

mergedDF = pd.concat([nonFraudDF, fraudDF])

mergedDF.head()
mergedDF.shape
plt.figure(figsize  = (15,30))

for i in enumerate(mergedDF.columns.drop('Class')):

    plt.subplot(10, 3, i[0]+1)

    sns.distplot(mergedDF[i[1]])
plt.figure(figsize  = (15,30))

for i in enumerate(mergedDF.columns.drop('Class')):

    plt.subplot(10, 3, i[0]+1)

    sns.boxplot(data=mergedDF, x=i[1])
def treatOutliers(col, df):

    q4 = df[col].quantile(0.95)

    df[col][df[col] >=  q4] = q4

    

    q1 = df[col].quantile(0.05)

    df[col][df[col] <=  q1] = q1

    

    return df
columns = mergedDF.columns.drop('Class')

for i in enumerate(columns):

    mergedDF = treatOutliers(i[1], mergedDF)
## We can see outliers are treated well and data is much better condition now.



plt.figure(figsize  = (15,30))

for i in enumerate(mergedDF.columns.drop('Class')):

    plt.subplot(10, 3, i[0]+1)

    sns.boxplot(data=mergedDF, x=i[1])
corr = mergedDF.corr()

corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

corr_df = corr.unstack().reset_index()

corr_df.columns = ['Variable1', 'Variable2', 'Correlation']

corr_df.dropna(subset = ['Correlation'], inplace=True)

corr_df['Correlation'] = round(corr_df['Correlation'].abs(), 2)

corr_df.sort_values(by = 'Correlation', ascending=False).head(10)
plt.figure(figsize  = (25,15))

sns.heatmap(mergedDF.corr(), annot=True, cmap='RdYlGn')
train_set, test_set = train_test_split(mergedDF, 

                                       train_size=0.7, 

                                       stratify=mergedDF.Class, 

                                       shuffle = True, 

                                       random_state=100)
print(train_set.shape)

print(test_set.shape)
# Futher divide the dataset in X_train and y_train

y_train = train_set.pop('Class')

X_train = train_set
logreg = LogisticRegression()

rfe = RFE(logreg, n_features_to_select=15 )

rfe = rfe.fit(X_train, y_train)
#useful columns according to rfe

useful_cols = X_train.columns[rfe.support_]

useful_cols
# Not useful columns according to rfe

X_train.columns[~rfe.support_]
X_train_rfe = X_train[useful_cols]

X_train_rfe.head()
def checkVIF():

    vif = pd.DataFrame()

    vif['Feaures'] = X_train_sm.columns

    vif['VIF'] = [VIF(X_train_sm.values, i) for i in range(X_train_sm.shape[1])]

    vif = vif.sort_values(by = 'VIF', ascending=False)

    return vif
# Logistic Regression Model

X_train_sm = sm.add_constant(X_train_rfe)

lm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

lm.fit().summary()
checkVIF()
X_train_sm = X_train_sm.drop(columns=['V27'])

lm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

lm.fit().summary()
checkVIF()
X_train_sm = X_train_sm.drop(columns=['V12'])

lm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

lm.fit().summary()
checkVIF()
X_train_sm = X_train_sm.drop(columns=['V24'])

lm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

lm.fit().summary()
checkVIF()
X_train_sm = X_train_sm.drop(columns=['V22'])

lm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

lm = lm.fit()

lm.summary()
checkVIF()
X_train_sm = X_train_sm.drop(columns=['V15'])

lm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

lm = lm.fit()

lm.summary()
checkVIF()
X_train_sm = X_train_sm.drop(columns=['V14'])

lm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

lm = lm.fit()

lm.summary()
checkVIF()
X_train_sm = X_train_sm.drop(columns=['V10'])

lm = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())

lm = lm.fit()

lm.summary()
checkVIF()
y_train_pred = lm.predict(X_train_sm)

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred_final = pd.DataFrame({'Class': y_train.values, 'Class_Prob': y_train_pred})

y_train_pred_final['id'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Class_Predicted'] = y_train_pred_final.Class_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
def drawRoc( actual, probs ):

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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Class, y_train_pred_final.Class_Prob, 

                                         drop_intermediate = False )
drawRoc(y_train_pred_final.Class, y_train_pred_final.Class_Prob)
numbers = [float(x)/10 for x in range(10)]



for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Class_Prob.map(lambda x: 1 if x > i else 0)



y_train_pred_final.head()
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

    

print(cutoff_df)
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Class_Prob.map(lambda x: 1 if x > 0.37 else 0)

y_train_pred_final.head()
y_train_pred_final['final_predicted'].value_counts()
cm = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final.final_predicted)

cm
accuracy = metrics.accuracy_score(y_train_pred_final.Class, y_train_pred_final.final_predicted)*100



tn, fp, fn, tp = metrics.confusion_matrix(y_train_pred_final.Class, y_train_pred_final.final_predicted).ravel()

specificity = tn / (tn + fp)*100



recall = metrics.recall_score(y_train_pred_final.Class, y_train_pred_final.final_predicted)*100



precision = metrics.precision_score(y_train_pred_final.Class, y_train_pred_final.final_predicted)*100



f1_score = metrics.f1_score(y_train_pred_final.Class, y_train_pred_final.final_predicted)*100



print("Accuracy: {0} %".format(round(accuracy, 2)))

print("Specificity: {0} %".format(round(specificity, 2)))

print("Recall: {0} %".format(round(recall, 2)))

print("Precision: {0} %".format(round(precision, 2)))

print("F1-Score: {0} %".format(round(f1_score, 2)))
print(metrics.classification_report(y_train_pred_final.Class, y_train_pred_final.final_predicted))
y_test = test_set.pop('Class')

X_test = test_set
cols = X_train_sm.columns

X_test_final = X_test[cols.drop('const')]



X_test_final = sm.add_constant(X_test_final)

X_test_final.head()
y_test_pred = lm.predict(X_test_final)

y_test_pred = y_test_pred.values.reshape(-1)
y_test_pred_final = pd.DataFrame({'Class': y_test.values, 'Class_Prob': y_test_pred})

y_test_pred_final['id'] = y_test.index

y_test_pred_final.head()
y_test_pred_final['Class_Predicted'] = y_test_pred_final.Class_Prob.map(lambda x: 1 if x > 0.38 else 0)

y_test_pred_final.head()
metrics.confusion_matrix(y_test_pred_final.Class, y_test_pred_final.Class_Predicted)
accuracy = metrics.accuracy_score(y_test_pred_final.Class, y_test_pred_final.Class_Predicted)*100



tn, fp, fn, tp = metrics.confusion_matrix(y_test_pred_final.Class, y_test_pred_final.Class_Predicted).ravel()

specificity = tn / (tn + fp)*100



recall = metrics.recall_score(y_test_pred_final.Class, y_test_pred_final.Class_Predicted)*100



precision = metrics.precision_score(y_test_pred_final.Class, y_test_pred_final.Class_Predicted)*100



f1_score = metrics.f1_score(y_test_pred_final.Class, y_test_pred_final.Class_Predicted)*100



print("Accuracy: {0} %".format(round(accuracy, 2)))

print("Specificity: {0} %".format(round(specificity, 2)))

print("Recall: {0} %".format(round(recall, 2)))

print("Precision: {0} %".format(round(precision, 2)))

print("F1-Score: {0} %".format(round(f1_score, 2)))
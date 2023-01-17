import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import time

%matplotlib inline
import os
import warnings
warnings.filterwarnings('ignore')
# data exploration
df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
df.head()
# set naming convention
df = df.rename(columns={'oldbalanceOrg':'oldBalanceSender', 'newbalanceOrig':'newBalanceSender', 'oldbalanceDest':'oldBalanceReceiver', 'newbalanceDest':'newBalanceReceiver', 'nameOrig':'nameSender', 'nameDest':'nameReceiver'})
df.info()
# are there nulls?
df.isnull().any()
# Is there no information for destination merchants?
print('The column information states that there are no receiving bank information (how much money is in their account) for customers who\'s name starts with M. In the dataset, there are {} rows with data on customers who\'s name starts with M.'
      .format(len(df[df.nameReceiver.str.startswith('M')][(df['oldBalanceReceiver'] != 0) | (df['newBalanceReceiver'] != 0)])))
# What about sending merchants?
print('There are {} rows with data on customers who\'s name starts with M for sending money.'
      .format(len(df[df.nameSender.str.startswith('M')][(df['oldBalanceSender'] != 0) | (df['newBalanceSender'] != 0)])))
# verify limitations of isFlaggedFraud
df[(df['amount'] > 200000) & (df['isFraud'] == 1) & (df['type'] == 'TRANSFER')].head(10)
# condition for isFlaggedFraud doesn't seem to match actual data. The only consistency is that all isFlaggedFraud are greater than $200,000 and isFraud is true. However, the reverse is not true.
# where does fraud occur?
df[df['isFraud'] == 1].type.drop_duplicates()
# fraud occurs only in TRANSFER and CASH_OUTS
print ('There are a total of {} fraudulent transactions out of {} transactions, or {:.2f}%.'
      .format(len(df[df['isFraud'] == 1]), len(df), (len(df[df['isFraud'] == 1]) / len(df) * 100)))
# check to see if there are any merchants with fraudulent charges
print('There are a total of {} fraudulent transactions out of {} transactions for Merchants.'
      .format(len(df[(df['isFraud'] == 1) & (df.nameReceiver.str.startswith('M'))]), len(df[df.nameReceiver.str.startswith('M')])))
# check how many fraudulent transactions have empty newBalanceDest
print('There are {} fraudulent transactions out of {} with 0 balance for the receiving account, or {:.2f}%.'
      .format(len(df[(df['newBalanceReceiver'] == 0) & (df['isFraud'] == 1)]),
              len(df[df['newBalanceReceiver'] == 0]),
             (len(df[(df['newBalanceReceiver'] == 0) & (df['isFraud'] == 1)])/len(df[df['newBalanceReceiver'] == 0]) * 100)))
# check how many non fraudulent transactions have empty newBalanceDest
print('There are {} non fraudulent transactions out of {} with 0 balance for the receiving account, or {:.2f}%.'
      .format(len(df[(df['newBalanceReceiver'] == 0) & (df['isFraud'] == 0)]),
              len(df[df['newBalanceReceiver'] == 0]),
             (len(df[(df['newBalanceReceiver'] == 0) & (df['isFraud'] == 0)])/len(df[df['newBalanceReceiver'] == 0]) * 100)))
# find average amount sent for fraudulent charges
df[df['isFraud'] == 1]['amount'].describe()
# data cleaning
# only two types of transactions occurs in fraud
df_clean = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]
# fraud occurs only in customers who's name does not start with M
df_clean = df_clean[~df_clean.nameSender.str.startswith('M') | ~df_clean.nameReceiver.str.startswith('M')]
df_clean.head()
# clean up the data, remove unnecessary columns
df_clean.drop(['nameSender', 'nameReceiver', 'isFlaggedFraud', 'step'], 1, inplace=True)

# only two values for type, convert to bool; TRANSFER = 1, CASH_OUT = 0
df_clean['type'] = np.where(df_clean['type'] == 'TRANSFER', 1, 0)
df_clean = df_clean.reset_index(drop=True)
df_clean.head()
df_clean.info()
# there is some descrepancy in the data. oldBalanceSender - amount should equal newBalanceSender
# and oldBalanceReceiver + amount should equal newBalanceReceiver but doesn't always occur. create features
# and remove previous balance from both sender and receiver
df_feature = pd.DataFrame(df_clean)
df_feature['errorBalanceSender'] = df_feature.oldBalanceSender - df_feature.amount + df_feature.newBalanceSender
df_feature['errorBalanceReceiver'] = df_feature.oldBalanceReceiver + df_feature.amount - df_feature.newBalanceReceiver
df_feature.drop(['oldBalanceSender', 'oldBalanceReceiver'], 1, inplace=True)
df_feature = df_feature.rename(columns={'newBalanceSender':'balanceSender', 'newBalanceReceiver':'balanceReceiver'})
df_feature['noErrors'] = np.where((df_feature['errorBalanceSender'] == 0) & (df_feature['errorBalanceReceiver'] == 0), 1, 0)
df_feature.head(5)
df_fraud = df_feature[df_feature.isFraud == 1]
df_notFraud = df_feature[df_feature.isFraud == 0]
for col in df_feature.loc[:, ~df_feature.columns.isin(['type', 'isFraud'])]:
    sns.distplot(df_fraud[col])
    sns.distplot(df_notFraud[col])
    plt.legend(['Fraud', 'Not Fraud'], ncol=2, loc='upper right')
    plt.show()
# explore amount, errorBalanceSender more indepth
print(df_fraud.amount.describe())
print(df_notFraud.amount.describe())
print(df_fraud.errorBalanceSender.describe())
print(df_notFraud.errorBalanceSender.describe())
#find how many are wrong 
f, axes = plt.subplots(ncols=4, figsize=(14, 4), sharex=True)
sns.despine(left=True)
axes[0].set_title('Fraudulent Charges')
axes[1].set_title('Fraudulent Charges')
axes[2].set_title('Non Fraudulent Charges')
axes[3].set_title('Non Fraudulent Charges')
sns.distplot(df_fraud.errorBalanceSender, ax=axes[0])
sns.distplot(df_fraud.errorBalanceReceiver, ax=axes[1])
sns.distplot(df_notFraud.errorBalanceSender, ax=axes[2])
sns.distplot(df_notFraud.errorBalanceReceiver, ax=axes[3])
plt.setp(axes, yticks=[])
plt.tight_layout()
sns.distplot(df_fraud.amount)
plt.title('Fraud Amount Distribution')
df_fraud.amount.describe()
sns.distplot(df_notFraud.amount)
plt.title('Non Fraud Amount Distribution')
df_notFraud.amount.describe()
def printErrorOrigin(df):
    print ('Number of charges with 0 error balance from the Originating account is', 
       len(df[df.errorBalanceSender == 0]), 'or ~', (int)((len(df[(df.errorBalanceSender == 0)])/len(df))*100), '%')

def printErrorDest(df):
    print ('Number of charges with 0 error balance from the Destination account is', 
       len(df[df.errorBalanceReceiver == 0]), 'or ~', (int)((len(df[df.errorBalanceReceiver == 0])/len(df))*100), '%')
print('Fraudulent Charges')
printErrorOrigin(df_fraud)
printErrorDest(df_fraud)
print('-' * 40)
print('Non Fraudulent Charges')
printErrorOrigin(df_notFraud)
printErrorDest(df_notFraud)
ax = sns.heatmap(df_feature.corr(), vmin=-.25)
ax.set_title('All Transactions')
ax = sns.heatmap(df_fraud.loc[:, ~df_fraud.columns.isin(['isFraud'])].corr(), vmin=-.25)
ax.set_title('Fraud Transactions')
ax = sns.heatmap(df_notFraud.loc[:, ~df_notFraud.columns.isin(['isFraud'])].corr(), vmin=-.25)
ax.set_title('Non Fraud Transactions')
def setTrainingData(df, test_size):
    X = df.loc[:, ~df.columns.isin(['isFraud'])]
    Y = df.isFraud

    return train_test_split(X, Y, test_size=test_size)

trainX, testX, trainY, testY = setTrainingData(df_feature, .2)
def drawConfusionMatrix(cm, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion Matrix of Transactions')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+labels)
    ax.set_yticklabels(['']+labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def correctFraudCount(y, y_pred):
    labels = ['Not Fraud', 'Fraud']
    cm = confusion_matrix(y, y_pred)
    print(pd.DataFrame(confusion_matrix(y, y_pred),
                       ['Actual Not Fraud', 'Actual Fraud'],
                       ['Predicted Not Fraud', 'Predicted Fraud']))
    y = y.values.reshape(-1, 1)
    count, total = [0, 0]

    for i in range(len(y)):
        if (y[i]==1):
            if (y_pred[i] == 1):
                count = count + 1
            total = total + 1
    print(count, 'fraudulent charges correctly identified out of a total of', total, 'fraudulent charges or {:.3f}%'.format(count/total*100))
    drawConfusionMatrix(cm, labels)
    
def printModel(model, testX, testY, y_pred):
    print('Percent Accuracy: {:.3f}%'.format(model.score(testX, testY)*100))
    correctFraudCount(testY, y_pred)

def runModel(name, model, trainX, trainY, testX, testY):
    print('-' * 20, name, '-' * 20)
    start_time = time.time()
    model.fit(trainX, trainY)
    print("--- Model Fitting in %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    y_pred = model.predict(testX)
    print("--- Model Predicting in %s seconds ---" % (time.time() - start_time))
    printModel(model, testX, testY, y_pred)
# logistic regression - check inverse of regularization strength
lr = LogisticRegression(C=1e10)
lr2 = LogisticRegression(C=1e5)
lr3 = LogisticRegression(C=1)
lr4 = LogisticRegression(C=1e-5)
lr5 = LogisticRegression(C=1e-10)

runModel('Logistic Regression C=1e10', lr, trainX, trainY, testX, testY)
runModel('Logistic Regression C=1e5', lr2, trainX, trainY, testX, testY)
runModel('Logistic Regression C=1 (Default)', lr3, trainX, trainY, testX, testY)
runModel('Logistic Regression C=1e-5', lr4, trainX, trainY, testX, testY)
runModel('Logistic Regression C=1e-10', lr5, trainX, trainY, testX, testY)
# logistic regression - check penalty
lr = LogisticRegression(penalty='l1')
lr2 = LogisticRegression(penalty='l2')

runModel('Logistic Regression l1 Penalty', lr, trainX, trainY, testX, testY)
runModel('Logistic Regression l2 Penalty (Default)', lr2, trainX, trainY, testX, testY)
# logistic regression - check max iteration
lr = LogisticRegression(max_iter=10)
lr2 = LogisticRegression(max_iter=100)
lr3 = LogisticRegression(max_iter=1000)

runModel('Logistic Regression 10 Max Iteration', lr, trainX, trainY, testX, testY)
runModel('Logistic Regression 100 Max Iteration (Default)', lr2, trainX, trainY, testX, testY)
runModel('Logistic Regression 200 Max iteration', lr3, trainX, trainY, testX, testY)
# random forest - check max features
randomForest = RandomForestClassifier(max_features = 'auto', random_state=9834)
randomForest2 = RandomForestClassifier(max_features = None, random_state=9834)
randomForest3 = RandomForestClassifier(max_features = (int)(len(trainX.columns)/2), random_state=9834)

runModel('Random Forest Max Features = Sqrt(# of Features) (Default)', randomForest, trainX, trainY, testX, testY)
runModel('Random Forest Max Features = # of Features ', randomForest2, trainX, trainY, testX, testY)
runModel('Random Forest Max Features = Half the # of Features) ', randomForest3, trainX, trainY, testX, testY)
# random forest - check max depth
randomForest = RandomForestClassifier(max_depth=2, random_state=9834)
randomForest2 = RandomForestClassifier(max_depth=4, random_state=9834)
randomForest3 = RandomForestClassifier(max_depth=6, random_state=9834)
randomForest4 = RandomForestClassifier(max_depth=8, random_state=9834)
randomForest5 = RandomForestClassifier(max_depth=None, random_state=9834)

runModel('Random Forest Max Depth = 2', randomForest, trainX, trainY, testX, testY)
runModel('Random Forest Max Depth = 4', randomForest2, trainX, trainY, testX, testY)
runModel('Random Forest Max Depth = 6', randomForest3, trainX, trainY, testX, testY)
runModel('Random Forest Max Depth = 8', randomForest4, trainX, trainY, testX, testY)
runModel('Random Forest Max Depth = None (Default)', randomForest5, trainX, trainY, testX, testY)
# random forest - check number of trees
randomForest = RandomForestClassifier(n_estimators=5, max_depth=8, random_state=9834)
randomForest2 = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=9834)
randomForest3 = RandomForestClassifier(n_estimators=15, max_depth=8, random_state=9834)
randomForest4 = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=9834)

runModel('Random Forest Number of Trees = 5', randomForest, trainX, trainY, testX, testY)
runModel('Random Forest Number of Trees = 10 (Default)', randomForest2, trainX, trainY, testX, testY)
runModel('Random Forest Number of Trees = 15', randomForest3, trainX, trainY, testX, testY)
runModel('Random Forest Number of Trees = 20', randomForest4, trainX, trainY, testX, testY)
# gradient boosting - check learning rate
gradientBoost = GradientBoostingClassifier(learning_rate=0.1, random_state=9833)
gradientBoost2 = GradientBoostingClassifier(learning_rate=0.25, random_state=9833)
gradientBoost3 = GradientBoostingClassifier(learning_rate=0.5, random_state=9833)
gradientBoost4 = GradientBoostingClassifier(learning_rate=0.75, random_state=9833)

runModel('Gradient Boosting Learning Rate = .1 (Default)', gradientBoost, trainX, trainY, testX, testY)
runModel('Gradient Boosting Learning Rate = .25', gradientBoost2, trainX, trainY, testX, testY)
runModel('Gradient Boosting Learning Rate = .5', gradientBoost3, trainX, trainY, testX, testY)
runModel('Gradient Boosting Learning Rate = .75', gradientBoost4, trainX, trainY, testX, testY)
# gradient boosting - check number of estimators
gradientBoost = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5, random_state=9833)
gradientBoost2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, random_state=9833)
gradientBoost3 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.5, random_state=9833)
gradientBoost4 = GradientBoostingClassifier(n_estimators=500, learning_rate=0.5, random_state=9833)

runModel('Gradient Boosting # of Estimators = 10', gradientBoost, trainX, trainY, testX, testY)
runModel('Gradient Boosting # of Estimators = 100 (Default)', gradientBoost2, trainX, trainY, testX, testY)
runModel('Gradient Boosting # of Estimators = 200', gradientBoost3, trainX, trainY, testX, testY)
runModel('Gradient Boosting # of Estimators = 500', gradientBoost4, trainX, trainY, testX, testY)
# gradient boosting - check max depth
gradientBoost = GradientBoostingClassifier(max_depth=1, random_state=9833)
gradientBoost2 = GradientBoostingClassifier(max_depth=3, random_state=9833)
gradientBoost3 = GradientBoostingClassifier(max_depth=5, random_state=9833)
gradientBoost4 = GradientBoostingClassifier(max_depth=7, random_state=9833)

runModel('Gradient Boosting Max Depth = 1', gradientBoost, trainX, trainY, testX, testY)
runModel('Gradient Boosting Max Depth = 3 (Default)', gradientBoost2, trainX, trainY, testX, testY)
runModel('Gradient Boosting Max Depth = 5', gradientBoost3, trainX, trainY, testX, testY)
runModel('Gradient Boosting Max Depth = 7', gradientBoost4, trainX, trainY, testX, testY)
# gradient boosting - check max features
gradientBoost = GradientBoostingClassifier(max_features = 'auto', max_depth=7, random_state=9833)
gradientBoost2 = GradientBoostingClassifier(max_features = None, max_depth=7, random_state=9833)
gradientBoost3 = GradientBoostingClassifier(max_features = (int)(len(trainX.columns)/2), max_depth=7, random_state=9833)

runModel('Gradient Boost Max Features = Sqrt(# of Features) (Default)', gradientBoost, trainX, trainY, testX, testY)
runModel('Gradient Boost Max Features = # of Features', gradientBoost2, trainX, trainY, testX, testY)
runModel('Gradient Boost Max Features = Half the # of Features', gradientBoost3, trainX, trainY, testX, testY)
# check test size .3
trainX, testX, trainY, testY = setTrainingData(df_feature, .3)
lr = LogisticRegression()
randomForest = RandomForestClassifier(max_depth=8)
gradientBoost = GradientBoostingClassifier(learning_rate=.5, max_depth=7)

runModel('Logistic Regression Test Size 30%', lr, trainX, trainY, testX, testY)
runModel('Random Forest Test Size 30%', randomForest, trainX, trainY, testX, testY)
runModel('Gradient Boosting Test Size 30%', gradientBoost, trainX, trainY, testX, testY)
# check test size .5
trainX, testX, trainY, testY = setTrainingData(df_feature, .5)
lr = LogisticRegression()
randomForest = RandomForestClassifier(max_depth=8)
gradientBoost = GradientBoostingClassifier(learning_rate=.5, max_depth=7)

runModel('Logistic Regression Test Size 50%', lr, trainX, trainY, testX, testY)
runModel('Random Forest Test Size 50%', randomForest, trainX, trainY, testX, testY)
runModel('Gradient Boosting Test Size 50%', gradientBoost, trainX, trainY, testX, testY)
# data is highly skewed, keep data of fradulent charges, but use subsample of non fraudulent charges
df_fraud = df_feature[df_feature['isFraud']==1]
df_notFraud = df_feature[df_feature['isFraud']==0].sample(n=len(df_fraud))

sample_data = pd.concat([df_fraud, df_notFraud], ignore_index=True)
trainX, testX, trainY, testY = setTrainingData(sample_data, .3)
runModel('Logistic Regression', lr, trainX, trainY, testX, testY)
runModel('Random Forest', randomForest, trainX, trainY, testX, testY)
runModel('Gradient Boosting', gradientBoost, trainX, trainY, testX, testY)
# data is highly skewed, keep data of fradulent charges, but use subsample of non fraudulent charges
df_fraud = df_feature[df_feature['isFraud']==1]
df_notFraud = df_feature[df_feature['isFraud']==0].sample(n=(int)(len(df_fraud)/2))

sample_data = pd.concat([df_fraud, df_notFraud], ignore_index=True)
trainX, testX, trainY, testY = setTrainingData(sample_data, .3)
runModel('Logistic Regression', lr, trainX, trainY, testX, testY)
runModel('Random Forest', randomForest, trainX, trainY, testX, testY)
runModel('Gradient Boosting', gradientBoost, trainX, trainY, testX, testY)
feature_importance = gradientBoost.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, testX.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
# try to reduce components
df_reduced = sample_data.loc[:, ~sample_data.columns.isin(['balanceSender', 'type', 'errorBalanceReceiver'])]

trainX, testX, trainY, testY = setTrainingData(df_reduced, .3)
runModel('Logistic Regression', lr, trainX, trainY, testX, testY)
runModel('Random Forest', randomForest, trainX, trainY, testX, testY)
runModel('Gradient Boosting', gradientBoost, trainX, trainY, testX, testY)
df_reduced.columns
X = df_reduced[['amount', 'balanceReceiver', 'errorBalanceSender', 'noErrors']]
Y = df_reduced.isFraud
models = [lr, randomForest, gradientBoost]
names = ['Logistic Regression', 'Random Forest', 'Gradient Boost']
i = 0

print('Cross Validation Scores')
for model in models:
    print('-'*40, names[i], '-'*40)
    print(cross_val_score(model, X, Y, cv=10))
    i += 1
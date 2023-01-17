import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
%matplotlib inline
fraud = pd.read_csv('../input/creditcard.csv')
fraud.shape
fraud.columns
fraud.describe()
#Determine missing values across dataframe
fraud.info()
#First look at Time
sns.distplot(fraud.Time)
plt.title('Distribution of Time')
plt.show()
#Now look at Amount
sns.boxplot(x=fraud['Amount'])
plt.title('Distribution of Amount')
plt.show()
fraud_total = fraud['Class'].sum()
print('Percent Fraud: ' + str(round((fraud_total/fraud.shape[0])*100, 2)) + '%')
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.title("Time < 100,000")
fraud[fraud['Time']<100000]['Time'].hist()


plt.subplot(1,2,2)
plt.title("Time >= 100,000")
fraud[fraud['Time']>=100000]['Time'].hist()

plt.tight_layout()
plt.show()
fraud['10k_time'] = np.where(fraud.Time<100000, 1,0)
features = fraud.drop(['Time'], 1)
# look at distribution below $5.00
np.log10(features[features.Amount < 5]['Amount']+1).hist()
# how many frauds are actually 0 dollars?
print("Non-Fraud Zero dollar Transactions:")
display(features[(features.Amount == 0) & (features.Class == 0)]['Class'].count())
print("Fraudulent Zero dollar Transactions:")
display(features[(features.Amount == 0) & (features.Class == 1)]['Class'].count())
features = fraud[fraud.Amount > 0]
features.head()
features.Amount.quantile(.99)
display(features[(features.Amount < 1000) & (features.Class == 1)]['Class'].count()/features[features.Class==1].shape[0])
# set features equal to purchases less than $1000

# features = features[features.Amount<1000]
sns.distplot(np.log10(features['Amount'][features.Amount>=2]))
# create feature for > $2.00 transaction
features['dollar2'] = np.where(features.Amount > 2, 1, 0)
# features['dollar_1000'] = np.where(features.Amount > 1000, 1, 0)
features = features.drop(['Amount'], 1)
features.head()
# Set X and y for model
X = features.drop(['Class'], 1)
y = features['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# try logistic regression model

lr = LogisticRegression(penalty='l2', solver='liblinear')

# Fit the model.
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))
# display(cross_val_score(lr, X, y, cv=5))
y_pred = lr.predict(X)
print(classification_report(y, y_pred))
display(pd.crosstab(y_pred, y))
# Create smote object
sm = SMOTE(random_state=0)

# Create resampled data
X_res, y_res = sm.fit_resample(X, y)
# create new training and test set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=0)
# fit model
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))
# display(cross_val_score(lr, X_res, y_res, cv=5))

y_pred = lr.predict(X)
print(pd.crosstab(y_pred, y))

print(classification_report(y, y_pred))
#Set up function to run our model with different trees, criterion, max features and max depth
rfc = ensemble.RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1)
%time rfc.fit(X_train, y_train)
print('\n Percentage accuracy for Random Forest Classifier')
print(rfc.score(X_test, y_test)*100, '%')
# print(cross_val_score(rfc, X, Y, cv=5))

# display(cross_val_score(rfc, X_res, y_res, cv=5))
y_pred = rfc.predict(X)
print(pd.crosstab(y_pred, y))

print(classification_report(y, y_pred))


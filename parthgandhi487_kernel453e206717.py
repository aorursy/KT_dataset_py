import seaborn as sns #importing our visualization library

import matplotlib.pyplot as plt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
df = pd.read_csv('/kaggle/input/holida-package/Holiday_Package.csv')
df = df.dropna()

print(df.shape)

print(list(df.columns))
df.head()
df.shape
df.info()
# Dropping unwanted column

df.drop(["Unnamed: 0"],axis=1,inplace=True)
df.head()
# Check for duplicates of data

df.duplicated().sum()
#Check for any missing values

df.isna().sum()
# find categorical variables



categorical = [var for var in df.columns if df[var].dtype=='O']



print('There are {} categorical variables\n'.format(len(categorical)))



print('The categorical variables are :', categorical)

df['Holliday_Package'].unique()
# Data Exploration

df['Holliday_Package'].value_counts()
# Data Exploration

df['foreign'].value_counts()
df.groupby('Holliday_Package').mean()
sns.countplot(x='Holliday_Package',data=df, palette='hls')

plt.show()

plt.savefig('count_plot')
df.groupby('foreign').mean()
sns.countplot(x='foreign',data=df, palette='hls')

plt.show()

plt.savefig('count_plot')
df.groupby('no_older_children').mean()
df.groupby('no_young_children').mean()
%matplotlib inline

pd.crosstab(df.no_older_children,df.Holliday_Package).plot(kind='bar')
%matplotlib inline

pd.crosstab(df.no_young_children,df.Holliday_Package).plot(kind='bar')
table=pd.crosstab(df.no_young_children,df.foreign)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
table=pd.crosstab(df.no_older_children,df.foreign)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
df.age.hist()

plt.title('Histogram of Age')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('hist_age')
df.describe()
# construct box plot for continuous variables

plt.figure(figsize=(10,10))

df.boxplot()
sns.heatmap(df.corr(), annot=True)
sns.pairplot(df)
for column in df.columns:

    if df[column].dtype == 'object':

        print(column.upper(),': ',df[column].nunique())

        print(df[column].value_counts().sort_values())

        print('\n')
# Converting Categorical to Numerical Variable

for feature in df.columns: 

    if df[feature].dtype == 'object':

        df[feature] = pd.Categorical(df[feature]).codes 
df.head()
# Train-Test Split

# Copy all the predictor variables into X dataframe

X = df.drop(['Holliday_Package'],axis=1)



# Copy target into the y dataframe. 

y = df.Holliday_Package
# Split X and y into training and test set in 70:30 ratio

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1)

tuned_parameters = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}

clf = GridSearchCV(LogisticRegression(solver='liblinear'), tuned_parameters, cv=3, scoring="accuracy")

clf.fit(X_train, y_train)
# Fit the Logistic Regression model

model = LogisticRegression(solver='newton-cg',max_iter=10000,penalty='l2',verbose=True,n_jobs=-1)

model.fit(X_train, y_train)
ytrain_predict = model.predict(X_train)

ytest_predict = model.predict(X_test)
ytest_predict_prob=model.predict_proba(X_test)

pd.DataFrame(ytest_predict_prob).head()
# Accuracy - Train Data

model.score(X_train, y_train)
# Train Model Roc_AUC SCore

# predict probabilities

probs = model.predict_proba(X_train)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y_train, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

train_fpr, train_tpr, train_thresholds = roc_curve(y_train, probs)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(train_fpr, train_tpr)
# Accuracy - Test Data

model.score(X_test, y_test)
# Test model roc auc score

# predict probabilities

probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

test_auc = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

test_fpr, test_tpr, test_thresholds = roc_curve(y_test, probs)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(test_fpr, test_tpr)
# Confusion matrix on train data

confusion_matrix(y_train, ytrain_predict)
print(classification_report(y_train, ytrain_predict))
# Confusion Matrix for Test Data

cnf_matrix=confusion_matrix(y_test, ytest_predict)

cnf_matrix
#Test Data Accuracy

test_acc=model.score(X_test,y_test)

test_acc
print(classification_report(y_test, ytest_predict))
# Implementing the model

import statsmodels.api as sm

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary())
# Linear Discriminate Analysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Build LDA Model

# Refer details for LDA at http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

clf = LinearDiscriminantAnalysis()

model1=clf.fit(X_train,y_train)

model1
# Predicting Train Data

# Predict it

# Predict it

pred_class = model1.predict(X_train)
print(classification_report(y_train, pred_class))
# Confusion matrix on train data

#generate Confusion Matrix



confusion_matrix(y_train, pred_class)
#Predicting Test data

model2=clf.fit(X_test,y_test)

model2
pred_class2 = model2.predict(X_test)
print(classification_report(y_test, pred_class2))
# Confusion matrix on test data

confusion_matrix(y_test, pred_class2)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns
# load datasets

df = pd.read_csv("/kaggle/input/hepatitis.data")

df.head()
cols = ["Class","AGE","SEX","STEROID","ANTIVIRALS","FATIGUE","MALAISE","ANOREXIA","LIVER BIG","LIVER FIRM",

        "SPLEEN PALPABLE","SPIDERS","ASCITES","VARICES","BILIRUBIN","ALK PHOSPHATE", 

        "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"]
df = pd.read_csv("/kaggle/input/hepatitis.data", names=cols)

df.head()
df.columns = df.columns.str.lower().str.replace(' ','_')

df = df.replace('?',0)

df.head()
#df.dtypes

df.columns[df.dtypes=='object']
# ignoring 'albumin' and 'bilirubin' as they are float values

df[['steroid', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm',

       'spleen_palpable', 'spiders', 'ascites', 'varices', 

       'alk_phosphate', 'sgot', 'protime']] = df[['steroid', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm',

       'spleen_palpable', 'spiders', 'ascites', 'varices', 

       'alk_phosphate', 'sgot', 'protime']].astype(int)



df[['albumin','bilirubin']]=df[['albumin','bilirubin']].astype(float)
df.dtypes
df.shape
## check for missing values 

df.isnull().sum()
df.describe()
## values 

target_label = {"Die":1, "Live":2}

# plotting 

plt.figure(figsize=(8,4))

df['class'].value_counts().plot(kind='bar')
### Gender classification

# 1=male  2=female

print(df['sex'].unique())

print(df['sex'].value_counts())



#plot gender

plt.figure(figsize=(8,4))

df['sex'].value_counts().plot(kind='bar')
### frequency distribution table using Age range

### dividing age groups

labels = ["< 10","10-20","20-30","30-40","40-50","50-60", "60-70"," > 70"]

bins = [0,10,20,30,40,50,60,70,80]

freq_df = df.groupby(pd.cut(df['age'], bins=bins, labels=labels)).size()

freq_df = freq_df.reset_index(name='count')

freq_df
# pie chart

labels = ["< 10","10-20","20-30","30-40","40-50","50-60", "60-70"," > 70"]

fig1, ax1 = plt.subplots()

ax1.pie(freq_df['count'], labels = labels, autopct = '1%.1f%%')

ax1.axis('equal')

plt.show()
# plot of frequency

width=0.6

plt.bar(freq_df['age'],freq_df['count'],width)

plt.ylabel('Counts')

plt.title('Frequency count of Age')
# Methods

# boxplt, scatterplot, Zscore, InterQuartile range
# Boxplot ( Univariate )

import seaborn as sns

sns.boxplot(df['age'])
sns.boxplot(df['alk_phosphate'])
# scatterplot ( multivariate )

sns.scatterplot(df['age'], df['albumin'])
sns.scatterplot(x= df['albumin'], y=df['age'],hue = df['sex'], palette = ['green','red'], data=df)
# using IQR

# H-spread / Mid_spread

# measure the statistical dispersion

# IQR = quantile 3(75)- 1(25)
q1 = df.quantile(q = 0.25)

q3 = df.quantile(q = 0.75)

IQR = q3-q1

IQR
## get actual datapoint that is an outlier

# True = Outliers

(df < (q1-1.5*IQR))| (df > (q3+1.5*IQR))
# removing all values that are outliers i.e True values

df_no_outlier  = df[~((df < (q1-1.5*IQR))| (df > (q3+1.5*IQR))).any(axis=1)]

df_no_outlier.head()
print(df_no_outlier.shape)

print(df.shape)
# plot of distribution of data

df.hist(bins=50, figsize=(20,15))

plt.show()
df_no_outlier.hist(bins=50, figsize=(20,15))

plt.show()
# SelectKbest

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
# features and labels

df.head()
x_features = df[['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise',

       'anorexia', 'liver_big', 'liver_firm', 'spleen_palpable', 'spiders',

       'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin',

       'protime', 'histology']]



y_label = df['class']
skb = SelectKBest(score_func=chi2, k=10)

best_feature_fit = skb.fit(x_features, y_label)
print("scores: ",best_feature_fit.scores_)
# transform

b_2 = best_feature_fit.transform(x_features)

b_2
# mapping to features and values

f_score = pd.DataFrame(best_feature_fit.scores_,columns=['Feature Scores'])

f_score.head()
features_cols = pd.DataFrame(x_features.columns, columns=['Features Names']) 

features_cols.head()
# concat those 2 df

# higher the number, the more importatnt feature 

best_feat_df = pd.concat([f_score,features_cols], axis=1)

best_feat_df
# get 10 high values

best_feat_df.nlargest(10,'Feature Scores')
# Recurssive Feature Elimination

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression() 
rfe = RFE(lr, 8)

rfe_fit = rfe.fit(x_features, y_label)
# selection 

selected_features = pd.DataFrame(rfe_fit.support_, columns=['Selected Features'])# selection 

ranking_features = pd.DataFrame(rfe_fit.ranking_, columns=['Ranking Features'])

rfe_feature_df = pd.concat([features_cols, selected_features,ranking_features], axis=1)

rfe_feature_df
# eliminate lower values ranking or True



### Feature_Importance extraction

# Extra tree classifier

from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()

clf.fit(x_features,y_label)
print(clf.feature_importances_)
feature_importance_df = pd.Series(clf.feature_importances_, index=x_features.columns)

feature_importance_df
df.corr()
#heat map for correlation

plt.figure(figsize=(20,10))

sns.heatmap(x_features.corr(), annot=True)
## Model deployment##

# Feature and Labels

# train test split

# Logistic Regression

# KNN 

# DCT



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
# Initial features

x_features.head()
y_labels = df['class']

y_labels.shape
# Selected Features

best_feat_df.nlargest(12,'Feature Scores')['Features Names'].unique()
x_features_best = df[['age', 'sex', 'steroid', 'antivirals', 'fatigue',

       'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin',

       'protime', 'histology']]
# original dataset

X_train, X_test, y_train, y_test = train_test_split(x_features,y_labels, test_size = 0.30, random_state = 3)
# best features of dataset

X_train_b, X_test_b , y_train_b, y_test_b = train_test_split(x_features_best,y_labels, test_size = 0.30, random_state = 3)
# Logistic Regression

lr = LogisticRegression(max_iter=210, C=2)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Training Accuracy on New feature dataset", accuracy_score(y_train, lr.predict(X_train)))

print("Testing Accuracy on Original dataset", accuracy_score(y_test, y_pred))
lr_best = LogisticRegression(max_iter=210, C=2)

lr_best.fit(X_train_b, y_train_b)

y_pred_b = lr_best.predict(X_test_b)

print("Training Accuracy on New feature dataset", accuracy_score(y_train_b, lr_best.predict(X_train_b)))

print("Testing Accuracy on New feature dataset", accuracy_score(y_test_b, y_pred_b))
## KNN

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Training Accuracy on original dataset", accuracy_score(y_train, knn.predict(X_train)))

print("Testing Accuracy on Original dataset", accuracy_score(y_test, y_pred))
## KNN

knn_best = KNeighborsClassifier(n_neighbors=5)

knn_best.fit(X_train_b, y_train_b)

y_pred_b = knn_best.predict(X_test_b)

print("Training Accuracy on New feature dataset", accuracy_score(y_train_b, knn_best.predict(X_train_b)))

print("Testing Accuracy on New feature dataset", accuracy_score(y_test_b, y_pred_b))
## Decision Tree

dct = DecisionTreeClassifier()

dct.fit(X_train, y_train)

y_pred = dct.predict(X_test)

print("Training Accuracy on original dataset", accuracy_score(y_train, dct.predict(X_train)))

print("Testing Accuracy on Original dataset", accuracy_score(y_test, y_pred))
dct_best = DecisionTreeClassifier()

dct_best.fit(X_train_b, y_train_b)

y_pred_b = dct_best.predict(X_test_b)

print("Training Accuracy on New feature dataset", accuracy_score(y_train_b, dct_best.predict(X_train_b)))

print("Testing Accuracy on New feature dataset", accuracy_score(y_test_b, y_pred_b))
## single prediction

x1 = X_test.iloc[1]

x1
y_labels.iloc[1]
## Single Prediction

x1_pred = lr.predict(np.array(x1).reshape(1,-1))

x1_pred
x1_pred = knn.predict(np.array(x1).reshape(1,-1))

x1_pred
x1_pred = dct.predict(np.array(x1).reshape(1,-1))

x1_pred
# 1- Die 

# 2-Live
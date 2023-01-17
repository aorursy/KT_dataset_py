# numerical analysis
import numpy as np
# storing and processing in dataframes
import pandas as pd
# simple plotting
import matplotlib.pyplot as plt
# advanced plotting
import seaborn as sns

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
# scaling features
from sklearn.preprocessing import StandardScaler
# selecting important features
from sklearn.feature_selection import RFECV
# k nearest neighbors model
from sklearn.neighbors import KNeighborsClassifier
# accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
# plot style
sns.set_style('whitegrid')
# color palettes
pal = ['#0e2433', '#ff007f']
# read data
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

# first few rows
df.head()
# no. of rows and columns
df.shape
# columns names
df.columns
# random rows
# df.sample(5)
# descriptive statistics
df.describe(include='all')
# consise summary of dataframe
# df.info()
# no. of na values in each columns
df.isna().sum()
# no of values in each class
print(df['diagnosis'].value_counts())

# plot class distribution
sns.countplot(df['diagnosis'], palette=pal, alpha=0.8)
plt.show()
fig, ax = plt.subplots()
m = ax.hist(df[df["diagnosis"] == "M"]['radius_mean'], bins=20, range=(0, 30), 
            label = "Malignant", alpha=0.7, color='#232121')
b = ax.hist(df[df["diagnosis"] == "B"]['radius_mean'], bins=20, range=(0, 30), 
            label = "Benign", alpha=0.7, color='#df2378')
plt.xlabel("Radius")
plt.ylabel("Count")
plt.title("Mean Radius")
plt.legend()
plt.show()
print('Min radius of benign cancer :', df[df['diagnosis']=='B']['radius_mean'].min())
print('Max radius of benign cancer :', df[df['diagnosis']=='B']['radius_mean'].max())
print('Min radius of malignant cancer :', df[df['diagnosis']=='M']['radius_mean'].min())
print('Min radius of malignant cancer :', df[df['diagnosis']=='M']['radius_mean'].max())
# figure size
plt.figure(figsize=(25, 12))
# plot heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdGy')
# show figure
plt.show()
# # figure size
plt.figure(figsize=(6, 6))
# # plot pairplot
sns.pairplot(df, hue="diagnosis", palette=pal)
# # show figure
plt.plot()
#sns.boxplot(data = df , x)
#sns.boxplot?
# Equalize class distribution
# Drop unwanted columns

df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
print(df.shape)
# encoding diagnosis data

df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x=='M' else 0)
# scale features
#scaler = StandardScaler()
#df_trans = pd.DataFrame(scaler.fit_transform(X))
# pca



# plt.figure(figsize=(18, 6))
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=df_melt)
# plt.show()
# plt.scatter_matrix(df)
# features and labels
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# model initialization
model = KNeighborsClassifier()

# model fitting
model.fit(X_train, y_train)

# predict using the model
pred = model.predict(X_test)

# model validation
print(accuracy_score(pred, y_test))
print(confusion_matrix(pred, y_test))
print(classification_report(pred, y_test))
sns.heatmap(confusion_matrix(pred, y_test), annot=True, fmt="d")
# # random forest classifier
# model = RandomForestClassifier() 

# # recursive feature elimination with cross validation
# rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
# rfecv = rfecv.fit(X_train, y_train)

# # predict using the model
# pred = model.predict(X_test)

# print(rfecv.n_features_)
# print(X_train.columns[rfecv.support_])

# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score of number of selected features")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()

# # model validation
# print(accuracy_score(pred, y_test))
# print(confusion_matrix(pred, y_test))
# sns.heatmap(confusion_matrix(pred, y_test), annot=True, fmt="d")

# # roc-auc
# probs = model.predict_proba(X_test)
# preds = probs[:,1]
# fpr, tpr, threshold = roc_curve(y_test, preds)
# roc_auc = auc(fpr, tpr)

# # roc-auc plot
# plt.title('ROC')
# plt.plot(fpr, tpr, 'b--', label='AUC = %0.2f' % roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.legend(loc='lower right')
# plt.show()
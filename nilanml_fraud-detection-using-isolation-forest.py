import pandas as pd

import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv('../input/creditcard.csv')

df.head()
df.dtypes
df.describe()
df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt



corr = df.corrwith(df['Class']).reset_index()

corr.columns = ['Index','Correlations']

corr = corr.set_index('Index')

corr = corr.sort_values(by=['Correlations'], ascending = False)

plt.figure(figsize=(4,15))

fig = sns.heatmap(corr, annot=True, fmt="g", cmap='YlGnBu')

plt.title("Correlation of Variables with Class")

plt.show()
plt.figure(figsize=(15,4))

fig = sns.distplot(df['Time'], kde=False, color="green")

plt.show()
plt.figure(figsize=(8,4))

fig = sns.violinplot(x=df["Amount"], color="lightblue")

plt.show()
plt.figure(figsize=(8,4))

fig = plt.scatter(x=df[df['Class'] == 1]['Time'], y=df[df['Class'] == 1]['Amount'], color="c")

plt.title("Time vs Transaction Amount in Fraud Cases")

plt.show()
plt.figure(figsize=(8,4))

fig = plt.scatter(x=df[df['Class'] == 0]['Time'], y=df[df['Class'] == 0]['Amount'], color="dodgerblue")

plt.title("Time vs Transaction Amount in Legit Cases")

plt.show()
df.hist(figsize=(20,20), color = "salmon")

plt.show()
plt.figure(figsize=(7,5))

fig = sns.countplot(x="Class", data=df)

plt.show()
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import train_test_split
inliers = df[df.Class==0]

inliers = inliers.drop(['Class'], axis=1)

outliers = df[df.Class==1]

outliers = outliers.drop(['Class'], axis=1)

inliers_train, inliers_test = train_test_split(inliers, test_size=0.30, random_state=42)
inliers_train.head()
inliers_test.head()
model = IsolationForest()

model.fit(inliers_train)

inlier_pred_test = model.predict(inliers_test)

outlier_pred = model.predict(outliers)
print("Accuracy in Detecting Legit Cases:", list(inlier_pred_test).count(1)/inlier_pred_test.shape[0])

print("Accuracy in Detecting Fraud Cases:", list(outlier_pred).count(-1)/outlier_pred.shape[0])
from sklearn.neighbors import LocalOutlierFactor



model = LocalOutlierFactor(novelty=True)

model.fit(inliers_train)

inlier_pred_test = model.predict(inliers_test)

outlier_pred = model.predict(outliers)
print("Accuracy in Detecting Legit Cases:", list(inlier_pred_test).count(1)/inlier_pred_test.shape[0])

print("Accuracy in Detecting Fraud Cases:", list(outlier_pred).count(-1)/outlier_pred.shape[0])
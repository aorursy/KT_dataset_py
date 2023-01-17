# Ignore Warnings

import warnings

warnings.filterwarnings('ignore')
# Import Main Packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Import Main Packages For Visualization

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

# Show Our Fils Data

import os

print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

sub = pd.read_csv("../input/sample_submission.csv")



print("Data are Ready!!")
train_df.head()
if (train_df.isnull().values.any() == False):

    print("No Missing Data")

else:

    train_df.isnull().sum()
print("Train Data Size {}\nTest Data Size {}".format(train_df.shape, test_df.shape))
# Show Labels Values

train_df['target'].value_counts()
# Describe 0 Value

train_df[train_df.target == 0].describe()
# Describe 1 Value

train_df[train_df.target == 1].describe()
features = train_df.drop(['ID_code', 'target'], axis=1)

label = train_df['target']
# EDA

fig, ax = plt.subplots(1, 2, figsize=(20, 8))



sns.countplot(label, ax=ax[0])

sns.violinplot(x=label.values, y=label.index.values, ax=ax[1])
trn_corr = features.corr()

trn_corr = trn_corr.values.flatten()

trn_corr = trn_corr[trn_corr != 1]



plt.figure(figsize=(20, 8))

sns.distplot(trn_corr, color="Green", label="train")

plt.xlabel("Correlation values found in train (except 1)")

plt.ylabel("Density")

plt.title("Are there correlations between features?"); 

plt.legend();
train_correlations = train_df.drop(["target"], axis=1).corr()

train_correlations = train_correlations.values.flatten()

train_correlations = train_correlations[train_correlations != 1]



test_correlations = test_df.corr()

test_correlations = test_correlations.values.flatten()

test_correlations = test_correlations[test_correlations != 1]



plt.figure(figsize=(20,8))

sns.distplot(train_correlations, color="Red", label="train")

sns.distplot(test_correlations, color="Green", label="test")

plt.xlabel("Correlation values found in train (except 1)")

plt.ylabel("Density")

plt.title("Are there correlations between features?"); 

plt.legend();
# Import Models

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB



from sklearn.pipeline import make_pipeline



from sklearn.model_selection import train_test_split # You Can Comment It



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X = features.values.astype('float64')

y = label.values.astype('float64')
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = GaussianNB() # Set Model

model.fit(X, y) # Fit Features and labels



y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
plt.figure(figsize=(12, 8))

mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True)

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.title("Confusion Matrix");
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_curve, auc



fpr, tpr, thr = roc_curve(y, model.predict_proba(X)[:,1])

plt.figure(figsize=(12, 8))

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot')

auc(fpr, tpr) * 100
from sklearn.pipeline import make_pipeline # Import pipeline

from sklearn.preprocessing import QuantileTransformer # For Processing Data.



# Set Model

pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())

pipeline.fit(X, y)
p_pred = pipeline.predict(X_test)
accuracy_score(y_test, p_pred)
print(classification_report(y_test, p_pred))
from sklearn.metrics import roc_curve, auc



fpr, tpr, thr = roc_curve(y, pipeline.predict_proba(X)[:,1])

plt.figure(figsize=(12, 8))

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic Plot')

auc(fpr, tpr) * 100
# Show Test File

test_df.head()
# Drop Unused columns

x_test = test_df.drop(['ID_code'], 1).values
gnb_pred = model.predict_proba(x_test)[:, 1] # GaussianNB => gnb

pip_pred = pipeline.predict_proba(x_test)[:, 1] # Pipeline => pip
mean_pred = (gnb_pred + pip_pred) / 2.0
sub.head()
sub['target'] = gnb_pred

sub.to_csv('gnb_submission.csv', index=False)

sub.head()
sub['target'] = pip_pred

sub.to_csv('pip_submission.csv', index=False)

sub.head()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()
df.describe()
df.isna().sum().any()
df.Class.value_counts()
sns.countplot(x='Class', data=df)
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])



plt.show()
from sklearn.preprocessing import RobustScaler

# We will use RobustScaler as it is less prone to outliers.
rob_scaler = RobustScaler()

amount_ = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
time_ = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

df.insert(0, 'Amount_', amount_)
df.insert(1, 'Time_', time_)

# Amount and Time are now scaled!

df.head()
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# Lets shuffle the data before creating the subsamples

# Suffling
df = df.sample(frac=1)

# Splitting
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

balanced_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle again
balanced_df = balanced_df.sample(frac=1, random_state=42)

balanced_df.head()
sns.countplot(x='Class', data=balanced_df)
# Correlating
corr = balanced_df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20})
# Box-plots
variables = balanced_df.columns.values[:-1]

i = 0

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(5,6,figsize=(24,18))

for feature in variables:
    i += 1
    plt.subplot(5,6,i)
    sns.boxplot(y=feature, x='Class', data=balanced_df)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

# New_df is from the random undersample data (fewer instances)
X = balanced_df.drop('Class', axis=1)
y = balanced_df['Class']


# T-SNE Implementation
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)

# PCA Implementation
X_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)

# TruncatedSVD
X_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)

# Plotting decompositions
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_cluster = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_cluster = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_cluster, red_cluster])


# PCA scatter plot
ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_cluster, red_cluster])

# TruncatedSVD scatter plot
ax3.scatter(X_svd[:,0], X_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_svd[:,0], X_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_cluster, red_cluster])

plt.show()
# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 

import collections

# Let us use the balanced data, undersampling.
X = balanced_df.drop('Class', axis=1)
y = balanced_df['Class']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting to arrays.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


# Using the following four classifiers
classifiers = {
    "Logisitic Regression": LogisticRegression(),
    "K Nearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

# Validating the scores, ROC Cureves, Confusion Matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_curve, confusion_matrix

results = pd.DataFrame(columns=['Accuracy %', 'Cross Val %', 'F1-Score'])
rocs = {}
confusion_matrices = {}
for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    cv_score = cross_val_score(classifier, X_train, y_train, cv=5)
    accuracy = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)    
    results.loc[key] = [100*round(accuracy,4), round(cv_score.mean(),4)*100, round(f1,2)]
    
    rocs[key] = roc_curve(y_test, y_pred)
    confusion_matrices[key] = confusion_matrix(y_test, y_pred)
results
fig, ax = plt.subplots(2, 2,figsize=(14,6))
fig.suptitle('Comparing Confusion Matrices', fontsize = 18)

i=0;
for k,cm in confusion_matrices.items():
    i += 1
    ax = plt.subplot(2,2,i)
    sns.heatmap(cm, annot=True, cmap=plt.cm.Greens)
    ax.set_title(k, fontsize=12)
    ax.set_xticklabels(['', ''], fontsize=8, rotation=90)
    ax.set_yticklabels(['', ''], fontsize=8, rotation=360)

plt.show()
# Some mandatory Libraries
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

#EDA Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Classifier Libraries
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Other Libraries
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

%matplotlib inline
warnings.filterwarnings("ignore")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
print(df.columns)
print('Our total data point: ', df.shape)
df.describe()
df.isnull().sum()

# No null values present in data frame.
# The classes are heavily skewed we need to solve this issue later.
print('No. of no frauds in persentage: ', round(df['Class'].value_counts()[0]/len(df) * 100,2))
print('No. of frauds in persentage: ', round(df['Class'].value_counts()[1]/len(df) * 100,2))
ax = sns.countplot(x='Class', data=df)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 50

ax1.hist(df.Time[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Time[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()

print ("Fraud")
print (df.Amount[df.Class == 1].describe())
print ()
print ("Normal")
print (df.Amount[df.Class == 0].describe())

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))

bins = 30

ax1.hist(df.Amount[df.Class == 1], bins = bins)
ax1.set_title('Fraud')

ax2.hist(df.Amount[df.Class == 0], bins = bins)
ax2.set_title('Normal')

plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.yscale('log')
plt.show()

df['Amount_max_fraud'] = 1
df.loc[df.Amount <= 2125.87, 'Amount_max_fraud'] = 0

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))

ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
ax1.set_title('Fraud')

ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

#Select only the anonymized features.
import matplotlib.gridspec as gridspec

v_features = df.iloc[:,1:29].columns
plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, cn in enumerate(df[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

with sns.axes_style("white"):
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(15,10))
    ax = sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)

f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x='Class', y='V2', data=df, ax=axes[0])
axes[0].set_title('V2 vs Class')

sns.boxplot(x='Class', y='V4', data=df, ax=axes[1])
axes[1].set_title('V4 vs Class')

sns.boxplot(x='Class', y='V11', data=df, ax=axes[2])
axes[2].set_title('V11 vs Class')

sns.boxplot(x='Class', y='V19', data=df, ax=axes[3])
axes[3].set_title('V19 vs Class')

plt.show()
# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)

f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x='Class', y='V10', data=df, ax=axes[0])
axes[0].set_title('V10 vs Class')

sns.boxplot(x='Class', y='V12', data=df, ax=axes[1])
axes[1].set_title('V12 vs Class')

sns.boxplot(x='Class', y='V14', data=df,  ax=axes[2])
axes[2].set_title('V14 vs Class')

sns.boxplot(x='Class', y='V17', data=df, ax=axes[3])
axes[3].set_title('V17 vs Class')

plt.show()
#to edit later 
print('Our current dataset shape based on class is', df[df['Class'] == 0].shape,' and ',df[df['Class'] == 1].shape)
# Class count
count_class_0, count_class_1 = df.Class.value_counts()

# Divide by class
df_0 = df[df['Class'] == 0]
df_1 = df[df['Class'] == 1]

# Create new dataset
df_1_over_sampled = df_1.sample(count_class_0, replace=True)
df_final = pd.concat([df_0, df_1_over_sampled], axis=0)

print('New dataset contain: ', df_final[df_final['Class'] == 0].shape, df_final[df_final['Class'] == 1].shape)
print('Distribution of the Classes in the subsample dataset')
print(df_final['Class'].value_counts()/len(df_final))
# New_df is from the random undersample data (fewer instances)

X = df_final.drop('Class', axis=1)
Y = df_final['Class']
# Before StandardScaler:

f, axes = plt.subplots(ncols=2, figsize=(10,4))

sns.distplot(df_final['Amount'], kde=False, color="g", ax=axes[0])

sns.distplot(df_final['Time'], kde=False, color="g", ax=axes[1])

plt.tight_layout()
# Create the Scaler object
scaler = StandardScaler()

# Fit and Transform Amount
scaler.fit(df_final['Amount'].values.reshape(-1, 1))
df_final['std_Amount'] = scaler.transform(df_final['Amount'].values.reshape(-1, 1))

# Fit and Transform Time
scaler.fit(df_final['Time'].values.reshape(-1, 1))
df_final['std_Time'] = scaler.transform(df_final['Time'].values.reshape(-1, 1))

# Delete old features
df_final.drop(['Time','Amount'], axis=1, inplace=True)
# After StandardScaler:

f, axes = plt.subplots(ncols=2, figsize=(10,4))

sns.distplot(df_final['std_Amount'], kde=False, color="g", ax=axes[0])

sns.distplot(df_final['std_Time'], kde=False, color="g", ax=axes[1])

plt.tight_layout()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=22)
# Define classifiers with default parameters.

classifiers = {
    "Logisitic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Linear Regression":linear_model.LinearRegression(),
    "Gaussian NB": GaussianNB()
}
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train) 
    training_score = cross_val_score(classifier, X_train, y_train, cv=3)
    print('Classifiers: ',name, 'has a training score of', round(training_score.mean(),2) * 100)
# LogisticRegression
params = {"penalty": ['l1', 'l2'],
          'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

gs = GridSearchCV(LogisticRegression(), 
                  params, 
                  cv = 3, 
                  n_jobs=-1)
gs_results = gs.fit(X_train, y_train)

log_reg = gs.best_estimator_ # store best estimators for future analysis

print('Best Accuracy: ', gs_results.best_score_)
print('Best Parametrs: ', gs_results.best_params_)
# KNN
params = {'n_neighbors':list(range(1, 50, 2)), 'weights':['uniform', 'distance']}

gs = GridSearchCV(KNeighborsClassifier(),
                  params, 
                  cv = 3, 
                  n_jobs=-1)
gs_results = gs.fit(X_train, y_train)

knears_neighbors = gs.best_estimator_ # store best estimators for future analysis

print('Best Accuracy: ', gs_results.best_score_)
print('Best Parametrs: ', gs_results.best_params_)
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=3)
print('LR CV Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=3)
print('KNN CV Score: ', round(knears_score.mean() * 100, 2).astype(str) + '%')
model = LogisticRegression(penalty = 'l2', C = 0.1)
model.fit(X_train,y_train)
pridict = model.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, pridict))
cm = confusion_matrix(y_test, pridict)
labels = ['No Fraud', 'Fraud']
print(pd.DataFrame(cm, index=labels, columns=labels))
lr_pred_prob = model.predict_proba(X_test)[:,1]
fpr,tpr,thrsld = roc_curve(y_test,lr_pred_prob)
print('AUC score:',roc_auc_score(y_test,lr_pred_prob))
# Ploting ROC Curve 

plt.figure(figsize=(7,5))
plt.plot([0,1],[0,1])
plt.plot(fpr,tpr,':', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid()
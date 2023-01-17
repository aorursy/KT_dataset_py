#importing packages

%matplotlib inline

import scipy.stats as stats

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
#shape

print('This data frame has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))
#peek at data

df.sample(5)
#info

df.info()
#numerical summary -> only non-anonymized columns of interest

pd.set_option('precision', 3)

df.loc[:, ['Time', 'Amount']].describe()
#visualizations of time and amount

plt.figure(figsize=(10,8))

plt.title('Distribution of Time Feature')

sns.distplot(df.Time)
plt.figure(figsize=(10,8))

plt.title('Distribution of Monetary Value Feature')

sns.distplot(df.Amount)
#fraud vs. normal transactions 

counts = df.Class.value_counts()

normal = counts[0]

fraudulent = counts[1]

perc_normal = (normal/(normal+fraudulent))*100

perc_fraudulent = (fraudulent/(normal+fraudulent))*100

print('There were {} non-fraudulent transactions ({:.3f}%) and {} fraudulent transactions ({:.3f}%).'.format(normal, perc_normal, fraudulent, perc_fraudulent))
plt.figure(figsize=(8,6))

sns.barplot(x=counts.index, y=counts)

plt.title('Count of Fraudulent vs. Non-Fraudulent Transactions')

plt.ylabel('Count')

plt.xlabel('Class (0:Non-Fraudulent, 1:Fraudulent)')
corr = df.corr()

corr
#heatmap

corr = df.corr()

plt.figure(figsize=(12,10))

heat = sns.heatmap(data=corr)

plt.title('Heatmap of Correlation')
#skewness

skew_ = df.skew()

skew_
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler2 = StandardScaler()

#scaling time

scaled_time = scaler.fit_transform(df[['Time']])

flat_list1 = [item for sublist in scaled_time.tolist() for item in sublist]

scaled_time = pd.Series(flat_list1)
#scaling the amount column

scaled_amount = scaler2.fit_transform(df[['Amount']])

flat_list2 = [item for sublist in scaled_amount.tolist() for item in sublist]

scaled_amount = pd.Series(flat_list2)
#concatenating newly created columns w original df

df = pd.concat([df, scaled_amount.rename('scaled_amount'), scaled_time.rename('scaled_time')], axis=1)

df.sample(5)
#dropping old amount and time columns

df.drop(['Amount', 'Time'], axis=1, inplace=True)
#manual train test split using numpy's random.rand

mask = np.random.rand(len(df)) < 0.9

train = df[mask]

test = df[~mask]

print('Train Shape: {}\nTest Shape: {}'.format(train.shape, test.shape))
train.reset_index(drop=True, inplace=True)

test.reset_index(drop=True, inplace=True)
#how many random samples from normal transactions do we need?

no_of_frauds = train.Class.value_counts()[1]

print('There are {} fraudulent transactions in the train data.'.format(no_of_frauds))
#randomly selecting 442 random non-fraudulent transactions

non_fraud = train[train['Class'] == 0]

fraud = train[train['Class'] == 1]
selected = non_fraud.sample(no_of_frauds)

selected.head()
#concatenating both into a subsample data set with equal class distribution

selected.reset_index(drop=True, inplace=True)

fraud.reset_index(drop=True, inplace=True)
subsample = pd.concat([selected, fraud])

len(subsample)
#shuffling our data set

subsample = subsample.sample(frac=1).reset_index(drop=True)

subsample.head(10)
new_counts = subsample.Class.value_counts()

plt.figure(figsize=(8,6))

sns.barplot(x=new_counts.index, y=new_counts)

plt.title('Count of Fraudulent vs. Non-Fraudulent Transactions In Subsample')

plt.ylabel('Count')

plt.xlabel('Class (0:Non-Fraudulent, 1:Fraudulent)')
#taking a look at correlations once more

corr = subsample.corr()

corr = corr[['Class']]

corr
#negative correlations smaller than -0.5

corr[corr.Class < -0.5]
#positive correlations greater than 0.5

corr[corr.Class > 0.5]
#visualizing the features w high negative correlation

f, axes = plt.subplots(nrows=2, ncols=4, figsize=(26,16))



f.suptitle('Features With High Negative Correlation', size=35)

sns.boxplot(x="Class", y="V3", data=subsample, ax=axes[0,0])

sns.boxplot(x="Class", y="V9", data=subsample, ax=axes[0,1])

sns.boxplot(x="Class", y="V10", data=subsample, ax=axes[0,2])

sns.boxplot(x="Class", y="V12", data=subsample, ax=axes[0,3])

sns.boxplot(x="Class", y="V14", data=subsample, ax=axes[1,0])

sns.boxplot(x="Class", y="V16", data=subsample, ax=axes[1,1])

sns.boxplot(x="Class", y="V17", data=subsample, ax=axes[1,2])

f.delaxes(axes[1,3])
#visualizing the features w high positive correlation

f, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,9))



f.suptitle('Features With High Positive Correlation', size=20)

sns.boxplot(x="Class", y="V4", data=subsample, ax=axes[0])

sns.boxplot(x="Class", y="V11", data=subsample, ax=axes[1])
#Only removing extreme outliers

Q1 = subsample.quantile(0.25)

Q3 = subsample.quantile(0.75)

IQR = Q3 - Q1



df2 = subsample[~((subsample < (Q1 - 2.5 * IQR)) |(subsample > (Q3 + 2.5 * IQR))).any(axis=1)]
len_after = len(df2)

len_before = len(subsample)

len_difference = len(subsample) - len(df2)

print('We reduced our data size from {} transactions by {} transactions to {} transactions.'.format(len_before, len_difference, len_after))
from sklearn.manifold import TSNE



X = df2.drop('Class', axis=1)

y = df2['Class']
#t-SNE

X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
# t-SNE scatter plot

import matplotlib.patches as mpatches



f, ax = plt.subplots(figsize=(24,16))





blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')

red_patch = mpatches.Patch(color='#AF0000', label='Fraud')



ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax.set_title('t-SNE', fontsize=14)



ax.grid(True)



ax.legend(handles=[blue_patch, red_patch])
def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn
# train test split

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)
X_train = X_train.values

X_validation = X_test.values

y_train = y_train.values

y_validation = y_test.values
print('X_shapes:\n', 'X_train:', 'X_validation:\n', X_train.shape, X_validation.shape, '\n')

print('Y_shapes:\n', 'Y_train:', 'Y_validation:\n', y_train.shape, y_validation.shape)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier




##Spot-Checking Algorithms



models = []



models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('SVM', SVC()))

models.append(('XGB', XGBClassifier()))

models.append(('RF', RandomForestClassifier()))



#testing models



results = []

names = []



for name, model in models:

    kfold = KFold(n_splits=10, random_state=42)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')

    results.append(cv_results)

    names.append(name)

    msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())

    print(msg)







#Compare Algorithms



fig = plt.figure(figsize=(12,10))

plt.title('Comparison of Classification Algorithms')

plt.xlabel('Algorithm')

plt.ylabel('ROC-AUC Score')

plt.boxplot(results)

ax = fig.add_subplot(111)

ax.set_xticklabels(names)

plt.show()



#visualizing RF

model = RandomForestClassifier(n_estimators=10)



# Train

model.fit(X_train, y_train)

# Extract single tree

estimator = model.estimators_[5]



from sklearn.tree import export_graphviz

# Export as dot file

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = X.columns.tolist(),

                class_names = ['0',' 1'],

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')
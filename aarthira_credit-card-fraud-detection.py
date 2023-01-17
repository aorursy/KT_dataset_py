# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline



from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns
#Reading the dataset

df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#To view the dataset

df.head()
df.describe()
#To view the dimension of the data

df.shape
#To see if there is any null values in the dataset

df.isnull().values.any()
#Number of frauds and regulars

fraud = len(df[df['Class']==1])



regular = len(df[df['Class']==0])
print('number of frauds are',fraud)

print('number of regulars are', regular)
#Visualising the class 

sns.countplot('Class', data = df)

plt.title('Transaction class distribution')



#Visualising all the col in data

df.hist(figsize=(30,30))

plt.show()
#To find the correlation between the data

f = plt.figure(figsize=(10, 10))

plt.matshow(df.corr(), f.number)

plt.show()
#Scaling the amount and time as they aren't scaled

from sklearn.preprocessing import StandardScaler, RobustScaler

standscale = StandardScaler()

df['scaled_amount'] = standscale.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = standscale.fit_transform(df['Time'].values.reshape(-1,1))
#Dropping the already existing amount and time

df.drop(['Time','Amount'], axis=1, inplace=True)
df.head()
#Inserting the scaled amount and time

scaled_amount = df['scaled_amount']

scaled_time = df['scaled_time']



df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

df.insert(0, 'scaled_amount', scaled_amount)

df.insert(1, 'scaled_time', scaled_time)

df.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
#Splitting the data into train and test by crossfolding

X = df.drop('Class', axis=1)

y = df['Class']

kf = KFold(n_splits = 5, random_state = None, shuffle = False)

for train_index, test_index in kf.split(X):

    print("Train:", train_index, "Test:", test_index)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    



    



#Equalling distributing data so that the data isn't biased and shuffling it for better prediction

df = df.sample(frac = 1)

fraud_df = df.loc[df['Class'] == 1]

regular_df = df.loc[df['Class'] == 0][:492]



distributed_df = pd.concat([fraud_df, regular_df])

new_df = distributed_df.sample(frac=1, random_state=42)

new_df.head()
#Visualising the equally distributed data

sns.countplot('Class', data=new_df)

plt.title('Equally Distributed Class')

plt.show()
#Setting up IQR 

v14 = new_df['V14'].loc[new_df['Class'] == 1].values

q25 = np.percentile(v14, 25)

q75 = np.percentile(v14, 75)

v14_iqr = q75 - q25

print('iqr of v14 is {}'.format(v14_iqr))



v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))



outliers = [x for x in v14 if x < v14_lower or x > v14_upper]

print(' V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V14 outliers:{}'.format(outliers))



#Dropping the data that lie out of IQR

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)

v12 = new_df['V12'].loc[new_df['Class'] == 1].values

q25 = np.percentile(v12, 25)

q75 = np.percentile(v12, 75)

v12_iqr = q75 - q25

print('iqr of v12 is {}'.format(v12_iqr))



v12_cut_off = v12_iqr * 1.5

v12_lower = q25 - v12_cut_off

v12_upper = q75 + v12_cut_off

print('Cut Off: {}'.format(v12_cut_off))

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))



outliers = [x for x in v12 if x < v12_lower or x > v12_upper]

print(' V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V12 outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)

v10 = new_df['V10'].loc[new_df['Class'] == 1].values

q25 = np.percentile(v10, 25)

q75 = np.percentile(v10, 75)

v10_iqr = q75 - q25

print('iqr of v10 is {}'.format(v10_iqr))



v10_cut_off = v10_iqr * 1.5

v10_lower = q25 - v12_cut_off

v10_upper = q75 + v12_cut_off

print('Cut Off: {}'.format(v10_cut_off))

print('V10 Lower: {}'.format(v10_lower))

print('V10 Upper: {}'.format(v10_upper))



outliers = [x for x in v10 if x < v10_lower or x > v10_upper]

print('V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V10 outliers:{}'.format(outliers))



new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

#Visualising the reformed data using box plot

f,(ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,5))



sns.boxplot(x="Class", y="V14", data=new_df,ax=ax1)

ax1.set_title("V14  outliers")



sns.boxplot(x = 'Class', y = 'V12', data = new_df, ax= ax2)

ax2.set_title("V12 outliers")



sns.boxplot(x = 'Class', y = 'V10', data = new_df, ax= ax3)

ax3.set_title("V10 outliers")

from sklearn.decomposition import PCA
#Reducing dimension using principal component analysis

X = new_df.drop('Class', axis=1)

y = new_df['Class']

pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
pca.shape
X = new_df.drop('Class', axis=1)

y = new_df['Class']
#Splitting the data into train and test

from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
#Implementing logistic regression classifier model

model = LogisticRegression()
#Fitting the training data with the classifier model

model.fit(X_train, y_train)
#Predicting the test set

pred = model.predict(X_test)
#Evaluating the model using cross val score

from sklearn.model_selection import cross_val_score

training_score = cross_val_score(model, X_train, y_train, cv=5)

print(training_score)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs = 4, n_estimators = 100, criterion = 'gini', verbose= False)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.svm import SVC
model2 = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',max_iter=-1)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, pred2))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = (11.0, 9.0)



#Supressing Warnings

import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')

train_df.head()
test_df = pd.read_csv('../input/test.csv')

test_df.head()
#Checking for empty or null records

null_data = train_df.isnull().sum()

print('Total null data in the training dataset is {}'.format(sum(null_data)))
sns.countplot(train_df['target'])
bought = train_df[train_df['target'] == 1].shape[0]



print('Training dataset has {} records out of which {}% have done transaction'.format(train_df.shape[0], 100*(bought/train_df.shape[0])))
#We will use 150000 records for training and the rest for testing out model



#For training

df_train_bal = train_df[:150000]



#For testing

df_test_bal = train_df[150000:]
#Creating a balanced dataset with comparable counts of both the outcome classes 0 and 1

df_train_bal_0 = df_train_bal[df_train_bal['target'] == 0]

df_train_bal_1 = df_train_bal[df_train_bal['target'] == 1]



#Selected 50000 non-transactions

df_train_bal_0 = df_train_bal_0.sample(50000)



#Final trainign dataset with balanced records, Randomize the dataset

df_train_bal = df_train_bal_0.append(df_train_bal_1).sample(frac=1)
df_train_bal['target'].value_counts()
corr = df_train_bal.corr()

sns.heatmap(corr)
corr['target'].sort_values(ascending=False)
#For training

X = df_train_bal.drop(['target','ID_code'], axis=1)

X.head()
y = df_train_bal.iloc[:,1:2]



print('Unique target values are :')

print(y['target'].value_counts())
#For testing

X_test = df_test_bal.drop(['target','ID_code'], axis=1)



y_test = df_test_bal.iloc[:,1:2]
from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression()

lr_model.fit(X, y)
lr_model.score(X_test, y_test)
#Cross Validation

from sklearn.model_selection import cross_val_score



# Perform 6-fold cross validation

scores = cross_val_score(lr_model, X_test, y_test, cv=6)



print('Cross-validated scores:', scores)
#Processing actual Testing data before submission

test_df_clean = test_df.drop('ID_code', axis=1)

test_df_clean.head()
#Applying PCA on both training and test dataset

from sklearn.decomposition import PCA



pca = PCA()

#Training data

X_train_pca = pca.fit_transform(X)
#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Satander Dataset Explained Variance')

plt.show()
#Now applyting PCA for 100 componenets

pca_best = PCA(n_components=100)

X_train_pca = pca_best.fit_transform(X)

X_test_pca = pca_best.transform(X_test)

X_actual_test = pca_best.transform(test_df_clean)



#Coveting numpy array back to Dataframe

X_train_pca = pd.DataFrame(X_train_pca)

X_test_pca= pd.DataFrame(X_test_pca)

X_actual_test = pd.DataFrame(X_actual_test)
#Applying best param to LR model

lr_model_best = LogisticRegression(C=10)

lr_model_best.fit(X_train_pca, y)
#Cross Validation of Best Model

from sklearn.model_selection import cross_val_score



# Perform 6-fold cross validation

scores = cross_val_score(lr_model_best, X_test_pca, y_test, cv=6)



print('Cross-validated scores:', scores)
test_pred = lr_model_best.predict(X_actual_test)
submission = pd.DataFrame({

        "ID_code": test_df["ID_code"],

        "target": test_pred

    })
submission.to_csv('submission.csv', index=False)
submission.head(10)
submission['target'].value_counts()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np 

import os





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")



print(train.shape)

print(test.shape)

train.head()
X=train.iloc[:,1:24].values

y=train.iloc[:,24].values

X1=test.iloc[:,1:24].values
temp_df = train.isnull().sum().reset_index()

temp_df['Percentage'] = (temp_df[0]/len(train))*100

temp_df.columns = ['Column Name', 'Number of null values', 'Null values in percentage']

print(f"The length of dataset is \t {len(train)}")

temp_df

from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()

X[:,3] = labelencoder_X_1.fit_transform(X[:,3])

X[:,4] = labelencoder_X_1.fit_transform(X[:,4])

X[:,5] = labelencoder_X_1.fit_transform(X[:,5])

X[:,6] = labelencoder_X_1.fit_transform(X[:,6])

X[:,7] = labelencoder_X_1.fit_transform(X[:,7])

X[:,8] = labelencoder_X_1.fit_transform(X[:,8])

X[:,9] = labelencoder_X_1.fit_transform(X[:,9])

X[:,10] = labelencoder_X_1.fit_transform(X[:,10])

X[:,11] = labelencoder_X_1.fit_transform(X[:,11])

X[:,12] = labelencoder_X_1.fit_transform(X[:,12])

X[:,13] = labelencoder_X_1.fit_transform(X[:,13])

X[:,14] = labelencoder_X_1.fit_transform(X[:,14])

X[:,16] = labelencoder_X_1.fit_transform(X[:,16])

X[:,17] = labelencoder_X_1.fit_transform(X[:,17])

X[:,18] = labelencoder_X_1.fit_transform(X[:,18])

X[:,19] = labelencoder_X_1.fit_transform(X[:,19])

X[:,20] = labelencoder_X_1.fit_transform(X[:,20])







y = labelencoder_X_1.fit_transform(y)



X1[:,3] = labelencoder_X_1.fit_transform(X1[:,3])

X1[:,4] = labelencoder_X_1.fit_transform(X1[:,4])

X1[:,5] = labelencoder_X_1.fit_transform(X1[:,5])

X1[:,6] = labelencoder_X_1.fit_transform(X1[:,6])

X1[:,7] = labelencoder_X_1.fit_transform(X1[:,7])



X1[:,8] = labelencoder_X_1.fit_transform(X1[:,8])

X1[:,9] = labelencoder_X_1.fit_transform(X1[:,9])

X1[:,10] = labelencoder_X_1.fit_transform(X1[:,10])

X1[:,11] = labelencoder_X_1.fit_transform(X1[:,11])

X1[:,12] = labelencoder_X_1.fit_transform(X1[:,12])

X1[:,13] = labelencoder_X_1.fit_transform(X1[:,13])

X1[:,14] = labelencoder_X_1.fit_transform(X1[:,14])

X1[:,16] = labelencoder_X_1.fit_transform(X1[:,16])

X1[:,17] = labelencoder_X_1.fit_transform(X1[:,17])

X1[:,18] = labelencoder_X_1.fit_transform(X1[:,18])

X1[:,19] = labelencoder_X_1.fit_transform(X1[:,19])

X1[:,20] = labelencoder_X_1.fit_transform(X1[:,20])



# Let us Import the Important Libraries  to train our Model for Machine Learning 

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split  

from sklearn.model_selection import cross_val_score   

from sklearn.preprocessing import Imputer  

from sklearn.preprocessing import StandardScaler   #
from imblearn.over_sampling import SMOTE, ADASYN

from collections import Counter

X_resampled, y_resampled = SMOTE().fit_resample(X, y)

#print(sorted(Counter(y_resampled).items()))



X_resampled.shape,y_resampled.shape
X=pd.DataFrame(X)

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

bestfeatures = SelectKBest(score_func=f_classif, k=10)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
X=pd.DataFrame(X)

X1=pd.DataFrame(X1)

X4=X.iloc[:,[1]+[4]+[5]+[9]+[15]+[16]+[18]+[19]+[20]+[22]].values

X5=X1.iloc[:,[1]+[4]+[5]+[9]+[15]+[16]+[18]+[19]+[20]+[22]].values
sc_X=StandardScaler()

x_train=sc_X.fit_transform(X4)



x_test = sc_X.fit_transform(X5)
from sklearn.decomposition import PCA

pca = PCA(n_components=None)

x_train = pca.fit_transform(x_train)



x_test = pca.fit_transform(x_test)

explained_variance = pca.explained_variance_ratio_

explained_variance

pca = PCA(n_components=5)

x_train = pca.fit_transform(x_train)



x_test = pca.fit_transform(x_test)
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

model = GaussianNB()



# Train the model using the training sets

model.fit(x_train, y)

accuracy = cross_val_score(estimator=model, X=x_train, y=y, cv=25)

print(f"The accuracy of the Gaussian Naive Bayes Model is \t {accuracy.mean()}") 

print(f"The deviation in the accuracy is \t {accuracy.std()}")

pred = model.predict_proba(x_train)[:, 1]

score = roc_auc_score(y, pred)



print("score: ", score)
submission["id"] = test_id

submission["target"] = model.predict_proba(x_test)[:, 1]




submission.to_csv('cat.csv', index=False)
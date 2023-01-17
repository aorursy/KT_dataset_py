import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_df=pd.read_csv('emotions.csv')

train_df.head()
train_df.describe
# Data contains variables: fft(Fast fourier Tranform), correlate, entropy, logm, eigen, covmat, min_q, max_q

# moments, mean, stddev

# All are numerical type variable (float)

#output is Label : Postive/Negative/Neutral

#class distribution from column label (Output is label)

plt.figure(figsize=(12,5))

sns.countplot(x=train_df.label, color='red')

plt.title('Brain Wave Data', fontsize=14)

plt.xlabel('Class Label', fontsize=14)

plt.ylabel('Class count', fontsize=14)
#Null data

train_df.isnull().sum()
label_df=train_df['label']

train_df.drop('label', axis=1, inplace=True)

train_df.head()
#Using cross validation (10 fold in this case)

#Pipeline based approach

#No of dimensions are high. Hence we will start with random forest classifier which works well on high-dimension data

#Since its probablity based classifier, no pre-processing stages like scaling or noise removal are required

#not affected by scale factors
%%time

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, train_test_split





model_randomForest=Pipeline(steps=[('random_forest', RandomForestClassifier())])

scores=cross_val_score(model_randomForest, train_df, label_df, cv=10, scoring='accuracy')

print('Accuracy for Random Forest = ', scores.mean())
%%time



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, train_test_split





model_logisticRegression=Pipeline(steps=[('scalar', StandardScaler()),

                                         ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])

scores=cross_val_score(model_logisticRegression, train_df, label_df, cv=10, scoring='accuracy')

print('Accuracy for Logistic Regression= ', scores.mean())
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()

scaled_df=scaler.fit_transform(train_df)

pca=PCA(n_components=20)

pca_vectors=pca.fit_transform(scaled_df)

for index, var in enumerate(pca.explained_variance_ratio_):

    print("Explained variance ratio by Principal Component ", (index+1) ," : " , var)
#Using mathematical mapping 2549 variables mapped to 20 variables

#Of 2549 variables, 10 are of most importance
plt.figure(figsize=(25,8))

sns.scatterplot(x=pca_vectors[:,0], y=pca_vectors[:,1],

               hue=label_df)

plt.title('PC V/s Class', fontsize=14)

plt.xlabel('PC 1', fontsize=14)

plt.ylabel('PC 2', fontsize=14)

plt.xticks(rotation='vertical');
# it can be seen that if we use Logistic regression the first classifier will seperate NEUTRAL class from other two

# and the second classifier will seperate NEGATIVE and POSITIVE

# Applying Logistic regression model on 2 main PCs
%%time



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, train_test_split



model_lg_pca=Pipeline(steps=[('scaler', StandardScaler()),

                            ('pca', PCA(n_components=2)),

                            ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga',max_iter=200 ))])

scores=cross_val_score(model_lg_pca, train_df, label_df, cv=10, scoring='accuracy')

print('Accuracy for Logistic Regression :', scores.mean())
# Taking 10 PCs and running the model 
%%time

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, train_test_split



model_lg_pca_10=Pipeline(steps=[('Scaler', StandardScaler()),

                               ('pca', PCA(n_components=10)),

                               ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])



scores=cross_val_score(model_lg_pca_10, train_df, label_df, cv=10, scoring='accuracy')

print('Accuracy for Logistic Regressionwith 10 PCs :', scores.mean())
%%time



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, train_test_split

model_mlp=Pipeline(steps=[('scaler', StandardScaler()),

                         ('mlp_classifier', MLPClassifier(hidden_layer_sizes=(1275, 637)))])

scores=cross_val_score(model_mlp, train_df, label_df, cv=10, scoring='accuracy')

print('Accuracy for ANN Classifier: ', scores.mean())
# General convention is to start with 50% of the data size for the first hidden layer 

# and 50% of previous size in subsequent layer 

# Number of hidden layers can be taken as a hyper-parameter and can be used to tune for better accuracy

# Hidden layers in this ccase is 2

# Or number of hidden neurons =  average of the input and output layers summed together.

# The upper bound on the number of hidden neurons that won't result in over-fitting is: 𝑁ℎ=𝑁𝑠/(𝛼∗(𝑁𝑖+𝑁𝑜))

# 𝑁𝑖= number of input neurons.

# 𝑁𝑜= number of output neurons.

# 𝑁𝑠= number of samples in training data set.

# α= an arbitrary scaling factor usually 2-10.
%%time



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score, train_test_split



model_SVM=Pipeline(steps=[('Scaler', StandardScaler()),

                         ('svm', LinearSVC())])

scores=cross_val_score(model_SVM, train_df, label_df, cv=10, scoring='accuracy')

print('Accuracy for Linear SVM :', scores.mean())

%%time

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score, train_test_split

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import xgboost as xgb



model_xgb=Pipeline(steps=

                   [('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])

scores=cross_val_score(model_xgb, train_df, label_df, cv=10, scoring='accuracy')

print('Accuracy for Extreme Gradient Boosting is :', scores.mean())
# xgboost performs well in GPU Machines

# os has been imported due to dead kernel problem

# CONCLUSIONS

# 1. For Accuracy XGBoost is most favourable

# 2. Random FOrest is a perfect choice if "time taken" is also considered

# 3. Simple classifiers like Logistic regression can give better accuracy with poper feauture engineering

# 4. other classifiers don't need much feauture engineering effort
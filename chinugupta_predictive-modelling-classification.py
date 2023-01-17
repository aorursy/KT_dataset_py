# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas_profiling

import sys

import math

import numpy.random as nr

import scipy.stats as ss

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn.preprocessing as skpe

import sklearn.model_selection as ms

import sklearn.metrics as sklm

import sklearn.linear_model as lm

from sklearn import tree

from sklearn import neighbors

from sklearn import ensemble

import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std
path="../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv"

df=pd.read_csv(path)

df.head()
df.info()
df.describe()
# A brief overview and detailed EDA of this dataset

pandas_profiling.ProfileReport(df)
# Histogram for univariate analysis

fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in df.items():

    sns.distplot(v,kde=False,rug=True,ax=axs[index]) # rug is used to see the frequency density on the x-scale

    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# Box Plots to detect outliers

fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in df.items():

    sns.boxplot(y=k, data=df, ax=axs[index])

    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# Checking correlation bw different variables

plt.figure(figsize=(18,18))

sns.heatmap(df.corr(),vmax=.7,cbar=True,annot=True)  
fig, (axis1,axis2) = plt.subplots(2,1,figsize=(10,8))



sns.barplot(x='quality', y='fixed acidity', data=df, ax=axis1)

sns.barplot(x='quality', y='volatile acidity', data=df, ax=axis2)
fig, (axis1,axis2) = plt.subplots(2,1,figsize=(10,8))



sns.barplot(x='quality', y='citric acid', data=df, ax=axis1)

sns.barplot(x='quality', y='residual sugar', data=df, ax=axis2)
fig, (axis1,axis2) = plt.subplots(2,1,figsize=(10,8))



sns.barplot(x='quality', y='chlorides', data=df, ax=axis1)

sns.barplot(x='quality', y='total sulfur dioxide', data=df, ax=axis2)
fig, (axis1,axis2) = plt.subplots(2,1,figsize=(10,8))



sns.barplot(x='quality', y='free sulfur dioxide', data=df, ax=axis1)

sns.barplot(x='quality', y='total sulfur dioxide' , data=df, ax=axis2)
fig,(axis1,axis2) = plt.subplots(2,1,figsize=(10,8))



sns.barplot(x='quality', y='pH', data=df, ax=axis1)

sns.barplot(x='quality', y='sulphates', data=df, ax=axis2)
sns.barplot(x='quality', y='alcohol' , data=df)
# Making bins in order to classify these bins as good or bad (binary classificaion)

bins = (2, 6, 8)

groups = ['Bad', 'Good']

df['quality'] = pd.cut(df['quality'], bins = bins, labels = groups)





# Encoding these binary gruoups into 0, 1, 2, etc.(categorical to numerical, because most of the machine learning models are not able to interpret categorical varaibles)

# For this purpose we use label encoder

le = skpe.LabelEncoder()





# Fitting and transforming these features

df['quality'] = le.fit_transform(df['quality'])



sns.countplot(df['quality'])
# Seperating dependent and independent variables 

X = df.drop('quality', axis = 1)

y = df['quality']
# Splitting the data into train and test set

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.25, random_state = 111,stratify=y)
# Feature scaling

scaler=skpe.StandardScaler()



X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
clf_rfc = ensemble.RandomForestClassifier(n_estimators=300)

clf_rfc.fit(X_train, y_train)

pred_rfc = clf_rfc.predict(X_test)



# Model performance

print(sklm.classification_report(y_test, pred_rfc))
def print_metrics(labels, scores):

    metrics = sklm.precision_recall_fscore_support(labels, scores)

    conf = sklm.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])

    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])

    print('')

    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))

    print(' ')

    print('           Positive      Negative')

    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])

    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])

    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])

    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])





    

print_metrics(y_test, pred_rfc) 
log_reg=lm.LogisticRegression(penalty='l2')

log_reg.fit(X_train,y_train)

probabilities = log_reg.predict_proba(X_test)  # Predicting probablities of the quality of wine
# Function to classify the wine as good or bad based on the probability (if it is less than 0.5, then it is bad otherwise it is good)

def score_model(probs, threshold):

    return np.array([1 if x > threshold else 0 for x in probs[:,1]])

scores = score_model(probabilities, 0.5)

print(np.array(scores[:15]))

print(np.array(y_test[:15]))
def print_metrics(labels, scores):

    metrics = sklm.precision_recall_fscore_support(labels, scores)

    conf = sklm.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])

    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])

    print('')

    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))

    print(' ')

    print('           Positive      Negative')

    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])

    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])

    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])

    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])





    

print_metrics(y_test, scores)   
# Plotting ROC-AUC curve

# This curve is used measure the performance of the model by area under the curve.

def plot_auc(labels, probs):

    ## Compute the false positive rate, true positive rate

    ## and threshold along with the AUC

    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])

    auc = sklm.auc(fpr, tpr)

    

    ## Plot the result

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

plot_auc(y_test, probabilities) 
# We can also use GridSearchCV but it takes a lot of time and searches through all hyperparameters even though they are not necessarily required.

# Hence, RandomizedSearchCV is possibly the best method you could use

param_dist={'n_estimators':[100,200,300,400,500,600],'criterion':['gini','entropy'],'max_depth':ss.randint(1,15),'max_features':ss.randint(1,9),'min_samples_leaf':ss.randint(1,9)}

clf=ensemble.RandomForestClassifier()

clf_cv=ms.RandomizedSearchCV(clf,param_distributions=param_dist,cv=5)

clf_cv.fit(X_train,y_train)

print("Tuned Random Forest Parameters: {}".format(clf_cv.best_params_)) 

print("Best score is {}".format(clf_cv.best_score_)) 
# Putting these hyperparameters into our model

clf=ensemble.RandomForestClassifier(criterion='entropy',max_depth=12,max_features=2,min_samples_leaf=5,n_estimators=300)

clf.fit(X_train,y_train)
# Now let's see how this model works on an unseen data.For this we are using K-Fold CrossValidation

rfc_cv = ms.cross_val_score(estimator = clf , X = X_train, y = y_train, cv = 10)

rfc_cv.mean()
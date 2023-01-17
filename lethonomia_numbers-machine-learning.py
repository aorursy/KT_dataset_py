import numpy as np

import pandas as pd

#import tensorflow as tf

random_seed = 2

#tf.set_random_seed(random_seed)

np.random.seed(random_seed) 

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier  



random_state = 3

import itertools

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set()
y_values = [85.114,86.485,85.285,85.285,85.228,86.100,90.228,90.514,90.428,90.428,90.428,

            92.385,91.957,66.357,64.242,92.500]

fig = plt.figure(figsize=(14,5), dpi=80)

_ =  sns.lineplot(x=range(len(y_values)),y=y_values,linestyle='--', marker='o', color='b')

plt.xlabel('Submission # (Not including errors)')

plt.ylabel('Percent Accuracy')

arrowprops=dict(color='black',headwidth=2,headlength=2,width=.5)

plt.annotate(xy=(0,85.114),xytext=(0,87),arrowprops=arrowprops,

             s="Plain Run with Decision Tree")

plt.annotate(xy=(6,90.228),xytext=(6,92),arrowprops=arrowprops,

             s="Manual folds w/ RandomForest")

plt.annotate(xy=(11,92.385),xytext=(10,93.5),arrowprops=arrowprops,

             s="Added KFold w/ hard VotingClassifier")

plt.annotate(xy=(13,66.357),xytext=(11,68),arrowprops=arrowprops,

             s="StandardScaler.\nDo not use.")

plt.annotate(xy=(15,92.500),xytext=(15,93.5),arrowprops=arrowprops,

             s="92.5% (Variance)")
#'../input/train.csv'- use in Kaggle as the csv file

train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
#Save solution to train set as y

y = train['label'].values

print(y[0:5])

#Drop solutions from train set

train.drop('label',axis=1,inplace=True)

print(train[0:5])
print(train.shape)

#Reshape arrays for image display

df_test = np.array(test).reshape((-1,28,28))

df_train = np.array(train).reshape((-1,28,28))

print(df_train)
fig = plt.figure(figsize=(20,5))

#36 total images; 3x12

for i in range(36):

    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(df_train[i]),cmap = "bone_r") #set to black and white images

    ax.set_title(y[i]) #What the number displayed is

plt.show()
#Reshape train, test for machine learning algorithms

df_train = df_train.reshape(len(df_train),-1)

df_test = df_test.reshape(len(df_test),-1)

print(df_train)

df_train.shape
#created var variables to differentiate steps

var_train = train.copy()

var_y = y.copy() 

var_test = test.copy()
variance = np.var(var_train,axis=0)

#Find the variance of all the pixels
print(variance[50:80])
variance.shape
fig = plt.figure(figsize=(14,5), dpi=80)

ax1 = fig.add_subplot(111)

_ = sns.barplot(y=variance,x=list(range(len(variance))))

plt.xlabel("Pixel Number")

plt.ylabel("Variance")

plt.title("Variance of Pixels")
fig = plt.figure(figsize=(14,5), dpi=80)

ax1 = fig.add_subplot(111)

_ = sns.barplot(y=variance[50:101],x=list(range(50,101)))

plt.xlabel("Pixel Number 50-100")

plt.ylabel("Variance")

plt.title("Variance of Pixels 50-100")
_ = plt.hist(variance,bins=30)

plt.xlabel("Variance")

plt.ylabel("Count")

plt.title("Histogram of Variance")
bins = pd.cut(variance,30).unique()

print(bins)

#30 bins done by evenly spaced increments (in range)
#Drops pixels of low variance

for i in range(0,len(variance)):

    if variance[i] < 432.052:

        var_train.drop("pixel"+ str(i),axis=1,inplace=True)

        var_test.drop("pixel"+ str(i),axis=1,inplace=True)

var_train.head()
#convert variable test/train to arrays

var_test = np.array(var_test)

var_train = np.array(var_train)

print(var_train.shape)

print(var_train)
#cross validation method initially made in Housing Loans

kfolds = KFold(n_splits=5, shuffle=True, random_state=random_state)



def model_fitter(model, X=var_train,y=var_y,test=var_test):

    cv = cross_validate(model, X, y,cv=kfolds,scoring='accuracy',

                        return_train_score=False, return_estimator=True)

    print("Best Score is: %s, located at %s"%(max(cv['test_score']),list(cv['test_score']).index(max(cv['test_score']))))

    best_rfc = cv['estimator'][list(cv['test_score']).index(max(cv['test_score']))]

    print(best_rfc)

    return(best_rfc)
# Gaussian Naive Bayes

gaussian = GaussianNB()

f_gaussian = model_fitter(gaussian)
# Decision Tree

decision_tree = DecisionTreeClassifier(random_state=random_state)

f_decision_tree = model_fitter(decision_tree)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100,random_state=random_state)

f_random_forest = model_fitter(random_forest)
VotingPredictor = VotingClassifier(estimators=[('Gaussian', f_gaussian),('random_forest', f_random_forest),('decision_tree', f_decision_tree)], voting='hard', n_jobs=4)

f_VotingPredictor = model_fitter(VotingPredictor)

VotingPredictor_predictions = f_VotingPredictor.predict(var_test)

print(VotingPredictor_predictions)
ans = pd.DataFrame()

ans['Label'] = VotingPredictor_predictions

ans.index.name='ImageId'

ans = ans.reset_index()

ans['ImageId'] = ans['ImageId'] + 1

ans.to_csv('ans.csv',index=False)

ans.head()

#0.90514
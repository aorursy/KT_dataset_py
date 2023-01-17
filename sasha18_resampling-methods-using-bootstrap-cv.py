#Import numerical libraries

import numpy as np

from numpy import array

import pandas as pd



#Import graphical plotting libraries

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Import resampling and modeling algorithms

from sklearn.utils import resample # for Bootstrap sampling

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



#KFold CV

from sklearn.model_selection import KFold, LeaveOneOut

from sklearn.model_selection import cross_val_score



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/handson-pima/Hands on Exercise Feature Engineering_ pima-indians-diabetes (1).csv')

data.head()



values = data.values
#Lets configure Bootstrap



n_iterations = 10  #No. of bootstrap samples to be repeated (created)

n_size = int(len(data) * 0.50) #Size of sample, picking only 50% of the given data in every bootstrap sample
#Lets run Bootstrap

stats = list()

for i in range(n_iterations):



    #prepare train & test sets

    train = resample(values, n_samples = n_size) #Sampling with replacement..whichever is not used in training data will be used in test data

    test = np.array([x for x in values if x.tolist() not in train.tolist()]) #picking rest of the data not considered in training sample

    

    #fit model

    model = DecisionTreeClassifier()

    model.fit(train[:,:-1], train[:,-1]) #model.fit(X_train,y_train) i.e model.fit(train set, train label as it is a classifier)

    

    #evaluate model

    predictions = model.predict(test[:,:-1]) #model.predict(X_test)

    score = accuracy_score(test[:,-1], predictions) #accuracy_score(y_test, y_pred)

    #caution, overall accuracy score can mislead when classes are imbalanced

    

    print(score)

    stats.append(score)
#Lets plot the scores to better understand this visually



plt.hist(stats)

plt.figure(figsize = (10,5))
#Lets find Confidence intervals



a = 0.95 # for 95% confidence

p = ((1.0 - a)/2.0) * 100 #tail regions on right and left .25 on each side indicated by P value (border)

                          # 1.0 is total area of this curve, 2.0 is actually .025 thats the space we would want to be 

                            #left on either side

lower = max(0.0, np.percentile(stats,p))



p = (a + ((1.0 - a)/ 2.0)) * 100 #p is limits

upper = min(1.0, np.percentile(stats,p))

print('%.1f confidence interval %.1f%% and %.1f%%' %(a*100, lower*100, upper*100))
#Create separate arrays such that only values are considered as X, y

values = data.values

X = values[:,0:8]

y = values[:,8]



#Split the data into train,test set

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.50, random_state = 1)
#Lets configure Cross Validation

#default value of n_splits = 10

kfold = KFold(n_splits = 50, random_state = 7)

model = LogisticRegression()

results = cross_val_score(model,X,y,cv = kfold)
print(results)
#What's the accuracy of this model using KFold CV



print('Accuracy:  %.3f%% (%.3f%%)' % (results.mean()*100.0, results.std()*100.0))
# scikit-learn k-fold cross-validation



# data sample

data = array([10,20,30,40,50,60,70,80,90,100])

# prepare cross validation

loocv = LeaveOneOut()

# enumerate splits

for train, test in loocv.split(data):

    print('train: %s, test: %s' % (data[train], data[test]))
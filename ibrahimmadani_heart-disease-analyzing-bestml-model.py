# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from time import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')
df.shape
df.head()
df.target.unique()
print('cp:',df.cp.unique())

print('fbs:',df.fbs.unique())

print('restecg:',df.restecg.unique())

print('exang:',df.exang.unique())

print('slope:',df.slope.unique())

print('thal:',df.thal.unique())
d1 = pd.get_dummies(df['cp'], prefix='cp')

d2 = pd.get_dummies(df['restecg'], prefix='restecg')

d3 = pd.get_dummies(df['slope'], prefix='slope')

d4 = pd.get_dummies(df['thal'], prefix='thal')
df_new = pd.concat([df,d1,d2,d3,d4], axis=1)
df_new.head()
df_new.drop(['cp','restecg','slope','thal'],axis=1, inplace=True)
X = df_new.drop(['target'], axis=1)

y = df_new['target']
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score
scale = StandardScaler()

X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=33)
def Classifiers():

    classifiers = []

    

    from sklearn.linear_model import LogisticRegression

    classifiers.append(LogisticRegression()) #no variations

                       

    from sklearn.tree import DecisionTreeClassifier

    classifiers.append(DecisionTreeClassifier())  #no variations

    

    from sklearn.neighbors import KNeighborsClassifier

    for n in range(2,10):

        classifiers.append(KNeighborsClassifier(n_neighbors=n))

    #classifiers.append(KNeighborsClassifier()) #n_neighbors

                       

    from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier

    classifiers.append(AdaBoostClassifier())

    classifiers.append(BaggingClassifier()) 

    classifiers.append(GradientBoostingClassifier())

    #classifiers.append(RandomForestClassifier())  #n_estimators

    for n in range(10,160):

        classifiers.append(RandomForestClassifier(n_estimators=n))

    

    from sklearn.svm import LinearSVC, SVC

    classifiers.append(SVC())

    classifiers.append(LinearSVC())

    

    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

    classifiers.append(BernoulliNB())

    classifiers.append(GaussianNB())

    #classifiers.append(MultinomialNB())

    

    return classifiers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def model_run(model, X_train, X_test, y_train, y_test):

    

    #Training

    start_time = time()

    print('Training the data on : ',model.__class__.__name__)

    model = model.fit(X_train, y_train)

    end_time = time()

    #print('Total time taken : ', end_time - start_time)

    

    #predictions



    predictions_train = model.predict(X_train[:100])

    #print('Training score is :',accuracy_score(y_train[:100],predictions_train))

    pred_train_score = accuracy_score(y_train[:100],predictions_train)

    predicions_test = model.predict(X_test)

    print('Test Score is :',accuracy_score(y_test,predicions_test))

    pred_test_score = accuracy_score(y_test,predicions_test)

    return pred_train_score, pred_test_score
import warnings

warnings.filterwarnings('ignore')
train_set_scores = {}

test_set_scores = {}

for i in range(10,160):

    #print('No of Estimators are:',i)

    pred_train, pred_test = model_run(RandomForestClassifier(n_estimators=i), X_train, X_test, y_train, y_test)

    #print(pred_train)

    train_set_scores[i]= pred_train

    test_set_scores[i]= pred_test

#import operator

print("Optimum number of estimators are ",max(train_set_scores.items(), key=operator.itemgetter(1))[0],' with train score : ',max(train_set_scores.values()) )

print("Optimum number of estimators are ",max(test_set_scores.items(), key=operator.itemgetter(1))[0],' with test score : ',max(test_set_scores.values()) )

    
model_run(RandomForestClassifier(n_estimators=18), X_train, X_test, y_train, y_test)
def main():

    import operator

    classifiers = Classifiers()

    test_set_results = {}

    #best_randomforest_model = None

    RandomForest_testScore = 0

    KNeighbors_testScore = 0    

    for model in classifiers:



        if 'RandomForest' in str(model):

            pred_train, pred_test = model_run(model, X_train, X_test, y_train, y_test)

            if pred_test>RandomForest_testScore:

                print(pred_test,'check 1')

                RandomForest_testScore = pred_test

                model_name = str(model).split('(')[0]

                test_set_results[model_name] = RandomForest_testScore

        elif 'KNeighbors' in str(model):

            pred_train, pred_test = model_run(model, X_train, X_test, y_train, y_test)

            if pred_test>KNeighbors_testScore:

                KNeighbors_testScore = pred_test

                model_name = str(model).split('(')[0]

                test_set_results[model_name] = KNeighbors_testScore            

        else:

            pred_train, pred_test = model_run(model, X_train, X_test, y_train, y_test)

            model_name = str(model).split('(')[0]

            test_set_results[model_name] = pred_test

        #print('Test Score is: ',pred_test)

    print('Best Model for the given data is :',max(test_set_results.items(), key=operator.itemgetter(1))[0],' with test score : ',max(test_set_results.values()) )

    return test_set_results
results = main()
import matplotlib.pyplot as plt



clrs = ['grey' if (x < max(results.values())) else 'red' for x in results.values() ]

plt.bar(range(len(results)), list(results.values()), align='center', color = clrs)

plt.xticks(range(len(results)), list(results.keys()), rotation = 'vertical')



plt.show()
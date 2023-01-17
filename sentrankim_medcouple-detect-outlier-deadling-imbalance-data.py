!pip install scipy
!pip install statistics
!pip install weightedstats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from statsmodels.stats.stattools import medcouple
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import math
from statistics import median
from scipy.stats import skew
import weightedstats as ws

dataset = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
label = dataset['Class']
origin_data = dataset # use detection outlier
dataset.describe()
dataset['Time'].describe()
dataset['Amount'].describe()
dataset.fillna(np.nan)
dataset.isnull().sum().max()
graph_time = sns.distplot(dataset['Time'], color="m", label="Skewness : %.2f"%(dataset['Time'].skew()))
graph_time = graph_time.legend(loc="best")
g = sns.distplot(dataset['Amount'], color="m", label="Skewness : %.2f"%(dataset['Amount'].skew()))
g = g.legend(loc="best")
def log_transform(feature):
    dataset[feature] = dataset[feature].map(lambda x:np.log(x) if x>0 else 0 )
log_transform('Time')
graph_time = sns.distplot(dataset['Time'], color="m", label="Skewness : %.2f"%(dataset['Time'].skew()))
graph_time = graph_time.legend(loc="best")
log_transform('Amount')
g = sns.distplot(dataset['Amount'], color="m", label="Skewness : %.2f"%(dataset['Amount'].skew()))
g = g.legend(loc="best")
rforest_checker = RandomForestClassifier(random_state = 0)
rforest_checker.fit(dataset, label)
importances_df = pd.DataFrame(rforest_checker.feature_importances_, columns=['Feature_Importance'],
                              index=dataset.columns)
importances_df.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
print(importances_df)
"""
    The purpose of this class is to find MC measure. 
    You might reference my Github: https://github.com/tks1998/statistical-function-and-algorithm-ML-/blob/master/medcople.py
    Complexity : O(nlogn)
    The code is the implementation of the formula on wiki. But it is still having some issues, since there are errors arised from finite-precision floating point arithmetic (when compare two float number). 
    I tested 1000 tests and compare function Medcouple with statsmodels, total errors for 1000 tests ~2.2, That means that each test I deviate by about 0.022.
    Function medcouple from Statsmodels is implemented with complexity O(n^2). If applying the problems (n ~ 285000), I do not have no enough RAM to run the test. ***So that I must implement the function???***.   
"""
class Med_couple:
    
    def __init__(self,data):
        self.data = np.sort(data,axis = None)[::-1] # sorted decreasing  
        self.med = np.median(self.data)
        self.scale = 2*np.amax(np.absolute(self.data))
        self.Zplus = [(x-self.med)/self.scale for x in self.data if x>=self.med]
        self.Zminus = [(x-self.med)/self.scale for x in self.data if x<=self.med]
        self.p = len(self.Zplus)
        self.q = len(self.Zminus)
    
    def H(self,i,j):
        a = self.Zplus[i]
        b = self.Zminus[j]

        if a==b:
            return np.sign(self.p - 1 - i - j)
        else:
            return (a+b)/(a-b)

    def greater_h(self,u):

        P = [0]*self.p

        j = 0

        for i in range(self.p-1,-1,-1):
            while j < self.q and self.H(i,j)>u:
                j+=1
            P[i]=j-1
        return P

    def less_h(self,u):

        Q = [0]*self.p

        j = self.q - 1

        for i in range(self.p):
            while j>=0 and self.H(i,j) < u:
                j=j-1
            Q[i]=j+1
        
        return Q
    #Kth pair algorithm (Johnson & Mizoguchi)
    def kth_pair_algorithm(self):
        L = [0]*self.p
        R = [self.q-1]*self.p

        Ltotal = 0

        Rtotal = self.p*self.q

        medcouple_index = math.floor(Rtotal / 2)

        while Rtotal - Ltotal > self.p:

            middle_idx = [i for i in range(self.p) if L[i]<=R[i]]
            row_medians = [self.H(i,math.floor((L[i]+R[i])/2)) for i in middle_idx]

            weight = [R[i]-L[i] + 1 for i in middle_idx]

            WM = ws.weighted_median(row_medians,weights = weight)
            
            P = self.greater_h(WM)

            Q = self.less_h(WM)

            Ptotal = np.sum(P)+len(P) 
            Qtotal = np.sum(Q)

            if medcouple_index <= Ptotal-1:
                R = P
                Rtotal = Ptotal
            else:
                if medcouple_index > Qtotal - 1:
                    L = Q
                    Ltotal = Qtotal
                else:
                    return WM
        remaining = np.array([])
       
        for i in range(self.p):
            for j in range(L[i],R[i]+1):
                remaining = np.append(remaining,self.H(i,j))

        find_index = medcouple_index-Ltotal

        k_minimum_element = remaining[np.argpartition(remaining,find_index)] # K-element algothrm  
    
        return k_minimum_element[find_index]
       
"""
    Reference from gits.github of joseph-allen : https://gist.github.com/joseph-allen/14d72af86689c99e1e225e5771ce1600 
"""
def detection_outlier(n,df):
    
    outlier_indices = []
    
    for col in df.columns:
        Q1 = np.percentile(df[col],25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR 
        medcouple = Med_couple(np.array(df[col])).kth_pair_algorithm()
        if (medcouple >=0):
            outlier_list_col = df[(df[col] < Q1 - outlier_step*math.exp(-3.5*medcouple)) | (df[col] > Q3 + outlier_step*math.exp(4*medcouple) )].index
        else:
            outlier_list_col = df[(df[col] < Q1 - outlier_step*math.exp(-4*medcouple)) | (df[col] > Q3 + outlier_step*math.exp(3.5*medcouple) )].index
        outlier_indices.extend(outlier_list_col)     
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers   
index_outlier = detection_outlier(20,origin_data)
print(index_outlier)
"""
    Removing outlier from dataset using medcouple method
"""
x_train = dataset
x_train = x_train.drop(index_outlier, axis = 0).reset_index(drop=True)
y_train = x_train['Class']
x_train = x_train.drop(columns = ['Class'])
"""
    reference from kaggle: https://www.kaggle.com/lane203j/methods-and-common-mistakes-for-evaluating-models
"""
class CreditCard:
    def __init__(self,clf,data,label,k_fold = 10):
        self.clf = clf
        self.data = data
        self.label = label
        self.k_fold = k_fold
    
    def ROC_score(self,x, y):
        precisions, recalls,_ = precision_recall_curve(y, self.clf.predict_proba(x)[:,1], pos_label=1)
        return auc(recalls, precisions)
    
    def calculate_score_model(self,num_random_state = None, is_shuffle = False):
        
        skf = StratifiedKFold(n_splits=self.k_fold, random_state=num_random_state, shuffle=is_shuffle)
        
        sum = 0
        
        for train_idx, test_idx in skf.split(self.data,self.label):
            
            x_train = self.data.loc[train_idx]
            y_train = self.label.loc[train_idx]
            
            x_test = self.data.loc[test_idx]
            y_test = self.label.loc[test_idx]
            self.clf.fit(x_train, y_train)
            sum+=self.ROC_score(x_test,y_test)
            
        return sum/self.k_fold    
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights
clf1 = LogisticRegression(max_iter = 1000000,class_weight = {0:0.50070004, 1: 357.62311558})
logstic = CreditCard(clf1,x_train,y_train,10).calculate_score_model()
logstic
steps = [('over', SMOTE()), ('model', LogisticRegression())]

pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(pipeline, x_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC : %.3f' % mean(scores))
model = LogisticRegression()
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, x_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))
model = LogisticRegression()
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', model)]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, dataset, label, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()
voting = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under), ('model', voting)]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(pipeline, dataset, label, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))

"""
    loaded data
"""
y_train  = dataset['Class']
x_train  = dataset.drop(columns = ['Class'])
steps = [('over', SMOTE()), ('model', HuberRegressor())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, x_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))
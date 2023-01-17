import pandas as pd

import numpy as np

import random as rnd



# visualization

import re

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold 

from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

from sklearn.metrics import accuracy_score

from datetime import datetime

from dateutil import parser





sp4=pd.read_csv('../input/SP4_Update.csv')

sp5=pd.read_csv('../input/SP5_Update.csv')

sp4=sp4.drop(["Product","CustomStoreCatManCategory","SKUDisplayName","LinePromotionID","OrderPromotionID","LinePromotionID"],axis=1)

sp5=sp5.drop(["Product","CustomStoreCatManCategory","SKUDisplayName","LinePromotionID","OrderPromotionID","LinePromotionID"],axis=1)

sp4['HowOld']='1'

sp4['IsStudentDiscount']=0

sp4['IsLinePromotion']=0

sp4['IsPromotion']=0

sp5['HowOld']='1'

sp5['IsStudentDiscount']=0

sp5['IsLinePromotion']=0

sp5['IsPromotion']=0

for i in range(len(sp4['LinePromotionname'])):

    if(sp4['LinePromotionname'].values[i-1] == 'StudentPriceAVE'):

        sp4['IsStudentDiscount'].values[i-1] = 1

    if(pd.isnull(sp4['LinePromotionname'].values[i-1])==False):

        sp4['IsLinePromotion'].values[i-1] = 1

        

for i in range(len(sp5['LinePromotionname'])):

    if(sp5['LinePromotionname'].values[i-1] == 'StudentPriceAVE'):

        sp5['IsStudentDiscount'].values[i-1] = 1

    if(pd.isnull(sp5['LinePromotionname'].values[i-1])==False):

        sp5['IsLinePromotion'].values[i-1] = 1

        

for i in range(len(sp4['OrderPromotionname'])):

     if(pd.isnull(sp4['OrderPromotionname'].values[i-1])==False):

        sp4['IsPromotion'].values[i-1] = 1



for i in range(len(sp5['OrderPromotionname'])):

     if(pd.isnull(sp5['OrderPromotionname'].values[i-1])==False):

        sp5['IsPromotion'].values[i-1] = 1

        

for i in range(len(sp4['LaunchedDate'])):

        d1=datetime.strptime(sp4['LaunchedDate'].values[i-1], "%m/%d/%Y").date()

        d2 = datetime.strptime(str(datetime.now().date()), "%Y-%m-%d")

        sp4['HowOld'].values[i-1] =  1/(d2.date() - d1).days



for i in range(len(sp5['LaunchedDate'])):

        d1=datetime.strptime(sp5['LaunchedDate'].values[i-1], "%Y-%m-%d %H:%M:%S").date()

        d2 = datetime.strptime(str(datetime.now().date()), "%Y-%m-%d")

        sp5['HowOld'].values[i-1] =  1/(d2.date() - d1).days

        

sp4=sp4.drop(["OrderPromotionname","LinePromotionname","LaunchedDate"],axis=1)   

sp5=sp5.drop(["OrderPromotionname","LinePromotionname","LaunchedDate"],axis=1)  





pl=sp4.pivot_table( 'ProductKey','IsPromotion','TotalUnits',aggfunc='count').plot(kind='bar')

#pl=sp4.pivot_table( 'ProductKey','IsLinePromotion','TotalUnits',aggfunc='count').plot(kind='bar')

#pl=sp4.pivot_table( 'ProductKey','IsStudentDiscount','TotalUnits',aggfunc='count').plot(kind='bar')

pl=sp4.pivot_table( 'ProductKey','ProductName','TotalUnits',aggfunc='count').plot(kind='area')

#pl=sp4.pivot_table( 'ProductKey','OrderPromotionKey','TotalUnits',aggfunc='count').plot(kind='area')















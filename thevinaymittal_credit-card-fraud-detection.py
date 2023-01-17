#First we'll import neccessary frameworks



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing, model_selection, metrics
#Next load dataset



card = pd.read_csv('../input/creditcard.csv')
#Next do some preprocessing



#step 1: Have a quick look



card.describe()
card.info()
#Step 2 DIFFERS , Drop-Impute-Fillna-Features-Encode-Replace-Scaling



#lets look into fradu vs valid cases in our class



fraud = card[card['Class']==1]

valid= card[card["Class"]==0]



outliers = len(fraud)/float(len(valid))



#print (outliers)



##Step 3 : Specifying etcetra options





seed = 42

results =[]

KFold = model_selection.KFold(n_splits=10, random_state=seed)

scoring = 'accuracy'
# After preprocessing lets visualize and correlate



card.hist(figsize=(20,20))
corrmat = card.corr()



fig= plt.figure(figsize=(12,9))



sns.heatmap(corrmat, vmax=0.8, square= True)



plt.show()
#Next, Define XY and find suitable model



x = np.array(card.drop(["Class"],axis=1))

y = np.array(card["Class"])



#xTrain,xTest, yTrain, yTest = model_selection.train_test_split(x,y,test_size=0.2)

#Select model



#model 1/2

from sklearn.ensemble import IsolationForest 



#model 2/2

from sklearn.neighbors import LocalOutlierFactor 



#Initializing

reg1= IsolationForest ( max_samples = len(x), contamination = outliers , random_state= seed  )

reg2= LocalOutlierFactor ( n_neighbors= 20 , contamination = outliers)



#Fit

reg1.fit(x, y)





#Predict

scores1 = reg1.decision_function(x)

predict1= reg1.predict(x)



#Metric

#metrics.mean_squared_error(predict1,YTest)

#Fit



#Predict

predict2 = reg2.fit_predict(x)

scores2= reg2.negative_outlier_factor_



#Metric

#metrics.mean_squared_error(predict2,YTest)

predict1[predict1==1] = 0

predict1[predict1==-1] = 1



n_errors= (predict1 != y).sum()



print ( metrics.accuracy_score (y, predict1))

predict2[predict2==1] = 0

predict2[predict2==-1] = 1



n_errors= (predict2 != y).sum()



print (metrics.classification_report (y, predict2))
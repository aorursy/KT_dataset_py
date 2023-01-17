# Import basic libraries 

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt        

%matplotlib inline

import seaborn as sn

from scipy import stats



import sklearn.metrics as metrics

import sklearn.preprocessing as skp

import sklearn.model_selection as skm



import warnings

warnings.filterwarnings("ignore")





#import classification modules

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import RandomizedSearchCV as rs



# first neural network with keras tutorial

from sklearn.metrics import accuracy_score,precision_score, recall_score, roc_auc_score,roc_curve 
final = pd.read_csv("/home/amir/DataScience/Module 3/Final Exam/hepatitis.csv")
final
print("Shape of The Data:",final.shape)

print("Null Values:\n",final.isnull().sum())

print("Data Types:\n",final.dtypes)

print("Unique Values:\n",final.nunique())
col = ['class','sex','steroid','antivirals','fatigue','malaise','anorexia','liver_big','liver_firm','spleen_palable','spiders','ascites','varices','histology']



for i in col:

    final[i]= final[i].astype(str)

final.dtypes
BOLD = '\033[1m'

END = '\033[0m'

numcols = final.select_dtypes(include=np.number)

    

for col in numcols:



    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,3))

    

    sn.boxplot(final[col], linewidth=1, ax = ax1)

    

    final[col].hist(ax = ax2)



    plt.tight_layout()

    

    plt.show()

    

    print(BOLD+col.center(115)+END)
stringcols = final.select_dtypes(exclude=np.number)



fig = plt.figure(figsize = (8,10))



for i,col in enumerate(stringcols):

    

    fig.add_subplot(6,3,i+1)

    

    final[col].value_counts().plot(kind = 'barh' ,fontsize=10)

    

    plt.tight_layout()

    

    plt.title(col)     

plt.figure(figsize=(12, 10))



corr = final.select_dtypes(include=np.number).corr()



ax = sn.heatmap(corr,vmin=-1, vmax=1, center=0,square=True, annot = True)



ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');



ax.set_yticklabels(ax.get_yticklabels(),rotation=0,horizontalalignment='right');
colz = ['protime']

col_zscore = np.abs(stats.zscore(final[colz]))

col_zscore

out_z = np.where(col_zscore>3)

out_z
final = final[(col_zscore < 3).all(axis =1)]

final
final = pd.get_dummies(final,drop_first = True)

final.dtypes
#Separating the label

y = final['class_2'].copy()



#Separating the features

X = final.drop('class_2',axis=1)

print("Shape of X: ", X.shape)

print("Shape of y: ", y.shape)



features = X.columns

features
X = skp.StandardScaler().fit(X).transform(X)  

X
#create train-test split parts for manual split

trainX, testX, trainy, testy= skm.train_test_split(X, y, test_size=0.25, random_state=99)

print("\n Shape of train split: ")

print(trainX.shape, trainy.shape)

print("\n Shape of test split: ")

print(testX.shape, testy.shape)
#Random Forest  

  

clf = RandomForestClassifier(n_estimators = 200)

clf.fit(trainX,trainy)

    

feature_imp = clf.feature_importances_

feature_imp
feature_array = np.array(features)

feature_array
zipfeatureimp = pd.DataFrame(list(zip(feature_array,feature_imp)))



print(zipfeatureimp.sort_values(by = 1,ascending = False))
#AdaBoost with Randomize Search



parameters = {'n_estimators':(10,200,10),'learning_rate':range(1,20,1)}



clf_tree = AdaBoostClassifier()

clf =rs(clf_tree,parameters,cv =5, scoring = 'precision')

clf.fit(trainX,trainy)



predictions = clf.predict(testX)

fpr1 , tpr1, _ = roc_curve(testy,predictions)

    

print(" accuracy of AdaBoost Classifier (%)", accuracy_score(testy,predictions)*100)



print("Precision of AdaBoost Classifierr (%)",precision_score(testy,predictions)*100)



print("Recall of AdaBoost Classifierr (%)",recall_score(testy,predictions)*100)



print("Best score (%)",clf.best_score_*100)



print("Best Parameters (%)",clf.best_params_)



print("Best Estimator (%)",clf.best_estimator_)

#Decision Tree with Randomize Search



parameters = {'min_samples_split' : range(10,200,10), 'max_depth' : range(1,20,1),'random_state':(1,20,1)}



clf_tree = DecisionTreeClassifier()

clf =rs(clf_tree,parameters,cv =5, scoring = 'precision')

clf.fit(trainX,trainy)



predictions = clf.predict(testX)

fpr2 , tpr2, _ = roc_curve(testy,predictions)

    

print(" accuracy of Decision Tree Classifier (%)", accuracy_score(testy,predictions)*100)



print("Precision of Decision Tree Classifierr (%)",precision_score(testy,predictions)*100)



print("Recall of Decision Tree Classifierr (%)",recall_score(testy,predictions)*100)



print("Best score (%)",clf.best_score_*100)



print("Best Parameters (%)",clf.best_params_)



print("Best Estimator (%)",clf.best_estimator_)

#GradientBoostingClassifier with Randomize Search



parameters = {'n_estimators':(10,200,10), 'max_depth':(1,20,1), 'random_state':(1,20,1)}



clf_tree = GradientBoostingClassifier()

clf =rs(clf_tree,parameters, scoring = 'precision')

clf.fit(trainX,trainy)

    

predictions = clf.predict(testX)

fpr3 , tpr3, _ = roc_curve(testy,predictions)



print(" accuracy of Gradient Boosting Classifier (%)", accuracy_score(testy,predictions)*100)



print("Precision of Gradient Boosting Classifierr (%)",precision_score(testy,predictions)*100)



print("Recall of Gradient Boosting Classifierr (%)",recall_score(testy,predictions)*100)



print("Best score (%)",clf.best_score_*100)



print("Best Parameters (%)",clf.best_params_)



print("Best Estimator (%)",clf.best_estimator_)

roc_auc1 = metrics.auc(fpr1 ,tpr1)

roc_auc2 = metrics.auc(fpr2 ,tpr2)

roc_auc3 = metrics.auc(fpr3 ,tpr3)



plt.figure()

plt.title('Recevier Operating Chatactersticks')

plt.plot(fpr1 , tpr1 , 'b', label ='roc default auc = %0.2f' % roc_auc1)

plt.plot(fpr2 , tpr2 , 'r', label ='roc grid search auc = %0.2f' % roc_auc2)

plt.plot(fpr3 , tpr3 , 'g', label ='roc random search auc = %0.2f' % roc_auc3)

plt.legend(loc = 'lower right')

plt.plot([0 , 1],[0,1],'r--')

plt.xlim([0 , 1])

plt.ylim([0 , 1])

plt.ylabel('True Positive rate')

plt.xlabel('False Positive rate')

plt.show()
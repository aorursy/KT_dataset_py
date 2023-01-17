import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# read data into dataset variable

train = pd.read_csv("../input/train.csv") #[:20000]

test = pd.read_csv("../input/test.csv") #[:10000]

#train=train.append(test)

train.describe().T

from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}





X = train.drop('target',axis=1).fillna(0) 



def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    





Y=train['target'].fillna(0)

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)

#poly = PolynomialFeatures(2)

#X=poly.fit_transform(X)



#print(X)



names = [

         #'ElasticNet',

         #'SVC',

         #'kSVC',

         #'KNN',

         'DecisionTree',

         'RandomForestClassifier',

         #'GridSearchCV',

         #'HuberRegressor',

         #'Ridge',

         #'Lasso',

         #'LassoCV',

         #'Lars',

         #'BayesianRidge',

         #'SGDClassifier',

         #'RidgeClassifier',

         #'LogisticRegression',

         #'OrthogonalMatchingPursuit',

         #'RANSACRegressor',

         ]



classifiers = [

    #ElasticNetCV(cv=10, random_state=0),

    #SVC(),

    #SVC(kernel = 'rbf', random_state = 0),

    #KNeighborsClassifier(n_neighbors = 1),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 200),

    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),

    #HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    #Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    #Lasso(alpha=0.05),

    #LassoCV(),

    #Lars(n_nonzero_coefs=10),

    #BayesianRidge(),

    #SGDClassifier(),

    #RidgeClassifier(),

    #LogisticRegression(),

    #OrthogonalMatchingPursuit(),

    #RANSACRegressor(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



temp=zip(names,classifiers,correction)

#print(temp)



for name, clf,correct in temp:

    regr=clf.fit(X,Y)

    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)

    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score



    # Confusion Matrix

    print(name,'Confusion Matrix')

    print(confusion_matrix(Y, np.round(regr.predict(X) ) ) )

    print('--'*40)



    # Classification Report

    print('Classification Report')

    print(classification_report(Y,np.round( regr.predict(X) ) ))



    # Accuracy

    print('--'*40)

    logreg_accuracy = round(accuracy_score(Y, np.round( regr.predict(X) ) ) * 100,2)

    print('Accuracy', logreg_accuracy,'%')

    

    # Create a submission file

    sub = pd.DataFrame()

    sub['id'] = test['id']

    sub['target'] = regr.predict(test)

    sub.to_csv(name, index=False)



    print(sub.head())



    
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected = True)

import plotly.graph_objs as go

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer

from sklearn.decomposition import PCA
df = pd.read_csv('../input/slice_localization_data.csv')[:100000]



df.describe().T





from sklearn.linear_model import OrthogonalMatchingPursuitCV,RANSACRegressor,LogisticRegression,MultiTaskElasticNet,HuberRegressor, Ridge, Lasso,LassoCV,LarsCV,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns



from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNet,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



X = df.drop('reference',axis=1) # we only take the first two features.

#le = preprocessing.LabelEncoder()

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)



    

#le.fit(train['Outcome'])

#print(list(le.classes_))



Y=np.round(df['reference'])

#scaler = MinMaxScaler()

#scaler.fit(X)

#X=scaler.transform(X)

#poly = PolynomialFeatures(2)

#X=poly.fit_transform(X)





names = [

         'DecisionTree',

         'RandomForestClassifier',    

         #'ElasticNet',

         #'KNN',

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

         #'SVC',

         #'kSVC',

         ]



classifiers = [

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 200,n_jobs=-1),    

    #ElasticNet(alpha=0.1),

    #KNeighborsClassifier(n_neighbors = 5,n_jobs=-1),

    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),

    #HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    #Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    #Lasso(alpha=0.5),

    #LassoCV(),

    #LarsCV(n_jobs=-1),

    #BayesianRidge(),

    #SGDClassifier(n_jobs=-1),

    #RidgeClassifier(),

    #LogisticRegression(n_jobs=-1),

    #OrthogonalMatchingPursuitCV(n_jobs=-1),

    #SVC(),

    #SVC(kernel = 'rbf', random_state = 0),    

    #RANSACRegressor(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



model=zip(names,classifiers,correction)

print(model)



for name, clf,correct in model:

    regr=clf.fit(X,Y)

    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)

    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score



    # Confusion Matrix

    print(name,'Confusion Matrix')

    conf=confusion_matrix(Y, np.round(regr.predict(X) ) )     

    #label = Y.sort_values().unique()

    #sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label, cmap="YlGnBu")

    #plt.show()



    print(confusion_matrix(Y, np.round(regr.predict(X) ) ) )

    print('--'*40)



    # Classification Report

    print(name,'Classification Report')

    print(classification_report(Y,np.round( regr.predict(X) ) ))



    # Accuracy

    print('--'*40)

    logreg_accuracy = round(accuracy_score(Y, np.round( regr.predict(X) ) ) * 100,2)

    print('Accuracy', logreg_accuracy,'%')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/creditcard.csv")

import seaborn as sns

import matplotlib.pyplot as plt # Visuals

plt.style.use('ggplot') # Using ggplot2 style visuals 



f, ax = plt.subplots(figsize=(11, 15))



ax.set_axis_bgcolor('#fafafa')

ax.set(xlim=(-.05, 50))

plt.ylabel('Dependent Variables')

plt.title("Box Plot of Pre-Processed Data Set")

ax = sns.boxplot(data = train, 

  orient = 'h', 

  palette = 'Set2')







#sns.set(style="ticks", color_codes=True)

#g = sns.pairplot(train, hue='Dx')



    

def plot_corr(df,size=10):

    import matplotlib.pyplot as plt

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''



    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns);

    plt.yticks(range(len(corr.columns)), corr.columns);



plot_corr(train)
new_col= train.groupby('Class').mean()

print(new_col.head().T)

train=train.fillna(0)

from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns



def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)





from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}



n_col=36

X = train.drop(['Class'],axis=1) 

Y=train['Class']

#X=X.fillna(value=0)

#scaler = MinMaxScaler()

#scaler.fit(X)

#X=scaler.transform(X)

#poly = PolynomialFeatures(2)

#X=poly.fit_transform(X)





names = [

         'DecisionTree',

         'RandomForestClassifier',    

         #'ElasticNet',

         #'SVC',

         #'kSVC',

         'KNN',

         #'GridSearchCV',

         'HuberRegressor',

         'Ridge',

         'Lasso',

         'LassoCV',

         'Lars',

         'BayesianRidge',

         'SGDClassifier',

         'RidgeClassifier',

         'LogisticRegression',

         'OrthogonalMatchingPursuit',

         #'RANSACRegressor',

         ]



classifiers = [

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators = 200),

    #ElasticNetCV(cv=10, random_state=0),

    #SVC(),

    #SVC(kernel = 'rbf', random_state = 0),

    KNeighborsClassifier(n_neighbors = 1),

    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),

    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),

    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),

    Lasso(alpha=0.05),

    LassoCV(),

    Lars(n_nonzero_coefs=10),

    BayesianRidge(),

    SGDClassifier(),

    RidgeClassifier(),

    LogisticRegression(),

    OrthogonalMatchingPursuit(),

    #RANSACRegressor(),

]

correction= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



models=zip(names,classifiers,correction)

   

for name, clf,correct in models:

    regr=clf.fit(X,Y)

    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)

    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score

    

    # Confusion Matrix

    print(name,'Confusion Matrix')

    conf=confusion_matrix(Y, np.round(regr.predict(X) ) )     

    label = ["0","1"]

    sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label, cmap="YlGnBu")

    plt.show()

    

    print('--'*40)



    # Classification Report

    print(name,'Classification Report')

    classif=classification_report(Y,np.round( regr.predict(X) ) )

    print(classif)





    # Accuracy

    print('--'*40)

    logreg_accuracy = round(accuracy_score(Y, np.round( regr.predict(X) ) ) * 100,2)

    print('Accuracy', logreg_accuracy,'%')
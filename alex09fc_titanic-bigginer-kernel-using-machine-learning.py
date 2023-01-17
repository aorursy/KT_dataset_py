import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.feature_selection import RFE

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

from sklearn.naive_bayes import GaussianNB 

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline 

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor 

from sklearn.ensemble import ExtraTreesRegressor 

from sklearn.ensemble import AdaBoostRegressor 

from sklearn.metrics import mean_squared_error



from pandas.tools.plotting import scatter_matrix

from sklearn.model_selection import KFold 

from sklearn.model_selection import cross_val_score 

from sklearn.model_selection import GridSearchCV
#my first kernel any comments/feedback would really help!!

Titanic_df= pd.read_csv('../input/train.csv',index_col = False)
Titanic_df.head()
Titanic_df.fillna(value=0, inplace=True)

Titanic_df.drop(["Cabin"],1,inplace =True)

Titanic_df.drop(["Embarked"],1,inplace =True)

Titanic_df.drop(["PassengerId"],1,inplace =True)

Titanic_df.drop(["Name"],1,inplace =True)

Titanic_df.drop(["Ticket"],1,inplace =True)

Titanic_df.drop(["Fare"],1,inplace =True)

Titanic_df.drop(["SibSp"],1,inplace =True)
Titanic_df=Titanic_df.apply(LabelEncoder().fit_transform)
Titanic_df.head()
Titanic_df.shape
Titanic_df.describe
Titanic_df.skew
Titanic_df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1) 

plt.show()

scatter_matrix(Titanic_df)

plt.show()
correlations = Titanic_df.corr() 

# plot correlation matrix 

fig = plt.figure() 

ax = fig.add_subplot(111) 

cax = ax.matshow(correlations, vmin=-1, vmax=1) 

fig.colorbar(cax)

plt.show()

seed = 7

scoring = 'neg_mean_squared_error'
X= np.array(Titanic_df.drop(["Survived"],1),dtype=np.float64)

y= np.array(Titanic_df["Survived"],dtype=np.float64)
sc = StandardScaler()

sc.fit(X,y)

sc.transform(X,y)
X_train ,X_test, y_train,y_test = train_test_split(X,y,test_size =0.20, random_state = 15)
X.shape
Titanic_df.dtypes
Titanic_df.groupby("Pclass").sum()
Titanic_df.groupby(['Sex','Survived'])['Survived'].count()
estimator = SVR(kernel="linear")

selector = RFE(estimator, 5, step=1)

selector = selector.fit(X,y)
selector.support_
selector.ranking_
models = []

models.append(('LR', LogisticRegression())) 

models.append(('LDA', LinearDiscriminantAnalysis())) 

models.append(('KNN', KNeighborsClassifier())) 

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC())) 

results = [] 

names = [] 

for name, model in models: 

    kfold = KFold(n_splits=10, random_state=seed) 

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results) 

    names.append(name)

    mod = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 

    print(mod)

ensembles = [] 

ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))

ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())]))) 

ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))

ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))

results = [] 

names = []

for name, model in ensembles: 

    kfold = KFold(n_splits=10, random_state=seed) 

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results) 

    names.append(name) 

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 

    print(msg)



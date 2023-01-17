import pandas as pd

import numpy as np

df=pd.read_csv('../input/Iris.csv')
df.head(10)
# When we have categorical values in the data set, we can create a table and sumarize it

df.describe(include=['O'])
missing_data=df.isnull()
missing_data.head(5)
missing_data.sum()
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("--------------------------------")
#Finding the porcentage of  missing data

round(((missing_data.sum()/len(missing_data))*100), 4)
df.shape
df.info()
#On classification problems you need to know how balanced the class values are.( This is an example)

df.groupby('Species').size() 
#Lets check the types 

df.dtypes

#Apply only when we have categorical values in our X



#df = pd.get_dummies(df)

# Apply when our target is categorical and we need to calculate the correlation 

from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

df["Target"] = lb_make.fit_transform(df["Species"])

df[["Species", "Target"]].head(1)
# We can analyze all the data set 

df.describe()
#We can analyze any colums separate

df['SepalLengthCm'].describe()
# When we have categorical values in the data set, we can create a table and sumarize it

df.describe(include=['O'])
df.corr()
corr_matrix= df.corr()
#To check a correlation with our target



corr_matrix['Target'].sort_values(ascending=False)

import seaborn as sns
sns.heatmap(df.corr(), vmin=-1, vmax=1.0, annot=True)
df.skew()
from matplotlib import pyplot as plt

df.hist(bins=10, figsize=(20,15))

plt.show()
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,15))

plt.show()
from pandas.plotting import scatter_matrix



scatter_matrix(df,figsize=(20,20))

plt.show()
import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

X = df.drop(['Species','Target','Id'],axis=1)  #independent columns

y = df['Species']    #target column

#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=4)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
import pandas as pd

import numpy as np





X =  df.drop(['Species','Target','Id'],axis=1) #independent columns

y = df['Species']     #target column i.e price range

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
X_train=df.drop(['Id','Target','Species'],axis=1).values

y_train=df['Species'].values
models = []

models.append(('PER', Perceptron( max_iter=1000,tol=1e-3)))

models.append(('XGB', XGBClassifier(objective="binary:logistic", random_state=42)))

models.append(('LSVC',  LinearSVC(max_iter=10000)))

models.append(('SGDC',  SGDClassifier(max_iter=1000,tol=1e-3)))

models.append(('LR', LogisticRegression(solver='lbfgs',max_iter=10000, multi_class='auto')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(gamma='scale',max_iter=1000)))

models.append(('AB', AdaBoostClassifier()))

models.append(('GBM', GradientBoostingClassifier()))

models.append(('RF', RandomForestClassifier(n_estimators=100)))

models.append(('ET', ExtraTreesClassifier(n_estimators=100)))



# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=42)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
fig = pyplot.figure()

fig.suptitle( ' Algorithm Comparison ' )

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
X=df.drop(['Id','Target','Species'],axis=1).values

y=df['Species'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,

test_size=0.20, random_state=42)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

logreg_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log

print(accuracy_score(y_test, logreg_pred))

print(confusion_matrix(y_test, logreg_pred))

print(classification_report(y_test, logreg_pred))
# Support Vector Machines



svc = SVC()

svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_svc

print(accuracy_score(y_test, svc_pred ))

print(confusion_matrix(y_test, svc_pred ))

print(classification_report(y_test, svc_pred ))
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn

print(accuracy_score(y_test, knn_pred))

print(confusion_matrix(y_test, knn_pred ))

print(classification_report(y_test, knn_pred ))
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

gaussian_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian

print(accuracy_score(y_test, gaussian_pred))

print(confusion_matrix(y_test, gaussian_pred ))

print(classification_report(y_test,gaussian_pred ))

#GaussianNB(priors=None, var_smoothing=1e-09)
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, y_train)

perceptron_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)

acc_perceptron

print(accuracy_score(y_test, perceptron_pred))

print(confusion_matrix(y_test, perceptron_pred))

print(classification_report(y_test,perceptron_pred))
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

lsvc_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc

print(accuracy_score(y_test, lsvc_pred ))

print(confusion_matrix(y_test, lsvc_pred ))

print(classification_report(y_test,lsvc_pred ))
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

sgd_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd

print(accuracy_score(y_test, sgd_pred  ))

print(confusion_matrix(y_test, sgd_pred  ))

print(classification_report(y_test,sgd_pred  ))
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

detree_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree

print(accuracy_score(y_test, detree_pred ))

print(confusion_matrix(y_test, detree_pred ))

print(classification_report(y_test,detree_pred ))
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

rforest_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest

print(accuracy_score(y_test, rforest_pred  ))

print(confusion_matrix(y_test, rforest_pred  ))

print(classification_report(y_test,rforest_pred  ))
#Linear Discriminant Analysis

clf = LinearDiscriminantAnalysis()

clf.fit(X_train, y_train)

clf_pred= clf.predict(X_test)

acc_clf = round(clf.score(X_train, y_train) * 100, 2)

acc_clf

print(accuracy_score(y_test, clf_pred ))

print(confusion_matrix(y_test, clf_pred  ))

print(classification_report(y_test,clf_pred  ))

#LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.0001)

# Ada Boost Classifier

AB = AdaBoostClassifier()

AB.fit(X_train, y_train)

AB_pred= AB.predict(X_test)

acc_AB = round(AB.score(X_train, y_train) * 100, 2)

acc_AB

print(accuracy_score(y_test, AB_pred))

print(confusion_matrix(y_test, AB_pred  ))

print(classification_report(y_test,AB_pred ))

#AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)

# Gradient Boosting Classifier

GBC = GradientBoostingClassifier()

GBC.fit(X_train, y_train)

GBC_pred= GBC.predict(X_test)

acc_GBC = round(GBC.score(X_train, y_train) * 100, 2)

acc_GBC

print(accuracy_score(y_test, GBC_pred))

print(confusion_matrix(y_test,GBC_pred ))

print(classification_report(y_test,GBC_pred ))

#GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
#ExtraTreesClassifier

ETC=ExtraTreesClassifier(n_estimators=100)

ETC.fit(X_train, y_train)

ETC_pred= ETC.predict(X_test)

acc_ETC = round(ETC.score(X_train, y_train) * 100, 2)

acc_ETC

print(accuracy_score(y_test, ETC_pred))

print(confusion_matrix(y_test,ETC_pred))

print(classification_report(y_test,ETC_pred ))

#ExtraTreesClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)
#XGBClassifier(objective

xgbs = XGBClassifier(objective="binary:logistic", random_state=42)

xgbs.fit(X_train, y_train)

xgbs_pred= xgbs.predict(X_test)

acc_xgbs = round(xgbs.score(X_train, y_train) * 100, 2)

acc_xgbs

print(accuracy_score(y_test, xgbs_pred))

print(confusion_matrix(y_test,xgbs_pred))

print(classification_report(y_test,xgbs_pred ))
models = pd.DataFrame({

    'Model': ['LogisticRegression', 'Support Vector Machines',

              'KNeighbors', 'Gaussian Naive Bayes', 'Perceptron', 'LinearSVC', 

              'Stochastic Gradient Descent', 'Decision Tree', 'RandomForest', 'Linear Discriminant Analysis', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier', 'XGBClassifier'],

    'Score': [acc_log,acc_svc,acc_knn,acc_gaussian,acc_perceptron,acc_linear_svc,acc_sgd,acc_decision_tree,acc_random_forest,acc_clf,acc_AB,acc_GBC,acc_ETC,acc_xgbs

]})

models.sort_values(by='Score', ascending=False)
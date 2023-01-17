#imports

import pandas as pd

from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
Test=pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

Train=pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

Y=Train['Class']

train = Train.drop(['Unnamed: 0', 'Class'], axis=1)

test=Test.drop('Unnamed: 0', axis=1)
train.describe()
class1=Train[Train.Class==1]

class0=Train[Train.Class==0]

class1.describe()
class0.describe()
import matplotlib.pyplot as plt



labels = 'Class 0', 'Class 1'

sizes = [len(class0), len(class1)]

colors = ['red', 'blue']



plt.pie(sizes, labels=labels, colors=colors)

plt.show()
#There's 1520 samples of class 1, we can either oversample, or undersample. IDK if I'm doing something wrong, but neither gave me good 

#results.

#I'd say it's because this is the default accuracy you could get, if it was balanced. Like, the model predicts class 0 by default, and gets 

#a lot right because there's a higher chance of it being class 0, locally.

listo=['V2', 'V3', 'V4','V5','V7','V8','V9',]

for feature in listo:

    onehot=pd.get_dummies(train[feature], prefix=str(feature), prefix_sep='_')

    train=train.drop(feature, axis=1)

    train=train.join(onehot)

for feature in listo:

    onehot=pd.get_dummies(test[feature], prefix=str(feature), prefix_sep='_')

    test=test.drop(feature, axis=1)

    test=test.join(onehot)

    
class1.describe()
class0=class0.sample(frac=1)#shuffling

class0_cut=class0[0:1800]

train=class1.append(class0_cut)

Y=train['Class']

train=train.drop(['Unnamed: 0'], axis=1)#We'll update train as needed to view characteristics

for feature in listo:

    onehot=pd.get_dummies(train[feature], prefix=str(feature), prefix_sep='_')

    train=train.drop(feature, axis=1)

    train=train.join(onehot)
train.head()
train.describe()
import numpy as np

import seaborn as sns

#get correlations of each features in dataset

corrmat = train.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(37,37))

#plot heat map

g=sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
featlist=list(train.columns)

#featlist.remove("Unnamed: 0")

#featlist.remove("Class")



for feature in featlist:

    x=Train[feature]

    y=Train['Class']

    plt.scatter(x,y)

    plt.xlabel(feature)

    plt.ylabel("Class")

    plt.show()
train.describe()

#yeah we still have the class variable, removing now

#train=train.drop("Class", axis=1)
#we'll be using XGBoost, because it's a very effective ensemble method.

#We'll also try using SVC, but XGB gets a better rep in most cases.

model=XGBClassifier()

X_train, X_test, y_train, y_test = train_test_split(train, Y, test_size=0.20, random_state=130)

model.fit(X_train, y_train, eval_metric=precision_score)

y_pred=model.predict(X_test)

score = precision_score(y_test, y_pred)

print("Score on undersampled, vanilla features, and XGB no tuning: "+str(score))
#Grid Search CV

#It took around 3 hours to run this locally, I'll just take the values from it. I tried commiting this notebook

#and I'm pretty sure that it'll time out before this block finishes executing, hence commenting out the complex bits

from sklearn.model_selection import GridSearchCV

gammas = np.array([1,0.1,0.01,0]) #best fit found to be 1

min_child_weights = np.array([1, 0.1, 0.01,0]) #0.01

max_depths= np.array([4,5,6,7]) #6

scale_pos_weights=np.array([0.5, 0.7, 1]) #1

subsamples= np.array([0.5, 0.7, 0.1]) #0.5

model = XGBClassifier()



"""

grid = GridSearchCV(estimator=model, param_grid=dict(subsample=subsamples,

                                                     scale_pos_weight=scale_pos_weights,

                                                     max_depth=max_depths,

                                                     min_child_weight=min_child_weights,

                                                     gamma=gammas

                                                    ))

grid.fit(train, Y)

print(grid)

# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.subsample)

print(grid.best_estimator_.scale_pos_weight)

print(grid.best_estimator_.max_depth)

print(grid.best_estimator_.min_child_weight)

print(grid.best_estimator_.gamma)

"""
#Let's see if that was useful

model = XGBClassifier(subsample=0.5,

                      scale_pos_weight=1,

                      max_depth=6,

                      min_child_weight=1,

                      gamma=0.01

                        )

X_train, X_test, y_train, y_test = train_test_split(train, Y, test_size=0.20, random_state=130)

model.fit(X_train, y_train, eval_metric=precision_score)

y_pred=model.predict(X_test)

score = precision_score(y_test, y_pred)

print("Score on undersampled, OHE, with tuned XGB model: "+str(score))
#Let's try an SVC

from sklearn import svm

model = svm.SVC(max_iter=1000)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

model.fit(X_train, y_train)

y_pred=model.predict(X_test)

score = precision_score(y_test, y_pred)

print("Score on undersampled, OHE features, with untuned SVC model: "+str(score))
train.describe()
from imblearn.over_sampling import BorderlineSMOTE

Y=Train['Class']

X_exp, y_exp = BorderlineSMOTE().fit_resample(train, Y)

X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, test_size=0.20, random_state=130)



model=XGBClassifier()

X=X_exp

y=y_exp

from imblearn.under_sampling import TomekLinks



#With how Borderline SMOTE makes more neighbours in nearly similar directions, I thought implementing TOMEK link removal would be good



tl = TomekLinks(return_indices=True, ratio='majority')

X_tl, y_tl, id_tl = tl.fit_sample(X, y)





def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()

    

plot_2d_space(X,y,"standard")

plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')



#implementing a function for easier KStratified folds test, because there's no default test.

def get_score(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    return model.score(X_test, y_test)

from sklearn.model_selection import StratifiedKFold

folds=StratifiedKFold(n_splits=7, shuffle=True, random_state=130)

vallist=[]

from statistics import mean

for train_set, test_set in folds.split(X_tl,y_tl):

    print(get_score(model, X_tl[train_set], X_tl[test_set], y_tl[train_set], y_tl[test_set]))

    vallist.append(get_score(model, X_tl[train_set], X_tl[test_set], y_tl[train_set], y_tl[test_set]))
mean(vallist)
model = XGBClassifier(subsample=0.5,

                      scale_pos_weight=1,

                      max_depth=6,

                      min_child_weight=1,

                      gamma=0.01

                        )

X_train, X_test, y_train, y_test = train_test_split(X_tl, y_tl, test_size=0.20, random_state=130)

model.fit(X_train, y_train, eval_metric=precision_score)

y_pred=model.predict(X_test)

score = precision_score(y_test, y_pred)

print("Score on BorderlineSMOTE TLinked, OHE , with tuned XGB model: "+str(score))
Theory=model.predict_proba(test.values)

SMOTETLOHETunedXGB=pd.DataFrame()

SMOTETLOHETunedXGB['Id']=Test['Unnamed: 0']

SMOTETLOHETunedXGB['PredictedValue'] = Theory[:,1]

#SMOTETLOHETunedXGB.to_csv("../input/webclubrecruitment2019/SMOTETLOHETunedXGB.csv", index=False)
#Feature selection with Chi2

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=16)

X=np.absolute(train.values)

y=Y.values

fit= test.fit(X, y)

features = fit.transform(X)



features = pd.DataFrame.from_records(features)
model = XGBClassifier(subsample=0.5,

                      scale_pos_weight=1,

                      max_depth=6,

                      min_child_weight=1,

                      gamma=0.01

                        )

train = Train.drop(['Unnamed: 0','Class'], axis=1)

train=train[['V1','V6', 'V12', 'V14', 'V10', 'V13', 'V6']]

X_exp, y_exp = BorderlineSMOTE().fit_resample(train, Y)

X=X_exp

y=y_exp

tl = TomekLinks(return_indices=True, ratio='majority')

X_tl, y_tl, id_tl = tl.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_tl, y_tl, test_size=0.20, random_state=130)

model.fit(X_train, y_train, eval_metric=precision_score)

y_pred=model.predict(X_test)

score = precision_score(y_test, y_pred)

print("Score on BorderlineSMOTE TLinked, selected features, with tuned XGB model: "+str(score))

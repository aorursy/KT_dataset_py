# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('/kaggle/input/glass/glass.csv')



df.sample(5)
X = df.drop(['Type'], axis=1)

Y = df['Type']
df.corr()['Type'].abs().sort_values(ascending=False)

df['Type'].unique()
sns.countplot(x='Type', data=df)
sns.boxplot('Type', 'RI', data =df)
import matplotlib.pylab as plt

from sklearn import preprocessing

from scipy.stats import skew

from scipy.stats import boxcox



# getting features names to loop

classes = X.columns.values



# This will contain the unskewed features

X_unsk = pd.DataFrame()



# looping through the 

for c in classes:

    scaled = preprocessing.scale(X[c]) 

    boxcox_scaled = preprocessing.scale(boxcox(X[c] + np.max(np.abs(X[c]) +1) )[0])

    

    # Populating 

    X_unsk[c] = boxcox_scaled

    

    #Next We calculate Skewness using skew in scipy.stats

    skness = skew(scaled)

    boxcox_skness = skew(boxcox_scaled)

    

    #We draw the histograms 

    figure = plt.figure()

    # First the original data shape

    figure.add_subplot(121)   

    plt.hist(scaled,facecolor='red',alpha=0.55) 

    plt.xlabel(c + " - Transformed") 

    plt.title("Skewness: {0:.2f}".format(skness)) 

    

    # then the unskewed

    figure.add_subplot(122) 

    plt.hist(boxcox_scaled,facecolor='yellow',alpha=0.55) 

    plt.title("Skewness: {0:.2f}".format(boxcox_skness)) 



    plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import math



# Here I use the unskewed dataset

X = X_unsk

X_tr, X_ts, y_tr, y_ts = train_test_split(X, Y, test_size=0.40, random_state=42)



rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

param_grid = { "criterion" : ["gini", "entropy"]

              , "min_samples_leaf" : [1, 5, 10]

              , "min_samples_split" : [2, 4, 10, 12, 16]

              , "n_estimators": [100, 125, 200]}

gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

gs = gs.fit(X_tr, y_tr)
print(gs.best_score_)

print(gs.best_params_)
bp = gs.best_params_

rf = RandomForestClassifier( criterion=bp['criterion'], 

                             n_estimators=bp['n_estimators'],

                             min_samples_split=bp['min_samples_split'],

                             min_samples_leaf=bp['min_samples_leaf'],

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)



rf.fit(X_tr, y_tr)

pred = rf.predict(X_ts)



score = rf.score(X_ts, y_ts)

print("Score: %.3f" % (score))
pd.concat((pd.DataFrame(X.columns, columns = ['variable']), 

           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)[:20]
from sklearn.metrics import confusion_matrix

import itertools

#print(y_ts.values)

#print(pred)



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).round(decimals=2)

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



cnf_matrix = confusion_matrix(y_ts.values, pred)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=np.sort(y_ts.unique()), normalize=False,

                      title='Confusion matrix, without normalization')
from xgboost import XGBClassifier



# Here I use the unskewed dataset

X = X_unsk

X_tr, X_ts, y_tr, y_ts = train_test_split(X, Y, test_size=0.4, random_state=42)

xgb = XGBClassifier()



param_grid = { "max_depth" : [5]

              , "learning_rate" : [0.125]

              , "n_estimators": [50]

              , "reg_lambda": [.1]}

gs = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

gs = gs.fit(X_tr, y_tr)
print(gs.best_score_)

print(gs.best_params_)
bp = gs.best_params_

xgb = XGBClassifier( max_depth=bp['max_depth'], 

                             n_estimators=bp['n_estimators'],

                             learning_rate=bp['learning_rate'],

                   reg_lambda=bp['reg_lambda'])



xgb.fit(X_tr, y_tr)

pred = xgb.predict(X_ts)



score = xgb.score(X_ts, y_ts)

print("Score: %.3f" % (score))
pd.concat((pd.DataFrame(X.columns, columns = ['variable']), 

           pd.DataFrame(xgb.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)[:20]
cnf_matrix = confusion_matrix(y_ts.values, pred)



plt.figure()

plot_confusion_matrix(cnf_matrix, classes=np.sort(y_ts.unique()), normalize=False,

                      title='Confusion matrix, without normalization')
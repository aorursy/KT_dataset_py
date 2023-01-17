import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.sample(5)
# New imports 

import matplotlib.colors as colors
df = df.rename(columns={'trestbps': 'restbp'})
df.columns
df.info()
df.isnull().sum()
df.describe()
x = df.drop(columns=['target']).copy()

y = df['target'].copy()
x_en = pd.get_dummies(x,columns=['cp','restecg','slope','ca','thal'])
from sklearn.model_selection import train_test_split as tts 

x_train,x_test,y_train,y_test = tts(x_en,y,test_size=0.4,random_state=23)
from sklearn.preprocessing import scale

x_train_s = scale(x_train)

x_test_s  = scale(x_test)
from sklearn.svm import SVC

svc = SVC(random_state=344)

svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)
score = svc.score(x_test,y_test)

score
def confusion(test, predict, labels, title='Confusion Matrix'):

    '''

        test: true label of test data, must be one dimensional

        predict: predicted label of test data, must be one dimensional

        labels: list of label names, ie: ['positive', 'negative']

        title: plot title

    '''



    bins = len(labels)

    # Make a 2D histogram from the test and result arrays

    pts, xe, ye = np.histogram2d(test, predict, bins)



    # For simplicity we create a new DataFrame

    pd_pts = pd.DataFrame(pts.astype(int), index=labels, columns=labels )

    

    # Display heatmap and add decorations

    hm = sns.heatmap(pd_pts, annot=True, fmt="d")    

    hm.axes.set_title(title, fontsize=20)

    hm.axes.set_xlabel('Predicted', fontsize=18)

    hm.axes.set_ylabel('Actual', fontsize=18)



    return None
confusion(y_test, y_pred, ['Does not have HD', 'Has HD'], title='Support Vector Classifier')
# making a parameters grid

param_grid = [{'C':[1,10,100,1000],

               'gamma':[0.001,0.0001],

              'kernel':['rbf']}] #radial basis function



from sklearn.model_selection import GridSearchCV



optimal_params = GridSearchCV(SVC(),param_grid,cv=10,verbose=0)

optimal_params.fit(x_train_s,y_train)

optimal_params.best_params_
svc = SVC(random_state=334,C=10,gamma=0.0001,kernel='rbf')

svc.fit(x_train_s,y_train)

y_pred = svc.predict(x_test)

score = svc.score(x_test,y_test)

score
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(x_train_s,y_train)

score = rfc.score(x_test,y_test)

score
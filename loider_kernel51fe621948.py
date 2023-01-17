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
import pandas as pd

import numpy as np

import sklearn

import statsmodels.api as sm

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix, f1_score, make_scorer

import matplotlib.mlab as mlab

%matplotlib inline



heart_dis_df = pd.read_csv("../input/framingham-heart-study-dataset/framingham.csv")



print('The number of samples into the train data is {}.'.format(heart_dis_df.shape[0]))



heart_dis_df.head()
heart_dis_df.isnull().sum()
#heart_dis_df["glucose"].fillna((heart_dis_df["glucose"].mean()), inplace=True)

heart_dis_df.drop(['education'],axis=1,inplace=True)

heart_dis_df.dropna(axis=0,inplace=True)

print('The number of samples after drop NA data is {}.'.format(heart_dis_df.shape[0]))



heart_dis_df.isnull().sum()
X=heart_dis_df.iloc[:,:-1]

y=heart_dis_df.iloc[:,-1]



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)
def draw_histograms(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')

        ax.set_title(feature+" Distribution",color='DarkRed')

        

    fig.tight_layout()  

    plt.show()

draw_histograms(heart_dis_df,heart_dis_df.columns,6,3)


sns.countplot(x='TenYearCHD',data=heart_dis_df)

heart_dis_df.TenYearCHD.value_counts()
heart_dis_df.describe()
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.

# The "accuracy" scoring is proportional to the number of correct classifications

import warnings

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')

    rfecv.fit(X, y)



print("Optimal number of features: %d" % rfecv.n_features_)

print('Selected features: %s' % list(X.columns[rfecv.support_]))



# Plot number of features VS. cross-validation scores

plt.figure(figsize=(10,6))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
Selected_features = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

#Selected_features = ['male', 'age', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']



X = X[Selected_features]



plt.subplots(figsize=(15, 5))

sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")

plt.show()
from sklearn.dummy import DummyClassifier



logreg=LogisticRegression()

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)



sklearn.metrics.accuracy_score(y_test,y_pred)

print("Accuracy_score : ", sklearn.metrics.accuracy_score(y_test,y_pred))

print("F1_score : ", f1_score(y_test, y_pred, average="macro"))



dummy = DummyClassifier(strategy="constant", random_state=0, constant=1)

dummy.fit(X_train,y_train)



print("Dummy f1 score:", f1_score(y_test, dummy.predict(X_test), average="micro"))
from sklearn.model_selection import GridSearchCV



f1 = make_scorer(f1_score , average='macro')



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    grid={"C":np.logspace(-30,30,1), "penalty":["l1","l2"]}# l1 lasso l2 ridge

    logreg=LogisticRegression()

    logreg_cv=GridSearchCV(logreg,grid,cv=10, scoring=f1)

    logreg_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)

print("F1_score : ",logreg_cv.best_score_)
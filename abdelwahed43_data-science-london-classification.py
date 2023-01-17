# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# read csv (comma separated value) into data

train = pd.read_csv('../input/data-science-london-scikit-learn/train.csv', header=None)

trainLabel = pd.read_csv('../input/data-science-london-scikit-learn/trainLabels.csv', header=None)

test = pd.read_csv('../input/data-science-london-scikit-learn/test.csv', header=None)

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
print('Data Description')

print('The shape of our training set: ',train.shape[0], 'rows ', 'and', train.shape[1]  , 'columns'  )

print('The shape of our testing set: ',trainLabel.shape[0], 'rows', 'and', trainLabel.shape[1], 'column')

print('The shape of our testing set: ',test.shape[0], 'rows', 'and', test.shape[1], 'columns')



print(train.columns.values)
# preview the data from head

train.head(3)
# split data train into Numeric and Categorocal

numeric = train.select_dtypes(exclude='object')

categorical = train.select_dtypes(include='object')
print("\nNumber of categorical features : ",(len(categorical.axes[1])))

print("\n", categorical.axes[1])

categorical.head()
##train.describe(include=['O'])
print("\nNumber of numeric features : ",(len(numeric.axes[1])))

print("\n", numeric.axes[1])
train.describe()
train.info()

print('_'*50)

test.info()
#missing data in Traing examples

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_data.head(3)
#missing data in Traing Label (target label) examples

total = trainLabel.isnull().sum().sort_values(ascending=False)

percent = (trainLabel.isnull().sum()/trainLabel.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_data.head(3)
#missing data in Test examples

total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



missing_data.head(3)
print(train.shape)

print(trainLabel.shape)

print(test.shape)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import LabelEncoder



# NAIBE BAYES

from sklearn.naive_bayes import GaussianNB

#KNN

from sklearn.neighbors import KNeighborsClassifier

#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

#SVM

from sklearn.svm import SVC

#DECISON TREE

from sklearn.tree import DecisionTreeClassifier

#XGBOOST

from xgboost import XGBClassifier

#AdaBoosting Classifier

from sklearn.ensemble import AdaBoostClassifier

#GradientBoosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

#HistGradientBoostingClassifier

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier



from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV









from sklearn.preprocessing import StandardScaler ,Normalizer , MinMaxScaler, RobustScaler 

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")
X_train,X_test,y_train,y_test = train_test_split(train,trainLabel,test_size=0.30, random_state=101)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
# Importing libraries

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.mixture import GaussianMixture

from sklearn.svm import SVC



X = np.r_[train,test]

print('X shape :',X.shape)

print('\n')



# USING THE GAUSSIAN MIXTURE MODEL 



#The Bayesian information criterion (BIC) can be used to select the number of components in a Gaussian Mixture in an efficient way. 

#In theory, it recovers the true number of components only in the asymptotic regime

lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)



#The GaussianMixture comes with different options to constrain the covariance of the difference classes estimated: 

# spherical, diagonal, tied or full covariance.

cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:

    for n_components in n_components_range:

        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)

        gmm.fit(X)

        bic.append(gmm.aic(X))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm

            

best_gmm.fit(X)

gmm_train = best_gmm.predict_proba(train)

gmm_test = best_gmm.predict_proba(test)
X_train,X_test,y_train,y_test = train_test_split(gmm_train,trainLabel,test_size=0.30, random_state=101)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
sk_fold = StratifiedKFold(10,shuffle=True, random_state=42)





X_train= X_train





X_train_1= pd.DataFrame(gmm_train).values



X_test= X_test



X_submit =  pd.DataFrame(gmm_test).values



g_nb = GaussianNB()

knn = KNeighborsClassifier()

ran_for  = RandomForestClassifier()

log_reg = LogisticRegression()

svc = SVC()

tree= DecisionTreeClassifier()

xgb = XGBClassifier()



ada_boost = AdaBoostClassifier()

grad_boost = GradientBoostingClassifier(n_estimators=100)

hist_grad_boost = HistGradientBoostingClassifier()









clf = [("Naive Bayes",g_nb,{}),\

       ("K Nearest",knn,{"n_neighbors":[3,5,6,7,8,9,10],"leaf_size":[25,30,35]}),\

       ("Random Forest",ran_for,{"n_estimators":[10, 50, 100, 200,400],"max_depth":[3, 10, 20, 40],"random_state":[99],"min_samples_leaf":[5,10,20,40,50],"bootstrap":[False]}),\

       ("Logistic Regression",log_reg,{"penalty":['l2'],"C":[100, 10, 1.0, 0.1, 0.01] , "solver":['saga']}),\

       ("Support Vector",svc,{"kernel": ["linear","rbf"],"gamma":[0.05,0.0001,0.01,0.001],"C":[0.1, 1, 10, 100, 1000]},),\

      

       ("Decision Tree", tree, {}),\

       ("XGBoost",xgb,{"n_estimators":[200],"max_depth":[3,4,5],"learning_rate":[.01,.1,.2],"subsample":[.8],"colsample_bytree":[1],"gamma":[0,1,5],"lambda":[.01,.1,1]}),\

       

       ("Adapative Boost",ada_boost,{"n_estimators":[100],"learning_rate":[.6,.8,1]}),\

       ("Gradient Boost",grad_boost,{}),\

     

       ("Histogram GB",hist_grad_boost,{"loss":["binary_crossentropy"],"min_samples_leaf":[5,10,20,40,50],"l2_regularization":[0,.1,1]})]





stack_list=[]

train_scores = pd.DataFrame(columns=["Name","Train Score","Test Score"])



i=0

for name,clf1,param_grid in clf:

    clf = GridSearchCV(clf1,param_grid=param_grid,scoring="accuracy",cv=sk_fold,return_train_score=True)

    clf.fit(X_train,y_train) #.reshape(-1,1)

    y_pred = clf.best_estimator_.predict(X_test)

    cm = confusion_matrix(y_test,y_pred)

    

    #train_scores.loc[i]= [name,cross_val_score(clf,X_train,y_train,cv=sk_fold,scoring="accuracy").mean(),(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]

    train_scores.loc[i]= [name,clf.best_score_,(cm[0,0]+cm[1,1,])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])]

    stack_list.append(clf.best_estimator_)

    i=i+1

    

est = [("g_nb",stack_list[0]),\

       ("knn",stack_list[1]),\

       ("ran_for",stack_list[2]),\

       ("log_reg",stack_list[3]),\

       ("svc",stack_list[4]),\

       ("dec_tree",stack_list[5]),\

       ("XGBoost",stack_list[6]),\

       ("ada_boost",stack_list[7]),\

       ("grad_boost",stack_list[8]),\

       ("hist_grad_boost",stack_list[9])]













sc = StackingClassifier(estimators=est,final_estimator = None,cv=sk_fold,passthrough=False)

sc.fit(X_train,y_train)

y_pred = sc.predict(X_test)

cm1 = confusion_matrix(y_test,y_pred)

y_pred_train = sc.predict(X_train)

cm2 = confusion_matrix(y_train,y_pred_train)

train_scores.append(pd.Series(["Stacking",(cm2[0,0]+cm2[1,1,])/(cm2[0,0]+cm2[0,1]+cm2[1,0]+cm2[1,1]),(cm1[0,0]+cm1[1,1,])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])],index=train_scores.columns),ignore_index=True)
print(X_train_1.shape)

print(trainLabel.shape)

print(X_submit.shape)
# Fitting our model

stack_list[3].fit(X_train_1,trainLabel)

y_submit = stack_list[3].predict(X_submit)



y_submit= pd.DataFrame(y_submit)

y_submit.index +=1







# FRAMING OUR SOLUTION

y_submit.columns = ['Solution']

y_submit['Id'] = np.arange(1,y_submit.shape[0]+1)

y_submit = y_submit[['Id', 'Solution']]







print(y_submit.shape)

print(y_submit.head(8))
y_submit.to_csv('Submission.csv',index=False)
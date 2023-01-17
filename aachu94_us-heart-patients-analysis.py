import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn import metrics

import statsmodels.api as sm

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report

from imblearn.combine import SMOTETomek
us=pd.read_csv("/kaggle/input/heart-patients/US_Heart_Patients.csv")

us.head()
us.info()
us.isnull().sum()
us=us.drop("education",axis=1)
us.shape
us.dropna(subset=['BPMeds'],inplace=True)
us.totChol.describe()
us["totChol"].fillna(us["totChol"].mean(),inplace=True)
us.BMI.describe()
us["BMI"].fillna(us["BMI"].mean(),inplace=True)
us.dropna(subset=['heartRate'],inplace=True)
us.glucose.describe()
us["glucose"].fillna(us["glucose"].mean(),inplace=True)
us.dropna(subset=['cigsPerDay'],inplace=True)
us.isnull().sum()
def draw_histograms(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')

        ax.set_title(feature+" Distribution",color='DarkRed')

        

    fig.tight_layout()  

    plt.show()

draw_histograms(us,us.columns,6,3)
us.TenYearCHD.value_counts().plot.bar()

plt.show()
sns.pairplot(data=us)

plt.show()
us.describe().T
X=us.drop(['TenYearCHD'],axis=1)

y=us['TenYearCHD']

print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtr=DecisionTreeClassifier(random_state=100,max_depth=4,min_samples_leaf=2)

dtr.fit(X_train,y_train)
gini_pred=dtr.predict(X_test)
gini_train=dtr.predict(X_train)
print('accuracy for train:',accuracy_score(y_train,gini_train))

print('accuracy for test:',accuracy_score(y_test,gini_pred))

print('difference between train and test:',accuracy_score(y_train,gini_train)-accuracy_score(y_test,gini_pred))

print('Classification Report:\n',classification_report(y_test,gini_pred))
entropy_dtr=DecisionTreeClassifier(criterion='entropy',splitter='best',random_state=100,max_depth=4,min_samples_leaf=1,min_samples_split=3)

entropy_dtr.fit(X_train,y_train)
entropy_pred=entropy_dtr.predict(X_test)
entropy_train=entropy_dtr.predict(X_train)
print('accuracy for train:',accuracy_score(y_train,entropy_train))

print('accuracy for test:',accuracy_score(y_test,entropy_pred))

print('difference between train and test:',accuracy_score(y_train,entropy_train)-accuracy_score(y_test,entropy_pred))

print('Classification Report:\n',classification_report(y_test,entropy_pred))
rfc = RandomForestClassifier(random_state=42)
import time

np.random.seed(42)

start = time.time()



param_dist = {'max_depth': [2, 3, 4],'bootstrap': [True, False],'max_features': ['auto', 'sqrt', 'log2', None],'criterion': ['gini', 'entropy']}



cv_rf = GridSearchCV(rfc, cv = 10,param_grid=param_dist, n_jobs = 3)



cv_rf.fit(X_train, y_train)

print('Best Parameters using grid search: \n', cv_rf.best_params_)

end = time.time()

print('Time taken in grid search: {0: .2f}'.format(end - start))
rfc_fit=rfc.set_params(criterion = 'gini',max_features = 'auto',max_depth = 3,bootstrap=True)
rfc_fitted=rfc.fit(X_train,y_train)
rfc_predict=rfc.predict(X_test)
print(classification_report(y_test,rfc_predict))
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
sklearn.metrics.accuracy_score(y_test,y_pred)
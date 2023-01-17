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
hp=pd.read_csv('/kaggle/input/heart-patients/US_Heart_Patients.csv')
hp.info()
hp.isnull().sum()*100/len(hp)
hp.shape
hp.drop('education',axis=1,inplace=True)
hp.dropna(subset=['cigsPerDay'],inplace=True)
hp.dropna(subset=['BPMeds'],inplace=True)
hp.totChol
hp.totChol.fillna(hp.totChol.mean(),inplace=True)
hp.glucose
hp.glucose.fillna(hp.glucose.mean(),inplace=True)
hp.dropna(subset=['heartRate'],inplace=True)
hp.BMI.describe()
hp.BMI.fillna(hp.BMI.mean(),inplace=True)
hp.isnull().sum()*100/len(hp)
import seaborn as sns

import matplotlib.pyplot as  plt
hp.TenYearCHD.value_counts().plot.barh(grid=True)

plt.show()
plt.figure(figsize=(20,8))

sns.heatmap(hp.corr(),annot=True)

plt.show()
plt.figure(figsize=(20,8))

sns.stripplot(data=hp,y='age',x='cigsPerDay',hue='TenYearCHD')

plt.show()
sns.scatterplot(data=hp,y='diaBP',x='sysBP',hue='TenYearCHD')

plt.show()
sns.scatterplot(data=hp,y='diaBP',x='sysBP',hue='prevalentHyp')

plt.show()
sns.barplot(data=hp,x='diabetes',y='glucose',hue='TenYearCHD')

plt.show()
sns.barplot(data=hp,x='currentSmoker',y='cigsPerDay',hue='TenYearCHD')

plt.show()
hp.shape
bmi=hp['BMI']

def cat_BMI(bmi):

    if float(bmi)<=18.5 :

        return 'Underweight'

    elif 18.5<float(bmi)<=25:

        return 'Normal'

    elif 25<float(bmi)<=30:

        return 'Overweight'

    else:

        return 'Obese'
hp['cat_BMI']=hp['BMI'].apply(cat_BMI)
hp['cat_BMI'].value_counts()*100/len(hp['cat_BMI'])
ct_bmi_tychd=pd.crosstab(hp['cat_BMI'],hp['TenYearCHD'])
ct_bmi_tychd.plot.bar()

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.feature_selection import RFE
X=hp.drop(['TenYearCHD','cat_BMI'],axis=1)

y=hp.TenYearCHD
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtr=DecisionTreeClassifier()

np.random.seed(42)



param_dist = {'criterion':['gini','entropy'],'min_samples_split':[2,3,4],'max_depth':[4,5,6],'min_samples_leaf':[1,2,3]}



cv_dtr = GridSearchCV(dtr, cv = 5,param_grid=param_dist, n_jobs = 3)



cv_dtr.fit(X_train, y_train)

print('Best Parameters using grid search: \n', cv_dtr.best_params_)
dtr=dtr.set_params(criterion= 'gini', max_depth= 4, min_samples_leaf= 1, min_samples_split= 3)
dtr_fit=dtr.fit(X_train,y_train)

dtr_predict=dtr.predict(X_test)

dtr_train_predict=dtr.predict(X_train)

print('accuracy score of train:',accuracy_score(y_train,dtr_train_predict))

print('accuracy score of test:',accuracy_score(y_test,dtr_predict))

print('classification report:\n',classification_report(y_test,dtr_predict))
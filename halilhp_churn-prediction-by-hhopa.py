# data analysis libraries:

import numpy as np

import pandas as pd



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier 

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

import numpy as np, pandas as pd, os, gc

from sklearn.model_selection import GroupKFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np,gc # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 500)

import numpy as np

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from warnings import filterwarnings

filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
# Read train and test data with pd.read_csv():

data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
data.head(10)
data.info()
data.nunique()

data.select_dtypes(include="object").nunique()
for i in data.select_dtypes(include="object"):

    print(data.select_dtypes(include="object")[i].value_counts())
data['Gender']=LabelEncoder().fit_transform(data['Gender'])

data['Surname']=LabelEncoder().fit_transform(data['Surname'])

dms= pd.get_dummies(data[['Geography']])



ndata=pd.concat([data, dms], axis=1)

ndata.head()
ndata.isnull().sum().any()
ndata.corrwith(data["Exited"], method="spearman")
ndata[["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]].describe().T
bins = [0, 25, 35, 50, 65, 95, np.inf]

mylabels = [ 'Young', 'Young_Adult','Adult', 'Senior', "Old", 'Death']

ndata['AgeGroup'] = pd.cut(ndata["Age"], bins, labels = mylabels)

ndata["AgeGroup"]=LabelEncoder().fit_transform(ndata["AgeGroup"])







ndata.head(30)

# Coef.of variance



columns=["CreditScore",

        "Age",

        "Tenure",

        "Balance",

        "EstimatedSalary"]



for i in columns:

    ndata["VC_"+i]=ndata[i].std()/ndata[i].mean()



ndata.head(10)
ndata.head()
plt.subplots(figsize=(20,15))

sns.heatmap(ndata.corr(), annot=True);
#In the mid tenure level there is less exit.



g= sns.factorplot(x = "Tenure", y = "Exited", data = ndata, kind = "bar", size = 7)

g.set_ylabels("Churn Probability")

plt.show()
g= sns.factorplot(x = "Gender", y = "Exited", data = ndata, kind = "bar", size = 5)

g.set_ylabels("Churn Probability")

plt.show()
g= sns.factorplot(x = "AgeGroup", y = "Exited", data = ndata, kind = "bar", size = 5)

g.set_ylabels("Churn Probability")

plt.show()
sns.barplot(x="Exited",y= "Balance", hue= 'Geography', data= ndata);
sns.barplot(x="IsActiveMember" ,y= "Exited" , hue= 'Geography',data=ndata) ;
sns.barplot(x="NumOfProducts" ,y="Exited"  ,  hue= 'Geography', data=ndata) ;
ndata["EstimatedSalary"].describe()
bins = [0, 50607,  99041,149383,np.inf]

mylabels = [ '1',  '2', '3',"4"]

ndata["EstimatedSalaryGroup"] = pd.cut(ndata["EstimatedSalary"], bins, labels = mylabels)

ndata.head()
ndata = ndata.drop(["RowNumber"],axis=1)

(sns

 .FacetGrid(ndata,

              hue = "Exited",

              height = 5,

              xlim = (10, 99))

 .map(sns.kdeplot, "Age", shade= True)

 .add_legend()

);
Ktrain, Ktest = train_test_split(ndata, test_size=0.30, random_state=4)

y_Ktest=Ktest["Exited"]

X_Ktest=Ktest.drop(["Exited"], axis=1)
Ktrain.head()
y=Ktrain['Exited']

X=Ktrain.drop(['Exited',"CustomerId","Surname", 'Geography'], axis=1).astype('float64')

X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.30, random_state=4)

def compML (ndata, y, algorithm):

    

    model=algorithm().fit(X_train, y_train)

    y_pred=model.predict(X_test)

    accuracy= accuracy_score(y_test, y_pred)

    #return accuracy

    model_name= algorithm.__name__

    print(model_name,": ",accuracy)
models = [LogisticRegression,

          KNeighborsClassifier,

          GaussianNB,

          SVC,

          DecisionTreeClassifier,

          RandomForestClassifier,

          GradientBoostingClassifier,

          LGBMClassifier,

          XGBClassifier,

          #CatBoostClassifier

         ]
for x in models:

    compML(ndata,"Exited",x)
clf=GradientBoostingClassifier().fit(X_train, y_train)

y_pred=clf.predict(X_test)

accuracy_score(y_test, y_pred)

#Model tuning

clf
GBM_params = {"loss":[ 'deviance', 'exponential'],

             "min_samples_split":[2,3,5],

             "n_estimators":[100,200,500],

             "min_samples_leaf":[1,2,5],

             }
GBM_cv_model = GridSearchCV(clf, 

                            GBM_params, 

                            cv=10, n_jobs=-1, 

                            verbose=2).fit(X_train, y_train)
GBM_cv_model.best_params_
clf_tuned = GradientBoostingClassifier(min_samples_split= 2,

                                        min_samples_leaf= 5,

                                       learning_rate= 0.1,

                                       max_depth= 3,

                                       n_estimators= 100,

                                       subsample= 1).fit(X_train, y_train)

y_pred=clf.predict(X_test)

accuracy_score(y_test, y_pred)
Importance = pd.DataFrame({'Importance':clf_tuned.feature_importances_*100},

                         index = X_train.columns)



Importance.sort_values(by = 'Importance',

                      axis = 0,

                      ascending = True).plot(kind = 'barh',

                                            color = '#d62728',

                                            figsize=(10,6), 

                                            edgecolor='white')

plt.xlabel('Variable Importance')

plt.gca().legend_ = None
#Ktest_Exited=Ktest["Exited"]

#Ktest=Ktest.drop(["Exited"], axis=1)

X_Ktest= X_Ktest.drop(["Geography","CustomerId","Surname"], axis=1).astype('float64')
predictions= clf.predict(X_Ktest)
real_test_y=pd.DataFrame(y_Ktest)

real_test_y["predictions"]=predictions



real_test_y.loc[:,"predictions"]=round(real_test_y.loc[:,"predictions"] ).astype(int)



real_test_y.head()
accuracy_score(real_test_y.loc[:,"Exited"],real_test_y.loc[:,"predictions"] )
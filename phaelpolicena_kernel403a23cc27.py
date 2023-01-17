# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.utils import check_array

from sklearn.metrics import mean_squared_error





from xgboost import XGBClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import linear_model

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import SGDClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')

ids = pd.read_csv('../input/sampleSubmission.csv')

test.head(5)
data.shape
data. describe()
data.info()
data.isnull().any()
data.loc[data['TotalCharges'].isna()] = 0
test.isnull().any()
test.loc[test['TotalCharges'].isna()] = 0
def correlate(data,t1,t2):

    numerical_column = ['int64','float64'] #select only numerical features to find correlation

    plt.figure(figsize=(t1,t2))

    sns.heatmap(

        data.select_dtypes(include=numerical_column).corr(),

        cmap=plt.cm.RdBu,

        vmax=1.0,

        linewidths=0.1,

        linecolor='white',

        square=True,

        annot=True

    )
correlate(data, 5,5)
X = data.drop(['customerID','Churn'], 1)

x = pd.get_dummies(X)

x = x.drop(['gender_0','Partner_0','Dependents_0','PaymentMethod_0','MultipleLines_0','PaperlessBilling_0','StreamingMovies_0','PhoneService_0','InternetService_0','DeviceProtection_0','TechSupport_0','OnlineSecurity_0','OnlineBackup_0'],1)



y = pd.get_dummies(data.Churn)

y = y.drop([0],1)



teste = test.drop(['customerID'], 1)



teste = pd.get_dummies(teste)

teste = teste.drop(['gender_0','Partner_0','Dependents_0','PaymentMethod_0','MultipleLines_0','PaperlessBilling_0','StreamingMovies_0','PhoneService_0','InternetService_0','DeviceProtection_0','TechSupport_0','OnlineSecurity_0','OnlineBackup_0'],1)

y.head(2)

test_np = np.array(teste)

X=np.array(x)

Y=np.array(y)
test_np.shape
x_Train,x_Test,y_Train,y_Test=train_test_split(X,Y,test_size=0.30)
scaler = StandardScaler()

sdsfds = scaler.fit_transform(x_Train)
def score(model,x,y):

    prob=model.predict_proba(x)

    prob = prob[:, 1]

    auc = roc_auc_score(y, prob)

    print('AUC: {}\nROC_AUC: {}\n {}'.format(model.score(x,y),auc,prob[:10]))
knn=KNeighborsClassifier(n_neighbors=3)

asa = knn.fit(x_Train,y_Train[:,1])

score(knn,x_Test,y_Test[:,1])
probs = knn.predict_proba(test_np)
xgb = XGBClassifier()

xgb.fit(x_Train,y_Train[:,1])

score(xgb,x_Test,y_Test[:,1])
prob=xgb.predict_proba(test_np)
clf = RandomForestClassifier(n_estimators=10)

clf.fit(x_Train,y_Train[:,1])

score(clf, x_Test,y_Test[:,1])
ada = AdaBoostClassifier(n_estimators=100, random_state=0)

ada.fit(x_Train,y_Train[:,1])

score(ada,x_Test,y_Test[:,1])
prob3 = ada.predict_proba(test_np)
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,

    max_depth=1, random_state=0).fit(x_Train, y_Train[:,1])

score(gbc, x_Test, y_Test[:,1])

rgl = LogisticRegression(random_state=0, solver='lbfgs',

                         multi_class='multinomial').fit(x_Train, y_Train[:,1])

score(rgl, x_Test, y_Test[:,1])
prob4 = rgl.predict_proba(test_np)
BC_model = BaggingClassifier().fit(x_Train, y_Train[:,1])

score(BC_model, x_Test, y_Test[:,1])
dtc = DecisionTreeClassifier(random_state=0).fit(x_Train, y_Train[:,1])

score(dtc, x_Test, y_Test[:,1])
etc = ExtraTreesClassifier(n_estimators=10, max_depth=None,

    min_samples_split=2, random_state=0).fit(x_Train, y_Train[:,1])

score(etc, x_Test, y_Test[:,1])
sgd = SGDClassifier(loss="log", penalty="elasticnet", max_iter=5).fit(x_Train, y_Train[:,1])

score(sgd, x_Test, y_Test[:,1])
submission = pd.DataFrame({

    "Id": ids.Id, 

    "Expected": probs[:,1]

})



submission.head()
submission.to_csv('sampleSubmission.csv', index=False)
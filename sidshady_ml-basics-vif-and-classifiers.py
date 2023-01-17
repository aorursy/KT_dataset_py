# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





df = pd.read_csv("../input/data.csv")

# Any results you write to the current directory are saved as output.
df.info()
list = ['Unnamed: 32','id']



df.drop(list,axis=1,inplace=True)
df.head()
X = df.drop("diagnosis",axis=1)

y = df['diagnosis']
diag_map = {"M":1,"B":0}



df['diagnosis'] = df['diagnosis'].map(diag_map)
df.head()
mean_feats = [i for i in df.columns if i.endswith("mean")]

se_feats = [i for i in df.columns if i.endswith("se")]

worst_feats = [i for i in df.columns if i.endswith("worst")]
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X[mean_feats] = scaler.fit_transform(X[mean_feats].as_matrix())
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import Imputer



from statsmodels.stats.outliers_influence import variance_inflation_factor



class ReduceVIF(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):

        # From looking at documentation, values between 5 and 10 are "okay".

        # Above 10 is too high and so should be removed.

        self.thresh = thresh

        

        # The statsmodel function will fail with NaN values, as such we have to impute them.

        # By default we impute using the median value.

        # This imputation could be taken out and added as part of an sklearn Pipeline.

        if impute:

            self.imputer = Imputer(strategy=impute_strategy)



    def fit(self, X, y=None):

        print('ReduceVIF fit')

        if hasattr(self, 'imputer'):

            self.imputer.fit(X)

        return self



    def transform(self, X, y=None):

        print('ReduceVIF transform')

        columns = X.columns.tolist()

        if hasattr(self, 'imputer'):

            X = pd.DataFrame(self.imputer.transform(X), columns=columns)

        return ReduceVIF.calculate_vif(X, self.thresh)



    @staticmethod

    def calculate_vif(X, thresh=5.0):

        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified

        dropped=True

        while dropped:

            variables = X.columns

            dropped = False

            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            

            max_vif = max(vif)

            if max_vif > thresh:

                maxloc = vif.index(max_vif)

                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')

                X = X.drop([X.columns.tolist()[maxloc]], axis=1)

                dropped=True

        return X

    

    def calculate_vif_2(X, thresh=5.0):

        

        variables = range(X.shape[1])

        dropped=True

        while dropped:

            dropped=False

            vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]



            maxloc = vif.index(max(vif))

            if max(vif) > thresh:

                print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))

                del variables[maxloc]

                dropped=True



        print('Remaining variables:')

        print(X.columns[variables])

        return X[variables]
transformer = ReduceVIF()
X = transformer.fit_transform(df[mean_feats], y)



X.head()
important_feats = []
important_feats.append('concavity_mean')

important_feats.append('symmetry_mean')
X = transformer.fit_transform(df[se_feats], y)

X.head()
important_feats.extend(['texture_se','area_se','concavity_se'])
X = transformer.fit_transform(df[worst_feats], y)

X.head()
important_feats.extend(['area_worst','concavity_worst','symmetry_worst'])
important_feats
X = df[important_feats]

y = df['diagnosis']
from sklearn.linear_model import LogisticRegression 



lmodel = LogisticRegression()
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)
lmodel.fit(X_train,y_train)
log_y_preds = lmodel.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report



print(confusion_matrix(y_test,log_y_preds))

print(classification_report(y_test,log_y_preds))
from sklearn.tree import DecisionTreeClassifier 



dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
dtree_y_preds = dtree.predict(X_test)
print(confusion_matrix(y_test,dtree_y_preds))

print(classification_report(y_test,dtree_y_preds))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators = 80)



rfc.fit(X_train,y_train)
rfc_y_preds = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_y_preds))

print(classification_report(y_test,rfc_y_preds))
from sklearn.svm import SVC 



svc_model = SVC()
svc_model.fit(X_train,y_train)

svc_y_preds = svc_model.predict(X_test)
print(confusion_matrix(y_test,svc_y_preds))

print(classification_report(y_test,svc_y_preds))
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 



grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train,y_train)
grid.best_params_
svc_model = SVC(C=1,gamma=0.001)
svc_model.fit(X_train,y_train)

svc_y_preds = svc_model.predict(X_test)

print(confusion_matrix(y_test,svc_y_preds))

print(classification_report(y_test,svc_y_preds))
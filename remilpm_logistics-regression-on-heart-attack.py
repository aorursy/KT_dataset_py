#Logistics regression on Heart Attack

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import Imputer



from statsmodels.stats.outliers_influence import variance_inflation_factor

import os

print(os.listdir("../input"))

Heart_Attack1=pd.read_csv("../input/HeartDisease.csv")

Heart_Attack1.head()

#=============================================================================================================

#Find the number of rows and columns in dataset

#=============================================================================================================

Heart_Attack1.shape
#=============================================================================================================

#Find the inavlid data

#=============================================================================================================

Heart_Attack1.isnull().sum()
#=============================================================================================================

#remove the inavlid data

#=============================================================================================================

Heart_Attack2=Heart_Attack1.dropna()

#=============================================================================================================

#Now check for inavlid data in new dataset

#=============================================================================================================

Heart_Attack2.isnull().sum()
#=============================================================================================================

#Find the number of rows and columns in dataset

#=============================================================================================================

Heart_Attack2.shape
#=============================================================================================================

#Logistics regression assumptions

#Model should have no multicollinearity, that means independent variables should not depend on each other

# Remove TenYearCHD, dependent variable as first step 

##=============================================================================================================

Heart_Attack3=Heart_Attack2.copy()

Heart_Attack3.pop('TenYearCHD')

Heart_Attack3.head()

from statsmodels.stats.outliers_influence import variance_inflation_factor



class ReduceVIF(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=5, impute=True, impute_strategy='median'):

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

#=============================================================================================================

# Remove columns having higher VIF factor ot having high multicollinearity

#=============================================================================================================

transformer = ReduceVIF()



# Only use 10 columns for speed in this example

Heart_Attack4 = transformer.fit_transform(Heart_Attack3)



Heart_Attack4.head()
#=============================================================================================================

# Relationship between gender and TenYearCHD

#=============================================================================================================

%matplotlib inline

pd.crosstab(Heart_Attack2.male,Heart_Attack2.TenYearCHD).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('Female                 Male')

plt.ylabel('Ten year CHD')

plt.savefig('Relationship')



#=============================================================================================================

# Male is a good predictor because it influences a great deal on TenYearCHD

#=============================================================================================================
#=============================================================================================================

# Relationship between education and TenYearCHD

#=============================================================================================================

pd.crosstab(Heart_Attack2.education,Heart_Attack2.TenYearCHD).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('education')

plt.ylabel('Ten year CHD')

plt.savefig('Relationship')



#=============================================================================================================

# education is a good predictor because it influences a great deal on TenYearCHD

#=============================================================================================================
#=============================================================================================================

# Relationship between currentSmoker and TenYearCHD

#=============================================================================================================

%matplotlib inline

pd.crosstab(Heart_Attack2.currentSmoker,Heart_Attack2.TenYearCHD).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('currentSmoker')

plt.ylabel('Ten year CHD')

plt.savefig('Relationship')



#=============================================================================================================

# currentSmoker is  a good predictor

#=============================================================================================================
#=============================================================================================================

# Relationship between cigsPerDay and TenYearCHD

#=============================================================================================================

%matplotlib inline

pd.crosstab(Heart_Attack2.cigsPerDay,Heart_Attack2.TenYearCHD).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('cigsPerDay')

plt.ylabel('Ten year CHD')

plt.savefig('Relationship')



#=============================================================================================================

# cigsPerDay is not a  good predictor 

#=============================================================================================================
#=============================================================================================================

# Relationship between BPMeds and TenYearCHD

#=============================================================================================================

pd.crosstab(Heart_Attack2.BPMeds,Heart_Attack2.TenYearCHD).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('BPMeds')

plt.ylabel('Ten year CHD')

plt.savefig('Relationship')



#=============================================================================================================

# BPMeds is a  good predictor 

#=============================================================================================================
#=============================================================================================================

# Relationship between prevalentStroke and TenYearCHD

#=============================================================================================================

pd.crosstab(Heart_Attack2.prevalentStroke,Heart_Attack2.TenYearCHD).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('prevalentStroke')

plt.ylabel('Ten year CHD')

plt.savefig('Relationship')



#=============================================================================================================

# prevalentStroke is  a good predictor 

#=============================================================================================================
#=============================================================================================================

# Relationship between prevalentHyp and TenYearCHD

#=============================================================================================================

pd.crosstab(Heart_Attack2.prevalentHyp,Heart_Attack2.TenYearCHD).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('prevalentHyp')

plt.ylabel('Ten year CHD')

plt.savefig('Relationship')



#=============================================================================================================

# prevalentHyp is a good predictor because it influences a great deal on TenYearCHD

#=============================================================================================================
#=============================================================================================================

# Relationship between diabetes and TenYearCHD

#=============================================================================================================

pd.crosstab(Heart_Attack2.diabetes,Heart_Attack2.TenYearCHD).plot(kind='bar')

plt.title('Relationship')

plt.xlabel('diabetes                 ')

plt.ylabel('Ten year CHD')

plt.savefig('Relationship')



#=============================================================================================================

# diabetes is  a good predictor 

#=============================================================================================================
X = Heart_Attack2.as_matrix(['male','education','currentSmoker','BPMeds','prevalentStroke','prevalentHyp','diabetes'])

y = Heart_Attack2['TenYearCHD']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)

       #lets scale the data

X = Heart_Attack2.as_matrix(['male','education','currentSmoker','BPMeds','prevalentStroke','prevalentHyp','diabetes'])

y = Heart_Attack2['TenYearCHD']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=10)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train= sc.transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

#Create XGBoost model

Heart_Attack_Pred_Model = XGBRegressor()

#fit the model

Heart_Attack_Pred_Model.fit(X_train, y_train)

#make predictions

predictions = Heart_Attack_Pred_Model.predict(X_test)

# Calculate MAE

print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, y_test)))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

    

from sklearn.preprocessing import OneHotEncoder

%matplotlib inline
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

data.head()
data['Churn'].value_counts()
data.shape
data.dtypes
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.isnull().sum()
categorical_var = list(data.dtypes.loc[data.dtypes == 'object'].index)

print(len(categorical_var))

print(categorical_var)
categorical_var.remove('customerID')
fig, ax =plt.subplots(6,3,figsize=(12,20))





sns.countplot(data['gender'], ax=ax[0][0])

sns.countplot(data['Partner'], ax=ax[0][1])

sns.countplot(data['Dependents'], ax=ax[0][2])



sns.countplot(data['PhoneService'], ax=ax[1][0])

sns.countplot(data['MultipleLines'], ax=ax[1][1])

sns.countplot(data['InternetService'], ax=ax[1][2])



sns.countplot(data['OnlineSecurity'], ax=ax[2][0])

sns.countplot(data['OnlineBackup'], ax=ax[2][1])

sns.countplot(data['DeviceProtection'], ax=ax[2][2])



sns.countplot(data['TechSupport'], ax=ax[3][0])

sns.countplot(data['StreamingTV'], ax=ax[3][1])

sns.countplot(data['StreamingMovies'], ax=ax[3][2])



sns.countplot(data['Contract'], ax=ax[4][0])

sns.countplot(data['PaperlessBilling'], ax=ax[4][1])

sns.countplot(data['PaymentMethod'], ax=ax[4][2])



sns.countplot(data['Churn'], ax=ax[5][0])



fig.show()
continuous_var = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']

data.describe()
nd = pd.melt(data, value_vars = continuous_var)

n1 = sns.FacetGrid (nd, col='variable', col_wrap=2, sharex=False, sharey = False)

n1 = n1.map(sns.distplot, 'value')

n1
corr = data[continuous_var].corr()

sns.heatmap(corr)
print (corr['TotalCharges'].sort_values(ascending=False), '\n') 
sns.jointplot(x=data['TotalCharges'], y=data['tenure'])
for var in categorical_var:

    if var!='Churn':

        test = data.groupby([var,'Churn'])

        print(test.size(),'\n\n')
import scipy.stats as stats

from scipy.stats import chi2_contingency



class ChiSquare:

    def __init__(self, dataframe):

        self.df = dataframe

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)



df = data

#Initialize ChiSquare Class

cT = ChiSquare(df)



#Feature Selection

for var in categorical_var:

    cT.TestIndependence(colX=var,colY="Churn" ) 
# ANOVA test

import scipy.stats as stats

    

for var in continuous_var:    

    result = stats.f_oneway(data[var][data['Churn'] == 'Yes'], 

                            data[var][data['Churn'] == 'No'])

    print(var)

    print(result)
from sklearn.feature_selection import SelectKBest

from scipy.stats import ttest_ind



t_stat = []

for var in continuous_var:

    var_no_churn = data[var][data["Churn"] == "No"]

    var_yes_churn = data[var][data["Churn"] == "Yes"]

    t_value = ttest_ind(var_no_churn, var_yes_churn, equal_var=False)

    print(var)

    print(t_value)

    #t_stat.append(t_value)
data.isnull().sum()
data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
categorical_var
#first convert all the string columns to categorical form

for var in categorical_var:

    data[var] = data[var].astype('category')
data[categorical_var] = data[categorical_var].apply(lambda x: x.cat.codes)
target = data['Churn']

data=data.drop('customerID',axis=1)

all_columns = list(data.columns)

all_columns.remove('Churn')
import warnings

warnings.filterwarnings('ignore')

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



X = data[all_columns] # Features

y = data['Churn'] # Target variable



# Feature extraction

model = LogisticRegression()

rfe = RFE(model, 8)

fit = rfe.fit(X, y)

print("Num Features: %s" % (fit.n_features_))

print("Selected Features: %s" % (fit.support_))

print("Feature Ranking: %s" % (fit.ranking_))
selected_features_rfe = list(fit.support_)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics



final_features_rfe = []    

for status, var in zip(selected_features_rfe, all_columns):

    if status == True:

        final_features_rfe.append(var)

        

final_features_rfe
X_rfe_lr = data[final_features_rfe]

y = data['Churn']



X_train_rfe_lr,X_test_rfe_lr,y_train_rfe_lr,y_test_rfe_lr=train_test_split(X_rfe_lr,y,test_size=0.25,random_state=0)



lr_model = LogisticRegression()



# fit the model with data

lr_model.fit(X_train_rfe_lr,y_train_rfe_lr)

y_pred_rfe_lr=lr_model.predict(X_test_rfe_lr)



acc_rfe_lr = metrics.accuracy_score(y_test_rfe_lr, y_pred_rfe_lr)

print("Accuracy: ",acc_rfe_lr)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# instantiate the model (using the default parameters)

lr_model_single = LogisticRegression()



# fit the model with data

lr_model_single.fit(X_train,y_train)

y_pred=lr_model_single.predict(X_test)



lr_acc = metrics.accuracy_score(y_test, y_pred)

print("Accuracy: ",lr_acc)
# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
from sklearn.metrics import roc_curve, auc

fpr_1, tpr_1, thresholds = roc_curve(y_test, y_pred_rfe_lr)

fpr_2, tpr_2, thresholds = roc_curve(y_test, y_pred)

roc_auc_1 = auc(fpr_1, tpr_1)

roc_auc_2 = auc(fpr_2, tpr_2)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(fpr_1,tpr_1, color='red',label = 'AUC = %0.2f' % roc_auc_1)

plt.plot(fpr_2,tpr_2, color='green',label = 'AUC = %0.2f' % roc_auc_2)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
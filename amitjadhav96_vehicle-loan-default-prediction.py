import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

dftrain = pd.read_csv('../input/lt-vehicle-loan-default-prediction/train.csv', parse_dates = ["Date.of.Birth","DisbursalDate"])
dftest = pd.read_csv('../input/lt-vehicle-loan-default-prediction/test.csv', parse_dates = ["Date.of.Birth","DisbursalDate"])

#Check train test 
print(dftrain.shape)
print(dftest.shape)

dfinfo = pd.read_csv('../input/lt-vehicle-loan-default-prediction/data_dictionary.csv')

dfinfo
dftrain.head(5)
dftrain.dtypes
dftrain.describe()
import pandas_profiling
pf1=pandas_profiling.ProfileReport(dftrain)

pf1
dftrain["PERFORM_CNS.SCORE.DESCRIPTION"].value_counts()
def risk_level(risk):
    list1 = risk.split("-")
    if len(list1)==1:
        return "Risk_Unavaliable"
    else:
        return list1[1]

dftrain["Risk_level"] = dftrain["PERFORM_CNS.SCORE.DESCRIPTION"].apply(lambda x: risk_level(x))
dftest["Risk_level"] = dftest["PERFORM_CNS.SCORE.DESCRIPTION"].apply(lambda x: risk_level(x))
dftrain["Risk_level"][:5]
from datetime import date

def AgeinYears(date,df):
    disbDate = df["DisbursalDate"][i]
    return disbDate.year - date.year - ((disbDate.month, disbDate.day) < (date.month, date.day))
AgeinYrsTrain=[]
for i in range(len(dftrain)):
    AgeinYrsTrain.append(AgeinYears(dftrain["Date.of.Birth"][i],dftrain))
AgeinYrsTest=[]
for i in range(len(dftest)):
    AgeinYrsTest.append(AgeinYears(dftest["Date.of.Birth"][i],dftest))
    
dftrain["AgeinYrs"]=AgeinYrsTrain
dftest["AgeInYears"]=AgeinYrsTest
dftrain.drop(["UniqueID","Employee_code_ID"], axis=1, inplace=True)
dftest.drop(["UniqueID","Employee_code_ID"], axis=1, inplace=True)
def duration(duration):
    list1=duration.split(" ")
    sumyrs = float(list1[0][:-3]) + float(list1[1][:-3])/12
    return round(sumyrs,2)
dftrain["AVERAGE.ACCT.AGE_Years"] = dftrain["AVERAGE.ACCT.AGE"].apply(lambda x: duration(x))
dftest["AVERAGE.ACCT.AGE_Years"] = dftest["AVERAGE.ACCT.AGE"].apply(lambda x: duration(x))
dftrain["CREDIT.HISTORY.LENGTH_Years"] = dftrain["CREDIT.HISTORY.LENGTH"].apply(lambda x: duration(x))
dftest["CREDIT.HISTORY.LENGTH_Years"] = dftest["CREDIT.HISTORY.LENGTH"].apply(lambda x: duration(x))
dftrain["CREDIT.HISTORY.LENGTH_Years"][:5]
dftrain.drop(["AVERAGE.ACCT.AGE","Aadhar_flag","Current_pincode_ID","Date.of.Birth","DisbursalDate","Driving_flag","MobileNo_Avl_Flag","PAN_flag","PRI.DISBURSED.AMOUNT","SEC.ACTIVE.ACCTS","Passport_flag","supplier_id","branch_id","VoterID_flag","State_ID","SEC.SANCTIONED.AMOUNT","SEC.OVERDUE.ACCTS","SEC.NO.OF.ACCTS","SEC.INSTAL.AMT","SEC.DISBURSED.AMOUNT","SEC.CURRENT.BALANCE"], axis=1, inplace=True)
dftest.drop(["AVERAGE.ACCT.AGE","Aadhar_flag","Current_pincode_ID","Date.of.Birth","DisbursalDate","Driving_flag","MobileNo_Avl_Flag","PAN_flag","PRI.DISBURSED.AMOUNT","SEC.ACTIVE.ACCTS","Passport_flag","supplier_id","branch_id","VoterID_flag","State_ID","SEC.SANCTIONED.AMOUNT","SEC.OVERDUE.ACCTS","SEC.NO.OF.ACCTS","SEC.INSTAL.AMT","SEC.DISBURSED.AMOUNT","SEC.CURRENT.BALANCE"], axis=1, inplace=True)
dftrain.drop(["PERFORM_CNS.SCORE.DESCRIPTION","CREDIT.HISTORY.LENGTH"], axis=1, inplace=True)
dftest.drop(["PERFORM_CNS.SCORE.DESCRIPTION","CREDIT.HISTORY.LENGTH"], axis=1, inplace=True)
dftrain.head(10)
dftrain.isnull().sum()
#Replacing negative age with positive age. Assume typing error
dftrain["AgeinYrs"][dftrain["AgeinYrs"]<0]=-dftrain["AgeinYrs"]
dftrain["AgeinYrs"].describe()
dftrain.dropna(subset = ["Employment.Type"], inplace=True)
dftest.dropna(subset = ["Employment.Type"], inplace=True)
dftrain.isnull().sum()
pf2=pandas_profiling.ProfileReport(dftrain)

pf2
dftrain["manufacturer_id"].value_counts()
dftrain.drop(dftrain[dftrain.manufacturer_id==156].index, inplace=True)
dftest["manufacturer_id"].value_counts()
dftest.drop(dftest[dftest.manufacturer_id==155].index, inplace=True)
dftest["manufacturer_id"].value_counts()
dftrain_onehot1 = pd.get_dummies(dftrain, columns=['manufacturer_id',"Employment.Type","Risk_level"], prefix = ['MID_',"ET_","RL_"],drop_first=True)
dftest_onehot1 = pd.get_dummies(dftest, columns=['manufacturer_id',"Employment.Type","Risk_level"], prefix = ['MID_',"ET_","RL_"],drop_first=True)
dftrain_onehot1.head()
#Reference Dummy: ManufacturerID-45,EmployeeType: Salaried, Risk_Type, Risk_level: High Risk 
dftest_onehot1.head()
X= dftrain_onehot1.loc[:, dftrain_onehot1.columns != 'loan_default']
Y= dftrain_onehot1["loan_default"]
X.columns
variables = ['disbursed_amount', 'asset_cost', 'ltv', 'PERFORM_CNS.SCORE',
       'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS',
       'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'PRIMARY.INSTAL.AMT',
       'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
       'NO.OF_INQUIRIES', 'AgeinYrs', 'AVERAGE.ACCT.AGE_Years',
       'CREDIT.HISTORY.LENGTH_Years']
X.head()
Y.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training= X.copy()
X.head()
scaled_training[variables] = scaler.fit_transform(scaled_training[variables])
scaled_training.head(5)
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
X_resampled, y_resampled = SMOTE().fit_resample(scaled_training, Y)
print(sorted(Counter(y_resampled).items()))
X_resampled_ADS, y_resampled_ADS = ADASYN().fit_resample(scaled_training, Y)
print(sorted(Counter(y_resampled).items()))
X_resampled_ADS
y_resampled_ADS
X_resampled
x_train, x_test, y_train, y_test = train_test_split(X_resampled_ADS, y_resampled_ADS, test_size=0.30, random_state=42)
x_train.head()
Counter(y_train)
Counter(y_test)
log1 = LogisticRegression(penalty='l1',solver="liblinear", max_iter=1000).fit(x_train,y_train)
log1.score(x_train, y_train)
predTrain = log1.predict(x_train)
predTrain[:5]
PredictionsTrain = pd.DataFrame(columns=["Prediction","Actual"])
PredictionsTrain["Prediction"]=predTrain
PredictionsTrain["Actual"] = y_train.tolist()

PredictionsTrain.head()
log1.score(x_test, y_test)
predTest = log1.predict(x_test)

predTest
PredictionsTest = pd.DataFrame(columns=["Prediction","Actual"])
PredictionsTest["Prediction"]=predTest
PredictionsTest["Actual"] = y_test.tolist()

PredictionsTest.head()
CoeffLogR = pd.DataFrame(columns=["Variable","Coefficients"])
CoeffLogR["Variable"]=X.columns
CoeffLogR["Coefficients"]=log1.coef_.tolist()[0]
CoeffLogR.sort_values("Coefficients", ascending = False)

# confusion matrix
print('Confusion Matrix')
print(pd.DataFrame(confusion_matrix(y_test, predTest)))
print('Accuracy',accuracy_score(y_test, predTest))
print('Recall',recall_score(y_test, predTest))
print('F1_score',f1_score(y_test, predTest))
print('ROC-AUC_score',roc_auc_score(y_test, predTest))

regrRM = RandomForestClassifier(n_estimators=300)
regrRM.fit(x_train, y_train)
regrRM.score(x_train, y_train)
regrRM.score(x_test, y_test)
predTestRF = regrRM.predict(x_test)
predTestRF
print('Accuracy',accuracy_score(y_test, predTestRF))
print('Recall',recall_score(y_test, predTestRF))
print('F1_score',f1_score(y_test, predTestRF))
print('ROC-AUC_score',roc_auc_score(y_test, predTestRF))

# Confusion matrix
print('Confusion Matrix')
print(pd.DataFrame(confusion_matrix(y_test, predTestRF)))

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
# Create the random grid
rm_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf2 = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
rf2_random = RandomizedSearchCV(estimator = rf2, param_distributions = rm_grid, n_iter = 2, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf2_random.fit(x_train, y_train)
rf2_random.best_params_
rf2_random.score(x_train, y_train)
rf2_random.score(x_test, y_test)
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(n_estimators=300,learning_rate=0.1, random_state=42)
adb.fit(x_train, y_train)
adb.score(x_train, y_train)
adb.score(x_test, y_test)
predTestadb = adb.predict(x_test)
predTestadb
print('Accuracy',accuracy_score(y_test, predTestadb))
print('Recall',recall_score(y_test, predTestadb))
print('F1_score',f1_score(y_test, predTestadb))
print('ROC-AUC_score',roc_auc_score(y_test, predTestadb))
print('Confusion Matrix')
print(pd.DataFrame(confusion_matrix(y_test, predTestadb)))
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=200)
gbt.fit(x_train, y_train)
gbt.score(x_train, y_train)
gbt.score(x_test, y_test)
predTestgbt = gbt.predict(x_test)
predTestgbt
print('Accuracy',accuracy_score(y_test, predTestgbt))
print('Recall',recall_score(y_test, predTestgbt))
print('F1_score',f1_score(y_test, predTestgbt))
print('ROC-AUC_score',roc_auc_score(y_test, predTestgbt))

##Fitting to Test data provided
scaled_testing=dftest_onehot1.copy()
scaled_training.head()
scaled_testing.head()
scaled_testing=scaled_testing.rename(columns={"AgeInYears": "AgeinYrs"})
scaled_testing[variables] = scaler.fit_transform(scaled_testing[variables])
pred = rf2_random.predict(scaled_testing)
pred

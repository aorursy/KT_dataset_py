import pandas as pd
import numpy as np
import re
from datetime import datetime, date

from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

train_df = pd.read_csv('../input/Project Train Dataset_single_quote.csv').drop(['CUST_COD'], axis=1)

# NOTE: in the original project all the manipulations made on data were replicated on the test dataset
# to give them the same format. Being useless in this notebook to produce a submission file all the manipulations 
# performed on that second dataset are commented.
#test_df = pd.read_csv('Project Test Dataset.csv',sep=';').drop(['CUST_COD'], axis=1)

train_df.dropna(thresh=(len(train_df.columns)-1),inplace=True)
def getClassFromSEX(SEX):
    if str(SEX) == "M":
        return 0
    elif str(SEX) == "F":
        return 1
    else:
        return np.nan
    
train_df['SEX'] = train_df['SEX'].apply(getClassFromSEX)

#test_df['SEX'] = test_df['SEX'].apply(getClassFromSEX)
def getClassFromMARRIAGE (MARRIAGE):
    if str(MARRIAGE) == "married":
        return 2
    elif str(MARRIAGE) == "single":
        return 1
    elif str(MARRIAGE) == "other":
        return 0
    else:
        return np.nan
    
train_df['MARRIAGE'] = train_df['MARRIAGE'].apply(getClassFromMARRIAGE)

#test_df['MARRIAGE'] = test_df['MARRIAGE'].apply(getClassFromMARRIAGE)
def getClassFromEDUCATION(EDUCATION):
    if str(EDUCATION) == "graduate school":
        return 3
    elif str(EDUCATION) == "university":
        return 2
    elif str(EDUCATION) == "high school":
        return 1
    elif str(EDUCATION) == 'other':
        return 0
    else:
        return np.nan
    
train_df['EDUCATION'] = train_df['EDUCATION'].apply(getClassFromEDUCATION)

#test_df['EDUCATION'] = test_df['EDUCATION'].apply(getClassFromEDUCATION)
def getAgeFromTrainBIRTH_DATE(BIRTH_DATE):
    today = date.today()
    if str(BIRTH_DATE) != "nan":
        pattern = re.compile("([0-9]+)/([0-9]+)/([0-9]+)")
        if pattern.match(BIRTH_DATE):
            born = datetime.strptime(BIRTH_DATE, "%d/%m/%Y")
        else:
            born = datetime.strptime(BIRTH_DATE, "%Y-%m-%dT%H:%M:%S")
    else:
        return np.nan
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

train_df['AGE'] = train_df['BIRTH_DATE'].apply(getAgeFromTrainBIRTH_DATE)
#test_df['AGE'] = test_df['BIRTH_DATE'].apply(getAgeFromTrainBIRTH_DATE)

train_df = train_df.drop(['BIRTH_DATE'],axis=1)
#test_df = test_df.drop(['BIRTH_DATE'],axis=1)
months = ['JUL','AUG','SEP','OCT','NOV','DEC']
bill_amts = ['BILL_AMT_' + month for month in months]
pay_amts = ['PAY_AMT_' + month for month in months]
pays  = ['PAY_' + month for month in months]
def getCardUsage(BILL_ROW):
    sum = 0
    for bill_month in bill_amts:
        if BILL_ROW[bill_month] != 0:
            sum+=1
    return sum

train_df['CARD_USAGE'] = train_df[bill_amts].apply(getCardUsage,axis=1)
#test_df['CARD_USAGE'] = test_df[bill_amts].apply(getCardUsage,axis=1)
def getPaymentHabits(PAY_ROW):
    sum = 0
    for pay_month in pays:
        if PAY_ROW[pay_month] > 0:
            sum+=1
    return sum

train_df['PAY_HABIT'] = train_df[pays].apply(getPaymentHabits,axis=1)
#test_df['PAY_HABIT'] = test_df[pays].apply(getPaymentHabits,axis=1)

def getBillFeature(DATAFRAME):
    regr = LinearRegression()
    x = np.array(range(len(bill_amts))).reshape(-1,1)
    bill_trends = []
    bill_stds = []
    for index, row in DATAFRAME.iterrows():
        row_bill_amts = np.array(row[bill_amts],dtype=float)
        # BILLs std
        bill_stds.append(np.std(row_bill_amts))
        # BILLs trend
        row_bill_amts = row_bill_amts.reshape(-1,1)
        regr.fit(x,row_bill_amts)
        bill_trends.append(regr.coef_[0][0])
    return bill_trends, bill_stds

bill_trends, bill_stds = getBillFeature(train_df)
train_df['BILL_TREND'] = bill_trends
train_df['BILL_STD'] = bill_stds

#bill_trends, bill_stds = getBillFeature(test_df)
#test_df['BILL_TREND'] = bill_trends
#test_df['BILL_STD'] = bill_stds

def maxDelta(BILL_ROW):
    max_delta=BILL_ROW[bill_amts[1]]-BILL_ROW[bill_amts[0]]
    for i in range(1,len(bill_amts)-1):
        if(BILL_ROW[bill_amts[i+1]]-BILL_ROW[bill_amts[i]]>max_delta):
            max_delta=BILL_ROW[bill_amts[i+1]]-BILL_ROW[bill_amts[i]]
    return max_delta
        
train_df['MAX_DELTA'] = train_df[bill_amts].apply(maxDelta,axis=1)
#test_df['MAX_DELTA'] = test_df[bill_amts].apply(maxDelta,axis=1)

train_df['NORMALIZED_DELTA']=train_df['MAX_DELTA']/train_df['LIMIT_BAL']
#test_df['NORMALIZED_DELTA']=test_df['MAX_DELTA']/test_df['LIMIT_BAL']
stat_df = train_df[['SEX','LIMIT_BAL','AGE','MARRIAGE','EDUCATION']].copy()
stat_df.dropna(thresh=(len(stat_df.columns)),inplace=True)
imputed_df = train_df.copy()
regr = LinearRegression()
train_age_indeces = ['SEX','LIMIT_BAL','MARRIAGE','EDUCATION']
train_age = np.array(stat_df[train_age_indeces])
x = train_age.reshape(-1, len(train_age_indeces))
target_age = np.array(stat_df['AGE'])
y = target_age.reshape(-1, 1)
regr.fit(train_age,target_age)
for index,row in imputed_df[imputed_df['AGE'].isnull()].iterrows():
    prediction = regr.predict(np.array(imputed_df.loc[index,train_age_indeces]).reshape(-1,len(train_age_indeces)))
    imputed_df.loc[index,'AGE'] = prediction
train_marriage_indeces = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'AGE']
train_marriage = stat_df[train_marriage_indeces]
target_marriage = stat_df['MARRIAGE']

dt = LogisticRegression()
dt.fit(train_marriage,target_marriage)

for index,row in imputed_df[imputed_df['MARRIAGE'].isnull()].iterrows():
    x = imputed_df.loc[index,train_marriage_indeces]
    prediction = dt.predict(np.array(x).reshape(-1,len(x)))[0]
    imputed_df.loc[index,'MARRIAGE'] = prediction
train_education_indeces = ['SEX','LIMIT_BAL','AGE','MARRIAGE']
train_education = stat_df[train_education_indeces]
target_education = stat_df['EDUCATION']
dt.fit(train_education,target_education)


for index,row in imputed_df[imputed_df['EDUCATION'].isnull()].iterrows():
    x = imputed_df.loc[index,train_education_indeces]
    prediction = dt.predict(np.array(x).reshape(-1,len(x)))[0]
    imputed_df.loc[index,'EDUCATION'] = prediction
train_sex_indeces = ['EDUCATION','LIMIT_BAL','AGE','MARRIAGE']
train_sex = stat_df[train_sex_indeces]
target_sex = stat_df['SEX']
dt.fit(train_sex,target_sex)

for index,row in imputed_df[imputed_df['SEX'].isnull()].iterrows():
    x = imputed_df.loc[index,train_sex_indeces]
    prediction = dt.predict(np.array(x).reshape(-1,len(x)))[0]
    imputed_df.loc[index,'SEX'] = prediction
selected_best_features = ['PAY_HABIT', 'PAY_DEC','NORMALIZED_DELTA', 'BILL_AMT_DEC',
                          'EDUCATION','PAY_AMT_DEC', 'PAY_NOV','PAY_AMT_NOV','CARD_USAGE']
imputed_df = (imputed_df - imputed_df.min()) / (imputed_df.max() - imputed_df.min())
xg = XGBClassifier(learning_rate=0.2 ,max_depth = 4, n_estimators = 61,objective="rank:pairwise",base_score=0.25)
scores = cross_val_score(xg, train_df.drop(['DEFAULT PAYMENT JAN'],axis=1), train_df["DEFAULT PAYMENT JAN"], 
                         cv = 10, scoring="f1", n_jobs=-1)                                         
print("XGBoost F1: %.4f" % (np.average(scores)))
print("Std dev: "+ str(np.std(scores)))
scores = cross_val_score(xg, imputed_df.drop(['DEFAULT PAYMENT JAN'],axis=1), imputed_df["DEFAULT PAYMENT JAN"], 
                         cv = 10, scoring="f1", n_jobs=-1)                                         
print("XGBoost F1: %.4f" % (np.average(scores)))
print("Std dev: "+ str(np.std(scores)))
log_reg = LogisticRegression()
scores = cross_val_score(log_reg, imputed_df.drop(['DEFAULT PAYMENT JAN'],axis=1), imputed_df["DEFAULT PAYMENT JAN"], 
                         cv = 10, scoring="f1", n_jobs=-1)                                         
print("Logistic Regression F1: %.4f" % (np.average(scores)))
print("Std dev: "+ str(np.std(scores)))

knn = KNeighborsClassifier(n_neighbors=19)
scores = cross_val_score(knn, imputed_df[selected_best_features], imputed_df["DEFAULT PAYMENT JAN"], 
                         cv = 10, scoring="f1", n_jobs=-1)                                         
print("KNN F1: %.4f" % (np.average(scores)))
print("Std dev: "+ str(np.std(scores)))
nb = GaussianNB()
scores = cross_val_score(nb, imputed_df[selected_best_features], imputed_df["DEFAULT PAYMENT JAN"], 
                         cv = 10, scoring="f1", n_jobs=-1)                                         
print("Naive Bayes F1: %.4f" % (np.average(scores)))
print("Std dev: "+ str(np.std(scores)))
lsvm = SVC(kernel="linear")
scores = cross_val_score(lsvm, imputed_df[selected_best_features], imputed_df["DEFAULT PAYMENT JAN"], 
                         cv = 10, scoring="f1", n_jobs=-1)  
print("SVM linear F1: %.4f" % (np.average(scores)))
print("Std dev: "+ str(np.std(scores)))
nn = MLPClassifier(solver="lbfgs", activation="tanh", alpha=0.001, hidden_layer_sizes=(10,))
scores = cross_val_score(nn, imputed_df.drop(['DEFAULT PAYMENT JAN'],axis=1), imputed_df["DEFAULT PAYMENT JAN"], 
                         cv = 10, scoring="f1", n_jobs=-1)                                         
print("Neural Network F1: %.4f" % (np.average(scores)))
print("Std dev: "+ str(np.std(scores)))
xg.fit(train_df.drop(['DEFAULT PAYMENT JAN'],axis=1), train_df["DEFAULT PAYMENT JAN"])

print("The model has been trained.")
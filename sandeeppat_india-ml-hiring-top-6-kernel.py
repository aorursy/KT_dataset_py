# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

import os

from sklearn.metrics import f1_score

from bayes_opt import BayesianOptimization

from lightgbm import LGBMClassifier

from lightgbm import *

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

pd.set_option('display.max.rows',5000)

pd.set_option('display.max.columns',5000)



# Any results you write to the current directory are saved as output.
os.listdir('../input/')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.drop(columns=['loan_id'],inplace=True)

test.drop(columns=['loan_id'],inplace=True)
train.isnull().sum()
test.isnull().sum()
wedge = [train['m13'].value_counts()[0],train['m13'].value_counts()[1]]



perc = [train['m13'].value_counts()[0]/len(train),

        train['m13'].value_counts()[1]/len(train)

       ]

plt.pie(wedge,labels=['Normal - '+ format(perc[0]*100, '.2f') + '%','Anomaly - '],

        shadow=True,radius = 2.0)
def origin_dateYear(x):

    year = x.split('-')[0]

    return year



def origin_dateMonth(x):

    month = x.split('-')[1]

    return month



def origin_dateDay(x):

    day = x.split('-')[2]

    return day



def first_paymentMonth(x):

    month = x.split('/')[0]

    return month



def first_paymentYear(x):

    year = x.split('/')[1]

    return year



def number_of_borrowers(x):

    if(x==1):

        return "One"

    else:

        return "Two"





def origin_dateYearTest(x):

    year = x.split('/')[0]

    return year



def origin_dateMonthTest(x):

    month = x.split('/')[1]

    return month



def origin_dateDayTest(x):

    day = x.split('/')[2]

    return day



def first_paymentMonthTest(x):

    month = x.split('-')[0]

    return month



def first_paymentYearTest(x):

    year = x.split('-')[1]

    return year



def number_of_borrowersTest(x):

    if(x==1):

        return "One"

    else:

        return "Two"

    



train['origin_Year'] = train['origination_date'].apply(origin_dateYear)

train['origin_Month'] = train['origination_date'].apply(origin_dateMonth)

train['origin_Day'] = train['origination_date'].apply(origin_dateDay)



train['firstPayMonth'] = train['first_payment_date'].apply(first_paymentMonth)

train['firstPayYear'] = train['first_payment_date'].apply(first_paymentYear)



train['numBorrowers'] = train['number_of_borrowers'].apply(number_of_borrowers)



test['origin_Year'] = test['origination_date'].apply(origin_dateYear)

test['origin_Month'] = test['origination_date'].apply(origin_dateMonthTest)

test['origin_Day'] = test['origination_date'].apply(origin_dateDayTest)



test['firstPayMonth'] = test['first_payment_date'].apply(first_paymentMonthTest)

test['firstPayYear'] = test['first_payment_date'].apply(first_paymentYearTest)



test['numBorrowers'] = test['number_of_borrowers'].apply(number_of_borrowersTest)
train['months'] = train['loan_term']/30

test['months'] = test['loan_term']/30



train['unpaid_per_month'] = train['unpaid_principal_bal']/train['months']

test['unpaid_per_month'] = test['unpaid_principal_bal']/test['months']



train['approxIncome'] = train['unpaid_principal_bal']/train['debt_to_income_ratio']

test['approxIncome'] = test['unpaid_principal_bal']/test['debt_to_income_ratio']



train['approxValue'] = train['unpaid_principal_bal']/train['loan_to_value']

test['approxValue'] = test['unpaid_principal_bal']/test['loan_to_value']



train['payPerPerson'] = train['unpaid_principal_bal']/train['number_of_borrowers']

test['payPerPerson'] = test['unpaid_principal_bal']/test['number_of_borrowers']



train['selfPercent'] = 100 - train['insurance_percent']

test['selfPercent'] = 100 - test['insurance_percent']



train['uncoveredLoan'] = (train['selfPercent']/100) * train['unpaid_principal_bal']

test['uncoveredLoan'] = (test['selfPercent']/100) * test['unpaid_principal_bal']



train['totalDelinquency'] = train['m1'] + train['m2'] + train['m3'] + train['m4'] + train['m5'] + train['m6'] + train['m7'] + train['m8'] + train['m9'] + train['m10'] + train['m11'] + train['m12'] 

test['totalDelinquency'] = test['m1'] + test['m2'] + test['m3'] + test['m4'] + test['m5'] + test['m6'] + test['m7'] + test['m8'] + test['m9'] + test['m10'] + test['m11'] + test['m12'] 



train['delinqProb'] = train['totalDelinquency']/12

test['delinqProb'] = test['totalDelinquency']/12
def convert(x):

    if(x=='Apr'):

        return 4

    elif(x=='Mar'):

        return 3

    elif(x=='May'):

        return 5

    else:

        return 2



test['firstPayMonth'] = test['firstPayMonth'].apply(convert) 
train['intMulUnpaid'] = train['interest_rate'] * train['unpaid_principal_bal']

test['intMulUnpaid'] = test['interest_rate'] * test['unpaid_principal_bal']



train.drop(columns=['origination_date','first_payment_date'],inplace=True)

test.drop(columns=['origination_date','first_payment_date'],inplace=True)



train['lag'] = train['firstPayMonth'].astype('int') - train['origin_Month'].astype('int')

test['lag'] = test['firstPayMonth'].astype('int') - test['origin_Month'].astype('int')



train['capabilityRatio'] = train['debt_to_income_ratio']/train['loan_to_value']

test['capabilityRatio'] = test['debt_to_income_ratio']/test['loan_to_value']



train['sumCreditScore'] = train['borrower_credit_score'] + train['co-borrower_credit_score']

test['sumCreditScore'] = test['borrower_credit_score'] + test['co-borrower_credit_score']



train['sumCreditScore'] = train['sumCreditScore']/train['number_of_borrowers']

test['sumCreditScore'] = test['sumCreditScore']/test['number_of_borrowers']



def CIBIL_trend(x):

    a=''

    if(x<250):

        a='Insufficent Information'

    elif((x>=250) and (x<=550)):

        a='Poor'

    elif((x>550) and (x<=650)):

        a='Fair'

    elif((x>650) and (x<=790)):

        a='Good'

    elif((x>790) and (x<=900)):

        a='Excellent'

    else:

        a='Others'

    return a



train.loc[70926,'sumCreditScore'] = train.loc[70926,'sumCreditScore']/2



train['remark'] = train['sumCreditScore'].apply(CIBIL_trend)

test['remark'] = test['sumCreditScore'].apply(CIBIL_trend)
categorical = ['source','financial_institution','loan_purpose','insurance_type','origin_Month',

               'firstPayMonth','numBorrowers','remark','origin_Year','origin_Day','firstPayYear']
powers = set(list(test.columns)) - set(categorical) - set(['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12'])



for i in powers:

    print(i)

    train['inverse'+i]=1/(train[i]+0.01)

    train['square'+i] = np.square(train[i])

    train['cube'+i] = np.power(train[i],3)

    train['sqrt'+i] = np.sqrt(train[i])

    train['cbrt'+i] = np.power(train[i],(1/3))

    

    test['inverse'+i]=1/(test[i]+0.01)

    test['square'+i] = np.square(test[i])

    test['cube'+i] = np.power(test[i],3)

    test['sqrt'+i] = np.sqrt(test[i])

    test['cbrt'+i] = np.power(test[i],(1/3))
!pip install catboost



from catboost import CatBoostClassifier





f1 = 0

probasPred = np.zeros(len(test))

probasPredTrain = np.zeros(len(train))



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(train.drop(columns=['m13']), train['m13'].values))):

    X_train = (train.drop(columns=['m13'])).iloc[trn_idx]

    X_test = (train.drop(columns=['m13'])).iloc[val_idx]

    y_train = train['m13'].iloc[trn_idx]

    y_test = train['m13'].iloc[val_idx]

    clf = CatBoostClassifier(class_weights=[1,1.5],cat_features = categorical,silent=True,iterations=100)

    #Iterations have been set to 100 to make commiting easy. It was set to 1000 while actual submission.

    clf.fit(X_train,y_train)

    preds = clf.predict(X_test)

    f1 += f1_score(y_test,preds)

    print('Current f1 is:' + str(f1/(fold_+1)))

    probasPredTrain += clf.predict_proba((train.drop(columns=['m13'])))[:,1]

    probasPred += clf.predict_proba(test)[:,1]
probasPredTrain = probasPredTrain/5

probasPred = probasPred/5



threshF1 = []

threshVal = []



for i in np.linspace(0.05,0.95,100):

    threshF1.append(f1_score(train['m13'],(probasPredTrain>i)*1))

    threshVal.append(i)



plt.plot(threshVal,threshF1)



np.array(threshF1).argmax()



labels = (probasPred>threshVal[(np.array(threshF1).argmax())])*1
data = pd.concat([train.drop(columns=['m13']),test])



data.head()



data = pd.get_dummies(data,columns=categorical,prefix='dum')



train_new = data.iloc[:len(train)]

test_new = data.iloc[len(train):]



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train_new = sc.fit_transform(train_new)

test_new = sc.transform(test_new)



# from sklearn.decomposition import PCA

# pca = PCA(n_components = 180)

# train_new = pca.fit_transform(train_new)

# test_new = pca.transform(test_new)
from xgboost import XGBClassifier

from imblearn.under_sampling import TomekLinks
f1 = 0

probasPred = np.zeros(len(test))

probasPredTrain = np.zeros(len(train))



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(train_new, train['m13'].values))):

    X_train = (train_new)[trn_idx]

    X_test = (train_new)[val_idx]

    y_train = train['m13'][trn_idx]

    y_test = train['m13'][val_idx]

    clf = XGBClassifier(colsample_bytree=0.57574225920725,learning_rate=0.12524888976943854,max_delta_step=0.23218201667538563,

             max_depth=15,min_child_weight=11.522073006203208,reg_alpha=6.779851915666066,

              reg_lambda=6.682134255966473,scale_pos_weight=2.556586526453567,subsample=0.7805883424057105,

            eval_metric='auc')

    clf.fit(X_train,y_train)

    preds = clf.predict(X_test)

    f1 += f1_score(y_test,preds)

    print('Current f1 is:' + str(f1/(fold_+1)))

    #probasPredTrain += clf.predict_proba((train_new))[:,1]

    probasPredTrain[val_idx] = clf.predict_proba(X_test)[:,1]

    probasPred += clf.predict_proba(test_new)[:,1]
probasPred = probasPred/5
threshF1 = []

threshVal = []



for i in np.linspace(0.05,0.95,100):

    threshF1.append(f1_score(train['m13'],(probasPredTrain>i)*1))

    threshVal.append(i)



plt.plot(threshVal,threshF1)
print(np.array(threshF1).max())

print(np.array(threshF1).argmax())

labels = (probasPred>threshVal[(np.array(threshF1).argmax())])*1
f1 = 0

probasPred = np.zeros(len(test))

probasPredTrain = np.zeros(len(train))



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(train_new, train['m13'].values))):

    X_train = (train_new)[trn_idx]

    X_test = (train_new)[val_idx]

    y_train = train['m13'][trn_idx]

    y_test = train['m13'][val_idx]

    clf = LGBMClassifier(scale_pos_weight=10.12,learning_rate=0.1322,max_depth=36,

                     min_data_in_leaf=213,reg_alpha=11.01,reg_lambda=9.281,num_leaves=996)

    clf.fit(X_train,y_train)

    preds = clf.predict(X_test)

    f1 += f1_score(y_test,preds)

    print('Current f1 is:' + str(f1/(fold_+1)))

    probasPredTrain[val_idx] = clf.predict_proba(X_test)[:,1]

    probasPred += clf.predict_proba(test_new)[:,1]
probasPred = probasPred/5
threshF1 = []

threshVal = []



for i in np.linspace(0.05,0.95,100):

    threshF1.append(f1_score(train['m13'],(probasPredTrain>i)*1))

    threshVal.append(i)



plt.plot(threshVal,threshF1)
print(np.array(threshF1).max())

print(np.array(threshF1).argmax())

labels = (probasPred>threshVal[(np.array(threshF1).argmax())])*1
# f1 = 0

# probasPred = np.zeros(len(test))

# probasPredTrain = np.zeros(len(train))



# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(train_new, train['m13'].values))):

#     X_train = (train_new)[trn_idx]

#     X_test = (train_new)[val_idx]

#     y_train = train['m13'][trn_idx]

#     y_test = train['m13'][val_idx]

#     tk = TomekLinks()

#     X_train_res, y_train_res = tk.fit_resample(X_train, y_train.ravel())

#     clf = LGBMClassifier(scale_pos_weight=10.12,learning_rate=0.1322,max_depth=36,

#                      min_data_in_leaf=213,reg_alpha=11.01,reg_lambda=9.281,num_leaves=996)

#     clf.fit(X_train_res,y_train_res)

#     preds = clf.predict(X_test)

#     f1 += f1_score(y_test,preds)

#     print('Current f1 is:' + str(f1/(fold_+1)))

#     probasPredTrain[val_idx] = clf.predict_proba(X_test)[:,1]

#     probasPred.append(clf.predict(test_new))
# probasPred = probasPred/5
# threshF1 = []

# threshVal = []



# for i in np.linspace(0.05,0.95,100):

#     threshF1.append(f1_score(train['m13'],(probasPredTrain>i)*1))

#     threshVal.append(i)



# plt.plot(threshVal,threshF1)
# print(np.array(threshF1).max())

# print(np.array(threshF1).argmax())

# labels = (probasPred>threshVal[(np.array(threshF1).argmax())])*1
# f1 = 0

# probasPred = np.zeros(len(test))

# probasPredTrain = np.zeros(len(train))



# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(train_new, train['m13'].values))):

#     X_train = (train_new)[trn_idx]

#     X_test = (train_new)[val_idx]

#     y_train = train['m13'][trn_idx]

#     y_test = train['m13'][val_idx]

#     tk = TomekLinks()

#     X_train_res, y_train_res = tk.fit_resample(X_train, y_train.ravel())

#     clf = XGBClassifier(colsample_bytree=0.5126,learning_rate=0.02512,max_delta_step=5.431,max_depth=15,min_child_weight=4.108,reg_alpha=6.16,

#                         reg_lambda=4.111,scale_pos_weight=3.9,subsample=0.5776,eval_metric='auc',tree_method='gpu_hist')

#     clf.fit(X_train_res,y_train_res)

#     preds = clf.predict(X_test)

#     f1 += f1_score(y_test,preds)

#     print('Current f1 is:' + str(f1/(fold_+1)))

#     probasPredTrain[val_idx] = clf.predict_proba(X_test)[:,1]

#     probasPred.append(clf.predict(test_new))
# probasPred = probasPred/5
# threshF1 = []

# threshVal = []



# for i in np.linspace(0.05,0.95,100):

#     threshF1.append(f1_score(train['m13'],(probasPredTrain>i)*1))

#     threshVal.append(i)



# plt.plot(threshVal,threshF1)
# print(np.array(threshF1).max())

# print(np.array(threshF1).argmax())

# labels = (probasPred>threshVal[(np.array(threshF1).argmax())])*1
tester = pd.read_csv('../input/test.csv')



submission = pd.DataFrame({'loan_id':tester['loan_id'],'m13':labels})



from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "subm.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission)
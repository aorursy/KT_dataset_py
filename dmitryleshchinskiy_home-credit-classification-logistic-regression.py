from IPython.display import Image

Image("../input/homecredit/home_credit.png")
%config IPCompleter.greedy=True

import pandas as pd

import numpy as nm

import matplotlib.pyplot as plt

import warnings

import random

warnings.filterwarnings("ignore")
def chooseColor():

    colors=['b', 'g', 'r','c','m','y']

    return colors[random.randint(0,5)]
def NanTooMany(dataq):

    mis_val_percent = pd.DataFrame(100 * dataq.isnull().sum() / len(dataq))

    mis_val_percent= mis_val_percent[mis_val_percent[0]>=50]

    print("Missing values in percentage, columns to drop")

    print(mis_val_percent.sort_values(by=0,ascending=False))

    mis_val_percent.index.name="name"

    for index in mis_val_percent.iterrows(): 

        dataq=dataq.drop(index[0],axis=1)

    return dataq
def checkMatches(data,dat):

    sat=set(data['SK_ID_CURR']) & set(dat['SK_ID_CURR'])

    match=len(sat)/len(data['SK_ID_CURR'])

    print("percent of matching records: "+ str(match))

    if match>0.70:

        dataq=pd.merge(data, dat, how='left', on='SK_ID_CURR')

    return dataq


application = pd.read_csv("../input/homecredit-labeled/application_train.csv")

bureau = pd.read_csv("../input/home-credit-default-risk/bureau.csv")

previous_app = pd.read_csv("../input/home-credit-default-risk/previous_application.csv")

application=NanTooMany(application)

bureau=NanTooMany(bureau)

previous_app=NanTooMany(previous_app)


d= application['TARGET'].value_counts()

plt.figure(figsize=(10,1))

plt.title("TARGET")

y_pos = nm.arange(len(d.index))

        # Create horizontal bars

plt.barh( d.index, d.values,color= chooseColor())

 

        # Create names on the y-axis

plt.yticks(y_pos, ['Repaid','Defaulted'])

plt.show()
##adding overdue column 1 if overdue for a app exists

bureau_overdue= bureau[['SK_ID_CURR','AMT_CREDIT_SUM_OVERDUE']]

bureau_overdue= bureau_overdue.groupby(by='SK_ID_CURR').max().reset_index()

bureau_overdue['overdue']=bureau_overdue['AMT_CREDIT_SUM_OVERDUE'].apply(lambda x: 1 if x >0 else 0)

bureau_overdue=bureau_overdue.drop('AMT_CREDIT_SUM_OVERDUE', axis=1)

application1= checkMatches(application,bureau_overdue)

d=application1['overdue'].value_counts()

d
plt.figure(figsize=(10,1))

plt.barh(d.index,d.values,color= chooseColor())

plt.title("Overdue")

y_pos = nm.arange(len(d.index))

plt.yticks(y_pos, ['Repaid on time','Overdue payment'])

plt.show()
##adding new customer column 1 if new customer

has_previous_application=set(previous_app['SK_ID_CURR']) & set(application['SK_ID_CURR'])

dt={'SK_ID_CURR':list(has_previous_application)}

newCustomers=pd.DataFrame(dt)

newCustomers['new_customer']=0

application1=checkMatches(application1,newCustomers)

application1['new_customer']=application1['new_customer'].apply(lambda x: 1 if x!=0 else 0)

d=application1['new_customer'].value_counts()

d
plt.figure(figsize=(10,1))



plt.barh(d.index,d.values,color= chooseColor())

plt.title("New customer")

y_pos = nm.arange(len(d.index))

        # Create names on the y-axis

plt.yticks(y_pos, ['Already has credit','Does not have credits'])

plt.show()
##adding column last contract status 1 is positive ( approved and unused offer) othrwise 0

hello=previous_app[['SK_ID_CURR','SK_ID_PREV', 'NAME_CONTRACT_STATUS']].groupby(by='SK_ID_CURR').max().reset_index()

hello['lastContractStatus']= hello['NAME_CONTRACT_STATUS'].apply(lambda x: 1 if x=='Approved'or x=='Unused offer' else 0)

new_pv_app=hello[['SK_ID_CURR','lastContractStatus']]

application1=checkMatches(application1,new_pv_app)

d=application1['lastContractStatus'].value_counts()

d
plt.figure(figsize=(10,1))

plt.barh(d.index,d.values,color= chooseColor())

plt.title("Status of the last application")

y_pos = nm.arange(len(d.index))

 

        # Create names on the y-axis

plt.yticks(y_pos, ['Negative','Positive'])

plt.show()
application1['DAYS_BIRTH']= pd.to_numeric(application1['DAYS_BIRTH'].abs()/365, downcast="signed")

application1['YEARS_BIRTH']=application1['DAYS_BIRTH']

application1=application1.drop('DAYS_BIRTH',axis=1)

application1['YEARS_BIRTH'].describe()
application1['YEARS_BIRTH'].plot(kind='density',figsize=(5,3), title="Applicants Age")
application1['DAYS_EMPLOYED']= pd.to_numeric(application1['DAYS_EMPLOYED'].abs(), downcast="signed")

application1['MONTHS_EMPLOYED']=application1['DAYS_EMPLOYED']/30

application1=application1.drop('DAYS_EMPLOYED',axis=1)

med=application1['MONTHS_EMPLOYED'].median()

application1['MONTHS_EMPLOYED']=application1['MONTHS_EMPLOYED'].apply(lambda x :med if x>4800 else x)

application1['MONTHS_EMPLOYED'].describe()
application1['MONTHS_EMPLOYED'].plot(kind='density',figsize=(5,3), title="Employee Seniority")
coro=application1.filter(regex=('FLAG_DOCUMENT_') )

coro['SK_ID_CURR']=application1['SK_ID_CURR']

coro['TARGET']=application1['TARGET']

coro['totalDocuments']=coro[['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4',

                    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',

                    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',

                    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']].sum(axis=1)

d=coro['totalDocuments'].value_counts()

d
hu=pd.crosstab(coro['TARGET'],coro['totalDocuments'])

hu.head()
##checking is there is association : H0=variables are independent

from scipy import stats

chi2, p, dof, ex=stats.chi2_contingency(hu)

print(str(p)+" p < 0.05 reject H0. Variables are not independent and \n there is a relationship between number of provided documents and default")
coro['document_provided']=coro['totalDocuments'].apply(lambda x: 1 if x > 0  else 0)

d=coro['document_provided'].value_counts()

d
plt.figure(figsize=(10,1))

plt.barh(d.index,d.values,color=chooseColor())

plt.title("Documents provided by customer")

y_pos = nm.arange(len(d.index))



        # Create names on the y-axis

plt.yticks(y_pos, ['Not provided','Provided'])

plt.show()
application1=application1[application1.columns.drop(list(application1.filter(regex='FLAG_DOCUMENT_')))]

application1['document_provided']=coro['document_provided']

application1['totalDocuments']=coro['totalDocuments']

##current credit volume

bureau['AMT_CREDIT_SUM_DEBT']=bureau['AMT_CREDIT_SUM_DEBT'].fillna(0).abs()

credit=pd.DataFrame({'SK_ID_CURR':bureau['SK_ID_CURR'], 'debt':bureau['AMT_CREDIT_SUM_DEBT']})

credit=credit.groupby(by='SK_ID_CURR').sum().reset_index()

application3=checkMatches(application1,credit)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

application3['debt'].describe()





plt.title("Existing debt no extreme values")

application3.boxplot(column='debt',showfliers=False, figsize=(5,5))
##if al credit talen by customer have been repaid

bureau['CREDIT_SCORE']=bureau['CREDIT_ACTIVE'].apply(lambda x :1 if x=="Closed" else -200)

nn={'SK_ID_CURR':bureau['SK_ID_CURR'],'repaidCredit':bureau['CREDIT_SCORE']}

gooCustomer=pd.DataFrame(nn)

gooCustomer=gooCustomer.groupby(by='SK_ID_CURR').sum().reset_index()

gooCustomer['repaidCredit']=gooCustomer['repaidCredit'].apply(lambda x:1 if x>0 else 0)

application4= checkMatches(application3, gooCustomer)

d= application4['repaidCredit'].value_counts()

d
plt.figure(figsize=(10,1))

plt.barh(d.index,d.values,color=chooseColor())

plt.title("All credits been repaid")

y_pos = nm.arange(len(d.index))

 



 

        # Create names on the y-axis

plt.yticks(y_pos, ['No details provided','All been repaid'])

plt.show()
##bureau1 = pd.read_csv("bureau.csv")

##if the last credit has been repaid

lastCredit=pd.DataFrame({"dateClosed":bureau['DAYS_CREDIT_ENDDATE'],'SK_ID_CURR':bureau['SK_ID_CURR'],'status':bureau['CREDIT_ACTIVE']})

lastCredit['dateClosed']=lastCredit['dateClosed'].apply(lambda x :x if x<=0 else 1)

lastCredit['status']=lastCredit['status'].apply(lambda x :1 if x=="Closed" else 0)

lastCredit=lastCredit[lastCredit['dateClosed']<=0]

lastCredit=lastCredit.groupby(by='SK_ID_CURR').max().reset_index()

lastCreditPaid=pd.DataFrame(lastCredit[lastCredit['status']==1])

lastCreditFail=pd.DataFrame(lastCredit[lastCredit['status']==0])

print("repaid customers: "+str(len(lastCreditPaid))+ " defauled customers: "+ str(len(lastCreditFail)))

lastCreditPaid.drop('status',axis=1)

lastCreditFail.drop('status',axis=1)

application5= checkMatches(application4,lastCreditPaid)

application5['repaidLastCredit']= application5['dateClosed'].apply(lambda x:1 if x<=0 else 0)

application5=application5.drop('dateClosed',axis=1)

application5=application5.drop('status', axis=1)

d= application5['repaidLastCredit'].value_counts()

d
plt.figure(figsize=(10,1))

plt.barh(d.index,d.values,color=chooseColor())

plt.title("The customers who repaid the last credit")

y_pos = nm.arange(len(d.index))

 

        # Create horizontal bars

plt.barh( d.index, d.values)

 

        # Create names on the y-axis

plt.yticks(y_pos, ['No details','Repaid'])

plt.show()
application5.columns
quant= application5[['TARGET','CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',  'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',  'CNT_FAM_MEMBERS', 

          'HOUR_APPR_PROCESS_START', 'EXT_SOURCE_2','REGION_POPULATION_RELATIVE', 'EXT_SOURCE_3','YEARS_BEGINEXPLUATATION_AVG', 'FLOORSMAX_AVG',

        'YEARS_BEGINEXPLUATATION_MODE', 'FLOORSMAX_MODE', 'YEARS_BEGINEXPLUATATION_MEDI', 'FLOORSMAX_MEDI', 'TOTALAREA_MODE',

 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 

        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',

'YEARS_BIRTH', 'MONTHS_EMPLOYED',  'totalDocuments', 'debt']]

from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()

scaler.fit(quant)

qnt=pd.DataFrame(scaler.transform(quant), columns= quant.columns)

quant.corr('spearman')['TARGET'].sort_values(ascending=False)
import seaborn as s

visual=quant[['TARGET','AMT_INCOME_TOTAL','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','debt',

'DAYS_REGISTRATION','DEF_30_CNT_SOCIAL_CIRCLE','FLOORSMAX_AVG','MONTHS_EMPLOYED','YEARS_BIRTH','EXT_SOURCE_2','EXT_SOURCE_3']]

s.pairplot(visual)
def flagReplace(list,data):

    for i in list:

        data[i]=data[i].apply(lambda x:1 if x=="Y" else 0)

    return data
def removeOutliers(array,data):

    for i in array:

        quant=data[i].quantile(0.99)

        med=data[i].median()

        ##data[i]=data[i][data[i]<quant]

        data[i]=data[i].apply(lambda x:med if x >quant else x)

    return data
def replaceColumn(list,train1, train):

    for i in list:

        train[i]=train1[i]

    return train



def fillNaNumber(list,train, number):

    for i in list:

        if number=="median":

            train[i]=train[i].fillna(train[i].median())

           

        elif number== "freq":

            train[i] = train[i].fillna(train[i].value_counts().index[0])

        else:

            train[i]=train[i].fillna(number)

    return train
prp= application5[['SK_ID_CURR','TARGET','EXT_SOURCE_2','EXT_SOURCE_3','debt','CNT_CHILDREN','new_customer','totalDocuments','overdue','lastContractStatus','document_provided','repaidCredit','repaidLastCredit','AMT_INCOME_TOTAL','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_REGISTRATION',

                    'DEF_30_CNT_SOCIAL_CIRCLE','FLOORSMAX_AVG','MONTHS_EMPLOYED','YEARS_BIRTH','CNT_FAM_MEMBERS', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]

prp=fillNaNumber(['EXT_SOURCE_2','EXT_SOURCE_3','debt','CNT_CHILDREN','new_customer','totalDocuments','document_provided','repaidLastCredit'],prp,0)

prp=fillNaNumber(['repaidCredit','overdue','lastContractStatus'],prp,2)

prp=fillNaNumber(['AMT_INCOME_TOTAL','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_REGISTRATION',

                    'DEF_30_CNT_SOCIAL_CIRCLE','FLOORSMAX_AVG','MONTHS_EMPLOYED','YEARS_BIRTH','CNT_CHILDREN'],prp,"median")

prp=fillNaNumber(['CNT_FAM_MEMBERS', 'CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY'],prp,"freq")

prp=flagReplace(['FLAG_OWN_CAR','FLAG_OWN_REALTY'],prp)

prp['CODE_GENDER']=prp['CODE_GENDER'].apply(lambda x :1 if x=="M" else 2)

prp=removeOutliers(['AMT_INCOME_TOTAL','debt','DEF_30_CNT_SOCIAL_CIRCLE'],prp)

from sklearn.preprocessing import MinMaxScaler 

scaler = MinMaxScaler()

arr=['debt','AMT_INCOME_TOTAL','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_REGISTRATION','DEF_30_CNT_SOCIAL_CIRCLE','FLOORSMAX_AVG','MONTHS_EMPLOYED','YEARS_BIRTH']

norm = pd.DataFrame(prp[arr], columns=arr)

scaler.fit(norm)

trainN=pd.DataFrame(scaler.transform(norm), columns=arr)

t=replaceColumn(arr,trainN,prp)

final= t
from sklearn.model_selection import train_test_split

import statsmodels.formula.api as smf

train, test = train_test_split(final, test_size=0.3, random_state=1)

train.shape
train.columns
def findlinear(prepare,train):

    results={}

    formul="TARGET~"

    for cl in prepare:

        formul1=formul

        formul+=cl

        res = smf.logit(formul, data = train).fit()

        check=False

        for p in res.pvalues.values:

            if p>=0.05:

                check=True

                formul=formul1

        if check == False:

            results[formul]=res.prsquared

            formul+="+"

    return pd.Series(results)



def findPolyn(prepare, power,train):

    results={}

    formul="TARGET~"

    for cl in prepare:

        formul1=formul

        if cl[0:2]!="C(":

            formulL=formul+cl

            formulP=formul+"nm.power( "+cl+" ,"+str(power)+" )"

            resL = smf.logit(formulL, data = train).fit()

            resP= smf.logit(formulP, data = train).fit()

            checkL=False

            checkP=False

            for p in resL.pvalues.values:

                if p>=0.05:

                    checkL=True

                

            for p in resP.pvalues.values:

                if p>=0.05:

                    checkP=True

            final=""

            if  checkL==True and checkP==False:

                formul= formulP

                results[formul]=resP.prsquared

                formul+="+"

            elif checkP==True and checkL==False:

                formul= formulL

                results[formul]=resL.prsquared

                formul+="+"

            elif  checkL==False and checkP==False and resL.prsquared > resP.prsquared:

                formul= formulL

                results[formul]=resL.prsquared

                formul+="+"

            elif  checkL==False and checkP==False and resL.prsquared < resP.prsquared:

                formul= formulP

                results[formul]=resP.prsquared

                formul+="+"

            elif checkL==True and checkP==True:

                formul=formul1

        else:

            formul+=cl

            

            res = smf.logit(formul, data = train).fit()

            check=False

            for p in res.pvalues.values:

                if p>=0.05:

                    check=True

                    formul=formul1

            if check == False:

                results[formul]=res.prsquared

                formul+="+"

        

    return pd.Series(results)
'''

prepare= ['EXT_SOURCE_2', 'EXT_SOURCE_3','totalDocuments','DAYS_LAST_PHONE_CHANGE','debt','FLOORSMAX_AVG', 'MONTHS_EMPLOYED', 'YEARS_BIRTH', 'CNT_FAM_MEMBERS',

       'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DEF_30_CNT_SOCIAL_CIRCLE',

          'C(FLAG_OWN_CAR)', 'C(FLAG_OWN_REALTY)','C(new_customer)', 'C(overdue)',

         'C(repaidCredit)','C(repaidLastCredit)']



'''

prepare= ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'debt',

       'CNT_CHILDREN', 'totalDocuments',

        'AMT_INCOME_TOTAL', 'DAYS_LAST_PHONE_CHANGE',

       'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DEF_30_CNT_SOCIAL_CIRCLE',

       'FLOORSMAX_AVG', 'MONTHS_EMPLOYED', 'YEARS_BIRTH', 'CNT_FAM_MEMBERS','C(document_provided)', 'C(repaidCredit)',

       'C(FLAG_OWN_CAR)', 'C(FLAG_OWN_REALTY)', 'C(overdue)',

       'C(repaidLastCredit)','C(lastContractStatus)']



linear=findlinear(prepare,train)

print(linear.sort_values(ascending=False).head(3))

polynomial=findPolyn(prepare,2,train)

print(polynomial.sort_values(ascending=False).head(3))


formulas=['TARGET~EXT_SOURCE_2+EXT_SOURCE_3+debt+CNT_CHILDREN+totalDocuments+DAYS_LAST_PHONE_CHANGE+DAYS_ID_PUBLISH+DAYS_REGISTRATION+DEF_30_CNT_SOCIAL_CIRCLE+FLOORSMAX_AVG+MONTHS_EMPLOYED+CNT_FAM_MEMBERS+C(new_customer)+C(FLAG_OWN_CAR)+C(overdue)+C(document_provided)+C(repaidLastCredit)',

'TARGET~EXT_SOURCE_2+EXT_SOURCE_3+debt+CNT_CHILDREN+totalDocuments+DAYS_LAST_PHONE_CHANGE+DAYS_ID_PUBLISH+DAYS_REGISTRATION+DEF_30_CNT_SOCIAL_CIRCLE+FLOORSMAX_AVG+MONTHS_EMPLOYED+CNT_FAM_MEMBERS+C(new_customer)+C(FLAG_OWN_CAR)+C(overdue)+C(document_provided)',

'TARGET~EXT_SOURCE_2+EXT_SOURCE_3+debt+CNT_CHILDREN+totalDocuments+DAYS_LAST_PHONE_CHANGE+DAYS_ID_PUBLISH+DAYS_REGISTRATION+DEF_30_CNT_SOCIAL_CIRCLE+FLOORSMAX_AVG+MONTHS_EMPLOYED+CNT_FAM_MEMBERS+C(new_customer)+C(FLAG_OWN_CAR)+C(overdue) ']

    

sqrtFormulas=['TARGET~EXT_SOURCE_2+nm.power( EXT_SOURCE_3 ,2 )+debt+CNT_CHILDREN+totalDocuments+nm.power( AMT_INCOME_TOTAL ,2 )+DAYS_LAST_PHONE_CHANGE+DAYS_ID_PUBLISH+DAYS_REGISTRATION+DEF_30_CNT_SOCIAL_CIRCLE+FLOORSMAX_AVG+MONTHS_EMPLOYED+CNT_FAM_MEMBERS+C(new_customer)+C(overdue)+C(document_provided)+C(repaidCredit)+C(repaidLastCredit)',

'TARGET~EXT_SOURCE_2+nm.power( EXT_SOURCE_3 ,2 )+debt+CNT_CHILDREN+totalDocuments+nm.power( AMT_INCOME_TOTAL ,2 )+DAYS_LAST_PHONE_CHANGE+DAYS_ID_PUBLISH+DAYS_REGISTRATION+DEF_30_CNT_SOCIAL_CIRCLE+FLOORSMAX_AVG+MONTHS_EMPLOYED+CNT_FAM_MEMBERS+C(new_customer)+C(overdue)+C(document_provided)+C(repaidCredit)',

'TARGET~EXT_SOURCE_2+nm.power( EXT_SOURCE_3 ,2 )+debt+CNT_CHILDREN+totalDocuments+nm.power( AMT_INCOME_TOTAL ,2 )+DAYS_LAST_PHONE_CHANGE+DAYS_ID_PUBLISH+DAYS_REGISTRATION+DEF_30_CNT_SOCIAL_CIRCLE+FLOORSMAX_AVG+MONTHS_EMPLOYED+CNT_FAM_MEMBERS+C(new_customer)+C(overdue)+C(document_provided)']

def compareMods(formulas,train):

    for f in formulas:

        res = smf.logit(f, data = train).fit()

        print("R^2: "+str(res.prsquared)+" RMSE: "+ str(nm.sqrt(nm.square(res.resid_response).sum()) / len(res.resid_response)))

print("----------------Linear Models-----------------")  

compareMods(formulas,train)

print("----------------Polynomial Models-----------------")

compareMods(sqrtFormulas,train)
dummies= pd.get_dummies(prp[['repaidCredit','overdue','lastContractStatus']], columns=['repaidCredit','overdue','lastContractStatus'])

trainDummy=final

trainDummy=trainDummy.drop(['repaidCredit','overdue','lastContractStatus'],axis=1)

trainDummy=trainDummy.join(dummies)

trainDummy.columns
def CV(data,target):

    clf = LogisticRegressionCV(cv=5, random_state=0, scoring="roc_auc").fit(data, target)

    print("AUC :"+str(clf.score(data,target)))
from sklearn.linear_model import LogisticRegressionCV



target= trainDummy['TARGET']

data= trainDummy[['EXT_SOURCE_2','EXT_SOURCE_3','debt','CNT_CHILDREN','new_customer','totalDocuments','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_REGISTRATION','DEF_30_CNT_SOCIAL_CIRCLE','FLOORSMAX_AVG','MONTHS_EMPLOYED','CNT_FAM_MEMBERS','FLAG_OWN_CAR','repaidCredit_0.0',

       'repaidCredit_1.0', 'repaidCredit_2.0', 'overdue_0.0', 'overdue_1.0',

       'overdue_2.0','lastContractStatus_0.0', 'lastContractStatus_1.0',

       'lastContractStatus_2.0']]



CV(data,target)

trainDummySqrt=trainDummy

trainDummySqrt['EXT_SOURCE_3']=trainDummySqrt['EXT_SOURCE_3']**2

#trainDummySqrt['YEARS_BIRTH']=trainDummySqrt['YEARS_BIRTH']**2

trainDummySqrt['AMT_INCOME_TOTAL']=trainDummySqrt['AMT_INCOME_TOTAL']**2

data=trainDummySqrt[['EXT_SOURCE_2','EXT_SOURCE_3','debt','CNT_CHILDREN','new_customer','totalDocuments','AMT_INCOME_TOTAL','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH','DAYS_REGISTRATION','DEF_30_CNT_SOCIAL_CIRCLE','FLOORSMAX_AVG','MONTHS_EMPLOYED','CNT_FAM_MEMBERS','repaidCredit_0.0',

       'repaidCredit_1.0', 'repaidCredit_2.0', 'overdue_0.0', 'overdue_1.0',

       'overdue_2.0', 'document_provided', 'repaidLastCredit']]

#data1=trainDummySqrt[['EXT_SOURCE_2','EXT_SOURCE_3','totalDocuments','DAYS_LAST_PHONE_CHANGE','debt','FLOORSMAX_AVG','MONTHS_EMPLOYED','YEARS_BIRTH','AMT_INCOME_TOTAL','DAYS_ID_PUBLISH','DAYS_REGISTRATION','DEF_30_CNT_SOCIAL_CIRCLE','new_customer_1','overdue_0.0', 'overdue_1.0', 'overdue_2.0']]

#data2=trainDummySqrt[['EXT_SOURCE_2','EXT_SOURCE_3','totalDocuments','DAYS_LAST_PHONE_CHANGE','debt','FLOORSMAX_AVG','MONTHS_EMPLOYED','YEARS_BIRTH','AMT_INCOME_TOTAL','DAYS_ID_PUBLISH','DAYS_REGISTRATION','DEF_30_CNT_SOCIAL_CIRCLE','new_customer_1']]

CV(data,target)

#CV(data1,target)

#CV(data2,target)

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty":["l1","l2"]}

logit=LogisticRegression()

cv=GridSearchCV(logit,grid,cv=5,scoring="roc_auc")

cv.fit(data,target)



print("Tuned parameters:",cv.best_params_)

print("AUC score for the best parametrized model:",cv.best_score_)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

log=LogisticRegression(C= 10, penalty='l1',solver='liblinear').fit(X_train,y_train)

res=log.predict(X_test)

from sklearn.metrics import confusion_matrix

print("-----------CONFUSION MATRIX----------")

print(confusion_matrix(y_test, res, labels=[0,1]))

start='TARGET~EXT_SOURCE_2+nm.power( EXT_SOURCE_3 ,2 )+debt+CNT_CHILDREN+C(new_customer)+totalDocuments+nm.power( AMT_INCOME_TOTAL ,2 )+DAYS_LAST_PHONE_CHANGE+DAYS_ID_PUBLISH+DAYS_REGISTRATION+DEF_30_CNT_SOCIAL_CIRCLE+FLOORSMAX_AVG+MONTHS_EMPLOYED+CNT_FAM_MEMBERS+C(overdue)+C(document_provided)+C(repaidCredit)+C(repaidLastCredit)'



res = smf.logit(start, data = train).fit()

print(res.summary())



ss={"Feature":res.params.index, "Coef":res.params.values}

ss=pd.DataFrame(ss)

ss.plot.bar(x='Feature', y='Coef', rot=90,title="Feature importance")



import matplotlib.pyplot as plm

from sklearn.metrics import roc_curve, auc

yhat= res.predict(test)



false_positive_rate, true_positive_rate, thresholds = roc_curve(test['TARGET'], yhat)

false_positive_rate_fake, true_positive_rate_fake, thresholds2 = roc_curve(test['TARGET'], test['TARGET'])

plm.rcParams["figure.figsize"] = (10,10)

plm.title("ROC-AUC")

#plt.xlabel('False Positive')

#plm.ylabel('True Positive')

plm.plot(false_positive_rate, true_positive_rate, label="The model's ROC")

plm.plot(false_positive_rate_fake, true_positive_rate_fake,color="green",linestyle="--", label="Ideal ROC")

plm.plot([0,1],[0,1], color="red", label="Predicts at chance model")

plm.legend()
from sklearn.metrics import roc_auc_score

print( "Models AUC score : "+ str(roc_auc_score(test['TARGET'], yhat)))
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=0)

print("Decision tree AUC score "+str(cross_val_score(clf, data, target, cv=5, scoring="roc_auc").mean()))
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)

print("K-Neighbors classifier AUC score "+str(cross_val_score(neigh, data, target, cv=5, scoring="roc_auc").mean()))
from IPython.display import HTML

HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

The raw code for this IPython notebook is by default hidden for easier reading.

To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
#Préparation mise en forme - 8 mn
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import collections
import gc
deb=time.time()
path="../input"
listfiles={"application_train":("SK_ID_CURR",["SK_ID_CURR"]),"application_test":("SK_ID_CURR",["SK_ID_CURR"]),"bureau":("SK_ID_CURR",["SK_ID_CURR","SK_ID_BUREAU"]),
           "bureau_balance":("SK_ID_BUREAU",["SK_ID_BUREAU"]),"POS_CASH_balance":("SK_ID_PREV",["SK_ID_CURR","SK_ID_PREV"]),"previous_application":("SK_ID_CURR",["SK_ID_CURR","SK_ID_PREV"]),
           "installments_payments":("SK_ID_PREV",["SK_ID_CURR","SK_ID_PREV"]),"credit_card_balance":("SK_ID_PREV",["SK_ID_CURR","SK_ID_PREV"])}
links=[('bureau_balance','bureau'),('installments_payments','previous_application'),('credit_card_balance','previous_application'),('POS_CASH_balance','previous_application'),
      ('previous_application','full'),('bureau','full')]
#Suivi perfs
ftime=[("Start",time.time(),0)]
def timer(title,showtime=True):
    global ftime
    newtime=(title,time.time(),time.time()-ftime[-1][1])
    ftime.append(newtime)
    if showtime==True:
        print(newtime[0] + " in " + str(np.round(newtime[2],2)))
    else:
        print(newtime[0]) 
data={x:(pd.read_csv(path +"/"+ x + '.csv')).sort_values(listfiles[x][0],ascending=True) for x in listfiles}
listetodel=list(set(list(data["application_train"])[44:91])^set(["FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]))
print("liste ok")
for i in listfiles:
    data[i].index=list(range(data[i].shape[0]))
data["full"]=pd.concat([data["application_train"],data["application_test"]],axis=0,sort=True)
data["full"].index=list(range(data["full"].shape[0]))
indtest=pd.DataFrame(data["full"]["SK_ID_CURR"].iloc[data["application_train"].shape[0]:])
train_shape,test_shape=data["application_train"].shape[0],data["application_test"].shape[0]
del data["application_train"],data["application_test"]
dicovars={}
timer("Files read and sorted by key")

#Fonctions de préparation des données
def agregate(dataf,key,table): #dataf doit être triée par key
    global dicovars
    continuous, discrete,dichot=set_groups(dataf)
    if len(dichot) > 0 : dichot = dichot + [key]
    vals=sorted(list(set(dataf[key])))
    size=len(vals)
    nbint=10
    pace=size//nbint+nbint
    indmin=dataf[key].index[0]
    dicodf={}
    for k,i in enumerate(list(range(pace,size+pace,pace-1))):
        valmax=vals[i-pace:i-1][-1]
        indmax=dataf[key][dataf[key]==valmax].index.tolist()[-1]
        dicosets={}
        if len(continuous) > 0 : dicosets["continuous"]=dataf[continuous].loc[indmin:indmax,:].groupby(key).agg(["sum","max","min"])
        if len(dichot) > 0 : dicosets["dichot"]=dataf[dichot].loc[indmin:indmax,:].groupby(key).agg(["sum"])
        dicosets["count"]=pd.DataFrame(dataf.loc[indmin:indmax,:].groupby(key)[key].agg("count"))
        ind=pd.MultiIndex(levels=[[table],["count"]],labels=[[0],[0]])
        dicosets["count"].columns=ind
        for x in list(dicosets): dicosets[x].columns=[x[0]+"_"+x[1] for x in dicosets[x].columns]
        dicodf[k]=pd.concat([dicosets[x] for x in list(dicosets)],axis=1)
        del dicosets
        indmin=indmax+1
    dataf_tsf=pd.concat([dicodf[k] for k in list(dicodf)],axis=0)
    timer("Agreggation done")
    return dataf_tsf

def binar(series,table):
    global dicovars
    #AJOUT
    setserie=set(series)
    if len(setserie)==2:
        var=pd.DataFrame(pd.get_dummies(series).iloc[:,0])
    else: var=pd.get_dummies(series)
    name=series.name
    var.columns=[str(x) + "_"+ name for x in var.columns]
    dicovars[table+ "_"+str(name)]=var.columns.tolist()
    return var

def set_groups(dataf):
    num=dataf.columns[(dataf.dtypes!="object").tolist()].tolist()
    nom=dataf.columns[(dataf.dtypes=="object").tolist()].tolist()
    continuous,discrete,dichot = [],nom,[]
    for i in num:
        if len(set(dataf[i].loc[:1000]))>3: continuous.append(i)
        else:
            setvar=set(dataf[i])
            if len(setvar)>3: continuous.append(i)
            elif (setvar==set([0,1]) and dataf[i].count()==dataf.shape[0]): dichot.append(i)
            else: discrete.append(i)
    timer("Groups set")
    return(continuous, discrete,dichot)

def fill_values(dataf,continuous, discrete,dichot,table,fill_cont=True):
    dicodf={}
    #Variables discrètes
    if len(discrete) >0:
        dataf[discrete]=dataf[discrete].fillna('missing')
        timer("Missing values filled for discrete vars")
        dicodf["discrete"]=pd.concat([binar(dataf[x],table) for x in discrete],axis=1)
    #Variables continues
    liste=list(set(dataf)^set(discrete))
    if len(liste)>0:
        dicodf["Others"]=dataf[liste]
    fusion=pd.concat([dicodf[x] for x in list(dicodf)],axis=1)
    return fusion

def process(dataf,keys,keyforagg,name):
    liste=list(set(dataf)^set(keys))
    dataf=dataf[liste+[keyforagg]]
    columnswithoutkeys=list(set(dataf)^set([keyforagg]))
    continuous, discrete,dichot =set_groups(dataf[columnswithoutkeys])
    dataf=fill_values(dataf,continuous, discrete,dichot,name,fill_cont=False)
    liste=dataf.columns.tolist()
    for i in range(len(liste)):
        if liste[i] not in keys: liste[i]= str(liste[i]+"_" +name)
    dataf.columns=liste
    dataf=agregate(dataf,keyforagg,name)
    dataf[keyforagg]=dataf.index.tolist()
    dataf.index=list(range(dataf.shape[0]))
    return dataf

diconame={"bureau":"B","bureau_balance":"BB","POS_CASH_balance":"PCB","previous_application":"PA","installments_payments":"IP","credit_card_balance":"CCB"}
for i in links:
    timer("Processing " + i[0],showtime=False)
    data[i[0]]=process(data[i[0]],listfiles[i[0]][1],listfiles[i[0]][0],diconame[i[0]])
    gc.collect()
    timer("Merging " + i[0] + " with " +i[1],showtime=False)
    print(data[i[0]].shape)
    data[i[1]]=data[i[1]].merge(right=data[i[0]],how='left',on=listfiles[i[0]][0])
    #data[i[1]].loc[:,list(data[i[0]])]=data[i[1]].loc[:,list(data[i[0]])].fillna(0)
    del data[i[0]]
    timer("Merging completed")
    print(data[i[1]].shape)
    gc.collect()
    

columnswithoutkeys=list(set(data["full"])^set(["SK_ID_CURR"]))
continuous, discrete,dichot =set_groups(data["full"][columnswithoutkeys])
data["full"]=fill_values(data["full"],continuous, discrete,dichot,"BB",fill_cont=False)
gc.collect()

gc.collect()
print('total preprocessing time : ' + str(time.time()-deb))

data["full"]=data["full"][data["full"].count()[data["full"].count()>0].index.tolist()]

#Selecting vars
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import random


data["full"]["Annuity_income_ratio"]=data["full"]["AMT_ANNUITY"]/data["full"]["AMT_INCOME_TOTAL"]
data["full"]['Credit_annuity_ratio'] = data["full"]['AMT_CREDIT'] / data["full"]['AMT_ANNUITY']
data["full"]['NEW_SOURCES_PROD'] = data["full"]['EXT_SOURCE_1'] * data["full"]['EXT_SOURCE_2'] * data["full"]['EXT_SOURCE_3']
data["full"]['NEW_EXT_SOURCES_MEAN'] = data["full"][['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
data["full"]['NEW_CREDIT_TO_GOODS_RATIO'] = data["full"]['AMT_CREDIT'] / data["full"]['AMT_GOODS_PRICE']
data["full"]['NEW_EMPLOY_TO_BIRTH_RATIO'] = data["full"]['DAYS_EMPLOYED'] / data["full"]['DAYS_BIRTH']
data["full"]['NEW_PHONE_TO_EMPLOY_RATIO'] = data["full"]['DAYS_LAST_PHONE_CHANGE'] / data["full"]['DAYS_EMPLOYED']
data["full"]['NEW_SCORES_STD'] = data["full"][['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)


#FI

liste=list(filter(lambda x :x !="TARGET",list(data["full"])))

X_train, X_val, y_train, y_val = train_test_split((data["full"].iloc[: train_shape,:])[liste], data["full"]["TARGET"].iloc[:train_shape], test_size=0.2, random_state=42)
X_test=(data["full"].iloc[train_shape:,:])[liste]
del data["full"]
gc.collect()

listetodel=['2_STATUS_BB_sum_B_min',
 '3_STATUS_BB_sum_B_max',
 '3_STATUS_BB_sum_B_min',
 '4_STATUS_BB_sum_B_max',
 '4_STATUS_BB_sum_B_min',
 'AMT_BALANCE_CCB_sum_PA_sum',
 'AMT_CREDIT_SUM_OVERDUE_B_min',
 'AMT_DRAWINGS_ATM_CURRENT_CCB_min_PA_max',
 'AMT_DRAWINGS_ATM_CURRENT_CCB_min_PA_min',
 'AMT_DRAWINGS_CURRENT_CCB_min_PA_max',
 'AMT_DRAWINGS_OTHER_CURRENT_CCB_min_PA_max',
 'AMT_DRAWINGS_OTHER_CURRENT_CCB_min_PA_min',
 'AMT_DRAWINGS_OTHER_CURRENT_CCB_min_PA_sum',
 'AMT_DRAWINGS_OTHER_CURRENT_CCB_sum_PA_max',
 'AMT_DRAWINGS_POS_CURRENT_CCB_min_PA_max',
 'AMT_DRAWINGS_POS_CURRENT_CCB_min_PA_sum',
 'AMT_INST_MIN_REGULARITY_CCB_min_PA_max',
 'AMT_INST_MIN_REGULARITY_CCB_sum_PA_sum',
 'AMT_RECIVABLE_CCB_min_PA_min',
 'AMT_REQ_CREDIT_BUREAU_HOUR',
 'AMT_TOTAL_RECEIVABLE_CCB_max_PA_sum',
 'AMT_TOTAL_RECEIVABLE_CCB_min_PA_max',
 'AMT_TOTAL_RECEIVABLE_CCB_sum_PA_sum',
 'APARTMENTS_AVG',
 'APARTMENTS_MEDI',
 'APARTMENTS_MODE',
 'Additional Service_NAME_GOODS_CATEGORY_PA_sum',
 'Advertising_ORGANIZATION_TYPE',
 'Agriculture_ORGANIZATION_TYPE',
 'Amortized debt_NAME_CONTRACT_STATUS_PCB_sum_PA_max',
 'Amortized debt_NAME_CONTRACT_STATUS_PCB_sum_PA_min',
 'Amortized debt_NAME_CONTRACT_STATUS_PCB_sum_PA_sum',
 'Animals_NAME_GOODS_CATEGORY_PA_sum',
 'Another type of loan_CREDIT_TYPE_B_sum',
 'Approved_NAME_CONTRACT_STATUS_CCB_sum_PA_max',
 'Approved_NAME_CONTRACT_STATUS_CCB_sum_PA_min',
 'Approved_NAME_CONTRACT_STATUS_CCB_sum_PA_sum',
 'Approved_NAME_CONTRACT_STATUS_PCB_sum_PA_max',
 'Approved_NAME_CONTRACT_STATUS_PCB_sum_PA_min',
 'Approved_NAME_CONTRACT_STATUS_PCB_sum_PA_sum',
 'BASEMENTAREA_AVG',
 'BASEMENTAREA_MEDI',
 'BASEMENTAREA_MODE',
 'Bad debt_CREDIT_ACTIVE_B_sum',
 'Building a house or an annex_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Business development_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Buying a garage_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Buying a holiday home / land_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Buying a home_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Buying a new car_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Buying a used car_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'CLIENT_CODE_REJECT_REASON_PA_sum',
 'CNT_CREDIT_PROLONG_B_min',
 'CNT_CREDIT_PROLONG_B_sum',
 'CNT_DRAWINGS_ATM_CURRENT_CCB_min_PA_max',
 'CNT_DRAWINGS_ATM_CURRENT_CCB_min_PA_min',
 'CNT_DRAWINGS_ATM_CURRENT_CCB_min_PA_sum',
 'CNT_DRAWINGS_CURRENT_CCB_min_PA_max',
 'CNT_DRAWINGS_CURRENT_CCB_min_PA_min',
 'CNT_DRAWINGS_CURRENT_CCB_min_PA_sum',
 'CNT_DRAWINGS_OTHER_CURRENT_CCB_max_PA_max',
 'CNT_DRAWINGS_OTHER_CURRENT_CCB_max_PA_min',
 'CNT_DRAWINGS_OTHER_CURRENT_CCB_min_PA_max',
 'CNT_DRAWINGS_OTHER_CURRENT_CCB_min_PA_min',
 'CNT_DRAWINGS_OTHER_CURRENT_CCB_min_PA_sum',
 'CNT_DRAWINGS_OTHER_CURRENT_CCB_sum_PA_max',
 'CNT_DRAWINGS_OTHER_CURRENT_CCB_sum_PA_min',
 'CNT_DRAWINGS_POS_CURRENT_CCB_min_PA_sum',
 'CNT_INSTALMENT_MATURE_CUM_CCB_min_PA_max',
 'CNT_INSTALMENT_MATURE_CUM_CCB_min_PA_min',
 'CNT_INSTALMENT_MATURE_CUM_CCB_min_PA_sum',
 'COMMONAREA_AVG',
 'COMMONAREA_MEDI',
 'COMMONAREA_MODE',
 'CREDIT_DAY_OVERDUE_B_min',
 'Canceled_NAME_CONTRACT_STATUS_PCB_sum_PA_max',
 'Canceled_NAME_CONTRACT_STATUS_PCB_sum_PA_min',
 'Canceled_NAME_CONTRACT_STATUS_PCB_sum_PA_sum',
 'Car dealer_CHANNEL_TYPE_PA_sum',
 'Car repairs_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Cars_NAME_PORTFOLIO_PA_sum',
 'Cash loan (non-earmarked)_CREDIT_TYPE_B_sum',
 'Cashless from the account of the employer_NAME_PAYMENT_TYPE_PA_sum',
 'Cleaning_ORGANIZATION_TYPE',
 'Cooking staff_OCCUPATION_TYPE',
 'Culture_ORGANIZATION_TYPE',
 'DAYS_FIRST_DRAWING_PA_max',
 'Demand_NAME_CONTRACT_STATUS_CCB_sum_PA_max',
 'Demand_NAME_CONTRACT_STATUS_CCB_sum_PA_min',
 'Demand_NAME_CONTRACT_STATUS_CCB_sum_PA_sum',
 'Demand_NAME_CONTRACT_STATUS_PCB_sum_PA_min',
 'Demand_NAME_CONTRACT_STATUS_PCB_sum_PA_sum',
 'Direct Sales_NAME_GOODS_CATEGORY_PA_sum',
 'ELEVATORS_AVG',
 'ELEVATORS_MEDI',
 'ELEVATORS_MODE',
 'ENTRANCES_AVG',
 'ENTRANCES_MEDI',
 'ENTRANCES_MODE',
 'Education_NAME_GOODS_CATEGORY_PA_sum',
 'Electricity_ORGANIZATION_TYPE',
 'Emergency_ORGANIZATION_TYPE',
 'Everyday expenses_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'FLAG_CONT_MOBILE',
 'FLAG_DOCUMENT_15',
 'FLAG_DOCUMENT_2',
 'FLAG_DOCUMENT_5',
 'FLAG_DOCUMENT_9',
 'FLAG_EMP_PHONE',
 'FLOORSMAX_AVG',
 'FLOORSMAX_MEDI',
 'FLOORSMAX_MODE',
 'FLOORSMIN_AVG',
 'FLOORSMIN_MEDI',
 'FLOORSMIN_MODE',
 'Fitness_NAME_GOODS_CATEGORY_PA_sum',
 'Furniture_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Gardening_NAME_GOODS_CATEGORY_PA_sum',
 'Gasification / water supply_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Group of people_NAME_TYPE_SUITE_PA_sum',
 'Hobby_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'House Construction_NAME_GOODS_CATEGORY_PA_sum',
 'Industry: type 10_ORGANIZATION_TYPE',
 'Industry: type 11_ORGANIZATION_TYPE',
 'Industry: type 13_ORGANIZATION_TYPE',
 'Industry: type 2_ORGANIZATION_TYPE',
 'Industry: type 6_ORGANIZATION_TYPE',
 'Industry: type 7_ORGANIZATION_TYPE',
 'Industry: type 8_ORGANIZATION_TYPE',
 'Insurance_NAME_GOODS_CATEGORY_PA_sum',
 'Insurance_ORGANIZATION_TYPE',
 'Interbank credit_CREDIT_TYPE_B_sum',
 'Journey_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'LANDAREA_AVG',
 'LANDAREA_MEDI',
 'LANDAREA_MODE',
 'LIVE_CITY_NOT_WORK_CITY',
 'LIVINGAPARTMENTS_AVG',
 'LIVINGAPARTMENTS_MEDI',
 'LIVINGAPARTMENTS_MODE',
 'LIVINGAREA_AVG',
 'LIVINGAREA_MEDI',
 'LIVINGAREA_MODE',
 'Legal Services_ORGANIZATION_TYPE',
 'Loan for business development_CREDIT_TYPE_B_sum',
 'Loan for purchase of shares (margin lending)_CREDIT_TYPE_B_sum',
 'Loan for the purchase of equipment_CREDIT_TYPE_B_sum',
 'Loan for working capital replenishment_CREDIT_TYPE_B_sum',
 'MLM partners_NAME_SELLER_INDUSTRY_PA_sum',
 'MONTHS_BALANCE_CCB_max_PA_min',
 'MONTHS_BALANCE_CCB_sum_PA_sum',
 'Maternity leave_NAME_INCOME_TYPE',
 'Medical Supplies_NAME_GOODS_CATEGORY_PA_sum',
 'Medicine_NAME_GOODS_CATEGORY_PA_sum',
 'Medicine_ORGANIZATION_TYPE',
 'Mobile operator loan_CREDIT_TYPE_B_sum',
 'Mobile_ORGANIZATION_TYPE',
 'Money for a third person_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Monolithic_WALLSMATERIAL_MODE',
 'NONLIVINGAPARTMENTS_AVG',
 'NONLIVINGAPARTMENTS_MEDI',
 'NONLIVINGAPARTMENTS_MODE',
 'NONLIVINGAREA_AVG',
 'NONLIVINGAREA_MEDI',
 'NONLIVINGAREA_MODE',
 'NUM_INSTALMENT_NUMBER_IP_min_PA_max',
 'NUM_INSTALMENT_VERSION_IP_min_PA_min',
 'Office Appliances_NAME_GOODS_CATEGORY_PA_sum',
 'Other_B_NAME_TYPE_SUITE',
 'Other_NAME_GOODS_CATEGORY_PA_sum',
 'POS others without interest_PRODUCT_COMBINATION_PA_sum',
 'Payments on other loans_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Private service staff_OCCUPATION_TYPE',
 'Purchase of electronic equipment_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Real estate loan_CREDIT_TYPE_B_sum',
 'Refusal to name the goal_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'Refused_NAME_CONTRACT_STATUS_CCB_sum_PA_max',
 'Refused_NAME_CONTRACT_STATUS_CCB_sum_PA_min',
 'Refused_NAME_CONTRACT_STATUS_CCB_sum_PA_sum',
 'Religion_ORGANIZATION_TYPE',
 'Returned to the store_NAME_CONTRACT_STATUS_PCB_sum_PA_min',
 'SK_DPD_CCB_max_PA_max',
 'SK_DPD_CCB_sum_PA_sum',
 'SK_DPD_DEF_CCB_max_PA_sum',
 'SK_DPD_DEF_PCB_min_PA_max',
 'SK_DPD_DEF_PCB_min_PA_sum',
 'SK_DPD_PCB_min_PA_max',
 'SK_DPD_PCB_min_PA_sum',
 'SYSTEM_CODE_REJECT_REASON_PA_sum',
 'Sent proposal_NAME_CONTRACT_STATUS_CCB_sum_PA_max',
 'Sent proposal_NAME_CONTRACT_STATUS_CCB_sum_PA_min',
 'Sent proposal_NAME_CONTRACT_STATUS_CCB_sum_PA_sum',
 'Services_ORGANIZATION_TYPE',
 'Signed_NAME_CONTRACT_STATUS_CCB_sum_PA_max',
 'Signed_NAME_CONTRACT_STATUS_CCB_sum_PA_min',
 'Signed_NAME_CONTRACT_STATUS_CCB_sum_PA_sum',
 'TOTALAREA_MODE',
 'Telecom_ORGANIZATION_TYPE',
 'Tourism_NAME_GOODS_CATEGORY_PA_sum',
 'Tourism_NAME_SELLER_INDUSTRY_PA_sum',
 'Trade: type 1_ORGANIZATION_TYPE',
 'Trade: type 4_ORGANIZATION_TYPE',
 'Trade: type 5_ORGANIZATION_TYPE',
 'Trade: type 6_ORGANIZATION_TYPE',
 'Transport: type 1_ORGANIZATION_TYPE',
 'Transport: type 2_ORGANIZATION_TYPE',
 'Unemployed_NAME_INCOME_TYPE',
 'University_ORGANIZATION_TYPE',
 'Unknown type of loan_CREDIT_TYPE_B_sum',
 'Vehicles_NAME_GOODS_CATEGORY_PA_sum',
 'Weapon_NAME_GOODS_CATEGORY_PA_sum',
 'XNA_NAME_CLIENT_TYPE_PA_sum',
 'XNA_NAME_CONTRACT_STATUS_PCB_sum_PA_max',
 'XNA_NAME_CONTRACT_STATUS_PCB_sum_PA_sum',
 'XNA_NAME_CONTRACT_TYPE_PA_sum',
 'YEARS_BEGINEXPLUATATION_AVG',
 'YEARS_BEGINEXPLUATATION_MEDI',
 'YEARS_BEGINEXPLUATATION_MODE',
 'YEARS_BUILD_AVG',
 'YEARS_BUILD_MEDI',
 'YEARS_BUILD_MODE',
 'currency 4_CREDIT_CURRENCY_B_sum',
 'missing_NAME_TYPE_SUITE',
 'missing_PRODUCT_COMBINATION_PA_sum',
 '4_STATUS_BB_sum_B_sum',
 'Industry: type 5_ORGANIZATION_TYPE',
 'N_FLAG_LAST_APPL_PER_CONTRACT_PA_sum',
 'AMT_RECEIVABLE_PRINCIPAL_CCB_sum_PA_sum',
 'Security_ORGANIZATION_TYPE',
 'CNT_DRAWINGS_OTHER_CURRENT_CCB_sum_PA_sum',
 'Jewelry_NAME_SELLER_INDUSTRY_PA_sum',
 'Completed_NAME_CONTRACT_STATUS_CCB_sum_PA_max']

listfin=list((set(listetodel)^set(X_train))&set(X_train))
X_train=X_train[listfin]
X_val=X_val[listfin]
X_test=X_test[listfin]
gc.collect()
timer("Variables selected")

#Modèle LGBM
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#Jeu d'entraînement final
X_train=pd.concat([X_train,X_val],axis=0)
y_train=pd.concat([y_train,y_val],axis=0)
del X_val,y_val

gc.collect() 
timer("Data ready for modelling")
vect=np.copy(y_train)
vect[vect==1]=4
vect[vect==0]=1
clf = LGBMClassifier(learning_rate =0.075, num_boost_round=1500,  nthread=8, seed=27,colsample_bytree=1, max_depth=3,
                     min_child_weight=87.5467,min_split_gain=0.0950,num_leaves=22,reg_alpha=0.0019,reg_lambda=0.0406,subsample=0.8709)
clf.fit(X_train, y_train, eval_metric= 'auc', verbose= 100)
timer("Model created.")

#Submission
sub=pd.Series(clf.predict_proba(X_test)[:,1],name="TARGET")
sub.loc[sub<0]=0
sub.loc[sub>1]=1
sub.index=indtest.index
submission=pd.concat([indtest,sub],axis=1)
submission.to_csv('submission.csv', index=False)
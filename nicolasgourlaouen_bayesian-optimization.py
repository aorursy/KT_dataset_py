#Data preprocessing - 7 mn
from sklearn.model_selection import train_test_split
import time
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import collections
import gc
deb=time.time()
path="../input"

#Dictionary which contains the keys of every table. {table:(PRIMARY_KEY,LIST_OF_KEYS)}
listfiles={"application_train":("SK_ID_CURR",["SK_ID_CURR"]),"application_test":("SK_ID_CURR",["SK_ID_CURR"]),"bureau":("SK_ID_CURR",["SK_ID_CURR","SK_ID_BUREAU"]),
           "bureau_balance":("SK_ID_BUREAU",["SK_ID_BUREAU"]),"POS_CASH_balance":("SK_ID_PREV",["SK_ID_CURR","SK_ID_PREV"]),"previous_application":("SK_ID_CURR",["SK_ID_CURR","SK_ID_PREV"]),
           "installments_payments":("SK_ID_PREV",["SK_ID_CURR","SK_ID_PREV"]),"credit_card_balance":("SK_ID_PREV",["SK_ID_CURR","SK_ID_PREV"])}

#Function which displays execution time
ftime=[("Start",time.time(),0)]
def timer(title,showtime=True):
    global ftime
    newtime=(title,time.time(),time.time()-ftime[-1][1])
    ftime.append(newtime)
    if showtime==True:
        print(newtime[0] + " in " + str(np.round(newtime[2],2)))
    else:
        print(newtime[0]) 

#Reading of the file; they are placed in a dictionary
data={x:(pd.read_csv(path +"/"+ x + '.csv')).sort_values(listfiles[x][0],ascending=True) for x in listfiles}
for i in listfiles:
    data[i].index=list(range(data[i].shape[0]))

#Concatenation of training and testing sets for preprocessing
data["full"]=pd.concat([data["application_train"],data["application_test"]],axis=0,sort=True)
data["full"].index=list(range(data["full"].shape[0]))
train_shape,test_shape=data["application_train"].shape[0],data["application_test"].shape[0]
del data["application_train"],data["application_test"]
gc.collect()
timer("Files read and sorted by key")

#Function which distinguishes continuous, categorical and binary features
def set_groups(dataf):
    num=dataf.columns[(dataf.dtypes!="object").tolist()].tolist() #Numerical features
    nom=dataf.columns[(dataf.dtypes=="object").tolist()].tolist() #Object features
    continuous,discrete,dichot = [],nom,[]
    for i in num:
        notnull=dataf[i][pd.isnull(dataf[i])==False]
        if len(set(notnull.loc[:200]))>2: continuous.append(i) 
        #Numerical features with more than 2 distinct values are considered continuous
        #We first only look at the 200 first values in order to limit calculation time
            
        else:
            setvar=set(notnull)
            if len(setvar)>2: continuous.append(i)
            elif setvar==set([0,1]): dichot.append(i)
            #and dataf[i].count()==dataf.shape[0]
            else: discrete.append(i)
    timer("Groups set")
    return(continuous, discrete,dichot)

"""Agreggation function, which takes as input the table to agreggate (which has to be sorted by key), aggregation key 
and the first 2 letters of the name of the table (to rename features).
The purpose of this function is to divide the table into 10 subparts which are agreggated individually and then concatenated 
to limit memory usage. This allows us to avoid memory issues."""
def agregate(dataf,key,table): 
    continuous, discrete,dichot=set_groups(dataf)
    if len(dichot) > 0 : dichot = dichot + [key]
    vals=sorted(list(set(dataf[key]))) #sorted values of the features (in each subpart we will take 1/10 of these values)
    nbint=10 #Number of subparts
    pace=len(vals)//nbint+nbint #pace, number of distinct values in each subpart
    indmin=dataf[key].index[0]
    dicodf={}
    for k,i in enumerate(list(range(pace,len(vals)+pace,pace-1))):
        valmax=vals[i-pace:i-1][-1] #last value of the split
        indmax=dataf[key][dataf[key]==valmax].index.tolist()[-1] #max index : index of the last line whose value is valmax 
        dicosets={}
        if len(continuous) > 0 : dicosets["continuous"]=dataf[continuous].loc[indmin:indmax,:].groupby(key).agg(["sum","max","min"]) #Agreggation of continuous features
        if len(dichot) > 0 : dicosets["dichot"]=dataf[dichot].loc[indmin:indmax,:].groupby(key).agg(["sum"]) #Agreggation of binary features
        dicosets["count"]=pd.DataFrame(dataf.loc[indmin:indmax,:].groupby(key)[key].agg("count")) #Count of the number of rows per key
        ind=pd.MultiIndex(levels=[[table],["count"]],labels=[[0],[0]]) 
        dicosets["count"].columns=ind
        for x in list(dicosets): dicosets[x].columns=[x[0]+"_"+x[1] for x in dicosets[x].columns] #Renaming of the columns
        dicodf[k]=pd.concat([dicosets[x] for x in list(dicosets)],axis=1) #Concatenation of binary features, continuous features and count
        del dicosets
        indmin=indmax+1 #max index for next subpart
    dataf_tsf=pd.concat([dicodf[k] for k in list(dicodf)],axis=0) #Concatenation of the 10 subparts
    timer("Agreggation done")
    return dataf_tsf

#This function gets dummy features from a categorical feature 
def binar(series,table):
    setserie=set(series)
    if len(setserie)==2:
        var=pd.DataFrame(pd.get_dummies(series).iloc[:,0])
    else: var=pd.get_dummies(series)
    name=series.name
    var.columns=[str(x) + "_"+ name for x in var.columns]
    return var

#This function preprocesses categorical features (filling missing values and getting dummy features)
def fill_values(dataf,table):
        dataf=dataf.fillna('missing')
        timer("Missing values filled for discrete vars")
        return pd.concat([binar(dataf[x],table) for x in list(dataf)],axis=1)

#This function preprocesses a table: it defines features types, agreggates and renames the features
def process(dataf,keys,keyforagg,name):
    liste=list(set(dataf)^set(keys)) #Features that are not a key
    dataf=dataf[liste+[keyforagg]] #Features that are not a key + primary key
    continuous, discrete,dichot =set_groups(dataf[liste])
    if len (discrete)>0:
        var=fill_values(dataf[discrete],name)
        dataf=dataf.drop(columns=discrete)
        dataf=pd.concat([dataf,var],axis=1)
    liste=dataf.columns.tolist()
    for i in range(len(liste)):
        if liste[i] not in keys: liste[i]= str(liste[i]+"_" +name) #Renaming
    dataf.columns=liste
    dataf=agregate(dataf,keyforagg,name)
    dataf[keyforagg]=dataf.index.tolist()
    dataf.index=list(range(dataf.shape[0]))
    return dataf

#First letters of the tables to rename the features 
diconame={"bureau":"B","bureau_balance":"BB","POS_CASH_balance":"PCB","previous_application":"PA","installments_payments":"IP","credit_card_balance":"CCB"}

#List of the successive merging between tables that will be made
links=[('bureau_balance','bureau'),('installments_payments','previous_application'),('credit_card_balance','previous_application'),('POS_CASH_balance','previous_application'),
      ('previous_application','full'),('bureau','full')]
for i in links:
    timer("Processing " + i[0],showtime=False)
    data[i[0]]=process(data[i[0]],listfiles[i[0]][1],listfiles[i[0]][0],diconame[i[0]])
    gc.collect()
    timer("Merging " + i[0] + " with " +i[1],showtime=False)
    print(data[i[0]].shape)
    data[i[1]]=data[i[1]].merge(right=data[i[0]],how='left',on=listfiles[i[0]][0]) #Merging
    del data[i[0]]
    timer("Merging completed")
    print(data[i[1]].shape)
    gc.collect()

#Processing of the final testing set
columnswithoutkeys=list(set(data["full"])^set(["SK_ID_CURR"]))
continuous, discrete,dichot =set_groups(data["full"][columnswithoutkeys])
var=fill_values(data["full"][discrete],"Main") 
data["full"]=data["full"].drop(columns=discrete)
data["full"]=pd.concat([data["full"],var],axis=1)
gc.collect()
print('total preprocessing time : ' + str(time.time()-deb))

#Feature Engineering, inspired by https://www.kaggle.com/ogrellier/lighgbm-with-selected-features
data["full"]["ANNUITY_INCOME_RATIO"]=data["full"]["AMT_ANNUITY"]/data["full"]["AMT_INCOME_TOTAL"]
data["full"]['CREDIT_ANNUITY_RATIO'] = data["full"]['AMT_CREDIT'] / data["full"]['AMT_ANNUITY']
data["full"]['EXT_SOURCES_PRODUCT'] = data["full"]['EXT_SOURCE_1'] * data["full"]['EXT_SOURCE_2'] * data["full"]['EXT_SOURCE_3']
data["full"]['EXT_SOURCES_MEAN'] = data["full"][['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
data["full"]['CREDIT_GROODS_PRICE_RATIO'] = data["full"]['AMT_CREDIT'] / data["full"]['AMT_GOODS_PRICE']
data["full"]['DAYS_EMPLOYED_DAYS_BIRTH_RATIO'] = data["full"]['DAYS_EMPLOYED'] / data["full"]['DAYS_BIRTH']
data["full"]['DAYS_LAST_PHONE_CHANGE_DAYS_EMPLOYED_RATIO'] = data["full"]['DAYS_LAST_PHONE_CHANGE'] / data["full"]['DAYS_EMPLOYED']
data["full"]['EXT_SOURCES_STD'] = data["full"][['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
data["full"]["DOCUMENTS_SUM"]=data["full"][['FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6',
                                       'FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11',
                                       'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
                                       'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']].mean(axis=1)
data["full"]["REQ_CREDIT_BUREAU_DAY_SUM"]=data["full"][['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',
                                            'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].sum(axis=1)
data["full"]["AMT_INCOME_TOTAL_CNT_CHILDREN_RATIO"]=data["full"]["AMT_INCOME_TOTAL"]/data["full"]["CNT_CHILDREN"]
data["full"]["CONTACT_FLAGS_MEAN"]=data["full"][['FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL']].mean(axis=1)
data["full"]["LIVE_WORK_PLACE_MEAN"]=data["full"][["REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].mean(axis=1)
data["full"]["LIVE_WORK_REGION_MEAN"]=data["full"][['REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION']].mean(axis=1)
data["full"]["REGION_RATING_MEAN"]=data["full"][["REGION_RATING_CLIENT","REGION_RATING_CLIENT_W_CITY"]].mean(axis=1)

from sklearn.model_selection import train_test_split
#Validation split
liste=list(filter(lambda x :x !="TARGET",list(data["full"]))) #List of features for training set

X_train, X_val, y_train, y_val = train_test_split((data["full"].iloc[: train_shape,:])[liste], data["full"]["TARGET"].iloc[:train_shape], test_size=0.2, random_state=42)
del data["full"]
gc.collect()

#Deletion of the features whose importance is < 5  
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
 'Completed_NAME_CONTRACT_STATUS_CCB_sum_PA_max',
 'Co-op apartment_NAME_HOUSING_TYPE',
 'AMT_RECIVABLE_CCB_min_PA_sum',
 'Education_NAME_CASH_LOAN_PURPOSE_PA_sum',
 'IT staff_OCCUPATION_TYPE',
 'Businessman_NAME_INCOME_TYPE',
 'Group of people_NAME_TYPE_SUITE',
 'FLAG_DOCUMENT_17',
 'SK_DPD_CCB_min_PA_max',
 'FLAG_DOCUMENT_7',
 'FLAG_DOCUMENT_19',
 'NUM_INSTALMENT_NUMBER_IP_min_PA_min',
 'SK_DPD_DEF_PCB_min_PA_min',
 'SK_DPD_DEF_CCB_min_PA_min',
 'FLAG_DOCUMENT_21',
 'FLAG_DOCUMENT_12',
 'FLAG_DOCUMENT_4',
 'XNA_NAME_CONTRACT_STATUS_PCB_sum_PA_min',
 'FLAG_MOBIL',
 'SK_DPD_CCB_min_PA_min',
 'Other_A_NAME_TYPE_SUITE',
 'SK_DPD_PCB_min_PA_min',
 'SK_DPD_DEF_CCB_min_PA_sum',
 'SK_DPD_CCB_min_PA_sum',
 'HR staff_OCCUPATION_TYPE',
 'Realty agents_OCCUPATION_TYPE',
 'SK_DPD_DEF_CCB_min_PA_max',
 'Secretaries_OCCUPATION_TYPE',
 'Student_NAME_INCOME_TYPE',
 'Unknown_NAME_FAMILY_STATUS',
 'XNA_CODE_GENDER',
 'FLAG_DOCUMENT_20',
 'FLAG_DOCUMENT_10']
listfin=list((set(listetodel)^set(X_train))&set(X_train))
X_train=X_train[listfin]
X_val=X_val[listfin]
gc.collect()
timer("Features selected")
#Bayesian Optimization 
#Inspired by https://github.com/fmfn/BayesianOptimization/blob/master/examples/xgboost_example.py
from bayes_opt import BayesianOptimization
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import lightgbm

def lgbm_evaluate(min_child_weight, colsample_bytree, max_depth, subsample, reg_alpha, min_split_gain, num_leaves, reg_lambda):
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['num_leaves'] = int(num_leaves)
    params['min_split_gain'] = max(min(min_split_gain, 1), 0)
    cv_result = lightgbm.cv(params, dset, num_boost_round=num_rounds, nfold=5,
             seed=seed,early_stopping_rounds=50,metrics=params["eval_metric"])
    return cv_result['auc-mean'][-1]

dset=lightgbm.Dataset(X_train,y_train)
num_rounds = 3000
seed=27
num_iter = 1 #The number of init points and iterations is reduced here to limit calculation time
init_points = 1
params = {'eta': 0.1,'eval_metric': 'auc','verbose_eval': True,'seed': seed,'learning_rate':0.075}

xgbBO = BayesianOptimization(lgbm_evaluate, {'min_child_weight': (1, 100),
                                            'colsample_bytree': (0.1, 1),
                                            'max_depth': (3, 10),
                                            'subsample': (0.5, 1),
                                            'reg_alpha':(0.001,0.1),
                                             'num_leaves':(5,30),
                                             'reg_lambda':(0.001,0.1),
                                             'min_split_gain':(0.001,0.1)})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    xgbBO.maximize(init_points=init_points, n_iter=num_iter)

timer("Bayesian optimization done")
"""Results after 5 init points + 23 iterations (LR 0.075)
Initialization
------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |   colsample_bytree |   max_depth |   min_child_weight |   min_split_gain |   num_leaves |   reg_alpha |   reg_lambda |   subsample |
1 | 07m15s |    0.78273 |             0.2230 |      5.0221 |            60.8569 |           0.0702 |      27.5701 |      0.0303 |       0.0908 |      0.7513 |
2 | 08m53s |    0.78333 |             0.4374 |      5.7995 |            83.2610 |           0.0605 |      22.0037 |      0.0342 |       0.0478 |      0.7822 |
3 | 07m21s |    0.78038 |             0.4037 |      7.3570 |             4.6034 |           0.0884 |      26.5799 |      0.0134 |       0.0556 |      0.5838 |
4 | 09m05s |    0.78229 |             0.3912 |      7.5857 |            61.2343 |           0.0591 |      14.9409 |      0.0952 |       0.0982 |      0.5772 |
5 | 08m10s |    0.78247 |             0.8361 |      5.7197 |            35.7080 |           0.0634 |      24.9980 |      0.0434 |       0.0201 |      0.8601 |
Bayesian Optimization
------------------------------------------------------------------------------------------------------------------------------------------------------------------
 Step |   Time |      Value |   colsample_bytree |   max_depth |   min_child_weight |   min_split_gain |   num_leaves |   reg_alpha |   reg_lambda |   subsample |
6 | 12m21s |    0.78259 |             0.3025 |      3.0498 |             2.1907 |           0.0548 |       5.1220 |      0.0620 |       0.0389 |      0.9556 |
7 | 15m50s |    0.78349 |             0.9750 |      3.0477 |            96.6492 |           0.0338 |       5.7353 |      0.0150 |       0.0389 |      0.9321 |
8 | 11m23s |    0.78329 |             0.8831 |      3.4421 |            99.6374 |           0.0482 |       28.7641 |      0.0032 |       0.0337 |      0.5488 | 
9 | 07m02s |    0.78119 |             0.6957 |      9.6027 |            99.7800 |           0.0454 |       29.3847 |      0.0203 |   0.0387 |      0.9947 | 
10 | 12m20s |    0.78376 |             0.9828 |      3.0181 |            69.1695 |           0.0747 |      19.2079 |    0.0920 |       0.0126 |      0.9660 | 
11 | 10m45s |    0.78340 |             0.1279 |      3.3402 |            98.4080 |           0.0853 |      17.4850 |0.0691 |       0.0399 |      0.9555 | 
12 | 11m43s |    0.78327 |             0.7992 |      3.0025 |            30.3096 |           0.0507 |      17.1483 |      0.0314 |       0.0125 |0.5029 | 
13 | 13m29s |    0.78305 |             0.1602 |      3.0050 |            75.7882 |           0.0140 |       5.2456 |      0.0715 |       0.0193 |      0.6042 |
14 | 11m09s |    0.78339 |             0.9921 |      3.0894 |            82.7038 |           0.0268 |      16.4945 |      0.0943 |0.0990 |      0.5637 | 
15 | 14m28s |    0.78200 |             0.6964 |      9.6255 |            19.6806 |           0.0692|       5.0011 |      0.0034 |       0.0369 |      0.9083 | 
16 | 11m46s |    0.78382 |             0.5720 |      3.0847 |            87.7469 |           0.0615 |      29.6365 |      0.0718 |       0.0079 |      0.9715 |
17 | 13m45s |    0.78393 |             0.7771 |      3.3805 |            42.2123 |           0.0909 |       7.6089 |      0.0012 |       0.0163 |      0.9653 |
18 | 13m20s |    0.78360 |             0.1744 |      3.1154 |            78.5347 |           0.0215 |      21.4539 |      0.0038 |       0.0087 |      0.9634| 
19 | 16m21s |    0.78305 |             0.9683 |      9.0393 |            85.6758 |           0.0940 |       5.6542 |0.0254 |       0.0055 |      0.8214 | 
20 | 13m34s |    0.78286 |             0.9842 |      3.4246 |            32.6800 |           0.0880 |       5.0256 |      0.0154 |       0.0137 |      0.6332 |
21 | 11m15s |    0.78290 |             0.1251 |      3.0474 |            20.4025 |           0.0842 |      14.4449 |      0.0824 |       0.0836 |      0.9790 |
22 | 12m30s |    0.78297 |             0.9195 |      3.0675 |             1.3053 |           0.0789 |      15.6313 |      0.0380 |       0.0585 |      0.9143 |
23 | 11m44s |    0.78342 |             0.3581 |      3.2036 |            45.8195 |           0.0916 |      20.6174 |      0.0198 |       0.0091 |      0.9412 |
24 | 09m48s |    0.78079 |             0.8315 |      9.4943 |             1.0214 |           0.0894 |       9.5139 |      0.0592 |       0.0515 |      0.9795 |
25 | 09m30s |    0.78261 |             0.9753 |      3.0721 |            22.3576 |           0.0967 |      28.5416 |      0.0605 |       0.0514 |      0.9965 |
26 | 12m16s |    0.78335 |             0.9697 |      3.0857 |            61.0296 |           0.0454 |       6.2085 |      0.0292 |       0.0106 |      0.9881 |
27 | 12m24s |    0.78372 |             0.9712 |      3.4425 |            75.9584 |           0.0760 |      28.3897 |      0.0261 |       0.0546 |      0.6105 |
28 | 11m59s |    0.78389 |             0.9868 |      3.3512 |            87.5467 |           0.0950 |      21.6978 |      0.0019 |       0.0406 |      0.8709 |
Total processing time: 321 min
"""
#Final LGBM model with optimized parameters
clf = LGBMClassifier(learning_rate =0.075, num_boost_round=1500,  nthread=8, seed=27,colsample_bytree=1, max_depth=3,
                     min_child_weight=87.5467,min_split_gain=0.0950,num_leaves=22,reg_alpha=0.0019,reg_lambda=0.0406,subsample=0.8709)
clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric= 'auc', verbose= 100, early_stopping_rounds= 50)
#AUC score is measured on both training and validation sets. It is displayed every 100 iterations 
#Training is stopped if there has not been any improvement of AUC score on validation set in 50 iterations

score=np.round(roc_auc_score(y_val, clf.predict_proba(X_val)[:,1]),4) 
print("Final score on validation set: " + str(np.round(score,3)))
timer("Model created.")
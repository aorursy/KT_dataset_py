import pandas as pd
import sklearn.model_selection as skms
import numpy as np
import sklearn.ensemble as ske
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import sklearn.decomposition as skd
import sklearn.linear_model as sklm
import lightgbm as lgb 
mode = "lgb"
def get_test_train_data():
    train_data = pd.read_csv("../input/application_train.csv")
   
    test_data = pd.read_csv("../input/application_test.csv")
    categorical_col = ["NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
                   "NAME_TYPE_SUITE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS",
                   "NAME_HOUSING_TYPE","OCCUPATION_TYPE","WEEKDAY_APPR_PROCESS_START","ORGANIZATION_TYPE",
                   "FONDKAPREMONT_MODE","HOUSETYPE_MODE","WALLSMATERIAL_MODE","EMERGENCYSTATE_MODE"]
    
    train_data_categorical = pd.get_dummies(train_data[categorical_col])
    train_data = train_data.drop(categorical_col, axis=1)
    train_data[train_data_categorical.columns] = train_data_categorical
    train_data = train_data.fillna(0)
    
    test_data_categorical = pd.get_dummies(test_data[categorical_col])
    test_data = test_data.drop(categorical_col, axis=1)
    test_data[test_data_categorical.columns] = test_data_categorical
    test_data= test_data.fillna(0)
    
    col_names = list(train_data.columns.values)
    test_col_names = list(test_data.columns.values)
    col_names.remove("SK_ID_CURR")
    col_names.remove("TARGET")
    for col in col_names:
        if col not in test_col_names:
            print("removing col as not found in test ", col)
            col_names.remove(col)
    return train_data, test_data, col_names
def get_test_train_split(train_data, test_data, col_names):
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    
    train_feature_data = train_data[col_names]
    test_feature_data = test_data[col_names]
    
    train_label_data = np.array(train_data["TARGET"])
    train_feature_data = np.array(train_feature_data)
    
    X_train, X_valid, y_train, y_valid = skms.train_test_split(train_feature_data, train_label_data, test_size=0.2)
    
    X_test = np.array(test_feature_data)
    return X_train, X_valid, y_train, y_valid, X_test
def get_model(X_train, y_train, num_leaves,max_depth,max_bin, learning_rate,num):
    if mode == "lgb":
        lgb_data=lgb.Dataset(X_train,label=y_train)
        param = {'num_leaves':num_leaves, 'objective':'binary','max_depth':max_depth,'learning_rate':learning_rate,'max_bin':max_bin}
        param['metric'] = ['auc','binary_logloss']
        num_round=num

        lgbm=lgb.train(param,lgb_data,num_round)
        return lgbm
    else:
        lrmodel = ske.RandomForestClassifier(n_estimators=800,max_depth=10,min_samples_split=2,min_samples_leaf=1)
        #lrmodel = sklm.LogisticRegression(penalty="l1")
        lrmodel.fit(X_train, y_train)
        return lrmodel
def print_auc(lrmodel, X_valid, y_valid):
    if mode == "lgb":
        pred=lrmodel.predict(X_valid)
    else:
        pred = lrmodel.predict_proba(X_valid)[:,1]
   # print(pred)
    auc = skm.roc_auc_score(y_valid, pred)
  #  print("ROC AUC on valid data is:",auc)
    return auc
def gen_result(lrmodel, X_test):
    if mode == "lgb":
        pred = lrmodel.predict(X_test)
    else:
        pred = lrmodel.predict_proba(X_test)[:,1]
    result = {"SK_ID_CURR":np.array(test_data["SK_ID_CURR"])}
    result["TARGET"] = list(pred)
    df_result = pd.DataFrame(result)
    df_result.to_csv("credit_result.csv", index=False)
    
def print_cols(data):
    coldata = data.head(10).T
    batch_size = 10
    tot_cols = coldata.shape[0]
    num_batch = tot_cols//batch_size
    for i in range(num_batch+1):
        print("Batch: ", i)
        print(coldata[i*batch_size:(i + 1)*batch_size])
def AddCountFeature(data, colName, train_data, test_data):
    train_data = train_data.reset_index().set_index("SK_ID_CURR")
    test_data = test_data.reset_index().set_index("SK_ID_CURR")
    if data.index.name == "SK_ID_CURR":
        data = data.reset_index()
    data[colName] = 0
    grouped_data = data[["SK_ID_CURR",colName]].groupby(["SK_ID_CURR"]).count()
    train_data = train_data.join(grouped_data)
    test_data = test_data.join(grouped_data)
    custom_features.append(colName)
    return train_data, test_data
def AddGroupedFeature(filter_name, data, agg, train_data, test_data):
    train_data = train_data.reset_index().set_index("SK_ID_CURR")
    test_data = test_data.reset_index().set_index("SK_ID_CURR")
    data = data.reset_index()
    
    grouped_data = data.groupby(["SK_ID_CURR"]).agg(agg)
    print(grouped_data.columns.values)
    grouped_data.columns = grouped_data.columns.map(lambda x: filter_name + "_" + x[0]+"_"+x[1])
    colNames = grouped_data.columns.values
    print(grouped_data.columns.values)
    train_data = train_data.join(grouped_data).fillna(0)
    test_data = test_data.join(grouped_data).fillna(0)
    for col in colNames:
        custom_features.append(col)
    return train_data, test_data
    
def gen_custom_features(train_data, test_data):
    custom_features = []
    if 1==2:
        gen_ratio_col(train_data, test_data, custom_features,"CNT_CHILDREN","CNT_FAM_MEMBERS","CHILD_FAM_RATIO")
        gen_ratio_col(train_data, test_data, custom_features,"AMT_CREDIT","AMT_INCOME_TOTAL","CREDIT_INCOME_RATIO")
        gen_ratio_col(train_data, test_data, custom_features,"AMT_ANNUITY","AMT_INCOME_TOTAL","ANNUITY_INCOME_RATIO")
        
        feature_cols = ["AMT_INCOME_TOTAL", "DAYS_BIRTH","DAYS_EMPLOYED","YEARS_BUILD_AVG","LIVINGAREA_AVG",
                               "NAME_CONTRACT_TYPE_Cash loans","NAME_CONTRACT_TYPE_Revolving loans",
                               "CODE_GENDER_F","CODE_GENDER_M", "FLAG_OWN_CAR_N","FLAG_OWN_CAR_Y",
                               "FLAG_OWN_REALTY_N","FLAG_OWN_REALTY_Y","EMERGENCYSTATE_MODE_Yes",
                               "NAME_EDUCATION_TYPE_Academic degree",#"NAME_INCOME_TYPE_Maternity leave",
                               "NAME_INCOME_TYPE_Student","NAME_INCOME_TYPE_Unemployed",
                               "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]
    else:
        feature_cols = ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","DAYS_BIRTH","AMT_GOODS_PRICE",
                      "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
                      'CNT_FAM_MEMBERS', 'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'ELEVATORS_AVG', 'FLAG_OWN_REALTY_Y']                                                     
    for col in feature_cols:
        custom_features.append(col)
    
    return custom_features
def add_bureau_features(train_data, test_data):
    bureau_data = pd.read_csv("../input/bureau.csv")
    bureau_data = bureau_data.fillna(0)
    print(bureau_data["CREDIT_TYPE"].unique())
    print(bureau_data["CREDIT_ACTIVE"].unique())

    bureau_data["CREDIT_DAYS"] = bureau_data["DAYS_CREDIT_ENDDATE"] -bureau_data["DAYS_CREDIT"]

    loc_non_creditcard = bureau_data["CREDIT_TYPE"]!="Credit card"
    bureau_data.loc[loc_non_creditcard, "CREDIT_TYPE"] = "Non Credit Card"

    loc_sold_bad_loan = bureau_data["CREDIT_ACTIVE"] == "Sold"
    bureau_data.loc[loc_sold_bad_loan, "CREDIT_ACTIVE"] = "Bad"

    loc_sold_bad_loan = bureau_data["CREDIT_ACTIVE"] == "Bad debt"
    bureau_data.loc[loc_sold_bad_loan, "CREDIT_ACTIVE"] = "Bad"

    print(bureau_data["CREDIT_TYPE"].unique())
    print(bureau_data["CREDIT_ACTIVE"].unique())
    agg = {"AMT_CREDIT_SUM":["sum"],
           "CREDIT_DAYS":["sum"],
           "AMT_CREDIT_MAX_OVERDUE":["max"],
           "CREDIT_DAY_OVERDUE":"max",
           "SK_ID_BUREAU":"count"
          }

    print("Tot bureau rows",len(bureau_data))

    datalist = []
    loc_credit_card = bureau_data["CREDIT_TYPE"]=="Credit card" 
    bureau_data_for_credit_card = bureau_data[loc_credit_card]
    datalist.append(("CC", bureau_data_for_credit_card))
    loc_notcredit_card = bureau_data["CREDIT_TYPE"]!="Credit card" 
    bureau_data_for_notcredit_card = bureau_data[loc_notcredit_card]
    datalist.append(("NonCC", bureau_data_for_notcredit_card))

    for credit_type_data_touple in datalist:
        credit_type = credit_type_data_touple[0]
        credit_type_data = credit_type_data_touple[1]
        print("Tot bureau credit card rows",len(credit_type_data))

        status_list = ["Closed","Bad","Active"]
        for status in status_list:
            credit_type_data_for_status = \
                credit_type_data[credit_type_data["CREDIT_ACTIVE"] == status]
            print("Tot bureau credit card " + status + " loan rows",len(credit_type_data_for_status))
            filter_name = credit_type + "_" + status
            train_data, test_data = AddGroupedFeature(filter_name, credit_type_data_for_status, 
                                agg,train_data, test_data)


    #Credit_card_closed_loan_totamount,Credit_card_closed_loan_totduration
    #Credit_card_closed_loan_delayedamt,
    if 1==2:
          
        custom_features.remove("NonCC_Closed_AMT_CREDIT_SUM_sum")
        custom_features.remove("NonCC_Closed_CREDIT_DAYS_sum")
        custom_features.remove("NonCC_Closed_SK_ID_BUREAU_count") 
        custom_features.remove("AMT_ANNUITY")
        custom_features.remove("CC_Closed_AMT_CREDIT_MAX_OVERDUE_max")
        custom_features.remove("NonCC_Closed_AMT_CREDIT_MAX_OVERDUE_max")
        custom_features.remove("FLAG_OWN_REALTY_Y")
        custom_features.remove("CC_Bad_CREDIT_DAYS_sum")    
        custom_features.remove("CC_Active_SK_ID_BUREAU_count")
        custom_features.remove("NonCC_Active_CREDIT_DAY_OVERDUE_max")
        
    #custom_features.remove("CC_Closed_CREDIT_DAY_OVERDUE_max")
    #custom_features.remove("CC_Bad_SK_ID_BUREAU_count")
    #custom_features.remove("AMT_INCOME_TOTAL") 
    #custom_features.remove("CC_Active_AMT_CREDIT_MAX_OVERDUE_max")
    return train_data, test_data, bureau_data


def add_bureau_bal_features(train_data, test_data, bureau_data):
    bureau_balance_data = pd.read_csv("../input/bureau_balance.csv")
    bureau_balance_data = bureau_balance_data.fillna(0)
    bureau_balance_data.set_index("SK_ID_BUREAU")
    if bureau_data.index.name != "SK_ID_BUREAU":
        bureau_data = bureau_data.set_index("SK_ID_BUREAU")
    bureau_balance_data = bureau_balance_data.join(bureau_data[["SK_ID_CURR"]])
    bureau_balance_data = bureau_balance_data.reset_index()
    agg = {
           "SK_ID_BUREAU":"count"
          }

    loc_status5 = bureau_balance_data["STATUS"] == "5"
    loc_status4 = bureau_balance_data["STATUS"] == "4"
    loc_status3 = bureau_balance_data["STATUS"] == "3"
    loc_status2 = bureau_balance_data["STATUS"] == "2"
    bureau_balance_data.loc[loc_status5,"STATUS"] = "1"
    bureau_balance_data.loc[loc_status4,"STATUS"] = "1"
    bureau_balance_data.loc[loc_status3,"STATUS"] = "1"
    bureau_balance_data.loc[loc_status2,"STATUS"] = "1"
    
    datalist = [] #[("all", bureau_balance_data)]
    loc_mon1 = bureau_balance_data["MONTHS_BALANCE"]>=-6
    bureau_balance_data_for_mon1 = bureau_balance_data[loc_mon1]
    datalist.append(("mon1", bureau_balance_data_for_mon1))

    for data_touple in datalist:
        mon_type = data_touple[0]
        mon_data = data_touple[1]
        status_list = mon_data["STATUS"].unique()
        print(status_list)
        for status in ["1"]: #status_list:
            mon_data_for_status = mon_data[mon_data["STATUS"] == status].reset_index()

            filter_name = mon_type + "_" + status
            print(filter_name)
            train_data, test_data = AddCountFeature(mon_data_for_status, filter_name+"_"+"count", train_data, test_data)


    #Credit_card_closed_loan_totamount,Credit_card_closed_loan_totduration
    #Credit_card_closed_loan_delayedamt,
    return train_data, test_data, bureau_balance_data
def add_credit_card_features(train_data, test_data):
    credit_card_balance_data = pd.read_csv("../input/credit_card_balance.csv")
    credit_card_balance_data = credit_card_balance_data.fillna(0)
    credit_card_balance_data["Available_Credit"] = credit_card_balance_data["AMT_CREDIT_LIMIT_ACTUAL"] - \
                                                credit_card_balance_data["AMT_BALANCE"]
    
    monloc = credit_card_balance_data["MONTHS_BALANCE"] >= -12
    credit_card_balance_data_recent = credit_card_balance_data.loc[monloc]
    
    
    agg = {
           "AMT_CREDIT_LIMIT_ACTUAL":["sum","mean"],
           "AMT_PAYMENT_TOTAL_CURRENT":["sum","mean"],
           "AMT_TOTAL_RECEIVABLE":["sum","mean"],
           "SK_DPD":["sum","mean"],
           "Available_Credit":["sum","mean"],
           "SK_ID_CURR":["count"]
          }
    
    list_status = credit_card_balance_data_recent["NAME_CONTRACT_STATUS"].unique()
    print(list_status)
    for status in ["Active"]: #list_status:
        statusloc = credit_card_balance_data_recent["NAME_CONTRACT_STATUS"] == status
        credit_card_balance_data_recent_for_status = credit_card_balance_data_recent[statusloc]
        train_data, test_data = AddGroupedFeature("credit_card_"+status, credit_card_balance_data_recent_for_status, 
                                agg,train_data, test_data)
        
    return train_data, test_data, credit_card_balance_data
def add_POS_CASH_features(train_data, test_data):
    POS_CASH_balance_data = pd.read_csv("../input/POS_CASH_balance.csv")
    POS_CASH_balance_data = POS_CASH_balance_data.fillna(0)
    
    loc_recent = POS_CASH_balance_data["MONTHS_BALANCE"] == -1
    POS_CASH_balance_data_curr = POS_CASH_balance_data[loc_recent]
    agg = {"CNT_INSTALMENT_FUTURE":"sum"}
    train_data, test_data = AddGroupedFeature("curr_pos_cash", POS_CASH_balance_data_curr, 
                                agg,train_data, test_data)
        
    loc_recent = POS_CASH_balance_data["MONTHS_BALANCE"] >= -12
    POS_CASH_balance_data_recent = POS_CASH_balance_data[loc_recent]
    agg = {"SK_DPD":"sum"}
    train_data, test_data = AddGroupedFeature("recent_pos_cash", POS_CASH_balance_data_recent, 
                                agg,train_data, test_data)
    return train_data, test_data, POS_CASH_balance_data
def add_prev_appl_features(train_data, test_data):
    previous_application_data = pd.read_csv("../input/previous_application.csv")
    previous_application_data = previous_application_data.fillna(0)
    loc_unused  = previous_application_data["NAME_CONTRACT_STATUS"] == "Unused offer"
    previous_application_data.loc[loc_unused, "NAME_CONTRACT_STATUS"] = "Approved"
    status_list = previous_application_data["NAME_CONTRACT_STATUS"].unique()
    agg = {"AMT_APPLICATION":{"sum":"sum"},
           "AMT_CREDIT":{"sum":"sum"},
           "SK_ID_PREV":{"count":"count"}
          }
    for status in status_list:
        status_loc = previous_application_data["NAME_CONTRACT_STATUS"] == status
        previous_application_data_for_status = previous_application_data[status_loc]

        train_data, test_data = AddGroupedFeature("prev_appl_"+status, previous_application_data_for_status, 
                                    agg,train_data, test_data)
    custom_features.remove("prev_appl_Canceled_AMT_CREDIT_sum")
    return train_data, test_data, previous_application_data
def add_inst_features(train_data, test_data):
    installments_payments_data = pd.read_csv("../input/installments_payments.csv")
    installments_payments_data = installments_payments_data.fillna(0)
    
    installments_payments_data["DIFF_DAYS"] = installments_payments_data["DAYS_INSTALMENT"] - \
                                            installments_payments_data["DAYS_ENTRY_PAYMENT"]
    installments_payments_data["DIFF_AMT"] = installments_payments_data["AMT_INSTALMENT"] - \
                                            installments_payments_data["AMT_PAYMENT"]
    installments_payments_data["DIFF_DAYS"] = \
              installments_payments_data["DIFF_DAYS"].apply(lambda x: -x if x < 0 else 0)
    installments_payments_data["DIFF_AMT"] = \
              installments_payments_data["DIFF_DAYS"].apply(lambda x: x if x > 0 else 0)
    
    #I want to know how was the payment history in last six months
    #That is in last six months how many installments were delayed and by how much amount
    
    sixmon_loc = installments_payments_data["DAYS_INSTALMENT"] >= -180
    installments_payments_data_sixmon = installments_payments_data[sixmon_loc]
    
    delayed_loc = installments_payments_data_sixmon["DIFF_DAYS"] > 0
    installments_payments_data_sixmon_delayed = installments_payments_data_sixmon[delayed_loc]
    agg = { 
             "AMT_INSTALMENT":["mean"],
             "DIFF_DAYS" : ["mean"],
             "SK_ID_PREV":["count"]
          }
    train_data, test_data = AddGroupedFeature("inst_delayed" , installments_payments_data_sixmon_delayed, 
                                    agg,train_data, test_data)
    
    less_loc = installments_payments_data_sixmon["DIFF_AMT"] > 0
    installments_payments_data_sixmon_less = installments_payments_data_sixmon[less_loc]
    agg = { 
             "DIFF_AMT" : ["sum"],
             "SK_ID_PREV":["count"]
          }
    train_data, test_data = AddGroupedFeature("inst_less" , installments_payments_data_sixmon_less, 
                                    agg,train_data, test_data)
    
    ontime_loc = installments_payments_data_sixmon["DIFF_DAYS"] == 0
    installments_payments_data_sixmon_ontime = installments_payments_data_sixmon[ontime_loc]
    agg = { 
             "AMT_INSTALMENT":["mean"],
             "DIFF_DAYS" : ["mean"],
             "SK_ID_PREV":["count"]
          }
    train_data, test_data = AddGroupedFeature("inst_ontime" , installments_payments_data_sixmon_ontime, 
                                    agg,train_data, test_data)
    
    more_loc = installments_payments_data_sixmon["DIFF_AMT"] > 0
    installments_payments_data_sixmon_more = installments_payments_data_sixmon[more_loc]
    agg = { 
             "DIFF_AMT" : ["sum"],
             "SK_ID_PREV":["count"]
          }
    train_data, test_data = AddGroupedFeature("inst_more" , installments_payments_data_sixmon_more, 
                                    agg,train_data, test_data)
    
    return train_data, test_data, installments_payments_data
train_data, test_data, col_names = get_test_train_data()
custom_features = gen_custom_features(train_data, test_data)
train_data, test_data, bureau_data = add_bureau_features(train_data, test_data)#0.736 to 0.74
train_data, test_data, bureau_balance_data = add_bureau_bal_features(train_data, test_data, bureau_data)
#0.74 to 0.75
train_data, test_data, credit_card_data = add_credit_card_features(train_data, test_data)
#0.75 to 0.757
train_data, test_data, pos_cash_data = add_POS_CASH_features(train_data, test_data)
#0.757 to 0.76
train_data, test_data, previous_application_data = add_prev_appl_features(train_data, test_data)
#0.76 to 0.765
train_data, test_data, inst_data = add_inst_features(train_data, test_data)
#0.76 to 0.765
#What ratio of installments are delayed
#What ratio of installments are short in amount

X_train, X_valid, y_train, y_valid, X_test = get_test_train_split(train_data, test_data, custom_features)
model = get_model(X_train, y_train,80,10,90,0.02,1000)
auc = print_auc(model, X_valid, y_valid)
print("auc",auc)
if 1==2:
    for col in train_data.columns:
        if col not in custom_features:
            if col != "TARGET" and col != "CODE_GENDER_XNA" and col in test_data.columns:
                custom_features.append(col)
                X_train, X_valid, y_train, y_valid, X_test = get_test_train_split(train_data, test_data, custom_features)
                model = get_model(X_train, y_train)
                aucnew = print_auc(model, X_valid, y_valid)
                if aucnew - auc >= 0:
                    custom_features.remove(col)
                else:
                    auc = aucnew
                    print(col, aucnew)
                    
if 1==2:
    for col in custom_features:
        custom_features.remove(col)
        X_train, X_valid, y_train, y_valid, X_test = get_test_train_split(train_data, test_data, custom_features)
        model = get_model(X_train, y_train,80,10,90,0.02)
        aucnew = print_auc(model, X_valid, y_valid)
        if aucnew - auc < 0:
            custom_features.append(col)
        else:
            auc = aucnew
            print(col, aucnew)
            modelnew=model
            
print("auc",auc)     
test_data = test_data.reset_index()
gen_result(model, X_test)
print(custom_features)

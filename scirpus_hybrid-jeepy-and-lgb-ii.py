import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import roc_auc_score
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
def UseGPFeatures(data):
    v = pd.DataFrame()
    v["i0"] = np.tanh((((((-1.0*((((np.maximum(((data["EXT_SOURCE_2"])), ((data["EXT_SOURCE_3"])))) - (data["te_OCCUPATION_TYPE"])))))) - (((((((data["EXT_SOURCE_2"]) + (data["EXT_SOURCE_3"]))) * 2.0)) + (data["EXT_SOURCE_3"]))))) * 2.0)) 
    v["i1"] = np.tanh(((((data["te_OCCUPATION_TYPE"]) + (((((data["NAME_CONTRACT_STATUS_Refused"]) - (((((data["EXT_SOURCE_3"]) * 2.0)) * 2.0)))) + (((data["te_ORGANIZATION_TYPE"]) - (((((data["EXT_SOURCE_1"]) * 2.0)) * 2.0)))))))) * 2.0)) 
    v["i2"] = np.tanh(((((((data["te_OCCUPATION_TYPE"]) - (0.686567))) - (((((((data["EXT_SOURCE_3"]) + (data["EXT_SOURCE_1"]))) + (data["EXT_SOURCE_2"]))) * 2.0)))) * 2.0)) 
    v["i3"] = np.tanh(((((((((((data["NAME_CONTRACT_STATUS_Refused"]) + (((data["te_OCCUPATION_TYPE"]) - (data["EXT_SOURCE_2"]))))) - (0.686567))) - (data["EXT_SOURCE_3"]))) * 2.0)) - (((data["EXT_SOURCE_2"]) + (data["EXT_SOURCE_3"]))))) 
    v["i4"] = np.tanh(((((((data["DAYS_CREDIT"]) - (data["EXT_SOURCE_3"]))) - (data["EXT_SOURCE_3"]))) + (((np.where(data["EXT_SOURCE_1"]>0, -3.0, data["DAYS_BIRTH"] )) - (data["EXT_SOURCE_1"]))))) 
    v["i5"] = np.tanh(((((data["NAME_CONTRACT_STATUS_Refused"]) + (((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) + (((data["te_NAME_EDUCATION_TYPE"]) + (((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) - ((((((data["EXT_SOURCE_2"]) * 2.0)) + (data["EXT_SOURCE_2"]))/2.0)))))))) * 2.0)))) * 2.0)) 
    v["i6"] = np.tanh(((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) + ((((((((((data["NAME_PRODUCT_TYPE_walk_in"]) - (data["EXT_SOURCE_1"]))) + (np.where(data["EXT_SOURCE_1"]>0, -2.0, data["te_ORGANIZATION_TYPE"] )))/2.0)) - (data["EXT_SOURCE_3"]))) * 2.0)))) 
    v["i7"] = np.tanh((-1.0*((((((((data["EXT_SOURCE_2"]) + (((data["DAYS_FIRST_DRAWING"]) + (data["EXT_SOURCE_3"]))))) * 2.0)) + (((data["EXT_SOURCE_2"]) + (((0.697674) + (data["EXT_SOURCE_1"])))))))))) 
    v["i8"] = np.tanh(((data["REGION_RATING_CLIENT_W_CITY"]) + (((((((((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) + (data["te_NAME_EDUCATION_TYPE"]))) - (data["EXT_SOURCE_1"]))) - (data["EXT_SOURCE_3"]))) - (data["EXT_SOURCE_2"]))) * 2.0)))) 
    v["i9"] = np.tanh(((data["NAME_CONTRACT_STATUS_Refused"]) + (((((((((((((data["cc_bal_AMT_BALANCE"]) - (data["EXT_SOURCE_2"]))) - (data["CODE_GENDER"]))) + (data["cc_bal_AMT_BALANCE"]))) - (0.416667))) * 2.0)) - (data["EXT_SOURCE_3"]))))) 
    v["i10"] = np.tanh(((((((((((data["ty__Microloan"]) * 2.0)) * (((14.800000) / 2.0)))) - (data["EXT_SOURCE_2"]))) + (((data["FLAG_DOCUMENT_3"]) - (data["EXT_SOURCE_3"]))))) + (data["CNT_INSTALMENT_FUTURE"]))) 
    v["i11"] = np.tanh(((((data["te_ORGANIZATION_TYPE"]) - (np.where((((data["EXT_SOURCE_1"]) + (data["NAME_CONTRACT_TYPE"]))/2.0)>0, 3.0, data["EXT_SOURCE_1"] )))) + (((data["NAME_YIELD_GROUP_high"]) - (data["EXT_SOURCE_2"]))))) 
    v["i12"] = np.tanh(((((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (((data["te_FLAG_OWN_CAR"]) + (((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) - (((data["SK_ID_PREV_y"]) * 2.0)))) + (data["CNT_INSTALMENT_FUTURE"]))))))) - (data["EXT_SOURCE_3"]))) + (data["CNT_INSTALMENT_FUTURE"]))) 
    v["i13"] = np.tanh(((data["te_CODE_GENDER"]) + (((np.maximum(((data["NAME_PRODUCT_TYPE_walk_in"])), ((((data["REG_CITY_NOT_LIVE_CITY"]) + (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"])))))) + (((((((data["cc_bal_CNT_DRAWINGS_CURRENT"]) + (((data["ty__Microloan"]) * 2.0)))) * 2.0)) * 2.0)))))) 
    v["i14"] = np.tanh(((((((((((((((data["ty__Microloan"]) + (data["te_NAME_EDUCATION_TYPE"]))/2.0)) + (data["te_NAME_CONTRACT_TYPE"]))/2.0)) - (data["AMT_ANNUITY"]))) + (data["CNT_PAYMENT"]))) * 2.0)) - (((data["FLAG_OWN_CAR"]) + (data["AMT_ANNUITY"]))))) 
    v["i15"] = np.tanh((((((((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) - (data["DAYS_FIRST_DRAWING"]))) + (((data["DAYS_ID_PUBLISH"]) + (((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) * 2.0)))))/2.0)) - (np.tanh((((data["ca__Closed"]) * 2.0)))))) + (data["cc_bal_AMT_BALANCE"]))) 
    v["i16"] = np.tanh(np.where(data["NAME_YIELD_GROUP_low_action"]>0, ((data["NAME_CONTRACT_STATUS_Refused"]) - (3.0)), ((((data["REGION_RATING_CLIENT_W_CITY"]) + (((data["avg_buro_buro_bal_status_1"]) + ((((data["NAME_YIELD_GROUP_high"]) + (data["NAME_CONTRACT_STATUS_Refused"]))/2.0)))))) + (data["avg_buro_buro_bal_status_1"])) )) 
    v["i17"] = np.tanh(((np.maximum(((np.maximum(((data["CNT_PAYMENT"])), ((data["te_NAME_FAMILY_STATUS"]))))), ((data["NAME_CLIENT_TYPE_New"])))) + (((np.minimum(((data["AMT_ANNUITY_x"])), ((((np.maximum(((data["ty__Microloan"])), ((data["CNT_PAYMENT"])))) - (data["RATE_DOWN_PAYMENT"])))))) * 2.0)))) 
    v["i18"] = np.tanh((((((((((((((data["ty__Microloan"]) * (11.714300))) - (data["ty__Mortgage"]))) + (data["te_WALLSMATERIAL_MODE"]))/2.0)) - (data["inst_NUM_INSTALMENT_VERSION"]))) - (data["PRODUCT_COMBINATION_POS_industry_with_interest"]))) - (((data["NAME_GOODS_CATEGORY_Furniture"]) * 2.0)))) 
    v["i19"] = np.tanh(((((data["SK_ID_PREV_y"]) + (-3.0))) * (((((data["SK_ID_PREV_y"]) - (data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]))) + (((0.797872) + (data["NAME_YIELD_GROUP_low_action"]))))))) 
    v["i20"] = np.tanh(((((data["avg_buro_buro_bal_status_1"]) + (((data["avg_buro_buro_bal_status_1"]) + (((data["PRODUCT_COMBINATION_Cash_X_Sell__high"]) + ((((((((data["ty__Microloan"]) * 2.0)) * 2.0)) + (((data["CNT_INSTALMENT_FUTURE"]) * (data["CNT_INSTALMENT_FUTURE"]))))/2.0)))))))) * 2.0)) 
    v["i21"] = np.tanh(((data["DAYS_LAST_PHONE_CHANGE"]) + (((((((11.714300) * (((((((data["OWN_CAR_AGE"]) + (data["CODE_REJECT_REASON_SCOFR"]))) * 2.0)) - (((data["inst_AMT_PAYMENT"]) - (data["CODE_REJECT_REASON_SCOFR"]))))))) * 2.0)) / 2.0)))) 
    v["i22"] = np.tanh(((0.576923) + (((data["AMT_ANNUITY_x"]) + (((((data["AMT_ANNUITY_x"]) + (np.maximum(((data["AMT_ANNUITY_x"])), ((np.maximum(((data["DEF_60_CNT_SOCIAL_CIRCLE"])), ((data["CNT_INSTALMENT_FUTURE"]))))))))) - (data["CHANNEL_TYPE_Channel_of_corporate_sales"]))))))) 
    v["i23"] = np.tanh(((data["SK_DPD"]) + ((((((((data["te_NAME_EDUCATION_TYPE"]) + (data["te_FLAG_OWN_CAR"]))/2.0)) - (data["RATE_DOWN_PAYMENT"]))) + (np.where(data["te_NAME_FAMILY_STATUS"]>0, 0.797872, ((data["SK_DPD"]) - (0.494118)) )))))) 
    v["i24"] = np.tanh(((((((data["te_ORGANIZATION_TYPE"]) + (data["CODE_REJECT_REASON_HC"]))/2.0)) + (((data["te_CODE_GENDER"]) - (((data["NAME_YIELD_GROUP_low_normal"]) + (((data["CHANNEL_TYPE_Channel_of_corporate_sales"]) + (((data["ty__Mortgage"]) * (data["ty__Mortgage"]))))))))))/2.0)) 
    v["i25"] = np.tanh((((((((1.220590) - (data["FLOORSMAX_MODE"]))) + (data["CNT_INSTALMENT_FUTURE"]))/2.0)) + (((14.800000) * (np.maximum(((data["ty__Microloan"])), ((data["SK_DPD"])))))))) 
    v["i26"] = np.tanh((-1.0*(((((((0.494118) + (((((np.minimum(((data["inst_AMT_PAYMENT"])), ((data["MONTHS_BALANCE"])))) * 2.0)) + (data["NAME_YIELD_GROUP_low_action"]))))/2.0)) + (np.maximum(((data["CHANNEL_TYPE_Channel_of_corporate_sales"])), ((((data["AMT_CREDIT_SUM_LIMIT"]) * 2.0)))))))))) 
    v["i27"] = np.tanh((((np.tanh((data["PRODUCT_COMBINATION_Cash_Street__low"]))) + ((-1.0*((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) + (((((data["PRODUCT_COMBINATION_Cash_Street__low"]) + (((data["ORGANIZATION_TYPE"]) - (data["CNT_PAYMENT"]))))) - (data["cc_bal_CNT_DRAWINGS_CURRENT"])))))))))/2.0)) 
    v["i28"] = np.tanh((((((0.062500) + (data["DAYS_REGISTRATION"]))/2.0)) - (np.where((((data["SK_ID_PREV_y"]) + ((((data["FLAG_DOCUMENT_18"]) + (data["Active"]))/2.0)))/2.0)>0, data["Active"], ((data["PRODUCT_COMBINATION_Cash_X_Sell__low"]) + (data["FLAG_DOCUMENT_16"])) )))) 
    v["i29"] = np.tanh((((((data["NAME_FAMILY_STATUS"]) + (np.minimum(((data["PRODUCT_COMBINATION_Cash_Street__low"])), (((-1.0*((np.maximum(((data["ty__Mortgage"])), ((((np.maximum(((data["PRODUCT_COMBINATION_Cash_Street__low"])), (((-1.0*((data["te_FLAG_DOCUMENT_18"]))))))) / 2.0))))))))))))/2.0)) - (data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"]))) 
    v["i30"] = np.tanh(np.where((-1.0*((data["cc_bal_AMT_TOTAL_RECEIVABLE"])))>0, data["CNT_PAYMENT"], (((((((data["FLAG_DOCUMENT_13"]) < (np.tanh((((data["SK_DPD"]) * (0.797872))))))*1.)) + (data["ca__Sold"]))) + (data["SK_DPD"])) )) 
    v["i31"] = np.tanh((((data["cc_bal_AMT_BALANCE"]) + ((-1.0*((np.maximum(((data["CODE_REJECT_REASON_SCO"])), (((((np.maximum(((data["FLAG_PHONE"])), ((data["FLAG_DOCUMENT_18"])))) + (np.minimum(((data["cc_bal_AMT_RECIVABLE"])), ((data["FLAG_PHONE"])))))/2.0)))))))))/2.0)) 
    v["i32"] = np.tanh((((((((((data["NAME_PAYMENT_TYPE_XNA"]) - (data["FLAG_DOCUMENT_13"]))) - (data["te_FLAG_CONT_MOBILE"]))) + (((((data["DAYS_ID_PUBLISH"]) + (data["te_FLAG_WORK_PHONE"]))) - (data["ty__Car_loan"]))))/2.0)) / 2.0)) 
    v["i33"] = np.tanh((((((data["avg_buro_buro_bal_status_0"]) < (-1.0))*1.)) - (np.maximum(((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) * 2.0))), (((((np.minimum((((3.99197196960449219))), ((data["FLAG_DOCUMENT_16"])))) + (data["DAYS_ENDDATE_FACT"]))/2.0))))))) 
    v["i34"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, ((0.323944) - (data["AMT_ANNUITY_x"])), np.minimum(((((data["AMT_ANNUITY_x"]) + (0.323944)))), (((((1.105260) > (np.maximum(((data["FLAG_DOCUMENT_13"])), ((data["AMT_ANNUITY_x"])))))*1.)))) )) 
    v["i35"] = np.tanh(np.where(data["AMT_GOODS_PRICE_x"]>0, (-1.0*((data["AMT_GOODS_PRICE_x"]))), ((data["AMT_GOODS_PRICE_x"]) + ((((0.036585) > (((np.maximum(((np.tanh((data["AMT_GOODS_PRICE_x"])))), ((-2.0)))) - (data["CHANNEL_TYPE_AP___Cash_loan_"]))))*1.))) )) 
    v["i36"] = np.tanh(((data["DAYS_EMPLOYED"]) + (((((0.494118) + (data["DAYS_EMPLOYED"]))) * (np.where(data["FLAG_DOCUMENT_8"]>0, 0.494118, ((14.800000) * (1.857140)) )))))) 
    v["i37"] = np.tanh(np.minimum(((0.494118)), ((np.minimum(((((np.minimum((((1.0))), (((((data["te_NAME_INCOME_TYPE"]) + (0.680000))/2.0))))) + (data["AMT_CREDIT_x"])))), (((((np.tanh((data["AMT_ANNUITY_x"]))) > (data["AMT_CREDIT_x"]))*1.)))))))) 
    v["i38"] = np.tanh(((((np.where(data["PRODUCT_COMBINATION_Cash_X_Sell__middle"]>0, data["CNT_PAYMENT"], (-1.0*((np.maximum(((((data["CHANNEL_TYPE_AP___Cash_loan_"]) * (data["PRODUCT_COMBINATION_Cash_X_Sell__middle"])))), ((((data["AMT_REQ_CREDIT_BUREAU_QRT"]) * (11.714300)))))))) )) * 2.0)) - (data["AMT_REQ_CREDIT_BUREAU_QRT"]))) 
    v["i39"] = np.tanh(((data["NAME_GOODS_CATEGORY_Direct_Sales"]) - ((((np.maximum(((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])), ((((0.494118) * (data["NAME_TYPE_SUITE_Unaccompanied"])))))) + ((((data["Active"]) + (data["SK_ID_PREV_y"]))/2.0)))/2.0)))) 
    v["i40"] = np.tanh((((((0.576923) - (np.maximum(((data["inst_NUM_INSTALMENT_VERSION"])), (((((data["NFLAG_INSURED_ON_APPROVAL"]) < (((((data["SK_DPD"]) * 2.0)) * 2.0)))*1.))))))) + ((((data["SK_DPD"]) + (((data["NFLAG_INSURED_ON_APPROVAL"]) / 2.0)))/2.0)))/2.0)) 
    v["i41"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, 11.714300, (((((data["inst_AMT_PAYMENT"]) < (np.tanh(((-1.0*((np.maximum(((np.minimum(((0.576923)), ((0.680000))))), ((((data["te_REG_REGION_NOT_LIVE_REGION"]) * 2.0)))))))))))*1.)) * 2.0) )) 
    v["i42"] = np.tanh((-1.0*(((((data["FLAG_DOCUMENT_18"]) + ((((np.minimum(((0.697674)), ((((np.tanh(((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) < (data["cc_bal_SK_DPD_DEF"]))*1.)))) - (data["te_FLAG_OWN_REALTY"])))))) < (data["te_OCCUPATION_TYPE"]))*1.)))/2.0))))) 
    v["i43"] = np.tanh((((((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) * (data["te_ORGANIZATION_TYPE"]))) + (((((((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) + (data["te_ORGANIZATION_TYPE"]))/2.0)) + (np.where(data["cc_bal_AMT_BALANCE"]>0, data["te_CNT_CHILDREN"], (((data["FLAG_DOCUMENT_2"]) + (data["cc_bal_AMT_TOTAL_RECEIVABLE"]))/2.0) )))/2.0)))/2.0)) 
    v["i44"] = np.tanh(np.where(data["CNT_INSTALMENT_FUTURE"]>0, (((0.062500) > ((-1.0*((data["te_WEEKDAY_APPR_PROCESS_START"])))))*1.), ((np.minimum(((data["ty__Mortgage"])), ((np.tanh(((-1.0*(((((data["ty__Mortgage"]) > (data["cc_bal_SK_DPD_DEF"]))*1.)))))))))) * 2.0) )) 
    v["i45"] = np.tanh(((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + ((-1.0*((data["CODE_GENDER"])))))/2.0)) + (np.where(data["CODE_GENDER"]>0, (((-1.0*((data["PRODUCT_COMBINATION_Cash_X_Sell__low"])))) * (1.105260)), data["DAYS_FIRST_DUE"] )))/2.0)) 
    v["i46"] = np.tanh((-1.0*((((np.tanh(((((((((data["avg_buro_buro_bal_status_0"]) + ((((((data["avg_buro_buro_bal_status_0"]) + (data["avg_buro_buro_bal_status_0"]))) < (data["FLAG_DOCUMENT_13"]))*1.)))/2.0)) * 2.0)) - (0.043478))))) - (0.043478)))))) 
    v["i47"] = np.tanh((((((((data["FLAG_DOCUMENT_2"]) - (np.minimum(((data["inst_AMT_PAYMENT"])), ((((0.036585) * 2.0))))))) > (0.576923))*1.)) - (np.maximum(((data["YEARS_BUILD_MODE"])), ((((((data["inst_AMT_PAYMENT"]) / 2.0)) / 2.0))))))) 
    v["i48"] = np.tanh(np.where(data["AMT_CREDIT_x"]>0, 0.339286, (((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) + ((((data["DEF_30_CNT_SOCIAL_CIRCLE"]) + (((data["AMT_CREDIT_x"]) + ((((((data["AMT_CREDIT_x"]) > (data["NAME_YIELD_GROUP_high"]))*1.)) - (data["NAME_YIELD_GROUP_high"]))))))/2.0)))/2.0) )) 
    v["i49"] = np.tanh((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + (((((-1.0*(((((((data["te_REG_REGION_NOT_LIVE_REGION"]) + (data["te_LIVE_REGION_NOT_WORK_REGION"]))/2.0)) / 2.0))))) + ((-1.0*((((np.maximum(((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])), ((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])))) / 2.0))))))/2.0)))/2.0)) * 2.0)) 
    v["i50"] = np.tanh(((((-1.0*((data["CHANNEL_TYPE_Channel_of_corporate_sales"])))) + (np.minimum(((((3.0) - (data["PRODUCT_COMBINATION_Cash_Street__low"])))), (((-1.0*((((data["AMT_CREDIT_SUM_LIMIT"]) + ((((data["WEEKDAY_APPR_PROCESS_START_SATURDAY"]) + (data["CHANNEL_TYPE_Channel_of_corporate_sales"]))/2.0)))))))))))/2.0)) 
    v["i51"] = np.tanh(((((((((np.minimum(((data["EXT_SOURCE_2"])), ((((-2.0) * (data["EXT_SOURCE_2"])))))) - (-2.0))) * (data["EXT_SOURCE_2"]))) * 2.0)) * 2.0)) 
    v["i52"] = np.tanh(((np.minimum(((np.minimum((((-1.0*((data["FLAG_DOCUMENT_13"]))))), ((data["FLAG_DOCUMENT_13"]))))), ((data["NAME_GOODS_CATEGORY_Direct_Sales"])))) + ((((data["DAYS_LAST_PHONE_CHANGE"]) > (np.tanh((((3.0) - (1.220590))))))*1.)))) 
    v["i53"] = np.tanh(((np.minimum(((np.minimum(((((data["FLAG_DOCUMENT_16"]) * (data["WEEKDAY_APPR_PROCESS_START_SUNDAY"])))), (((-1.0*((data["FLAG_DOCUMENT_14"])))))))), ((((data["FLAG_DOCUMENT_16"]) * (data["NAME_CLIENT_TYPE_Refreshed"])))))) * 2.0)) 
    v["i54"] = np.tanh(np.minimum((((-1.0*((data["cc_bal_cc_bal_status__Sent_proposal"]))))), ((((((np.where(data["cc_bal_cc_bal_status__Sent_proposal"]>0, data["cc_bal_cc_bal_status__Sent_proposal"], (((0.031746) + (np.where(data["NONLIVINGAPARTMENTS_MODE"]>0, data["NONLIVINGAPARTMENTS_MODE"], data["AMT_CREDIT_y"] )))/2.0) )) * 2.0)) * 2.0))))) 
    v["i55"] = np.tanh((((((((((((data["Returned_to_the_store"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0)) - (data["ty__Mortgage"]))) + (data["cc_bal_CNT_DRAWINGS_CURRENT"]))/2.0)) + (((data["HOUR_APPR_PROCESS_START_y"]) * (((((data["Returned_to_the_store"]) * 2.0)) * 2.0)))))/2.0)) 
    v["i56"] = np.tanh(((np.where(data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]>0, (((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) > (0.323944))*1.)) * (((0.686567) - (data["REGION_POPULATION_RELATIVE"])))), 2.235290 )) * (((data["REGION_POPULATION_RELATIVE"]) + (0.323944))))) 
    v["i57"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, 1.105260, ((((((data["FLAG_DOCUMENT_2"]) / 2.0)) - (((((-1.0*(((((data["ca__Sold"]) + (data["FLAG_WORK_PHONE"]))/2.0))))) + (np.tanh((data["Active"]))))/2.0)))) / 2.0) )) 
    v["i58"] = np.tanh((-1.0*((np.maximum(((data["FLAG_DOCUMENT_11"])), ((np.maximum(((((data["FLAG_DOCUMENT_18"]) / 2.0))), (((((((np.minimum(((data["FLAG_DOCUMENT_18"])), ((((data["AMT_REQ_CREDIT_BUREAU_MON"]) * (data["FLAG_DOCUMENT_18"])))))) / 2.0)) < (data["NAME_GOODS_CATEGORY_Gardening"]))*1.))))))))))) 
    v["i59"] = np.tanh(np.where((((1.0) > ((((0.043478) < (((data["NAME_GOODS_CATEGORY_Direct_Sales"]) - (data["SK_DPD"]))))*1.)))*1.)>0, 3.526320, np.minimum(((data["te_NAME_HOUSING_TYPE"])), ((0.043478))) )) 
    v["i60"] = np.tanh(((data["cc_bal_SK_DPD_DEF"]) - ((((((0.697674) + (data["HOUSETYPE_MODE"]))/2.0)) * ((((((data["NONLIVINGAPARTMENTS_MODE"]) + (data["HOUSETYPE_MODE"]))/2.0)) * (data["nans"]))))))) 
    v["i61"] = np.tanh(((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + (((np.minimum(((data["FLAG_DOCUMENT_2"])), ((data["NAME_GOODS_CATEGORY_Direct_Sales"])))) + (np.tanh(((((np.where(data["inst_SK_ID_PREV"]>0, 0.680000, data["SK_ID_PREV_y"] )) < ((-1.0*((0.697674)))))*1.)))))))) 
    v["i62"] = np.tanh(np.minimum(((((data["SK_ID_PREV_x"]) * (((data["CNT_INSTALMENT_FUTURE"]) * 2.0))))), (((((((np.where(data["CNT_INSTALMENT_FUTURE"]>0, ((data["SK_ID_PREV_x"]) / 2.0), data["CODE_REJECT_REASON_SCO"] )) * (data["SK_ID_PREV_x"]))) + (data["CNT_PAYMENT"]))/2.0))))) 
    v["i63"] = np.tanh(np.minimum(((0.494118)), ((np.maximum(((data["te_NAME_FAMILY_STATUS"])), (((((data["PRODUCT_COMBINATION_Card_X_Sell"]) + ((((((((((data["te_NAME_FAMILY_STATUS"]) * 2.0)) * (data["PRODUCT_COMBINATION_Card_X_Sell"]))) + (data["REG_CITY_NOT_LIVE_CITY"]))) + (data["REG_CITY_NOT_LIVE_CITY"]))/2.0)))/2.0)))))))) 
    v["i64"] = np.tanh((((((data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"]) < (((1.220590) - (2.0))))*1.)) - ((((((data["cc_bal_cc_bal_status__Sent_proposal"]) + ((((data["SK_ID_PREV_y"]) + (data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))/2.0)))/2.0)) * (((2.0) / 2.0)))))) 
    v["i65"] = np.tanh(((data["cc_bal_AMT_DRAWINGS_OTHER_CURRENT"]) + (((((data["AMT_CREDIT_y"]) * ((((0.680000) < (((((-1.0) * 2.0)) + (data["AMT_CREDIT_y"]))))*1.)))) * (((data["AMT_CREDIT_y"]) + (data["AMT_CREDIT_y"]))))))) 
    v["i66"] = np.tanh((((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["FLAG_DOCUMENT_6"], ((((-1.0*((data["FLAG_DOCUMENT_11"])))) + (data["SK_ID_PREV_x"]))/2.0) )) + (np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, (-1.0*((data["te_FLAG_DOCUMENT_6"]))), data["cc_bal_CNT_DRAWINGS_CURRENT"] )))/2.0)) 
    v["i67"] = np.tanh(((data["EXT_SOURCE_3"]) - (((0.494118) - (((((data["EXT_SOURCE_3"]) + (((data["EXT_SOURCE_3"]) + (2.0))))) * (data["EXT_SOURCE_3"]))))))) 
    v["i68"] = np.tanh(((np.minimum((((((data["NAME_CONTRACT_STATUS_Approved"]) < (data["ca__Closed"]))*1.))), (((-1.0*((np.where((((data["ca__Closed"]) < (1.220590))*1.)>0, data["ca__Closed"], data["inst_AMT_PAYMENT"] )))))))) * 2.0)) 
    v["i69"] = np.tanh((((((data["CODE_REJECT_REASON_XAP"]) / 2.0)) + (np.where((((data["CODE_REJECT_REASON_XAP"]) > (0.576923))*1.)>0, (((data["SK_ID_BUREAU"]) < (((-2.0) / 2.0)))*1.), data["SK_ID_BUREAU"] )))/2.0)) 
    v["i70"] = np.tanh((((((0.697674) * 2.0)) + ((((((6.0)) * 2.0)) * (np.where(data["AMT_CREDIT_SUM_DEBT"]>0, 11.714300, data["AMT_CREDIT_SUM_DEBT"] )))))/2.0)) 
    v["i71"] = np.tanh((((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) < ((-1.0*((((data["AMT_CREDIT_SUM"]) + (np.where((-1.0*((data["cc_bal_SK_DPD_DEF"])))>0, 0.416667, np.tanh((0.416667)) ))))))))*1.)) * (2.235290))) 
    v["i72"] = np.tanh(((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + (((data["Returned_to_the_store"]) * (np.maximum(((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Furniture"])), ((0.680000))))), ((((((np.maximum(((data["ty__Car_loan"])), ((data["Returned_to_the_store"])))) * 2.0)) * 2.0))))))))) 
    v["i73"] = np.tanh((((((((((((-1.0*(((((1.105260) < (data["NAME_EDUCATION_TYPE"]))*1.))))) - (data["cc_bal_AMT_DRAWINGS_POS_CURRENT"]))) * (data["NAME_EDUCATION_TYPE"]))) - (data["cc_bal_AMT_DRAWINGS_POS_CURRENT"]))) * (data["NAME_EDUCATION_TYPE"]))) - (data["cc_bal_AMT_DRAWINGS_POS_CURRENT"]))) 
    v["i74"] = np.tanh((((((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) + ((((((((data["FLAG_DOCUMENT_2"]) - ((((data["DAYS_LAST_DUE_1ST_VERSION"]) + ((-1.0*((data["NAME_GOODS_CATEGORY_Insurance"])))))/2.0)))) * 2.0)) + (((data["SK_DPD"]) + (data["REGION_RATING_CLIENT_W_CITY"]))))/2.0)))/2.0)) / 2.0)) 
    v["i75"] = np.tanh((((-1.0*(((((data["cc_bal_SK_DPD_DEF"]) < (data["ty__Mortgage"]))*1.))))) - (((((np.minimum(((data["YEARS_BEGINEXPLUATATION_MODE"])), (((((data["ty__Mortgage"]) > (data["YEARS_BEGINEXPLUATATION_MODE"]))*1.))))) * 2.0)) * 2.0)))) 
    v["i76"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, ((((data["DAYS_LAST_DUE"]) - (data["DAYS_FIRST_DUE"]))) - (data["cc_bal_CNT_INSTALMENT_MATURE_CUM"])), ((((((data["DAYS_FIRST_DUE"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0)) > (1.220590))*1.) )) 
    v["i77"] = np.tanh(((np.where(data["AMT_REQ_CREDIT_BUREAU_DAY"]>0, data["AMT_REQ_CREDIT_BUREAU_DAY"], ((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) * (((np.where(data["cc_bal_AMT_TOTAL_RECEIVABLE"]>0, data["AMT_REQ_CREDIT_BUREAU_DAY"], 0.686567 )) + (data["MONTHS_BALANCE"])))) )) - (data["FLAG_DOCUMENT_15"]))) 
    v["i78"] = np.tanh((((((np.where(((data["cc_bal_SK_DPD_DEF"]) - (data["FLAG_DOCUMENT_2"]))>0, data["inst_AMT_PAYMENT"], data["te_FLAG_DOCUMENT_20"] )) < (((-1.0) - ((-1.0*(((((0.031746) + (1.0))/2.0))))))))*1.)) / 2.0)) 
    v["i79"] = np.tanh(np.minimum((((-1.0*((((((((1.27607381343841553)) < (data["NAME_YIELD_GROUP_high"]))*1.)) / 2.0)))))), ((((((data["AMT_CREDIT_x"]) + (0.797872))) + (((data["AMT_CREDIT_x"]) + (0.797872)))))))) 
    v["i80"] = np.tanh(((((((((1.0) - (data["NAME_CONTRACT_STATUS_Canceled"]))) < (data["CHANNEL_TYPE_AP___Cash_loan_"]))*1.)) < (np.where(np.tanh((data["NAME_CONTRACT_STATUS_Canceled"]))>0, data["CHANNEL_TYPE_AP___Cash_loan_"], (((((data["NONLIVINGAPARTMENTS_MODE"]) + (data["CHANNEL_TYPE_AP___Cash_loan_"]))/2.0)) / 2.0) )))*1.)) 
    v["i81"] = np.tanh(np.where(data["PRODUCT_COMBINATION_Cash_Street__low"]>0, np.where((((data["FLAG_DOCUMENT_2"]) < (data["ty__Microloan"]))*1.)>0, data["PRODUCT_COMBINATION_Cash_Street__low"], -2.0 ), (((data["NAME_GOODS_CATEGORY_Construction_Materials"]) > (data["cc_bal_SK_DPD_DEF"]))*1.) )) 
    v["i82"] = np.tanh(((((np.minimum(((0.0)), ((data["FLAG_DOCUMENT_14"])))) - (((data["FLAG_DOCUMENT_14"]) - ((((data["te_ORGANIZATION_TYPE"]) > ((((((2.0) + (data["te_NAME_CONTRACT_TYPE"]))) + (3.0))/2.0)))*1.)))))) * 2.0)) 
    v["i83"] = np.tanh(np.maximum(((data["FLAG_DOCUMENT_2"])), ((np.maximum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((np.maximum((((((((0.036585) < (data["cc_bal_SK_DPD_DEF"]))*1.)) * 2.0))), ((((data["ty__Microloan"]) - (((((6.0)) + ((6.0)))/2.0))))))))))))) 
    v["i84"] = np.tanh(((((0.0) - (((data["NAME_SELLER_INDUSTRY_Connectivity"]) * (((data["avg_buro_MONTHS_BALANCE"]) * (data["NAME_SELLER_INDUSTRY_Connectivity"]))))))) * (((((((data["NAME_SELLER_INDUSTRY_Connectivity"]) < (data["avg_buro_MONTHS_BALANCE"]))*1.)) < (data["NAME_SELLER_INDUSTRY_Connectivity"]))*1.)))) 
    v["i85"] = np.tanh(np.minimum((((-1.0*((data["NAME_GOODS_CATEGORY_Sport_and_Leisure"]))))), ((((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) + (data["NAME_GOODS_CATEGORY_Sport_and_Leisure"]))) * (data["WEEKDAY_APPR_PROCESS_START_TUESDAY"])))))) 
    v["i86"] = np.tanh((((((2.0) - (1.0))) < (np.where(data["CNT_INSTALMENT_FUTURE"]>0, (((((data["cc_bal_SK_DPD"]) - (((data["DAYS_FIRST_DUE"]) / 2.0)))) + (data["CNT_INSTALMENT_FUTURE"]))/2.0), data["DAYS_FIRST_DUE"] )))*1.)) 
    v["i87"] = np.tanh((((-1.0*((((((((np.minimum(((data["cc_bal_SK_DPD"])), (((-1.0*((data["inst_AMT_PAYMENT"]))))))) + (np.minimum(((data["NAME_SELLER_INDUSTRY_Connectivity"])), ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * 2.0))))))/2.0)) < (data["inst_AMT_PAYMENT"]))*1.))))) * (data["MONTHS_BALANCE"]))) 
    v["i88"] = np.tanh(((((((2.0) < (data["NAME_SELLER_INDUSTRY_Connectivity"]))*1.)) + ((((((data["RATE_DOWN_PAYMENT"]) > (data["cc_bal_SK_DPD"]))*1.)) * ((-1.0*((((0.680000) - (((0.043478) * 2.0))))))))))/2.0)) 
    v["i89"] = np.tanh((-1.0*((np.maximum(((((data["FLAG_DOCUMENT_13"]) * 2.0))), ((((((0.062500) + (data["NAME_TYPE_SUITE_Children"]))) + (((data["NAME_CLIENT_TYPE_Refreshed"]) + (-3.0))))))))))) 
    v["i90"] = np.tanh(((np.where(data["NONLIVINGAPARTMENTS_MODE"]>0, np.maximum(((((((data["NONLIVINGAPARTMENTS_MODE"]) * 2.0)) * ((((data["NONLIVINGAPARTMENTS_MODE"]) > (2.235290))*1.))))), ((data["cc_bal_SK_DPD"]))), data["NONLIVINGAPARTMENTS_MODE"] )) - (data["NONLIVINGAPARTMENTS_MODE"]))) 
    v["i91"] = np.tanh(np.maximum(((((((((data["DAYS_LAST_DUE"]) > ((((data["LIVINGAPARTMENTS_MEDI"]) < (0.323944))*1.)))*1.)) > ((((data["LIVINGAPARTMENTS_MEDI"]) < ((-1.0*((data["cc_bal_SK_DPD_DEF"])))))*1.)))*1.))), (((-1.0*((data["cc_bal_SK_DPD_DEF"]))))))) 
    v["i92"] = np.tanh(((14.800000) * ((((data["ENTRANCES_MEDI"]) < ((-1.0*((np.where(data["PRODUCT_COMBINATION_Card_X_Sell"]>0, 0.686567, ((0.686567) + (((0.697674) + (3.526320)))) ))))))*1.)))) 
    v["i93"] = np.tanh(np.minimum(((data["cc_bal_SK_DPD"])), (((((((-1.0*(((((data["FLAG_DOCUMENT_15"]) + ((((data["NAME_CASH_LOAN_PURPOSE_Everyday_expenses"]) + (((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) / 2.0)))/2.0)))/2.0))))) - (0.036585))) + (1.0)))))) 
    v["i94"] = np.tanh(np.minimum(((((data["ty__Another_type_of_loan"]) - (((((((data["ty__Another_type_of_loan"]) * 2.0)) * 2.0)) * 2.0))))), (((((np.tanh((((((data["REGION_POPULATION_RELATIVE"]) * 2.0)) * 2.0)))) + (0.494118))/2.0))))) 
    v["i95"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["FLAG_DOCUMENT_16"], ((((data["cu__currency_3"]) - (np.maximum(((data["cc_bal_cc_bal_status__Sent_proposal"])), (((((data["FONDKAPREMONT_MODE"]) > (1.0))*1.))))))) - (data["FLAG_DOCUMENT_16"])) )) 
    v["i96"] = np.tanh(np.maximum((((((np.minimum(((data["te_WEEKDAY_APPR_PROCESS_START"])), ((0.043478)))) + (np.maximum((((-1.0*((data["FLAG_CONT_MOBILE"]))))), ((((data["FLAG_DOCUMENT_2"]) / 2.0))))))/2.0))), (((((-1.0) + (data["NAME_CASH_LOAN_PURPOSE_Car_repairs"]))/2.0))))) 
    v["i97"] = np.tanh((-1.0*((np.where(((1.220590) - (np.maximum((((((data["NAME_GOODS_CATEGORY_Gardening"]) + (data["inst_NUM_INSTALMENT_VERSION"]))/2.0))), ((data["FLAG_DOCUMENT_11"])))))>0, data["FLAG_DOCUMENT_16"], (7.99752998352050781) ))))) 
    v["i98"] = np.tanh(((np.minimum(((np.maximum(((((((((-1.0*((0.576923)))) / 2.0)) > (data["AMT_CREDIT_y"]))*1.))), ((data["NAME_SELLER_INDUSTRY_Auto_technology"]))))), ((((data["AMT_CREDIT_y"]) - ((-1.0*((0.576923))))))))) * (2.0))) 
    v["i99"] = np.tanh((((((data["Active"]) > (0.576923))*1.)) * (((data["Active"]) * ((((-1.0*((((np.minimum(((data["Active"])), ((data["CNT_INSTALMENT"])))) + (0.323944)))))) * 2.0)))))) 
    v["i100"] = np.tanh(np.where(data["AMT_CREDIT_y"]>0, ((data["CNT_INSTALMENT_FUTURE"]) * (np.maximum((((((data["AMT_CREDIT_y"]) > (1.105260))*1.))), ((((0.043478) * (data["CNT_INSTALMENT_FUTURE"]))))))), (((data["NAME_YIELD_GROUP_XNA"]) > (1.105260))*1.) )) 
    v["i101"] = np.tanh(np.minimum(((0.686567)), ((np.minimum(((((data["FLAG_EMAIL"]) * (data["inst_AMT_PAYMENT"])))), ((((1.857140) - (data["NAME_CONTRACT_STATUS_Canceled"]))))))))) 
    v["i102"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, data["FLAG_PHONE"], ((data["AMT_CREDIT_y"]) * ((((((0.576923) < (np.where(data["ca__Sold"]>0, data["AMT_CREDIT_y"], data["cc_bal_SK_DPD_DEF"] )))*1.)) * 2.0))) )) 
    v["i103"] = np.tanh((((((((data["PRODUCT_COMBINATION_POS_household_without_interest"]) > ((((data["PRODUCT_COMBINATION_POS_household_without_interest"]) < (((data["FLAG_PHONE"]) / 2.0)))*1.)))*1.)) / 2.0)) + (((((data["FLAG_PHONE"]) * (data["NAME_GOODS_CATEGORY_Direct_Sales"]))) * 2.0)))) 
    v["i104"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["FLAG_DOCUMENT_6"], ((((((np.tanh((data["FLAG_DOCUMENT_6"]))) + (((((data["FLAG_DOCUMENT_2"]) - (((0.0) - (data["NAME_GOODS_CATEGORY_Insurance"]))))) * 2.0)))/2.0)) + (data["cc_bal_SK_DPD_DEF"]))/2.0) )) 
    v["i105"] = np.tanh((-1.0*((((data["SK_ID_PREV_y"]) * (((((((data["te_ORGANIZATION_TYPE"]) + (data["NAME_GOODS_CATEGORY_Direct_Sales"]))/2.0)) + ((((((data["te_ORGANIZATION_TYPE"]) + (data["FLAG_DOCUMENT_2"]))/2.0)) * ((-1.0*((data["te_FLAG_DOCUMENT_20"])))))))/2.0))))))) 
    v["i106"] = np.tanh((-1.0*(((((0.576923) > ((((((np.minimum(((np.minimum(((data["te_NAME_INCOME_TYPE"])), ((data["NAME_GOODS_CATEGORY_Direct_Sales"]))))), ((0.043478)))) - (data["SK_ID_PREV_y"]))) + ((3.83449053764343262)))/2.0)))*1.))))) 
    v["i107"] = np.tanh((-1.0*((((np.maximum(((data["NAME_SELLER_INDUSTRY_MLM_partners"])), (((((1.105260) < (data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"]))*1.))))) * 2.0))))) 
    v["i108"] = np.tanh(((np.where((((data["NAME_SELLER_INDUSTRY_Connectivity"]) + (0.494118))/2.0)>0, ((data["te_NAME_INCOME_TYPE"]) / 2.0), 0.043478 )) * (((0.323944) + ((((data["te_FLAG_DOCUMENT_20"]) + (data["te_FLAG_DOCUMENT_3"]))/2.0)))))) 
    v["i109"] = np.tanh(((np.maximum(((((((((((data["FLAG_DOCUMENT_2"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0)) * 2.0)) + (((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * 2.0)))/2.0))), ((((data["cu__currency_3"]) * 2.0))))) * (((((data["te_FLAG_DOCUMENT_20"]) * 2.0)) * 2.0)))) 
    v["i110"] = np.tanh((((0.323944) + (np.minimum(((np.minimum(((np.tanh((data["WEEKDAY_APPR_PROCESS_START_MONDAY"])))), ((((0.697674) * 2.0)))))), ((((data["CODE_REJECT_REASON_VERIF"]) * (((((data["FLAG_DOCUMENT_17"]) * (data["CODE_REJECT_REASON_VERIF"]))) / 2.0))))))))/2.0)) 
    v["i111"] = np.tanh(np.tanh((np.where((((((((((0.494118) + (data["DAYS_CREDIT_ENDDATE"]))/2.0)) < ((0.0)))*1.)) > (0.031746))*1.)>0, 11.714300, np.minimum(((0.494118)), ((data["DAYS_CREDIT_ENDDATE"]))) )))) 
    v["i112"] = np.tanh((((-1.0*((np.where(((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) / 2.0)>0, (((-1.0*((data["avg_buro_buro_count"])))) / 2.0), (((data["avg_buro_buro_count"]) > ((((2.235290) > (((data["CODE_REJECT_REASON_SCO"]) / 2.0)))*1.)))*1.) ))))) * 2.0)) 
    v["i113"] = np.tanh(((((((((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Buying_a_home"])))) + (((data["NAME_CASH_LOAN_PURPOSE_Payments_on_other_loans"]) * (((((data["NAME_CASH_LOAN_PURPOSE_Payments_on_other_loans"]) + (np.maximum(((data["AMT_REQ_CREDIT_BUREAU_YEAR"])), ((((0.043478) / 2.0))))))) * 2.0)))))/2.0)) * 2.0)) * 2.0)) 
    v["i114"] = np.tanh(((np.where(data["cc_bal_AMT_DRAWINGS_OTHER_CURRENT"]>0, np.where(data["PRODUCT_COMBINATION_Cash"]>0, (-1.0*((data["avg_buro_buro_bal_status_X"]))), data["cc_bal_AMT_DRAWINGS_OTHER_CURRENT"] ), data["PRODUCT_COMBINATION_Cash"] )) / 2.0)) 
    v["i115"] = np.tanh(((data["AMT_CREDIT_y"]) * (np.where(0.494118>0, (((data["cc_bal_SK_DPD_DEF"]) > (((((data["FLAG_DOCUMENT_17"]) + (0.797872))) + (np.tanh((data["AMT_CREDIT_x"]))))))*1.), data["AMT_CREDIT_x"] )))) 
    v["i116"] = np.tanh(np.where(data["FLAG_DOCUMENT_2"]>0, 3.0, ((data["NAME_CONTRACT_STATUS_Canceled"]) * (((((data["NAME_CONTRACT_STATUS_Canceled"]) * (data["NAME_CONTRACT_STATUS_Canceled"]))) * (np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Furniture"])), ((data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))))) )) 
    v["i117"] = np.tanh(((data["FLAG_DOCUMENT_2"]) - (((data["AMT_GOODS_PRICE_x"]) * (((1.105260) * (((((((((0.062500) < (data["PRODUCT_COMBINATION_Cash"]))*1.)) * (0.323944))) < (data["PRODUCT_COMBINATION_Cash"]))*1.)))))))) 
    v["i118"] = np.tanh(np.where(data["AMT_CREDIT_x"]>0, np.where(data["AMT_ANNUITY_x"]>0, (((data["AMT_ANNUITY_x"]) > (np.minimum(((0.494118)), ((data["AMT_CREDIT_x"])))))*1.), -2.0 ), (((data["NAME_GOODS_CATEGORY_Direct_Sales"]) > (0.043478))*1.) )) 
    v["i119"] = np.tanh(((((((data["FLAG_EMP_PHONE"]) + (0.339286))) * ((((((((8.0)) * (((data["FLAG_EMP_PHONE"]) + (data["DAYS_EMPLOYED"]))))) * 2.0)) * 2.0)))) * 2.0)) 
    v["i120"] = np.tanh((((-1.0*(((((0.680000) < ((((data["DAYS_BIRTH"]) + (0.062500))/2.0)))*1.))))) * 2.0)) 
    v["i121"] = np.tanh((((0.062500) + ((-1.0*(((((data["LIVE_CITY_NOT_WORK_CITY"]) > (np.minimum(((((data["cc_bal_SK_DPD_DEF"]) + (((data["DAYS_EMPLOYED"]) - (data["NAME_GOODS_CATEGORY_Insurance"])))))), ((0.416667)))))*1.))))))/2.0)) 
    v["i122"] = np.tanh(np.where((((data["SK_ID_BUREAU"]) > (2.0))*1.)>0, data["SK_ID_BUREAU"], (((data["DAYS_ID_PUBLISH"]) > (((((((0.0) + (3.526320))/2.0)) + (data["DAYS_ID_PUBLISH"]))/2.0)))*1.) )) 
    v["i123"] = np.tanh(((((data["te_FLAG_WORK_PHONE"]) * ((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) > (data["FLAG_DOCUMENT_2"]))*1.)))) + (((((0.339286) * (np.minimum(((((data["te_FLAG_WORK_PHONE"]) - (data["te_REG_REGION_NOT_LIVE_REGION"])))), ((data["NAME_CONTRACT_STATUS_Approved"])))))) / 2.0)))) 
    v["i124"] = np.tanh(np.where(data["REGION_RATING_CLIENT_W_CITY"]>0, (((((0.036585) / 2.0)) + ((((data["FLAG_DOCUMENT_2"]) > ((((data["cc_bal_SK_DPD_DEF"]) + (data["NAME_TYPE_SUITE_Unaccompanied"]))/2.0)))*1.)))/2.0), (((data["NAME_HOUSING_TYPE"]) > ((-1.0*((-3.0)))))*1.) )) 
    v["i125"] = np.tanh((-1.0*((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["te_OCCUPATION_TYPE"], ((data["te_OCCUPATION_TYPE"]) * (((((((data["AMT_REQ_CREDIT_BUREAU_MON"]) > (0.686567))*1.)) < (np.tanh((data["AMT_REQ_CREDIT_BUREAU_MON"]))))*1.))) ))))) 
    v["i126"] = np.tanh((-1.0*((np.minimum(((((data["EXT_SOURCE_1"]) + (((data["EXT_SOURCE_1"]) + ((((((1.0) * 2.0)) + (3.0))/2.0))))))), (((((np.tanh((data["EXT_SOURCE_1"]))) > (data["EXT_SOURCE_1"]))*1.)))))))) 
    v["i127"] = np.tanh(np.where((((data["DAYS_BIRTH"]) > (1.105260))*1.)>0, ((1.105260) - (data["DAYS_BIRTH"])), np.where(data["te_NAME_INCOME_TYPE"]>0, data["DAYS_BIRTH"], np.tanh((0.031746)) ) )) 
    v["i128"] = np.tanh(np.tanh((np.minimum(((0.576923)), ((np.where((((((((data["FLAG_DOCUMENT_3"]) - ((-1.0*((data["DAYS_ID_PUBLISH"])))))) + (0.416667))/2.0)) / 2.0)>0, 0.416667, data["DAYS_ID_PUBLISH"] ))))))) 
    v["i129"] = np.tanh((((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) + (np.tanh((data["NAME_CONTRACT_TYPE"]))))/2.0)) * (((data["CNT_FAM_MEMBERS"]) + (((data["NAME_GOODS_CATEGORY_Clothing_and_Accessories"]) * (((data["te_NAME_CONTRACT_TYPE"]) - (data["FLAG_DOCUMENT_17"]))))))))) 
    v["i130"] = np.tanh(np.minimum(((((np.minimum(((data["FLAG_DOCUMENT_2"])), ((((data["AMT_CREDIT_x"]) / 2.0))))) * (data["AMT_REQ_CREDIT_BUREAU_MON"])))), ((((((((0.697674) + (((0.339286) + (data["AMT_CREDIT_x"]))))) * 2.0)) * 2.0))))) 
    v["i131"] = np.tanh(((np.minimum(((((np.where(data["AMT_GOODS_PRICE_x"]>0, 3.526320, data["AMT_GOODS_PRICE_x"] )) * (data["FLAG_DOCUMENT_17"])))), ((((((data["AMT_GOODS_PRICE_x"]) * 2.0)) - (np.minimum(((-2.0)), ((data["FLAG_DOCUMENT_21"]))))))))) * 2.0)) 
    v["i132"] = np.tanh(((np.where(data["NAME_CLIENT_TYPE_Repeater"]>0, np.maximum(((0.062500)), ((data["FLAG_DOCUMENT_21"]))), ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (0.036585))) - (((data["FLAG_DOCUMENT_21"]) + (0.036585)))) )) * 2.0)) 
    v["i133"] = np.tanh(((((((-1.0*(((((((0.797872) / 2.0)) > ((((-1.0*((data["AMT_GOODS_PRICE_x"])))) * 2.0)))*1.))))) + ((((1.220590) < (data["FLAG_DOCUMENT_5"]))*1.)))/2.0)) + (data["cc_bal_SK_DPD_DEF"]))) 
    v["i134"] = np.tanh((((data["AMT_CREDIT_x"]) > (((0.797872) * ((((np.minimum(((0.323944)), (((((((data["AMT_CREDIT_x"]) / 2.0)) + (0.339286))/2.0))))) < (data["AMT_CREDIT_x"]))*1.)))))*1.)) 
    v["i135"] = np.tanh((-1.0*((((data["FLAG_DOCUMENT_15"]) + ((((data["AMT_GOODS_PRICE_x"]) > (((0.697674) + (np.minimum(((data["AMT_GOODS_PRICE_x"])), (((((0.12030962109565735)) - (data["AMT_GOODS_PRICE_x"])))))))))*1.))))))) 
    v["i136"] = np.tanh(((14.800000) * (((data["AMT_CREDIT_x"]) - (((data["AMT_GOODS_PRICE_x"]) - (((((((3.25225424766540527)) < (data["AMT_CREDIT_x"]))*1.)) / 2.0)))))))) 
    v["i137"] = np.tanh((((((((data["te_NAME_CONTRACT_TYPE"]) < ((((((data["NAME_YIELD_GROUP_low_normal"]) / 2.0)) < (data["NAME_CONTRACT_TYPE"]))*1.)))*1.)) / 2.0)) * (((data["NAME_CONTRACT_TYPE"]) + (np.tanh((data["FLAG_WORK_PHONE"]))))))) 
    v["i138"] = np.tanh((((((((0.797872) * (data["AMT_CREDIT_x"]))) + (((data["Completed"]) / 2.0)))) > ((((((((-1.0*((data["AMT_CREDIT_x"])))) > (((data["Completed"]) / 2.0)))*1.)) < (data["te_FLAG_DOCUMENT_8"]))*1.)))*1.)) 
    v["i139"] = np.tanh(((((((np.maximum((((((data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"]) + (1.857140))/2.0))), ((data["inst_AMT_PAYMENT"])))) < (0.576923))*1.)) > ((((data["cc_bal_SK_DPD_DEF"]) > (((data["FLAG_DOCUMENT_2"]) * (1.857140))))*1.)))*1.)) 
    v["i140"] = np.tanh((((((((((((data["EXT_SOURCE_2"]) < ((-1.0*((2.235290)))))*1.)) * 2.0)) * 2.0)) + (np.tanh((data["EXT_SOURCE_2"]))))) / 2.0)) 
    v["i141"] = np.tanh((-1.0*(((((data["AMT_GOODS_PRICE_x"]) > ((((((((((0.576923) / 2.0)) < (data["AMT_GOODS_PRICE_x"]))*1.)) * 2.0)) - (np.maximum(((((0.416667) / 2.0))), ((data["AMT_GOODS_PRICE_x"])))))))*1.))))) 
    v["i142"] = np.tanh(((np.tanh(((((((((((data["AMT_GOODS_PRICE_x"]) > (0.797872))*1.)) < (np.tanh((data["AMT_GOODS_PRICE_x"]))))*1.)) < ((((data["DAYS_LAST_DUE_1ST_VERSION"]) < (np.tanh((data["AMT_GOODS_PRICE_x"]))))*1.)))*1.)))) / 2.0)) 
    v["i143"] = np.tanh(np.where(data["NAME_CONTRACT_STATUS_Refused"]>0, (-1.0*(((((data["CODE_REJECT_REASON_HC"]) > (3.0))*1.)))), ((0.339286) * ((((0.576923) < ((((data["ty__Consumer_credit"]) > (0.797872))*1.)))*1.))) )) 
    v["i144"] = np.tanh(((np.minimum(((np.where(data["NAME_GOODS_CATEGORY_Computers"]>0, ((data["AMT_CREDIT_y"]) / 2.0), (((((data["AMT_CREDIT_y"]) < (data["NAME_GOODS_CATEGORY_Insurance"]))*1.)) / 2.0) ))), ((np.where(data["NAME_GOODS_CATEGORY_Computers"]>0, data["AMT_CREDIT_y"], data["YEARS_BUILD_AVG"] ))))) / 2.0)) 
    v["i145"] = np.tanh(np.where(data["NAME_CASH_LOAN_PURPOSE_Furniture"]>0, (((data["FLAG_DOCUMENT_14"]) + (data["FLAG_DOCUMENT_18"]))/2.0), np.minimum((((-1.0*(((((data["FLAG_DOCUMENT_18"]) + (((data["FLAG_DOCUMENT_14"]) + (data["FLAG_DOCUMENT_17"]))))/2.0)))))), (((-1.0*((data["FLAG_DOCUMENT_18"])))))) )) 
    v["i146"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["FLAG_DOCUMENT_18"], ((data["cc_bal_SK_DPD_DEF"]) - (((data["FLAG_DOCUMENT_18"]) * (((data["te_FLAG_DOCUMENT_14"]) + (np.minimum(((((data["te_FLAG_DOCUMENT_14"]) * 2.0))), ((data["FLAG_DOCUMENT_18"]))))))))) )) 
    v["i147"] = np.tanh((((((np.tanh((((data["ca__Bad_debt"]) + (np.tanh((1.857140))))))) < (data["AMT_DOWN_PAYMENT"]))*1.)) + (np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), (((((data["te_FLAG_DOCUMENT_18"]) + ((-1.0*((data["AMT_DOWN_PAYMENT"])))))/2.0))))))) 
    v["i148"] = np.tanh(np.where((((0.680000) < (data["AMT_DOWN_PAYMENT"]))*1.)>0, ((data["AMT_DOWN_PAYMENT"]) / 2.0), ((data["NAME_GOODS_CATEGORY_Insurance"]) - (((np.maximum(((np.tanh((data["SK_DPD"])))), ((data["AMT_DOWN_PAYMENT"])))) / 2.0))) )) 
    v["i149"] = np.tanh(((((data["MONTHS_BALANCE"]) * (((3.526320) * 2.0)))) * (((data["NAME_GOODS_CATEGORY_Insurance"]) + ((-1.0*(((((((data["inst_AMT_PAYMENT"]) * 2.0)) > ((((data["ca__Bad_debt"]) + (11.714300))/2.0)))*1.))))))))) 
    v["i150"] = np.tanh((((((-1.0) + (((1.857140) / 2.0)))) < (np.where(data["SK_DPD"]>0, -1.0, np.tanh((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, 1.857140, data["SK_DPD"] ))) )))*1.)) 
    v["i151"] = np.tanh((-1.0*(((((((data["inst_SK_ID_PREV"]) > ((((np.tanh((0.697674))) + (np.minimum(((((0.697674) + (((((data["inst_SK_ID_PREV"]) * 2.0)) * 2.0))))), ((data["HOUSETYPE_MODE"])))))/2.0)))*1.)) / 2.0))))) 
    v["i152"] = np.tanh(((((((0.031746) * (((np.maximum((((-1.0*((data["WALLSMATERIAL_MODE"]))))), ((0.686567)))) - (data["WALLSMATERIAL_MODE"]))))) * (np.maximum((((-1.0*((data["EXT_SOURCE_2"]))))), ((data["SK_DPD"])))))) * 2.0)) 
    v["i153"] = np.tanh(((((((data["CNT_INSTALMENT_FUTURE"]) > (((np.maximum(((data["CNT_INSTALMENT_FUTURE"])), ((data["NAME_GOODS_CATEGORY_Fitness"])))) * 2.0)))*1.)) + ((((((data["CNT_INSTALMENT_FUTURE"]) > (3.0))*1.)) - (((data["FLAG_DOCUMENT_17"]) * 2.0)))))/2.0)) 
    v["i154"] = np.tanh(((np.where(data["NAME_TYPE_SUITE_Unaccompanied"]>0, ((data["FLAG_DOCUMENT_21"]) + (data["FLAG_DOCUMENT_21"])), ((0.043478) + ((((data["FLAG_DOCUMENT_21"]) > (np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["SK_DPD"], 0.576923 )))*1.))) )) * 2.0)) 
    v["i155"] = np.tanh(np.maximum(((data["cc_bal_SK_DPD_DEF"])), ((((data["SK_DPD"]) - (((((((data["SK_DPD"]) * 2.0)) * (data["cc_bal_cc_bal_status__Sent_proposal"]))) - (((1.0) - (((11.714300) * 2.0))))))))))) 
    v["i156"] = np.tanh(np.where(data["NAME_FAMILY_STATUS"]>0, data["DAYS_EMPLOYED"], np.where(data["NONLIVINGAPARTMENTS_MODE"]>0, data["DAYS_EMPLOYED"], np.minimum(((0.697674)), ((np.where(data["DAYS_EMPLOYED"]>0, data["AMT_CREDIT_x"], 0.031746 )))) ) )) 
    v["i157"] = np.tanh(np.where(data["te_NAME_INCOME_TYPE"]>0, np.where(data["NONLIVINGAPARTMENTS_MODE"]>0, data["te_FLAG_WORK_PHONE"], np.where(data["NAME_GOODS_CATEGORY_Fitness"]>0, data["te_NAME_INCOME_TYPE"], 0.043478 ) ), np.where(data["NAME_GOODS_CATEGORY_Fitness"]>0, data["te_NAME_INCOME_TYPE"], data["NONLIVINGAPARTMENTS_MODE"] ) )) 
    v["i158"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, ((data["cc_bal_cc_bal_status__Sent_proposal"]) - (0.323944)), np.minimum(((((data["NONLIVINGAREA_MODE"]) * ((-1.0*((0.323944))))))), (((-1.0*((data["cc_bal_cc_bal_status__Sent_proposal"])))))) )) 
    v["i159"] = np.tanh((((((((np.where((-1.0*((data["cc_bal_SK_DPD_DEF"])))>0, data["ORGANIZATION_TYPE"], ((data["NAME_GOODS_CATEGORY_Fitness"]) / 2.0) )) - (-3.0))) < (data["ORGANIZATION_TYPE"]))*1.)) * 2.0)) 
    v["i160"] = np.tanh((((((-1.0*(((-1.0*((((data["NAME_GOODS_CATEGORY_Fitness"]) * (((data["FLAG_DOCUMENT_13"]) - (((data["HOUSETYPE_MODE"]) + (data["NONLIVINGAPARTMENTS_MODE"]))))))))))))) * 2.0)) * 2.0)) 
    v["i161"] = np.tanh(np.where(data["WALLSMATERIAL_MODE"]>0, (((((-1.0*((0.062500)))) * (((data["WALLSMATERIAL_MODE"]) + (data["NAME_GOODS_CATEGORY_Fitness"]))))) * 2.0), 0.062500 )) 
    v["i162"] = np.tanh(((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (((((3.526320) + (((data["te_WALLSMATERIAL_MODE"]) + (((((data["FLAG_DOCUMENT_2"]) + (((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) * 2.0)))) * 2.0)))))) * (data["te_WALLSMATERIAL_MODE"]))))) 
    v["i163"] = np.tanh(((((np.maximum(((data["ca__Bad_debt"])), (((((((11.714300) < ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + (data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]))/2.0)))*1.)) * 2.0))))) - (np.minimum(((1.105260)), ((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"])))))) * 2.0)) 
    v["i164"] = np.tanh(((((data["FLAG_WORK_PHONE"]) * (((0.043478) - (np.minimum(((((((data["NAME_GOODS_CATEGORY_Fitness"]) * (data["NAME_CASH_LOAN_PURPOSE_Furniture"]))) * 2.0))), (((-1.0*((data["NAME_SELLER_INDUSTRY_MLM_partners"]))))))))))) * 2.0)) 
    v["i165"] = np.tanh(((((data["PRODUCT_COMBINATION_Card_X_Sell"]) * ((((data["PRODUCT_COMBINATION_Card_X_Sell"]) > (np.where(np.where(data["cc_bal_AMT_BALANCE"]>0, 2.0, data["cc_bal_AMT_BALANCE"] )>0, 2.0, 11.714300 )))*1.)))) - (((data["FLAG_DOCUMENT_15"]) / 2.0)))) 
    v["i166"] = np.tanh(((((np.minimum(((data["NAME_SELLER_INDUSTRY_Auto_technology"])), ((((0.797872) - (data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"])))))) - ((((((0.797872) * (((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) / 2.0)))) < (data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"]))*1.)))) / 2.0)) 
    v["i167"] = np.tanh((((3.0) < (((0.323944) + (np.maximum(((np.maximum(((data["cc_bal_AMT_BALANCE"])), ((data["AMT_CREDIT_y"]))))), ((((np.where(data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]>0, data["cc_bal_AMT_BALANCE"], 3.0 )) + (data["cc_bal_CNT_DRAWINGS_POS_CURRENT"])))))))))*1.)) 
    v["i168"] = np.tanh(((((np.minimum((((((1.78492116928100586)) - (data["AMT_CREDIT_SUM"])))), (((((0.797872) > ((((((1.220590) > (data["AMT_CREDIT_SUM"]))*1.)) + (((data["AMT_CREDIT_SUM"]) / 2.0)))))*1.))))) * 2.0)) * 2.0)) 
    v["i169"] = np.tanh(np.where(np.where((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) > (data["NAME_CASH_LOAN_PURPOSE_Furniture"]))*1.)>0, 3.526320, (((data["PRODUCT_COMBINATION_Cash_X_Sell__middle"]) > (3.0))*1.) )>0, (((data["PRODUCT_COMBINATION_Cash_X_Sell__middle"]) > (3.0))*1.), data["SK_ID_BUREAU"] )) 
    v["i170"] = np.tanh(((np.minimum(((data["NFLAG_INSURED_ON_APPROVAL"])), ((0.680000)))) * ((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) < (np.tanh((np.tanh(((((-1.0*((np.maximum(((0.576923)), ((((data["NFLAG_INSURED_ON_APPROVAL"]) / 2.0)))))))) / 2.0)))))))*1.)))) 
    v["i171"] = np.tanh((((((0.031746) + (((((data["NAME_PORTFOLIO_Cash"]) / 2.0)) / 2.0)))/2.0)) - (((((data["inst_SK_ID_PREV"]) / 2.0)) * (((((data["inst_SK_ID_PREV"]) / 2.0)) * (data["NAME_PORTFOLIO_Cash"]))))))) 
    v["i172"] = np.tanh((((((1.105260) > (data["PRODUCT_COMBINATION_Cash"]))*1.)) * (np.minimum(((np.minimum(((0.062500)), ((((data["PRODUCT_COMBINATION_Cash"]) * (data["inst_AMT_PAYMENT"]))))))), ((((data["NFLAG_INSURED_ON_APPROVAL"]) * (data["inst_AMT_PAYMENT"])))))))) 
    v["i173"] = np.tanh(((((((3.526320) * (((((np.maximum((((((data["AMT_CREDIT_MAX_OVERDUE"]) > (np.tanh((data["AMT_CREDIT_MAX_OVERDUE"]))))*1.))), ((data["NAME_CASH_LOAN_PURPOSE_Medicine"])))) + (data["AMT_CREDIT_MAX_OVERDUE"]))) * 2.0)))) * 2.0)) * 2.0)) 
    v["i174"] = np.tanh((((data["avg_buro_buro_bal_status_2"]) < (((np.where(data["CNT_INSTALMENT_FUTURE"]>0, np.minimum(((data["DAYS_CREDIT"])), ((np.minimum(((data["DAYS_CREDIT"])), ((((1.220590) - (data["DAYS_CREDIT"])))))))), data["CNT_INSTALMENT_FUTURE"] )) / 2.0)))*1.)) 
    v["i175"] = np.tanh(((np.tanh((data["EXT_SOURCE_3"]))) * (((0.416667) - ((((((((data["EXT_SOURCE_3"]) < (np.minimum(((np.tanh((data["FLAG_DOCUMENT_2"])))), ((-2.0)))))*1.)) * 2.0)) * 2.0)))))) 
    v["i176"] = np.tanh(np.where(data["SK_DPD"]>0, data["CNT_INSTALMENT"], (((((((np.maximum(((data["NAME_SELLER_INDUSTRY_Connectivity"])), ((data["NAME_SELLER_INDUSTRY_Connectivity"])))) * 2.0)) * (data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))) + (((data["CNT_INSTALMENT"]) * (data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))))/2.0) )) 
    v["i177"] = np.tanh(((data["CNT_INSTALMENT"]) * ((((-1.0*((np.maximum(((((((((((0.339286) < (data["FLAG_DOCUMENT_2"]))*1.)) + (data["DAYS_CREDIT"]))/2.0)) * (data["DAYS_CREDIT"])))), (((-1.0*((0.0)))))))))) / 2.0)))) 
    v["i178"] = np.tanh(np.maximum((((((((((data["cc_bal_AMT_DRAWINGS_OTHER_CURRENT"]) > (((3.0) / 2.0)))*1.)) * 2.0)) * 2.0))), (((((3.0) < (((data["CNT_INSTALMENT_FUTURE"]) + ((((data["cc_bal_SK_DPD_DEF"]) > (data["NAME_GOODS_CATEGORY_Weapon"]))*1.)))))*1.))))) 
    v["i179"] = np.tanh(((np.where(data["SK_DPD"]>0, data["CNT_INSTALMENT"], data["AMT_REQ_CREDIT_BUREAU_DAY"] )) + ((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) < ((((data["SK_DPD"]) + (((data["AMT_REQ_CREDIT_BUREAU_DAY"]) + (np.maximum(((data["AMT_REQ_CREDIT_BUREAU_DAY"])), ((data["AMT_REQ_CREDIT_BUREAU_DAY"])))))))/2.0)))*1.)))) 
    v["i180"] = np.tanh(((np.minimum((((((data["NAME_SELLER_INDUSTRY_Connectivity"]) > (((np.tanh((data["NAME_SELLER_INDUSTRY_Connectivity"]))) * (1.857140))))*1.))), ((np.where(data["DAYS_CREDIT"]>0, 0.0, np.tanh((data["NAME_SELLER_INDUSTRY_Connectivity"])) ))))) / 2.0)) 
    v["i181"] = np.tanh((((-1.0*((np.tanh((np.where(np.tanh((data["cc_bal_SK_DPD_DEF"]))>0, data["NAME_YIELD_GROUP_high"], np.where(data["NAME_YIELD_GROUP_high"]>0, np.where(data["SK_DPD"]>0, data["NAME_YIELD_GROUP_high"], data["SK_DPD"] ), 0.036585 ) ))))))) * 2.0)) 
    v["i182"] = np.tanh(np.minimum(((((((2.0) + (data["PRODUCT_COMBINATION_POS_mobile_with_interest"]))) - (((data["NAME_YIELD_GROUP_high"]) * 2.0))))), ((((np.minimum(((0.323944)), ((0.680000)))) / 2.0))))) 
    v["i183"] = np.tanh(((((((((data["PRODUCT_COMBINATION_Cash_Street__high"]) > (np.tanh((0.416667))))*1.)) + ((((data["SK_DPD"]) + ((((((-1.0*((data["PRODUCT_COMBINATION_Cash_Street__high"])))) * (data["SK_DPD"]))) / 2.0)))/2.0)))/2.0)) / 2.0)) 
    v["i184"] = np.tanh((((2.0) < (((data["PRODUCT_COMBINATION_POS_household_with_interest"]) + (((data["LIVINGAPARTMENTS_MEDI"]) - (((((0.0) * 2.0)) - (((np.where(data["cc_bal_cc_bal_status__Refused"]>0, data["cc_bal_cc_bal_status__Refused"], data["LIVINGAPARTMENTS_MEDI"] )) * 2.0)))))))))*1.)) 
    v["i185"] = np.tanh(((((data["ENTRANCES_MEDI"]) * (((data["NAME_GOODS_CATEGORY_Insurance"]) * 2.0)))) - (np.tanh(((((np.tanh(((((data["ENTRANCES_MEDI"]) + (0.576923))/2.0)))) + (np.tanh((data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))/2.0)))))) 
    v["i186"] = np.tanh(((np.where(data["CODE_REJECT_REASON_VERIF"]>0, data["DAYS_FIRST_DUE"], np.maximum(((((0.062500) * (data["DAYS_FIRST_DUE"])))), (((-1.0*((data["NAME_GOODS_CATEGORY_Weapon"])))))) )) - (np.maximum(((data["FLAG_DOCUMENT_17"])), ((data["NAME_GOODS_CATEGORY_Weapon"])))))) 
    v["i187"] = np.tanh(((((data["NAME_GOODS_CATEGORY_Other"]) - ((((((data["NAME_GOODS_CATEGORY_Other"]) / 2.0)) + ((10.0)))/2.0)))) * (np.maximum(((data["NAME_GOODS_CATEGORY_Other"])), ((0.0)))))) 
    v["i188"] = np.tanh(np.minimum(((data["ty__Another_type_of_loan"])), ((((data["cc_bal_SK_DPD"]) - (np.where(data["FLAG_DOCUMENT_14"]>0, data["FLAG_DOCUMENT_14"], (((((data["FLAG_DOCUMENT_14"]) + (np.maximum(((data["NAME_GOODS_CATEGORY_Fitness"])), ((data["ty__Another_type_of_loan"])))))/2.0)) / 2.0) ))))))) 
    v["i189"] = np.tanh((((-1.0*((((0.697674) - ((((((0.680000) - ((((data["NAME_PORTFOLIO_POS"]) < (0.680000))*1.)))) < (np.tanh(((((0.680000) + (data["NAME_PORTFOLIO_POS"]))/2.0)))))*1.))))))) / 2.0)) 
    v["i190"] = np.tanh(np.minimum(((((np.where(data["NAME_GOODS_CATEGORY_XNA"]>0, ((data["NAME_GOODS_CATEGORY_XNA"]) / 2.0), ((data["PRODUCT_COMBINATION_Cash_Street__high"]) / 2.0) )) * (np.minimum((((1.0))), ((data["PRODUCT_COMBINATION_Cash_Street__high"]))))))), ((((((data["NAME_GOODS_CATEGORY_XNA"]) / 2.0)) / 2.0))))) 
    v["i191"] = np.tanh(np.tanh((np.where(data["cc_bal_cc_bal_status__Refused"]>0, ((np.tanh((((data["NFLAG_LAST_APPL_IN_DAY"]) + (((((((data["NFLAG_LAST_APPL_IN_DAY"]) - (0.062500))) * 2.0)) - (0.062500))))))) / 2.0), data["NFLAG_LAST_APPL_IN_DAY"] )))) 
    v["i192"] = np.tanh(((np.maximum(((data["cc_bal_SK_DPD"])), (((((2.0) < (((data["AMT_CREDIT_x"]) * (np.where(data["cc_bal_SK_DPD"]>0, 0.686567, 1.220590 )))))*1.))))) * (((data["AMT_CREDIT_x"]) - (0.043478))))) 
    v["i193"] = np.tanh(((np.minimum(((data["cc_bal_SK_DPD_DEF"])), ((data["cc_bal_SK_DPD"])))) - (np.where((10.0)>0, (((0.323944) < (((data["YEARS_BEGINEXPLUATATION_MEDI"]) + ((((data["cc_bal_SK_DPD_DEF"]) + (data["YEARS_BEGINEXPLUATATION_MEDI"]))/2.0)))))*1.), 0.043478 )))) 
    v["i194"] = np.tanh((((data["LIVINGAREA_MODE"]) > (np.maximum(((data["TOTALAREA_MODE"])), ((np.where(((data["FLOORSMAX_AVG"]) * (np.maximum(((data["LIVINGAREA_AVG"])), ((data["TOTALAREA_MODE"])))))>0, ((data["LIVINGAREA_AVG"]) / 2.0), data["LIVINGAREA_MODE"] ))))))*1.)) 
    v["i195"] = np.tanh((((-1.0*((np.where(data["WALLSMATERIAL_MODE"]>0, np.where(((0.494118) * (data["FLAG_CONT_MOBILE"]))>0, 0.686567, data["FLAG_CONT_MOBILE"] ), (((((0.036585) / 2.0)) > (0.686567))*1.) ))))) / 2.0)) 
    v["i196"] = np.tanh(((((((data["NAME_CASH_LOAN_PURPOSE_Everyday_expenses"]) * (data["NAME_CASH_LOAN_PURPOSE_Buying_a_home"]))) + ((((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Buying_a_home"])))) + (((data["NAME_CASH_LOAN_PURPOSE_Everyday_expenses"]) * (data["cc_bal_SK_DPD_DEF"]))))))) * 2.0)) 
    v["i197"] = np.tanh(np.where(np.where(data["cc_bal_cc_bal_status__Refused"]>0, (((data["cu__currency_3"]) < (0.323944))*1.), np.where(data["FLAG_DOCUMENT_2"]>0, data["FLAG_EMAIL"], 1.220590 ) )>0, ((data["cc_bal_cc_bal_status__Refused"]) + (data["cu__currency_3"])), data["FLAG_DOCUMENT_2"] )) 
    v["i198"] = np.tanh(((((np.maximum(((data["FLAG_DOCUMENT_2"])), (((((data["NAME_GOODS_CATEGORY_Medicine"]) > (data["cc_bal_SK_DPD_DEF"]))*1.))))) + (((((data["NAME_YIELD_GROUP_XNA"]) * 2.0)) * (data["cc_bal_SK_DPD_DEF"]))))) * 2.0)) 
    v["i199"] = np.tanh(np.where((((0.680000) + (data["DAYS_FIRST_DRAWING"]))/2.0)>0, np.where(data["DAYS_FIRST_DRAWING"]>0, (-1.0*((0.062500))), (((10.75054359436035156)) / 2.0) ), 0.043478 )) 
    v["i200"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, data["FLAG_EMAIL"], np.where(data["cc_bal_SK_DPD_DEF"]>0, data["FLAG_EMAIL"], np.tanh(((((0.043478) + (((((0.494118) * ((-1.0*((data["FLAG_EMAIL"])))))) / 2.0)))/2.0))) ) )) 
    v["i201"] = np.tanh(((((data["te_FLAG_EMAIL"]) * (np.where(((((14.800000) / 2.0)) - (((data["NAME_YIELD_GROUP_XNA"]) + (((data["NAME_TYPE_SUITE_Group_of_people"]) - (data["te_FLAG_EMAIL"]))))))>0, data["NAME_TYPE_SUITE_Group_of_people"], data["NAME_YIELD_GROUP_XNA"] )))) * 2.0)) 
    v["i202"] = np.tanh(((data["NAME_TYPE_SUITE_Group_of_people"]) * ((((((((np.maximum(((((0.031746) / 2.0))), ((data["FLAG_DOCUMENT_10"])))) - (1.105260))) - (data["NAME_TYPE_SUITE_Group_of_people"]))) + (data["NAME_CASH_LOAN_PURPOSE_Furniture"]))/2.0)))) 
    v["i203"] = np.tanh(((np.where(data["NAME_YIELD_GROUP_XNA"]>0, data["PRODUCT_COMBINATION_POS_household_without_interest"], 0.043478 )) * ((((0.062500) < ((((((1.220590) > (data["NAME_YIELD_GROUP_XNA"]))*1.)) - (data["cc_bal_SK_DPD_DEF"]))))*1.)))) 
    v["i204"] = np.tanh(((np.minimum((((((data["PRODUCT_COMBINATION_Cash"]) > (((np.tanh((0.416667))) - (((np.tanh((data["NAME_YIELD_GROUP_XNA"]))) + (data["NAME_PORTFOLIO_POS"]))))))*1.))), (((((0.680000) < (data["NAME_YIELD_GROUP_XNA"]))*1.))))) / 2.0)) 
    v["i205"] = np.tanh((-1.0*((np.maximum(((((-2.0) * (0.043478)))), ((((data["NAME_CONTRACT_STATUS_Approved"]) * (((data["AMT_REQ_CREDIT_BUREAU_WEEK"]) + (((data["CNT_INSTALMENT"]) * (data["AMT_REQ_CREDIT_BUREAU_WEEK"]))))))))))))) 
    v["i206"] = np.tanh(np.where(((data["cc_bal_SK_DPD_DEF"]) * ((-1.0*((data["CODE_REJECT_REASON_LIMIT"])))))>0, data["NAME_GOODS_CATEGORY_XNA"], ((0.043478) * (np.where(data["CODE_REJECT_REASON_LIMIT"]>0, 3.0, 0.043478 ))) )) 
    v["i207"] = np.tanh(np.minimum(((data["cc_bal_SK_DPD"])), ((((data["cc_bal_SK_DPD"]) * (np.where(data["FLAG_EMAIL"]>0, -3.0, (((((11.714300) < (data["cc_bal_SK_DPD"]))*1.)) + (data["FLAG_EMAIL"])) ))))))) 
    v["i208"] = np.tanh((((((0.797872) < (data["inst_AMT_INSTALMENT"]))*1.)) * ((((((0.797872) + ((((((2.0) / 2.0)) > (data["inst_AMT_INSTALMENT"]))*1.)))) > (np.tanh((data["AMT_CREDIT_y"]))))*1.)))) 
    v["i209"] = np.tanh((((-1.0*(((((np.where(data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]>0, 11.714300, ((np.where(0.323944>0, 0.323944, data["inst_AMT_PAYMENT"] )) * 2.0) )) < (((np.tanh((data["inst_AMT_PAYMENT"]))) * 2.0)))*1.))))) / 2.0)) 
    v["i210"] = np.tanh((((data["inst_AMT_PAYMENT"]) < ((-1.0*((np.maximum(((np.tanh((0.680000)))), ((np.where(data["NAME_PORTFOLIO_Cards"]>0, 0.680000, (((0.043478) > ((-1.0*((data["NAME_PORTFOLIO_Cards"])))))*1.) )))))))))*1.)) 
    v["i211"] = np.tanh((((((data["NAME_CONTRACT_TYPE_Consumer_loans"]) * (data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))) + ((((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) + (np.minimum(((((data["AMT_CREDIT_y"]) / 2.0))), ((data["AMT_CREDIT_y"])))))/2.0)) * ((((data["NAME_CONTRACT_TYPE_Consumer_loans"]) > (0.339286))*1.)))))/2.0)) 
    v["i212"] = np.tanh(np.where(((data["NAME_PORTFOLIO_Cards"]) + (((0.036585) * 2.0)))>0, ((((((data["NAME_GOODS_CATEGORY_Insurance"]) < (data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]))*1.)) < ((((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]) > (data["cc_bal_AMT_PAYMENT_CURRENT"]))*1.)))*1.), data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"] )) 
    v["i213"] = np.tanh(((data["cc_bal_CNT_DRAWINGS_CURRENT"]) * (((((data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"]) * (data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"]))) - (np.minimum(((data["cc_bal_AMT_PAYMENT_CURRENT"])), ((((11.714300) * ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) > (data["cc_bal_CNT_DRAWINGS_CURRENT"]))*1.))))))))))) 
    v["i214"] = np.tanh(np.where(np.tanh((data["CODE_REJECT_REASON_CLIENT"]))>0, data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"], ((np.where(data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]>0, ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) / 2.0)) / 2.0), ((np.minimum(((data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"])), ((data["NAME_PAYMENT_TYPE_Cash_through_the_bank"])))) / 2.0) )) / 2.0) )) 
    v["i215"] = np.tanh(((((((0.062500) - (((data["FLAG_EMAIL"]) * ((-1.0*((((data["FLAG_DOCUMENT_4"]) + (data["NAME_GOODS_CATEGORY_Direct_Sales"])))))))))) - (data["NAME_GOODS_CATEGORY_Weapon"]))) - (((data["FLAG_EMAIL"]) * (0.062500))))) 
    v["i216"] = np.tanh((-1.0*((((((data["cc_bal_cc_bal_status__Sent_proposal"]) + (data["FLAG_CONT_MOBILE"]))) * (((0.323944) + (((-1.0) * (data["NAME_YIELD_GROUP_XNA"])))))))))) 
    v["i217"] = np.tanh((((((1.857140) < (data["avg_buro_buro_bal_status_2"]))*1.)) - (np.maximum((((((0.031746) < (((np.maximum(((data["cc_bal_SK_DPD_DEF"])), ((data["avg_buro_buro_bal_status_2"])))) * 2.0)))*1.))), ((data["cc_bal_cc_bal_status__Sent_proposal"])))))) 
    v["i218"] = np.tanh(((data["cc_bal_SK_ID_PREV"]) * ((((((0.031746) < (data["FLAG_OWN_CAR"]))*1.)) + ((((((data["cc_bal_SK_ID_PREV"]) > (1.857140))*1.)) - (0.062500))))))) 
    v["i219"] = np.tanh((((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]) > ((((3.0) + ((-1.0*((np.where(((data["NAME_CASH_LOAN_PURPOSE_Car_repairs"]) + (data["DAYS_LAST_DUE"]))>0, ((data["NAME_CASH_LOAN_PURPOSE_Car_repairs"]) + (data["DAYS_LAST_DUE"])), 3.0 ))))))/2.0)))*1.)) 
    v["i220"] = np.tanh((-1.0*(((((0.680000) < (((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) * (((np.maximum(((0.043478)), ((((((0.680000) * 2.0)) * (data["DAYS_FIRST_DUE"])))))) + (((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) * 2.0)))))))*1.))))) 
    v["i221"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, np.minimum(((data["DAYS_LAST_DUE"])), ((data["PRODUCT_COMBINATION_Card_Street"]))), (((((((((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) < (((data["DAYS_LAST_DUE"]) + (data["DAYS_LAST_DUE"]))))*1.)) + (0.062500))/2.0)) < (data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]))*1.) )) 
    v["i222"] = np.tanh((((data["inst_AMT_PAYMENT"]) < ((-1.0*(((((((-1.0*((data["inst_AMT_PAYMENT"])))) / 2.0)) + (0.323944)))))))*1.)) 
    v["i223"] = np.tanh(((((((data["cc_bal_SK_DPD_DEF"]) + (np.where(data["cc_bal_SK_DPD"]>0, 1.105260, (((data["cc_bal_SK_DPD"]) < (data["NAME_SELLER_INDUSTRY_Auto_technology"]))*1.) )))/2.0)) < ((((data["NAME_SELLER_INDUSTRY_Auto_technology"]) > (14.800000))*1.)))*1.)) 
    v["i224"] = np.tanh(np.where(data["cc_bal_SK_DPD_DEF"]>0, (-1.0*((data["DAYS_FIRST_DUE"]))), ((((data["cc_bal_SK_DPD_DEF"]) - (((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (data["DAYS_FIRST_DUE"]))))) * 2.0) )) 
    v["i225"] = np.tanh((((((data["cc_bal_SK_DPD"]) > (((14.800000) - (np.maximum(((((np.where(data["cc_bal_SK_DPD"]>0, (((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))/2.0), data["FLAG_DOCUMENT_2"] )) * 2.0))), ((data["NAME_GOODS_CATEGORY_Direct_Sales"])))))))*1.)) * 2.0)) 
    v["i226"] = np.tanh((((((3.0) < ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + (data["CHANNEL_TYPE_Car_dealer"]))/2.0)))*1.)) - (np.where(data["CHANNEL_TYPE_Car_dealer"]>0, data["CHANNEL_TYPE_Car_dealer"], ((data["FLAG_DOCUMENT_17"]) * ((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Purchase_of_electronic_equipment"]))))) )))) 
    v["i227"] = np.tanh(((0.036585) + (((0.062500) + (np.where(data["AMT_CREDIT_MAX_OVERDUE"]>0, ((data["AMT_CREDIT_MAX_OVERDUE"]) * (14.800000)), ((0.036585) + (((data["AMT_CREDIT_MAX_OVERDUE"]) * (14.800000)))) )))))) 
    v["i228"] = np.tanh(np.minimum((((-1.0*((((np.tanh((data["AMT_DOWN_PAYMENT"]))) / 2.0)))))), (((((data["REGION_RATING_CLIENT_W_CITY"]) + ((((data["AMT_DOWN_PAYMENT"]) < (data["DAYS_CREDIT"]))*1.)))/2.0))))) 
    v["i229"] = np.tanh((((((-2.0) > (data["te_REGION_RATING_CLIENT"]))*1.)) * (((((((((data["te_WALLSMATERIAL_MODE"]) * (data["AMT_CREDIT_y"]))) + (((data["AMT_GOODS_PRICE_y"]) * (data["te_REGION_RATING_CLIENT"]))))) * 2.0)) * 2.0)))) 
    v["i230"] = np.tanh(np.minimum(((((data["inst_AMT_INSTALMENT"]) + ((((((data["AMT_APPLICATION"]) > (data["te_WALLSMATERIAL_MODE"]))*1.)) * 2.0))))), ((((((((data["inst_AMT_INSTALMENT"]) < (0.797872))*1.)) < ((-1.0*((data["AMT_DOWN_PAYMENT"])))))*1.))))) 
    v["i231"] = np.tanh(np.where(data["FLAG_OWN_CAR"]>0, (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + (((((((data["CNT_INSTALMENT"]) < ((-1.0*((data["NAME_GOODS_CATEGORY_Insurance"])))))*1.)) + ((-1.0*((1.0)))))/2.0)))/2.0), ((np.tanh((0.062500))) * 2.0) )) 
    v["i232"] = np.tanh(((((0.031746) + (((data["CODE_GENDER"]) * ((-1.0*(((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) < (data["NAME_GOODS_CATEGORY_Insurance"]))*1.))))))))) + (((np.tanh((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]))) * ((-1.0*((0.797872)))))))) 
    v["i233"] = np.tanh((((data["cc_bal_AMT_RECIVABLE"]) > (np.maximum(((data["cc_bal_AMT_RECEIVABLE_PRINCIPAL"])), ((np.tanh((data["cc_bal_AMT_RECEIVABLE_PRINCIPAL"])))))))*1.)) 
    v["i234"] = np.tanh((-1.0*((((np.where((-1.0*((data["DAYS_LAST_DUE_1ST_VERSION"])))>0, data["inst_AMT_PAYMENT"], np.tanh((np.tanh((data["inst_SK_ID_PREV"])))) )) / 2.0))))) 
    v["i235"] = np.tanh(((((data["inst_AMT_INSTALMENT"]) * (((((((((data["AMT_GOODS_PRICE_y"]) * (((1.105260) / 2.0)))) > (1.0))*1.)) < (((data["AMT_GOODS_PRICE_y"]) - (1.105260))))*1.)))) * 2.0)) 
    v["i236"] = np.tanh(((data["AMT_DOWN_PAYMENT"]) * (((((np.tanh(((((data["inst_AMT_PAYMENT"]) > (((data["te_OCCUPATION_TYPE"]) * (0.036585))))*1.)))) - (np.where(data["inst_AMT_PAYMENT"]>0, data["te_OCCUPATION_TYPE"], data["AMT_DOWN_PAYMENT"] )))) / 2.0)))) 
    v["i237"] = np.tanh(((((((-1.0*((data["inst_AMT_PAYMENT"])))) * (np.where(0.036585>0, data["AMT_DOWN_PAYMENT"], (-1.0*((data["inst_AMT_PAYMENT"]))) )))) + ((-1.0*(((((0.416667) < (((data["inst_AMT_PAYMENT"]) * 2.0)))*1.))))))/2.0)) 
    v["i238"] = np.tanh((((0.680000) < (((np.where(data["AMT_GOODS_PRICE_y"]>0, np.where(data["AMT_DOWN_PAYMENT"]>0, data["AMT_DOWN_PAYMENT"], data["CNT_INSTALMENT_FUTURE"] ), ((data["AMT_DOWN_PAYMENT"]) - (data["NAME_CASH_LOAN_PURPOSE_Business_development"])) )) / 2.0)))*1.)) 
    v["i239"] = np.tanh((((0.339286) < ((((0.339286) + (np.where(data["CNT_INSTALMENT_FUTURE"]>0, (((data["CNT_INSTALMENT"]) < ((((data["CNT_INSTALMENT_FUTURE"]) > (data["CNT_INSTALMENT"]))*1.)))*1.), data["CNT_INSTALMENT"] )))/2.0)))*1.)) 
    v["i240"] = np.tanh((-1.0*((np.minimum((((((data["CNT_INSTALMENT"]) > ((((data["CNT_INSTALMENT"]) > (0.323944))*1.)))*1.))), ((np.maximum((((((((data["CNT_INSTALMENT"]) > (data["CNT_PAYMENT"]))*1.)) / 2.0))), ((data["inst_AMT_PAYMENT"])))))))))) 
    v["i241"] = np.tanh((-1.0*((((np.where(data["AMT_DOWN_PAYMENT"]>0, data["avg_buro_buro_bal_status_2"], (-1.0*((np.where(data["avg_buro_buro_bal_status_2"]>0, ((((1.0)) < (((data["avg_buro_buro_bal_status_2"]) / 2.0)))*1.), data["avg_buro_buro_bal_status_2"] )))) )) * 2.0))))) 
    v["i242"] = np.tanh(((data["avg_buro_buro_bal_status_4"]) - (((data["avg_buro_buro_bal_status_2"]) - (np.where(-1.0>0, ((data["inst_AMT_PAYMENT"]) * 2.0), ((((data["inst_AMT_PAYMENT"]) * 2.0)) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"])) )))))) 
    v["i243"] = np.tanh(((data["CNT_INSTALMENT_FUTURE"]) * (np.maximum((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) > (((2.235290) - ((((data["AMT_DOWN_PAYMENT"]) + (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))/2.0)))))*1.))), ((((-3.0) - (0.680000)))))))) 
    v["i244"] = np.tanh(np.tanh((((((((data["AMT_CREDIT_y"]) * (np.maximum(((((-2.0) * (0.036585)))), ((np.minimum(((0.036585)), ((np.tanh((((data["te_FLAG_EMAIL"]) * 2.0)))))))))))) * 2.0)) * 2.0)))) 
    v["i245"] = np.tanh(((((data["avg_buro_buro_bal_status_4"]) * (data["te_FLAG_DOCUMENT_7"]))) * (((((((data["avg_buro_buro_bal_status_4"]) - (0.576923))) - (((((((0.0) < (11.714300))*1.)) < (data["te_FLAG_DOCUMENT_4"]))*1.)))) * 2.0)))) 
    v["i246"] = np.tanh(np.where((((1.105260) + ((((1.105260) + (data["te_FLAG_EMAIL"]))/2.0)))/2.0)>0, ((((data["te_FLAG_EMAIL"]) * (data["avg_buro_buro_bal_status_2"]))) - (data["avg_buro_buro_bal_status_4"])), np.tanh((0.416667)) )) 
    v["i247"] = np.tanh(((np.minimum(((((((np.minimum((((-1.0*((((((data["avg_buro_buro_bal_status_4"]) * ((((data["avg_buro_buro_bal_status_4"]) < (1.857140))*1.)))) * 2.0)))))), ((data["cc_bal_cc_bal_status__Refused"])))) * 2.0)) * 2.0))), ((data["cc_bal_cc_bal_status__Refused"])))) * 2.0)) 
    v["i248"] = np.tanh(np.maximum(((-3.0)), (((-1.0*((np.where(data["NAME_YIELD_GROUP_XNA"]>0, (((0.686567) < (((data["avg_buro_buro_bal_status_2"]) * 2.0)))*1.), (-1.0*((((data["avg_buro_buro_bal_status_2"]) * (((data["avg_buro_buro_bal_status_2"]) * 2.0)))))) )))))))) 
    v["i249"] = np.tanh(((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (((np.minimum(((((1.0) - (np.where(data["avg_buro_buro_bal_status_4"]>0, 0.494118, data["NAME_CONTRACT_TYPE_Revolving_loans"] ))))), ((np.where(data["avg_buro_buro_bal_status_4"]>0, 0.797872, data["NAME_CONTRACT_TYPE_Revolving_loans"] ))))) / 2.0)))) 
    v["i250"] = np.tanh((-1.0*(((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) > (np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["avg_buro_buro_bal_status_2"], (-1.0*(((((data["avg_buro_buro_bal_status_2"]) > ((((11.714300) + (-1.0))/2.0)))*1.)))) )))*1.))))) 
    v["i251"] = np.tanh(((np.where((((((-3.0) < (data["DAYS_LAST_DUE_1ST_VERSION"]))*1.)) + ((-1.0*((data["CODE_REJECT_REASON_SCOFR"])))))>0, data["cc_bal_SK_DPD_DEF"], -3.0 )) * (np.where(data["avg_buro_buro_bal_status_4"]>0, data["DAYS_LAST_DUE_1ST_VERSION"], 11.714300 )))) 
    v["i252"] = np.tanh((((1.0) < ((((((np.where((((data["cc_bal_AMT_PAYMENT_CURRENT"]) > (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))*1.)>0, data["CODE_REJECT_REASON_SCOFR"], 0.697674 )) * (data["cc_bal_AMT_PAYMENT_CURRENT"]))) + (data["cc_bal_AMT_PAYMENT_CURRENT"]))/2.0)))*1.)) 
    v["i253"] = np.tanh(((((((((data["CODE_REJECT_REASON_SCOFR"]) * (((data["avg_buro_buro_bal_status_2"]) * (((data["avg_buro_buro_bal_status_2"]) * (data["inst_AMT_PAYMENT"]))))))) / 2.0)) * (((data["avg_buro_buro_bal_status_2"]) + (data["CODE_REJECT_REASON_SCOFR"]))))) * 2.0)) 
    v["i254"] = np.tanh(((11.714300) * (((np.where(((data["avg_buro_buro_bal_status_4"]) * (data["avg_buro_buro_bal_status_2"]))>0, data["inst_AMT_PAYMENT"], ((data["inst_AMT_PAYMENT"]) + (14.800000)) )) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))))) 
    v["i255"] = np.tanh(np.minimum(((np.maximum(((data["AMT_GOODS_PRICE_y"])), (((-1.0*((data["CNT_INSTALMENT_FUTURE"])))))))), ((((data["CODE_REJECT_REASON_SCOFR"]) * (np.where(0.494118>0, (-1.0*((data["CNT_INSTALMENT_FUTURE"]))), ((0.0) - (0.036585)) ))))))) 
    v["i256"] = np.tanh((((((-1.0*((((data["YEARS_BEGINEXPLUATATION_MEDI"]) * (np.where(data["avg_buro_buro_bal_status_2"]>0, (-1.0*((data["YEARS_BEGINEXPLUATATION_MEDI"]))), np.tanh((14.800000)) ))))))) - (data["FLAG_DOCUMENT_4"]))) * 2.0)) 
    v["i257"] = np.tanh(((((((((data["avg_buro_buro_count"]) / 2.0)) / 2.0)) + (((((np.where(data["PRODUCT_COMBINATION_Cash_X_Sell__low"]>0, data["avg_buro_buro_count"], data["NAME_GOODS_CATEGORY_Weapon"] )) / 2.0)) + (data["PRODUCT_COMBINATION_Cash_X_Sell__low"]))))) * (data["avg_buro_buro_count"]))) 
    v["i258"] = np.tanh(((np.maximum(((data["avg_buro_MONTHS_BALANCE"])), ((((np.tanh(((((data["SK_ID_BUREAU"]) < (((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) < (data["avg_buro_MONTHS_BALANCE"]))*1.)) > (data["avg_buro_MONTHS_BALANCE"]))*1.)))*1.)))) * (data["SK_ID_BUREAU"])))))) * (data["SK_ID_BUREAU"]))) 
    v["i259"] = np.tanh(((((((np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), (((((2.235290) + (data["DAYS_CREDIT"]))/2.0))))) + (-3.0))/2.0)) > (np.minimum(((0.323944)), ((data["avg_buro_buro_count"])))))*1.)) 
    v["i260"] = np.tanh(np.minimum(((((data["TOTALAREA_MODE"]) * (np.tanh(((-1.0*((data["avg_buro_buro_bal_status_2"]))))))))), ((np.maximum(((np.minimum(((data["TOTALAREA_MODE"])), ((data["TOTALAREA_MODE"]))))), ((np.minimum(((data["avg_buro_buro_bal_status_2"])), ((data["DAYS_CREDIT"])))))))))) 
    v["i261"] = np.tanh(np.minimum(((((data["CODE_REJECT_REASON_SCOFR"]) * ((((-1.0*(((((0.697674) < (np.minimum(((data["CODE_REJECT_REASON_SCOFR"])), ((data["DAYS_LAST_DUE_1ST_VERSION"])))))*1.))))) * 2.0))))), ((((data["CODE_REJECT_REASON_SCOFR"]) * (data["NAME_PAYMENT_TYPE_Cashless_from_the_account_of_the_employer"])))))) 
    v["i262"] = np.tanh(((((data["avg_buro_buro_count"]) * (np.minimum(((data["avg_buro_buro_bal_status_4"])), ((((((9.0)) > (((data["avg_buro_buro_count"]) * (np.minimum(((((data["avg_buro_buro_bal_status_2"]) * 2.0))), ((data["avg_buro_buro_bal_status_4"])))))))*1.))))))) * (data["avg_buro_buro_bal_status_2"]))) 
    v["i263"] = np.tanh((((((data["avg_buro_buro_count"]) > (np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((np.tanh((11.714300)))))))*1.)) * (((11.714300) * (((11.714300) * (data["AMT_DOWN_PAYMENT"]))))))) 
    v["i264"] = np.tanh(np.minimum((((-1.0*((((np.maximum(((data["NAME_SELLER_INDUSTRY_Tourism"])), (((((14.800000) < (data["avg_buro_buro_bal_status_4"]))*1.))))) * 2.0)))))), (((-1.0*((np.maximum(((data["NAME_GOODS_CATEGORY_Weapon"])), (((((data["avg_buro_buro_bal_status_4"]) > (0.062500))*1.))))))))))) 
    v["i265"] = np.tanh(((data["SK_DPD"]) * (np.where((((data["SK_DPD"]) < (data["avg_buro_buro_bal_status_4"]))*1.)>0, (-1.0*((data["SK_DPD"]))), ((data["avg_buro_buro_bal_status_4"]) + (((-1.0) * (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * 2.0))))) )))) 
    v["i266"] = np.tanh((-1.0*((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (np.where(data["avg_buro_buro_bal_status_4"]>0, ((0.576923) * 2.0), data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"] ))))))) 
    v["i267"] = np.tanh(np.minimum(((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (np.where(data["avg_buro_buro_bal_status_4"]>0, data["CODE_REJECT_REASON_SYSTEM"], data["NAME_GOODS_CATEGORY_Insurance"] ))))), ((((data["avg_buro_buro_bal_status_4"]) - (np.where(data["avg_buro_buro_bal_status_4"]>0, ((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * 2.0), data["NAME_GOODS_CATEGORY_Fitness"] ))))))) 
    v["i268"] = np.tanh(np.where(((data["FLAG_DOCUMENT_4"]) + (np.tanh((((data["cc_bal_SK_DPD_DEF"]) * 2.0)))))>0, -3.0, (((data["cc_bal_SK_DPD"]) > (np.maximum(((11.714300)), ((11.714300)))))*1.) )) 
    v["i269"] = np.tanh(((np.maximum(((3.0)), (((9.0))))) * (((np.maximum(((data["cc_bal_SK_DPD_DEF"])), ((((((4.0)) < (np.maximum(((((data["CHANNEL_TYPE_AP___Cash_loan_"]) / 2.0))), ((data["CHANNEL_TYPE_AP___Cash_loan_"])))))*1.))))) * (data["CHANNEL_TYPE_AP___Cash_loan_"]))))) 
    v["i270"] = np.tanh(np.tanh((((((((np.tanh((data["avg_buro_buro_bal_status_4"]))) * (data["CHANNEL_TYPE_AP___Cash_loan_"]))) * (11.714300))) * (np.maximum(((data["CHANNEL_TYPE_AP___Cash_loan_"])), ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * 2.0))))))))) 
    v["i271"] = np.tanh(np.minimum((((((np.where(data["cc_bal_SK_DPD_DEF"]>0, data["avg_buro_buro_bal_status_2"], 0.031746 )) < ((((data["cc_bal_cc_bal_status__Refused"]) + ((((0.339286) < (data["avg_buro_buro_bal_status_2"]))*1.)))/2.0)))*1.))), (((((data["avg_buro_buro_bal_status_2"]) < (3.526320))*1.))))) 
    v["i272"] = np.tanh((-1.0*((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((((-3.0) + (((data["NAME_CLIENT_TYPE_Refreshed"]) - (((np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((((11.714300) - (data["avg_buro_buro_bal_status_4"])))))) * 2.0)))))))))))) 
    v["i273"] = np.tanh(np.minimum(((np.minimum((((((12.49790954589843750)) - (data["avg_buro_buro_bal_status_4"])))), ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (np.where(((data["avg_buro_buro_bal_status_4"]) / 2.0)>0, data["NAME_CASH_LOAN_PURPOSE_Business_development"], (-1.0*((data["avg_buro_buro_bal_status_4"]))) )))))))), ((11.714300)))) 
    v["i274"] = np.tanh((((-1.0*(((((((data["avg_buro_buro_bal_status_2"]) * (data["avg_buro_buro_bal_status_4"]))) < (np.maximum(((data["FLAG_DOCUMENT_17"])), ((np.tanh((data["FLAG_DOCUMENT_17"])))))))*1.))))) * 2.0)) 
    v["i275"] = np.tanh(((data["cc_bal_SK_DPD_DEF"]) + (((((np.where(((data["FLAG_DOCUMENT_10"]) - (data["cc_bal_SK_DPD_DEF"]))>0, data["cu__currency_3"], data["cc_bal_SK_DPD_DEF"] )) - ((((data["cu__currency_3"]) > (data["cc_bal_SK_DPD_DEF"]))*1.)))) / 2.0)))) 
    v["i276"] = np.tanh((-1.0*(((((11.714300) < (((np.maximum(((((14.800000) / 2.0))), (((((np.maximum(((0.036585)), ((2.0)))) + (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))/2.0))))) + (data["avg_buro_buro_bal_status_4"]))))*1.))))) 
    v["i277"] = np.tanh(np.maximum(((0.031746)), (((((0.031746) < (((data["NAME_YIELD_GROUP_XNA"]) * (np.minimum(((data["cc_bal_SK_DPD_DEF"])), ((np.where(0.416667>0, 0.031746, data["NAME_PORTFOLIO_XNA"] ))))))))*1.))))) 
    v["i278"] = np.tanh((((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((data["cu__currency_3"])))) + (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (((np.where(data["cu__currency_3"]>0, data["cu__currency_3"], data["NAME_GOODS_CATEGORY_Fitness"] )) * (((data["cu__currency_3"]) + (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))))))))/2.0)) 
    v["i279"] = np.tanh(((np.where(((((((data["NAME_GOODS_CATEGORY_Sport_and_Leisure"]) + (data["NAME_PORTFOLIO_XNA"]))/2.0)) + (data["REG_REGION_NOT_WORK_REGION"]))/2.0)>0, 0.680000, data["NAME_GOODS_CATEGORY_Sport_and_Leisure"] )) * (((data["NAME_GOODS_CATEGORY_Sport_and_Leisure"]) * ((((-1.0*((1.857140)))) * 2.0)))))) 
    v["i280"] = np.tanh(((((data["CHANNEL_TYPE_Stone"]) * 2.0)) * ((((0.062500) + (((((np.minimum(((data["CHANNEL_TYPE_Stone"])), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))) * (np.tanh((np.tanh((data["NAME_CONTRACT_STATUS_Approved"]))))))) / 2.0)))/2.0)))) 
    v["i281"] = np.tanh(np.where(data["FLAG_DOCUMENT_4"]>0, -3.0, ((((np.where(data["SK_ID_PREV_x"]>0, ((((((-3.0) + (data["SK_ID_PREV_x"]))/2.0)) > (2.235290))*1.), data["NAME_GOODS_CATEGORY_Direct_Sales"] )) * 2.0)) * 2.0) )) 
    v["i282"] = np.tanh(((np.where(((data["NAME_SELLER_INDUSTRY_MLM_partners"]) * (data["FLAG_DOCUMENT_15"]))>0, ((data["NAME_GOODS_CATEGORY_Fitness"]) * (data["FLAG_DOCUMENT_15"])), (((((((data["NAME_SELLER_INDUSTRY_MLM_partners"]) * (data["FLAG_DOCUMENT_15"]))) * 2.0)) + (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))/2.0) )) * 2.0)) 
    v["i283"] = np.tanh((((((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, 11.714300, data["FLAG_DOCUMENT_5"] )) * (data["te_FLAG_DOCUMENT_20"]))) > ((((0.416667) + (0.416667))/2.0)))*1.)) 
    v["i284"] = np.tanh(((((data["te_FLAG_DOCUMENT_3"]) * (((np.maximum((((((0.494118) < (data["cc_bal_SK_DPD_DEF"]))*1.))), (((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (2.235290))) > (0.0))*1.))))) * 2.0)))) * 2.0)) 
    v["i285"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, (-1.0*((((data["te_FLAG_DOCUMENT_6"]) * (2.0))))), ((data["te_FLAG_DOCUMENT_18"]) * ((((((data["nans"]) > ((((-1.0*((data["FLAG_DOCUMENT_18"])))) / 2.0)))*1.)) / 2.0))) )) 
    v["i286"] = np.tanh((((((((data["te_FLAG_DOCUMENT_8"]) < ((-1.0*((((((data["te_FLAG_DOCUMENT_8"]) * (data["FLAG_DOCUMENT_21"]))) + (((data["cc_bal_SK_DPD_DEF"]) + (((((7.04159450531005859)) + (data["te_FLAG_DOCUMENT_20"]))/2.0))))))))))*1.)) * 2.0)) * 2.0)) 
    v["i287"] = np.tanh(np.minimum(((np.maximum(((data["te_LIVE_REGION_NOT_WORK_REGION"])), ((data["NAME_GOODS_CATEGORY_Weapon"]))))), (((((((((((data["FLAG_DOCUMENT_10"]) * (data["FLAG_DOCUMENT_10"]))) > (data["cc_bal_SK_DPD_DEF"]))*1.)) * 2.0)) - ((((data["NAME_GOODS_CATEGORY_Weapon"]) + (data["te_LIVE_REGION_NOT_WORK_REGION"]))/2.0))))))) 
    v["i288"] = np.tanh(((((np.maximum(((np.maximum(((0.031746)), (((-1.0*((data["NAME_GOODS_CATEGORY_Insurance"])))))))), ((data["REG_REGION_NOT_WORK_REGION"])))) - (data["te_FLAG_DOCUMENT_20"]))) * (np.maximum(((0.062500)), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))))) 
    v["i289"] = np.tanh(((np.where(data["NAME_GOODS_CATEGORY_Education"]>0, data["te_REG_REGION_NOT_LIVE_REGION"], ((data["te_REG_REGION_NOT_LIVE_REGION"]) * (data["te_CODE_GENDER"])) )) * (((data["te_REG_REGION_NOT_LIVE_REGION"]) + (data["te_OCCUPATION_TYPE"]))))) 
    v["i290"] = np.tanh((((((data["FLAG_DOCUMENT_3"]) * ((((0.323944) < (data["RATE_DOWN_PAYMENT"]))*1.)))) + (np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, data["FLAG_DOCUMENT_3"], ((data["te_REG_REGION_NOT_LIVE_REGION"]) * (data["RATE_DOWN_PAYMENT"])) )))/2.0)) 
    v["i291"] = np.tanh(((np.where(((data["AMT_DOWN_PAYMENT"]) + (((0.697674) / 2.0)))>0, (((data["AMT_DOWN_PAYMENT"]) > (0.062500))*1.), data["AMT_DOWN_PAYMENT"] )) * (data["FLAG_DOCUMENT_6"]))) 
    v["i292"] = np.tanh(np.where((((-1.0*((data["te_NAME_CONTRACT_TYPE"])))) - (np.maximum(((0.797872)), ((data["NAME_CONTRACT_TYPE"])))))>0, data["NAME_CONTRACT_TYPE"], ((((np.tanh((data["FLAG_DOCUMENT_6"]))) / 2.0)) / 2.0) )) 
    v["i293"] = np.tanh(((((((((((((((data["FLAG_DOCUMENT_2"]) + (data["AMT_DOWN_PAYMENT"]))/2.0)) > (2.0))*1.)) - (data["NAME_GOODS_CATEGORY_Education"]))) - (data["NAME_GOODS_CATEGORY_Education"]))) - (data["FLAG_DOCUMENT_17"]))) - (data["NAME_GOODS_CATEGORY_Education"]))) 
    v["i294"] = np.tanh(np.maximum(((np.minimum((((-1.0*((data["AMT_INCOME_TOTAL"]))))), ((np.tanh(((((data["AMT_INCOME_TOTAL"]) < (((((1.220590) - (2.0))) / 2.0)))*1.)))))))), ((((data["AMT_INCOME_TOTAL"]) - (2.0)))))) 
    v["i295"] = np.tanh((((((data["AMT_INCOME_TOTAL"]) > ((((1.220590) > (-3.0))*1.)))*1.)) + (((3.0) * ((((data["AMT_ANNUITY_x"]) > (np.where(data["AMT_CREDIT_x"]>0, 1.220590, 2.235290 )))*1.)))))) 
    v["i296"] = np.tanh((((((data["NAME_GOODS_CATEGORY_Insurance"]) > (((data["AMT_ANNUITY_x"]) + (np.where(0.031746>0, 1.220590, data["NAME_GOODS_CATEGORY_Insurance"] )))))*1.)) - (0.062500))) 
    v["i297"] = np.tanh((-1.0*((np.where(data["te_HOUSETYPE_MODE"]>0, 0.062500, ((data["te_NAME_CONTRACT_TYPE"]) * (np.tanh((np.minimum(((np.maximum(((data["te_NAME_CONTRACT_TYPE"])), ((data["FLAG_DOCUMENT_3"]))))), (((-1.0*((data["te_NAME_CONTRACT_TYPE"])))))))))) ))))) 
    v["i298"] = np.tanh(np.minimum(((((np.where(data["te_ORGANIZATION_TYPE"]>0, (-1.0*(((((0.697674) < (((data["HOUR_APPR_PROCESS_START_x"]) / 2.0)))*1.)))), data["AMT_INCOME_TOTAL"] )) * (data["te_ORGANIZATION_TYPE"])))), ((np.maximum(((data["te_ORGANIZATION_TYPE"])), ((data["HOUR_APPR_PROCESS_START_x"]))))))) 
    v["i299"] = np.tanh(np.maximum(((np.maximum((((((data["AMT_INCOME_TOTAL"]) > (1.105260))*1.))), (((((data["PRODUCT_COMBINATION_Cash_X_Sell__high"]) > (2.235290))*1.)))))), ((0.062500)))) 
    v["i300"] = np.tanh(((np.minimum(((0.697674)), ((np.where(data["AMT_DOWN_PAYMENT"]>0, data["DAYS_ID_PUBLISH"], 0.062500 ))))) * (data["NAME_YIELD_GROUP_XNA"]))) 
    v["i301"] = np.tanh(((((((((1.0) < (data["te_NAME_EDUCATION_TYPE"]))*1.)) > (((np.maximum(((data["FLAG_EMP_PHONE"])), ((data["te_NAME_EDUCATION_TYPE"])))) / 2.0)))*1.)) * 2.0)) 
    v["i302"] = np.tanh((((((data["ORGANIZATION_TYPE"]) > (3.526320))*1.)) - (((3.526320) * ((((((data["NAME_CONTRACT_STATUS_Canceled"]) * (((data["ORGANIZATION_TYPE"]) * (data["ORGANIZATION_TYPE"]))))) > (3.526320))*1.)))))) 
    v["i303"] = np.tanh(((data["ORGANIZATION_TYPE"]) * (((np.maximum((((((3.526320) < (data["ORGANIZATION_TYPE"]))*1.))), (((((data["PRODUCT_COMBINATION_Cash_X_Sell__high"]) > ((((data["AMT_DOWN_PAYMENT"]) < (0.036585))*1.)))*1.))))) / 2.0)))) 
    v["i304"] = np.tanh(np.minimum(((((data["AMT_INCOME_TOTAL"]) * (((data["NAME_YIELD_GROUP_high"]) * (data["PRODUCT_COMBINATION_Cash_X_Sell__high"])))))), ((((data["NAME_YIELD_GROUP_high"]) * ((-1.0*((data["AMT_INCOME_TOTAL"]))))))))) 
    v["i305"] = np.tanh(np.where(data["NAME_TYPE_SUITE_Other_A"]>0, data["AMT_DOWN_PAYMENT"], np.tanh((((((-1.0*((data["avg_buro_buro_count"])))) > (((((-1.0*((data["avg_buro_buro_count"])))) > (data["AMT_DOWN_PAYMENT"]))*1.)))*1.))) )) 
    v["i306"] = np.tanh((((((data["avg_buro_buro_bal_status_C"]) > ((((0.416667) + ((-1.0*((np.minimum(((0.043478)), ((-2.0))))))))/2.0)))*1.)) * (((1.0) - (data["NAME_TYPE_SUITE_Other_A"]))))) 
    v["i307"] = np.tanh((-1.0*((((((((0.323944) + (data["AMT_GOODS_PRICE_x"]))/2.0)) < (np.where(data["AMT_ANNUITY_x"]>0, (((((data["AMT_ANNUITY_x"]) < (data["AMT_GOODS_PRICE_x"]))*1.)) * (data["AMT_ANNUITY_x"])), data["AMT_ANNUITY_x"] )))*1.))))) 
    v["i308"] = np.tanh(np.where(data["ty__Credit_card"]>0, ((np.tanh(((((np.tanh((data["ty__Credit_card"]))) > ((((((0.339286) * (0.339286))) + (0.339286))/2.0)))*1.)))) / 2.0), data["NAME_CASH_LOAN_PURPOSE_Furniture"] )) 
    v["i309"] = np.tanh(((((data["AMT_DOWN_PAYMENT"]) / 2.0)) + (((14.800000) * ((-1.0*(((((((data["AMT_DOWN_PAYMENT"]) - (1.857140))) > (0.036585))*1.))))))))) 
    v["i310"] = np.tanh(np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, ((-1.0) - (data["EXT_SOURCE_3"])), (((data["EXT_SOURCE_3"]) > (0.680000))*1.) )) 
    v["i311"] = np.tanh((((((((data["EXT_SOURCE_3"]) * (data["NAME_TYPE_SUITE_Other_A"]))) * (data["NAME_TYPE_SUITE_Other_A"]))) + ((((np.minimum(((data["nans"])), (((-1.0*((data["EXT_SOURCE_3"]))))))) + (data["NAME_TYPE_SUITE_Other_A"]))/2.0)))/2.0)) 
    v["i312"] = np.tanh(((((np.tanh(((((3.0) < (((((data["FONDKAPREMONT_MODE"]) - (0.323944))) - (np.maximum(((data["te_WALLSMATERIAL_MODE"])), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))))))*1.)))) * 2.0)) * 2.0)) 
    v["i313"] = np.tanh((((((((data["SK_ID_PREV_x"]) * (data["SK_ID_BUREAU"]))) < (((((data["SK_ID_BUREAU"]) + (-3.0))) / 2.0)))*1.)) * (np.minimum(((1.0)), ((((data["SK_ID_BUREAU"]) + (-3.0)))))))) 
    v["i314"] = np.tanh(((3.526320) * (np.where(((data["PRODUCT_COMBINATION_POS_mobile_with_interest"]) + (0.062500))>0, (((((data["SK_ID_BUREAU"]) > (2.235290))*1.)) * (data["SK_ID_PREV_x"])), 0.043478 )))) 
    v["i315"] = np.tanh(np.minimum(((((((data["SK_ID_PREV_x"]) * ((-1.0*((data["avg_buro_buro_bal_status_3"])))))) * 2.0))), ((((np.maximum(((data["te_HOUSETYPE_MODE"])), ((((((data["avg_buro_buro_bal_status_3"]) * 2.0)) * 2.0))))) * 2.0))))) 
    v["i316"] = np.tanh(np.tanh((((0.043478) + (((0.043478) + ((((((0.494118) + (data["WALLSMATERIAL_MODE"]))/2.0)) - ((((data["WALLSMATERIAL_MODE"]) > ((((data["te_FONDKAPREMONT_MODE"]) > (0.043478))*1.)))*1.)))))))))) 
    v["i317"] = np.tanh(np.where(np.tanh((np.minimum((((((np.minimum(((0.036585)), ((0.036585)))) > ((((((-1.0*((data["AMT_CREDIT_SUM_DEBT"])))) / 2.0)) / 2.0)))*1.))), ((data["SK_ID_BUREAU"])))))>0, data["SK_ID_BUREAU"], data["AMT_CREDIT_SUM_DEBT"] )) 
    v["i318"] = np.tanh(((data["ty__Consumer_credit"]) * (np.where(((data["AMT_CREDIT_SUM"]) * (0.686567))>0, ((((data["AMT_CREDIT_SUM"]) * (0.686567))) + (data["ca__Closed"])), data["NAME_GOODS_CATEGORY_Insurance"] )))) 
    v["i319"] = np.tanh((((((((2.235290) < (data["AMT_GOODS_PRICE_x"]))*1.)) * (np.where(np.minimum(((2.235290)), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))>0, data["AMT_CREDIT_SUM_DEBT"], data["DAYS_LAST_DUE"] )))) * (2.235290))) 
    v["i320"] = np.tanh((((((((1.220590) < (((data["ty__Consumer_credit"]) - ((((((data["inst_AMT_PAYMENT"]) > (data["AMT_CREDIT_SUM"]))*1.)) + (data["AMT_CREDIT_SUM"]))))))*1.)) / 2.0)) * (data["ca__Closed"]))) 
    v["i321"] = np.tanh(((np.where(((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * 2.0)>0, ((((data["ty__Consumer_credit"]) - (-2.0))) * 2.0), np.minimum((((-1.0*(((((data["AMT_CREDIT_SUM"]) > (0.323944))*1.)))))), ((0.686567))) )) / 2.0)) 
    v["i322"] = np.tanh((((((2.235290) < ((((data["nans"]) + ((((0.680000) < (((data["te_WALLSMATERIAL_MODE"]) / 2.0)))*1.)))/2.0)))*1.)) * (data["nans"]))) 
    v["i323"] = np.tanh(np.minimum((((-1.0*(((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) > ((((data["te_WALLSMATERIAL_MODE"]) > (data["NAME_GOODS_CATEGORY_Insurance"]))*1.)))*1.)))))), ((np.tanh(((-1.0*(((((data["te_WALLSMATERIAL_MODE"]) > ((((0.697674) > (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))*1.)))*1.)))))))))) 
    v["i324"] = np.tanh((((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) + (0.339286))/2.0)) * ((-1.0*(((((np.minimum((((((((0.062500) < (data["te_WALLSMATERIAL_MODE"]))*1.)) + (data["NAME_CASH_LOAN_PURPOSE_Furniture"])))), ((data["WALLSMATERIAL_MODE"])))) + (data["te_HOUSETYPE_MODE"]))/2.0))))))) 
    v["i325"] = np.tanh(((((data["CNT_INSTALMENT"]) * (((np.where(np.tanh((data["ty__Credit_card"]))>0, data["NAME_GOODS_CATEGORY_Direct_Sales"], (-1.0*((data["inst_AMT_PAYMENT"]))) )) + (((data["inst_AMT_PAYMENT"]) * (data["ty__Credit_card"]))))))) / 2.0)) 
    v["i326"] = np.tanh(((((((((-1.0*((np.minimum(((data["CNT_INSTALMENT"])), ((data["CNT_INSTALMENT"]))))))) * (((data["CNT_INSTALMENT"]) / 2.0)))) + (data["inst_AMT_PAYMENT"]))/2.0)) * (((data["ty__Consumer_credit"]) + (data["ty__Credit_card"]))))) 
    v["i327"] = np.tanh(np.where(data["CODE_REJECT_REASON_CLIENT"]>0, data["PRODUCT_COMBINATION_POS_industry_without_interest"], np.where((((data["inst_AMT_PAYMENT"]) < ((((((data["PRODUCT_COMBINATION_POS_industry_without_interest"]) > (1.0))*1.)) / 2.0)))*1.)>0, (-1.0*((data["PRODUCT_COMBINATION_POS_industry_without_interest"]))), data["PRODUCT_COMBINATION_POS_industry_without_interest"] ) )) 
    v["i328"] = np.tanh(((np.where(data["inst_AMT_INSTALMENT"]>0, ((np.maximum(((data["NAME_CONTRACT_STATUS_Unused_offer"])), ((data["ty__Consumer_credit"])))) / 2.0), np.minimum(((0.036585)), (((((data["NAME_CONTRACT_STATUS_Unused_offer"]) + (data["ty__Consumer_credit"]))/2.0)))) )) / 2.0)) 
    v["i329"] = np.tanh((-1.0*((((((((((((0.686567) < (data["te_NAME_HOUSING_TYPE"]))*1.)) > (((((data["te_NAME_HOUSING_TYPE"]) / 2.0)) + ((((((3.0) > (data["FLAG_DOCUMENT_10"]))*1.)) / 2.0)))))*1.)) * 2.0)) * 2.0))))) 
    v["i330"] = np.tanh(np.where(data["te_DAYS_ID_PUBLISH"]>0, (((((((1.220590) - (2.0))) / 2.0)) > (data["AMT_CREDIT_SUM"]))*1.), (((data["AMT_CREDIT_SUM"]) > (np.maximum(((0.323944)), ((data["NAME_TYPE_SUITE_Other_B"])))))*1.) )) 
    v["i331"] = np.tanh(np.where(((((np.where(((data["te_DAYS_ID_PUBLISH"]) / 2.0)>0, 1.0, ((data["AMT_CREDIT_x"]) / 2.0) )) / 2.0)) / 2.0)>0, ((((data["te_DAYS_ID_PUBLISH"]) / 2.0)) * (data["te_DAYS_ID_PUBLISH"])), data["te_DAYS_ID_PUBLISH"] )) 
    v["i332"] = np.tanh((((data["inst_AMT_PAYMENT"]) < ((-1.0*(((((((0.043478) / 2.0)) + (((np.maximum(((3.0)), ((((data["NAME_TYPE_SUITE_Other_B"]) - (1.857140)))))) - (data["NAME_TYPE_SUITE_Other_B"]))))/2.0))))))*1.)) 
    v["i333"] = np.tanh(((np.minimum((((((((((8.0)) - (data["inst_AMT_PAYMENT"]))) - (((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) / 2.0)))) * 2.0))), ((((((((data["NAME_CASH_LOAN_PURPOSE_Furniture"]) / 2.0)) * 2.0)) / 2.0))))) * (3.0))) 
    v["i334"] = np.tanh(((((np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, 3.526320, ((np.maximum(((np.maximum(((data["cu__currency_3"])), ((data["FLAG_DOCUMENT_13"]))))), ((-2.0)))) * 2.0) )) * (((data["cu__currency_3"]) * 2.0)))) * 2.0)) 
    v["i335"] = np.tanh(((np.where(data["cc_bal_cc_bal_status__Sent_proposal"]>0, np.minimum(((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"])), ((data["Amortized_debt"]))))), (((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Buying_a_used_car"])))))), ((((((-1.0*((data["cc_bal_cc_bal_status__Sent_proposal"])))) / 2.0)) < (data["NAME_CASH_LOAN_PURPOSE_Buying_a_used_car"]))*1.) )) * 2.0)) 
    v["i336"] = np.tanh(((((((((data["Amortized_debt"]) + (0.031746))/2.0)) + ((((0.416667) + (data["Amortized_debt"]))/2.0)))/2.0)) - ((((data["Amortized_debt"]) > (data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"]))*1.)))) 
    v["i337"] = np.tanh(np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Refusal_to_name_the_goal"])), (((((7.0)) + (((14.800000) - (np.maximum(((np.maximum(((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Business_development"])), ((data["NAME_GOODS_CATEGORY_Education"]))))), ((14.800000))))), ((data["NAME_CASH_LOAN_PURPOSE_Furniture"]))))))))))) 
    v["i338"] = np.tanh(np.where(data["cc_bal_cc_bal_status__Sent_proposal"]>0, ((data["NAME_CONTRACT_STATUS_Unused_offer"]) * (data["cc_bal_cc_bal_status__Sent_proposal"])), (-1.0*(((((((data["CODE_REJECT_REASON_CLIENT"]) * (data["CODE_REJECT_REASON_CLIENT"]))) < (((data["NAME_CONTRACT_STATUS_Unused_offer"]) * 2.0)))*1.)))) )) 
    v["i339"] = np.tanh(((data["NAME_PORTFOLIO_XNA"]) * (((np.where(((data["WALLSMATERIAL_MODE"]) / 2.0)>0, (((data["WALLSMATERIAL_MODE"]) > (3.526320))*1.), (((data["NAME_CONTRACT_STATUS_Unused_offer"]) > ((6.0)))*1.) )) * (3.526320))))) 
    v["i340"] = np.tanh(((((((data["SK_ID_PREV_x"]) * (((np.maximum(((((-3.0) + (np.where(data["SK_ID_PREV_x"]>0, data["NAME_CASH_LOAN_PURPOSE_Business_development"], data["SK_ID_PREV_x"] ))))), ((data["CNT_INSTALMENT_FUTURE"])))) / 2.0)))) / 2.0)) - (0.036585))) 
    v["i341"] = np.tanh((-1.0*((((((np.where(data["AMT_DOWN_PAYMENT"]>0, 0.043478, (-1.0*((((np.where(data["NAME_GOODS_CATEGORY_Direct_Sales"]>0, 0.043478, 0.0 )) * 2.0)))) )) * 2.0)) * 2.0))))) 
    v["i342"] = np.tanh(np.where(((((data["NAME_YIELD_GROUP_high"]) * ((((data["te_FLAG_DOCUMENT_8"]) < (0.0))*1.)))) - (((1.220590) / 2.0)))>0, data["DAYS_TERMINATION"], (((data["te_FLAG_DOCUMENT_8"]) < (0.0))*1.) )) 
    v["i343"] = np.tanh(((0.062500) * (np.minimum(((data["DAYS_LAST_DUE"])), ((((data["te_FLAG_DOCUMENT_3"]) - (data["te_CODE_GENDER"])))))))) 
    v["i344"] = np.tanh(((((data["FLAG_DOCUMENT_5"]) * (np.minimum(((np.where((((data["NAME_CONTRACT_STATUS_Canceled"]) > (0.062500))*1.)>0, data["WALLSMATERIAL_MODE"], (-1.0*((data["WALLSMATERIAL_MODE"]))) ))), (((-1.0*((data["WALLSMATERIAL_MODE"]))))))))) / 2.0)) 
    v["i345"] = np.tanh((((-1.0*(((((data["FLAG_DOCUMENT_8"]) > ((((9.55103015899658203)) * ((((0.797872) > (((((data["AMT_DOWN_PAYMENT"]) * 2.0)) - ((0.0)))))*1.)))))*1.))))) * 2.0)) 
    v["i346"] = np.tanh(((((((3.0) + (((np.where(data["cc_bal_cc_bal_status__Refused"]>0, (-1.0*((data["FLAG_DOCUMENT_6"]))), data["DAYS_CREDIT"] )) + ((((0.0) < (data["FLAG_DOCUMENT_6"]))*1.)))))/2.0)) < (data["DAYS_CREDIT"]))*1.)) 
    v["i347"] = np.tanh(np.where((((-1.0*((data["AMT_DOWN_PAYMENT"])))) / 2.0)>0, (((((data["OCCUPATION_TYPE"]) > (2.0))*1.)) * (data["NAME_CONTRACT_STATUS_Canceled"])), np.minimum(((data["AMT_DOWN_PAYMENT"])), ((data["FLAG_DOCUMENT_5"]))) )) 
    v["i348"] = np.tanh(((np.where(((data["NAME_CONTRACT_STATUS_Approved"]) - ((((data["AMT_DOWN_PAYMENT"]) < (((((data["AMT_DOWN_PAYMENT"]) / 2.0)) * (data["OCCUPATION_TYPE"]))))*1.)))>0, 0.031746, (-1.0*((data["OCCUPATION_TYPE"]))) )) * (data["AMT_DOWN_PAYMENT"]))) 
    v["i349"] = np.tanh((((-1.0*((data["ty__Microloan"])))) * ((((((np.where((-1.0*((data["AMT_DOWN_PAYMENT"])))>0, (-1.0*((data["ty__Microloan"]))), (-1.0*((data["AMT_DOWN_PAYMENT"]))) )) * 2.0)) < (data["NAME_CONTRACT_STATUS_Approved"]))*1.)))) 
    v["i350"] = np.tanh((((-1.0) + ((((data["WEEKDAY_APPR_PROCESS_START_WEDNESDAY"]) > (((((data["NAME_CONTRACT_STATUS_Canceled"]) - (np.maximum((((-1.0*((0.062500))))), ((1.857140)))))) * 2.0)))*1.)))/2.0)) 
    v["i351"] = np.tanh((((np.where((-1.0*((data["AMT_DOWN_PAYMENT"])))>0, 2.0, ((data["cc_bal_cc_bal_status__Refused"]) / 2.0) )) < ((((-1.0*((1.105260)))) + (np.where(data["cc_bal_cc_bal_status__Refused"]>0, data["NAME_CASH_LOAN_PURPOSE_Repairs"], data["AMT_DOWN_PAYMENT"] )))))*1.)) 
    v["i352"] = np.tanh(((np.maximum(((((data["AMT_DOWN_PAYMENT"]) - (14.800000)))), ((data["CODE_REJECT_REASON_VERIF"])))) * (((((data["AMT_DOWN_PAYMENT"]) - (np.where(data["CODE_REJECT_REASON_VERIF"]>0, 0.062500, 1.220590 )))) / 2.0)))) 
    v["i353"] = np.tanh(np.where(data["te_OCCUPATION_TYPE"]>0, (((data["te_OCCUPATION_TYPE"]) < (0.680000))*1.), (((data["te_OCCUPATION_TYPE"]) < ((((-3.0) + ((((np.minimum(((0.494118)), ((3.0)))) < (data["AMT_DOWN_PAYMENT"]))*1.)))/2.0)))*1.) )) 
    v["i354"] = np.tanh(np.where(np.maximum(((data["NAME_GOODS_CATEGORY_Direct_Sales"])), ((-3.0)))>0, (-1.0*((np.minimum(((data["OBS_60_CNT_SOCIAL_CIRCLE"])), ((data["OBS_60_CNT_SOCIAL_CIRCLE"])))))), (((0.0) > ((7.0)))*1.) )) 
    v["i355"] = np.tanh(np.where(((np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, data["cc_bal_cc_bal_status__Refused"], (((data["DEF_60_CNT_SOCIAL_CIRCLE"]) > (((2.235290) + ((((6.48228216171264648)) / 2.0)))))*1.) )) * 2.0)>0, data["DEF_60_CNT_SOCIAL_CIRCLE"], ((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * 2.0) )) 
    v["i356"] = np.tanh(((((((11.51243686676025391)) + (14.800000))/2.0)) * (((data["cc_bal_SK_DPD_DEF"]) * (np.maximum(((data["ty__Microloan"])), ((data["inst_AMT_PAYMENT"])))))))) 
    v["i357"] = np.tanh((((-2.0) > ((-1.0*(((((((((((((data["AMT_DOWN_PAYMENT"]) < (data["cc_bal_SK_DPD_DEF"]))*1.)) * (data["inst_AMT_INSTALMENT"]))) / 2.0)) * (data["inst_AMT_INSTALMENT"]))) / 2.0))))))*1.)) 
    v["i358"] = np.tanh((((((data["ty__Microloan"]) > ((((11.714300) + ((((11.714300) + ((-1.0*((((data["AMT_DOWN_PAYMENT"]) - (((((-1.0*((data["ty__Microloan"])))) < (data["AMT_DOWN_PAYMENT"]))*1.))))))))/2.0)))/2.0)))*1.)) * 2.0)) 
    v["i359"] = np.tanh(((((data["inst_AMT_PAYMENT"]) * (((((data["AMT_DOWN_PAYMENT"]) * (data["inst_AMT_PAYMENT"]))) * (data["CHANNEL_TYPE_Car_dealer"]))))) - (((data["CHANNEL_TYPE_Car_dealer"]) * 2.0)))) 
    v["i360"] = np.tanh((((((0.576923) < (data["cc_bal_cc_bal_status__Refused"]))*1.)) - (((((np.maximum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((((data["te_FLAG_WORK_PHONE"]) * (0.036585)))))) + (0.043478))) + (data["CHANNEL_TYPE_Car_dealer"]))))) 
    v["i361"] = np.tanh(np.where(((data["cc_bal_SK_DPD_DEF"]) - (0.036585))>0, data["WEEKDAY_APPR_PROCESS_START"], ((11.714300) * (((data["cc_bal_SK_DPD_DEF"]) * (((((data["cc_bal_SK_DPD_DEF"]) * (11.714300))) + (data["inst_AMT_PAYMENT"])))))) )) 
    v["i362"] = np.tanh(np.where(0.031746>0, ((((np.maximum(((((((4.0)) < (data["inst_AMT_INSTALMENT"]))*1.))), ((((((4.0)) < (data["AMT_DOWN_PAYMENT"]))*1.))))) * (3.526320))) * (data["inst_AMT_INSTALMENT"])), 3.526320 )) 
    v["i363"] = np.tanh(((data["WEEKDAY_APPR_PROCESS_START"]) * ((-1.0*((np.tanh((np.maximum(((data["cc_bal_AMT_PAYMENT_CURRENT"])), ((((((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))) + ((-1.0*((np.tanh((data["cc_bal_AMT_PAYMENT_CURRENT"])))))))/2.0)))))))))))) 
    v["i364"] = np.tanh((-1.0*(((((((((6.0)) + (data["te_FLAG_DOCUMENT_5"]))) * 2.0)) * ((((data["inst_AMT_PAYMENT"]) > ((((data["te_FLAG_DOCUMENT_5"]) + ((4.66436243057250977)))/2.0)))*1.))))))) 
    v["i365"] = np.tanh((((data["inst_AMT_INSTALMENT"]) + ((((((data["CNT_INSTALMENT_FUTURE"]) > (3.0))*1.)) - (np.tanh((np.maximum(((np.tanh((data["inst_AMT_INSTALMENT"])))), (((-1.0*((0.036585))))))))))))/2.0)) 
    v["i366"] = np.tanh(((data["inst_AMT_PAYMENT"]) * ((-1.0*(((((((data["AMT_ANNUITY"]) > (2.235290))*1.)) * (np.maximum(((14.800000)), ((data["AMT_ANNUITY"]))))))))))) 
    v["i367"] = np.tanh(np.where(data["NAME_CASH_LOAN_PURPOSE_Building_a_house_or_an_annex"]>0, ((((((8.0)) + (((np.minimum(((data["NFLAG_INSURED_ON_APPROVAL"])), ((0.797872)))) * 2.0)))) < (data["NAME_CASH_LOAN_PURPOSE_Building_a_house_or_an_annex"]))*1.), ((-2.0) * (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])) )) 
    v["i368"] = np.tanh(((((np.minimum(((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (-1.0)))), ((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (data["AMT_DOWN_PAYMENT"]))) * 2.0))))) - (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))) * (data["AMT_DOWN_PAYMENT"]))) 
    v["i369"] = np.tanh(np.where(data["CNT_INSTALMENT"]>0, np.minimum(((((0.680000) - (data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"])))), ((np.maximum(((((data["CNT_INSTALMENT"]) - (data["CNT_INSTALMENT"])))), ((data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"])))))), ((0.036585) * (data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"])) )) 
    v["i370"] = np.tanh(((((-1.0*((np.where(((((1.857140) * (((data["NAME_GOODS_CATEGORY_Insurance"]) / 2.0)))) - (data["cc_bal_SK_DPD_DEF"]))>0, data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"], 0.043478 ))))) + (((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * (0.062500))))/2.0)) 
    v["i371"] = np.tanh(np.minimum(((0.416667)), ((((((((((1.105260) - ((((((0.30295735597610474)) / 2.0)) * (data["AMT_DOWN_PAYMENT"]))))) + (data["NFLAG_INSURED_ON_APPROVAL"]))/2.0)) < (((data["AMT_DOWN_PAYMENT"]) / 2.0)))*1.))))) 
    v["i372"] = np.tanh((((((data["cc_bal_SK_DPD_DEF"]) < (np.minimum(((data["NAME_YIELD_GROUP_middle"])), ((((((data["NAME_GOODS_CATEGORY_Insurance"]) * (data["NAME_YIELD_GROUP_middle"]))) * 2.0))))))*1.)) * (data["NAME_YIELD_GROUP_middle"]))) 
    v["i373"] = np.tanh((-1.0*((((np.minimum(((np.tanh(((((((np.tanh((0.339286))) > (data["NAME_YIELD_GROUP_middle"]))*1.)) / 2.0))))), (((((data["NAME_YIELD_GROUP_middle"]) > (data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"]))*1.))))) / 2.0))))) 
    v["i374"] = np.tanh(np.where((((0.680000) < (np.minimum((((((data["NAME_YIELD_GROUP_middle"]) + (0.0))/2.0))), ((data["AMT_CREDIT_SUM"])))))*1.)>0, data["AMT_CREDIT_SUM"], ((np.minimum(((data["AMT_CREDIT_SUM"])), ((data["AMT_CREDIT_SUM"])))) * (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])) )) 
    v["i375"] = np.tanh(((np.minimum(((((np.where(data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]>0, ((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]) - (14.800000)), ((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]) * (data["NAME_GOODS_CATEGORY_Computers"])) )) * 2.0))), ((((data["NAME_CASH_LOAN_PURPOSE_Gasification___water_supply"]) / 2.0))))) * 2.0)) 
    v["i376"] = np.tanh(np.where(data["te_NAME_HOUSING_TYPE"]>0, 0.323944, (((((-1.0*(((((data["te_NAME_HOUSING_TYPE"]) < (np.tanh((-2.0))))*1.))))) * 2.0)) * 2.0) )) 
    v["i377"] = np.tanh((((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))) * (np.where(data["NAME_YIELD_GROUP_middle"]>0, np.minimum(((-2.0)), ((((np.tanh((2.235290))) / 2.0)))), (((0.686567) + (data["NAME_YIELD_GROUP_middle"]))/2.0) )))) 
    v["i378"] = np.tanh((-1.0*((((((data["inst_AMT_PAYMENT"]) * (0.031746))) - ((((0.416667) < (np.where(data["te_NAME_HOUSING_TYPE"]>0, data["inst_AMT_PAYMENT"], ((data["AMT_CREDIT_SUM"]) * (data["inst_AMT_PAYMENT"])) )))*1.))))))) 
    v["i379"] = np.tanh((((((data["AMT_CREDIT_SUM"]) + (((0.043478) * 2.0)))) < (np.tanh((data["AMT_CREDIT_SUM_DEBT"]))))*1.)) 
    v["i380"] = np.tanh((((-1.0*(((((0.416667) + (np.where(np.minimum(((data["EXT_SOURCE_3"])), ((0.494118)))>0, data["ca__Active"], np.tanh((data["te_NAME_HOUSING_TYPE"])) )))/2.0))))) - (((0.323944) / 2.0)))) 
    v["i381"] = np.tanh(np.where(data["SK_ID_BUREAU"]>0, ((data["ty__Credit_card"]) * (data["SK_ID_BUREAU"])), (((((data["inst_AMT_PAYMENT"]) < (np.tanh((data["ty__Credit_card"]))))*1.)) * (((data["SK_ID_BUREAU"]) + (0.494118)))) )) 
    v["i382"] = np.tanh(((((((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * ((((data["DAYS_ID_PUBLISH"]) < (1.105260))*1.)))) * 2.0)) * 2.0)) * 2.0)) * (np.tanh((np.tanh((((data["DAYS_ID_PUBLISH"]) + (data["DAYS_ID_PUBLISH"]))))))))) 
    v["i383"] = np.tanh((-1.0*((np.maximum((((((data["DAYS_CREDIT_UPDATE"]) < (-1.0))*1.))), ((((((((((data["NAME_GOODS_CATEGORY_Insurance"]) < (14.800000))*1.)) * 2.0)) < (data["NAME_PRODUCT_TYPE_walk_in"]))*1.)))))))) 
    v["i384"] = np.tanh(((((((data["DAYS_CREDIT"]) * (np.where(0.036585>0, data["ca__Sold"], data["DAYS_CREDIT"] )))) * 2.0)) - (np.where(data["ca__Sold"]>0, data["ca__Sold"], data["ca__Sold"] )))) 
    v["i385"] = np.tanh(np.maximum(((((data["ca__Sold"]) - (((2.0) * 2.0))))), ((np.tanh((((data["DAYS_CREDIT"]) * ((((data["NAME_PRODUCT_TYPE_walk_in"]) > ((((3.526320) < (data["NAME_PRODUCT_TYPE_walk_in"]))*1.)))*1.))))))))) 
    v["i386"] = np.tanh(((((((data["NAME_CASH_LOAN_PURPOSE_Other"]) > (0.416667))*1.)) + (np.minimum(((((0.494118) / 2.0))), ((((data["DAYS_ENDDATE_FACT"]) * ((-1.0*((((0.494118) / 2.0)))))))))))/2.0)) 
    v["i387"] = np.tanh(np.minimum((((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) > ((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])))))*1.))), ((((0.576923) - (np.tanh((np.maximum(((np.tanh(((((data["AMT_REQ_CREDIT_BUREAU_YEAR"]) + (0.416667))/2.0))))), ((data["AMT_REQ_CREDIT_BUREAU_YEAR"]))))))))))) 
    v["i388"] = np.tanh(((data["ca__Sold"]) * (np.maximum((((-1.0*((data["ca__Sold"]))))), (((((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) < (data["FLAG_DOCUMENT_10"]))*1.)) + (((((data["ca__Sold"]) - (((11.714300) / 2.0)))) * 2.0))))))))) 
    v["i389"] = np.tanh((-1.0*((((np.minimum((((((data["DAYS_CREDIT"]) + (2.0))/2.0))), ((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * (data["DAYS_CREDIT"])))))) * (np.where(data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]>0, 2.0, data["DAYS_CREDIT"] ))))))) 
    v["i390"] = np.tanh(((data["AMT_CREDIT_SUM_DEBT"]) * (((data["AMT_CREDIT_SUM_DEBT"]) * (np.where(data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]>0, data["AMT_CREDIT_SUM_DEBT"], np.tanh((np.where(data["AMT_CREDIT_SUM_DEBT"]>0, data["NAME_GOODS_CATEGORY_Insurance"], np.minimum(((data["DAYS_LAST_DUE_1ST_VERSION"])), ((data["NAME_GOODS_CATEGORY_Insurance"]))) ))) )))))) 
    v["i391"] = np.tanh((((((((((((((data["AMT_CREDIT_MAX_OVERDUE"]) > (np.where(data["AMT_CREDIT_MAX_OVERDUE"]>0, 0.062500, data["FLAG_DOCUMENT_4"] )))*1.)) * 2.0)) + (np.tanh((data["AMT_CREDIT_MAX_OVERDUE"]))))) + (data["FLAG_DOCUMENT_4"]))) * 2.0)) * 2.0)) 
    v["i392"] = np.tanh(np.minimum(((((data["FLAG_DOCUMENT_15"]) * (data["NAME_CASH_LOAN_PURPOSE_Business_development"])))), ((((data["FLAG_EMAIL"]) * ((((((data["DAYS_LAST_DUE_1ST_VERSION"]) > (0.680000))*1.)) * (((data["FLAG_DOCUMENT_15"]) * (data["FLAG_EMAIL"])))))))))) 
    v["i393"] = np.tanh(np.minimum(((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) * (((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) + (data["NAME_GOODS_CATEGORY_Insurance"])))))), (((-1.0*(((((data["cu__currency_4"]) + ((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) > (((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) * (data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]))))*1.)))/2.0)))))))) 
    v["i394"] = np.tanh(((((((np.minimum(((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])), ((data["AMT_REQ_CREDIT_BUREAU_YEAR"])))) * (((0.043478) * 2.0)))) - (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))) - (((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * (((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) * (data["AMT_REQ_CREDIT_BUREAU_YEAR"]))))))) 
    v["i395"] = np.tanh(((np.where(data["cc_bal_SK_DPD_DEF"]>0, data["cc_bal_SK_DPD"], data["ty__Loan_for_working_capital_replenishment"] )) * ((-1.0*((((0.043478) - ((((14.800000) > ((((data["ty__Loan_for_working_capital_replenishment"]) + (2.0))/2.0)))*1.))))))))) 
    v["i396"] = np.tanh(((0.043478) + ((-1.0*((np.maximum((((((14.800000) < (data["ty__Loan_for_working_capital_replenishment"]))*1.))), ((np.where(data["ty__Loan_for_working_capital_replenishment"]>0, ((2.235290) * (data["cc_bal_SK_DPD"])), data["NAME_GOODS_CATEGORY_Insurance"] )))))))))) 
    v["i397"] = np.tanh((((9.0)) * ((((((0.680000) < (data["cc_bal_SK_DPD_DEF"]))*1.)) + ((((data["NAME_CASH_LOAN_PURPOSE_Hobby"]) > ((((data["cc_bal_SK_DPD_DEF"]) + (0.0))/2.0)))*1.)))))) 
    v["i398"] = np.tanh(np.where(data["cc_bal_SK_DPD"]>0, data["NAME_GOODS_CATEGORY_Insurance"], ((0.686567) * (np.minimum((((-1.0*((((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]) / 2.0)))))), (((((-1.0*((np.maximum(((data["FLAG_DOCUMENT_4"])), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"]))))))) / 2.0)))))) )) 
    v["i399"] = np.tanh(np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, data["te_FLAG_DOCUMENT_19"], ((((((data["te_FLAG_DOCUMENT_19"]) - (((((((13.52401924133300781)) < (data["avg_buro_buro_bal_status_4"]))*1.)) / 2.0)))) - (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))) * (data["avg_buro_buro_bal_status_4"])) )) 
    v["i400"] = np.tanh(((data["avg_buro_buro_bal_status_4"]) * (((((np.minimum(((data["cu__currency_4"])), ((data["te_FLAG_DOCUMENT_20"])))) - (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))) - (((np.maximum(((data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"])), ((data["cc_bal_SK_DPD"])))) * 2.0)))))) 
    v["i401"] = np.tanh(((np.where(data["te_FLAG_DOCUMENT_19"]>0, data["te_FLAG_DOCUMENT_19"], data["avg_buro_buro_bal_status_4"] )) * ((-1.0*((np.where(data["NAME_CASH_LOAN_PURPOSE_Business_development"]>0, data["te_FLAG_DOCUMENT_19"], data["avg_buro_buro_bal_status_4"] ))))))) 
    v["i402"] = np.tanh(np.minimum((((((data["NAME_GOODS_CATEGORY_Insurance"]) > (np.maximum(((np.minimum((((-1.0*((data["cu__currency_4"]))))), ((data["NAME_GOODS_CATEGORY_Insurance"]))))), ((data["te_FLAG_DOCUMENT_19"])))))*1.))), ((((data["te_FLAG_DOCUMENT_19"]) + ((((4.89787912368774414)) - (data["cu__currency_4"])))))))) 
    v["i403"] = np.tanh(np.where(data["FLAG_DOCUMENT_10"]>0, data["inst_AMT_PAYMENT"], ((((((data["inst_AMT_PAYMENT"]) * (data["NAME_CASH_LOAN_PURPOSE_Buying_a_holiday_home___land"]))) + (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (((data["inst_AMT_PAYMENT"]) + (data["HOUR_APPR_PROCESS_START_y"]))))))) * 2.0) )) 
    v["i404"] = np.tanh(np.tanh((np.tanh((np.tanh((np.tanh((((((data["AMT_CREDIT_y"]) + ((((-1.0*((data["AMT_APPLICATION"])))) - ((((data["AMT_APPLICATION"]) < (-1.0))*1.)))))) * 2.0)))))))))) 
    v["i405"] = np.tanh(((data["NAME_CLIENT_TYPE_Repeater"]) * (((0.036585) * (((((3.526320) * (((((data["NAME_CLIENT_TYPE_Repeater"]) / 2.0)) + (data["NAME_PRODUCT_TYPE_XNA"]))))) * 2.0)))))) 
    v["i406"] = np.tanh(((data["NAME_CONTRACT_TYPE_Revolving_loans"]) * (((((0.680000) + (data["AMT_DOWN_PAYMENT"]))) * (((data["inst_AMT_INSTALMENT"]) * (data["NAME_PORTFOLIO_Cards"]))))))) 
    v["i407"] = np.tanh(((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (((data["inst_AMT_PAYMENT"]) * ((((8.0)) * ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) > (np.minimum(((data["NAME_PRODUCT_TYPE_XNA"])), ((data["NAME_PRODUCT_TYPE_XNA"])))))*1.)))))))) * 2.0)) 
    v["i408"] = np.tanh((((0.680000) < (((((((data["inst_AMT_INSTALMENT"]) - (0.416667))) * ((((np.tanh((data["inst_AMT_INSTALMENT"]))) > ((((data["PRODUCT_COMBINATION_Card_X_Sell"]) < (data["te_FLAG_DOCUMENT_19"]))*1.)))*1.)))) / 2.0)))*1.)) 
    v["i409"] = np.tanh(np.minimum(((0.043478)), ((np.where(((((data["te_FLAG_DOCUMENT_19"]) * 2.0)) + (data["PRODUCT_COMBINATION_Card_X_Sell"]))>0, np.tanh((14.800000)), (-1.0*((data["PRODUCT_COMBINATION_Card_X_Sell"]))) ))))) 
    v["i410"] = np.tanh(((((data["te_FLAG_DOCUMENT_20"]) + (np.tanh((data["AMT_DOWN_PAYMENT"]))))) * (((14.800000) * ((((11.714300) < (((((data["AMT_DOWN_PAYMENT"]) + (data["NAME_SELLER_INDUSTRY_XNA"]))) + (data["NAME_SELLER_INDUSTRY_XNA"]))))*1.)))))) 
    v["i411"] = np.tanh((((-1.0*(((((data["NAME_GOODS_CATEGORY_Insurance"]) + (((((((((data["te_FLAG_DOCUMENT_20"]) / 2.0)) + (data["NFLAG_INSURED_ON_APPROVAL"]))/2.0)) + (np.where(data["NAME_CONTRACT_STATUS_Approved"]>0, data["CNT_INSTALMENT"], 0.697674 )))/2.0)))/2.0))))) / 2.0)) 
    v["i412"] = np.tanh(np.where(data["NAME_YIELD_GROUP_high"]>0, (((((0.686567) / 2.0)) < (data["inst_NUM_INSTALMENT_VERSION"]))*1.), (((data["NAME_YIELD_GROUP_low_action"]) > (((((((((data["inst_NUM_INSTALMENT_VERSION"]) / 2.0)) > (data["NAME_YIELD_GROUP_high"]))*1.)) > (data["AMT_APPLICATION"]))*1.)))*1.) )) 
    v["i413"] = np.tanh((((((((((((data["DAYS_REGISTRATION"]) / 2.0)) > (0.680000))*1.)) - (0.062500))) - (((data["DAYS_REGISTRATION"]) * (0.062500))))) - (((data["DAYS_REGISTRATION"]) * (0.062500))))) 
    v["i414"] = np.tanh(np.minimum((((((((data["inst_AMT_INSTALMENT"]) > (0.339286))*1.)) * ((-1.0*((data["PRODUCT_COMBINATION_POS_household_without_interest"]))))))), ((np.maximum(((data["NAME_YIELD_GROUP_low_normal"])), ((np.where(data["NAME_YIELD_GROUP_low_action"]>0, 0.043478, 0.797872 )))))))) 
    v["i415"] = np.tanh((((((data["inst_AMT_PAYMENT"]) > (0.494118))*1.)) * (np.minimum((((-1.0*(((((((((data["inst_AMT_PAYMENT"]) > (0.686567))*1.)) / 2.0)) / 2.0)))))), ((data["NAME_YIELD_GROUP_low_normal"])))))) 
    v["i416"] = np.tanh((((np.maximum(((np.where(((data["NAME_YIELD_GROUP_XNA"]) * (data["AMT_DOWN_PAYMENT"]))>0, data["inst_AMT_INSTALMENT"], (-1.0*((data["inst_AMT_INSTALMENT"]))) ))), ((0.797872)))) < ((((0.797872) < (data["inst_AMT_INSTALMENT"]))*1.)))*1.)) 
    v["i417"] = np.tanh(np.minimum((((-1.0*(((((3.526320) < (data["NAME_YIELD_GROUP_low_action"]))*1.)))))), ((((data["NAME_PRODUCT_TYPE_walk_in"]) * ((-1.0*((np.maximum((((((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))) * (2.0)))), ((data["NAME_YIELD_GROUP_low_action"])))))))))))) 
    v["i418"] = np.tanh(((((data["DAYS_LAST_DUE"]) * (np.where(-1.0>0, data["NAME_CASH_LOAN_PURPOSE_Business_development"], ((((6.0)) < ((((((data["PRODUCT_COMBINATION_Card_Street"]) + (2.235290))/2.0)) * 2.0)))*1.) )))) * 2.0)) 
    v["i419"] = np.tanh(((((data["REG_CITY_NOT_LIVE_CITY"]) * ((((np.maximum(((np.tanh((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))), ((data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"])))) + (((data["te_FLAG_DOCUMENT_19"]) * (data["PRODUCT_COMBINATION_Cash_X_Sell__low"]))))/2.0)))) - ((((3.526320) < (data["PRODUCT_COMBINATION_Cash_X_Sell__low"]))*1.)))) 
    v["i420"] = np.tanh(((np.minimum(((((data["NAME_YIELD_GROUP_high"]) / 2.0))), ((data["CHANNEL_TYPE_Car_dealer"])))) * (((((np.maximum(((np.tanh((data["inst_AMT_PAYMENT"])))), ((data["CHANNEL_TYPE_Car_dealer"])))) * 2.0)) + ((((data["NAME_YIELD_GROUP_high"]) + (data["CHANNEL_TYPE_Car_dealer"]))/2.0)))))) 
    v["i421"] = np.tanh((((((np.tanh((((np.tanh(((((data["AMT_GOODS_PRICE_y"]) > (0.576923))*1.)))) * (data["inst_AMT_INSTALMENT"]))))) > ((((1.857140) > (((data["AMT_ANNUITY"]) * (data["CNT_INSTALMENT_FUTURE"]))))*1.)))*1.)) * 2.0)) 
    v["i422"] = np.tanh((-1.0*(((((1.220590) < ((((-1.0*(((((-1.0*(((-1.0*(((((-1.0*((data["CNT_INSTALMENT"])))) - (((data["FLAG_DOCUMENT_4"]) / 2.0)))))))))) - (0.031746)))))) / 2.0)))*1.))))) 
    v["i423"] = np.tanh((((data["CNT_INSTALMENT_FUTURE"]) < (((np.minimum(((data["inst_AMT_INSTALMENT"])), (((((((data["CNT_INSTALMENT_FUTURE"]) / 2.0)) > ((-1.0*((0.323944)))))*1.))))) - (1.220590))))*1.)) 
    v["i424"] = np.tanh(np.minimum((((((data["CNT_INSTALMENT_FUTURE"]) > (((3.0) + (data["NAME_YIELD_GROUP_middle"]))))*1.))), ((((((data["AMT_CREDIT_y"]) - (np.minimum(((((data["NAME_YIELD_GROUP_middle"]) * 2.0))), ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) / 2.0))))))) / 2.0))))) 
    v["i425"] = np.tanh((((((data["te_FLAG_DOCUMENT_19"]) < (((1.220590) * (((data["inst_AMT_PAYMENT"]) + (-2.0))))))*1.)) * (((((data["inst_AMT_PAYMENT"]) + (-2.0))) - (data["CNT_INSTALMENT"]))))) 
    v["i426"] = np.tanh(((((np.minimum(((0.062500)), ((np.minimum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((data["NAME_SELLER_INDUSTRY_Connectivity"]))))))) * ((((data["inst_AMT_PAYMENT"]) + (np.minimum(((0.062500)), ((data["inst_AMT_PAYMENT"])))))/2.0)))) / 2.0)) 
    v["i427"] = np.tanh((-1.0*((np.where(data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"]>0, ((data["PRODUCT_COMBINATION_Card_X_Sell"]) * 2.0), (((((data["NAME_GOODS_CATEGORY_Photo___Cinema_Equipment"]) > ((-1.0*((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))))*1.)) * 2.0) ))))) 
    v["i428"] = np.tanh((((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]) > ((((np.minimum(((0.031746)), (((((data["NAME_GOODS_CATEGORY_Insurance"]) + (data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]))/2.0))))) < (((data["cc_bal_AMT_PAYMENT_CURRENT"]) - (data["NAME_GOODS_CATEGORY_Insurance"]))))*1.)))*1.)) 
    v["i429"] = np.tanh((((((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) > (0.697674))*1.)) * 2.0)) * (np.minimum((((-1.0*((0.0))))), ((((data["cc_bal_CNT_DRAWINGS_ATM_CURRENT"]) - ((14.39841270446777344))))))))) 
    v["i430"] = np.tanh((((((data["cc_bal_AMT_BALANCE"]) * (data["cc_bal_CNT_DRAWINGS_CURRENT"]))) > ((((data["cc_bal_CNT_DRAWINGS_CURRENT"]) > (((data["cc_bal_AMT_BALANCE"]) * ((((((((data["cc_bal_CNT_DRAWINGS_CURRENT"]) / 2.0)) / 2.0)) > (data["cc_bal_CNT_DRAWINGS_CURRENT"]))*1.)))))*1.)))*1.)) 
    v["i431"] = np.tanh(((np.minimum(((0.323944)), ((np.minimum(((((data["inst_NUM_INSTALMENT_VERSION"]) * (data["inst_NUM_INSTALMENT_VERSION"])))), (((((data["inst_NUM_INSTALMENT_VERSION"]) + (0.686567))/2.0)))))))) + ((((data["cc_bal_MONTHS_BALANCE"]) > (1.105260))*1.)))) 
    v["i432"] = np.tanh(np.minimum(((((data["inst_AMT_PAYMENT"]) * ((-1.0*(((((data["DAYS_FIRST_DRAWING"]) < (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + ((((((data["inst_AMT_PAYMENT"]) * (data["inst_AMT_PAYMENT"]))) > ((11.87245082855224609)))*1.)))))*1.)))))))), ((0.062500)))) 
    v["i433"] = np.tanh((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) - ((-1.0*((0.031746)))))) > ((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) * (data["te_FLAG_DOCUMENT_19"]))) > ((-1.0*((((0.036585) * 2.0))))))*1.)))*1.)) 
    v["i434"] = np.tanh((-1.0*(((((((np.minimum(((data["AMT_APPLICATION"])), ((data["YEARS_BUILD_AVG"])))) * ((((1.0) > (data["cc_bal_AMT_CREDIT_LIMIT_ACTUAL"]))*1.)))) > (((np.minimum(((data["AMT_APPLICATION"])), ((0.043478)))) / 2.0)))*1.))))) 
    v["i435"] = np.tanh(((data["YEARS_BUILD_MEDI"]) * ((((((((-1.0*((0.062500)))) > (((data["te_FLAG_DOCUMENT_19"]) + (data["AMT_APPLICATION"]))))*1.)) + (((data["te_FLAG_DOCUMENT_19"]) * ((((-1.0) + (data["YEARS_BUILD_MEDI"]))/2.0)))))/2.0)))) 
    v["i436"] = np.tanh(((np.where(data["AMT_CREDIT_y"]>0, ((((((0.062500) + ((10.65713787078857422)))/2.0)) + (data["AMT_CREDIT_y"]))/2.0), -3.0 )) * ((((3.526320) < (((data["AMT_ANNUITY"]) - (0.494118))))*1.)))) 
    v["i437"] = np.tanh(((((data["inst_AMT_INSTALMENT"]) * ((((np.where((((data["AMT_APPLICATION"]) > ((7.21875619888305664)))*1.)>0, data["AMT_APPLICATION"], data["CNT_INSTALMENT_FUTURE"] )) > ((6.57621288299560547)))*1.)))) * 2.0)) 
    v["i438"] = np.tanh(((3.526320) * (((data["inst_AMT_INSTALMENT"]) * ((((3.526320) < ((((data["AMT_APPLICATION"]) + (data["AMT_APPLICATION"]))/2.0)))*1.)))))) 
    v["i439"] = np.tanh(np.minimum(((data["CHANNEL_TYPE_Car_dealer"])), (((((((((data["CHANNEL_TYPE_Car_dealer"]) * (data["CHANNEL_TYPE_Car_dealer"]))) * (((0.036585) * (data["NAME_SELLER_INDUSTRY_Connectivity"]))))) + ((-1.0*((data["CHANNEL_TYPE_Car_dealer"])))))/2.0))))) 
    v["i440"] = np.tanh(((((np.minimum(((((np.minimum(((((0.036585) * 2.0))), ((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"])))) * 2.0))), ((data["FLAG_DOCUMENT_4"])))) * (np.tanh((data["DAYS_FIRST_DRAWING"]))))) - (((data["FLAG_DOCUMENT_4"]) / 2.0)))) 
    v["i441"] = np.tanh((-1.0*((((data["DAYS_LAST_DUE_1ST_VERSION"]) * (((data["DAYS_LAST_DUE_1ST_VERSION"]) * ((((data["cc_bal_CNT_INSTALMENT_MATURE_CUM"]) > ((((((((0.576923) < (data["te_FLAG_DOCUMENT_19"]))*1.)) / 2.0)) * 2.0)))*1.))))))))) 
    v["i442"] = np.tanh((((((0.323944) < (data["inst_NUM_INSTALMENT_NUMBER"]))*1.)) * ((((((data["inst_SK_ID_PREV"]) < ((((((0.494118) < (data["inst_NUM_INSTALMENT_NUMBER"]))*1.)) * (0.494118))))*1.)) * 2.0)))) 
    v["i443"] = np.tanh(np.where(data["cc_bal_AMT_DRAWINGS_CURRENT"]>0, 0.0, ((((((data["inst_NUM_INSTALMENT_NUMBER"]) * 2.0)) * ((((0.416667) < (((data["inst_NUM_INSTALMENT_NUMBER"]) - (data["inst_SK_ID_PREV"]))))*1.)))) * 2.0) )) 
    v["i444"] = np.tanh(((np.minimum(((data["inst_AMT_INSTALMENT"])), (((((data["SK_ID_PREV_x"]) + (np.minimum((((-1.0*((np.where(data["inst_AMT_INSTALMENT"]>0, data["inst_NUM_INSTALMENT_NUMBER"], data["cc_bal_CNT_INSTALMENT_MATURE_CUM"] )))))), ((0.036585)))))/2.0))))) / 2.0)) 
    v["i445"] = np.tanh((((((-1.0*((((data["inst_AMT_PAYMENT"]) * 2.0))))) + (np.tanh((0.686567))))) * (((np.minimum(((data["inst_AMT_PAYMENT"])), ((np.minimum(((data["inst_AMT_PAYMENT"])), ((0.062500))))))) * (data["PRODUCT_COMBINATION_Cash_Street__low"]))))) 
    v["i446"] = np.tanh(np.where(data["NAME_CONTRACT_STATUS_Canceled"]>0, np.where(data["te_FLAG_EMAIL"]>0, data["NAME_CASH_LOAN_PURPOSE_Buying_a_new_car"], (((0.494118) < (data["CNT_INSTALMENT_FUTURE"]))*1.) ), np.minimum((((((0.494118) < (data["CNT_INSTALMENT_FUTURE"]))*1.))), ((0.494118))) )) 
    v["i447"] = np.tanh(((np.minimum(((0.680000)), (((-1.0*((np.where((-1.0*((data["NAME_SELLER_INDUSTRY_Connectivity"])))>0, (((data["CNT_PAYMENT"]) > (2.235290))*1.), (((0.043478) < (-2.0))*1.) )))))))) * 2.0)) 
    v["i448"] = np.tanh(np.where((((data["PRODUCT_COMBINATION_Cash_Street__low"]) > (np.tanh(((6.05926561355590820)))))*1.)>0, (((data["AMT_APPLICATION"]) + (data["AMT_CREDIT_y"]))/2.0), np.minimum((((-1.0*((data["PRODUCT_COMBINATION_Cash_Street__low"]))))), ((((data["PRODUCT_COMBINATION_Cash_Street__low"]) * (data["CNT_INSTALMENT"]))))) )) 
    v["i449"] = np.tanh(((np.where((((2.235290) > (data["CNT_INSTALMENT_FUTURE"]))*1.)>0, (((data["CNT_PAYMENT"]) > (3.0))*1.), (((data["CNT_INSTALMENT_FUTURE"]) > (3.0))*1.) )) * 2.0)) 
    v["i450"] = np.tanh(np.minimum(((((data["te_FLAG_DOCUMENT_19"]) * ((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) < (data["PRODUCT_COMBINATION_Cash_Street__low"]))*1.))))), ((((((data["te_FLAG_DOCUMENT_19"]) * (np.maximum(((data["PRODUCT_COMBINATION_Cash_Street__low"])), ((data["te_FLAG_DOCUMENT_19"])))))) * (data["NAME_CASH_LOAN_PURPOSE_XNA"])))))) 
    v["i451"] = np.tanh(((((((np.maximum(((data["cc_bal_AMT_DRAWINGS_CURRENT"])), (((-1.0*((0.043478))))))) * (((1.220590) - (data["NAME_CASH_LOAN_PURPOSE_Education"]))))) + (((data["te_FLAG_DOCUMENT_19"]) / 2.0)))) * (data["NAME_CASH_LOAN_PURPOSE_Education"]))) 
    v["i452"] = np.tanh(np.minimum(((((((((2.0)) + (0.036585))) > ((((data["cc_bal_AMT_BALANCE"]) + (data["te_FLAG_DOCUMENT_19"]))/2.0)))*1.))), (((((data["cc_bal_AMT_BALANCE"]) > ((((2.0)) + (data["NAME_CASH_LOAN_PURPOSE_Business_development"]))))*1.))))) 
    v["i453"] = np.tanh((((0.062500) < (np.where(((data["DAYS_LAST_DUE_1ST_VERSION"]) - (0.323944))>0, ((((data["DAYS_LAST_DUE_1ST_VERSION"]) - (0.339286))) * ((((data["NAME_SELLER_INDUSTRY_Connectivity"]) > (data["DAYS_LAST_DUE_1ST_VERSION"]))*1.))), data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"] )))*1.)) 
    v["i454"] = np.tanh(np.tanh((np.tanh((((data["DAYS_FIRST_DRAWING"]) * (np.tanh(((((((data["DAYS_FIRST_DRAWING"]) < (((((-1.0*((data["DAYS_FIRST_DRAWING"])))) < (data["inst_NUM_INSTALMENT_NUMBER"]))*1.)))*1.)) * (data["inst_NUM_INSTALMENT_NUMBER"]))))))))))) 
    v["i455"] = np.tanh(((((((((data["cc_bal_AMT_DRAWINGS_CURRENT"]) > ((((data["te_FLAG_DOCUMENT_19"]) + (data["inst_AMT_PAYMENT"]))/2.0)))*1.)) < (data["cc_bal_AMT_DRAWINGS_CURRENT"]))*1.)) * (((((data["inst_AMT_PAYMENT"]) * 2.0)) * ((-1.0*((1.857140)))))))) 
    v["i456"] = np.tanh(np.where(data["inst_SK_ID_PREV"]>0, (((np.where(data["inst_SK_ID_PREV"]>0, 3.0, 1.105260 )) < (data["inst_SK_ID_PREV"]))*1.), ((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) * 2.0)) * 2.0)) * 2.0) )) 
    v["i457"] = np.tanh(np.where((((data["ty__Loan_for_working_capital_replenishment"]) < (data["FLAG_DOCUMENT_10"]))*1.)>0, np.where(data["NAME_PORTFOLIO_Cards"]>0, (-1.0*((data["CHANNEL_TYPE_Car_dealer"]))), data["cc_bal_AMT_DRAWINGS_CURRENT"] ), np.maximum(((((data["ty__Loan_for_working_capital_replenishment"]) * 2.0))), ((data["cc_bal_AMT_DRAWINGS_CURRENT"]))) )) 
    v["i458"] = np.tanh(((data["cc_bal_AMT_DRAWINGS_POS_CURRENT"]) * (np.minimum((((-1.0*((((data["cc_bal_CNT_DRAWINGS_CURRENT"]) * (((data["cc_bal_AMT_DRAWINGS_POS_CURRENT"]) * (data["cc_bal_AMT_DRAWINGS_POS_CURRENT"]))))))))), ((data["cc_bal_AMT_DRAWINGS_POS_CURRENT"])))))) 
    v["i459"] = np.tanh(np.maximum(((((data["ty__Loan_for_working_capital_replenishment"]) * (-2.0)))), ((np.maximum(((((data["ty__Loan_for_working_capital_replenishment"]) * (data["SK_ID_PREV_y"])))), (((((data["cc_bal_CNT_DRAWINGS_POS_CURRENT"]) + (((0.036585) + (-2.0))))/2.0)))))))) 
    v["i460"] = np.tanh((((((((((data["cc_bal_AMT_DRAWINGS_CURRENT"]) < (data["cc_bal_AMT_PAYMENT_CURRENT"]))*1.)) * 2.0)) * 2.0)) * (np.minimum(((data["cc_bal_AMT_PAYMENT_CURRENT"])), ((((0.043478) - (np.minimum(((data["te_FLAG_DOCUMENT_19"])), ((data["cc_bal_AMT_DRAWINGS_CURRENT"]))))))))))) 
    v["i461"] = np.tanh(((((((((((data["te_FLAG_DOCUMENT_19"]) * ((((data["cc_bal_AMT_PAYMENT_CURRENT"]) > (np.maximum(((((data["te_FLAG_DOCUMENT_19"]) + (2.0)))), ((0.697674)))))*1.)))) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i462"] = np.tanh((((-1.0*((np.where(data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]>0, (((data["cc_bal_AMT_DRAWINGS_ATM_CURRENT"]) > ((1.95135760307312012)))*1.), (((-1.0*((((((1.95135760307312012)) < (data["cc_bal_AMT_DRAWINGS_CURRENT"]))*1.))))) / 2.0) ))))) * 2.0)) 
    v["i463"] = np.tanh(((((np.where(np.minimum(((data["cc_bal_AMT_PAYMENT_CURRENT"])), ((data["ty__Loan_for_working_capital_replenishment"])))>0, 1.857140, data["FLAG_DOCUMENT_10"] )) * 2.0)) * ((((data["NAME_YIELD_GROUP_high"]) < ((-1.0*((0.576923)))))*1.)))) 
    v["i464"] = np.tanh(((data["te_FONDKAPREMONT_MODE"]) * (((((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + (np.where(data["cc_bal_AMT_DRAWINGS_CURRENT"]>0, data["cc_bal_AMT_DRAWINGS_CURRENT"], data["ty__Loan_for_working_capital_replenishment"] )))) + (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * 2.0)))))) 
    v["i465"] = np.tanh((((-1.0*((((((((data["ty__Loan_for_working_capital_replenishment"]) + (data["cc_bal_AMT_DRAWINGS_CURRENT"]))/2.0)) + (((data["ty__Loan_for_working_capital_replenishment"]) * ((-1.0*(((((data["cc_bal_AMT_DRAWINGS_CURRENT"]) + (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * 2.0)))/2.0))))))))/2.0))))) / 2.0)) 
    v["i466"] = np.tanh(((((((((-1.0*((((data["ty__Real_estate_loan"]) - ((-1.0*((data["cc_bal_AMT_PAYMENT_CURRENT"]))))))))) + (data["te_FLAG_DOCUMENT_19"]))) > ((((data["cc_bal_AMT_PAYMENT_CURRENT"]) < (data["te_FLAG_DOCUMENT_19"]))*1.)))*1.)) * 2.0)) 
    v["i467"] = np.tanh(((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) * 2.0)) * (((2.0) * 2.0)))) * ((((((((-1.0*((0.576923)))) + (data["cc_bal_AMT_TOTAL_RECEIVABLE"]))/2.0)) > (data["cc_bal_AMT_TOTAL_RECEIVABLE"]))*1.)))) 
    v["i468"] = np.tanh(((data["ty__Loan_for_working_capital_replenishment"]) + (np.where(((data["cc_bal_SK_DPD_DEF"]) + (data["NAME_GOODS_CATEGORY_Insurance"]))>0, data["cc_bal_AMT_DRAWINGS_CURRENT"], ((((((data["cc_bal_AMT_DRAWINGS_CURRENT"]) * 2.0)) * (data["ty__Loan_for_working_capital_replenishment"]))) * 2.0) )))) 
    v["i469"] = np.tanh((((((3.526320) < (((np.maximum(((((3.526320) / 2.0))), ((data["cc_bal_AMT_INST_MIN_REGULARITY"])))) - (((data["te_FLAG_DOCUMENT_19"]) * ((-1.0*((3.526320)))))))))*1.)) * (((data["cc_bal_AMT_INST_MIN_REGULARITY"]) / 2.0)))) 
    v["i470"] = np.tanh((-1.0*((np.where(data["ty__Loan_for_working_capital_replenishment"]>0, ((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) - ((((data["SK_ID_PREV_y"]) < (np.tanh((-3.0))))*1.))), data["NAME_CASH_LOAN_PURPOSE_Business_development"] ))))) 
    v["i471"] = np.tanh(((np.maximum(((data["cc_bal_SK_DPD_DEF"])), ((data["NAME_CASH_LOAN_PURPOSE_Business_development"])))) * ((((((14.800000) + ((((14.800000) + (((((-1.0*((data["cc_bal_SK_DPD_DEF"])))) + (data["te_FLAG_DOCUMENT_19"]))/2.0)))/2.0)))/2.0)) * (data["cc_bal_AMT_DRAWINGS_CURRENT"]))))) 
    v["i472"] = np.tanh((((((data["cc_bal_AMT_PAYMENT_CURRENT"]) > (((2.235290) - (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) * (((((data["cc_bal_AMT_PAYMENT_CURRENT"]) - (np.minimum(((0.494118)), ((data["cc_bal_AMT_PAYMENT_CURRENT"])))))) * 2.0)))))))*1.)) * 2.0)) 
    v["i473"] = np.tanh(((((((((((np.where(data["ty__Real_estate_loan"]>0, ((data["NAME_GOODS_CATEGORY_Insurance"]) * 2.0), ((data["NAME_GOODS_CATEGORY_Insurance"]) * (data["ty__Loan_for_working_capital_replenishment"])) )) / 2.0)) * 2.0)) * 2.0)) * 2.0)) * 2.0)) 
    v["i474"] = np.tanh(((data["ty__Loan_for_working_capital_replenishment"]) * ((((np.maximum(((np.tanh((data["ty__Loan_for_working_capital_replenishment"])))), ((data["FLAG_DOCUMENT_4"])))) + (((((data["NAME_CASH_LOAN_PURPOSE_Purchase_of_electronic_equipment"]) * (data["ty__Loan_for_working_capital_replenishment"]))) - (data["FLAG_DOCUMENT_10"]))))/2.0)))) 
    v["i475"] = np.tanh((((-1.0*(((((((14.800000) * 2.0)) < (((data["NAME_CASH_LOAN_PURPOSE_Purchase_of_electronic_equipment"]) + ((((((1.98723483085632324)) * (data["ty__Loan_for_working_capital_replenishment"]))) + (((data["FLAG_DOCUMENT_10"]) + (data["ty__Loan_for_working_capital_replenishment"]))))))))*1.))))) * 2.0)) 
    v["i476"] = np.tanh(np.where((((data["ty__Loan_for_working_capital_replenishment"]) > ((10.0)))*1.)>0, (-1.0*((np.maximum(((data["ty__Loan_for_working_capital_replenishment"])), ((np.maximum(((data["ty__Loan_for_working_capital_replenishment"])), ((data["ty__Loan_for_working_capital_replenishment"]))))))))), np.maximum(((data["ty__Loan_for_working_capital_replenishment"])), (((-1.0*((data["ty__Loan_for_working_capital_replenishment"])))))) )) 
    v["i477"] = np.tanh(np.minimum(((((((((data["avg_buro_buro_bal_status_1"]) * (data["EXT_SOURCE_3"]))) * (data["EXT_SOURCE_3"]))) * (data["EXT_SOURCE_3"])))), (((((((data["avg_buro_buro_bal_status_1"]) * (data["EXT_SOURCE_3"]))) > (np.tanh((0.680000))))*1.))))) 
    v["i478"] = np.tanh(((data["te_FLAG_DOCUMENT_19"]) * (np.where((((data["avg_buro_buro_bal_status_3"]) > (((data["te_FLAG_DOCUMENT_19"]) + (1.220590))))*1.)>0, 14.800000, (-1.0*((((data["avg_buro_buro_bal_status_3"]) * (data["te_FLAG_DOCUMENT_19"]))))) )))) 
    v["i479"] = np.tanh((-1.0*((((((data["FLAG_DOCUMENT_19"]) * 2.0)) * ((((data["FLAG_DOCUMENT_3"]) + ((((((data["FLAG_DOCUMENT_3"]) + (((data["te_FLAG_DOCUMENT_19"]) + (((data["FLAG_DOCUMENT_19"]) / 2.0)))))/2.0)) / 2.0)))/2.0))))))) 
    v["i480"] = np.tanh(((((data["te_FLAG_DOCUMENT_19"]) + (((data["CODE_REJECT_REASON_SYSTEM"]) * (data["NAME_TYPE_SUITE_Other_B"]))))) * (((data["NAME_TYPE_SUITE_Other_B"]) - (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) - (data["CODE_REJECT_REASON_SYSTEM"]))))))) 
    v["i481"] = np.tanh(((((((data["te_FLAG_DOCUMENT_17"]) - (data["te_FLAG_DOCUMENT_10"]))) * (11.714300))) * (np.where(((data["te_FLAG_DOCUMENT_17"]) - (data["te_FLAG_DOCUMENT_10"]))>0, ((data["te_FLAG_DOCUMENT_10"]) * 2.0), data["te_SK_ID_CURR"] )))) 
    v["i482"] = np.tanh((((((data["cc_bal_cc_bal_status__Refused"]) * 2.0)) > ((((data["SK_ID_PREV_y"]) > ((-1.0*((np.maximum(((0.043478)), (((((2.235290) > ((-1.0*((14.800000)))))*1.)))))))))*1.)))*1.)) 
    v["i483"] = np.tanh(((np.maximum(((np.tanh((np.maximum(((data["NAME_GOODS_CATEGORY_Insurance"])), ((((data["inst_AMT_PAYMENT"]) * (data["Active"]))))))))), ((0.0)))) * (((data["Completed"]) * (data["Active"]))))) 
    v["i484"] = np.tanh(np.minimum(((np.tanh(((((0.043478) > (data["inst_AMT_PAYMENT"]))*1.))))), ((((np.where(data["inst_AMT_PAYMENT"]>0, data["te_FLAG_DOCUMENT_19"], (((-1.0*(((((0.043478) + (data["inst_AMT_PAYMENT"]))/2.0))))) * 2.0) )) / 2.0))))) 
    v["i485"] = np.tanh(np.minimum(((((np.tanh((data["inst_AMT_INSTALMENT"]))) + (0.323944)))), ((((data["inst_AMT_INSTALMENT"]) * (np.where((((data["AMT_GOODS_PRICE_y"]) + (0.031746))/2.0)>0, 0.043478, data["inst_AMT_INSTALMENT"] ))))))) 
    v["i486"] = np.tanh((((((((((0.036585) * (((((data["DAYS_FIRST_DRAWING"]) * (((((data["te_REG_REGION_NOT_LIVE_REGION"]) / 2.0)) / 2.0)))) * 2.0)))) > ((((data["inst_AMT_PAYMENT"]) + (data["DAYS_FIRST_DRAWING"]))/2.0)))*1.)) / 2.0)) / 2.0)) 
    v["i487"] = np.tanh(np.minimum((((((0.043478) + (np.tanh(((-1.0*(((((data["EXT_SOURCE_3"]) < ((-1.0*((np.where(data["NAME_YIELD_GROUP_high"]>0, data["AMT_GOODS_PRICE_y"], 0.686567 ))))))*1.))))))))/2.0))), ((0.043478)))) 
    v["i488"] = np.tanh((((((((((-2.0) > (np.minimum((((((data["te_FLAG_DOCUMENT_20"]) < (0.036585))*1.))), ((data["DAYS_CREDIT"])))))*1.)) * 2.0)) * (np.maximum(((data["inst_AMT_PAYMENT"])), ((data["te_FLAG_DOCUMENT_20"])))))) * 2.0)) 
    v["i489"] = np.tanh(((((((data["DAYS_CREDIT"]) - (((data["DAYS_CREDIT_UPDATE"]) * (data["DAYS_CREDIT_UPDATE"]))))) / 2.0)) * ((((((data["DAYS_CREDIT"]) + (((data["DAYS_CREDIT_UPDATE"]) * (np.tanh((data["DAYS_CREDIT"]))))))/2.0)) / 2.0)))) 
    v["i490"] = np.tanh(((((data["DAYS_CREDIT"]) * (data["NAME_SELLER_INDUSTRY_Connectivity"]))) * (((((data["te_FLAG_CONT_MOBILE"]) * ((((0.494118) < (data["te_FLAG_CONT_MOBILE"]))*1.)))) - (data["te_FLAG_CONT_MOBILE"]))))) 
    v["i491"] = np.tanh(((((data["DAYS_CREDIT"]) * (np.maximum(((((0.062500) * (((data["inst_AMT_PAYMENT"]) - (((data["DAYS_CREDIT"]) * (((data["DAYS_CREDIT"]) * (data["inst_AMT_PAYMENT"])))))))))), ((0.062500)))))) * 2.0)) 
    v["i492"] = np.tanh(((0.062500) * (np.where(data["inst_AMT_PAYMENT"]>0, ((np.minimum((((-1.0*((data["inst_AMT_PAYMENT"]))))), ((data["te_FLAG_DOCUMENT_20"])))) * 2.0), (-1.0*((((data["te_FLAG_CONT_MOBILE"]) - (data["inst_AMT_PAYMENT"]))))) )))) 
    v["i493"] = np.tanh(np.where(((0.797872) - (data["DAYS_CREDIT_UPDATE"]))>0, ((((((data["AMT_CREDIT_x"]) / 2.0)) * (data["AMT_CREDIT_x"]))) * (data["inst_AMT_INSTALMENT"])), ((0.0) - (((data["AMT_CREDIT_x"]) / 2.0))) )) 
    v["i494"] = np.tanh(((((0.339286) * ((((((((data["DAYS_DECISION"]) / 2.0)) > (data["AMT_CREDIT_x"]))*1.)) - (data["DAYS_DECISION"]))))) * (data["AMT_CREDIT_x"]))) 
    v["i495"] = np.tanh((((((((0.062500) > (data["DAYS_CREDIT_UPDATE"]))*1.)) * (((np.maximum(((data["NAME_SELLER_INDUSTRY_Connectivity"])), (((((0.036585) > (data["inst_AMT_PAYMENT"]))*1.))))) / 2.0)))) * (((1.220590) / 2.0)))) 
    v["i496"] = np.tanh((((-1.0*((data["te_FLAG_DOCUMENT_11"])))) * (np.where(data["NAME_GOODS_CATEGORY_Insurance"]>0, 0.494118, np.minimum(((data["te_FLAG_DOCUMENT_11"])), ((11.714300))) )))) 
    v["i497"] = np.tanh(np.where(data["PRODUCT_COMBINATION_POS_mobile_without_interest"]>0, np.minimum(((np.minimum(((data["inst_AMT_PAYMENT"])), ((data["PRODUCT_COMBINATION_POS_mobile_without_interest"]))))), ((0.043478))), np.minimum(((0.043478)), ((np.maximum(((data["PRODUCT_COMBINATION_Cash"])), (((-1.0*((data["inst_AMT_PAYMENT"]))))))))) )) 
    v["i498"] = np.tanh(((np.where(np.where(data["AMT_GOODS_PRICE_y"]>0, data["te_FLAG_DOCUMENT_20"], 0.036585 )>0, 0.036585, (((np.tanh((data["te_FLAG_OWN_CAR"]))) + (data["AMT_GOODS_PRICE_y"]))/2.0) )) * (data["AMT_GOODS_PRICE_y"]))) 
    v["i499"] = np.tanh(((11.714300) * ((((((data["cc_bal_AMT_DRAWINGS_CURRENT"]) * (np.minimum(((data["CNT_INSTALMENT"])), (((((1.0) < (np.minimum(((data["CNT_INSTALMENT"])), ((data["cc_bal_AMT_DRAWINGS_CURRENT"])))))*1.))))))) > (0.494118))*1.)))) 
    v["i500"] = np.tanh((((-1.0*((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"])))) * ((((3.0) < ((((data["cc_bal_AMT_PAYMENT_TOTAL_CURRENT"]) + ((((((2.0) < (data["cc_bal_AMT_DRAWINGS_CURRENT"]))*1.)) * 2.0)))/2.0)))*1.)))) 
    v["i501"] = np.tanh(((((((data["te_FLAG_OWN_CAR"]) / 2.0)) / 2.0)) * (((((((data["NAME_CONTRACT_STATUS_Canceled"]) / 2.0)) * (data["NAME_GOODS_CATEGORY_Insurance"]))) - (((data["NAME_CONTRACT_STATUS_Canceled"]) / 2.0)))))) 
    v["i502"] = np.tanh(((((data["inst_AMT_PAYMENT"]) * (data["NAME_CONTRACT_STATUS_Canceled"]))) * (np.minimum((((4.64522552490234375))), (((-1.0*((data["cc_bal_AMT_INST_MIN_REGULARITY"]))))))))) 
    v["i503"] = np.tanh((-1.0*((((np.maximum((((((-1.0*((data["CNT_INSTALMENT"])))) * (0.339286)))), ((((((6.0)) < (data["inst_AMT_PAYMENT"]))*1.))))) * (np.maximum(((data["NAME_CONTRACT_STATUS_Canceled"])), ((data["CNT_INSTALMENT"]))))))))) 
    v["i504"] = np.tanh(np.minimum((((((0.043478) + (((3.526320) - (data["PRODUCT_COMBINATION_Cash"]))))/2.0))), ((((((0.043478) * (((data["PRODUCT_COMBINATION_Cash"]) - (data["te_FLAG_OWN_CAR"]))))) * 2.0))))) 
    v["i505"] = np.tanh(np.minimum(((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) * (data["DAYS_LAST_DUE_1ST_VERSION"])))), ((((((3.526320) * (data["DAYS_LAST_DUE_1ST_VERSION"]))) * ((((data["NAME_GOODS_CATEGORY_Direct_Sales"]) + (np.tanh(((((2.235290) < (data["CNT_INSTALMENT"]))*1.)))))/2.0))))))) 
    v["i506"] = np.tanh(np.minimum(((np.maximum(((((((-1.0*((data["ca__Closed"])))) < (data["ca__Closed"]))*1.))), ((data["CNT_INSTALMENT_FUTURE"]))))), (((((data["ty__Credit_card"]) > (np.maximum(((data["CNT_INSTALMENT_FUTURE"])), ((0.416667)))))*1.))))) 
    v["i507"] = np.tanh((((((data["te_FLAG_DOCUMENT_19"]) * (data["NAME_GOODS_CATEGORY_Direct_Sales"]))) + (np.where(data["te_FLAG_DOCUMENT_19"]>0, ((data["te_FLAG_DOCUMENT_19"]) * (data["NAME_GOODS_CATEGORY_Insurance"])), data["cc_bal_AMT_INST_MIN_REGULARITY"] )))/2.0)) 
    v["i508"] = np.tanh((((((data["cc_bal_AMT_INST_MIN_REGULARITY"]) > (np.maximum(((data["cc_bal_AMT_DRAWINGS_CURRENT"])), ((0.494118)))))*1.)) * ((-1.0*(((((np.maximum(((data["cc_bal_AMT_RECEIVABLE_PRINCIPAL"])), ((0.494118)))) > (((data["cc_bal_AMT_TOTAL_RECEIVABLE"]) - (0.0))))*1.))))))) 
    v["i509"] = np.tanh((((-1.0*((data["NAME_GOODS_CATEGORY_Direct_Sales"])))) * ((((-1.0*((data["ty__Real_estate_loan"])))) - ((((data["te_FLAG_DOCUMENT_19"]) + (((data["NAME_CASH_LOAN_PURPOSE_Business_development"]) + (((data["ty__Interbank_credit"]) + (data["FLAG_DOCUMENT_10"]))))))/2.0)))))) 
    v["i510"] = np.tanh(np.where(data["CNT_CREDIT_PROLONG"]>0, data["te_FLAG_DOCUMENT_19"], np.minimum(((((np.maximum(((np.minimum((((-1.0*((data["te_FLAG_DOCUMENT_19"]))))), ((0.323944))))), ((data["CNT_CREDIT_PROLONG"])))) * (data["AMT_REQ_CREDIT_BUREAU_MON"])))), ((((1.0) * 2.0)))) )) 
    v["i511"] = np.tanh((-1.0*((np.maximum((((((data["CNT_CREDIT_PROLONG"]) > ((((9.95406532287597656)) + (((data["FLAG_DOCUMENT_15"]) - (np.maximum(((data["FLAG_DOCUMENT_15"])), ((data["NAME_GOODS_CATEGORY_Education"])))))))))*1.))), ((((data["FLAG_DOCUMENT_15"]) - (data["CNT_CREDIT_PROLONG"])))))))))
    return v
gc.enable()

buro_bal = pd.read_csv('../input/bureau_balance.csv')
print('Buro bal shape : ', buro_bal.shape)

print('transform to dummies')
buro_bal = pd.concat([buro_bal, pd.get_dummies(buro_bal.STATUS, prefix='buro_bal_status')], axis=1).drop('STATUS', axis=1)

print('Counting buros')
buro_counts = buro_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
buro_bal['buro_count'] = buro_bal['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])

print('averaging buro bal')
avg_buro_bal = buro_bal.groupby('SK_ID_BUREAU').mean()

avg_buro_bal.columns = ['avg_buro_' + f_ for f_ in avg_buro_bal.columns]
del buro_bal
gc.collect()

print('Read Bureau')
buro = pd.read_csv('../input/bureau.csv')

print('Go to dummies')
buro_credit_active_dum = pd.get_dummies(buro.CREDIT_ACTIVE, prefix='ca_')
buro_credit_currency_dum = pd.get_dummies(buro.CREDIT_CURRENCY, prefix='cu_')
buro_credit_type_dum = pd.get_dummies(buro.CREDIT_TYPE, prefix='ty_')

buro_full = pd.concat([buro, buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum], axis=1)
# buro_full.columns = ['buro_' + f_ for f_ in buro_full.columns]

del buro_credit_active_dum, buro_credit_currency_dum, buro_credit_type_dum
gc.collect()

print('Merge with buro avg')
buro_full = buro_full.merge(right=avg_buro_bal.reset_index(), how='left', on='SK_ID_BUREAU', suffixes=('', '_bur_bal'))

print('Counting buro per SK_ID_CURR')
nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
buro_full['SK_ID_BUREAU'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

print('Averaging bureau')
avg_buro = buro_full.groupby('SK_ID_CURR').mean()
print(avg_buro.head())

del buro, buro_full
gc.collect()

print('Read prev')
prev = pd.read_csv('../input/previous_application.csv')

prev_cat_features = [
    f_ for f_ in prev.columns if prev[f_].dtype == 'object'
]

print('Go to dummies')
prev_dum = pd.DataFrame()
for f_ in prev_cat_features:
    prev_dum = pd.concat([prev_dum, pd.get_dummies(prev[f_], prefix=f_).astype(np.uint8)], axis=1)

prev = pd.concat([prev, prev_dum], axis=1)

del prev_dum
gc.collect()

print('Counting number of Prevs')
nb_prev_per_curr = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])

print('Averaging prev')
avg_prev = prev.groupby('SK_ID_CURR').mean()
print(avg_prev.head())
del prev
gc.collect()

print('Reading POS_CASH')
pos = pd.read_csv('../input/POS_CASH_balance.csv')

print('Go to dummies')
pos = pd.concat([pos, pd.get_dummies(pos['NAME_CONTRACT_STATUS'])], axis=1)

print('Compute nb of prevs per curr')
nb_prevs = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
pos['SK_ID_PREV'] = pos['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Go to averages')
avg_pos = pos.groupby('SK_ID_CURR').mean()

del pos, nb_prevs
gc.collect()

print('Reading CC balance')
cc_bal = pd.read_csv('../input/credit_card_balance.csv')

print('Go to dummies')
cc_bal = pd.concat([cc_bal, pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_status_')], axis=1)

nb_prevs = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
cc_bal['SK_ID_PREV'] = cc_bal['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

print('Compute average')
avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]

del cc_bal, nb_prevs
gc.collect()

print('Reading Installments')
inst = pd.read_csv('../input/installments_payments.csv')
nb_prevs = inst[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
inst['SK_ID_PREV'] = inst['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])

avg_inst = inst.groupby('SK_ID_CURR').mean()
avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]

print('Read data and test')
data = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
print('Shapes : ', data.shape, test.shape)


print('Read data and test')
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
print('Shapes : ', train.shape, train.shape)

categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]
categorical_feats
for f_ in categorical_feats:
    train[f_], indexer = pd.factorize(train[f_])
    test[f_] = indexer.get_indexer(test[f_])
    
train = train.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

del avg_buro, avg_prev
gc.collect()

ID = test.SK_ID_CURR
train.columns = train.columns.str.replace('[^A-Za-z0-9_]', '_')
test.columns = test.columns.str.replace('[^A-Za-z0-9_]', '_')
floattypes = []
inttypes = []
stringtypes = []
for c in test.columns:
    if(train[c].dtype=='object'):
        train[c] = train[c].astype('str')
        stringtypes.append(c)
    elif(train[c].dtype=='int64'):
        train[c] = train[c].astype('int32')
        inttypes.append(c)
    else:
        train[c] = train[c].astype('float32')
        floattypes.append(c)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in stringtypes:
    train['te_'+col] = 0.
    test['te_'+col] = 0.
    SMOOTHING = test[~test[col].isin(train[col])].shape[0]/test.shape[0]

    for f, (vis_index, blind_index) in enumerate(kf.split(train)):
        _, train.loc[blind_index, 'te_'+col] = target_encode(train.loc[vis_index, col], 
                                                            train.loc[blind_index, col], 
                                                            target=train.loc[vis_index,'TARGET'], 
                                                            min_samples_leaf=100,
                                                            smoothing=SMOOTHING,
                                                            noise_level=0.0)
        _, x = target_encode(train.loc[vis_index, col], 
                                          test[col], 
                                          target=train.loc[vis_index,'TARGET'], 
                                          min_samples_leaf=100,
                                          smoothing=SMOOTHING,
                                          noise_level=0.0)
        test['te_'+col] += (.2*x)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in inttypes:
    train['te_'+col] = 0.
    test['te_'+col] = 0.
    SMOOTHING = test[~test[col].isin(train[col])].shape[0]/test.shape[0]
    for f, (vis_index, blind_index) in enumerate(kf.split(train)):
        _, train.loc[blind_index, 'te_'+col] = target_encode(train.loc[vis_index, col], 
                                                            train.loc[blind_index, col], 
                                                            target=train.loc[vis_index,'TARGET'], 
                                                            min_samples_leaf=100,
                                                            smoothing=SMOOTHING,
                                                            noise_level=0.0)
        _, x = target_encode(train.loc[vis_index, col], 
                                              test[col], 
                                              target=train.loc[vis_index,'TARGET'], 
                                              min_samples_leaf=100,
                                              smoothing=SMOOTHING,
                                              noise_level=0.0)
        test['te_'+col] += (.2*x)
ntrainrows = train.shape[0]
test.insert(1,'TARGET',-1)
alldata = pd.concat([train,test])
del train ,test
gc.collect()
alldata['nans'] = alldata.isnull().sum(axis=1)
for col in inttypes[1:]:
    x = alldata[col].value_counts().reset_index(drop=False)
    x.columns = [col,'cnt_'+col]
    x['cnt_'+col]/=alldata.shape[0]
    alldata = alldata.merge(x,on=col,how='left')
features = list(set(alldata.columns).difference(['SK_ID_CURR','TARGET']))
alldata[features] = alldata[features].astype('float32')
for c in features:
    ss = StandardScaler()
    alldata.loc[~alldata[c].isnull(),c] = ss.fit_transform(alldata.loc[~alldata[c].isnull(),c].values.reshape(-1,1))
    alldata[c].fillna(alldata[c].mean(),inplace=True)
train = alldata[:ntrainrows]
test = alldata[ntrainrows:]
traintargets = train.TARGET.values
train = UseGPFeatures(train)
test = UseGPFeatures(test)
train['TARGET'] = traintargets
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
import gc

gc.enable()
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])
feats = [f for f in train.columns if f not in ['SK_ID_CURR','TARGET']]
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
    trn_x, trn_y = train[feats].iloc[trn_idx], train.iloc[trn_idx]['TARGET']
    val_x, val_y = train[feats].iloc[val_idx], train.iloc[val_idx]['TARGET']
    
    clf = LGBMClassifier(
        n_estimators=4000,
        learning_rate=0.03,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=100,
        #scale_pos_weight=12.5,
        silent=-1,
        verbose=-1,
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
Submission = pd.DataFrame({ 'SK_ID_CURR': ID,'TARGET': sub_preds })
Submission.to_csv("hybridII.csv", index=False)
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.utils import shuffle

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from pandas.api.types import CategoricalDtype



from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



import gc

import os

from functools import partial



print(f"Pandas version: {pd.__version__}")
INPUT_DIR = "../input/"



get_full_path = partial(os.path.join, INPUT_DIR)



PATH_APP_TRAIN = get_full_path("application_train.csv")

PATH_APP_TEST = get_full_path("application_test.csv")

PATH_PRE_APP = get_full_path("previous_application.csv")

PATH_INST_PAY = get_full_path("installments_payments.csv")

PATH_INST_PAY_PROCESSED = "installments_payments_preprocessed.csv"

PATH_POS_BALANCE = get_full_path("POS_CASH_balance")

PATH_CREDIT_BALANCE = get_full_path("credit_card_balance.csv")

PATH_BUREAU = get_full_path("bureau.csv")

PATH_BUREAU_BALANCE = get_full_path("bureau_balance.csv")



PERFORM_IMPUTATION = True

CROSS_VALIDATION_FOLD = 5

RANDOM_SEED = 0
LoanType = CategoricalDtype(["Cash loans", "Revolving loans", "Consumer loans"], False)

HouseType = CategoricalDtype(["block of flats", "terraced house", "specific housing"])

WeekDayType = CategoricalDtype(['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY'], True)

YesNoType = CategoricalDtype(["N", "Y"], True)

LoanStatusType = CategoricalDtype(['Approved', 'Refused', 'Canceled', 'Unused offer'], False)

EducationType = CategoricalDtype(["Incomplete higher", "Higher education", "Lower secondary",

                                  "Secondary / secondary special", "Academic degree"], True)

FamilyType = CategoricalDtype(["Civil marriage", "Married", "Separated", "Single / not married", "Widow"], False)

HousingType = CategoricalDtype(["House / apartment", "Rented apartment", "With parents",

                                "Municipal apartment", "Office apartment", "Co-op apartment"], False)

IncomeType = CategoricalDtype(["Unemployed", "Student", "State servant", "Working", "Commercial associate",

                               "Businessman", "Maternity leave", "Pensioner"], False)

AccompanyType = CategoricalDtype(["Unaccompanied", "Spouse", "partner", "Family", "Children",

                                  "Group of people", "Other_A", "Other_B"], False)

OccupationType = CategoricalDtype(["Accountants", "Cleaning staff", "Cooking staff", "Core staff",

                                   "Drivers", "HR staff", "High skill tech staff", "IT staff", "Laborers",

                                   "Low-skill Laborers", "Managers", "Medicine staff",

                                   "Private service staff", "Realty agents", "Sales staff", "Secretaries",

                                   "Security staff", "Waiters/barmen staff"], False)

CreditStatusType = CategoricalDtype(["Closed", "Active", "Sold", "Bad debt", "Signed"], True)

BureauBalanceStatusType = CategoricalDtype(["C", *map(str, range(6))], True)
def load_app_data(train_only=False):

    gender_type = CategoricalDtype(["M", "F"], False)

    yes_no_type2 = CategoricalDtype(["No", "Yes"], True)

    wall_material_type = CategoricalDtype(["Stone, brick", "Wooden", "Block", "Panel", 

                                           "Monolithic", "Mixed", "Others"], False)

    fondkapremont_type = CategoricalDtype(["reg oper account", "org spec account",

                                           "reg oper spec account", "not specified"], False)



    col_types = {

        "SK_ID_CURR": np.uint32, "TARGET": np.bool, "CODE_GENDER": gender_type, "NAME_CONTRACT_TYPE": LoanType,

        "FLAG_OWN_CAR": YesNoType, "FLAG_OWN_REALTY": YesNoType, "CNT_CHILDREN": np.uint8,

        "AMT_INCOME_TOTAL": np.float32, "AMT_CREDIT": np.float32,

        "AMT_ANNUITY": np.float32, "AMT_GOODS_PRICE": np.float32,

        "NAME_TYPE_SUITE": AccompanyType, "NAME_EDUCATION_TYPE": EducationType, "NAME_INCOME_TYPE": IncomeType,

        "NAME_FAMILY_STATUS": FamilyType, "NAME_HOUSING_TYPE": HousingType,

        "REGION_POPULATION_RELATIVE": np.float32,

        "REGION_RATING_CLIENT": np.uint8, "REGION_RATING_CLIENT_W_CITY": np.uint8,

        "WEEKDAY_APPR_PROCESS_START": WeekDayType, "HOUR_APPR_PROCESS_START": np.uint8,

        "DAYS_EMPLOYED": np.float32, "DAYS_BIRTH": np.int32,

        "DAYS_REGISTRATION": np.float32, "DAYS_ID_PUBLISH": np.float32,

        "OWN_CAR_AGE": np.float16,

        "EXT_SOURCE_1": np.float32, "EXT_SOURCE_2": np.float32, "EXT_SOURCE_3": np.float32,

        "FLAG_MOBIL": np.bool, "FLAG_EMP_PHONE": np.bool, "FLAG_WORK_PHONE": np.bool,

        "FLAG_CONT_MOBILE": np.bool, "FLAG_PHONE": np.bool, "FLAG_EMAIL": np.bool,

        "OCCUPATION_TYPE": OccupationType, "CNT_FAM_MEMBERS": np.float16,

        "REG_REGION_NOT_LIVE_REGION": np.bool, "REG_REGION_NOT_WORK_REGION": np.bool,

        "LIVE_REGION_NOT_WORK_REGION": np.bool, "REG_CITY_NOT_LIVE_CITY": np.bool,

        "REG_CITY_NOT_WORK_CITY": np.bool, "LIVE_CITY_NOT_WORK_CITY": np.bool,

        "ORGANIZATION_TYPE": "category",

        "OBS_30_CNT_SOCIAL_CIRCLE": np.float16, "DEF_30_CNT_SOCIAL_CIRCLE": np.float16,

        "OBS_60_CNT_SOCIAL_CIRCLE": np.float16, "DEF_60_CNT_SOCIAL_CIRCLE": np.float16,

        "DAYS_LAST_PHONE_CHANGE": np.float32,

        "AMT_REQ_CREDIT_BUREAU_HOUR": np.float16, "AMT_REQ_CREDIT_BUREAU_DAY": np.float16,

        "AMT_REQ_CREDIT_BUREAU_WEEK": np.float16, "AMT_REQ_CREDIT_BUREAU_MON": np.float16,

        "AMT_REQ_CREDIT_BUREAU_QRT": np.float16, "AMT_REQ_CREDIT_BUREAU_YEAR": np.float16,

        "HOUSETYPE_MODE": HouseType, "FONDKAPREMONT_MODE": fondkapremont_type,

        "WALLSMATERIAL_MODE": wall_material_type, "EMERGENCYSTATE_MODE": yes_no_type2

    }



    float_housing_columns = [

        "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",

        "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",

        "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG",

        "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",

        "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",

        "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE",

        "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE",

        "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE",

        "NONLIVINGAREA_MODE", "TOTALAREA_MODE",

        "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",

        "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI",

        "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",

        "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",

        # 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',

    ]



    col_types.update(("FLAG_DOCUMENT_" + str(v), np.bool) for v in range(2, 22))

    col_types.update((col, np.float16) for col in float_housing_columns)

    

    replace_dict = {"DAYS_EMPLOYED": {365243: np.nan}, "DAYS_LAST_PHONE_CHANGE": {0: np.nan}}

    na_values = ["XNA", "Unknown"]

    df_app_train = pd.read_csv(PATH_APP_TRAIN, index_col=0, na_values=na_values, dtype=col_types)

    df_app_train.replace(replace_dict, inplace=True)

    assert df_app_train.shape == (307511, 121)

    if train_only:

        return df_app_train

    else:

        df_app_test = pd.read_csv(PATH_APP_TEST, index_col=0, na_values=na_values, dtype=col_types)

        df_app_test.replace(replace_dict, inplace=True)

        assert df_app_test.shape == (48744, 120)

        return df_app_train, df_app_test



    

def load_prev_app_data():

    interest_types = CategoricalDtype(["low_action", "low_normal", "middle", "high"], True)

    col_types = {

        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "NAME_CONTRACT_TYPE": LoanType,

        "AMT_ANNUITY": np.float32, "AMT_APPLICATION": np.float32, "AMT_CREDIT": np.float32,

        "AMT_DOWN_PAYMENT": np.float32, "AMT_GOODS_PRICE": np.float32,

        "WEEKDAY_APPR_PROCESS_START": WeekDayType, "HOUR_APPR_PROCESS_START": np.uint8,

        "FLAG_LAST_APPL_PER_CONTRACT": YesNoType, "NFLAG_LAST_APPL_IN_DAY": np.bool,

        "RATE_DOWN_PAYMENT": np.float32, "RATE_INTEREST_PRIMARY": np.float32, "RATE_INTEREST_PRIVILEGED": np.float32,

        "NAME_CASH_LOAN_PURPOSE": "category", "NAME_CONTRACT_STATUS": LoanStatusType,

        "DAYS_DECISION": np.int16, "NAME_PAYMENT_TYPE": "category", "CODE_REJECT_REASON": "category",

        "NAME_TYPE_SUITE": "category", "NAME_CLIENT_TYPE": "category", "NAME_GOODS_CATEGORY": "category",

        "NAME_PORTFOLIO": "category", "NAME_PRODUCT_TYPE": "category", "CHANNEL_TYPE": "category",

        "SELLERPLACE_AREA": np.float32, "NAME_SELLER_INDUSTRY": "category", "CNT_PAYMENT": np.float16,

        "NAME_YIELD_GROUP": interest_types, "PRODUCT_COMBINATION": "category",

        "DAYS_FIRST_DRAWING": np.float32, "DAYS_FIRST_DUE": np.float32,

        "DAYS_LAST_DUE_1ST_VERSION": np.float32, "DAYS_LAST_DUE": np.float32, "DAYS_TERMINATION": np.float32,

        "NFLAG_INSURED_ON_APPROVAL": np.float16

    }



    d = {365243: np.nan}



    replace_dict = {"SELLERPLACE_AREA": {-1: np.nan},

                    "DAYS_FIRST_DRAWING": d, "DAYS_FIRST_DUE": d,

                    "DAYS_LAST_DUE_1ST_VERSION": d, "DAYS_LAST_DUE": d, "DAYS_TERMINATION": d}

    df_prev_app = pd.read_csv(PATH_PRE_APP, na_values=["XNA", ""], index_col=0,

                              keep_default_na=False, dtype=col_types)

    # df_prev_app.set_index("SK_ID_PREV")

    df_prev_app.replace(replace_dict, inplace=True)

    assert df_prev_app.shape == (1670214, 36), f"Incorrect shape for df_prev_app: {df_prev_app.shape}"

    return df_prev_app





def load_install_payments(load_processed=True):

    col_types = {

        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32,

        "NUM_INSTALMENT_VERSION": np.uint8, "NUM_INSTALMENT_NUMBER": np.uint16,

        "DAYS_INSTALMENT": np.int16, "DAYS_ENTRY_PAYMENT": np.float16,

        "AMT_INSTALMENT": np.float32, "AMT_PAYMENT": np.float32,

        "NUM_PAYMENTS": np.uint16, "AMT_OVERDUE": np.float32, "AMT_DPD30": np.float32

    }



    if load_processed:

        return pd.read_csv(PATH_INST_PAY_PROCESSED, dtype=col_types)

    else:

        df = pd.read_csv(PATH_INST_PAY, dtype=col_types)

        assert df.shape == (13605401, 8)

        return df





def load_pos_balance():

    col_types = {

        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "MONTHS_BALANCE": np.int16,

        "CNT_INSTALMENT": np.float16, "CNT_INSTALMENT_FUTURE": np.float16,

        "NAME_CONTRACT_STATUS": CreditStatusType,

        "SK_DPD": np.uint16, "SK_DPD_DEF": np.uint16

    }

    df = pd.read_csv(PATH_POS_BALANCE, na_values=["XNA"], dtype=col_types)

    assert df.shape == (10001358, 8)

    return df





def load_credit_balance_data():

    col_types = {

        "SK_ID_PREV": np.uint32, "SK_ID_CURR": np.uint32, "MONTHS_BALANCE": np.int16,

        "AMT_BALANCE": np.float32, "AMT_CREDIT_LIMIT_ACTUAL": np.float32,

        "AMT_DRAWINGS_ATM_CURRENT": np.float32, "AMT_DRAWINGS_CURRENT": np.float32,

        "AMT_DRAWINGS_OTHER_CURRENT": np.float32, "AMT_DRAWINGS_POS_CURRENT": np.float32,

        "AMT_INST_MIN_REGULARITY": np.float32, "AMT_PAYMENT_CURRENT": np.float32,

        "AMT_PAYMENT_TOTAL_CURRENT": np.float32, "AMT_RECEIVABLE_PRINCIPAL": np.float32,

        "AMT_RECIVABLE": np.float32, "AMT_TOTAL_RECEIVABLE": np.float32,

        "CNT_DRAWINGS_ATM_CURRENT": np.float16, "CNT_DRAWINGS_CURRENT": np.float16,

        "CNT_DRAWINGS_OTHER_CURRENT": np.float16, "CNT_DRAWINGS_POS_CURRENT": np.float16,

        "CNT_INSTALMENT_MATURE_CUM": np.float16, "NAME_CONTRACT_STATUS": "category",

        "SK_DPD": np.uint16, "SK_DPD_DEF": np.uint16

    }

    df = pd.read_csv(PATH_CREDIT_BALANCE, dtype=col_types)

    assert df.shape == (3840312, 23)

    return df





def load_bureau():

    currency_types = CategoricalDtype(["currency " + str(v) for v in range(1, 5)], False)

    col_types = {

        "SK_ID_BUREAU": np.uint32, "SK_ID_CURR": np.uint32, "CREDIT_ACTIVE": CreditStatusType,

        "CREDIT_CURRENCY": currency_types, "DAYS_CREDIT": np.float16, "CREDIT_DAY_OVERDUE": np.uint16,

        "DAYS_ENDDATE_FACT": np.float16, "DAYS_CREDIT_ENDDATE": np.float16,

        "AMT_CREDIT_MAX_OVERDUE": np.float32, "CNT_CREDIT_PROLONG": np.uint8,

        "AMT_CREDIT_SUM": np.float32, "AMT_CREDIT_SUM_DEBT": np.float32, "AMT_CREDIT_SUM_LIMIT": np.float32,

        "AMT_CREDIT_SUM_OVERDUE": np.float32, "AMT_ANNUITY": np.float32,

        "CREDIT_TYPE": "category"

    }

    df = pd.read_csv(PATH_BUREAU, index_col=1, dtype=col_types)

    assert df.shape == (1716428, 16)

    return df





def load_bureau_balance():

    col_types = {

        "SK_ID_BUREAU": np.uint32, "MONTHS_BALANCE": np.int16, "STATUS": BureauBalanceStatusType

    }

    df = pd.read_csv(PATH_BUREAU_BALANCE, na_values=["X"], dtype=col_types)

    assert df.shape == (27299925, 3)

    return df
def flatten_agg_df_columns(df_agg, prefix=None):

    if prefix is None:

        df_agg.columns = ['_'.join([c0, c1.upper()]) for c0, c1 in df_agg.columns]

    else:

        df_agg.columns = ['_'.join([prefix, c0, c1.upper()]) for c0, c1 in df_agg.columns]

    return df_agg



def clean_inst_pay(df_inst_pay):

    print("Cleaning installment payments data")

    df_inst_pay["DAYS_ENTRY_PAYMENT"].fillna(0, inplace=True)

    df_inst_pay["AMT_PAYMENT"].fillna(-1, inplace=True)

    df_inst_pay_valid_filter = (df_inst_pay["AMT_PAYMENT"] > 0) | (df_inst_pay["AMT_INSTALMENT"] > 0)

    print(f"Remove {(~df_inst_pay_valid_filter).sum():d} invalid records.")

    df_inst_pay_group = df_inst_pay[df_inst_pay_valid_filter].groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER",

                                                                       "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"])

    del df_inst_pay_valid_filter



    print("Aggregate multiple installments for one payment")

    df_inst_pay_group_cnt = df_inst_pay_group.size()

    df_inst_agg = flatten_agg_df_columns(df_inst_pay_group.agg({

        "SK_ID_CURR": ["min", "max"],

        "NUM_INSTALMENT_VERSION": ["max", "nunique"],

        "DAYS_INSTALMENT": ["min", "max"],

        "AMT_INSTALMENT": ["min", "max", "sum"]

    }))

    del df_inst_pay_group



    print("Processing 1")

    assert (df_inst_agg["SK_ID_CURR_MIN"] == df_inst_agg["SK_ID_CURR_MAX"]).all(axis=None), "Inconsistent SK_ID_CURR"

    df_inst_pay_processed = pd.DataFrame(index=df_inst_agg.index)

    df_inst_pay_processed["SK_ID_CURR"] = df_inst_agg["SK_ID_CURR_MIN"]



    df_inst_pay_group_cnt_distict = df_inst_agg["NUM_INSTALMENT_VERSION_NUNIQUE"]

    df_inst_pay_group_check = ((df_inst_pay_group_cnt == 2) |

                               (df_inst_pay_group_cnt_distict == 1))

    assert df_inst_pay_group_check.all(axis=None)

    del df_inst_pay_group_cnt, df_inst_pay_group_check

    df_inst_pay_processed["NUM_INSTALMENT_VERSION"] = df_inst_agg["NUM_INSTALMENT_VERSION_MAX"]



    assert (df_inst_agg["DAYS_INSTALMENT_MIN"] == df_inst_agg["DAYS_INSTALMENT_MAX"]).all(axis=None)

    df_inst_pay_processed["DAYS_INSTALMENT"] = df_inst_agg["DAYS_INSTALMENT_MIN"]



    df_agg_filter = (df_inst_pay_group_cnt_distict == 2)

    assert (df_agg_filter | (df_inst_agg["AMT_INSTALMENT_MIN"] == df_inst_agg["AMT_INSTALMENT_MAX"])).all(axis=None)

    df_inst_pay_processed["AMT_INSTALMENT"] = df_inst_agg["AMT_INSTALMENT_MIN"]

    df_inst_pay_processed.loc[df_agg_filter, "AMT_INSTALMENT"] = df_inst_agg["AMT_INSTALMENT_SUM"]

    print("%d payments aggregated" % df_agg_filter.sum())

    del df_inst_pay_group_cnt_distict, df_agg_filter



    df_inst_pay_processed.reset_index(inplace=True)

    # df_inst_pay_processed["DAYS_ENTRY_PAYMENT"].astype(np.float16, copy=False)

    df_inst_pay_processed["DAYS_ENTRY_PAYMENT"] = df_inst_pay_processed["DAYS_ENTRY_PAYMENT"].astype(np.float16,

                                                                                                     copy=False)

    df_inst_pay_processed["AMT_PAYMENT"] = df_inst_pay_processed["AMT_PAYMENT"].astype(np.float32, copy=False)

    df_inst_pay_processed["AMT_PAYMENT"].replace(-1, -np.inf, inplace=True)

    assert ((df_inst_pay_processed["AMT_PAYMENT"] >= 0) |

            (df_inst_pay_processed["DAYS_ENTRY_PAYMENT"] == 0)).all(axis=None)

    df_diff_entry_offset = df_inst_pay_processed["DAYS_ENTRY_PAYMENT"] - df_inst_pay_processed["DAYS_INSTALMENT"]

    df_inst_pay_processed["AMT_DUE_PAYMENT"] = (np.fmax(df_inst_pay_processed["AMT_PAYMENT"], 0) *

                                                (df_diff_entry_offset <= 0))

    df_inst_pay_processed["AMT_DUE30_PAYMENT"] = (np.fmax(df_inst_pay_processed["AMT_PAYMENT"], 0) *

                                                  (df_diff_entry_offset <= 30))



    df_inst_pay_group = df_inst_pay_processed.groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"])

    del df_diff_entry_offset, df_inst_pay_processed, df_inst_agg



    print("Aggregate multiple payments for one installment")

    df_inst_pay_group_cnt = df_inst_pay_group.size()

    df_inst_agg = flatten_agg_df_columns(df_inst_pay_group.agg({

        "SK_ID_CURR": ["min", "max"],

        # "NUM_INSTALMENT_VERSION": ["min", "max"],

        "DAYS_INSTALMENT": ["min", "max"],

        "DAYS_ENTRY_PAYMENT": ["min", "max"],

        "AMT_INSTALMENT": ["min", "max", "sum"],

        "AMT_PAYMENT": ["sum"],

        "AMT_DUE_PAYMENT": ["sum"],

        "AMT_DUE30_PAYMENT": ["sum"]

    }, skipna=False))

    del df_inst_pay_group

    print("Finish aggregations")

    

    gc.collect()



    print("Processing 2")

    assert (df_inst_agg["SK_ID_CURR_MIN"] == df_inst_agg["SK_ID_CURR_MAX"]).all(), "Inconsistent SK_ID_CURR"

    df_inst_pay_processed = pd.DataFrame(index=df_inst_agg.index)

    df_inst_pay_processed["SK_ID_CURR"] = df_inst_agg["SK_ID_CURR_MIN"]



    # df_inst_agg_INST_VER = df_inst_agg["NUM_INSTALMENT_VERSION"]

    # assert (df_inst_agg_INST_VER["min"] == df_inst_agg_INST_VER["max"]).all(axis=None), "Inconsistent NUM_INSTALMENT_VERSION"

    # df_inst_pay_processed["NUM_INSTALMENT_VERSION"] = df_inst_agg_INST_VER["min"]



    assert (df_inst_agg["DAYS_INSTALMENT_MIN"] ==

            df_inst_agg["DAYS_INSTALMENT_MAX"]).all(axis=None), "Inconsistent DAYS_INSTALMENT"

    df_inst_pay_processed["DAYS_INSTALMENT"] = df_inst_agg["DAYS_INSTALMENT_MIN"]



    df_inst_pay_processed["DAYS_FIRST_PAYMENT"] = df_inst_agg["DAYS_ENTRY_PAYMENT_MIN"].replace(0, np.nan)

    df_inst_pay_processed["DAYS_LAST_PAYMENT"] = df_inst_agg["DAYS_ENTRY_PAYMENT_MAX"].replace(0, np.nan)



    assert (df_inst_agg["AMT_INSTALMENT_MIN"] == df_inst_agg["AMT_INSTALMENT_MAX"]).all(axis=None)

    df_inst_pay_processed["AMT_INSTALMENT"] = df_inst_agg["AMT_INSTALMENT_MIN"]



    # Fix missing installment info

    # df_prev_app_ann = pd.read_csv(r"data\previous_application.csv", index_col=0, usecols=[0, 3])

    # df_inst_agg = df_inst_agg.join(df_prev_app_ann, how="left")

    #

    # df_annuity_check = ((df_inst_agg.index.get_level_values(2) != 1) | df_inst_agg["AMT_ANNUITY"].isna() |

    #                     (df_inst_agg["AMT_INSTALMENT_min"] == 0) |

    #                     ((df_inst_agg["AMT_ANNUITY"] - df_inst_agg["AMT_INSTALMENT_min"]).abs() < 0.01))

    # assert df_annuity_check.all(axis=None)

    # inst_fix_filter = ((df_inst_agg["NUM_INSTALMENT_VERSION"] == 1) & (df_inst_agg["AMT_INSTALMENT_min"] == 0))

    # df_inst_pay_processed.loc[inst_fix_filter, "AMT_INSTALMENT"] = df_inst_agg.loc[inst_fix_filter, "AMT_ANNUITY"]

    # del df_annuity_check, inst_fix_filter



    # inst_fix_filter = (df_inst_agg["AMT_INSTALMENT_min"] == 0)

    # df_inst_pay_processed.loc[inst_fix_filter, "AMT_INSTALMENT"] = df_inst_agg.loc[inst_fix_filter, "AMT_PAYMENT_sum"]

    # del inst_fix_filter



    df_inst_pay_invalid_filter = (df_inst_agg["AMT_PAYMENT_SUM"] < 0)

    assert ((~df_inst_pay_invalid_filter) | (df_inst_pay_group_cnt == 1)).all(axis=None)

    df_inst_pay_processed["AMT_PAYMENT"] = df_inst_agg["AMT_PAYMENT_SUM"]

    df_inst_pay_processed.loc[df_inst_pay_invalid_filter, "AMT_PAYMENT"] = np.nan

    assert (df_inst_pay_processed["AMT_PAYMENT"] != 0).all(axis=None)



    df_inst_pay_invalid_filter = df_inst_pay_processed["AMT_PAYMENT"].isnull()

    df_inst_pay_processed["NUM_PAYMENTS"] = df_inst_pay_group_cnt.astype(np.uint16)

    df_inst_pay_processed.loc[df_inst_pay_invalid_filter, "NUM_PAYMENTS"] = np.uint16(0)

    print("%d installments aggregated" % (df_inst_pay_group_cnt > 1).sum())

    del df_inst_pay_group_cnt, df_inst_pay_invalid_filter



    df_inst_pay_processed["AMT_OVERDUE"] = np.fmax(df_inst_pay_processed["AMT_INSTALMENT"] -

                                                   df_inst_agg["AMT_DUE_PAYMENT_SUM"], 0)

    df_inst_pay_processed["AMT_OVERDUE"] *= (df_inst_pay_processed["AMT_OVERDUE"] >= 0.01)

    df_inst_pay_processed["AMT_DPD30"] = np.fmax(df_inst_pay_processed["AMT_INSTALMENT"] -

                                                 df_inst_agg["AMT_DUE30_PAYMENT_SUM"], 0)

    df_inst_pay_processed["AMT_DPD30"] *= (df_inst_pay_processed["AMT_DPD30"] >= 0.01)

    df_inst_pay_processed["AMT_UNPAID"] = np.fmax(df_inst_pay_processed["AMT_INSTALMENT"] -

                                                  df_inst_pay_processed["AMT_PAYMENT"].fillna(0), 0)

    df_inst_pay_processed["AMT_UNPAID"] *= (df_inst_pay_processed["AMT_UNPAID"] >= 0.01)

    df_inst_pay_processed.reset_index(inplace=True)

    # df_inst_pay_processed.rename(columns={"NUM_INSTALMENT_NUMBER": "NUM_INSTALMENT_NUMBER",

    #                                       "NUM_INSTALMENT_VERSION": "INSTALMENT_VER"})

    del df_inst_agg

    

    print("Finish processing")



    output_columns = ["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",

                      "DAYS_INSTALMENT", "DAYS_FIRST_PAYMENT", "DAYS_LAST_PAYMENT", "NUM_PAYMENTS",

                      "AMT_INSTALMENT", "AMT_PAYMENT", "AMT_OVERDUE", "AMT_DPD30", "AMT_UNPAID"]

    df_inst_pay_processed.drop(df_inst_pay_processed.columns.drop(output_columns), axis=1, inplace=True)



#     df_inst_pay_processed.to_csv(r"installments_payments_processed.csv", index=False)

    print("Finish cleaning installment payment data")

    return df_inst_pay_processed





def merge_payment_info():

    df_inst_pay = load_install_payments()

    df_pos = load_pos_balance()

    df_inst_pay["MONTHS_BALANCE"] = df_inst_pay["DAYS_INSTALMENT"] // 30

    df_pos.set_index(["SK_ID_PREV", "MONTHS_BALANCE"], inplace=True, drop=True)

    df_merged = df_inst_pay.join(df_pos, on=["SK_ID_PREV", "MONTHS_BALANCE"], how="outer", rsuffix="_POS")

    t = df_merged[(df_merged.NAME_CONTRACT_STATUS != "Active") & df_merged.SK_ID_CURR.notnull()]

    return t
def group_values(col_orig, new_col_values):

    return pd.DataFrame({col: col_orig.isin(values) for col, values in new_col_values})



def get_preprocessed_bureau_data():

    df_bureau = load_bureau()

    df_bureau_balance = load_bureau_balance()



    df_bureau_balance["STATUS"] = df_bureau_balance["STATUS"].cat.codes - 1

    df_bureau_balance["HAS_OVERDUE"] = df_bureau_balance["STATUS"] > 0

    df_bureau_balance_agg = df_bureau_balance.groupby("SK_ID_BUREAU").agg({

        "HAS_OVERDUE": ["sum"],

        "STATUS": ["max"]

    })

    df_bureau_balance_agg.columns = ["CNT_OVERDUE", "MAX_OVERDUE_TIME"]

    

#     df_bureau_balance_L12_agg = (df_bureau_balance["MONTHS_BALANCE"] > -12).agg({

#         "HAS_OVERDUE": ["count", "sum"],

#         "STATUS": ["max"]

#     })

#     df_bureau_balance_L12_agg = ["NUM_CREDITS", "CNT_OVERDUE", "MAX_OVERDUE"]

    

    df_bureau = df_bureau.join(df_bureau_balance_agg)

    df_bureau["CNT_OVERDUE"].where(df_bureau["CNT_OVERDUE"] > 0,

                                   df_bureau["AMT_CREDIT_MAX_OVERDUE"].gt(0).astype(np.uint8, copy=False),

                                   inplace=True)

    

    df_bureau.loc[df_bureau["DAYS_ENDDATE_FACT"].isna() & (df_bureau["AMT_CREDIT_SUM_DEBT"] == 0) &

                  (df_bureau["DAYS_CREDIT_ENDDATE"] < 0), "DAYS_ENDDATE_FACT"] = df_bureau["DAYS_CREDIT_ENDDATE"]

    df_bureau["DURATION"] = df_bureau["DAYS_ENDDATE_FACT"].fillna(0) - df_bureau["DAYS_CREDIT"]

    df_bureau["CLOSE_DURATION_RATIO"] = ((df_bureau["DAYS_ENDDATE_FACT"] - df_bureau["DAYS_CREDIT"]) / 

                                         (df_bureau["DAYS_CREDIT_ENDDATE"] - df_bureau["DAYS_CREDIT"]))

    

    bureau_groups = df_bureau.groupby("SK_ID_CURR")

    df_bureau_agg = flatten_agg_df_columns(bureau_groups.agg({

        "CNT_OVERDUE": ["sum"],

        "DURATION": ["sum"],

        "AMT_CREDIT_MAX_OVERDUE": ["max"],

        "AMT_CREDIT_SUM_OVERDUE": ["max", "sum"],

        "DAYS_CREDIT": ["min", "max"],

        "CREDIT_DAY_OVERDUE": ["max", "mean"],

        "MAX_OVERDUE_TIME": ["max"],

        "AMT_CREDIT_SUM": ["max", "mean", "sum"],

        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],

        "CNT_CREDIT_PROLONG": ["max"],

        "DURATION": ["sum"],

        "DAYS_ENDDATE_FACT": ["max"],

        "CLOSE_DURATION_RATIO": ["min", "max", "mean"]



#         'DAYS_CREDIT': ['mean', 'var'],

#         'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],

#         'DAYS_CREDIT_UPDATE': ['mean'],

#         'AMT_CREDIT_MAX_OVERDUE': ['mean'],

#         'AMT_CREDIT_SUM_OVERDUE': ['mean'],

#         'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],

#         'AMT_ANNUITY': ['max', 'mean'],

#         'MONTHS_BALANCE_MIN': ['min'],

#         'MONTHS_BALANCE_MAX': ['max'],

#         'MONTHS_BALANCE_SIZE': ['mean', 'sum']

    }), "BUREAU")

    df_bureau_agg["BUREAU_OVERDUE_FREQ"] = df_bureau_agg["BUREAU_CNT_OVERDUE_SUM"] / df_bureau_agg["BUREAU_DURATION_SUM"]

    df_bureau_agg["BUREAU_LOAN_CNT"] = bureau_groups.size()



    # def func_bureau_agg(s):

    #     total_overdues = (s.CREDIT_DAY_OVERDUE > 0).sum()

    #     max_overdue = s.AMT_CREDIT_MAX_OVERDUE.max()

    #     # active_debt = s[s.CREDIT_ACTIVE == "Active"].AMT_CREDIT_SUM_DEBT.sum()

    #     return pd.Series([total_overdues, max_overdue], index=["total_overdues", "max_overdues"])

    # df_bureau_agg = df_bureau.groupby("SK_ID_CURR").apply(func_bureau_agg)



    bureau_active_groups = df_bureau[df_bureau.CREDIT_ACTIVE == "Active"].groupby("SK_ID_CURR")

    df_bureau_active_agg = flatten_agg_df_columns(bureau_active_groups.agg({

        "DAYS_CREDIT_ENDDATE": ["max"],

        "AMT_ANNUITY": ["sum"]

    }), "BUREAU_ACTIVE")

    df_bureau_agg = df_bureau_agg.join(df_bureau_active_agg, how="left")



    df_bureau_agg.fillna({

        "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX": 0,

        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": 0,

        "BUREAU_ACTIVE_AMT_CREDIT_SUM_DEBT_SUM": 0,

        # "BUREAU_ACTIVE_DAYS_CREDIT_ENDDATE_MAX": 100000

    }, inplace=True)



    # assert df_bureau_agg.notnull().all(axis=None)

    

    del df_bureau, df_bureau_balance

    return df_bureau_agg
def get_preprocessed_previous_app_data(load_inst_pay=True, load_credit_balance=True):

    # def func_prev_app_agg(s):

    #     return pd.Series({"has_assessed_risk": s.has_assessed_risk.sum(),

    #                       "max_refused": s[s.is_refused].AMT_APPLICATION.max(),

    #                       "total_approved": s[s.is_approved].size,

    #                       "max_prev_annuity": s[s.is_approved].AMT_ANNUITY.max()})



    df_prev_app = load_prev_app_data()

    df_prev_app["APPROVED_RATIO"] = df_prev_app["AMT_CREDIT"] / df_prev_app["AMT_APPLICATION"]

    df_prev_app["DOWN_PAYMENT_RATIO"] = df_prev_app["AMT_DOWN_PAYMENT"] / df_prev_app["AMT_CREDIT"]

    df_prev_app["CNT_ASSESSED_RISK"] = (df_prev_app["NAME_PRODUCT_TYPE"] == "x-sell").astype(np.uint8, copy=False)

    df_prev_app["NAME_YIELD_GROUP_CODE"] = df_prev_app["NAME_YIELD_GROUP"].cat.codes

    df_prev_app.loc[df_prev_app["NAME_CONTRACT_TYPE"] != "Revolving loans", "PAYMENT_DURATION_RATIO"] = (

        (df_prev_app["DAYS_LAST_DUE"] - df_prev_app["DAYS_FIRST_DUE"] + 30) / (30 * df_prev_app["CNT_PAYMENT"])

    )

    

    total_ratio = df_prev_app["AMT_ANNUITY"] * df_prev_app["CNT_PAYMENT"] / df_prev_app["AMT_CREDIT"]

    total_ratio = total_ratio[total_ratio.notna() & (df_prev_app["CNT_PAYMENT"] > 0)]

    interest_rate_est = 1 - 1 / total_ratio

    for _ in range(7):

        interest_rate_est = 1 - (1 - interest_rate_est ** df_prev_app["CNT_PAYMENT"]) / total_ratio

    df_prev_app["INTEREST_RATE"] = interest_rate_est

    del total_ratio, interest_rate_est

    

    prev_app_groups = df_prev_app[df_prev_app["NAME_CONTRACT_STATUS"] != "Canceled"].groupby("SK_ID_CURR")

    df_prev_app_agg = flatten_agg_df_columns(prev_app_groups.agg({

        "CNT_ASSESSED_RISK": ["sum"],

        "NAME_YIELD_GROUP_CODE": ["mean"],

        "DAYS_DECISION": ["max"]

        

#         'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],

#         'AMT_ANNUITY': ['min', 'max', 'mean'],

#         'AMT_APPLICATION': ['min', 'max', 'mean'],

#         'AMT_CREDIT': ['min', 'max', 'mean'],

#         'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],

#         'AMT_GOODS_PRICE': ['min', 'max', 'mean'],

#         'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],

#         'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],

#         'DAYS_DECISION': ['min', 'max', 'mean'],

#         'CNT_PAYMENT': ['mean', 'sum'],

    }), "PREV")



    prev_app_refused_groups = df_prev_app[df_prev_app.NAME_CONTRACT_STATUS == "Refused"].groupby("SK_ID_CURR")

    df_prev_refused_agg = flatten_agg_df_columns(prev_app_refused_groups.agg({

        "AMT_APPLICATION": ["max"],

        "RATE_INTEREST_PRIMARY": ["mean", "max"],

        "NAME_YIELD_GROUP_CODE": ["mean", "max"]

    }), "PREV_REFUSED")

    df_prev_refused_agg["PREV_REFUSED_CNT"] = prev_app_refused_groups.size()

    df_prev_app_agg = df_prev_app_agg.join(df_prev_refused_agg, how="left")

    del df_prev_refused_agg



    prev_app_approved_groups = df_prev_app[df_prev_app.NAME_CONTRACT_STATUS == "Approved"].groupby("SK_ID_CURR")

    df_prev_approved_agg = flatten_agg_df_columns(prev_app_approved_groups.agg({

        "AMT_APPLICATION": ["max"],

        "AMT_ANNUITY": ["max"],

        "DOWN_PAYMENT_RATIO": ["min", "max", "mean"],

        "RATE_INTEREST_PRIMARY": ["mean", "max"],

        "NAME_YIELD_GROUP_CODE": ["mean", "max"],

        "INTEREST_RATE": ["mean", "min", "max"],

        "PAYMENT_DURATION_RATIO": ["min", "max", "mean"]

    }), "PREV_APPROVED")

    df_prev_approved_agg["PREV_APPROVED_CNT"] = prev_app_approved_groups.size()

    df_prev_app_agg = df_prev_app_agg.join(df_prev_approved_agg, how="left")

    del df_prev_approved_agg

    del df_prev_app



    if load_inst_pay:

        if os.path.exists(PATH_INST_PAY_PROCESSED):

            df_inst_pay = load_install_payments()

        else:

            df_inst_pay = clean_inst_pay(load_install_payments(False))

            columns_to_write = ["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",

                    "DAYS_INSTALMENT", "DAYS_FIRST_PAYMENT", "DAYS_LAST_PAYMENT", "NUM_PAYMENTS",

                    "AMT_INSTALMENT", "AMT_PAYMENT", "AMT_OVERDUE", "AMT_DPD30", "AMT_UNPAID"]

            df_inst_pay.to_csv(PATH_INST_PAY_PROCESSED, index=False, columns=columns_to_write)

            # df_POS_CASH_balance = load_pos_balance()



        # df_prev_app_processed = df_prev_app[["SK_ID_CURR"]].copy()

        # df_prev_app_processed[""] = df_prev_app["NAME_CONTRACT_TYPE"]



        df_inst_pay["CNT_OVERDUE"] = (df_inst_pay.AMT_OVERDUE > 0)

        df_inst_pay["CNT_DPD30"] = (df_inst_pay.AMT_DPD30 > 0)

        df_inst_pay_groups = df_inst_pay.groupby("SK_ID_CURR")



        df_inst_pay_agg = flatten_agg_df_columns(df_inst_pay_groups.agg({

            "AMT_OVERDUE": ["max", "mean"],

            "CNT_OVERDUE": ["sum", "mean"],

            "AMT_DPD30": ["max", "mean"],

            "CNT_DPD30": ["sum", "mean"],

            "AMT_UNPAID": ["sum"]

        }), "INST_PAY")



        # print_null_columns(df_inst_pay_agg)

        df_prev_app_agg = df_prev_app_agg.join(df_inst_pay_agg, how="outer")

        del df_inst_pay,  df_inst_pay_agg



    if load_credit_balance:

        df_credit_card_balance = load_credit_balance_data()

        df_credit_card_balance.sort_values(["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE"], inplace=True)

         

        df_credit_card_balance["CNT_OVERDUE"] = (df_credit_card_balance["SK_DPD"] > 0).astype(np.uint8)

        df_credit_card_balance["CNT_OVERDUE_DEF"] = (df_credit_card_balance["SK_DPD_DEF"] > 0).astype(np.uint8)

        df_credit_card_balance["IS_BALANCE_HIGH"] = (2 * df_credit_card_balance["AMT_BALANCE"] > df_credit_card_balance["AMT_CREDIT_LIMIT_ACTUAL"]).astype(np.uint8)

        cc_groups = df_credit_card_balance.groupby("SK_ID_CURR")

        df_cc_agg = flatten_agg_df_columns(cc_groups.agg({

            "MONTHS_BALANCE": ["min", "max"],

            "SK_DPD": ["max", "mean"],

            "SK_DPD_DEF": ["max", "mean"],

            "CNT_OVERDUE": ["sum"],

            "CNT_OVERDUE_DEF": ["sum"],

            "IS_BALANCE_HIGH": ["mean", "max"],

        }), "CREDIT_CARD")

        

        df_cc_agg_prev = flatten_agg_df_columns(df_credit_card_balance.groupby("SK_ID_PREV").agg({

            "SK_ID_CURR": ["last"],

            "MONTHS_BALANCE": ["last"],

            "AMT_BALANCE": ["last"],

            "AMT_CREDIT_LIMIT_ACTUAL": ["max", "last"],

            "SK_DPD": ["last"],

            "SK_DPD_DEF": ["last"]

        }))

        df_cc_agg_prev["LAST_LIMIT_RATIO"] = df_cc_agg_prev["AMT_CREDIT_LIMIT_ACTUAL_LAST"] / df_cc_agg_prev["AMT_CREDIT_LIMIT_ACTUAL_MAX"]

        

        df_cc_agg = df_cc_agg.join(flatten_agg_df_columns(df_cc_agg_prev.groupby("SK_ID_CURR_LAST").agg({

            "AMT_CREDIT_LIMIT_ACTUAL_LAST": ["sum", "max"],

            "AMT_BALANCE_LAST": ["sum"],

            "LAST_LIMIT_RATIO": ["mean", "min"],

            "SK_DPD_LAST": ["max", "sum"],

            "SK_DPD_DEF_LAST": ["max", "sum"]

        }), "CREDIT_CARD"), how="inner")

        df_cc_agg["CREDIT_BALANCE_RATIO"] = (df_cc_agg["CREDIT_CARD_AMT_BALANCE_LAST_SUM"] / 

                                             df_cc_agg["CREDIT_CARD_AMT_CREDIT_LIMIT_ACTUAL_LAST_SUM"]).fillna(0)

        

        df_prev_app_agg = df_prev_app_agg.join(df_cc_agg, how="outer")

        del df_credit_card_balance, df_cc_agg



    df_prev_app_agg.fillna({

        "PREV_REFUSED_AMT_APPLICATION_MAX": 0,

        "PREV_REFUSED_CNT": 0,

        "PREV_APPROVED_AMT_APPLICATION_MAX": 0,

        "PREV_APPROVED_AMT_ANNUITY_MAX": 0,

        "PREV_APPROVED_CNT": 0,

        "PREV_CNT_ASSESSED_RISK_SUM": 0,

        "INST_PAY_AMT_OVERDUE_MAX": 0,

        "INST_PAY_CNT_OVERDUE_SUM": 0,

        "INST_PAY_CNT_OVERDUE_MEAN": 0,

        "INST_PAY_AMT_DPD30_MAX": 0,

        "INST_PAY_CNT_DPD30_SUM": 0,

        "INST_PAY_CNT_DPD30_MEAN": 0,

        "INST_PAY_AMT_UNPAID_SUM": 0,

        "CREDIT_CARD_MONTHS_BALANCE_MAX": 0,

        "CREDIT_CARD_MONTHS_BALANCE_MIN": 0,

        "CREDIT_CARD_SK_DPD_MAX": 0,

        "CREDIT_CARD_SK_DPD_DEF_MAX": 0,

        "CREDIT_CARD_CNT_OVERDUE_SUM": 0,

        "CREDIT_CARD_CNT_OVERDUE_DEF_SUM": 0

    }, inplace=True)



    # print(df_prev_app_agg.head())

    # print_null_columns(df_prev_app_agg)

    return df_prev_app_agg



def preprocess_app(transformers, df, df_bureau_agg, df_prev_app_agg, impute=True):

    perform_grouping = True

    excluded_columns = ["TARGET"]

    PASSTHROUGH_COLS = ["CNT_CHILDREN", "CNT_FAM_MEMBERS",

                         "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",

                         "REGION_POPULATION_RELATIVE",

                         "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",

                         "OWN_CAR_AGE",

                         "FLAG_EMP_PHONE",

                         # "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE",

                         # "FLAG_EMAIL",

                         "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",

                         # "HOUR_APPR_PROCESS_START",

                         # "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",

                         # "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",

                         "DAYS_LAST_PHONE_CHANGE",

                         "TOTALAREA_MODE", "AMT_REQ_CREDIT_BUREAU_MON"]

    # PASSTHROUGH_COLS.extend(("FLAG_DOCUMENT_" + str(i)) for i in range(2, 22))



    IMPUTER_FIX_VAL_COL_MAP = {

        0: ["AMT_GOODS_PRICE", "OBS_30_CNT_SOCIAL_CIRCLE",

            "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"],

        -1: ["OWN_CAR_AGE"]

    }

    MEAN_IMPUTER_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    MOST_FREQ_IMPUPTER_COLS = ["CNT_FAM_MEMBERS", "AMT_ANNUITY", "DAYS_LAST_PHONE_CHANGE"]



    housing_columns = ["APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",

                       "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",

                       "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG",

                       "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",

                       "NONLIVINGAREA_AVG",

                       "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",

                       "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE",

                       "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE",

                       "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE",

                       "NONLIVINGAREA_MODE",

                       "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI",

                       "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI",

                       "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",

                       "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",

                       # 'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'

                       ]



    df_app_processed = df[[]].copy()

    df_app_processed["NAME_CONTRACT_TYPE"] = df["NAME_CONTRACT_TYPE"].str.startswith("C").astype(np.uint8)

    df_app_processed["IS_MALE"] = df["CODE_GENDER"].cat.codes

    df_app_processed["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].cat.codes

    df_app_processed["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].cat.codes

    df_app_processed["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].cat.codes

    if perform_grouping:

        name_type_suite_groups = [("Acc_No", ["Unaccompanied"]),

                                  ("Acc_Fam_Ch", ["Family", "Children", "Group of people"]),

                                  ("Acc_Spouse", ["Spouse, partner"]),

                                  ("Acc_Other", ["Other_A", "Other_B"])]

        df_app_processed = pd.concat(

            [df_app_processed, group_values(df["NAME_TYPE_SUITE"], name_type_suite_groups)], axis=1, copy=False)



        family_status_groups = [("With_family", ["Married", "Civil marriage"]),

                                ("Without_family", ["Single / not married", "Separated", "Widow"])]

        df_app_processed = pd.concat(

            [df_app_processed, group_values(df["NAME_FAMILY_STATUS"], family_status_groups)], axis=1, copy=False)



        income_type_groups = [("Income_Job", ["Working", "Maternity leave"]),

                              ("Income_Commercial", ["Commercial associate", "Businessman"]),

                              ("Income_Pensioner", ["Pensioner"]),

                              ("Income_Servant", ["State servant"]),

                              ("Income_Other", ["Unemployed", "Student"])]

        df_app_processed = pd.concat(

            [df_app_processed, group_values(df["NAME_INCOME_TYPE"], income_type_groups)], axis=1, copy=False)



        organization_groups = [("Org_Missing", [np.nan]),

                               ("Org_Business_1", ["Business Entity Type 1"]),

                               ("Org_Business_2", ["Business Entity Type 2"]),

                               ("Org_Business_3", ["Business Entity Type 3"]),

                               ("Org_Government", ["Government"]),

                               ("Org_Self", ["Self-employed"]),

                               ("Org_Trade_7", ["Trade: type 7"]),

                               ("Org_Transport_3", ["Transport: type 3"]),

                               ("Org_Transport_4", ["Transport: type 4"]),

                               ("Org_Medicine", ["Medicine"]),

                               ("Org_Other", ["Other"]),

                               ("Org_Mix_0", ["Trade: type 6", "Transport: type 1", "Industry: type 12"]),

                               ("Org_Mix_1", ["Bank", "Military", "Police", "University", "Security Ministries"]),

                               ("Org_Mix_2", ["School", "Insurance", "Culture"]),

                               ("Org_Mix_3", ["Trade: type 5", "Trade: type 4", "Religion"]),

                               ("Org_Mix_4", ["Hotel", "Industry: type 10", "Medicine"]),

                               ("Org_Mix_5", ["Industry: type 3", "Realtor", "Agriculture",

                                              "Trade: type 3", "Industry: type 4", "Security"]),

                               ("Org_Mix_6", ["Industry: type 11", "Postal"]),

                               ("Org_Mix_7", ["Industry: type 13", "Industry: type 8", "Restaurant",

                                              "Construction", "Cleaning", "Industry: type 1"]),

                               ]

        df_app_processed = pd.concat(

            [df_app_processed, group_values(df["ORGANIZATION_TYPE"], organization_groups)], axis=1, copy=False)



        housing_groups = [

                          ("Housing_Missing", [np.nan]),

                          ("Housing_Own", ["House / apartment"]),

                          ("Housing_Provided", ["Municipal apartment", "Office apartment", "Co-op apartment"]),

                          ("Housing_Rent", ["Rented apartment"]),

                          ("Housing_Parent", ["With parents"])

                         ]

        df_app_processed = pd.concat(

            [df_app_processed, group_values(df["NAME_HOUSING_TYPE"], housing_groups)], axis=1, copy=False)

    else:

        df_app_processed = append_one_hot_encoding(df_app_processed, df["NAME_TYPE_SUITE"], prefix="Acc")

        df_app_processed = append_one_hot_encoding(df_app_processed, df["NAME_FAMILY_STATUS"], prefix="Fam")

        df_app_processed = append_one_hot_encoding(df_app_processed, df["NAME_INCOME_TYPE"], prefix="Income")

        df_app_processed = append_one_hot_encoding(df_app_processed, df["ORGANIZATION_TYPE"], prefix="Org")

        df_app_processed = append_one_hot_encoding(df_app_processed, df["NAME_HOUSING_TYPE"], prefix="Housing")



    df_app_processed[PASSTHROUGH_COLS] = df[PASSTHROUGH_COLS]

#     df_app_processed["EXT_SCORE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

    df_app_processed["ANN_PERCENT"] = df.AMT_ANNUITY / df.AMT_INCOME_TOTAL

    df_app_processed["EMPLOYED_PERCENT"] = df.DAYS_EMPLOYED / df.DAYS_BIRTH

    df_app_processed["PAYMENT_RATIO"] = df.AMT_CREDIT / df.AMT_ANNUITY

    # df_app_processed["LEFT_OVER"] = df.AMT_INCOME_TOTAL - df.AMT_ANNUITY



    if transformers:

        imputer = transformers["imputer"]

        imputer_cols = transformers["imputer_cols"]

    else:

        imputer_cols = []

        for cols in IMPUTER_FIX_VAL_COL_MAP.values():

            imputer_cols.extend(cols)

        imputer_cols.extend(MEAN_IMPUTER_COLS)

        imputer_cols.extend(MOST_FREQ_IMPUPTER_COLS)

        transformers["imputer_cols"] = imputer_cols

        assert len(imputer_cols) == len(set(imputer_cols))



        if impute:

            imputer = [("FixVal_Imputer_" + str(v), SimpleImputer(strategy="constant", fill_value=v), cols)

                       for v, cols in IMPUTER_FIX_VAL_COL_MAP.items()]

            imputer.append(("Mean_Imputer", SimpleImputer(strategy="mean"), MEAN_IMPUTER_COLS))

            imputer.append(("Most_Freq_Imputer", SimpleImputer(strategy="most_frequent"), MOST_FREQ_IMPUPTER_COLS))



            transformers["imputer"] = imputer = ColumnTransformer(imputer)



            imputer.fit(df[imputer_cols])

        else:

            transformers["imputer"] = imputer = ColumnTransformer([("No_Imputer", "passthrough", imputer_cols)])

            imputer.fit(df[imputer_cols])



    df_app_processed[imputer_cols] = pd.DataFrame(imputer.transform(df[imputer_cols]), index=df.index, copy=False)

    # df_app_processed[housing_columns] = df[housing_columns]



    if df_bureau_agg is not None:

        df_app_processed = df_app_processed.join(df_bureau_agg, how="left")

    if df_prev_app_agg is not None:

        df_app_processed = df_app_processed.join(df_prev_app_agg, how="left")

        df_app_processed["ANNUITY_RATIO"] = df_app_processed.AMT_ANNUITY / df_app_processed.PREV_APPROVED_AMT_ANNUITY_MAX



    # for name, col in df.iteritems():

    #     if not (col.isnull().any() or col.dtype == "object" or name in excluded_columns):

    #         # df_app_processed[name] = col

    #         print(name)



    return df_app_processed
def get_preprocessed_data(impute=True, random_seed=None):

    print("Reading data")

    df_app_train, df_app_test = load_app_data()

    print("Finish reading data")



    print("Preprocess training data")

    df_bureau_agg = None

    df_prev_app_agg = None

    df_bureau_agg = get_preprocessed_bureau_data()

    print("Finish preprocessing bureau data")

    df_prev_app_agg = get_preprocessed_previous_app_data(True, True)

    print("Finish preprocessing previous application data")



    df_app_train = shuffle(df_app_train, random_state=random_seed)

    transformers = dict()

    X_train = preprocess_app(transformers, df_app_train, df_bureau_agg, df_prev_app_agg, impute=impute)

    y_train = df_app_train["TARGET"]

    print("Finish preprocessing application data")



    print("Preprocess test data")

    X_test = preprocess_app(transformers, df_app_test, df_bureau_agg, df_prev_app_agg, impute=impute)



    print("Training data shape:", X_train.shape)

    X_train.info(verbose=5)

    

    return X_train, y_train, X_test



X_train, y_train, X_test = get_preprocessed_data(PERFORM_IMPUTATION, RANDOM_SEED)
def build_model_and_classify(X_train, y_train, X_test=None, classifier="xgb", early_stopping=True, 

                             cv=CROSS_VALIDATION_FOLD, tune_param=False, random_seed=None, 

                             output_path="submission.csv"):

    print("Initializing classifier")

    

    if classifier == "xgb":

        if tune_param:

            clf = XGBClassifier(seed=random_seed, tree_method="gpu_hist")

        else:

            clf = XGBClassifier(n_estimators=2000, min_child_weight=32, max_depth=6, # max_leaves=64,

                                min_split_loss=0.08, learning_rate=0.1,

                                reg_lambda=1, reg_alpha=0.6,

                                scale_pos_weight=3,

                                seed=random_seed, tree_method="gpu_hist")

            # clf = XGBClassifier(max_depth=8, min_child_weight=12, seed=1)

#             clf = XGBClassifier(n_estimators=10000, learning_rate=0.02, max_leaves=34, max_depth=8,

#                             colsample_bytree=0.95, subsample=0.872,

#                             reg_alpha=0.0415, reg_lambda=0.0735, min_split_loss=0.022,

#                             min_child_weight=39.33,

#                             seed=random_seed, tree_method="gpu_hist")

    elif classifier == "lgbm":

#         clf = LGBMClassifier(

#             n_jobs=8, n_estimators=200, learning_rate=0.02, num_leaves=34, max_depth=8,

#             colsample_bytree=0.95, subsample=0.872,

#             reg_alpha=0.0415, reg_lambda=0.0735,

#             min_split_gain=0.022, min_child_weight=39.33,

# #             early_stopping_round=10,

#             metric="auc",

#             tree_learner="data",

#             data_random_seed=random_seed,

# #             silent=-1,

#             verbose=1)

        clf = LGBMClassifier(

            boosting_type="goss", n_estimators=1000, learning_rate=0.00513,

            num_leaves=54, max_depth=10, subsample_for_bin=240000,

            reg_alpha=0.436193, reg_lambda=0.479169,

            min_split_gain=0.025, colsample_bytree=0.52,

            subsample=1, # 'is_unbalance': False,

            silent=-1, verbose=-1

        )

    else:

        # clf = GradientBoostingClassifier(max_depth=10, min_samples_split=15, verbose=5)

        # clf = DecisionTreeClassifier(class_weight=weight_dict, max_depth=15, min_samples_split=4)

        # clf = LogisticRegression(class_weight=weight_dict)

        raise ValueError("Unsupported classifier: " + classifier)



    if early_stopping:

        fit_orig = clf.fit

        VALIDATION_SIZE = 0.1

        def fit_wrapped(X, y, **kwargs):

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SIZE)

            fit_orig(X_train, y_train, eval_set=[(X_val, y_val)], 

                     early_stopping_rounds=50, eval_metric="auc")

        clf.fit = fit_wrapped

        

    if cv > 0:

        if tune_param:

            print("Tuning classifier parameters")

            params = {

                "n_estimators": [200, 400],

                "max_depth": [6, 8],

                "max_leaves": [64, 128],

                "min_child_weight": [24, 32],

                "min_split_loss": [0.02 * (1 << n) for n in range(6)],

                "learning_rate": [0.06 + 0.02 * (1 << n) for n in range(1, 4)],

                "reg_lambda": [0.2 * n for n in range(6)],

                "reg_alpha": [0.2 * n for n in range(6)],

                "scale_pos_weight": [1, 3, 5]

            }

#             model_selection_task = GridSearchCV(clf, param_grid=params, scoring="roc_auc", cv=cv, verbose=5)

            model_selection_task = RandomizedSearchCV(clf, params, n_iter=50, scoring="roc_auc", 

                                                      cv=cv, n_jobs=5, verbose=5)

            global results

            results = model_selection_task.fit(X_train, y_train)

            print(f"Best score: {results.best_score_}")

            print(f"Best parameters: {results.best_params_}")

            print(f"CV results: {results.cv_results_}")

            clf = results.best_estimator_

        else:

            print(f"Perform {cv:d}-fold cross validation")

            score_val = sum(cross_val_score(clf, X_train, y_train,

                                            cv=cv, scoring="roc_auc", verbose=10, n_jobs=5)) / cv

            print("Validation AUC: %.6f" % score_val)

    else:

        test_size = 0.1

        print(f"Perform hold-out validation (Test size: {test_size:.0%})")

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_seed)

        clf.fit(X_train, y_train)

        prob_val = clf.predict_proba(X_val)[:, 1]

        score_val = roc_auc_score(y_val, prob_val)

        print("Validation AUC: %.6f" % score_val)

    # print(clf.feature_importances_)



    if X_test is None:

        return



    print("Training classifier")

    clf.fit(X_train, y_train)



#     print("Dumping trained classifier")

#     from joblib import dump

#     dump(clf, 'boost_tree_gpu_0.joblib')



    print("Classify training and test set")

#     train_prob_df = pd.DataFrame(clf.predict_proba(X_train)[:, 1], index=X_train.index, columns=["PRED_PROB"])

#     train_prob_df.to_csv("train_prob.csv")

    test_prob_df = pd.DataFrame(clf.predict_proba(X_test)[:, 1], index=X_test.index, columns=["TARGET"])

    test_prob_df.to_csv(output_path)

    print("Finished classifying")
# %%timeit -n 1 -r 1



# Training with LGBM

build_model_and_classify(X_train, y_train, X_test, "lgbm", cv=CROSS_VALIDATION_FOLD, tune_param=True,

                         random_seed=RANDOM_SEED, output_path="submission_lgbm.csv")
# build_model_and_classify(X_train, y_train, X_test, random_seed=RANDOM_SEED)
build_model_and_classify(X_train, y_train, X_test, tune_param=True, random_seed=RANDOM_SEED,

                         output_path="submission_xgb.csv")
df_result = pd.DataFrame.from_records(results.cv_results_["params"])

df_result["rank"] = results.cv_results_["rank_test_score"]

df_result["score"] = results.cv_results_["mean_test_score"]

df_result.sort_values(by="rank", inplace=True)

df_result
for c in df_result.columns[:-2]:

    print(df_result.groupby(c)["score"].mean().sort_values())
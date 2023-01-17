import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%pylab inline

pylab.rcParams['figure.figsize'] = (10, 6)



limit_rows   = 7000000

train_df           = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,

                                                    "ind_nuevo":str,

                                                    "ult_fec_cli_1t":str,

                                                    "indext":str}, nrows=limit_rows)

unique_ids   = pd.Series(train_df["ncodpers"].unique())

limit_people = 12000

unique_id    = unique_ids.sample(n=limit_people)

train_df     = train_df[train_df.ncodpers.isin(unique_id)]

train_df.describe()
# convert the dates to standard Year-Month-Day format

train_df["fecha_dato"] = pd.to_datetime(train_df["fecha_dato"],format="%Y-%m-%d")

train_df["fecha_alta"] = pd.to_datetime(train_df["fecha_alta"],format="%Y-%m-%d")

train_df["fecha_dato"].unique()



# customers will be more likely to buy products at certain months of the year (Christmas bonuses?), so let's add a month column.

train_df["month"] = pd.DatetimeIndex(train_df["fecha_dato"]).month

train_df["age"]   = pd.to_numeric(train_df["age"], errors="coerce")

# missing values?

train_df.isnull().any()
# Let's see if we can fill in missing values by looking how many months of history these customers have.

months_active = train_df.loc[train_df["ind_nuevo"].isnull(),:].groupby("ncodpers", sort=False).size()

months_active.max()

# Looks like these are all new customers, so replace accordingly.

train_df.loc[train_df["ind_nuevo"].isnull(),"ind_nuevo"] = 1



train_df.antiguedad = pd.to_numeric(train_df.antiguedad,errors="coerce")

# the same people that we just determined were new customers

train_df.loc[train_df["antiguedad"].isnull(),"ind_nuevo"].describe()

train_df.loc[train_df.antiguedad.isnull(),"antiguedad"] = train_df.antiguedad.min()

train_df.loc[train_df.antiguedad <0, "antiguedad"]      = 0



train_df.loc[train_df.ind_actividad_cliente.isnull(),"ind_actividad_cliente"] = train_df["ind_actividad_cliente"].median()



# Some entries don't have the date they joined the company. Just give them something in the middle of the pack

dates=train_df.loc[:,"fecha_alta"].sort_values().reset_index()

median_date = int(np.median(dates.index.values))

train_df.loc[train_df.fecha_alta.isnull(),"fecha_alta"] = dates.loc[median_date,"fecha_alta"]

train_df["fecha_alta"].describe()



# Fill in missing with the more common status.

pd.Series([i for i in train_df.indrel]).value_counts()

train_df.loc[train_df.indrel.isnull(),"indrel"] = 1



# manual fix

train_df.loc[train_df.nomprov.isnull(),"nomprov"] = "UNKNOWN"

train_df.loc[train_df.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"



# fill in missing with median income by region

incomes = train_df.loc[train_df.renta.notnull(),:].groupby("nomprov").agg({"renta":{"MedianIncome":median}})

incomes.sort_values(by=("renta","MedianIncome"),inplace=True)

incomes.reset_index(inplace=True)

incomes.nomprov = incomes.nomprov.astype("category", categories=[i for i in train_df.nomprov.unique()],ordered=False)

incomes.head()



grouped        = train_df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()

new_incomes    = pd.merge(train_df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]

new_incomes    = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")

train_df.sort_values("nomprov",inplace=True)

train_df       = train_df.reset_index()

new_incomes    = new_incomes.reset_index()



train_df.loc[train_df.renta.isnull(),"renta"] = new_incomes.loc[train_df.renta.isnull(),"renta"].reset_index()

train_df.loc[train_df.renta.isnull(),"renta"] = train_df.loc[train_df.renta.notnull(),"renta"].median()

train_df.sort_values(by="fecha_dato",inplace=True)



train_df.loc[train_df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"] = 0

train_df.loc[train_df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0



train_df.drop(["tipodom","cod_prov"],axis=1,inplace=True)
# Age distribution

def age_distribution(df):

    with sns.plotting_context("notebook",font_scale=1.5):

        sns.set_style("whitegrid")

        sns.distplot(df["age"].dropna(),

                     bins=80,

                     kde=False,

                     color="tomato")

        plt.ylabel("Count")

        plt.xlim((15,100))

        

age_distribution(train_df)

# Let's separate the distribution and move the outliers to the mean of the closest one.

train_df.loc[train_df.age < 18,"age"]  = train_df.loc[(train_df.age >= 18) & (train_df.age <= 30),"age"].mean(skipna=True)

train_df.loc[train_df.age > 100,"age"] = train_df.loc[(train_df.age >= 30) & (train_df.age <= 100),"age"].mean(skipna=True)

train_df["age"].fillna(train_df["age"].mean(),inplace=True)

train_df["age"]                  = train_df["age"].astype(int)

# plot again

age_distribution(train_df)
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing, ensemble

import xgboost as xgb

from collections import defaultdict



import csv

import datetime

from operator import sub



usecols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',

       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',

       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',

       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']



df_train = pd.read_csv('../input/train_ver2.csv', usecols=usecols)

sample = pd.read_csv('../input/sample_submission.csv')



df_train = df_train.drop_duplicates(['ncodpers'], keep='last')

df_train.fillna(0, inplace=True)



models = {}

model_preds = {}

id_preds = defaultdict(list)

ids = df_train['ncodpers'].values

for c in df_train.columns:

    if c != 'ncodpers':

        print(c)

        y_train = df_train[c]

        x_train = df_train.drop([c, 'ncodpers'], 1)

        

        clf = LogisticRegression()

        clf.fit(x_train, y_train)

        p_train = clf.predict_proba(x_train)[:,1]

        

        models[c] = clf

        model_preds[c] = p_train

        for id, p in zip(ids, p_train):

            id_preds[id].append(p)

            

        print(roc_auc_score(y_train, p_train))

        

already_active = {}

for row in df_train.values:

    row = list(row)

    id = row.pop(0)

    active = [c[0] for c in zip(df_train.columns[1:], row) if c[1] > 0]

    already_active[id] = active

    

train_preds = {}

for id, p in id_preds.items():

    # Here be dragons

    preds = [i[0] for i in sorted([i for i in zip(df_train.columns[1:], p) if i[0] not in already_active[id]], key=lambda i:i [1], reverse=True)[:7]]

    train_preds[id] = preds

    

test_preds = []

for row in sample.values:

    id = row[0]

    p = train_preds[id]

    test_preds.append(' '.join(p))



sample['added_products'] = test_preds

sample.to_csv('submission.csv', index=False)
"""

Code based on BreakfastPirate Forum post

__author__ : SRK

"""

import csv

import datetime

from operator import sub

import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn import preprocessing, ensemble



mapping_dict = {

'ind_empleado'  : {-99:0, 'N':1, 'B':2, 'F':3, 'A':4, 'S':5},

'sexo'          : {'V':0, 'H':1, -99:2},

'ind_nuevo'     : {'0':0, '1':1, -99:2},

'indrel'        : {'1':0, '99':1, -99:2},

'indrel_1mes'   : {-99:0, '1.0':1, '1':1, '2.0':2, '2':2, '3.0':3, '3':3, '4.0':4, '4':4, 'P':5},

'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},

'indresi'       : {-99:0, 'S':1, 'N':2},

'indext'        : {-99:0, 'S':1, 'N':2},

'conyuemp'      : {-99:0, 'S':1, 'N':2},

'indfall'       : {-99:0, 'S':1, 'N':2},

'tipodom'       : {-99:0, '1':1},

'ind_actividad_cliente' : {'0':0, '1':1, -99:2},

'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:2},

'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},

'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}

}

cat_cols = list(mapping_dict.keys())



target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

target_cols = target_cols[2:]



def getTarget(row):

    tlist = []

    for col in target_cols:

        if row[col].strip() in ['', 'NA']:

            target = 0

        else:

            target = int(float(row[col]))

        tlist.append(target)

    return tlist



def getIndex(row, col):

    val = row[col].strip()

    if val not in ['','NA']:

        ind = mapping_dict[col][val]

    else:

        ind = mapping_dict[col][-99]

    return ind



def getAge(row):

    mean_age = 40.

    min_age = 20.

    max_age = 90.

    range_age = max_age - min_age

    age = row['age'].strip()

    if age == 'NA' or age == '':

        age = mean_age

    else:

        age = float(age)

        if age < min_age:

            age = min_age

        elif age > max_age:

            age = max_age

    return round( (age - min_age) / range_age, 4)



def getCustSeniority(row):

    min_value = 0.

    max_value = 256.

    range_value = max_value - min_value

    missing_value = 0.

    cust_seniority = row['antiguedad'].strip()

    if cust_seniority == 'NA' or cust_seniority == '':

        cust_seniority = missing_value

    else:

        cust_seniority = float(cust_seniority)

        if cust_seniority < min_value:

            cust_seniority = min_value

        elif cust_seniority > max_value:

            cust_seniority = max_value

    return round((cust_seniority-min_value) / range_value, 4)



def getRent(row):

    min_value = 0.

    max_value = 1500000.

    range_value = max_value - min_value

    missing_value = 101850.

    rent = row['renta'].strip()

    if rent == 'NA' or rent == '':

        rent = missing_value

    else:

        rent = float(rent)

        if rent < min_value:

            rent = min_value

        elif rent > max_value:

            rent = max_value

    return round((rent-min_value) / range_value, 6)



def processData(in_file_name, cust_dict):

    x_vars_list = []

    y_vars_list = []

    for row in csv.DictReader(in_file_name):

        # use only the four months as specified by breakfastpirate #

        if row['fecha_dato'] not in ['2015-05-28', '2015-06-28', '2016-05-28', '2016-06-28']:

            continue



        cust_id = int(row['ncodpers'])

        if row['fecha_dato'] in ['2015-05-28', '2016-05-28']:    

            target_list = getTarget(row)

            cust_dict[cust_id] =  target_list[:]

            continue



        x_vars = []

        for col in cat_cols:

            x_vars.append( getIndex(row, col) )

        x_vars.append( getAge(row) )

        x_vars.append( getCustSeniority(row) )

        x_vars.append( getRent(row) )



        if row['fecha_dato'] == '2016-06-28':

            prev_target_list = cust_dict.get(cust_id, [0]*22)

            x_vars_list.append(x_vars + prev_target_list)

        elif row['fecha_dato'] == '2015-06-28':

            prev_target_list = cust_dict.get(cust_id, [0]*22)

            target_list = getTarget(row)

            new_products = [max(x1 - x2,0) for (x1, x2) in zip(target_list, prev_target_list)]

            if sum(new_products) > 0:

                for ind, prod in enumerate(new_products):

                    if prod>0:

                        assert len(prev_target_list) == 22

                        x_vars_list.append(x_vars+prev_target_list)

                        y_vars_list.append(ind)



    return x_vars_list, y_vars_list, cust_dict

            

def runXGB(train_X, train_y, seed_val=0):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.05

    param['max_depth'] = 8

    param['silent'] = 1

    param['num_class'] = 22

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = 1

    param['subsample'] = 0.7

    param['colsample_bytree'] = 0.7

    param['seed'] = seed_val

    num_rounds = 50



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    model = xgb.train(plst, xgtrain, num_rounds)    

    return model





if __name__ == "__main__":

    start_time = datetime.datetime.now()

    data_path = "../input/"

    train_file =  open(data_path + "train_ver2.csv")

    x_vars_list, y_vars_list, cust_dict = processData(train_file, {})

    train_X = np.array(x_vars_list)

    train_y = np.array(y_vars_list)

    print(np.unique(train_y))

    del x_vars_list, y_vars_list

    train_file.close()

    print(train_X.shape, train_y.shape)

    print(datetime.datetime.now()-start_time)

    test_file = open(data_path + "test_ver2.csv")

    x_vars_list, y_vars_list, cust_dict = processData(test_file, cust_dict)

    test_X = np.array(x_vars_list)

    del x_vars_list

    test_file.close()

    print(test_X.shape)

    print(datetime.datetime.now()-start_time)



    print("Building model..")

    model = runXGB(train_X, train_y, seed_val=0)

    del train_X, train_y

    print("Predicting..")

    xgtest = xgb.DMatrix(test_X)

    preds = model.predict(xgtest)

    del test_X, xgtest

    print(datetime.datetime.now()-start_time)



    print("Getting the top products..")

    target_cols = np.array(target_cols)

    preds = np.argsort(preds, axis=1)

    preds = np.fliplr(preds)[:,:7]

    test_id = np.array(pd.read_csv(data_path + "test_ver2.csv", usecols=['ncodpers'])['ncodpers'])

    final_preds = [" ".join(list(target_cols[pred])) for pred in preds]

    out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})

    out_df.to_csv('submission2.csv', index=False)

    print(datetime.datetime.now()-start_time)
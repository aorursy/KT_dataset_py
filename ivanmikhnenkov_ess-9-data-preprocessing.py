# ESS round 9: features selection, recoding numerical missings into NaN, imputing values.





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk



# Setting up visualization

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ESS9 = pd.read_csv("../input/european-social-survey-round-9-ed-11/ESS9e01_1.csv")
# Choose only features that don't need to be recoded later (no nominal with >2 choices)

# Take features which would be used in draft model and categorize based on what miss dict to apply.



# No missings

features0 = ['dscrrce', 'dscrntn', 'dscrrlg', 'dscrlng', 'dscretn', 'dscrage', 'dscrgnd', 'dscrsex', 'dscrdsb', 'dscroth', 

             'pdwrk', 'edctn', 'uempla', 'uempli', 'dsbld', 'rtrd', 'cmsrv', 'hswrk', 'pdwrkp', 'edctnp', 'uemplap',

       'uemplip', 'dsbldp', 'rtrdp', 'cmsrvp', 'hswrkp']



# 7, 8, 9 are missings 

features1 = ['netusoft', 'polintr', 'psppsgva', 'actrolga', 'psppipla', 'cptppola', 'vote', 'contplt', 'wrkprty', 'wrkorg', 

             'badge', 'sgnptit','pbldmn', 'bctprd', 'pstplonl', 'clsprty', 'gincdif', 'freehms','hmsfmlsh', 'hmsacld', 

             'imsmetn', 'imdfetn', 'impcntr', 'sclact', 'crmvct', 'aesfdrk', 'health', 'hlthhmp', 'rlgblg', 'dscrgrp',

            'ctzcntr', 'brncntr', 'blgetmg', 'facntr', 'mocntr', 'evpdemp', 'evmar', 'bthcld', 'ggchld', 'anvcld','alvgptn', 

             'acldnmr', 'aftjbyc', 'advcyc', 'hincfel', 'frprtpl', 'gvintcz', 'poltran', 'topinfr', 'btminfr', 'wltdffr', 'recskil', 'recexp', 'recknow', 'recimg', 'recgndr',

       'sofrdst', 'sofrwrk', 'sofrpr', 'sofrprv', 'ppldsrv', 'jstprev',

       'pcmpinj', 'evlvptn', 'emprf14', 'emprm14', 'atncrse', 'ipcrtiv','imprich','ipeqopt','ipshabt','impsafe','impdiff','ipfrule','ipudrst', 'ipmodst', 'ipgdtim','impfree',

          'iphlppl', 'ipsuces', 'ipstrgv', 'ipadvnt', 'ipbhprp', 'iprspot', 'iplylfr', 'impenv', 'imptrad', 'impfun']



# 6, 7, 8, 9 are missings

features2 = ['prtdgcl', 'rlgblge', 'gndr', 'gndr2', 'lvgptnea', 'dvrcdeva', 'chldhhe', 'domicil', 'crpdwk', 'pdjobev', 'estsz', 

             'jbspv', 'icwhct','wrkac6m', 'uemp3m', 'uemp12m', 'uemp5yr', 'mbtru', 'grspfr', 'netifr','occinfr']



# 77, 88, 99 are missing

features3 = ['ppltrst', 'pplfair', 'pplhlp', 'trstprl', 'trstlgl', 'trstplc', 'trstplt', 'trstprt', 'trstep','trstun',

             'lrscale', 'stflife', 'stfeco', 'stfgov', 'stfdem', 'stfedu', 'stfhlth', 'euftf', 'imbgeco', 'imueclt', 

             'imwbcnt', 'happy', 'sclmeet', 'inprdsc', 'atchctr','atcherp', 'rlgdgr', 'rlgatnd', 'pray', 'plnftr', 'hhmmb',

            'maritalb', 'eduyrs', 'hinctnta']



# 66, 77, 88, 99 are missings

features4 = ['rlgdnm', 'rlgdnme', 'nbthcld', 'ngchld', 'rshpsts', 'wkdcorga', 'iorgact', 'tporgwk', 'occf14b', 'occm14b',

            'grsplet', 'netilet']



# 666, 777, 888, 999 are missings

features5 = ['agea', 'wkhct', 'wkhtot', 'wkhtotp']



# 7777, 8888, 9999 are missings

features6 = ['nwspol', 'yrbrn']



# 6666, 7777, 8888, 9999 are missings

features7 = ['netustm', 'livecnta', 'pdempyr', 'lvptnyr', 'maryr', 'fcldbrn', 'ycldbyr', 'ygcdbyr', 'pdjobyr']



# 0, 1111, 7777, 8888, 9999 are missings

features_s1 = ['lvpntyr']



# 33, 44, 55, 65, 77, 88, 99 are missings

features_s2 = ['vteurmmb']



# 0, 55, 66, 77, 88, 99 are missings

features_s3 = ['eiscedp', 'eisced', 'eiscedf', 'eiscedm']



# 3, 6, 7, 8, 9: missing

features_s4 = ['emplrel', 'wrkctra', 'emprelp']



# 55, 66, 77, 88, 99 are missings

features_s5 = ['ifredu', 'ifrjob', 'evfredu', 'evfrjob']



# 0, 777, 888, 999 are missings

features_SB1 = ['ageadlt', 'agemage', 'ageoage', 'iagpnt', 'tygledu', 'tygpnt', 'tolvpnt', 'tochld']

# 0, 111, 777, 888, 999 are missings

features_SB2 = ['iaglptn', 'iagmr', 'tyglvp', 'tygmr', 'towkht']

# 0, 111, 222, 777, 888, 999 are missings

features_SB3 = ['iagrtr', 'tygrtr']



weights = ['dweight', 'pweight']

# Verification that no feature were saved twice in different lists

all_vars =  features0 + features1 + features2 + features3 + features4 + features5 + features6 + features7 + features_s1 + features_s2 + features_s3 + features_s4 + features_s5 + features_SB1 + features_SB2 + features_SB3 + weights

all_features_unique = len(all_vars) == len(set(all_vars))

print("All features saved are unique (no feature were accidentally saved twice):", all_features_unique)
print("Overall number of variables:", len(all_vars))
# Dictionaries based on which the features numerical missings would be recoded to NaN

# One dictionary for each feature list (except features0 that don't contain any missings)



# Universal missing value

missing = pd.np.nan



# Dictionaries mapping numerics to missing var based on how features to recode were implemented. I check labels in SPSS.

missRecDict1 = {7: missing, 8: missing, 9: missing}

missRecDict2 = {6: missing, 7: missing, 8: missing, 9: missing}

missRecDict3 = {77: missing, 88: missing, 99: missing}

missRecDict4 = {66: missing, 77: missing, 88: missing, 99: missing}

missRecDict5 = {666: missing, 777: missing, 888: missing, 999: missing}

missRecDict6 = {7777: missing, 8888: missing, 9999: missing}

missRecDict7 = {6666: missing, 7777: missing, 8888: missing, 9999: missing}



missRecDict_s1 = {0: missing, 1111: missing, 7777: missing, 8888: missing, 9999: missing}

missRecDict_s2 = {33: missing, 44: missing, 55: missing, 65: missing, 77: missing, 88: missing, 99: missing}

missRecDict_s3 = {0: missing, 55: missing, 66: missing, 77: missing, 88: missing, 99: missing}

missRecDict_s4 = {3: missing, 6: missing, 7: missing, 8: missing, 9: missing}

missRecDict_s5 = {55: missing, 66: missing, 77: missing, 88: missing, 99: missing}



# Technically 0, 111, 222 are not missings but nominal answers when the majority of answers are scale, so recoded into missings to simplify things.

missRecDict_SB1 = {0: missing, 777: missing, 888: missing, 999: missing}

missRecDict_SB2 = {0: missing, 111: missing, 777: missing, 888: missing, 999: missing}

missRecDict_SB3 = {0: missing, 111: missing, 222: missing, 777: missing, 888: missing, 999: missing}

# Save to Data_m, where 'm' stands for "missings".

Data_m = pd.DataFrame()

Data_m[features0] = ESS9[features0]

Data_m[features1] = ESS9[features1].replace(missRecDict1)

Data_m[features2] = ESS9[features2].replace(missRecDict2)

Data_m[features3] = ESS9[features3].replace(missRecDict3)

Data_m[features4] = ESS9[features4].replace(missRecDict4)

Data_m[features5] = ESS9[features5].replace(missRecDict5)

Data_m[features6] = ESS9[features6].replace(missRecDict6)

Data_m[features7] = ESS9[features7].replace(missRecDict7)

Data_m[features_s1] = ESS9[features_s1].replace(missRecDict_s1)

Data_m[features_s2] = ESS9[features_s2].replace(missRecDict_s2)

Data_m[features_s3] = ESS9[features_s3].replace(missRecDict_s3)

Data_m[features_s4] = ESS9[features_s4].replace(missRecDict_s4)

Data_m[features_s5] = ESS9[features_s5].replace(missRecDict_s5)

Data_m[features_SB1] = ESS9[features_SB1].replace(missRecDict_SB1)

Data_m[features_SB2] = ESS9[features_SB2].replace(missRecDict_SB2)

Data_m[features_SB3] = ESS9[features_SB3].replace(missRecDict_SB3)

Data_m[weights] = ESS9[weights]
vars_change_dtype = Data_m.dtypes.loc[Data_m.dtypes == "O"].keys().tolist()



for var in vars_change_dtype:

    Data_m[var] = pd.to_numeric(Data_m[var], errors='coerce')

# Save Data_m with 226 columns and all missings encoded as NaN.

Data_m.to_csv('Data_missings.csv',index=False)
Data_m.count().loc[Data_m.count() < 18008]
# Dropping variables with counts less than 18008

low_count_cols = Data_m.count().loc[Data_m.count() < 18008].keys().tolist()

Data_dropped = Data_m.copy().drop(low_count_cols, axis = 1)
Data_dropped.describe()
from sklearn.impute import SimpleImputer



imputer_mf = SimpleImputer(missing_values = missing, strategy = "median")



# "imp" stands for imputing: which is dropping variables with count less than 18008 and imputing all variables with "most_frequent" strategy

Data_imp = pd.DataFrame(imputer_mf.fit_transform(Data_dropped))

Data_imp.columns = Data_dropped.columns
Data_imp.describe()
# Save Data_m with 211 columns and all missings (NaN) imputed by median value of its variable.

Data_imp.to_csv('Data_imputed.csv',index=False)
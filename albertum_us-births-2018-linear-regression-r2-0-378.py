# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LassoCV

from sklearn import metrics

from sklearn.preprocessing import PolynomialFeatures

from scipy import stats

%matplotlib inline



pd.set_option('display.max_columns',500)

pd.set_option('display.max_rows',500)
df = pd.read_csv('../input/us-births-2018/US_births(2018).csv', 

                 low_memory=False)
def drop_rows(df):

    '''

    Dropping rows where missing: 'DBWT', 'BMI', 'DBWT', 'WTGAIN', 'PWgt_R', 'DLMP_MM', 'DLMP_YY'

    '''

    df.drop(df[df['DBWT'].eq(9999)].index, inplace=True)    

    df.drop(df[df['BMI'].eq(99.9)].index, inplace=True)

    df.drop(df[df['DBWT'].eq(9999)].index, inplace=True)

    df.drop(df[df['WTGAIN'].eq(99)].index, inplace=True)

    df.drop(df[df['PWgt_R'].eq(999)].index, inplace=True)

    df.drop(df[df['DLMP_MM'].eq(99)].index, inplace=True)

    df.drop(df[df['DLMP_YY'].eq(9999)].index, inplace=True)

    

    df.drop(columns=['IMP_SEX'], inplace=True)

    return df

df = drop_rows(df)
def feature_engineer(df):

    '''

    Creating new column: 'first_birth', 'smoked', 'PRIORDEAD_cat', 'PRIORTERM_cat', 'PRIORLIVE_cat', 

    

    '''



    #creating new column called 'first_birth': Is the baby the Mom's first child? Yes:1 No:0

    df['first_birth'] = np.where(df['ILLB_R'].eq(888), 1, 0)

    df['plural_delivery'] = np.where(df['ILLB_R'].lt(4), 'Yes', 'No')





    

    #creating new column 'smoked': Did the mother smoke before pregnancy? Yes: Daily, No: None, Unknown: Unknown

    conditions = [df['CIG_0'].eq(0),

                  df['CIG_0'].eq(99)]

    choices = ['None',

               'Unknown']

    df['smoked'] = np.select(conditions, choices, 'Daily')



    

    #creating new column 'PRIORDEAD_cat': Did the mother previously have miscarriages? Yes: Yes, No: None, Unknown: Unknown

    conditions = [df['PRIORDEAD'].eq(0),

                  df['PRIORDEAD'].eq(99)]

    choices = ['None',

               'Unknown']

    df['PRIORDEAD_cat'] = np.select(conditions, choices, 'Yes')



    

    #creating new column 'PRIORTERM_cat': Did the mother previously have terminations? Yes: Yes, No:None, Unknown: Unknown

    conditions = [df['PRIORTERM'].eq(0),

                  df['PRIORTERM'].eq(99)]

    choices = ['None',

               'Unknown']

    df['PRIORTERM_cat'] = np.select(conditions, choices, 'Yes')



    #creating new column 'PRIORLIVE_cat': Did the mother previously birth living children: Yes: Yes, No: None, Unknown: Unknown

    conditions = [df['PRIORLIVE'].eq(0),

                  df['PRIORLIVE'].eq(99)]

    choices = ['None',

               'Unknown']

    df['PRIORLIVE_cat'] = np.select(conditions, choices, 'Yes')



    #creating new column 'pregnancy_length': An estimation of the gestation period by subtracting the month/year of last menses from month/year of baby born

    conditions = [(df['DOB_MM'] > df['DLMP_MM']) & (2018 == df['DLMP_YY']),

                  (df['DOB_MM'] > df['DLMP_MM']) & (2018 > df['DLMP_YY']),

                  (df['DOB_MM'] < df['DLMP_MM']) & (2018 > df['DLMP_YY'])]

    choices = [df['DOB_MM'] - df['DLMP_MM'],

               ((df['DOB_YY'] - df['DLMP_YY'])* 12) + df['DOB_MM'] - df['DLMP_MM'],

               ((df['DOB_YY'] - df['DLMP_YY'])* 12) - df['DLMP_MM'] + df['DOB_MM']]

    df['pregnancy_length'] = np.select(conditions,choices, 12)

    

    #creating new column 'MAGER_cat': Mother's age is < 18: Minor, Mother's age is >= 18: Adult

    df['MAGER_cat'] = np.where(df['MAGER'].lt(18),'Minor','Adult')



    #creating new column 'pregnancy_cat: Binning pregnancy_length to 5 bins: 'Early','8','9','10','Late'

    condition =[df['pregnancy_length'].eq(9),

                df['pregnancy_length'].eq(8),

                df['pregnancy_length'].eq(10),

                df['pregnancy_length'].lt(7)]

    choices = ['9',

              '8',

               '10',

               'Early']

    df['pregnancy_length_cat'] = np.select(condition,choices, 'Late')



    #creating new column 'BMI_log':np.log(BMI) to normalize BMI

    df['BMI_log'] = np.log(df['BMI'])

    

    #creating new column 'first_pregnancy': Is it mother's first live birth? Yes: 1, No: 0

    df['first_live_birth'] = np.where(df['ILP_R'].eq(888), 1, 0)

    

    #creating new column 'first_natal': Is it mother's first natality event? Yes: 1, No: 0

    df['first_natal'] = np.where(df['ILOP_R'].eq(888),1, 0)

    

    #adjusting 'PRECARE' values: if Unkonwn, impute 0

    df['PRECARE'] = np.where(df['PRECARE'].eq(99), 0, df['PRECARE'])

    

    #adjusting 'PREVIS' values: if Unkonwn, impute 0

    df['PREVIS'] = np.where(df['PREVIS'].eq(99), 0, df['PREVIS'])

    

    #creating new column 'T35AGE_older': Is mother's age older than 34? Yes: 1, No: 0

    df['T35AGE_older'] = np.where(df['MAGER'].gt(34), 1, 0)

    

    #creating new column 'MOM_weight': manually computing mom's weight incase of missing values

    df['MOM_weight'] = (df['M_Ht_In']**2)*df['BMI']/704

    

    #creating new column '...': Weight gained divided by Mom's Weight

    df['WTGAIN_div_MOM_weight'] = df['WTGAIN']/df['MOM_weight']

    

    #creating new column '...': Weight gained divided by gestation period

    df['WTGAIN_div_length'] = df['WTGAIN'] / df['pregnancy_length']

    

    #creating new column '...': calculating percentage of weight gained due to pregnancy

    df['WT_percent_gain'] = df['WTGAIN'] / df['PWgt_R']

    

    #adjusting 'MAR_IMP': Marriage imputed should be 0 if left blank

    df['MAR_IMP'] = np.where(df['MAR_IMP'].eq(' '),0,1)

    

    #adjusting 'DMAR': assigning blanks to a new variable 0 for unknowns

    df['DMAR'] = np.where(df['DMAR'].eq(' '),0,df['DMAR'])

    

    #creating new column '...': 

    df['pregnancy_length_sqrt'] =  np.sqrt(df['pregnancy_length'])

    

    #dropping rows that with gestation period greater than 12 and less than 5, treating them as outliers

    df.drop(df[df['pregnancy_length'].gt(12)].index,inplace=True)

    df.drop(df[df['pregnancy_length'].lt(5)].index,inplace=True)

    return df



df = feature_engineer(df)

df.reset_index(inplace=True, drop=True)
X = df[['ATTEND','BFACIL', 'smoked', 'DOB_MM', 'DMAR','FHISPX','FEDUC', 'FRACE6', 'first_birth', 'plural_delivery', 'first_live_birth', 'first_natal','pregnancy_length_sqrt',

        'IP_GON', 'LD_INDL', 'MAGER', 'T35AGE_older','MAR_IMP', 'MBSTATE_REC', 'MEDUC', 'MHISPX', 'MRAVE6', 'MTRAN', 'pregnancy_length', 'WTGAIN_div_MOM_weight','WTGAIN_div_length',

        'NO_INFEC','NO_MMORB','NO_RISKS','PAY', 'PAY_REC','PRECARE','PREVIS', 'PRIORDEAD_cat', 'PRIORLIVE_cat', 'PRIORTERM_cat', 'PWgt_R', 'BMI_log','M_Ht_In', 'MOM_weight',

        'RDMETH_REC', 'RESTATUS', 'RF_CESAR', 'SEX', 'WTGAIN','WT_percent_gain','MAGER_cat','pregnancy_length_cat','BMI'

]]



_X = pd.get_dummies(X, columns=['ATTEND', 'BFACIL','smoked', 'DOB_MM','DMAR','FHISPX','FEDUC','FRACE6', 'plural_delivery',

                                'IP_GON','LD_INDL', 'T35AGE_older','MAR_IMP', 'MBSTATE_REC', 'MEDUC', 'MHISPX', 'MRAVE6', 'MTRAN',

                                'NO_INFEC','NO_MMORB','NO_RISKS', 'PAY', 'PAY_REC','PRIORDEAD_cat', 'PRIORLIVE_cat','PRIORTERM_cat',

                                'RDMETH_REC', 'RESTATUS', 'RF_CESAR', 'SEX','MAGER_cat','pregnancy_length_cat']).copy()

y = df['DBWT']

_X.shape
def feat_eng_dummy(_X):

    _X['MAGER_smoked_Daily'] = _X['MAGER'] * _X['smoked_Daily']

    _X['NO_RISKS_1_length'] = _X['NO_RISKS_1'] * _X['pregnancy_length']

    _X['RDMETH_REC_3_length'] = _X['RDMETH_REC_3'] * _X['pregnancy_length']

    _X['RDMETH_REC_1_length'] = _X['RDMETH_REC_1'] * _X['pregnancy_length']

    _X['ATTEND_1_length'] = _X['pregnancy_length'] * _X['ATTEND_1']

    _X['MRAVE6_1_FRACE6_1'] = _X['MRAVE6_1'] * _X['FRACE6_1']

    _X['BFACIL_1_length'] = _X['pregnancy_length'] * _X['BFACIL_1']

    _X['BMI_log_length'] = _X['BMI_log'] * _X['pregnancy_length']

    _X['M_Ht_In_length'] = _X['DMAR_1'] * _X['pregnancy_length']

    _X['LD_INDL_N_length'] = _X['LD_INDL_N'] * _X['pregnancy_length']

    _X['MTRAN_Y_length'] = _X['MTRAN_Y'] * _X['pregnancy_length']

    _X['PRECARE_length'] = _X['PRECARE'] * _X['pregnancy_length']

    _X['PREVIS_length'] = _X['PREVIS'] * _X['pregnancy_length']

    _X['MOM_weight_length'] = _X['MOM_weight'] * _X['pregnancy_length']

    _X['RDMETH_REC_3_pregnancy_length_cat_9'] = _X['RDMETH_REC_3'] * _X['pregnancy_length_cat_9']

    _X['RF_CESAR_Y_pregnancy_length_cat_9'] = _X['RF_CESAR_Y'] * _X['pregnancy_length_cat_9']

    

    return _X



_X = feat_eng_dummy(_X)


_X_columns = _X.columns

categorical_columns = []

continuous_columns = []

for i in _X_columns:

    if _X[i].max() == 1:

        categorical_columns.append(i)

    else:

        continuous_columns.append(i)

        

del _X_columns
_X['DBWT'] = y

plt.figure(figsize=(14,10))

sns.distplot(_X['DBWT'])

plt.title('Distribution plot of Baby Weight in Grams')

plt.xlabel('Baby Weight in Grams')

plt.savefig('figure1.png');
plt.figure(figsize=(14,10))

sns.distplot(_X[_X['SEX_M'].eq(0)]['DBWT'], label = 'Female')

sns.distplot(_X[_X['SEX_M'].eq(1)]['DBWT'], label = 'Male')

plt.title('Distribution of Baby Weight Separated by Gender')

plt.xlabel('Baby Weight in Grams')

plt.legend();
# null: Male Baby Weight = Female Baby Weight

# alt: Male Baby Weight != Female Baby Weight

# alpha: 0.05

stats.f_oneway(_X[_X['SEX_M'].eq(0)]['DBWT'],

              _X[_X['SEX_M'].eq(1)]['DBWT'])

# reject null. There is significant evidence to suggest that male babies weigh differently than females.
plt.figure(figsize=(14,10))

sns.distplot(_X[_X['smoked_Daily'].eq(1)]['DBWT'], label = 'Used to Smoke Daily')

sns.distplot(_X[_X['smoked_None'].eq(1)]['DBWT'], label = 'Used to Never Smoke')

plt.title('Distribution of Baby Weight Separated by Mother Smoked History')

plt.xlabel('Baby Weight in Grams')

plt.legend();
# null: Baby Weight of Moms who used to Smoke Daily = Baby Weight of Moms who never smoked

# alt: Baby Weight of Moms who used to Smoke Daily != Baby Weight of Moms who never smoked

# alpha: 0.05

stats.f_oneway(_X[_X['smoked_Daily'].eq(1)]['DBWT'],

              _X[_X['smoked_None'].eq(1)]['DBWT'])

# reject null. There is significant evidence that moms who used to smoke have different baby weights than those who never smoked.
fig, ax = plt.subplots(figsize=(14,10))

sns.boxplot(x='MRAVE6',y='DBWT', data=df, palette='muted')

ax.set_title('Distribution of Baby Weight Separated by Race')

ax.set_xlabel('Mother\'s Race')

ax.set_ylabel('Baby Weight in Grams')

ax.set_xticklabels(['White(only)','Black(only)','AIAN(only)','Asian(only)','NHOPI(only)','More than one race']);
# null: Baby Weights of Moms of different Race are equal

# alt: Baby Weights of Moms of different Race are NOT equal

# alpha: 0.05

stats.f_oneway(_X[_X['MRAVE6_1'].eq(1)]['DBWT'],

              _X[_X['MRAVE6_2'].eq(1)]['DBWT'],

              _X[_X['MRAVE6_3'].eq(1)]['DBWT'],

              _X[_X['MRAVE6_4'].eq(1)]['DBWT'],

              _X[_X['MRAVE6_5'].eq(1)]['DBWT'],

              _X[_X['MRAVE6_6'].eq(1)]['DBWT'])

# reject null. There is significant evidence to suggest that the all race babies are not the same.
plt.figure(figsize=(14,10))

sns.boxplot(x='pregnancy_length',y='DBWT',data=df, palette = 'muted')

plt.title('Baby Weight depending on Total Months of Gestation')

plt.ylabel('Baby Weight by Grams')

plt.xlabel('Total Months of Gestation');
# null: Baby weights of all gestation periods are equal

# alt: Baby weights of all gestation periods are NOT equal

# alpha: 0.05

stats.f_oneway(df[df['pregnancy_length'].eq(5)]['DBWT'],

              df[df['pregnancy_length'].eq(6)]['DBWT'],

              df[df['pregnancy_length'].eq(7)]['DBWT'],

              df[df['pregnancy_length'].eq(8)]['DBWT'],

              df[df['pregnancy_length'].eq(9)]['DBWT'],

               df[df['pregnancy_length'].eq(10)]['DBWT'],

               df[df['pregnancy_length'].eq(11)]['DBWT'],

               df[df['pregnancy_length'].eq(12)]['DBWT'])

# reject null. There is significant evidence to suggest that that the length of gestation has an effect on baby weight
fig, ax = plt.subplots(figsize=(14,10))

sns.boxplot(x='pregnancy_length',y='DBWT',data=_X, hue = 'RDMETH_REC_3',palette = 'muted', ax=ax)

handles, _ = ax.get_legend_handles_labels()

ax.legend(loc='upper right', handles = handles, labels = ['No Cesar', 'Yes Cesar'])

ax.set_title('Baby Weight vs Total Months of Gestation with-without C-Section')

ax.set_ylabel('Baby Weight by Grams')

ax.set_xlabel('Total Months of Gestation');
_X['RDMETH_REC_3_pregnancy_length_cat_Early'] = _X['RDMETH_REC_3'] * _X['pregnancy_length_cat_Early']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

_X_columns = _X.columns

_X[continuous_columns] = scaler.fit_transform(_X[continuous_columns])

_X = pd.DataFrame(data=_X, columns = _X_columns)
_new_columns = [

#     'first_birth',

#  'first_pregnancy',

#  'first_natal',

 'MAGER',

#  'pregnancy_length',

#  'WTGAIN_div_MOM_weight',

 'WTGAIN_div_length',

 'PRECARE',

 'PREVIS',

#  'PWgt_R',

 'BMI_log',

 'M_Ht_In',

 'MOM_weight',

 'WTGAIN',

 'WT_percent_gain',

#  'BMI',

 'ATTEND_1',

#  'ATTEND_2',

 'ATTEND_3',

 'ATTEND_4',

#  'ATTEND_5',

#  'ATTEND_9',

#  'BFACIL_1',

 'BFACIL_2',

 'BFACIL_3',

#  'BFACIL_4',

 'BFACIL_5',

#  'BFACIL_6',

#  'BFACIL_7',

#  'BFACIL_9',

#  'smoked_Daily',

 'smoked_None',

#  'smoked_Unknown',

#  'DOB_MM_1',

#  'DOB_MM_2',

#  'DOB_MM_3',

#  'DOB_MM_4',

#  'DOB_MM_5',

#  'DOB_MM_6',

#  'DOB_MM_7',

#  'DOB_MM_8',

#  'DOB_MM_9',

#  'DOB_MM_10',

#  'DOB_MM_11',

#  'DOB_MM_12',

#  'DMAR_0',

#  'DMAR_1',

 'DMAR_2',

 'FHISPX_0',

#  'FHISPX_1',

#  'FHISPX_2',

#  'FHISPX_3',

#  'FHISPX_4',

#  'FHISPX_5',

#  'FHISPX_6',

#  'FHISPX_9',

#  'FEDUC_1',

#  'FEDUC_2',

#  'FEDUC_3',

#  'FEDUC_4',

#  'FEDUC_5',

 'FEDUC_6',

#  'FEDUC_7',

#  'FEDUC_8',

#  'FEDUC_9',

 'FRACE6_1',

#  'FRACE6_2',

#  'FRACE6_3',

 'FRACE6_4',

#  'FRACE6_5',

#  'FRACE6_6',

#  'FRACE6_9',

#  'plural_delivery_No',

 'plural_delivery_Yes',

#  'IP_GON_N',

#  'IP_GON_U',

#  'IP_GON_Y',

 'LD_INDL_N',

#  'LD_INDL_U',

#  'LD_INDL_Y',

#  'T35AGE_older_0',

 'T35AGE_older_1',

#  'MAR_IMP_1',

 'MBSTATE_REC_1',

#  'MBSTATE_REC_2',

#  'MBSTATE_REC_3',

 'MEDUC_1',

#  'MEDUC_2',

 'MEDUC_3',

 'MEDUC_4',

 'MEDUC_5',

 'MEDUC_6',

 'MEDUC_7',

#  'MEDUC_8',

#  'MEDUC_9',

 'MHISPX_0',

#  'MHISPX_1',

 'MHISPX_2',

#  'MHISPX_3',

#  'MHISPX_4',

#  'MHISPX_5',

 'MHISPX_6',

#  'MHISPX_9',

#  'MRAVE6_1',

 'MRAVE6_2',

#  'MRAVE6_3',

 'MRAVE6_4',

#  'MRAVE6_5',

 'MRAVE6_6',

#  'MTRAN_N',

#  'MTRAN_U',

 'MTRAN_Y',

 'NO_INFEC_0',

#  'NO_INFEC_1',

#  'NO_INFEC_9',

 'NO_MMORB_0',

#  'NO_MMORB_1',

#  'NO_MMORB_9',

 'NO_RISKS_0',

#  'NO_RISKS_1',

#  'NO_RISKS_9',

#  'PAY_1',

#  'PAY_2',

#  'PAY_3',

#  'PAY_4',

#  'PAY_5',

#  'PAY_6',

 'PAY_8',

#  'PAY_9',

 'PAY_REC_1',

#  'PAY_REC_2',

#  'PAY_REC_3',

#  'PAY_REC_4',

#  'PAY_REC_9',

#  'PRIORDEAD_cat_None',

#  'PRIORDEAD_cat_Unknown',

 'PRIORDEAD_cat_Yes',

#  'PRIORLIVE_cat_None',

#  'PRIORLIVE_cat_Unknown',

 'PRIORLIVE_cat_Yes',

#  'PRIORTERM_cat_None',

#  'PRIORTERM_cat_Unknown',

#  'PRIORTERM_cat_Yes',

#  'RDMETH_REC_1',

#  'RDMETH_REC_2',

 'RDMETH_REC_3',

#  'RDMETH_REC_4',

#  'RDMETH_REC_5',

#  'RDMETH_REC_6',

#  'RDMETH_REC_9',

 'RESTATUS_1',

#  'RESTATUS_2',

#  'RESTATUS_3',

#  'RESTATUS_4',

 'RF_CESAR_N',

#  'RF_CESAR_U',

#  'RF_CESAR_Y',

#  'SEX_F',

 'SEX_M',

#  'MAGER_cat_Adult',

 'MAGER_cat_Minor',

 'pregnancy_length_cat_10',

 'pregnancy_length_cat_8',

 'pregnancy_length_cat_9',

 'pregnancy_length_cat_Early',

#  'pregnancy_length_cat_Late',

 'MAGER_smoked_Daily',

 'NO_RISKS_1_length',

 'RDMETH_REC_3_length',

 'RDMETH_REC_1_length',

 'ATTEND_1_length',

 'MRAVE6_1_FRACE6_1',

 'BFACIL_1_length',

 'BMI_log_length',

#  'M_Ht_In_length',

 'LD_INDL_N_length',

 'MTRAN_Y_length',

 'PRECARE_length',

 'PREVIS_length',

 'MOM_weight_length',

 'RDMETH_REC_3_pregnancy_length_cat_9',

 'RF_CESAR_Y_pregnancy_length_cat_9',

 'pregnancy_length_sqrt',

'RDMETH_REC_3_pregnancy_length_cat_Early'

]


_X['DBWT'] = y

from statsmodels.formula.api import ols

formula = 'DBWT~'+'+'.join(_new_columns)

model = ols(formula=formula, data=_X).fit()

model.summary()
_X.drop(columns=['DBWT'],inplace=True)

del X

del df

_X[continuous_columns] = scaler.inverse_transform(_X[continuous_columns])
X_train, X_test, y_train, y_test = train_test_split(_X, y, random_state = 42, test_size=0.2)

scaler = StandardScaler()

X_train[continuous_columns] = scaler.fit_transform(X_train[continuous_columns])

X_test[continuous_columns] = scaler.transform(X_test[continuous_columns])
linreg = LinearRegression()

linreg.fit(X_train[_new_columns],y_train)

y_train_linreg = linreg.predict(X_train[_new_columns])

np.sqrt(metrics.mean_squared_error(y_train, y_train_linreg))
y_test_linreg = linreg.predict(X_test[_new_columns])

np.sqrt(metrics.mean_squared_error(y_test, y_test_linreg))
linreg.score(X_test[_new_columns],y_test)
lasso_cv = LassoCV(cv=5, random_state=42, verbose=1)

lasso_cv.fit(X_train[_new_columns], y_train)

y_train_lasso = lasso_cv.predict(X_train[_new_columns])

np.sqrt(metrics.mean_squared_error(y_train, y_train_lasso))
y_test_lasso = lasso_cv.predict(X_test[_new_columns])

np.sqrt(metrics.mean_squared_error(y_test, y_test_lasso))
lasso_cv.score(X_test[_new_columns],y_test)
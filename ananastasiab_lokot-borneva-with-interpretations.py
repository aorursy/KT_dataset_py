import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve,roc_auc_score

from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

from sklearn.compose import ColumnTransformer



from sklearn.ensemble import RandomForestClassifier

import xgboost

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from catboost import CatBoostClassifier, Pool

import datetime

import featuretools as ft

%matplotlib inline



import seaborn as sns



# import scorecardpy as sc
from pdpbox import pdp, get_dataset, info_plots

import lime

import lime.lime_tabular
import shap
data_train = pd.read_csv('/kaggle/input/hse-practical-ml-1/car_loan_train.csv',sep=',', decimal='.')

data_train
np.sum(data_train.isnull(),axis=0)
def extract_years_old(fmt_str, start_year=None):

    """pd.to_datetime(data['Date.of.Birth'], format='%d-%m-%y').apply(lambda x: x.year) didnt work correctly ad 68 returning 2068

    so invent bicycles"""

    year_now = start_year if start_year else datetime.datetime.now().year

    year_now = year_now // 100

    year = int(fmt_str.split('-')[2])

    if year > year_now:  # 1900

        return 100 + year_now - year

    else:  #

        return year_now - year
data_train['Employment.Type'] = np.where(pd.isnull(data_train['Employment.Type']),1,np.where(data_train['Employment.Type']=='Self employed',0,2))

data_train['AVERAGE.ACCT.AGE'] = pd.to_numeric(data_train['AVERAGE.ACCT.AGE'].str.split(' ',expand=True)[0].str.replace('yrs',''))*12+pd.to_numeric(data_train['AVERAGE.ACCT.AGE'].str.split(' ',expand=True)[1].str.replace('mon',''))

data_train['CREDIT.HISTORY.LENGTH'] = pd.to_numeric(data_train['CREDIT.HISTORY.LENGTH'].str.split(' ',expand=True)[0].str.replace('yrs',''))*12+pd.to_numeric(data_train['CREDIT.HISTORY.LENGTH'].str.split(' ',expand=True)[1].str.replace('mon',''))



data_train['Age'] = data_train['Date.of.Birth'].apply(extract_years_old)

data_train['DisbursalDate'] = pd.to_datetime(data_train['DisbursalDate'], 

                                       format='%d-%m-%y')



data_train['DisbursalDate_bin'] = np.where(data_train['DisbursalDate']<pd.Timestamp('2018-08-16 00:00:00'),1,

         np.where(data_train['DisbursalDate']<pd.Timestamp('2018-08-31 00:00:00'),2,

        np.where(data_train['DisbursalDate']<pd.Timestamp('2018-09-15 00:00:00'),3,

        np.where(data_train['DisbursalDate']<pd.Timestamp('2018-09-30 00:00:00'),4,

        np.where(data_train['DisbursalDate']<pd.Timestamp('2018-10-15 00:00:00'),5,

       6)))))

data_train['no_score_f'] = np.where(data_train['PERFORM_CNS.SCORE'] >20,0,1)

data_train['coborrower_f'] = np.where((data_train['SEC.NO.OF.ACCTS']>0),1,0)

data_train['has_ch_f'] =np.where((data_train['PRI.NO.OF.ACCTS']>0),1,0)
#основного заемщика и созаемщика схлопываем в одного

data_train['NO.OF.ACCTS_total'] = data_train['PRI.NO.OF.ACCTS']+data_train['SEC.NO.OF.ACCTS']

data_train['ACTIVE.ACCTS_total'] = data_train['PRI.ACTIVE.ACCTS']+data_train['SEC.ACTIVE.ACCTS']

data_train['OVERDUE.ACCTS_total'] = data_train['PRI.OVERDUE.ACCTS']+data_train['SEC.OVERDUE.ACCTS']

data_train['CURRENT.BALANCE_total'] = data_train['PRI.CURRENT.BALANCE']+data_train['SEC.CURRENT.BALANCE']

data_train['SANCTIONED.AMOUNT_total'] = data_train['PRI.SANCTIONED.AMOUNT']+data_train['SEC.SANCTIONED.AMOUNT']

data_train['DISBURSED.AMOUNT_total'] = data_train['PRI.DISBURSED.AMOUNT']+data_train['SEC.DISBURSED.AMOUNT']

data_train['INSTAL.AMT_total'] = data_train['PRIMARY.INSTAL.AMT']+data_train['SEC.INSTAL.AMT']



#выделяем маленькие суммы

data_train['INSTAL.AMT_total_bin'] = ((data_train['INSTAL.AMT_total']<20000)

                                                & (data_train['INSTAL.AMT_total']>0)).astype(int)

#доп фичи по возрасту

data_train['Age_bin'] = ((data_train['Age']<23)

                                                & (data_train['Age']>47)).astype(int)

data_train['Age_sqrt'] = data_train['Age'] ** 2
#доля активных, просроченных кредитов в КИ

#соотношения сумм

data_train['ACTIVE_loans_share'] = np.where(data_train['NO.OF.ACCTS_total'] >0,data_train['ACTIVE.ACCTS_total'] / data_train['NO.OF.ACCTS_total'],-100)

data_train['OVERDUE_loans_share'] = np.where(data_train['NO.OF.ACCTS_total'] >0,data_train['OVERDUE.ACCTS_total'] / data_train['NO.OF.ACCTS_total'],-100)

data_train['OVERDUE2ACT_loans_share'] = np.where(data_train['ACTIVE.ACCTS_total'] >0,data_train['OVERDUE.ACCTS_total']/data_train['ACTIVE.ACCTS_total'],-100)

data_train['Curr2disb_loans_share'] = np.where(data_train['disbursed_amount'] >0,data_train['CURRENT.BALANCE_total']/data_train['disbursed_amount'],-100)

data_train['disb2sanct_loans_share'] = np.where(data_train['SANCTIONED.AMOUNT_total'] >0,data_train['disbursed_amount']/data_train['SANCTIONED.AMOUNT_total'],-100)

data_train['Curr2disb_loans_share'] = np.where(data_train['disbursed_amount'] >0,data_train['CURRENT.BALANCE_total']/data_train['disbursed_amount'],-100)

data_train['curr2sanct_loans_share'] = np.where(data_train['SANCTIONED.AMOUNT_total'] >0,data_train['CURRENT.BALANCE_total']/data_train['SANCTIONED.AMOUNT_total'],-100)

#доля свежих, просроченных кредитов в КИ

data_train['new2old_accounts_total'] = np.where(data_train['NO.OF.ACCTS_total'] >0,data_train['NEW.ACCTS.IN.LAST.SIX.MONTHS']/data_train['NO.OF.ACCTS_total'],-100)

data_train['del2old_accounts_total'] = np.where(data_train['NO.OF.ACCTS_total'] >0,data_train['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']/data_train['NO.OF.ACCTS_total'],-100)

data_train['del2new_accounts_total'] = np.where(data_train['NEW.ACCTS.IN.LAST.SIX.MONTHS'] >0,data_train['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']/data_train['NEW.ACCTS.IN.LAST.SIX.MONTHS'],-100)

data_train['actterm2credhist_accounts_total'] = np.where(data_train['CREDIT.HISTORY.LENGTH'] >0,data_train['AVERAGE.ACCT.AGE']/data_train['CREDIT.HISTORY.LENGTH'],-100)

#No Bureau History Available

data_train['no_bki_f'] =np.where(data_train['PERFORM_CNS.SCORE.DESCRIPTION']== 'No Bureau History Available',1,0)
# #WOE биннинг для id

# bins_supplier_id = sc.woebin(pd.concat([data_train['supplier_id'], data_train['target']], axis=1),y='target')

# bins_state_ID = sc.woebin(pd.concat([data_train['State_ID'], data_train['target']], axis=1),y='target')

# bins_branch_id = sc.woebin(pd.concat([data_train['branch_id'], data_train['target']], axis=1),y='target')
# data_train['supplier_id'] = sc.woebin_ply(pd.DataFrame(data_train['supplier_id']), bins_supplier_id)

# data_train['State_ID'] = sc.woebin_ply(pd.DataFrame(data_train['State_ID']), bins_state_ID)

# data_train['branch_id'] = sc.woebin_ply(pd.DataFrame(data_train['branch_id']), bins_branch_id)
#disbursed_amount и magic

data_train['dd0'] = data_train['magic_0']/data_train['disbursed_amount']

data_train['dd1'] = data_train['magic_1']/data_train['disbursed_amount']

data_train['dd2'] = data_train['magic_2']/data_train['disbursed_amount']

data_train['dd3'] = data_train['magic_3']/data_train['disbursed_amount']

data_train['dd4'] = data_train['magic_4']/data_train['disbursed_amount']

data_train['dd5'] = data_train['magic_5']/data_train['disbursed_amount']



data_train['m0m1'] = data_train['magic_1'] / data_train['magic_0']

data_train['m1m2'] = data_train['magic_2'] / data_train['magic_1']

data_train['m2m3'] = data_train['magic_3'] / data_train['magic_2']

data_train['m3m4'] = data_train['magic_4'] / data_train['magic_3']

data_train['m4m5'] = data_train['magic_5'] / data_train['magic_4']



data_train['d0'] = np.where(data_train['magic_0'] > data_train['disbursed_amount'],1,0)

data_train['d1'] = np.where(data_train['magic_1'] > data_train['disbursed_amount'],1,0)

data_train['d2'] = np.where(data_train['magic_2'] > data_train['disbursed_amount'],1,0)

data_train['d3'] = np.where(data_train['magic_3'] > data_train['disbursed_amount'],1,0)

data_train['d4'] = np.where(data_train['magic_4'] > data_train['disbursed_amount'],1,0)

data_train['d5'] = np.where(data_train['magic_5'] > data_train['disbursed_amount'],1,0)



data_train['m1'] = np.where(data_train['magic_1'] > data_train['magic_0'],1,0)

data_train['m2'] = np.where(data_train['magic_2'] > data_train['magic_1'],1,0)

data_train['m3'] = np.where(data_train['magic_3'] > data_train['magic_2'],1,0)

data_train['m4'] = np.where(data_train['magic_4'] > data_train['magic_3'],1,0)

data_train['m5'] = np.where(data_train['magic_5'] > data_train['magic_4'],1,0)



data_train['d_sum'] = data_train['d0']+data_train['d1']+data_train['d2']+data_train['d3']+data_train['d4']+data_train['d5']

data_train['m_sum'] = data_train['m1']+data_train['m2']+data_train['m3']+data_train['m4']+data_train['m5']



#f

data_train['f1_ltv'] = data_train['f1']*100/data_train['ltv']

data_train['f2_ltv'] = data_train['f2']/data_train['asset_cost']
#группы переменных

#'PERFORM_CNS.SCORE','target',



#applications data

g1 = ['UniqueID',  'branch_id',

       'supplier_id', 'manufacturer_id', 'Current_pincode_ID',

       'Employment.Type', 'State_ID', 'Employee_code_ID', 'Aadhar_flag',

       'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag',

         'Age','DisbursalDate_bin',  'coborrower_f'

        ]

#BCH data

g2 = [ 'NEW.ACCTS.IN.LAST.SIX.MONTHS',

       'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'AVERAGE.ACCT.AGE',

       'CREDIT.HISTORY.LENGTH', 'NO.OF_INQUIRIES','no_score_f','has_ch_f',

       'NO.OF.ACCTS_total', 'ACTIVE.ACCTS_total', 'OVERDUE.ACCTS_total',

       'CURRENT.BALANCE_total', 'SANCTIONED.AMOUNT_total',

       'DISBURSED.AMOUNT_total', 'INSTAL.AMT_total']

g22 = ['ACTIVE_loans_share',

       'OVERDUE_loans_share', 'OVERDUE2ACT_loans_share',

       'Curr2disb_loans_share', 'disb2sanct_loans_share',

       'curr2sanct_loans_share', 'new2old_accounts_total',

       'del2old_accounts_total', 'del2new_accounts_total',

       'actterm2credhist_accounts_total']



#Loan data

g3 = ['disbursed_amount', 'asset_cost', 'ltv', 'magic_0', 'magic_1',

       'magic_2', 'magic_3', 'magic_4', 'magic_5', 'f1', 'f2']

g33 = ['dd0', 'dd1', 'dd2', 'dd3', 'dd4',

       'dd5', 'm0m1', 'm1m2', 'm2m3', 'm3m4', 'm4m5', 'd0', 'd1', 'd2', 'd3',

       'd4', 'd5', 'm1', 'm2', 'm3', 'm4', 'm5', 'd_sum', 'm_sum', 'f1_ltv',

       'f2_ltv']
from sklearn.base import BaseEstimator, TransformerMixin



# Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features



class TargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns, noise_level = 0):

        self.columns = columns

        self.maps = {}

        self.noise_level = noise_level



    def fit(self, X, y):

        for col in self.columns:

            self.maps[col] = self.target_encode(trn_series = X[col],

                                                target = y)

        return self



    def transform(self, X):

        for col in self.columns:

            trn_series = X[col]

            averages = self.maps[col]

            ft_trn_series = pd.merge(

                trn_series.to_frame(trn_series.name),

                averages,

                left_on = trn_series.name,

                right_index = True,

                how='left')['target'].rename(trn_series.name + '_mean').fillna(0)

            # pd.merge does not keep the index so restore it

            ft_trn_series.index = trn_series.index

            X[col] =  self.add_noise(ft_trn_series, self.noise_level)

        return X



    def add_noise(self, series, noise_level):

        return series * (1 + noise_level * np.random.randn(len(series)))



    def target_encode(self,

                      trn_series=None,

                      target=None,

                      min_samples_leaf=1,

                      smoothing=1):

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

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'})

        return averages

        # Apply averages to trn and tst series
data_train_counter = data_train.copy().drop(columns=['target'])

target = data_train['target']
TE = TargetEncoder(columns=['branch_id','supplier_id', 'manufacturer_id', 'Employee_code_ID'], noise_level = 0.01)
#drop 'UniqueID','Date.of.Birth','DisbursalDate', 'MobileNo_Avl_Flag', 

cols2drop = ['Date.of.Birth','DisbursalDate', 

                                  'MobileNo_Avl_Flag'

                                  , 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',

                                   'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',

                                   'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',

                                   'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',

                                   'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',

                'PERFORM_CNS.SCORE.DESCRIPTION']

data_train_sel = data_train.drop(columns=cols2drop)

TE.fit(data_train_sel.drop(columns=['target']), data_train_sel['target'])

data_train_sel = TE.transform(data_train_sel)
data_train_sel = data_train.drop(columns=cols2drop)
fig, ax = plt.subplots(int(np.ceil(data_train_sel.columns.size/2)),2, figsize=(10,60))

ax = ax.flatten()

for i, feature in enumerate(data_train_sel.columns.values):

    sns.kdeplot(data_train_sel[data_train_sel['target']==0][feature], bw=0.5, ax=ax[i])    

    sns.kdeplot(data_train_sel[data_train_sel['target']==1][feature], bw=0.5, ax=ax[i])

plt.show()
corr = data_train_sel.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



fig, ax = plt.subplots(figsize=(22, 18))

sns.heatmap(data_train_sel.corr(), mask=mask, cmap='viridis', center=0, square=True, cbar_kws={"shrink": .5})

plt.show()
data_test = pd.read_csv('/kaggle/input/hse-practical-ml-1/car_loan_test.csv',sep=',', decimal='.')
data_test['Employment.Type'] = np.where(pd.isnull(data_test['Employment.Type']),1,np.where(data_test['Employment.Type']=='Self employed',0,2))



data_test['AVERAGE.ACCT.AGE'] = pd.to_numeric(data_test['AVERAGE.ACCT.AGE'].str.split(' ',expand=True)[0].str.replace('yrs',''))*12+pd.to_numeric(data_test['AVERAGE.ACCT.AGE'].str.split(' ',expand=True)[1].str.replace('mon',''))

data_test['CREDIT.HISTORY.LENGTH'] = pd.to_numeric(data_test['CREDIT.HISTORY.LENGTH'].str.split(' ',expand=True)[0].str.replace('yrs',''))*12+pd.to_numeric(data_test['CREDIT.HISTORY.LENGTH'].str.split(' ',expand=True)[1].str.replace('mon',''))



data_test['DisbursalDate'] = pd.to_datetime(data_test['DisbursalDate'],format='%d-%m-%y')

# data_train['age'] = round((data_train['DisbursalDate'] - data_train['Date.of.Birth'])/np.timedelta64(1,'Y'),0)

data_test['Age'] = data_test['Date.of.Birth'].apply(extract_years_old)
data_test['DisbursalDate_bin'] = np.where(data_test['DisbursalDate']<pd.Timestamp('2018-08-16 00:00:00'),1,

         np.where(data_test['DisbursalDate']<pd.Timestamp('2018-08-31 00:00:00'),2,

        np.where(data_test['DisbursalDate']<pd.Timestamp('2018-09-15 00:00:00'),3,

        np.where(data_test['DisbursalDate']<pd.Timestamp('2018-09-30 00:00:00'),4,

        np.where(data_test['DisbursalDate']<pd.Timestamp('2018-10-15 00:00:00'),5,

       6)))))
data_test['no_score_f'] = np.where(data_test['PERFORM_CNS.SCORE'] >20,0,1)

data_test['coborrower_f'] = np.where((data_test['SEC.NO.OF.ACCTS']>0),1,0)

data_test['has_ch_f'] =np.where((data_test['PRI.NO.OF.ACCTS']>0),1,0)
data_test['DisbursalDate_bin'] = np.where(data_test['DisbursalDate']<pd.Timestamp('2018-08-16 00:00:00'),1,

         np.where(data_test['DisbursalDate']<pd.Timestamp('2018-08-31 00:00:00'),2,

        np.where(data_test['DisbursalDate']<pd.Timestamp('2018-09-15 00:00:00'),3,

        np.where(data_test['DisbursalDate']<pd.Timestamp('2018-09-30 00:00:00'),4,

        np.where(data_test['DisbursalDate']<pd.Timestamp('2018-10-15 00:00:00'),5,

       6)))))



data_test['no_score_f'] = np.where(data_test['PERFORM_CNS.SCORE'] >20,0,1)



data_test['coborrower_f'] = np.where((data_test['SEC.NO.OF.ACCTS']>0),1,0)

data_test['has_ch_f'] =np.where((data_test['PRI.NO.OF.ACCTS']>0),1,0)



data_test['NO.OF.ACCTS_total'] = data_test['PRI.NO.OF.ACCTS']+data_test['SEC.NO.OF.ACCTS']

data_test['ACTIVE.ACCTS_total'] = data_test['PRI.ACTIVE.ACCTS']+data_test['SEC.ACTIVE.ACCTS']

data_test['OVERDUE.ACCTS_total'] = data_test['PRI.OVERDUE.ACCTS']+data_test['SEC.OVERDUE.ACCTS']

data_test['CURRENT.BALANCE_total'] = data_test['PRI.CURRENT.BALANCE']+data_test['SEC.CURRENT.BALANCE']

data_test['SANCTIONED.AMOUNT_total'] = data_test['PRI.SANCTIONED.AMOUNT']+data_test['SEC.SANCTIONED.AMOUNT']

data_test['DISBURSED.AMOUNT_total'] = data_test['PRI.DISBURSED.AMOUNT']+data_test['SEC.DISBURSED.AMOUNT']

data_test['INSTAL.AMT_total'] = data_test['PRIMARY.INSTAL.AMT']+data_test['SEC.INSTAL.AMT']



data_test['INSTAL.AMT_total_bin'] = ((data_test['INSTAL.AMT_total']<20000)

                                                & (data_test['INSTAL.AMT_total']>0)).astype(int)

data_test['Age_bin'] = ((data_test['Age']<23)

                                                & (data_test['Age']>47)).astype(int)

data_test['Age_sqrt'] = data_test['Age'] ** 2



data_test['ACTIVE_loans_share'] = np.where(data_test['NO.OF.ACCTS_total'] >0,data_test['ACTIVE.ACCTS_total'] / data_test['NO.OF.ACCTS_total'],-100)

data_test['OVERDUE_loans_share'] = np.where(data_test['NO.OF.ACCTS_total'] >0,data_test['OVERDUE.ACCTS_total'] / data_test['NO.OF.ACCTS_total'],-100)

data_test['OVERDUE2ACT_loans_share'] = np.where(data_test['ACTIVE.ACCTS_total'] >0,data_test['OVERDUE.ACCTS_total']/data_test['ACTIVE.ACCTS_total'],-100)

data_test['Curr2disb_loans_share'] = np.where(data_test['disbursed_amount'] >0,data_test['CURRENT.BALANCE_total']/data_test['disbursed_amount'],-100)

data_test['disb2sanct_loans_share'] = np.where(data_test['SANCTIONED.AMOUNT_total'] >0,data_test['disbursed_amount']/data_test['SANCTIONED.AMOUNT_total'],-100)

data_test['Curr2disb_loans_share'] = np.where(data_test['disbursed_amount'] >0,data_test['CURRENT.BALANCE_total']/data_test['disbursed_amount'],-100)

data_test['curr2sanct_loans_share'] = np.where(data_test['SANCTIONED.AMOUNT_total'] >0,data_test['CURRENT.BALANCE_total']/data_test['SANCTIONED.AMOUNT_total'],-100)



data_test['new2old_accounts_total'] = np.where(data_test['NO.OF.ACCTS_total'] >0,data_test['NEW.ACCTS.IN.LAST.SIX.MONTHS']/data_test['NO.OF.ACCTS_total'],-100)

data_test['del2old_accounts_total'] = np.where(data_test['NO.OF.ACCTS_total'] >0,data_test['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']/data_test['NO.OF.ACCTS_total'],-100)

data_test['del2new_accounts_total'] = np.where(data_test['NEW.ACCTS.IN.LAST.SIX.MONTHS'] >0,data_test['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS']/data_test['NEW.ACCTS.IN.LAST.SIX.MONTHS'],-100)

data_test['actterm2credhist_accounts_total'] = np.where(data_test['CREDIT.HISTORY.LENGTH'] >0,data_test['AVERAGE.ACCT.AGE']/data_test['CREDIT.HISTORY.LENGTH'],-100)



data_test['no_bki_f'] =np.where(data_test['PERFORM_CNS.SCORE.DESCRIPTION']== 'No Bureau History Available',1,0)
data_test['ACTIVE_loans_share'] = np.where(data_test['NO.OF.ACCTS_total'] >0,data_test['ACTIVE.ACCTS_total'] / data_test['NO.OF.ACCTS_total'],-100)

data_test['OVERDUE_loans_share'] = np.where(data_test['NO.OF.ACCTS_total'] >0,data_test['OVERDUE.ACCTS_total'] / data_test['NO.OF.ACCTS_total'],-100)

data_test['OVERDUE2ACT_loans_share'] = np.where(data_test['ACTIVE.ACCTS_total'] >0,data_test['OVERDUE.ACCTS_total']/data_test['ACTIVE.ACCTS_total'],-100)

data_test['Curr2disb_loans_share'] = np.where(data_test['disbursed_amount'] >0,data_test['CURRENT.BALANCE_total']/data_test['disbursed_amount'],-100)

data_test['disb2sanct_loans_share'] = np.where(data_test['SANCTIONED.AMOUNT_total'] >0,data_test['disbursed_amount']/data_test['SANCTIONED.AMOUNT_total'],-100)

data_test['Curr2disb_loans_share'] = np.where(data_test['disbursed_amount'] >0,data_test['CURRENT.BALANCE_total']/data_test['disbursed_amount'],-100)

data_test['curr2sanct_loans_share'] = np.where(data_test['SANCTIONED.AMOUNT_total'] >0,data_test['CURRENT.BALANCE_total']/data_test['SANCTIONED.AMOUNT_total'],-100)



# data_test.drop(['Date.of.Birth','DisbursalDate', 

#                                   'MobileNo_Avl_Flag'

#                                   , 'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',

#                                    'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',

#                                    'PRI.DISBURSED.AMOUNT', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',

#                                    'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',

#                                    'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',], axis=1, inplace=True)



# data_test['supplier_id'] = sc.woebin_ply(pd.DataFrame(data_test['supplier_id']), bins_supplier_id)

# data_test['State_ID'] = sc.woebin_ply(pd.DataFrame(data_test['State_ID']), bins_state_ID)

# data_test['branch_id'] = sc.woebin_ply(pd.DataFrame(data_test['branch_id']), bins_branch_id)



data_test['dd0'] = data_test['magic_0']/data_test['disbursed_amount']

data_test['dd1'] = data_test['magic_1']/data_test['disbursed_amount']

data_test['dd2'] = data_test['magic_2']/data_test['disbursed_amount']

data_test['dd3'] = data_test['magic_3']/data_test['disbursed_amount']

data_test['dd4'] = data_test['magic_4']/data_test['disbursed_amount']

data_test['dd5'] = data_test['magic_5']/data_test['disbursed_amount']



data_test['m0m1'] = data_test['magic_1'] / data_test['magic_0']

data_test['m1m2'] = data_test['magic_2'] / data_test['magic_1']

data_test['m2m3'] = data_test['magic_3'] / data_test['magic_2']

data_test['m3m4'] = data_test['magic_4'] / data_test['magic_3']

data_test['m4m5'] = data_test['magic_5'] / data_test['magic_4']



data_test['d0'] = np.where(data_test['magic_0'] > data_test['disbursed_amount'],1,0)

data_test['d1'] = np.where(data_test['magic_1'] > data_test['disbursed_amount'],1,0)

data_test['d2'] = np.where(data_test['magic_2'] > data_test['disbursed_amount'],1,0)

data_test['d3'] = np.where(data_test['magic_3'] > data_test['disbursed_amount'],1,0)

data_test['d4'] = np.where(data_test['magic_4'] > data_test['disbursed_amount'],1,0)

data_test['d5'] = np.where(data_test['magic_5'] > data_test['disbursed_amount'],1,0)



data_test['m1'] = np.where(data_test['magic_1'] > data_test['magic_0'],1,0)

data_test['m2'] = np.where(data_test['magic_2'] > data_test['magic_1'],1,0)

data_test['m3'] = np.where(data_test['magic_3'] > data_test['magic_2'],1,0)

data_test['m4'] = np.where(data_test['magic_4'] > data_test['magic_3'],1,0)

data_test['m5'] = np.where(data_test['magic_5'] > data_test['magic_4'],1,0)



data_test['d_sum'] = data_test['d0']+data_test['d1']+data_test['d2']+data_test['d3']+data_test['d4']+data_test['d5']

data_test['m_sum'] = data_test['m1']+data_test['m2']+data_test['m3']+data_test['m4']+data_test['m5']



data_test['f1_ltv'] = data_test['f1']*100/data_test['ltv']

data_test['f2_ltv'] = data_test['f2']/data_test['asset_cost']

data_test_sel = TE.transform(data_test).drop(columns=cols2drop)
class Clock:

    """Author thinks it works better than %%time"""

    start_time = None

    

    @staticmethod

    def start():

        Clock.start_time = datetime.datetime.now()

        

    @staticmethod

    def stop():

        print('Time:', datetime.datetime.now() - Clock.start_time)

        

from tqdm.auto import tqdm as _tqdm
X_train, X_test, y_train, y_test = train_test_split(data_train_sel.drop(columns='target'), data_train['target'], test_size=0.2, random_state=42)
metric = roc_auc_score

df = X_train

y = y_train
from sklearn.model_selection import KFold

folds = KFold(n_splits=5).split(df, y)

params = {}

params['app'] = 'binary'

params['learning_rate'] = 0.01

params['metric'] = 'auc'

params['seed'] = 0
def permutation_importance(df, y_true, model, metric, steps=1):

    imp = OrderedDict()

    base_pred = model.predict(df)

    base_score = metric(y_true, base_pred)

    for col in df.columns:

        imp[col] = 0

        saved = df[col].copy()

        for i in range(steps):

            df[col] = np.random.permutation(df[col])

            pred = model.predict(df)

            score = metric(y_true, pred)

            diff = base_score - score

            imp[col] += diff

            df[col] = saved.values

        imp[col] /= steps

    return imp
es = ft.EntitySet()



# add entities (application table itself)

es.entity_from_dataframe(

    entity_id='apps', # define entity id

    dataframe=df, # select underlying data

    index='SK_ID_CURR', # define unique index column

    # specify some datatypes manually (if needed)

    variable_types={

        f: ft.variable_types.Categorical 

        for f in df.columns if f.startswith('FLAG_')

    }

)
import lightgbm as lgb

from collections import OrderedDict

folds = KFold(n_splits=5).split(df, y)

fold_scores = []

split_imp = np.zeros(df.shape[1])

gain_imp = np.zeros(df.shape[1])

perm_imp = np.zeros(df.shape[1])



Clock.start()

for k, (tr, te) in _tqdm(enumerate(folds)):

    xtr, xte = df.iloc[tr, :], df.iloc[te, :]

    ytr, yte = y.iloc[tr], y.iloc[te]

    xtr = lgb.Dataset(xtr, label=ytr)

    xval = lgb.Dataset(xte, label=yte, reference=xtr)

    model = lgb.train(params,

                           xtr,

                           num_boost_round=10000,

                           valid_sets=[xtr, xval],

                           early_stopping_rounds=50,

                           feval=None,

                           verbose_eval=100)

    pred = model.predict(xte)

    fold_scores.append(metric(yte, pred))

    print('Fold {}: {:.5f}'.format(k, fold_scores[-1]))



    split_imp += model.feature_importance(importance_type='split')

    gain_imp += model.feature_importance(importance_type='gain')

    perm_imp += list(permutation_importance(xte, yte, model, metric).values())

print('Mean score: {:.5f}'.format(np.mean(fold_scores)))

Clock.stop()



split_imp /= 5

gain_imp /= 5

perm_imp /= 5



all_imp = pd.DataFrame(index=df.columns)

all_imp['permutation'] = perm_imp

all_imp['gain'] = gain_imp

all_imp['split'] = split_imp
all_imp = all_imp.sort_values(['permutation'])

all_imp['permutation_f']  = np.where(all_imp['permutation']>0.0001,1,0)

all_imp.nlargest(50,['gain','permutation','split']).index
best_cols = all_imp.nlargest(50,['gain','permutation','split']).index
from catboost import CatBoostClassifier



model = CatBoostClassifier()

# 'loss_function': ['Logloss','BrierScore','NormalizedGini'],

grid = {

        'learning_rate': [0.03, 0.1],

        'n_estimators': [100,500,1000],

        'depth': [4, 6, 8],

        }



grid_search_result = model.grid_search(grid, 

                                       X=X_train, 

                                       y=y_train, 

                                       plot=True)
model = CatBoostClassifier(learning_rate=0.03,

        n_estimators=1000,

        depth= 6)

# train the model

# start = time.time()

model.fit(data_train_sel.drop(columns=['target']), data_train_sel['target'])

# end = time.time()
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

scores = cross_val_score(model, data_train_sel.drop(columns=['target']), data_train_sel['target'],

                         cv=cv, scoring='roc_auc')
y_train_true = data_train_sel['target']

y_train_scores_cb = model.predict_proba(data_train_sel.drop(columns=['target']))[:,1]

print(roc_auc_score(y_train_true, y_train_scores_cb))

plt.hist(y_train_scores_cb)
my_model_features = list(data_train_sel)
shap_test = shap.TreeExplainer(model).shap_values(data_train_sel.drop(columns=['target']))

shap.summary_plot(shap_test, data_train_sel.drop(columns=['target']),

                      max_display=25, auto_size_plot=True)
pdp_PERFORM_CNS = pdp.pdp_isolate(

    model=model, dataset=data_train_sel, model_features=my_model_features, feature='PERFORM_CNS.SCORE'

)

fig, axes = pdp.pdp_plot(pdp_PERFORM_CNS, 'PERFORM_CNS.SCORE')
pdp_ltv = pdp.pdp_isolate(

    model=model, dataset=data_train_sel, model_features=my_model_features, feature='ltv'

)

fig, axes = pdp.pdp_plot(pdp_ltv, 'LTV')
pdp_manufacturer_id = pdp.pdp_isolate(

    model=model, dataset=data_train_sel, model_features=my_model_features, feature='manufacturer_id'

)

fig, axes = pdp.pdp_plot(pdp_manufacturer_id, 'manufacturer_id')
explainer = lime.lime_tabular.LimeTabularExplainer(data_train[my_model_features].astype(int).values,  

mode='classification',training_labels=data_train['target'],feature_names=my_model_features)
j = 0

exp = explainer.explain_instance(data_train_sel.values[j], model.predict_proba, num_features=10)

exp.show_in_notebook(show_table=True)
data_test_sel['SK_ID_CURR'] = data_test_sel.index + data_train_sel.shape[0]
test_predictions = model.predict_proba(data_test_sel)[:,1]



data_test_predict = pd.read_csv('/kaggle/input/hse-practical-ml-1/car_ss.csv',sep=',', decimal='.')

data_test_predict['Predicted'] = test_predictions

data_test_predict.to_csv('submission.csv', index = False)



test_predictions = model.predict_proba(data_test_sel)[:,1]

plt.hist(test_predictions)
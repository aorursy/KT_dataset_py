# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score,confusion_matrix

from sklearn import metrics

from sklearn.metrics import classification_report

from catboost import CatBoostClassifier,cv,Pool

from xgboost import XGBClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from vecstack import stacking

import seaborn as sns

from sklearn.preprocessing import Binarizer

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', 120)

pd.set_option('display.max_rows', 100)
train = pd.read_csv('../input/cax-mortgagemodeling/CAX_MortgageModeling_Train.csv')

test = pd.read_csv('../input/cax-mortgagemodeling/CAX_MortgageModeling_Test.csv')

dataset = train.append(test,sort = False)

dataset['RESULT'] = dataset['RESULT'].replace('FUNDED',1)

dataset['RESULT'] = dataset['RESULT'].replace('NOT FUNDED',0)

dataset = dataset.drop(['Unique_ID'],axis = 1)
dataset.shape
dataset.describe()
dataset.info()
dataset.isna().sum()
print("Total no. of Unique AMORTIZATION:", dataset['AMORTIZATION'].nunique())

print("Total no. of Unique RATE:", dataset['RATE'].nunique())

print("Total no. of Unique MORTGAGE PURPOSE:", dataset['MORTGAGE PURPOSE'].nunique())

print("Total no. of Unique PAYMENT FREQUENCY:", dataset['PAYMENT FREQUENCY'].nunique())

print("Total no. of Unique PROPERTY TYPE:",dataset['PROPERTY TYPE'].nunique())

print("Total no. of Unique TERM:", dataset['TERM'].nunique())

print("Total no. of Unique GENDER:", dataset['GENDER'].nunique())

print("Total no. of Unique AGE RANGE:", dataset['AGE RANGE'].nunique())

print("Total no. of Unique INCOME TYPE:", dataset['INCOME TYPE'].nunique())

print("Total no. of Unique NAICS CODE:", dataset['NAICS CODE'].nunique())

dataset.columns
plt.rcParams['figure.figsize'] = (18, 5)



plt.subplot(1, 3, 1)

sns.distplot(dataset['PROPERTY VALUE'],  color = 'orange')

plt.title('PROPERTY VALUE')



plt.subplot(1, 3, 2)

sns.distplot(dataset['MORTGAGE PAYMENT'], color = 'pink')

plt.title('MORTGAGE PAYMENT')



plt.subplot(1, 3, 3)

sns.distplot(dataset['MORTGAGE AMOUNT'], color = 'red')

plt.title('MORTGAGE AMOUNT')



plt.show()
plt.rcParams['figure.figsize'] = (18, 5)



plt.subplot(1, 3, 1)

sns.distplot(np.log(dataset['PROPERTY VALUE']),  color = 'orange')

plt.title('PROPERTY VALUE')



plt.subplot(1, 3, 2)

sns.distplot(np.log(dataset['MORTGAGE PAYMENT']), color = 'pink')

plt.title('MORTGAGE PAYMENT')



plt.subplot(1, 3, 3)

sns.distplot(np.log(dataset['MORTGAGE AMOUNT']), color = 'red')

plt.title('MORTGAGE AMOUNT')



plt.show()
plt.rcParams['figure.figsize'] = (18, 5)







plt.subplot(1, 3, 1)

sns.distplot(np.log(dataset['INCOME']), color = 'pink')

plt.title('LTV')



plt.subplot(1, 3, 2)

sns.distplot((dataset['CREDIT SCORE']), color = 'red')

plt.title('TDS')



plt.show()
sns.distplot(np.log1p(dataset['CREDIT SCORE']), color = 'red')

plt.title('TDS')



plt.show()
sns.countplot(dataset['AMORTIZATION'], palette = 'muted')

plt.title('AMORTIZATION',  fontsize = 30)
sns.countplot(dataset['RATE'], palette = 'muted')

plt.title('RATE',  fontsize = 30)
sns.countplot(dataset['MORTGAGE PURPOSE'], palette = 'muted')

plt.title('MORTGAGE PURPOSE',  fontsize = 30)
sns.countplot(dataset['PAYMENT FREQUENCY'], palette = 'muted')

plt.title('PAYMENT FREQUENCY',  fontsize = 30)
sns.countplot(dataset['PROPERTY TYPE'], palette = 'muted')

plt.title('PROPERTY TYPE',  fontsize = 30)
sns.countplot(dataset['TERM'], palette = 'muted')

plt.title('TERM',  fontsize = 30)
sns.countplot(dataset['GENDER'], palette = 'muted')

plt.title('GENDER',  fontsize = 30)
sns.countplot(dataset['AGE RANGE'], palette = 'muted')

plt.title('AGE RANGE',  fontsize = 30)
sns.countplot(dataset['INCOME TYPE'], palette = 'muted')

plt.title('INCOME TYPE',  fontsize = 30)
sns.countplot(dataset['NAICS CODE'], palette = 'muted')

plt.title('NAICS CODE',  fontsize = 30)
sns.countplot((train['RESULT']), color = 'red')

plt.title('RESULT')



plt.show()
rate_weights = []

for i in dataset['RATE']:

    if i < 4 :

        rate_weights.append(20)

    elif (i >= 4) & (i < 6):

        rate_weights.append(10)

    else:

        rate_weights.append(2)

        

dataset['Rate_weights'] = rate_weights



gds_weights = []

for i in dataset['GDS']:

    if i < 32 :

        gds_weights.append(20)

    elif (i >= 32) & (i < 39):

        gds_weights.append(10)

    else:

        gds_weights.append(1)

        

dataset['Gds_weights'] = gds_weights



tds_weights = []

for i in dataset['TDS']:

    if i < 40 :

        tds_weights.append(20)

    elif (i >= 40) & (i < 44):

        tds_weights.append(10)

    else:

        tds_weights.append(1)

        

dataset['Tds_weights'] = tds_weights



dataset['MORTGAGE PURPOSE'] = dataset['MORTGAGE PURPOSE'].replace('Purchase',1)

dataset['MORTGAGE PURPOSE'] = dataset['MORTGAGE PURPOSE'].replace('Refinance',0)



def creditscores(x):

    score = ''

    if (x >= 0) & (x < 300) :

        score = 'Uknown'

    elif (x >= 300) & (x < 575) :

        score = 'Poor'

    elif (x >= 575) & (x < 660) :

        score = 'Fair'

    elif (x >= 660) & (x < 690) :

        score = 'Average'

    elif (x >= 660) & (x < 740) :

        score = 'Good'

    else:

        score = 'Best'

    return score

 

    

fsa_province = []

for i in dataset['FSA']:

    fsa_province.append(i[0])



dataset['FSA_Province'] = fsa_province



fsa_area = []

for i in dataset['FSA']:

    fsa_area.append(int(i[1]))



dataset['FSA_Area'] = fsa_area

dataset['FSA_Area'] = dataset['FSA_Area'].astype(int)    

    

fsa_geo = []

for i in dataset['FSA']:

    fsa_geo.append(i[2])

    

dataset['FSA_Geo'] = fsa_geo



dataset['Credit_score_description'] = dataset['CREDIT SCORE'].apply(creditscores)



def territories(x):

    dummy = ''

    if x == 'K' or x =='L' or x =='M' or x =='N' or x =='P':

        dummy = 20

    elif x =='G'or x =='H' or x =='J' :

        dummy = 15

    elif x =='V' :

        dummy = 9

    elif x =='T' :

        dummy = 8

    elif x =='A' :

        dummy = 7

    elif x =='B' :

        dummy = 6

    elif x =='R' :

        dummy = 5

    elif x =='E' :

        dummy = 4

    elif x =='C' :

        dummy = 3

    elif x =='X' :

        dummy = 2

    elif x =='W' :

        dummy = 1

    else:

        dummy = -1

    return dummy





def rural_urban(x):

    if x == 0:

        return 'Rural'

    else :

        return 'Urban'



    

dataset['Territories'] = dataset['FSA_Province'].apply(territories)



dataset['FSA_Urban_Rural'] = dataset['FSA_Area'].apply(rural_urban)



dataset['Income_per_month'] = dataset['INCOME'] / 12

  

def not_willing(x):

    if x['MORTGAGE PURPOSE'] == 0:

        if x['MORTGAGE AMOUNT'] > x['PROPERTY VALUE']:

            return 'higher refinance'

        else:

            return 'lower refinance'

    elif x['MORTGAGE PURPOSE'] == 1:

        if x['MORTGAGE AMOUNT'] > x['PROPERTY VALUE']:

            return 'higher purchase'

        else:

            return 'lower purchase'

    

dataset['not_willing'] = dataset.apply(not_willing, axis=1)  





def can_pay(x):

    if x['Income_per_month'] > x['MORTGAGE PAYMENT']:

        if (x['Income_per_month'] * 30) / 100 > x['MORTGAGE PAYMENT']:

            return 'can defo pay'

        else:

            return 'can maybe pay'

    else:

        return 'cant pay'



dataset['can_pay'] = dataset.apply(can_pay, axis=1)
def refinance_amt(x):

    if x['MORTGAGE PURPOSE'] == 0 :

        return  x['PROPERTY VALUE'] * (int(x['GDS']) / 100)

    else:

        return 0.001



dataset['refinance_amt'] = dataset.apply(refinance_amt, axis=1)
dataset['other_debts'] = dataset['GDS'] - dataset['TDS']

dataset['act_amt'] = dataset['PROPERTY VALUE'] * (dataset['LTV']/100)

dataset['bins_prop'] = pd.qcut(dataset['PROPERTY VALUE'], 6, labels=[1,2,3,4,5,6])

dataset['bins_mortpay'] = pd.qcut(dataset['MORTGAGE PAYMENT'], 6, labels=[1,2,3,4,5,6])

dataset['bins_mortamt'] = pd.qcut(dataset['MORTGAGE AMOUNT'], 6, labels=[1,2,3,4,5,6])

dataset['bins_gds'] = pd.qcut(dataset['GDS'], 6, labels=[1,2,3,4,5,6])

dataset['bins_ltv'] = pd.qcut(dataset['LTV'], 6, labels=[1,2,3,4,5,6])

dataset['bins_tds'] = pd.qcut(dataset['TDS'], 6, labels=[1,2,3,4,5,6])

dataset['bins_income'] = pd.qcut(dataset['INCOME'], 6, labels=[1,2,3,4,5,6])



dataset['Income_used'] = round((dataset['MORTGAGE PAYMENT']*12) * dataset['INCOME'] /100)

dataset['Income_per_month'] = dataset['INCOME'] / 12

dataset['Income_per_month'] = pd.qcut(dataset['Income_per_month'],6, labels = [1,2,3,4,5,6])



dataset['GDS_amount_income'] = dataset['GDS'] * dataset['INCOME'] / 100



vals_to_replace = {'Monthly':1, 'Bi-Weekly':0, 'Bi-Weekly Accelerated':0,'Semi-Monthly':0,'Weekly Accelerated':0,

                   'Weekly':0,}



dataset['PAYMENT FREQUENCY'] = dataset['PAYMENT FREQUENCY'].map(vals_to_replace)



def month_int(x):

    if x['PAYMENT FREQUENCY'] == 1 : return(x['RATE']/12) * x['MORTGAGE AMOUNT']

    else:return 0.001



dataset['month_intrst'] = dataset.apply(month_int, axis=1)





def refinance_amt(x):

    if x['MORTGAGE PURPOSE'] == 0 :

        return(x['PROPERTY VALUE'] * x['GDS']) / 100

    else:

        return 0.001



dataset['refinance_amt'] = dataset.apply(refinance_amt, axis=1)





dataset['bins_prop'] = pd.qcut(dataset['PROPERTY VALUE'], 20, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

dataset['bins_mortpay'] = pd.qcut(dataset['MORTGAGE PAYMENT'], 20, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

dataset['bins_mortamt'] = pd.qcut(dataset['MORTGAGE AMOUNT'], 20, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

dataset['bins_gds'] = pd.qcut(dataset['GDS'], 20, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

dataset['bins_tds'] = pd.qcut(dataset['TDS'], 20, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

dataset['bins_income'] = pd.qcut(dataset['INCOME'], 20, labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])



dataset['gds_of_tds'] = dataset['TDS'] - dataset['GDS']

dataset['binned_gds_of_tds'] = pd.qcut(dataset['gds_of_tds'], 10, labels=[1,2,3,4,5,6,7,8,9,10])

dataset['prop_value_missingss'] = (dataset['PROPERTY VALUE'] - dataset['MORTGAGE AMOUNT']) / dataset['MORTGAGE AMOUNT']
dataset['prop_value_by_mortg'] = (dataset['PROPERTY VALUE'] / dataset['MORTGAGE PAYMENT']).astype('float32')

dataset['prop_value_by_gds'] = (dataset['PROPERTY VALUE'] / (dataset['GDS'] + 2)).astype('float32')

dataset['prop_value_by_LTV'] = (dataset['PROPERTY VALUE'] / dataset['LTV']).astype('float32')

dataset['prop_value_by_TDS'] = (dataset['PROPERTY VALUE'] / (dataset['TDS'] + 2)).astype('float32')

dataset['prop_value_by_amort'] = (dataset['PROPERTY VALUE'] / dataset['AMORTIZATION']).astype('float32')

dataset['prop_value_by_RATE'] = (dataset['PROPERTY VALUE'] / dataset['RATE']).astype('float32')

dataset['prop_value_by_TERM'] = (dataset['PROPERTY VALUE'] / dataset['TERM']).astype('float32')

dataset['prop_value_by_INCOME'] = (dataset['PROPERTY VALUE'] / dataset['INCOME']).astype('float32')

#dataset['prop_value_by_CREDIT SCORE'] = (dataset['PROPERTY VALUE'] / dataset['CREDIT SCORE']).astype('float32')



dataset['mort_payment_by_GDS'] = (dataset['MORTGAGE PAYMENT'] * dataset['GDS']).astype('float32')

dataset['mort_payment_by_LTV'] = (dataset['MORTGAGE PAYMENT'] * dataset['LTV']).astype('float32')

dataset['mort_payment_by_TDS'] = (dataset['MORTGAGE PAYMENT'] * dataset['TDS']).astype('float32')

dataset['mort_payment_by_AMORTIZATION'] = (dataset['MORTGAGE PAYMENT'] * dataset['AMORTIZATION']).astype('float32')

dataset['mort_payment_by_MORTGAGE AMOUNT'] = ((dataset['MORTGAGE PAYMENT'] *  dataset['MORTGAGE AMOUNT'])/100).astype('float32')

dataset['mort_payment_by_RATE'] = (dataset['MORTGAGE PAYMENT'] * dataset['RATE']).astype('float32')

dataset['mort_payment_by_RATE2'] = ((dataset['MORTGAGE PAYMENT'] * dataset['RATE'])/100).astype('float32')

dataset['mort_payment_by_TERM'] = (dataset['MORTGAGE PAYMENT'] / dataset['TERM']).astype('float32')

dataset['mort_payment_by_INCOME'] = ((dataset['MORTGAGE PAYMENT'] * dataset['INCOME'])/100).astype('float32')



dataset['GDS_by_PROPERTY'] = ((dataset['GDS'] * dataset['PROPERTY VALUE'])/100).astype('float32')

dataset['GDS_by_MORTGAGE'] = ((dataset['GDS'] * dataset['MORTGAGE PAYMENT'])/100).astype('float32')

dataset['GDS_by_LTV'] = (dataset['GDS'] + dataset['LTV']).astype('float32')

dataset['GDS_by_TDS'] = (dataset['GDS'] + dataset['TDS']).astype('float32')

dataset['all_three'] = (dataset['GDS'] + dataset['LTV'] + dataset['TDS']).astype('float32')

dataset['GDS_by_AMORTIZATION'] = (dataset['GDS'] * dataset['AMORTIZATION']).astype('float32')

dataset['GDS_by_MORTGAGE AMOUNT'] = ((dataset['GDS'] * dataset['MORTGAGE AMOUNT'])/100).astype('float32')

dataset['GDS_by_RATE'] = (dataset['GDS'] * dataset['RATE']).astype('float32')

dataset['GDS_by_TERM'] = (dataset['GDS'] * dataset['TERM']).astype('float32')

dataset['GDS_by_INCOME'] = ((dataset['GDS'] * dataset['INCOME'])/100).astype('float32')



dataset['LTV_by_PROPERTY'] = ((dataset['LTV'] * dataset['PROPERTY VALUE'])/100).astype('float32')

dataset['LTV_by_MORTGAGE'] = ((dataset['LTV'] * dataset['MORTGAGE PAYMENT'])/100).astype('float32')

dataset['LTV_by_AMORTIZATION'] = (dataset['LTV'] * dataset['AMORTIZATION']).astype('float32')

dataset['LTV_by_MORTGAGE AMOUNT'] = ((dataset['LTV'] * dataset['MORTGAGE AMOUNT'])/100).astype('float32')

dataset['LTV_by_RATE'] = (dataset['LTV'] * dataset['RATE']).astype('float32')

dataset['LTV_by_TERM'] = (dataset['LTV'] * dataset['TERM']).astype('float32')

dataset['LTV_by_INCOME'] = ((dataset['LTV'] * dataset['INCOME'])/100).astype('float32')



dataset['TDS_by_PROPERTY'] = ((dataset['TDS'] * dataset['PROPERTY VALUE'])/100).astype('float32')

dataset['TDS_by_MORTGAGE'] = ((dataset['TDS'] * dataset['MORTGAGE PAYMENT'])/100).astype('float32')

dataset['TDS_by_AMORTIZATION'] = (dataset['TDS'] * dataset['AMORTIZATION']).astype('float32')

dataset['TDS_by_MORTGAGE AMOUNT'] = ((dataset['TDS'] * dataset['MORTGAGE AMOUNT'])/100).astype('float32')

dataset['TDS_by_RATE'] = (dataset['TDS'] * dataset['RATE']).astype('float32')

dataset['TDS_by_TERM'] = (dataset['TDS'] * dataset['TERM']).astype('float32')

dataset['TDS_by_INCOME'] = ((dataset['TDS'] * dataset['INCOME'])/100).astype('float32')



dataset['AMORTIZATION_by_RATE'] = (dataset['AMORTIZATION'] * dataset['RATE']).astype('float32')

dataset['AMORTIZATION_by_TERM'] = (dataset['AMORTIZATION'] * dataset['TERM']).astype('float32')

dataset['AMORTIZATION_by_INCOME TYPE'] = (dataset['AMORTIZATION'] * dataset['INCOME TYPE']).astype('float32')



dataset['MORTGAGE AMOUNT_by_PROPERTY VALUE'] = (dataset['MORTGAGE AMOUNT'] - dataset['PROPERTY VALUE']).astype('float32')

dataset['MORTGAGE AMOUNT_by_mortg'] = (dataset['MORTGAGE AMOUNT'] / dataset['MORTGAGE PAYMENT']).astype('float32')

#dataset['MORTGAGE AMOUNT_by_gds'] = (dataset['MORTGAGE AMOUNT'] / dataset['GDS']).astype('float32')

#dataset['MORTGAGE AMOUNT_by_LTV'] = (dataset['MORTGAGE AMOUNT'] / dataset['LTV']).astype('float32')

#dataset['MORTGAGE AMOUNT_by_TDS'] = (dataset['MORTGAGE AMOUNT'] / dataset['TDS']).astype('float32')

dataset['MORTGAGE AMOUNT_by_amort'] = (dataset['MORTGAGE AMOUNT'] / dataset['AMORTIZATION']).astype('float32')

dataset['MORTGAGE AMOUNT_by_RATE'] = (dataset['MORTGAGE AMOUNT'] / dataset['RATE']).astype('float32')

dataset['MORTGAGE AMOUNT_by_TERM'] = (dataset['MORTGAGE AMOUNT'] / dataset['TERM']).astype('float32')

dataset['MORTGAGE AMOUNT_by_INCOME'] = (dataset['MORTGAGE AMOUNT'] / dataset['INCOME']).astype('float32')

#dataset['MORTGAGE AMOUNT_by_CREDIT SCORE'] = (dataset['MORTGAGE AMOUNT'] / dataset['CREDIT SCORE']).astype('float32')



dataset['RATE_by_PROPERTY VALUE'] = ((dataset['RATE'] * dataset['PROPERTY VALUE']) / 100).astype('float32')

dataset['RATE_by_MORTGAGE PAYMENT'] = ((dataset['RATE'] * dataset['MORTGAGE PAYMENT']) / 100).astype('float32')

dataset['RATE_by_MORTGAGE AMOUNT'] = ((dataset['RATE'] * dataset['MORTGAGE AMOUNT']) / 100).astype('float32')

dataset['RATE_by_TERM'] = (dataset['RATE'] * dataset['TERM']).astype('float32')

dataset['RATE_by_INCOME'] = ((dataset['RATE'] * dataset['INCOME']) / 100).astype('float32')



dataset['TERM_by_PROPERTY VALUE'] = ((dataset['TERM'] * dataset['PROPERTY VALUE']) / 100).astype('float32')

dataset['TERM_by_MORTGAGE PAYMENT'] = (dataset['TERM'] * dataset['MORTGAGE PAYMENT']).astype('float32')

dataset['TERM_by_MORTGAGE AMOUNT'] = ((dataset['TERM'] * dataset['MORTGAGE AMOUNT']) / 100).astype('float32')

dataset['TERM_by_INCOME'] = ((dataset['TERM'] * dataset['INCOME']) / 100).astype('float32')



dataset['Mort_by_amt'] = dataset['MORTGAGE AMOUNT'] / dataset['MORTGAGE PAYMENT']

new_train,new_test = dataset[:len(train)],dataset[len(train):]

new_train.RESULT = new_train.RESULT.replace(-1,0)
stats = new_train.groupby(['INCOME TYPE'], as_index=False)['INCOME'].agg('mean')

stats.rename(columns={'INCOME': 'mean_by_incomet_inc'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['INCOME TYPE'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['INCOME TYPE'])



def further_inc_type(x):

    if x['mean_by_incomet_inc'] > x['INCOME']:

        return x['INCOME TYPE']  + 2

    else:

        return x['INCOME TYPE']  + 2

new_train['further_inc_type'] = new_train.apply(further_inc_type, axis=1)

new_test['further_inc_type'] = new_test.apply(further_inc_type, axis=1)



def purpose_default(x):

    if x['mean_by_incomet_inc'] > x['INCOME']:

        return 'may not pay'

    else:

        return 'may pay'





new_train['mean_by_incomet_inc2'] = new_train.apply(purpose_default, axis=1)

new_test['mean_by_incomet_inc2'] = new_test.apply(purpose_default, axis=1)





stats = new_train.groupby(['Credit_score_description'], as_index=False)['MORTGAGE PAYMENT'].agg('mean')

stats.rename(columns={'MORTGAGE PAYMENT': 'credi_scor mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description'])



stats = new_train.groupby(['Credit_score_description'], as_index=False)['PROPERTY VALUE'].agg('mean')

stats.rename(columns={'PROPERTY VALUE': 'credi_scor mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description'])



stats = new_train.groupby(['Credit_score_description'], as_index=False)['MORTGAGE AMOUNT'].agg('mean')

stats.rename(columns={'MORTGAGE AMOUNT': 'credi_scor mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description'])



stats = new_train.groupby(['INCOME TYPE'], as_index=False)['MORTGAGE PAYMENT'].agg('mean')

stats.rename(columns={'MORTGAGE PAYMENT': 'incom_mort mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['INCOME TYPE'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['INCOME TYPE'])



stats = new_train.groupby(['INCOME TYPE'], as_index=False)['PROPERTY VALUE'].agg('mean')

stats.rename(columns={'PROPERTY VALUE': 'imcom_pro mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['INCOME TYPE'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['INCOME TYPE'])



stats = new_train.groupby(['INCOME TYPE'], as_index=False)['MORTGAGE AMOUNT'].agg('mean')

stats.rename(columns={'MORTGAGE AMOUNT': 'imcom_mort mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['INCOME TYPE'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['INCOME TYPE'])





stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['PROPERTY VALUE'].agg('mean')

stats.rename(columns={'PROPERTY VALUE': 'prop_value mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['PROPERTY VALUE'].agg('mean')

stats.rename(columns={'PROPERTY VALUE': 'prop_value mean2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['PROPERTY VALUE'].agg('mean')

stats.rename(columns={'PROPERTY VALUE': 'prop_value mean3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])





stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['PROPERTY VALUE'].agg('max')

stats.rename(columns={'PROPERTY VALUE': 'prop_value max1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['PROPERTY VALUE'].agg('max')

stats.rename(columns={'PROPERTY VALUE': 'prop_value max2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['PROPERTY VALUE'].agg('max')

stats.rename(columns={'PROPERTY VALUE': 'prop_value max3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['PROPERTY VALUE'].agg('min')

stats.rename(columns={'PROPERTY VALUE': 'prop_value min1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['PROPERTY VALUE'].agg('min')

stats.rename(columns={'PROPERTY VALUE': 'prop_value min2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['PROPERTY VALUE'].agg('min')

stats.rename(columns={'PROPERTY VALUE': 'prop_value min3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['MORTGAGE PAYMENT'].agg('mean')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['MORTGAGE PAYMENT'].agg('mean')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value mean2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['MORTGAGE PAYMENT'].agg('mean')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value mean3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])





stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['MORTGAGE PAYMENT'].agg('max')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value max1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['MORTGAGE PAYMENT'].agg('max')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value max2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['MORTGAGE PAYMENT'].agg('max')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value max3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['MORTGAGE PAYMENT'].agg('min')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value min1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['MORTGAGE PAYMENT'].agg('min')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value min2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['MORTGAGE PAYMENT'].agg('min')

stats.rename(columns={'MORTGAGE PAYMENT': 'mort_pay_value min3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['MORTGAGE AMOUNT'].agg('mean')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['MORTGAGE AMOUNT'].agg('mean')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value mean2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['MORTGAGE AMOUNT'].agg('mean')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value mean3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])





stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['MORTGAGE AMOUNT'].agg('max')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value max1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['MORTGAGE AMOUNT'].agg('max')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value max2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['MORTGAGE AMOUNT'].agg('max')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value max3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['MORTGAGE AMOUNT'].agg('min')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value min1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['MORTGAGE AMOUNT'].agg('min')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value min2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['MORTGAGE AMOUNT'].agg('min')

stats.rename(columns={'MORTGAGE AMOUNT': 'mort_amt_value min3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['INCOME'].agg('mean')

stats.rename(columns={'INCOME': 'income_amt_value mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['INCOME'].agg('mean')

stats.rename(columns={'INCOME': 'income_amt_value mean2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['INCOME'].agg('mean')

stats.rename(columns={'INCOME': 'income_amt_value mean3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])





stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['INCOME'].agg('max')

stats.rename(columns={'INCOME': 'income_amt_value max1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['INCOME'].agg('max')

stats.rename(columns={'INCOME': 'income_amt_value max2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['INCOME'].agg('max')

stats.rename(columns={'INCOME': 'income_amt_value max3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['INCOME'].agg('min')

stats.rename(columns={'INCOME': 'income_amt_value min1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['INCOME'].agg('min')

stats.rename(columns={'INCOME': 'income_amt_value min2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['INCOME'].agg('min')

stats.rename(columns={'INCOME': 'income_amt_value min3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['TDS'].agg('mean')

stats.rename(columns={'TDS': 'tds_amt_value mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['TDS'].agg('mean')

stats.rename(columns={'TDS': 'tds_amt_value mean2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['TDS'].agg('mean')

stats.rename(columns={'TDS': 'tds_amt_value mean3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])





stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['TDS'].agg('max')

stats.rename(columns={'TDS': 'tds_amt_value max1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['TDS'].agg('max')

stats.rename(columns={'TDS': 'tds_amt_value max2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['TDS'].agg('max')

stats.rename(columns={'TDS': 'tds_amt_value max3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['TDS'].agg('min')

stats.rename(columns={'TDS': 'tds_amt_value min1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['TDS'].agg('min')

stats.rename(columns={'TDS': 'tds_amt_value min2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['TDS'].agg('min')

stats.rename(columns={'TDS': 'tds_amt_value min3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['GDS'].agg('mean')

stats.rename(columns={'GDS': 'gds_amt_value mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['GDS'].agg('mean')

stats.rename(columns={'GDS': 'gds_amt_value mean2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['GDS'].agg('mean')

stats.rename(columns={'GDS': 'gds_amt_value mean3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])





stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['GDS'].agg('max')

stats.rename(columns={'GDS': 'gds_amt_value max1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['GDS'].agg('max')

stats.rename(columns={'GDS': 'gds_amt_value max2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['GDS'].agg('max')

stats.rename(columns={'GDS': 'gds_amt_value max3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['GDS'].agg('min')

stats.rename(columns={'GDS': 'gds_amt_value min1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['GDS'].agg('min')

stats.rename(columns={'GDS': 'gds_amt_value min2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['GDS'].agg('min')

stats.rename(columns={'GDS': 'gds_amt_value min3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['LTV'].agg('mean')

stats.rename(columns={'LTV': 'ltv_amt_value mean1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['LTV'].agg('mean')

stats.rename(columns={'LTV': 'ltv_amt_value mean2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['LTV'].agg('mean')

stats.rename(columns={'LTV': 'ltv_amt_value mean3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])





stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['LTV'].agg('max')

stats.rename(columns={'LTV': 'ltv_amt_value max1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['LTV'].agg('max')

stats.rename(columns={'LTV': 'ltv_amt_value max2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['LTV'].agg('max')

stats.rename(columns={'LTV': 'ltv_amt_value max3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])



stats = new_train.groupby(['FSA', 'GENDER'], as_index=False)['LTV'].agg('min')

stats.rename(columns={'LTV': 'ltv_amt_value min1'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['FSA','GENDER'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['FSA','GENDER'])



stats = new_train.groupby(['Credit_score_description','FSA_Province'], as_index=False)['LTV'].agg('min')

stats.rename(columns={'LTV': 'ltv_amt_value min2'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['Credit_score_description','FSA_Province'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['Credit_score_description','FSA_Province'])



stats = new_train.groupby(['PROPERTY TYPE','FSA_Urban_Rural'], as_index=False)['LTV'].agg('min')

stats.rename(columns={'LTV': 'ltv_amt_value min3'}, inplace=True)

new_train = pd.merge(stats,new_train,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])

new_test = pd.merge(new_test,stats,how = 'left', on = ['PROPERTY TYPE','FSA_Urban_Rural'])
attempt1 =  new_train.pivot_table(values=['INCOME'],

                    index=['GENDER','FSA_Urban_Rural'], aggfunc=np.mean)

attempt1.rename(columns={'INCOME': 'gen_urb_rur income'}, inplace=True)

new_train = pd.merge(attempt1,new_train,how = 'left', on = ['GENDER','FSA_Urban_Rural'])

new_test = pd.merge(attempt1,new_test,how = 'left', on = ['GENDER','FSA_Urban_Rural'])



attempt2 = new_train.pivot_table(values=['INCOME'],

                    index=['GENDER','FSA_Province'], aggfunc=np.mean)

attempt2.rename(columns={'INCOME': 'gen_fsa_prov income'}, inplace=True)

new_train = pd.merge(attempt2,new_train,how = 'left', on = ['GENDER','FSA_Province'])

new_test = pd.merge(attempt2,new_test,how = 'left', on = ['GENDER','FSA_Province'])



attempt3 = new_train.pivot_table(values=['INCOME'],

                    index=['GENDER','AGE RANGE'], aggfunc=np.mean)

attempt3.rename(columns={'INCOME': 'gen_age_range income'}, inplace=True)

new_train = pd.merge(attempt3,new_train,how = 'left', on = ['GENDER','AGE RANGE'])

new_test = pd.merge(attempt3,new_test,how = 'left', on = ['GENDER','AGE RANGE'])



attempt4 = new_train.pivot_table(values=['INCOME'],

                    index=['GENDER','Credit_score_description'], aggfunc=np.mean)

attempt4.rename(columns={'INCOME': 'gen_c income'}, inplace=True)

new_train = pd.merge(attempt4,new_train,how = 'left', on = ['GENDER','Credit_score_description'])

new_test = pd.merge(attempt4,new_test,how = 'left', on = ['GENDER','Credit_score_description'])



attempt5 = new_train.pivot_table(values=['INCOME'],

                    index=['GENDER','PROPERTY TYPE'], aggfunc=np.mean)

attempt5.rename(columns={'INCOME': 'gen_prop_type income'}, inplace=True)

new_train = pd.merge(attempt5,new_train,how = 'left', on = ['GENDER','PROPERTY TYPE'])

new_test = pd.merge(attempt5,new_test,how = 'left', on = ['GENDER','PROPERTY TYPE'])



new_train['gen_urb_income_diff'] = new_train['gen_urb_rur income'] - new_train['INCOME']

new_test['gen_urb_income_diff'] = new_test['gen_urb_rur income'] - new_test['INCOME']



new_train['gen_fsa_income_diff'] = new_train['gen_fsa_prov income'] - new_train['INCOME']

new_test['gen_fsa_income_diff'] = new_test['gen_fsa_prov income'] - new_test['INCOME']



new_train['gen_age_income_diff'] = new_train['gen_age_range income'] - new_train['INCOME']

new_test['gen_age_income_diff'] = new_test['gen_age_range income'] - new_test['INCOME']



new_train['gen_c_income_diff'] = new_train['gen_c income'] - new_train['INCOME']

new_test['gen_c_income_diff'] = new_test['gen_c income'] - new_test['INCOME']



new_train['gen_prop_income_diff'] = new_train['gen_prop_type income'] - new_train['INCOME']

new_test['gen_prop_income_diff'] = new_test['gen_prop_type income'] - new_test['INCOME']
labelencoder = LabelEncoder()

new_train['NAICS CODE'] = labelencoder.fit_transform(new_train['NAICS CODE'])

new_train['FSA'] = labelencoder.fit_transform(new_train['FSA']) 

new_train['PAYMENT FREQUENCY'] = labelencoder.fit_transform(new_train['PAYMENT FREQUENCY'])

new_train['FSA_Geo'] = labelencoder.fit_transform(new_train['FSA_Geo'])

new_train['FSA_Province'] = labelencoder.fit_transform(new_train['FSA_Province'])

new_train['PROPERTY TYPE'] = labelencoder.fit_transform(new_train['PROPERTY TYPE'])

new_train['AGE RANGE'] = labelencoder.fit_transform(new_train['AGE RANGE']) 

new_train['GENDER'] = labelencoder.fit_transform(new_train['GENDER'])

new_train['Credit_score_description'] = labelencoder.fit_transform(new_train['Credit_score_description'])

new_train['FSA_Urban_Rural'] = labelencoder.fit_transform(new_train['FSA_Urban_Rural'])

new_train['mean_by_incomet_inc2'] = labelencoder.fit_transform(new_train['mean_by_incomet_inc2'])





new_test['NAICS CODE'] = labelencoder.fit_transform(new_test['NAICS CODE'])

new_test['FSA'] = labelencoder.fit_transform(new_test['FSA']) 

new_test['PAYMENT FREQUENCY'] = labelencoder.fit_transform(new_test['PAYMENT FREQUENCY'])

new_test['FSA_Geo'] = labelencoder.fit_transform(new_test['FSA_Geo'])

new_test['FSA_Province'] = labelencoder.fit_transform(new_test['FSA_Province'])

new_test['PROPERTY TYPE'] = labelencoder.fit_transform(new_test['PROPERTY TYPE'])

new_test['AGE RANGE'] = labelencoder.fit_transform(new_test['AGE RANGE']) 

new_test['GENDER'] = labelencoder.fit_transform(new_test['GENDER'])

new_test['Credit_score_description'] = labelencoder.fit_transform(new_test['Credit_score_description'])

new_test['FSA_Urban_Rural'] = labelencoder.fit_transform(new_test['FSA_Urban_Rural'])

new_test['mean_by_incomet_inc2'] = labelencoder.fit_transform(new_test['mean_by_incomet_inc2'])
matrix = new_train.corr()

upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
new_train['bins_prop'] = labelencoder.fit_transform(new_train['bins_prop'])

new_train['bins_mortamt'] = labelencoder.fit_transform(new_train['bins_mortamt']) 

new_train['bins_gds'] = labelencoder.fit_transform(new_train['bins_gds'])

new_train['bins_ltv'] = labelencoder.fit_transform(new_train['bins_ltv'])

new_train['bins_tds'] = labelencoder.fit_transform(new_train['bins_tds'])

new_train['bins_income'] = labelencoder.fit_transform(new_train['bins_income'])

new_train['Income_per_month'] = labelencoder.fit_transform(new_train['Income_per_month']) 

new_train['binned_gds_of_tds'] = labelencoder.fit_transform(new_train['binned_gds_of_tds'])

new_train['bins_mortamt'] = labelencoder.fit_transform(new_train['bins_mortamt'])

new_train['not_willing'] = labelencoder.fit_transform(new_train['not_willing'])

new_train['can_pay'] = labelencoder.fit_transform(new_train['can_pay'])





new_test['bins_prop'] = labelencoder.fit_transform(new_test['bins_prop'])

new_test['bins_mortamt'] = labelencoder.fit_transform(new_test['bins_mortamt']) 

new_test['bins_gds'] = labelencoder.fit_transform(new_test['bins_gds'])

new_test['bins_ltv'] = labelencoder.fit_transform(new_test['bins_ltv'])

new_test['bins_tds'] = labelencoder.fit_transform(new_test['bins_tds'])

new_test['bins_income'] = labelencoder.fit_transform(new_test['bins_income'])

new_test['Income_per_month'] = labelencoder.fit_transform(new_test['Income_per_month']) 

new_test['binned_gds_of_tds'] = labelencoder.fit_transform(new_test['binned_gds_of_tds'])

new_test['bins_mortamt'] = labelencoder.fit_transform(new_test['bins_mortamt'])

new_test['not_willing'] = labelencoder.fit_transform(new_test['not_willing'])

new_test['can_pay'] = labelencoder.fit_transform(new_test['can_pay'])
new_train = new_train.drop(to_drop,axis = 1)

new_test = new_test.drop(to_drop,axis = 1)
'''from catboost import CatBoostClassifier,Pool



clf = CatBoostClassifier(iterations=2000,

                                loss_function = 'Logloss',

                                depth=5,

                                eval_metric = 'F1',

                                learning_rate = 0.09161,

                                scale_pos_weight = 0.5,

                                



        )



clf.fit(X_train.drop(to_drop,axis = 1),y_train, eval_set = [(X_val.drop(to_drop,axis = 1),y_val)], early_stopping_rounds = 100,

       )

        #cat_features = categor'''
'''y_pred_cat = clf.predict(X_val.drop(to_drop,axis = 1))



print("Training Accuracy: ", clf.score(X_train.drop(to_drop,axis = 1), y_train))

print('Testing Accuarcy: ', clf.score(X_val.drop(to_drop,axis = 1), y_val))



# making a classification report

cr = classification_report(y_val,  y_pred_cat)

print(cr)



print(f1_score(y_val, y_pred_cat, average='macro'))

print(confusion_matrix(y_val, y_pred_cat))'''
'''from lightgbm import LGBMClassifier

model_lgb = LGBMClassifier(max_depth=6, learning_rate=0.01, n_estimators=5000, scale_pos_weight = 25

                                        )

model_lgb.fit(X_res, y_res,  eval_set = [(lx_val,y_val)], early_stopping_rounds = 100,

         )



#y_pred_lgb = (model_lgb.predict_proba(new_test.drop(['RESULT','MORTGAGE NUMBER',],axis = 1))[:,1] >= 0.5).astype(bool)

y_pred_lgb = model_lgb.predict(lx_val)

print(f1_score(y_val, y_pred_lgb, average='macro',))

print(confusion_matrix(y_val,y_pred_lgb))



print("Training Accuracy: ", model_lgb.score(X_res, y_res))

print('Testing Accuarcy: ', model_lgb.score(lx_val, y_val))'''
'''from lightgbm import LGBMClassifier

model_lgb = LGBMClassifier(scale_pos_weight = 0.6,drop_rate=0.9, min_data_in_leaf=100, max_bin=255,

                                 n_estimators=500,min_sum_hessian_in_leaf=1,importance_type='gain',learning_rate=0.1,bagging_fraction = 0.85,

                                 colsample_bytree = 1.0,feature_fraction = 0.1,lambda_l1 = 5.0,lambda_l2 = 3.0,max_depth =  9,

                                 min_child_samples = 55,min_child_weight = 5.0,min_split_gain = 0.1,num_leaves = 45,subsample = 0.75)

model_lgb.fit(X_train, y_train,  eval_set = [(X_val,y_val)], early_stopping_rounds = 100,

         )



#y_pred_lgb = (model_lgb.predict_proba(lx_val)[:,1] >= 0.3).astype(bool)

y_pred_lgb = model_lgb.predict(X_val)

print(f1_score(y_val, y_pred_lgb, average='macro',))

print(confusion_matrix(y_val,y_pred_lgb))



print("Training Accuracy: ", model_lgb.score(X_train, y_train))

print('Testing Accuarcy: ', model_lgb.score(X_val, y_val))'''
'''import numpy as np

import pandas as pd

import gc

import time

from contextlib import contextmanager

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

'''
'''def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data

    train_df = df[df['RESULT'].notnull()]

    test_df = df[df['RESULT'].isnull()]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    del df

    gc.collect()

    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)

    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['RESULT','MORTGAGE NUMBER','Unique_ID']]

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['RESULT'])):

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['RESULT'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['RESULT'].iloc[valid_idx]



        # LightGBM parameters found by Bayesian optimization

        clf = LGBMClassifier(scale_pos_weight =0.4,

            nthread=4,

            n_estimators=10000,

            learning_rate=0.02,

            num_leaves=34,

            colsample_bytree=0.9497036,

            subsample=0.8715623,

            max_depth=8,

            reg_alpha=0.041545473,

            reg_lambda=0.0735294,

            min_split_gain=0.0222415,

            min_child_weight=39.3259775,

            silent=-1,

            verbose=-1, )



        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

            eval_metric= 'F1', verbose= 200, early_stopping_rounds= 200)



        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits



        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = clf.feature_importances_

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y

        gc.collect()



    print('Full AUC score %.6f' % roc_auc_score(train_df['RESULT'], oof_preds))

    # Write submission file and plot feature importance

    df__sample_submission = pd.read_csv('../input/cax-mortgagemodeling/CAX_MortgageModeling_SubmissionFormat.csv')

    subb = pd.DataFrame()

    subb['Unique_ID'] = df__sample_submission['Unique_ID']

    subb['Result_Predicted']=  sub_preds

    subb['Result_Predicted'] = subb.Result_Predicted.replace(1,'FUNDED')

    subb['Result_Predicted'] = subb.Result_Predicted.replace(2,'FUNDED')

    subb['Result_Predicted'] = subb.Result_Predicted.replace(-0,'NOT FUNDED')

    subb['Result_Predicted'] = subb.Result_Predicted.replace(-1,'NOT FUNDED')

    subb['Result_Predicted'] = subb.Result_Predicted.replace(-2,'NOT FUNDED')

    subb.to_csv('CAX_MortgageModeling_SubmissionFormat.csv',index = 0)

    display_importances(feature_importance_df)

    return feature_importance_df



# Display/plot feature importance

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances01.png')







feat_importance = kfold_lightgbm(dataset2, num_folds= 10, stratified= False)'''

#lollol = pd.read_csv('CAX_MortgageModeling_SubmissionFormat.csv')
'''final = pd.read_csv('CAX_MortgageModeling_SubmissionFormat.csv')

new= []

for i in final.Result_Predicted:

    if i < 0.69:

        new.append(1)

    else:

        new.append(0)'''
'''final.Result_Predicted = new

final.Result_Predicted.value_counts()

final['Result_Predicted'] = final.Result_Predicted.replace(1,'NOT FUNDED')

final['Result_Predicted'] = final.Result_Predicted.replace(0,'FUNDED')

final.to_csv('CAX_MortgageModeling_SubmissionFormat.csv',index = 0)'''
'''from xgboost import XGBClassifier



meta_model = XGBClassifier(max_depth=6, learning_rate=0.07, n_estimators=1100,task_type = "GPU", scale_pos_weight =0.4, )



meta_model.fit(X_train,y_train,eval_set = [(X_val,y_val)], early_stopping_rounds = 30,)

y_pred_xg = meta_model.predict(X_val)



print("Training Accuracy: ", meta_model.score(X_train, y_train))

print('Testing Accuarcy: ', meta_model.score(X_val, y_val))



# making a classification report

print(f1_score(y_val, y_pred_xg, average='macro'))

print(confusion_matrix(y_val, y_pred_xg))'''
'''from imblearn.ensemble import BalancedBaggingClassifier

resampled_rf = BalancedBaggingClassifier(base_estimator=model_lgb,

                                         n_estimators=10,

                                         random_state=123)

pip_resampled = make_pipeline(RobustScaler(),

                              resampled_rf)

scores = cross_val_score(pip_resampled,

                         X_train, y_train,

                         scoring="roc_auc", cv= 2)

print(f"EasyEnsemble model's average AUC: {scores.mean():.3f}")'''
'''resampled_rf = BalancedBaggingClassifier(base_estimator=model_lgb,

                                         n_estimators=20,

                                         random_state=123)



resampled_rf.fit(X_train, y_train)

y_pred_xg = resampled_rf.predict(X_val)



print("Training Accuracy: ", meta_model.score(X_train, y_train))

print('Testing Accuarcy: ', meta_model.score(X_val, y_val))



# making a classification report

print(f1_score(y_val, y_pred_xg, average='macro'))

print(confusion_matrix(y_val, y_pred_xg))'''
'''y_pred_xg = (resampled_rf.predict_proba(X_val)[:,1] >= 0.31).astype(bool)



# making a classification report

print(f1_score(y_val, y_pred_xg, average='macro'))

print(confusion_matrix(y_val, y_pred_xg))'''
'''from sklearn.model_selection import learning_curve



model_lgb.fit(X_train, y_train)

train_sizes, train_scores, valid_scores = learning_curve(model_lgb, 

X=X_train, y=y_train, cv=3)



train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

valid_mean = np.mean(valid_scores, axis=1)

valid_std = np.std(valid_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o',markersize=5, label='training accuracy')

plt.plot(train_sizes, valid_mean, color='red', marker='o', markersize=5, label='valid accuracy')'''
'''from sklearn.model_selection import learning_curve



meta_model = XGBClassifier(max_depth=7, learning_rate=0.07, n_estimators=1100, scale_pos_weight =0.3, 

                        )



meta_model.fit(X_train, y_train)

train_sizes, train_scores, valid_scores = learning_curve(meta_model, 

X=X_train, y=y_train)



train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

valid_mean = np.mean(valid_scores, axis=1)

valid_std = np.std(valid_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o',markersize=5, label='training accuracy')

plt.plot(train_sizes, valid_mean, color='red', marker='o', markersize=5, label='valid accuracy')'''
'''from scipy.stats import randint as sp_randint



from scipy.stats import uniform as sp_uniform

param_test ={'num_leaves': sp_randint(6, 50), 

             'min_child_samples': sp_randint(100, 500), 

             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

             'subsample': sp_uniform(loc=0.2, scale=0.8), 

             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

             'scale_pos_weight': [0.1,0.5,0.9,1,3],

             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}





n_HP_points_to_test = 100



import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum

clf = lgb.LGBMClassifier(max_depth=-1, random_state=7, silent=True, metric='logloss', n_jobs=4, n_estimators=5000)

gs = RandomizedSearchCV(

    estimator=clf, param_distributions=param_test, 

    n_iter=n_HP_points_to_test,

    scoring='f1_macro',

    cv=3,

    refit=True,

    random_state=314,

    verbose=True)





gs.fit(X = X_train, y = y_train,eval_set = [(X_val,y_val)])

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))'''

'''from skopt import gp_minimize

from skopt.space import Real, Integer

from skopt.utils import use_named_args

from skopt.plots import plot_convergence

import time'''
'''class ModelOptimizer:

    best_score = None

    opt = None

    

    def __init__(self, model, X_train, y_train, categorical_columns_indices=None, n_fold=3, seed=1994, early_stopping_rounds=30, is_stratified=True, is_shuffle=True):

        self.model = model

        self.X_train = X_train

        self.y_train = y_train

        self.categorical_columns_indices = categorical_columns_indices

        self.n_fold = n_fold

        self.seed = seed

        self.early_stopping_rounds = early_stopping_rounds

        self.is_stratified = is_stratified

        self.is_shuffle = is_shuffle

        

        

    def update_model(self, **kwargs):

        for k, v in kwargs.items():

            setattr(self.model, k, v)

            

    def evaluate_model(self):

        pass

    

    def optimize(self, param_space, max_evals=10, n_random_starts=2):

        start_time = time.time()

        

        @use_named_args(param_space)

        def _minimize(**params):

            self.model.set_params(**params)

            return self.evaluate_model()

        

        opt = gp_minimize(_minimize, param_space, n_calls=max_evals, n_random_starts=n_random_starts, random_state=2405, n_jobs=-1)

        best_values = opt.x

        optimal_values = dict(zip([param.name for param in param_space], best_values))

        best_score = opt.fun

        self.best_score = best_score

        self.opt = opt

        

        print('optimal_parameters: {}\noptimal score: {}\noptimization time: {}'.format(optimal_values, best_score, time.time() - start_time))

        print('updating model with optimal values')

        self.update_model(**optimal_values)

        plot_convergence(opt)

        return optimal_values

    

class CatboostOptimizer(ModelOptimizer):

    def evaluate_model(self):

        validation_scores = catboost.cv(

        catboost.Pool(self.X_train, 

                      self.y_train, 

                      cat_features=self.categorical_columns_indices),

        self.model.get_params(), 

        nfold=self.n_fold,

        stratified=self.is_stratified,

        seed=self.seed,

        early_stopping_rounds=self.early_stopping_rounds,

        shuffle=self.is_shuffle,

        verbose=100,

        plot=False)

        self.scores = validation_scores

        test_scores = validation_scores.iloc[:, 2]

        best_metric = test_scores.max()

        return 1 - best_metric'''
'''import catboost

cb = catboost.CatBoostClassifier(n_estimators=4000, # use large n_estimators deliberately to make use of the early stopping

                         loss_function='Logloss',

                         eval_metric='AUC',

                         boosting_type='Ordered', # use permutations

                         random_seed=1994, 

                         scale_pos_weight = 0.4,

                         use_best_model=True)

cb_optimizer = CatboostOptimizer(cb, X_train, y_train)

params_space = [Real(0.001, 0.2, name='learning_rate'),]

cb_optimal_values = cb_optimizer.optimize(params_space)'''
new_train.to_csv("Train_Preprocessed_version.csv",index=False)



new_test.to_csv("Test_Preprocessed_version.csv",index=False)
import h2o

from h2o.automl import H2OAutoML
h2o.init()
h2o_train=h2o.import_file("Train_Preprocessed_version.csv")

h2o_test=h2o.import_file("Test_Preprocessed_version.csv")
x = h2o_train.columns

y = "RESULT"

x.remove(y)
h2o_train[y] = h2o_train[y].asfactor()

h2o_test[y] = h2o_test[y].asfactor()
h2o_train.describe()
#cols_to_change = ['AMORTIZATION','MORTGAGE PURPOSE','PAYMENT FREQUENCY','PROPERTY TYPE','TERM','FSA','AGE RANGE','INCOME TYPE','NAICS CODE','RESULT',

#        'GEN_Male','GEN_Unknown','FSA_Province','FSA_Area',

#       'FSA_Geo','FSA_Geo3','FSA_Geo4','FSA_Geo5','Rate_weights','binned_income','credit_score_binned','naics_score_binned']
#for col in cols_to_change:

#    h2o_train[col] = h2o_train[col].asfactor()
aml = H2OAutoML(max_models=15, seed=2,balance_classes = True,max_runtime_secs=10000,project_name="CAX_Mortgage",

                max_runtime_secs_per_model=450)

aml.train(x=x, y=y, training_frame=h2o_train)
aml.leaderboard
preds = aml.predict(h2o_test)
h2o.save_model(aml.leader, path="h2o model2")
df__sample_submission = pd.read_csv('../input/cax-mortgagemodeling/CAX_MortgageModeling_SubmissionFormat.csv')
preds=preds.as_data_frame()
df__sample_submission['Result_Predicted']=preds['predict']
df__sample_submission.to_csv("Best_AutoML.csv",index=False)
#new_train,new_test = dataset[:len(train)],dataset[len(train):]

#new_train.RESULT = new_train.RESULT.replace(-1,0)

#X, y = new_train.drop(['RESULT',],axis = 1), new_train.RESULT

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
#from sklearn.ensemble import GradientBoostingClassifier

#from sklearn.tree import DecisionTreeClassifier

#import lightgbm as lgb

#from sklearn.ensemble import AdaBoostClassifier

'''models = [

    ExtraTreesClassifier(random_state=0, n_jobs=-1, 

                         n_estimators=100, max_depth=3),

        

    RandomForestClassifier(random_state=0, n_jobs=-1, 

                           n_estimators=100, max_depth=3),

        

    XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, 

                  n_estimators=100, max_depth=3),

    

    GradientBoostingClassifier(),

    

    DecisionTreeClassifier(criterion = 'entropy', 

                                    random_state = 0),

    

    AdaBoostClassifier(),

    

]'''
'''S_train, S_test = stacking(models,                     # list of models

                           X_train,y_train,X_test,    # data

                           regression=False,           # classification task (if you need 

                                                       #     regression - set to True)

                           mode='oof_pred_bag',        # mode: oof for train set, predict test 

                                                       #     set in each fold and vote

                           needs_proba=False,          # predict class labels (if you need 

                                                       #     probabilities - set to True) 

                           save_dir=None,              # do not save result and log (to save 

                                                       #     in current dir - set to '.')

                           metric=f1_score,            # metric: callable

                           n_folds=4,                  # number of folds

                           stratified=True,            # stratified split for folds

                           shuffle=True,               # shuffle the data

                           random_state=0,             # ensure reproducibility

                           verbose=2)'''
'''df__sample_submission = pd.read_csv('../input/cax-mortgagemodeling/CAX_MortgageModeling_SubmissionFormat.csv')

subb = pd.DataFrame()

subb['Unique_ID'] = df__sample_submission['Unique_ID']

subb['Result_Predicted'] = y_pred

subb['Result_Predicted'] = subb.Result_Predicted.replace(1,'FUNDED')

subb['Result_Predicted'] = subb.Result_Predicted.replace(2,'FUNDED')

subb['Result_Predicted'] = subb.Result_Predicted.replace(-0,'NOT FUNDED')

subb['Result_Predicted'] = subb.Result_Predicted.replace(-1,'NOT FUNDED')

subb['Result_Predicted'] = subb.Result_Predicted.replace(-2,'NOT FUNDED')

subb.to_csv('CAX_MortgageModeling_SubmissionFormat.csv',index = 0)'''
#train.RESULT.value_counts()#
#subb.Result_Predicted.value_counts()
'''

rf_feature = rf.feature_importances_   #random forest

ada_feature = model_ada.feature_importances_ #adaboost decision

lgb_feature = model_lgb.feature_importances_ #lgbm

et_feature = model_etc.feature_importances_  #extra tree

dt_feature = dt.feature_importances_ #decision tree

gbc_feature = model_gbc.feature_importances_ #gradient boost

rf_features = [i for i in rf_feature]

ada_features = [i for i in ada_feature]

lgb_features = [i for i in lgb_feature]

et_features = [i for i in et_feature]

dt_features = [i for i in dt_feature]

gbc_features = [i for i in gbc_feature]

cols = [i for i in new_train.columns if  i not in 'RESULT']

'''
#len(new_train.columns)
#predictions_lol = y_pred_rf*0.2 + y_pred_ada*0.2 + y_pred_lgb*0.4 + y_pred_etc*0.3 + y_pred_dt*0.1

#hmmp = [round(i)  for i in predictions_lol]
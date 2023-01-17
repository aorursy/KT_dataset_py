#from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.plotting import scatter_matrix





import numpy as np

#import pandas as pd

from matplotlib import pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

from scipy.stats import kurtosis

from scipy.stats import skew

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from numpy.fft import *

from sklearn.model_selection import cross_val_score,train_test_split

import lightgbm as lgb

import xgboost as xgb

from sklearn.preprocessing import StandardScaler

from mlxtend.classifier import StackingClassifier

from mlxtend.classifier import StackingCVClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

import seaborn as sns

import matplotlib.style as style 

import eli5

from skopt import BayesSearchCV

from eli5.sklearn import PermutationImportance

style.use('ggplot')

import warnings

warnings.filterwarnings('ignore')







from imblearn.pipeline import make_pipeline

from imblearn import datasets

from imblearn.over_sampling import SMOTE



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold

from sklearn.metrics import recall_score, roc_auc_score



from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3, random_state=None)

# from xgboost import XGBClassifier

# model = XGBClassifier()

from sklearn.metrics import f1_score,classification_report



import matplotlib.pyplot as plt

from itertools import cycle



from sklearn import svm, datasets

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp

from sklearn.metrics import roc_auc_score

from eli5.sklearn import PermutationImportance





from sklearn.ensemble import ExtraTreesClassifier

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score,classification_report



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sample = pd.read_csv('/kaggle/input/datavidia2019/sample_submission.csv')

hotel = pd.read_csv('/kaggle/input/datavidia2019/hotel.csv')

flight = pd.read_csv('/kaggle/input/datavidia2019/flight.csv')

test = pd.read_csv('/kaggle/input/datavidia2019/test.csv')
flight.info()
flight.shape
hotel.info()
hotel.shape
flight['gender'].value_counts() 

# terdapat sebanyak 24 data yang tidak diketahui gendernya.
flight['trip'].value_counts()
flight['service_class'].value_counts()
flight['is_tx_promo'].value_counts()
flight['airlines_name'].value_counts()
flight['route'].value_counts()
flight.groupby(['is_tx_promo', 'gender']).count()
# is_tx_promo

print('female NO  :', (29539/(29539+38074))*100)

print('male NO    :', (38074/(38074+29539))*100)

print('female YES :', (27347/(22962+27347))*100)

print('male YES   :', (22963/(27347+22962))*100)
flight_gender = flight[flight['gender'] == 'None'] # filtering DataFrame untuk kolom fitur Gender == None

flight_gender.account_id.unique()
a1 = flight[flight['account_id'] == 'eaa8ec58eb416c13dd6f9d53ff88b2f2'] # 5 kali perjalanan tanpa promo M

a2 = flight[flight['account_id'] == '044eb7e13934d8dda219a7bc297e907d'] # 12 kali perjalanan            F

a3 = flight[flight['account_id'] == 'af59e51233e2da3877e9b520aae7cfbb'] # 3 kali perjalanan             F

a4 = flight[flight['account_id'] == '02789604494e3f45ebe767d538ff4fbc'] # 2 kali perjalanan             F

a5 = flight[flight['account_id'] == 'bbcc88153dbac780d0630f8ab4b39708'] # 1 kali perjalanan             F *terdapat data dengan price, is_tx_promo yang sama 

a6 = flight[flight['account_id'] == 'd9313670821c35b4f774dd482edc37ea'] # 1 kali perjalanan             M
for i in flight.loc[flight['account_id'] == 'eaa8ec58eb416c13dd6f9d53ff88b2f2'].index :

    flight['gender'][i] = 'M'



for i in flight.loc[flight['account_id'] == '044eb7e13934d8dda219a7bc297e907d'].index :

    flight['gender'][i] = 'F'

    

for i in flight.loc[flight['account_id'] == 'af59e51233e2da3877e9b520aae7cfbb'].index :

    flight['gender'][i] = 'F'



for i in flight.loc[flight['account_id'] == '02789604494e3f45ebe767d538ff4fbc'].index :

    flight['gender'][i] = 'F'



for i in flight.loc[flight['account_id'] == 'bbcc88153dbac780d0630f8ab4b39708'].index :

    flight['gender'][i] = 'F'

    

for i in flight.loc[flight['account_id'] == 'd9313670821c35b4f774dd482edc37ea'].index :

    flight['gender'][i] = 'M'
flight.gender.value_counts()
# Membuat list Feature Target

iscrosssell = list()

for i in range(len(flight)):

    if flight.iloc[i,]['hotel_id'] == 'None':

        iscrosssell.append('no')

    else:

        iscrosssell.append('yes')

        

# Membuat kolom baru di flight dataframe

flight['is_cross_sell'] = pd.Series(iscrosssell)

flight.is_cross_sell.replace(['yes', 'no'], [1,0], inplace = True)

data_awal = pd.concat([flight, test])
# 1 --> YES

# 0 --> NO

data_awal.is_tx_promo.replace(['YES', 'NO'], [1,0], inplace = True)

    

# 1 --> Male

# 0 --> Female

data_awal.gender.replace(['M','F'], [1,0], inplace=True)



# 1 --> Economy

# 0 --> Business

data_awal.service_class.replace(['ECONOMY','BUSINESS'], [1,0], inplace=True)



# 0 --> trip

# 1 --> roundtrip

# 2 --> round

data_awal.trip.replace(['trip','roundtrip','round'], [0,1,2], inplace=True)
# Menyimpan dataframe sebelum dirubah nama maskapai

data_train = data_awal

# Mengubah nama maskapai

data_awal.airlines_name.replace(['6c483c0812c96f8ec43bb0ff76eaf716','33199710eb822fbcfd0dc793f4788d30', '0a102015e48c1f68e121acc99fca9a05',

                             'ad5bef60d81ea077018f4d50b813153a', '74c5549aa99d55280a896ea50068a211','e35de6a36d385711a660c72c0286154a'

                                ,'9855a1d3de1c46526dde37c5d6fb758c','6872b49542519aea7ae146e23fab5c08'],

                            [1,2,3,4,5,6,7,8], inplace =True)
flight = data_awal.iloc[:117946]

test = data_awal.iloc[117946:]
flight.shape, test.shape
# Mencari account_id yang sama di dataset flight dan test.

kolom_account_id = list()



for i in test.account_id.unique():

    if i in flight.account_id.unique():

        kolom_account_id.append(i)

        

# Mengubah list ke dalam bentuk dataframe        

kolom_account_id = pd.DataFrame(pd.Series(kolom_account_id))

kolom_account_id.columns = ['Perpotongan_Akun']

kolom_account_id.head()
# Membuat DataFrame Data_flight

Data_flight = list()

for i in kolom_account_id.Perpotongan_Akun:

    for i in flight.loc[flight.account_id == i].index:

        Data_flight.append(tuple(flight.iloc[i]))

Data_flight = pd.DataFrame(Data_flight, columns = flight.columns)
# Membuat DataFrame Data_flight_no

Data_flight_no = flight.copy()

for i in kolom_account_id.Perpotongan_Akun:

    Data_flight_no = Data_flight_no[Data_flight_no['account_id'] != i]
# Membuat DataFrame Data_test

Data_test = list()

test_ = test.reset_index()

test_ = test_.drop(columns='index')

for i in kolom_account_id.Perpotongan_Akun:

    for i in test_.loc[test_.account_id == i].index:

        Data_test.append(tuple(test_.iloc[i]))

Data_test = pd.DataFrame(Data_test, columns = test_.columns)
# Membuat DataFrame Data_test_no

Data_test_no = test_.copy()

for i in kolom_account_id.Perpotongan_Akun:

    Data_test_no = Data_test_no[Data_test_no['account_id'] != i]
# Melihat Ukuran Baris dan Kolom dari setiap dataframe

Data_flight.shape, Data_flight_no.shape, Data_test.shape, Data_test_no.shape
# Melihat apakah ada account_id yang sama atau tidak sama pada ke-4 dataframe

len(Data_flight.account_id.unique()), len(Data_flight_no.account_id.unique()), len(Data_test.account_id.unique()), len(Data_test_no.account_id.unique())
fig, ax = plt.subplots(2,4 , figsize= (12,8))



sns.distplot(Data_flight.price, ax = ax[0,0])

ax[0,0].set_title('Distribusi Data_flight')



sns.distplot(Data_flight_no.price, ax = ax[0,1])

ax[0,1].set_title('Distribusi Data_flight_no')





sns.distplot(Data_test.price, ax = ax[0,2])

ax[0,2].set_title('Distribusi Data_test ')





sns.distplot(Data_test_no.price, ax = ax[0,3])

ax[0,3].set_title('Distribusi Data_test_no')





sns.distplot(Data_flight.member_duration_days, ax = ax[1,0])

ax[1,0].set_title('Distribusi Data_flight')



sns.distplot(Data_flight_no.member_duration_days, ax = ax[1,1])

ax[1,1].set_title('Distribusi Data_flight_no')





sns.distplot(Data_test.member_duration_days, ax = ax[1,2])

ax[1,2].set_title('Distribusi Data_test ')





sns.distplot(Data_test_no.member_duration_days, ax = ax[1,3])

ax[1,3].set_title('Distribusi Data_test_no')







plt.tight_layout()

plt.show()
Data_flight.price.median(), Data_flight_no.price.median(),  Data_test.price.median(), Data_test_no.price.median()
fig, ax = plt.subplots(2,4 , figsize= (15,10))



sns.countplot(Data_flight.is_tx_promo, ax = ax[0,0])

ax[0,0].set_title('Distribusi Data_flight')



sns.countplot(Data_flight_no.is_tx_promo ,ax = ax[0,1])

ax[0,1].set_title('Distribusi Data_flight_no')





sns.countplot(Data_test.is_tx_promo, ax = ax[0,2])

ax[0,2].set_title('Distribusi Data_test ')





sns.countplot(Data_test_no.is_tx_promo, ax = ax[0,3])

ax[0,3].set_title('Distribusi Data_test_no')





sns.countplot(Data_flight.no_of_seats, ax = ax[1,0])

ax[1,0].set_title('Distribusi Data_flight')



sns.countplot(Data_flight_no.no_of_seats, ax = ax[1,1])

ax[1,1].set_title('Distribusi Data_flight_no')





sns.countplot(Data_test.no_of_seats, ax = ax[1,2])

ax[1,2].set_title('Distribusi Data_test ')





sns.countplot(Data_test_no.no_of_seats, ax = ax[1,3])

ax[1,3].set_title('Distribusi Data_test_no')







plt.tight_layout()

plt.show()
fig, ax = plt.subplots(2,4 , figsize= (15,10))



sns.countplot(Data_flight.airlines_name, ax = ax[0,0])

ax[0,0].set_title('Distribusi Data_flight')



sns.countplot(Data_flight_no.airlines_name ,ax = ax[0,1])

ax[0,1].set_title('Distribusi Data_flight_no')





sns.countplot(Data_test.airlines_name, ax = ax[0,2])

ax[0,2].set_title('Distribusi Data_test ')





sns.countplot(Data_test_no.airlines_name, ax = ax[0,3])

ax[0,3].set_title('Distribusi Data_test_no')





sns.countplot(Data_flight.trip, ax = ax[1,0])

ax[1,0].set_title('Distribusi Data_flight')



sns.countplot(Data_flight_no.trip, ax = ax[1,1])

ax[1,1].set_title('Distribusi Data_flight_no')





sns.countplot(Data_test.trip, ax = ax[1,2])

ax[1,2].set_title('Distribusi Data_test ')





sns.countplot(Data_test_no.trip, ax = ax[1,3])

ax[1,3].set_title('Distribusi Data_test_no')







plt.tight_layout()

plt.show()
fig, ax = plt.subplots(2,2 , figsize= (20,20))





sns.countplot(x = 'is_cross_sell' , hue = 'gender', data= data_train, ax = ax[0,0])

sns.countplot(x = 'is_cross_sell', hue = 'trip', data = data_train, ax = ax[0,1])

sns.countplot(x = 'is_cross_sell', hue = 'service_class', data = data_train, ax = ax[1,0])

sns.countplot(x = 'is_cross_sell', hue = 'is_tx_promo', data = data_train, ax = ax[1,1])
fig, ax = plt.subplots(2 ,2 , figsize = (20, 20))



sns.countplot(x = 'is_cross_sell', hue = 'no_of_seats', data = data_train, ax = ax[0,0])

sns.countplot(x = 'is_cross_sell', hue = 'route', data = data_train, ax = ax[0,1])

sns.countplot(x ='is_cross_sell', hue = 'visited_city', data = data_train , ax = ax[1,0])
fig, ax = plt.subplots(2,2 , figsize = (20,20))





sns.distplot(data_train['price'], ax = ax[0,0])

sns.distplot(data_train['member_duration_days'], ax = ax[0,1])

sns.countplot(x = 'airlines_name', data = data_train, ax = ax[1,0])

sns.countplot(x = 'airlines_name', hue = 'is_cross_sell', data = data_train, ax = ax[1,1])
data_train['visited'] = data_train['visited_city'].apply(lambda x : x.split("[")[1])

data_train['visited'] = data_train['visited'].apply(lambda x : x.split("]")[0])

data_train['con_visited'] = data_train['visited'].apply(lambda x : x.count(",") + 1)
data_train['con_log'] = data_train['log_transaction'].apply(lambda x : x.count(",") +1)
data = data_train[['account_id', 'order_id']]

data_group = data['account_id'].value_counts().index.to_frame()

data_group['jumlah_transaksi'] = data['account_id'].value_counts().values

data_group = data_group.rename(columns = {0:'account_id'})

data_train = pd.merge(data_train, data_group , how = 'left', on= 'account_id')


data_train['kota1'] = data_train['visited'].apply(lambda x : x.split(",")[0])

data_train['kota2'] = data_train['visited'].apply(lambda x : x.split(",")[1])

data_train['kota3'] = data_train['visited'].apply(lambda x : x.split(",")[2])



data_train['kota4'] = 'None'

for i in range(len(data_train)):

    try :

        data_train['kota4'][i] = data_train['visited'][i].split(",")[3]

    except:

        continue

    



data_train['kota5'] = 'None'

for i in range(len(data_train)):

    try:

        data_train['kota5'][i] = data_train['visited'][i].split(",")[4]

    except:

        continue

    



    

import re



for i in range(len(data_train)):

    data_train['kota1'][i] = "".join(re.findall("[a-zA-Z]", data_train['kota1'][i]))    

    data_train['kota2'][i] = "".join(re.findall("[a-zA-Z]", data_train['kota2'][i]))

    data_train['kota3'][i] = "".join(re.findall("[a-zA-Z]", data_train['kota3'][i]))

    data_train['kota4'][i] = "".join(re.findall("[a-zA-Z]", data_train['kota4'][i]))

    data_train['kota5'][i] = "".join(re.findall("[a-zA-Z]", data_train['kota5'][i]))





data_train['Semarang'] = 0 

for i in range(len(data_train)):

    if data_train['kota1'][i] == 'Semarang':

        data_train['Semarang'][i] = 1

    elif data_train['kota2'][i] == 'Semarang':

        data_train['Semarang'][i] = 1

    elif data_train['kota3'][i] == 'Semarang':

        data_train['Semarang'][i] = 1

    elif data_train['kota4'][i] == 'Semarang':

        data_train['Semarang'][i] = 1  

    elif data_train['kota5'][i] == 'Semarang':

        data_train['Semarang'][i] = 1





data_train['Jogjakarta'] = 0 

for i in range(len(data_train)):

    if data_train['kota1'][i] == 'Jogjakarta':

        data_train['Jogjakarta'][i] = 1

    elif data_train['kota2'][i] == 'Jogjakarta':

        data_train['Jogjakarta'][i] = 1

    elif data_train['kota3'][i] == 'Jogjakarta':

        data_train['Jogjakarta'][i] = 1

    elif data_train['kota4'][i] == 'Jogjakarta':

        data_train['Jogjakarta'][i] = 1  

    elif data_train['kota5'][i] == 'Jogjakarta':

        data_train['Jogjakarta'][i] = 1

   

data_train['Aceh'] = 0 

for i in range(len(data_train)):

    if data_train['kota1'][i] == 'Aceh':

        data_train['Aceh'][i] = 1

    elif data_train['kota2'][i] == 'Aceh':

        data_train['Aceh'][i] = 1

    elif data_train['kota3'][i] == 'Aceh':

        data_train['Aceh'][i] = 1

    elif data_train['kota4'][i] == 'Aceh':

        data_train['Aceh'][i] = 1  

    elif data_train['kota5'][i] == 'Aceh':

        data_train['Aceh'][i] = 1



facet = sns.FacetGrid(data_train, hue="is_cross_sell",aspect=4)

facet.map(sns.kdeplot,'price',shade= True)

facet.set(xlim=(0, data_train['price'].max()))

facet.add_legend()

 

plt.xlim(4000000,1000000) 

data_train.loc[data_train['price'] <= 840000 , 'bin_price'] = 'pr_1'

data_train.loc[(data_train['price'] > 840000) & (data_train['price']<=1100000), 'bin_price' ] = 'pr_2'

data_train.loc[(data_train['price'] >1100000)& (data_train['price']<1350000), 'bin_price']= 'pr_3'

data_train.loc[(data_train['price'] > 1350000)& (data_train['price']<1760000),'bin_price']= 'pr_4'

data_train.loc[(data_train['price']>1760000), 'bin_price'] = 'pr_5'
facet = sns.FacetGrid(data_train, hue="is_cross_sell",aspect=4)

facet.map(sns.kdeplot,'member_duration_days',shade= True)

facet.set(xlim=(0, data_train['member_duration_days'].max()))

facet.add_legend()

 
data_train.loc[data_train['member_duration_days'] <= 340, "bin_member_duration_days"] = 'duration_1',

data_train.loc[(data_train['member_duration_days'] > 340) & (data_train['member_duration_days'] <= 735), "bin_member_duration_days" ] = 'duration_2',

data_train.loc[data_train['member_duration_days'] > 735, "bin_member_duration_days"] = 'duration_3',
def _kurtosis(x):

    return kurtosis(x)



def CPT5(x):

    den = len(x)*np.exp(np.std(x))

    return sum(np.exp(x))/den



def skewness(x):

    return skew(x)



def SSC(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    xn_i1 = x[0:len(x)-2]  # xn-1

    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)

    return sum(ans[1:]) 



def wave_length(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    return sum(abs(xn_i2-xn))

    

def norm_entropy(x):

    tresh = 3

    return sum(np.power(abs(x),tresh))



def SRAV(x):    

    SRA = sum(np.sqrt(abs(x)))

    return np.power(SRA/len(x),2)



def mean_abs(x):

    return sum(abs(x))/len(x)



def zero_crossing(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1

    return sum(np.heaviside(-xn*xn_i2,0))



def mean_change_of_abs_change(x):

    return np.mean(np.diff(np.abs(np.diff(x))))
# Membuat fitur berdasarkan mean,sum,median,quartile,max,min, dan beberapa statistik lainnya dari log_transaction.



data_train['log_transaction'] = data_train['log_transaction'].apply(lambda x : x.split("[")[1])

data_train['log_transaction'] = data_train['log_transaction'].apply(lambda x : x.split("]")[0])



data_train['list_log'] = data_train['log_transaction'].apply(lambda x : [float(x.split(",")[i]) for i in range(len(x.split(",")))] )



data_train['mean_log'] = data_train['list_log'].apply(lambda x : np.mean(x))

data_train['sum_log'] = data_train['list_log'].apply(lambda x :np.sum(x))

data_train['median_log'] = data_train['list_log'].apply(lambda x : np.percentile(x, 50))

data_train['quartile_1'] = data_train['list_log'].apply(lambda x : np.percentile(x, 25))

data_train['qurttile_3'] = data_train['list_log'].apply(lambda x :np.percentile(x , 75))

data_train['quartile_4'] = data_train['list_log'].apply(lambda x : np.percentile(x, 95))

data_train['max_log'] = data_train['list_log'].apply(lambda x :np.max(x))

data_train['min_log'] = data_train['list_log'].apply(lambda x :np.min(x))





data_train['range_log'] = data_train['list_log'].apply(lambda x : np.max(x) - np.min(x))

data_train['maxtomin_log'] = data_train['list_log'].apply(lambda x : np.max(x) / np.min(x))

data_train['mean_abs_chg_log'] = data_train['list_log'].apply( lambda x : np.mean(np.abs(np.diff(x))))

data_train['mean_change_of_abs_change'] = data_train['list_log'].apply(lambda x : mean_change_of_abs_change(x))



data_train['abs_max_log'] = data_train['list_log'].apply(lambda x : np.max(np.abs(x)))

data_train['abs_min_log'] = data_train['list_log'].apply(lambda x : np.min(np.abs(x)))

data_train['abs_avg_log'] = (data_train['abs_max_log'] + data_train['abs_min_log'])/2





data_train['iqr_log'] =  data_train['qurttile_3'] - data_train['quartile_1'] 

data_train['scc_log'] = data_train['list_log'].apply(lambda x : SSC(x))

data_train['skewness_log'] = data_train['list_log'].apply(lambda x : skewness(x))



data_train['wave_length_log'] = data_train['list_log'].apply(lambda x : wave_length(x))

data_train['kurtosis_log'] = data_train['list_log'].apply(lambda x : _kurtosis(x))

data_train['zero_crossing'] = data_train['list_log'].apply(lambda x : zero_crossing(x))
# Membuat fitur berdasarkan mean,sum,median,quartile,max,min, dan beberapa statistik lainnya dari price.



col = 'price'

agg = pd.DataFrame()



agg[str(col)+'_mean'] = data_train.groupby(['account_id'])[col].mean()

agg[str(col)+'_median'] = data_train.groupby(['account_id'])[col].median()

agg[str(col)+'_max'] = data_train.groupby(['account_id'])[col].max()

agg[str(col)+'_min'] = data_train.groupby(['account_id'])[col].min()

agg[str(col) + '_maxtoMin'] = agg[str(col) + '_max'] / agg[str(col) + '_min']

agg[str(col) + '_abs_max'] = data_train.groupby(['account_id'])[col].apply(lambda x: np.max(np.abs(x)))

agg[str(col) + '_abs_min'] = data_train.groupby(['account_id'])[col].apply(lambda x: np.min(np.abs(x)))

agg[str(col) + '_abs_avg'] = (agg[col + '_abs_min'] + agg[col + '_abs_max'])/2

agg[str(col)+'_mad'] = data_train.groupby(['account_id'])[col].mad()

agg[str(col)+'_q25'] = data_train.groupby(['account_id'])[col].quantile(0.25)

agg[str(col)+'_q75'] = data_train.groupby(['account_id'])[col].quantile(0.75)

agg[str(col)+'_q95'] = data_train.groupby(['account_id'])[col].quantile(0.95)

agg[str(col)+'_ssc'] = data_train.groupby(['account_id'])[col].apply(SSC)

agg[str(col)+'_mean_abs'] = data_train.groupby(['account_id'])[col].apply(mean_abs)

agg[str(col)+'_norm_entropy'] = data_train.groupby(['account_id'])[col].apply(norm_entropy)

agg[str(col)+'_SRAV'] = data_train.groupby(['account_id'])[col].apply(SRAV)

agg[str(col)+'_kurtosis'] = data_train.groupby(['account_id'])[col].apply(_kurtosis)

data_train = pd.merge( data_train, agg , how = "left", on = "account_id") 
# Membuat fitur berdasarkan mean,sum,median,quartile,max,min, dan beberapa statistik lainnya dari price.

col = 'member_duration_days'

agg = pd.DataFrame()



agg[str(col)+'_mean'] = data_train.groupby(['account_id'])[col].mean()

agg[str(col)+'_median'] = data_train.groupby(['account_id'])[col].median()

agg[str(col)+'_max'] = data_train.groupby(['account_id'])[col].max()

agg[str(col)+'_min'] = data_train.groupby(['account_id'])[col].min()

agg[str(col) + '_maxtoMin'] = agg[str(col) + '_max'] / agg[str(col) + '_min']

agg[str(col) + '_abs_max'] = data_train.groupby(['account_id'])[col].apply(lambda x: np.max(np.abs(x)))

agg[str(col) + '_abs_min'] = data_train.groupby(['account_id'])[col].apply(lambda x: np.min(np.abs(x)))

agg[str(col) + '_abs_avg'] = (agg[col + '_abs_min'] + agg[col + '_abs_max'])/2

agg[str(col)+'_mad'] = data_train.groupby(['account_id'])[col].mad()

agg[str(col)+'_q25'] = data_train.groupby(['account_id'])[col].quantile(0.25)

agg[str(col)+'_q75'] = data_train.groupby(['account_id'])[col].quantile(0.75)

agg[str(col)+'_q95'] = data_train.groupby(['account_id'])[col].quantile(0.95)

agg[str(col)+'_ssc'] = data_train.groupby(['account_id'])[col].apply(SSC)

agg[str(col)+'_mean_abs'] = data_train.groupby(['account_id'])[col].apply(mean_abs)

agg[str(col)+'_norm_entropy'] = data_train.groupby(['account_id'])[col].apply(norm_entropy)

agg[str(col)+'_SRAV'] = data_train.groupby(['account_id'])[col].apply(SRAV)

agg[str(col)+'_kurtosis'] = data_train.groupby(['account_id'])[col].apply(_kurtosis)

data_train = pd.merge( data_train, agg , how = "left", on = "account_id") 


data_train['Harga_Satuan_Bangku'] = (data_train.price) / (data_train.no_of_seats)

data_train['member_duration_week'] = (data_train.member_duration_days) / 7

data_train['member_duration_month'] = (data_train.member_duration_days) / 30

data_train['member_duration_year'] = (data_train.member_duration_days) / 365



data_train['durasi_kunjungan'] = data_train.member_duration_days / data_train.con_visited
clustering = data_train[['order_id', 'member_duration_days', 'price']]
wcs = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=1000, n_init=20, random_state=0)

    kmeans.fit(clustering[['price', 'member_duration_days']])

    wcs.append(kmeans.inertia_)

print(wcs)
plt.plot(range(1, 11), wcs)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

pred_y = kmeans.fit_predict(clustering[['price', 'member_duration_days']])

clustering['clustering_price_duration'] = pred_y.tolist()



clustering['clustering_price_duration'].value_counts()
wc = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeans.fit(clustering[['price']])

    wc.append(kmeans.inertia_)

plt.plot(range(1, 11), wc)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

pred_price = kmeans.fit_predict(clustering[['price']])





clustering['clustering_price'] = pred_price.tolist()

clustering['clustering_price'].value_counts()
wc = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)

    kmeans.fit(clustering[['member_duration_days']])

    wc.append(kmeans.inertia_)

plt.plot(range(1, 11), wc)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

pred_price = kmeans.fit_predict(clustering[['member_duration_days']])



clustering['clustering_duration'] = pred_price.tolist()

clustering['clustering_duration'].value_counts()



# Satukan data



clustering = clustering[['order_id', 'clustering_price_duration', 'clustering_duration']]

data_train = pd.merge(data_train, clustering, how = 'left', on = 'order_id')

data_train.head()
data_train['real_female'] = 0

data_train.loc[(data_train['gender'] == 0)  & (data_train['is_tx_promo'] == 1), 'real_female'] = 1



data_train['real_female'].value_counts()
data_train['rich_female'] = 0

data_train.loc[(data_train['gender'] == 0 ) &(data_train['service_class'] == 0), 'rich_female'] = 1



data_train.rich_female.value_counts()
data_train['rich_male'] = 0

data_train.loc[(data_train['gender']  == 1 ) & (data_train['service_class'] == 0 ), 'rich_male'] = 1



data_train.rich_male.value_counts()
data_train['rich_trip'] = 0

data_train.loc[(data_train['trip_trip']  == 1 ) & (data_train['service_class'] == 0 ), 'rich_trip'] = 1



data_train.rich_trip.value_counts()
data_train['rich_roundtrip'] = 0 

data_train.loc[(data_train['trip_roundtrip']  == 1 ) & (data_train['service_class'] == 0 ), 'rich_roundtrip'] = 1



data_train.rich_roundtrip.value_counts()
data_train.head()
## feature indvidu , pasangan , keluarga

data_train.loc[data_train['no_of_seats'] == 1 , 'berangkat_dengan'] = 'individu'

data_train.loc[data_train['no_of_seats'] == 2 , 'berangkat_dengan'] = 'couple'

data_train.loc[data_train['no_of_seats'] > 2 , 'berangkat_dengan'] = 'family'
facet = sns.FacetGrid(data_train, hue="is_cross_sell",aspect=4)

facet.map(sns.kdeplot,'jumlah_transaksi',shade= True)

facet.set(xlim=(0, data_train['jumlah_transaksi'].max()))

facet.add_legend()

## feature indvidu , pasangan , keluarga

data_train.loc[data_train['jumlah_transaksi'] <= 25 , 'berlangganan'] = 'yes'

data_train.loc[ (data_train['jumlah_transaksi'] > 25) & (data_train['jumlah_transaksi'] >= 118) , 'berlangganan'] = 'no'

data_train.loc[ (data_train['jumlah_transaksi'] > 118) & (data_train['jumlah_transaksi'] > 132) , 'berlangganan'] = 'yes'

data_train.loc[data_train['jumlah_transaksi'] >=  132 , 'berlangganan'] = 'no'

facet = sns.FacetGrid(data_train, hue="is_cross_sell",aspect=4)

facet.map(sns.kdeplot,'con_log',shade= True)

facet.set(xlim=(0, data_train['con_log'].max()))

facet.add_legend()

plt.xlim(0,30)
data_train.loc[data_train['con_log'] <= 15 , 'con_log_bin'] = 'yes'

data_train.loc[data_train['con_log'] > 15 , 'con_log_bin'] = 'no'

# Membuat dummies fitur yang tidak ordinal.

fitur = pd.get_dummies(data_train[['trip','airlines_name','bin_price', 'bin_member_duration_days']])

vis = pd.get_dummies(data_train['visited_city'], prefix = 'visited_city')



data_train = pd.concat([data_train, fitur], axis = 1)

data_train = pd.concat([data_train, vis], axis = 1)

data_train = data_train.drop(['visited_city'], axis = 1)

trip_with = pd.get_dummies(data_train[['berangkat_dengan', 'berlangganan', 'con_log_bin']], prefix = ['berangkat_dengan', 'berlangganan', 'con_log_bin'])



data_train = pd.concat([data_train, trip_with ], axis = 1)

data_train = data_train.drop(['berangkat_dengan', 'berlangganan', 'con_log_bin'], axis = 1)

col_con = data_train.columns.to_list()

exc = []

for column in col_con:

    try:

        data_train[column] = data_train[column].astype('int64')

    except:

        exc.append(column)

        

exc
data_train.head()
flight__ = data_train.iloc[:117946]

test__ = data_train.iloc[117946:]
test = data_train[data_train['is_cross_sell'] == 'None']

train = data_train[data_train['is_cross_sell'] != 'None']



train['is_cross_sell'] = train['is_cross_sell'].astype('int64')



y = train['is_cross_sell']

X = train.drop(['is_cross_sell','order_id'], axis = 1)



X.shape , y.shape

kf = KFold(n_splits=3 , random_state=42, shuffle=False)



smoter = SMOTE(random_state = 42)

    

X_train_cv_upsample = []

y_train_cv_upsample = []

    

    

X_val_cv = []

y_val_cv = []

    

    

for train_fold_index , val_fold_index in kf.split(X , y):

        

    X_train_fold , y_train_fold = X.iloc[train_fold_index], y[train_fold_index]

        

    X_val_fold , y_val_fold = X.iloc[val_fold_index], y[val_fold_index]

        

    X_train_fold_upsample , y_train_fold_upsample = smoter.fit_resample(X_train_fold.values, y_train_fold.values)

        

    # masukan data train yang sudah di oversampling

    X_train_cv_upsample.append(X_train_fold_upsample)

    y_train_cv_upsample.append(y_train_fold_upsample)

        

        

    #masukan data validation

    X_val_cv.append(X_val_fold)

    y_val_cv.append(y_val_fold)



    

X_train_1 = X_train_cv_upsample[0]

y_train_1 = y_train_cv_upsample[0]



X_val_1 = X_val_cv[0]

y_val_1 = y_val_cv[0]
result = []

result_cr  = []

result_auc = []



for index in range(len(X_train_cv_upsample)):

    X_train = X_train_cv_upsample[index]

    y_train = y_train_cv_upsample[index]

                   

    X_val = X_val_cv[index]

    

    y_val = y_val_cv[index]

                   

    mod = ensemble.ExtraTreesClassifier(random_state = 42).fit(X_train , y_train)

    

    # make prediction

    #pred = mod.predict(X_val)

    pred_prob = mod.predict_proba(X_val)

    

    #control threshold 

    threshold = 0.25 # threshold we set where the probability prediction must be above this to be classified as a '1'

    classes = pred_prob[:,1] # say it is the class in the second column you care about predictint

    classes[classes>=threshold] = 1

    classes[classes<threshold] = 0

    

    #cek 

    

    f1 = f1_score(y_val, classes)

    cr = classification_report(y_val , classes)

    auc = roc_auc_score(y_val, classes)

    

    result.append(f1)

    result_cr.append(cr)

    result_auc.append(auc)
result, np.mean(result)
for cr in result_cr:

    print(cr)
print(result_auc), np.mean(result_auc)
X__ = flight__.drop(columns = ['is_cross_sell', 'order_id','account_id'])

y_ = flight__.is_cross_sell
for train_index, test_index in skf.split(X__,y_):

    print('Train:', train_index, 'Validation:', test_index)

    X_train, X_test = X__.iloc[train_index], X__.iloc[test_index] 

    y_train, y_test = y_.iloc[train_index], y_.iloc[test_index]

    

    

    # applying SMOTE to our data and checking the class counts

    X_resampled_3, y_resampled_3 = SMOTE().fit_resample(X_train, y_train)

    X_train_res = pd.DataFrame(X_resampled_3)

    X_train_res.columns = X_train.columns

    model.fit(X_train_res, y_resampled_3)

    

    y_pred = model.predict(X_test)

    probs = model.predict_proba(X_test) # prediction on a new dataset X_new



    threshold = 0.25 # threshold we set where the probability prediction must be above this to be classified as a '1'

    classes = probs[:,1] # say it is the class in the second column you care about predictint

    classes[classes>=threshold] = 1

    classes[classes<threshold] = 0

    

    print()

    print(f1_score(y_test, classes, average='micro'))

    print('Report : ')

    print(classification_report(y_test, classes))

    print()

    print(roc_auc_score(y_test,classes))
# import lightgbm as lgb

# MLA = [

#     #Ensemble Methods

#     ensemble.ExtraTreesClassifier(),

#     ensemble.GradientBoostingClassifier(),

#     ensemble.RandomForestClassifier(),



    

#     #Navies Bayes

#     naive_bayes.BernoulliNB(),

#     naive_bayes.GaussianNB(),

    

#     #Nearest Neighbor

#     neighbors.KNeighborsClassifier(),

    

    

#     #Trees    

#     tree.DecisionTreeClassifier(),

#     tree.ExtraTreeClassifier(),

  

#     #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

#     XGBClassifier()  

#     lgb()

    

#     ]

# evaluasi = pd.DataFrame(columns = ['model'])



# evaluasi['model'] = MLA

# hasil = []



# for model in MLA:

#     # model

#     try:

#         mod = model.fit(X_train_1 , y_train_1)

    

#         # make prediction

#         pred = mod.predict(X_val_1)

#         f1 = f1_score(y_val_1, pred)

#         hasil.append(f1)

#     except:

#         print(model)

    

# evaluasi['f1_score'] = hasil

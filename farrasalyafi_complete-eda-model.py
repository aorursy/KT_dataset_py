#Importing Libraries
#Basic libraries
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import datetime
import math
from datetime import date
from scipy import stats

#Fetaure Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#Imbalance Dataset
from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

#Model Evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, f1_score,auc,roc_curve,roc_auc_score, precision_recall_curve
import scikitplot as skplt

#Modelling Algoritm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Load Data
data = pd.read_csv('../input/should-this-loan-be-approved-or-denied/SBAnational.csv').drop_duplicates()
data.head()
#Melihat info dari data yang kita punya seperti jumlah kolom, input, memorti, tipe data dll
data.info()
#Melihat apakah ada kolom yang inputnya kosong
data.isnull().sum()
#Kita akan merubah tipe kolom yang memiliki tanggal menjadi tipe date/tanggal
date_col = ['ApprovalDate', 'ChgOffDate','DisbursementDate']
data[date_col] = pd.to_datetime(data[date_col].stack(),format='%d-%b-%y').unstack()
#Merubah kolom ApprovalFY menjadi integer, walaupun sebenrnya dia adalah tahun, tapi agar lebih mudah 
data['ApprovalFY'].replace('1976A', 1976, inplace=True)
data['ApprovalFY']= data['ApprovalFY'].astype(int)
#Merubah Kolom Currency menjadi float
curr_col = ['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
data[curr_col] = data[curr_col].replace('[\$,]', '', regex=True).astype(float) 
#Merubah input MIS_Status dari string menjadi integer
data['MIS_Status'] = data['MIS_Status'].replace({'P I F': 0, 'CHGOFF':1})
data.MIS_Status.value_counts()
#Merubah input LowDoc yang string menjadi integer dan merubah input yang tidak sesuai menjadi NaN
data['LowDoc'] = data['LowDoc'].replace({'[C, S, A, R, 1, 0]':np.nan})
data['LowDoc'] = data['LowDoc'].replace({'N': 0, 'Y':1})
data['LowDoc'] = np.where((data['LowDoc'] != 0) & (data['LowDoc'] != 1), np.nan, data.LowDoc)
data.LowDoc.value_counts()
#Menangani input kolom RevLIneCr menjadi integer dan merubah input yang tidak sesuai di Kolom RevLineCr dengan menganggapnya sebagai NaN
data['RevLineCr'] = data['RevLineCr'].replace({'N': 0, 'Y':1, })
data['RevLineCr'] = data['RevLineCr'].replace({'0': 0, '1':1, })
data['RevLineCr'] = np.where((data['RevLineCr'] != 0) & (data['RevLineCr'] != 1), np.nan, data.RevLineCr)
data.RevLineCr.value_counts()
#Merubah kolom NewExist menjadi integer dan menangani yang salah input membuat existing business = 0 dan new business = 1
data['NewExist'] = data['NewExist'].replace({1.0: 0, 2.0:1, 0:np.nan}).fillna(0).astype(int)
data.NewExist.value_counts()
#Kolom UrbanRural sudah sesuai, tidak ada yang perlu di ubah. Kita hanya ingin meliat isinya seperti apa
data.UrbanRural.value_counts()
#Melihat kolom FranchiseCode, berdasarkan guideline
#jika kolom FranchiseCode = 0 atau = 1 maka dia tidak ada frnachise, selain itu maka dia ada franchise
data['FranchiseCode'] = data['FranchiseCode'].replace(1,0 )	
data['FranchiseCode'] = np.where((data.FranchiseCode != 0 ),1,data.FranchiseCode)

#Merubah nama kolom FranchiseCode menjadi Is_Franchised
data.rename(columns={"FranchiseCode": "Is_Franchised"}, inplace=True)
data.Is_Franchised.value_counts()
#Pada kolom CreateJob saya akan merubahnya menjadi categorcal
#jika 0 maka dia tidak membuat job, jika > 0 maka dia membuat job
data['CreateJob'] = np.where((data.CreateJob > 0 ),1,data.CreateJob)
data.rename(columns={"CreateJob": "Is_CreatedJob"}, inplace=True)
data.Is_CreatedJob.value_counts()
#Pad kolom RetainedJob saya akan merubahnya menjadi categorcal
#jika 0 maka dia tidak memiliki karyawan tetap, maka jika >0 maka dia memiliki karyawan
data['RetainedJob'] = np.where((data.RetainedJob > 0 ),1,data.RetainedJob)
data.rename(columns={"RetainedJob": "Is_RetainedJob"}, inplace=True)
data.Is_RetainedJob.value_counts()
#Loan Term dibagi menjadi 2, yakni yang jangka panjang >= 240 bulan (20 tahun) dan < 240 bulan (20 tahun), 
#ini berdasarkan guideline, jika 20 tahun atau diatasnya maka dia dibackup dengan properti jika kurang, maka sebaliknya
data['RealEstate'] = data['Term'].apply(lambda x: 1 if x >= 240 else 0)
#Kita akan membuat kolom baru yakni 'Recession'
#kolom ini berisi apakah si perusahaan ini aktif pada masa resesi dari (1 des 2007 - 30 jun 2009)
#jika aktif maka 1, jika tidak maka 0

#Pertama buat kolom perhitungan untuk merubah Kolom Term menjadi Daysterm dan kolom Active dengan menambahkan kolom
#Daysterm dengan kolom DisbursementDate
data['DaysTerm'] =  data['Term']*30
data['Active'] = data['DisbursementDate'] + pd.TimedeltaIndex(data['DaysTerm'], unit='D')

#Kedua kita aka membuat kolom Recession
startdate = datetime.datetime.strptime('2007-12-1', "%Y-%m-%d").date()
enddate = datetime.datetime.strptime('2009-06-30', "%Y-%m-%d").date()
data['Recession'] = data['Active'].apply(lambda x: 1 if startdate <= x <= enddate else 0)
#Menangani kolom NAICS, kita akan meerubahnya menjadi nama sektornya dan membuat kolom rate default setiap sektornya
#Berdasarkan guideline, dua  digit di awal adalah kode industrinya
ind_code = data['NAICS']

#Fungsi untuk mengambil ambil 2 digit awal dari kodenya
def get_code(ind_code):
    if ind_code <= 0:
        return 0
    return (ind_code // 10 ** (int(math.log(ind_code, 10)) - 1))

#Merubah 2 digit menjadi nama sektor
def sector_name(i):
    def_code = {11:'Agriculture, Forestry, Fishing & Hunting', 21:'Mining, Quarying, Oil & Gas',
                22:'Utilities', 23:'Constuction', 31:'Manufacturing', 32:'Manufacturing', 33:'Manufacturing',
                42:'Wholesale Trade', 44:'Retail Trade', 45:'Retail Trade', 48:'Transportation & Warehousing',
                49:'Transportation & Warehousing', 51:'Information', 52:'Finance & Insurance', 
                53:'Real Estate, Rental & Leasing', 54:'Professional, Scientific & Technical Service',
                55:'Management of Companies & Enterprise', 
                56:'Administrative, Support, Waste Management & Remediation Service',
                61:'Educational Service', 62:'Health Care & Social Assistance',
                71:'Arts, Entertainment & Recreation', 72:'Accomodation & Food Service',
                81:'Other Servieces (Ex: Public Administration)', 92:'Public Administration'
               }
    if i in def_code:
        return def_code[i]
    
def def_rate(i):
    sector_default = {21:0.08, 11:0.09, 55:0.10, 
                      62: 0.10, 22:0.14, 
                      92:0.15,54:0.19, 
                      42:0.19,31:0.19,
                      32:0.16,33:0.14,
                      81:0.20,71:0.21,
                      72:0.22,44:0.22,
                      45:0.23,23:0.23,
                      56:0.24,61:0.24,
                      51:0.25,48:0.27,
                      49:0.23,52:0.28,53:0.29}
    if i in sector_default:
        return sector_default[i]
    return np.nan
#Membuar kolom baru yaitu ind_code
data['ind_code'] = data.NAICS.apply(get_code)

#Memuat kolom baru yaitu Sector_name
data['Sector_name'] = data.ind_code.apply(sector_name)

#Membuat kolom baru yaitu Sector_rate
data['Sector_rate'] = data.NAICS.apply(get_code).apply(def_rate)
#Meliat kolom NAICS, ind_code, Sector_rate, Sector_name untuk memastikan sudah benar atau belum
data[['NAICS','ind_code', 'Sector_rate', 'Sector_name']].head()
#Berdasarkan guideline, kita akan membuat kolom State_rate karena setiap daerah memiliki default rate yang berbeda-beda
#Pertama kita hitung dulu default rate tiap daerah
def_state = data.groupby(['State', 'MIS_Status'])['State'].count().unstack('MIS_Status')
def_state['Def_Percent'] = def_state[1]/(def_state[1] + def_state[0])
def_state = def_state.drop(axis=1, columns=[0.0, 1.0]).round(1).to_dict()

#Kedua,membuat fungsi merubah daerah tersebut sesuai dengan default ratenya
def state_def_rate(i):
    def_state = {'AK': 0.1,'AL': 0.2, 'AR': 0.2, 'AZ': 0.2,'CA': 0.2, 'CO': 0.2, 'CT': 0.1, 'DC': 0.2,
                 'DE': 0.2, 'FL': 0.3, 'GA': 0.2, 'HI': 0.2, 'IA': 0.1, 'ID': 0.1, 'IL': 0.2, 'IN': 0.2, 
                 'KS': 0.1, 'KY': 0.2, 'LA': 0.2, 'MA': 0.1, 'MD': 0.2, 'ME': 0.1, 'MI': 0.2, 'MN': 0.1,
                 'MO': 0.2, 'MS': 0.2, 'MT': 0.1, 'NC': 0.2, 'ND': 0.1, 'NE': 0.1, 'NH': 0.1, 'NJ': 0.2,
                 'NM': 0.1, 'NV': 0.2, 'NY': 0.2, 'OH': 0.2, 'OK': 0.2, 'OR': 0.2, 'PA': 0.1, 'RI': 0.1,
                 'SC': 0.2, 'SD': 0.1, 'TN': 0.2, 'TX': 0.2, 'UT': 0.2, 'VA': 0.2, 'VT': 0.1, 'WA': 0.1,
                 'WI': 0.1, 'WV': 0.2, 'WY': 0.1}

    if i in def_state:
        return def_state[i]
    
#Ketiga membuat kolom State_rate    
data['State_rate'] = data.State.apply(state_def_rate)
#Memastikan Kolom State dan State_rate sudah sesuai
data[['State', 'State_rate']].head(10)
#Membuat kolom Portion SBA Aproved Loan
#kolom ini berisi persen antara jaminan yang diberikan dari SBA dibandingkan dengan pinjaman dari bank
data['Portion_SBA_Bank'] = data['SBA_Appv'] / data['GrAppv']
#Menurut guideline, data ini diambil dari tahun 1987 - 2014, namun karena kita diminta untuk memasukkan
#atau membuat kolom baru yakni Recession yang artinya pinjamanya harus melewati massa resessi pada tahun 2007 sampai 2009
#sehingga data yang diambil hanya sampai tahun 2010 karena rerata lama pinjaman hanya selama 5 tahun atau lebih
data = data[data['DisbursementDate'] <= pd.Timestamp(2010, 12, 31)]
#Kita melihat lagi semua dataset kita
data.info()
#Melihat berapa banyak data yang kosong
data.isnull().sum()
#Kita akan menghilangkan input yang kosong pad kolom NewExist dengan menggunakan kolom Is_Frenchised
#kita berasumsi jika Is_Franchised = 0 maka dia New Business, karena biasanya usaha baru tidak punya Franchise
#jika Is_Franchised = 1 maka dia existing business,kemungkinan dia punya franchise 
# sekarang kita cek asumsi kita, apakah benar dengan melihat perbandingan dua kolom tersebut
data[['NewExist', 'Is_Franchised']].head(10)

#ternyata asumsi kita salah, sehingga, kita akan drop saja input yang kosong ini, selain asumsi kita salah
# dana juga input yang kosong terbilang sangat kecil dibanding dengan totoal jumlah data kita
#Kita akan mencoba mengisi input yang kosong pada kolom LowDoc
#berdasrkan guideline, jika pinjaman < 150.000 maka dia 'Yes' dan jika pinjaman > 150.000 maka dia 'No'
# dan juga ada beberap input yang kami jadikan Nan jika diluar 'Yes' dan 'No'
#untuk mengisinya, kita akan menggunakan kolom DisbursementGross

data['LowDoc'] = np.where((data['LowDoc'] == np.nan) & (data['DisbursementGross'] < 150000),1,data.LowDoc)
data['LowDoc'] = np.where((data['LowDoc'] == np.nan) & (data['DisbursementGross'] >= 150000),0,data.LowDoc)

data = data[(data['LowDoc'] == 0) | (data['LowDoc'] == 1)]
#Kita cek lagi kolom LowDoc untuk memastikan
data.LowDoc.value_counts()
#Mengisi input yang kosong pada MIS_Status dengan menggunakan kolom CghOffDate
#jika dia ada tanggal di ChgOffDate, maka dia statusnya CHGOFF, jika tidak maka kosong tanggalnya
data['MIS_Status'] = np.where((data['MIS_Status'] == 0.0) & (data['ChgOffDate'] == np.nan),0,data.MIS_Status)
data['MIS_Status'] = np.where((data['MIS_Status'] == 1.0) & (data['ChgOffDate'] != np.nan),1,data.MIS_Status)

data = data[(data['MIS_Status'] == 0) | (data['MIS_Status'] == 1)]
#Kita cek lagi apakah sudah benar kolom MIS_Status dengan ChgOffDate
print(data[['MIS_Status', 'ChgOffDate']].head(10))
#Cek kembali kolom MIS_status
data.MIS_Status.value_counts()
#Kita aka drop kolom yang masih ada input yang kosong karena tidak ada gunanya dan sudah digantikan dengan
#kolom yang lainya untuk dilakukan EDA sebelum dipilih lagi mana kolom yang akan dimasukkan ke model
#berdasarkan hubunganya dengan target atau seberapa berdampaknya terhadap target
data = data.drop(axis=1, columns=['Name','Bank','NAICS', 'BankState',
                                  'ChgOffDate','ind_code', 'Active', 'DaysTerm'])
#Input yang hilang pada kolom LowDoc da MIS_Status tidak bisa diinput dengan kondisi yang telah dibuat
#sehingga kita drop rownya
data.dropna(subset=['City', 'State','LowDoc', 'MIS_Status', 
                    'Sector_rate', 'Sector_name', 'RevLineCr'], inplace=True)
#Kita cek kembali apakah masih ada kolom yang inputnya kosong
data.isnull().sum()
#Kita akan menyesuaikan tipe data dengan input datanya
data = data.astype({'UrbanRural': 'object', 
                    'RevLineCr': 'int64', 
                    'LowDoc':'int64', 
                    'MIS_Status':'int64'})
#Membuat plot jumlah pinjaman setiap tahunya
f, ax = plt.subplots(figsize=(16,9))
sns.barplot(x="ApprovalFY", y="DisbursementGross", color='Salmon', data=data)
plt.title('Jumlah Pinjaman Setiap Tahun', fontsize=20)
plt.xlabel('Tahun', fontsize=15)
plt.ylabel('Jumlah Pinjaman ($)', fontsize=15)
data.DisbursementGross.describe()
#Membuat plot jumlah pinjaman berdasarkan sektornya
f, ax = plt.subplots(figsize=(16,9))
sns.barplot(x="DisbursementGross", y="Sector_name", data=data)
plt.title('Jumlah Pinjaman Berdasarkan Sektor', fontsize=20)
plt.xlabel('Jumlah Pinjaman ($)', fontsize=15)
plt.ylabel('Nama Sektor', fontsize=15)
data.groupby('Sector_name')['DisbursementGross'].describe().style.highlight_max(color='green').highlight_min(color='blue')
#Melihat jumlah yang bayar dan gagal bayar setiap tahunya
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(x="ApprovalFY", data=data,hue='MIS_Status')
plt.title('Jumlah Bayar dan Gagal Bayar Setiap Tahun', fontsize=20)
plt.xlabel('Tahun', fontsize=15)
plt.ylabel('Jumlah Pinjaman ($)', fontsize=15)
plt.legend(["Tidak", "Gagal"],loc='upper right')
#Melihat jumlah yang bayar & gagal bayar pada setiap sektor
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Sector_name", hue="MIS_Status", data=data)
plt.title('Jumlah Gagal Bayar Bedasarkan Sektor', fontsize=20)
plt.xlabel('Jumlah Gagal Bayar', fontsize=15)
plt.ylabel('Nama Sektor', fontsize=15)
plt.legend(["Tidak", "Gagal"],loc='lower right')
pd.DataFrame(data.groupby('Sector_name')['MIS_Status'].value_counts()).unstack(level=1).style.highlight_max(color='green').highlight_min(color='blue')
#Membuat plot sektor yang aktif saat resesi global tahun 2008
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Sector_name", hue="Recession", data=data)
plt.title('Jumlah Aktif Saat Resesi Bedasarkan Sektor', fontsize=20)
plt.xlabel('Jumlah Aktif', fontsize=15)
plt.ylabel('Nama Sektor', fontsize=15)
plt.legend(["Tidak", "Aktif"],loc='lower right')
#Meliat sektor mana saja yang memiliki jaminan properti
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Sector_name", hue="RealEstate", data=data)
plt.title('Jumlah Sector Yang Memiliki Jaminan Properti', fontsize=20)
plt.xlabel('Jumlah', fontsize=15)
plt.ylabel('Nama Sektor', fontsize=15)
plt.legend(["Tidak", "Punya"],loc='lower right')
plt.show()
#Melihat lebih detail sektor dengan jaminan properti
pd.DataFrame(data.groupby('Sector_name')['RealEstate'].value_counts()).unstack(level=1).style.highlight_max(color='green').highlight_min(color='blue')
#Melihat lama pinjaman
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(data['Term'])
plt.title('Lama Pinjaman', fontsize=20)
plt.xlabel('Bulan', fontsize=15)
#Melihat detail pinjaman
data['Term'].describe() 
#Melihat pinjaman berdasarkan sektor
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x="Term", y="Sector_name", data=data)
plt.title('Lama Pinjaman', fontsize=20)
plt.xlabel('Bulan', fontsize=15)
plt.ylabel('Nama Sektor', fontsize=15)
#Detail pinjaman berdasarkan sektor
data.groupby('Sector_name')['Term'].describe().style.highlight_max(color='green').highlight_min(color='blue')
#Melihat jumlah lapangan pekergja setiap tahun
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(x="ApprovalFY", data=data,hue='Is_CreatedJob')
plt.title('Jumlah Pembuat Lapangan Pekerjaan Setiap Tahun', fontsize=20)
plt.xlabel('Tahun', fontsize=15)
plt.ylabel('Jumlah Lapangan Pekerjaan', fontsize=15)
plt.legend(["Tidak", "Membuat"],loc='upper right')
#Jumlah lapangan kera berdasrkan sektor
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Sector_name", hue="Is_CreatedJob", data=data)
plt.title('Jumlah Lapangan Pekerja Berdasarkan Sektor', fontsize=20)
plt.xlabel('Jumlah Lapangan Pekerja', fontsize=15)
plt.ylabel('Nama Sektor', fontsize=15)
plt.legend(["Tidak", "Ada"],loc='lower right')
#Detail Setiap sektor lapangan kerja
pd.DataFrame(data.groupby('Sector_name')['Is_CreatedJob'].value_counts()).unstack(level=1).style.highlight_max(color='green').highlight_min(color='blue')
#Melihat jumlah bisnis baru dan lama yang ikut SBA setiap tahun
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(x="ApprovalFY", data=data,hue='NewExist')
plt.title('Jumlah Bisnis Baru dan Lama Setiap Tahun', fontsize=20)
plt.xlabel('Tahun', fontsize=15)
plt.ylabel('Jumlah Bisnis', fontsize=15)
plt.legend(["Lama", "Baru"],loc='upper right')
#Membuat plot Jumla bisnis baru dan lama berdasarkan sektor
f, ax = plt.subplots(figsize=(16,9))
sns.countplot(y="Sector_name", hue="NewExist", data=data)
plt.title('Jumlah Bisnis Baru Atau Lama Berdasarkan Sektor', fontsize=20)
plt.xlabel('Jumlah Bisnis Baru', fontsize=15)
plt.ylabel('Nama Sektor', fontsize=15)
plt.legend(["Lama", "Baru"],loc='lower right')
#Detail Jumla bisnis baru dan lama berdasarkn sektor
pd.DataFrame(data.groupby('Sector_name')['NewExist'].value_counts()).unstack(level=1).style.highlight_max(color='green').highlight_min(color='blue')
#kita akan membuang kolom-kolom yang danggap tidak penting
# kolom LoanNr_ChkDgt tidak penting karena hanya id dari peminjam sudah digantikan dengan index
# kolom City, State, UrbanRural dan ZIP tidak perlu karena sudah kita ubah menjadi state rate
# kolom bank dan bank satet juga tidak terlalu penting
# kolom NAICS karen sudah digantikan dengan Sector_rate
# kolom ApprovalDate dan ApprovalFY karena hanya pencataan tanggal saja
# kolom Term dihapus karena sudah digantikan dengan RealEstate
# kolom UrbanRural karena tidak mempengaruhi target
# kolom LowDoc karena suda ada Disbursement Gross, LowDoc hanya dikelompokkan saja secara administartif
# kolom Active dan DaysTerm karena sudah digantikan dengan Recession
# kolom ind_code karena sudah ada Secator_rate
# kolom ChgOffDate karena dia sebernya sama dengan MIS_Status
# kolom DisbursementDate karena hanya tanggan pembayaran
# kolom SBA_Appv karena sudah digatikan dengan Portion_SBA_Bank
# kolom DisbursementDate sudah tidak digunakan lagi
# kolom Sector_name suda tidak digunakan lagi
data = data.drop(axis =1, columns = ['LoanNr_ChkDgt','City','State', 'Zip', 'UrbanRural', 'LowDoc',
                                    'ApprovalDate', 'ApprovalFY', 'SBA_Appv','DisbursementDate', 
                                     'Sector_name','BalanceGross', 'ChgOffPrinGr'])
#Kita akan menggunakan Inter Quartile Range untuk menangani ouliers
#Menentukan Limit
def limit(i):
    Q1 = data[i].quantile(0.25)
    Q3 = data[i].quantile(0.75)
    IQR = Q3 - Q1
    
    #menentukan upper limit biasa dan upper limit ekstim
    lower_limit = data[i].quantile(0.25) - (IQR * 1.5)
    lower_limit_extreme = data[i].quantile(0.25) - (IQR * 3)
    upper_limit = data[i].quantile(0.75) + (IQR * 1.5)
    upper_limit_extreme = data[i].quantile(0.75) + (IQR * 3)
    print('Lower Limit:', lower_limit)
    print('Lower Limit Extreme:', lower_limit_extreme)
    print('Upper Limit:', upper_limit)
    print('Upper Limit Extreme:', upper_limit_extreme)

#Mengitung persen outliers dari data    
def percent_outliers(i):
    Q1 = data[i].quantile(0.25)
    Q3 = data[i].quantile(0.75)
    IQR = Q3 - Q1
    
    #menentukan upper limit biasa dan upper limit ekstim
    lower_limit = data[i].quantile(0.25) - (IQR * 1.5)
    lower_limit_extreme = data[i].quantile(0.25) - (IQR * 3)
    upper_limit = data[i].quantile(0.75) + (IQR * 1.5)
    upper_limit_extreme = data[i].quantile(0.75) + (IQR * 3)
    #melihat persenan outliers terhadap total data
    print('Lower Limit: {} %'.format(data[(data[i] >= lower_limit)].shape[0]/ data.shape[0]*100))
    print('Lower Limit Extereme: {} %'.format(data[(data[i] >= lower_limit_extreme)].shape[0]/data.shape[0]*100))
    print('Upper Limit: {} %'.format(data[(data[i] >= upper_limit)].shape[0]/ data.shape[0]*100))
    print('Upper Limit Extereme: {} %'.format(data[(data[i] >= upper_limit_extreme)].shape[0]/data.shape[0]*100))
#Kita cek kolom DisbursemntGross
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=data['DisbursementGross'])
plt.title('DisbursementGross Ouliers', fontsize=20)
plt.xlabel('Jumlah', fontsize=15)
#kita akan cek limit outliers dan berapa persen dari data kita yang melebihi limit tersebut
print(limit('DisbursementGross'))
print('-'*50)
print(percent_outliers('DisbursementGross'))
#karena terdapat 10 % dari jumlah data yang kita punya, maka saya coba merubah datanya dengan menggunakan
#log transformation, karena jika ouliers dihilangkan sangat banyak sekali data yang hilang (10%)
data['DisbursementGross'] = np.log(data['DisbursementGross'])
data['DisbursementGross'].skew()
#kita akan cek limit outliers dan berapa persen dari data kita yang melebihi limit tersebut
print(limit('DisbursementGross'))
print('-'*50)
print(percent_outliers('DisbursementGross'))
#ternyata masih ada sekitar 1% outliers, karena jumlahnya terbilang kecil, maka kita drop saja
outliers1_drop = data[(data['DisbursementGross'] > 14.9)].index
data.drop(outliers1_drop, inplace=True)
#kita cek lagi apakah masiha ada outliers
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=data['DisbursementGross'])
plt.title('DisbursementGross Ouliers', fontsize=20)
plt.xlabel('Jumlah', fontsize=15)
#kita cek pada kolom GrAppv apakah ada outliers
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=data['GrAppv'])
plt.title('GrAppv Outliers', fontsize=20)
plt.xlabel('Jumlah', fontsize=15)
#kita akan cek limit outliers dan berapa persen dari data kita yang melebihi limit tersebut
print(limit('GrAppv'))
print('-'*50)
print(percent_outliers('GrAppv'))
data['GrAppv'] = np.log(data['GrAppv'])
data['GrAppv'].skew()
#kita akan cek limit outliers dan berapa persen dari data kita yang melebihi limit tersebut
print(limit('GrAppv'))
print('-'*50)
print(percent_outliers('GrAppv'))
#ternyata masih ada sekitar 1% outliers, karena jumlahnya terbilang kecil, maka kita drop saja
outliers2_drop = data[(data['GrAppv'] < 7.5)].index
data.drop(outliers2_drop, inplace=True)
#kita cek lagi pada kolom GrAppv apakah masih ada outliers
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=data['GrAppv'])
plt.title('GrAppv Outliers', fontsize=20)
plt.xlabel('Jumlah', fontsize=15)
#kita cek ouliers pada kolom NoEmp
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=data['NoEmp'])
plt.title('NoEmp Ouliers', fontsize=20)
plt.xlabel('Jumlah', fontsize=15)
#kita akan cek limit outliers dan berapa persen dari data kita yang melebihi limit tersebut
print(limit('NoEmp'))
print('-'*50)
print(percent_outliers('NoEmp'))
#pada kolom NoEmp, terdapat iput 0, aka saya anggap ini kesalahan, input, karena tidak mungkin sebuah perusahaan
#tidak memiliki karyawan
wrong_input = data[(data['NoEmp'] == 0)].index
data.drop(wrong_input, inplace=True)
#melakukan boxcox transformasi karena semua metode tela saya coba namun ini yang paling baik hasilnya
data['NoEmp']= stats.boxcox(data['NoEmp'])[0]
data['NoEmp'].skew()
#kita akan cek lagi limit outliers dan berapa persen dari data kita yang melebihi limit tersebut
print(limit('NoEmp'))
print('-'*50)
print(percent_outliers('NoEmp'))
#ternyata masih ada sekitar 0.02% outliers, karena jumlahnya terbilang kecil, maka kita drop saja
outliers3_drop = data[(data['NoEmp'] > 3.3)].index
data.drop(outliers3_drop, inplace=True)
#kita cek ouliers lagi pada kolom NoEmp
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=data['NoEmp'])
plt.title('NoEmp Ouliers', fontsize=20)
plt.xlabel('Jumlah', fontsize=15)
#kita cek ouliers pada kolom Term
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=data['Term'])
plt.title('Term Ouliers', fontsize=20)
plt.xlabel('Bulan', fontsize=15)
#Karena terdapat data yang lama pinajamanya 0 bulan dan 43 tahun 
#karena tidak mungkin ada waktu 0 dan 569 bulan (43 tahun) 
#sedangkan pengambilan data yang kita punya hanya dari tahun 1987–2010 (23 tahun),
#sehingga minimal 5 tahun (75 bulan) atau maksimal 23 tahun (276 bulan)
wrong_input_2 = data[(data['Term'] < 75)].index
wrong_input_3 = data[(data['Term'] > 276)].index
data.drop(wrong_input_2, inplace=True)
data.drop(wrong_input_3, inplace=True)
#kita cek lagi ouliers pada kolom Term
f, ax = plt.subplots(figsize=(16,9))
sns.boxplot(x=data['Term'])
plt.title('Term Ouliers', fontsize=20)
plt.xlabel('Bulan', fontsize=15)
#kita akan cek lagi limit outliers dan berapa persen dari data kita yang melebihi limit tersebut
print(limit('Term'))
print('-'*50)
print(percent_outliers('Term'))
#kita tidak akan drop outlier ini, karena selain jumlanya banyak (18%) ini belum tentu salah input, karena memang
#beberapa industri bisa mengambil jangka waktu pinjaman yang lama seperti oil & gas tau mining
#karena data memiliki jumlah input yang sangat banyak, maka saya akan menggunakan teknik feature importance pada fetaure selection
#kita akan memisahkan dulu independen dan dependen featurenya
#data = data.reset_index(drop=True) #reset index dulu biar urut indexnya
y = data['MIS_Status']
X = data.drop(columns=['MIS_Status'], axis=1)

#kita coba menggunakan fetaure importance pada model XGboost
model = XGBClassifier()
model.fit(X,y)

#Kita visualisasi feature yang penting-penting
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
f, ax = plt.subplots(figsize=(16,9))
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance', fontsize=20)
plt.ylabel('Features', fontsize=15)
plt.xlabel('Score', fontsize=15)
plt.show()
#Berdasarkan feature selection diatas, kita akan pilih, feature-feature tersebut dan membuang
#feature-feature yang tidak relefan dengan target
data = data[['RevLineCr', 'Term', 'Portion_SBA_Bank', 'GrAppv', 'State_rate', 'DisbursementGross',
            'Is_RetainedJob', 'Sector_rate', 'Recession', 'Is_CreatedJob', 'MIS_Status']]
data.shape
#Kita cek ratio target variabel
print(data.MIS_Status.value_counts())
print('-'*50)
print('MIS_Status (0): {} %'.format(data[(data['MIS_Status'] == 0)].shape[0]/data.shape[0]*100))
print('MIS_Status (1): {} %'.format(data[(data['MIS_Status'] == 1)].shape[0]/data.shape[0]*100))
#Visualisasi Imbalance Dataset Sebelum Dibenahi
sns.countplot("MIS_Status",data=data)
#pertama kita akan membagi data menjadi train dan test, namun perlu diingat, jika target data kita imbalance
#sehingga kita membagi data di traindan testnya harus sesuai, jadi tidak boleh dalam pembagian datanya ada yang 
#hanya berisi 0 atau yang mayoritas aja, makanya kita menggunakan stratify=y
y = data['MIS_Status']
X = data.drop(columns=['MIS_Status'], axis=1)
scale = StandardScaler()
X_scaled = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=27, stratify=y) #jangan lupa untuk stratify
#Disini saya mengggunaka SMOTE dan kemudian di undersampling lagi
over = SMOTE(sampling_strategy='minority')
under = RandomUnderSampler()
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

#sekarang kita fit ke training data kita
X_train, y_train = pipeline.fit_resample(X_train, y_train)
#Membuat fungsi yang nanti sekalin bisa training dan tes kemudian dievalusi
def model_eval(algo,X_train,y_train,X_test,y_test):
    algo.fit(X_train,y_train)
    y_train_ypred = algo.predict(X_train)
    y_train_prob = algo.predict_proba(X_train)[:,-1]

    #TEST

    y_test_ypred = algo.predict(X_test)
    y_test_prob = algo.predict_proba(X_test)[:,-1]
    y_probas = algo.predict_proba(X_test)
    
    #Confussion Matrix
    plot_confusion_matrix(algo, X_test, y_test)
    plt.show() 
    print('='*100)
    print('Classification Report: \n', classification_report(y_test, y_test_ypred, digits=3))
    print('='*100)
    
    #ROC Curve
    #fpr,tpr,thresholds = roc_curve(y_test,y_test_prob)
    skplt.metrics.plot_roc(y_test, y_probas,figsize=(16,9) )
    
    #PR Curve
    skplt.metrics.plot_precision_recall(y_test, y_probas, figsize=(16,9))
    plt.show()
#Using Logistic Regression
lr = LogisticRegression()
model_eval(lr,X_train,y_train,X_test,y_test)
#sekarang kita coba menggunakan Naive Bayes
nb = GaussianNB()
model_eval(nb,X_train,y_train,X_test,y_test)
#Sekarang kita coba menggunakan KNN
knn = KNeighborsClassifier()
model_eval(knn,X_train,y_train,X_test,y_test)
#Sekarang kita coba Random Forest
rf = RandomForestClassifier()
model_eval(rf,X_train,y_train,X_test,y_test)
#Sekarang kita coba menggunakan XGBoost
xgb = XGBClassifier()
model_eval(xgb,X_train,y_train,X_test,y_test)
#kita membuat function opmitasi, disinia saya menggunakan GridSearchCV
def model_opt(clf, params,X_train,y_train,X_test,y_test ):
    # Load GridSearchCV
    search = GridSearchCV(estimator=clf,
                          param_grid=params,
                          scoring = 'f1',
                          n_jobs = -1,
                          cv = 3,
                          verbose=True)

    # Train search object
    search.fit(X_train, y_train)
    
    best = search.best_estimator_
    best_model = best.fit(X_train, y_train)
    
    #### TEST

    y_test_ypred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:,-1]
    y_probas = best_model.predict_proba(X_test)
    
    print('Best parameters: \n',search.best_params_)
    print('='*70)
    #Confussion Matrix
    plot_confusion_matrix(algo, X_test, y_test)
    plt.show() 
    print('='*100)
    print('Classification Report: \n', classification_report(y_test, y_test_ypred, digits=3))
    print('='*100)
    
    #ROC Curve
    #fpr,tpr,thresholds = roc_curve(y_test,y_test_prob)
    skplt.metrics.plot_roc(y_test, y_probas,figsize=(16,9) )
    
    #PR Curve
    skplt.metrics.plot_precision_recall(y_test, y_probas, figsize=(16,9))
    plt.show()
params ={"learning_rate"    : [0.05, 0.10, 0.15] ,
         "max_depth"        : [ 3, 4, 5, 6],
         "min_child_weight" : [ 1, 3, 5, 7 ],
         "gamma"            : [ 0.0, 0.1, 0.2 ],
         "colsample_bytree" : [ 0.3, 0.4, 0.5] }

#Saya sudah mencari dengan GridSearcCV dan menemukan parameterterbaik
{'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.15, 'max_depth': 6, 'min_child_weight': 1}

#sekarnag kita coba pada model setelah ditunning
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bynode=1, colsample_bytree=0.5, gamma=0.0,
                    learning_rate=0.15, max_delta_step=0, max_depth=6,
                    min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                    nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    silent=None, subsample=1, verbosity=1)

model_eval(xgb,X_train,y_train,X_test,y_test)
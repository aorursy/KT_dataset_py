%load_ext autoreload
%autoreload 2

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib
from fastai.imports import *
from fastai.structured import *
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB  # Gaussian naive Bayes classifier
from sklearn.metrics import confusion_matrix
#dict = {}
#dict["now"] = 1
df = pd.read_csv("../input/Interview.csv")
headers = df.dtypes.index
print(len(headers))

#drop unnamed list with NaN value
drop_list = ['Unnamed: 23', 'Unnamed: 24','Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27']
df.drop(drop_list,axis = 1,inplace = True)
#get list of all column names
#headers = df.dtypes.index.count()
#Nama column sangat panjang
#print(headers)
#print(len(df.columns))
#Normalisasi nama kolom
df.columns = ['Date','Name','Industry','Location','Position','Skillset','Interview_Type','ID','Gender','Cand_Location',
              'Job_Location','Venue','Nat_Location','Permission','Hope','Three_Hours','Alternate_Number','Printout','Clear',
              'Shared','Expected','Observed','Status']
headers = df.dtypes.index
print(headers)
#Check unique dataset
#for c in df.columns:
#    print(c)
#    print(df[c].unique())
#Format data sangat tidak beraturan, mari pertama kita lihat variabel Date of Interview
#print(df['Date'].unique())
"""
    Tedapat beberapa ketidaknormalan pada variabel Date of Interview, kita harus normalisasi
    variabel ini, misal bentuk normalnya yaitu date/month/year
"""

#Normalisasi Date of Interview
#Buang 'jam'  interview
def normalize_date(date):
    date = date.str.strip()
    date = date.str.split("&").str[0]
    date = date.str.replace('–', '/')
    date = date.str.replace('.', '/')
    date = date.str.replace('Apr', '04')
    date = date.str.replace('-', '/')
    date = date.str.replace(' ', '/')
    date = date.str.replace('//+', '/')
    return date

df['Date'] = normalize_date(df['Date'])
#Baris ke 1233 semuanya NaN , jadi kita buang aja
df.drop(df.index[[1233]], inplace = True)
#print(df.ix[1233])

#Bikin kolom baru year month dan date untuk nanti digabungin lagi
df['Day'] = df['Date'].str.split("/").str[0]
df['Month'] = df['Date'].str.split("/").str[1]
df['Year'] = df['Date'].str.split("/").str[2]
df['Year'].replace(['16','15'],['2016','2015'],inplace  = True)
df['date'] = pd.to_datetime(pd.DataFrame({'year' : df['Year'],'month' : df['Month'],'day'  : df['Day']}),format = '%Y-%m-%d')
df.drop(['Date','Day','Month','Year'],axis = 1,inplace = True)
df.rename(columns = {'date':'Date'},inplace = True)
df = df[df['Date'] < '2018-01-01']
print(df['Date'].unique())
#print(df['Date'])

#Ubah jadi lowercase semua
df = df.apply(lambda x: x.astype(str).str.lower())
#Normalisasi Name
#print(df['Name'].value_counts())
#Standart Chartered Bank Chennai = Standard Chartered Bank

#Normalisasi Industry
print(df['Industry'].value_counts())
#IT Products and Services = IT Services = IT
df['Industry'].replace('it products and services','it')
df['Industry'].replace('it services','it')



#Normalisasi Skillset
df = df[df.Skillset.str.contains("am") == False]
df = df[df.Skillset.str.contains("pm") == False]
idx = df.Skillset.str.contains("java")
df.Skillset.loc[idx] = "java"
idx = df.Skillset.str.contains("sccm")
df.Skillset.loc[idx] = "sccm"
idx = df.Skillset.str.contains("ra")
df.Skillset.loc[idx] = "ra"
idx = df.Skillset.str.contains("lending")
df.Skillset.loc[idx] = "l & l"
idx = df.Skillset.str.contains("analytical")
df.Skillset.loc[idx] = "r&d"
idx = df.Skillset.str.contains("tech")
df.Skillset.loc[idx] = "technical lead"
idx = df.Skillset.str.contains("tl")
df.Skillset.loc[idx] = "technical lead"
print(df['Skillset'].unique())
#Java,AM,PM,sccm,ra

#Normalisasi Interview_Type
idx = df.Interview_Type.str.contains("scheduled walk in")
df.Interview_Type.loc[idx] = "scheduled walkin"
print(df['Interview_Type'].unique())

#Normalisasi Permission
idx = df.Permission.str.contains("nan")
df.Permission.loc[idx] = "not sure"
idx = df.Permission.str.contains("na")
df.Permission.loc[idx] = "not sure"
idx = df.Permission.str.contains("not yet")
df.Permission.loc[idx] = "not sure"
idx = df.Permission.str.contains("yet to confirm")
df.Permission.loc[idx] = "not sure"
print(df['Permission'].unique())

#Normalisasi Hope
idx = df.Hope.str.contains("nan")
df.Hope.loc[idx] = "not sure"
idx = df.Hope.str.contains("na")
df.Hope.loc[idx] = "not sure"
idx = df.Hope.str.contains("cant say")
df.Hope.loc[idx] = "not sure"
print(df['Hope'].unique())

#Normalisasi Three_Hours
idx = df.Three_Hours.str.contains("nan")
df.Three_Hours.loc[idx] = "not sure"
idx = df.Three_Hours.str.contains("na")
df.Three_Hours.loc[idx] = "not sure"
idx = df.Three_Hours.str.contains("no dont")
df.Three_Hours.loc[idx] = "no"
print(df['Three_Hours'].unique())

#Normalisasi Alternate_Number
idx = df.Alternate_Number.str.contains("nan")
df.Alternate_Number.loc[idx] = "no"
idx = df.Alternate_Number.str.contains("na")
df.Alternate_Number.loc[idx] = "no"
idx = df.Alternate_Number.str.contains("hav")
df.Alternate_Number.loc[idx] = "no"
print(df['Alternate_Number'].unique())

#NormalisasiPrintout
idx = df.Printout.str.contains("nan")
df.Printout.loc[idx] = "not sure"
idx = df.Printout.str.contains("na")
df.Printout.loc[idx] = "not sure"
idx = df.Printout.str.contains("no-")
df.Printout.loc[idx] = "no"
idx = df.Printout.str.contains("not yet")
df.Printout.loc[idx] = "not sure"
print(df['Printout'].unique())

#Normalisasi Clear
idx = df.Clear.str.contains("nan")
df.Clear.loc[idx] = "no"
idx = df.Clear.str.contains("no-")
df.Clear.loc[idx] = "no"
idx = df.Clear.str.contains("na")
df.Clear.loc[idx] = "no"
print(df['Clear'].unique())

#Normalisasi Shared
idx = df.Shared.str.contains("nan")
df.Shared.loc[idx] = "not sure"
idx = df.Shared.str.contains("na")
df.Shared.loc[idx] = "not sure"
idx = df.Shared.str.contains("not yet")
df.Shared.loc[idx] = "not sure"
idx = df.Shared.str.contains("yet to check")
df.Shared.loc[idx] = "not sure"
idx = df.Shared.str.contains("need to check")
df.Shared.loc[idx] = "not sure"
idx = df.Shared.str.contains("havent")
df.Shared.loc[idx] = "not sure"
print(df['Shared'].unique())

#Normalisasi Expected
idx = df.Expected.str.contains("nan")
df.Expected.loc[idx] = "not sure"
idx = df.Expected.str.contains("uncertain")
df.Expected.loc[idx] = "not sure"
idx = df.Expected.str.contains("10.30 am")
df.Expected.loc[idx] = "yes"
print(df['Expected'].unique())

#Normalisasi Observed
idx = df.Observed.str.contains("yes ")
df.Observed.loc[idx] = "yes"
idx = df.Observed.str.contains("no ")
df.Observed.loc[idx] = "no"
display(df.head().T)
print(df['Observed'].unique())
df['Observed'].replace(['yes', 'no'],[1, 0], inplace = True)
#print(df.tail())
model_df = df
#Transformasi variable Date
add_datepart(model_df, 'Date')
#Trasforamasi semua data menjadi numerik
train_cats(model_df)

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)

#display_all(model_df.tail().T)
#print(df.Observed)

df, y, nas = proc_df(model_df, "Observed")
display_all(df.tail().T)

#Bagi data menjadi 70% data training dan 30% data test
n_valid = int(len(model_df) * .3)
n_trn = len(model_df) - n_valid
X_train, X_valid = df[:n_trn].copy(),df[n_trn:].copy()
y_train, y_valid = y[:n_trn].copy(),y[n_trn:].copy()

#Classify
m_base = RandomForestClassifier(n_jobs = -1, oob_score=True)
%time m_base.fit(X_train, y_train)
print("Training Acc:", round(m_base.score(X_train, y_train),5)),
print("Validation Acc:", round(m_base.score(X_valid, y_valid),5)),
print("Out-of-Bag Acc:", round(m_base.oob_score_, 5))
draw_tree(m_base.estimators_[0], df, precision = 3)

feature_imp = rf_feat_importance(m_base, df)

def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend = False)

plot_fi(feature_imp)
print(feature_imp)
feature_imp = feature_imp[feature_imp.imp > 0.007]
df = df[feature_imp.cols].copy()
#df.drop("ID",axis = 1,inplace = True)

#Over-fitting
m_base = RandomForestClassifier(max_features='log2',min_samples_leaf=5,n_jobs = -1, oob_score=True)
X_train, X_valid = df[:n_trn].copy(),df[n_trn:].copy()
y_valid = np.reshape(y_valid, (-1, 1))
%time m_base.fit(X_train, y_train)
draw_tree(m_base.estimators_[0], df, precision = 3)
#prediction = m_base.predict(y_valid)
#confusion_matrix(y_valid, prediction)
print("Training Acc:", round(m_base.score(X_train, y_train),5)),
print("Validation Acc:", round(m_base.score(X_valid, y_valid), 5)),
print("Out-of-Bag Acc:", round(m_base.oob_score_, 5))




























































































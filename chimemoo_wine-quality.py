# Import semua library yang dibutuhkan

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualisation (2-D)

%matplotlib inline

import seaborn as sns # data visualisation (3-D)

plt.style.use('seaborn') # set style for graph
# Import file dataset

rawdata = pd.read_csv('//kaggle/input/winequality-white.csv')
# Menampilkan data

display(rawdata.head(), rawdata.shape) 
# Menampilkan informasi pada dataset

# Seperti jumlah missing value, dan type field

rawdata.info()
# Mendapatkan deskripsi dari dataset berupa count, mean, std, min, q1, q2, q3, dan max

# pada masing-masing field

rawdata.describe()
# Mengecek nilai kolerasi antar masing-masing field, untuk melihat pesebaran datanya

rawdata.corr()
# Menampilkan hasil visualisasi data Korelasi antar masing-masing field

plt.figure(figsize = (10, 8))

sns.heatmap(rawdata.corr(), square = True, cmap = 'Blues')
# Quality Class Values Count

target_count = rawdata['quality'].value_counts()

target_count.plot(kind='bar', title='Count (target)');
# Memvisualiasikan feature quality dan citric acid

plt.figure(figsize = (10, 6))

sns.set_context('talk')

sns.boxplot(rawdata['quality'], rawdata['citric acid'], data = rawdata)
outliers = []  # list data untuk menampung nilai ouliers



# metode yang digunakan untuk mendeteksi outliers yaitu interquartile range 

def detect_outliers(data): 

    quantile1, quantile3 = np.percentile(data, [25, 75])  # create two quantiles for 25% and 75%

    iqr_val = quantile3 - quantile1                       # interquantilerange value

    lower_bound_value = quantile1 - (1.5 * iqr_val)       # lower limit of the data, anything greater are not outliers

    upper_bound_value = quantile3 + (1.5 * iqr_val)       # upper limit of the data, anything less are not outliers

    

    for i in data:

        if lower_bound_value < i < upper_bound_value:     # if data[value] is greater than lbv and less than ubv than it is not considered as an outlier

            pass

        else:

            outliers.append(i)

            

    return lower_bound_value, upper_bound_value        # return lower bound and upper bound value for the data



feature_list = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

# #Delete outlier

# for i in feature_list:

#     mean = rawdata[i].mean()

#     bawah, atas = detect_outliers(rawdata[i])

#     rawdata[i] = rawdata[i].mask(rawdata[i] > atas, mean)
detect_outliers(rawdata['fixed acidity'])
rawdata.corr()
plt.figure(figsize = (10, 8))

sns.heatmap(rawdata.corr(), square = True, cmap = 'Blues')
from scipy import stats

k2, p = stats.normaltest(rawdata['fixed acidity'])

alpha = 1e-3

print("p = {:g}".format(p))

if p < alpha:  # null hypothesis: x comes from a normal distribution

    print("The null hypothesis can be rejected")

else:

    print("The null hypothesis cannot be rejected")
# Class count

cclass_6,cclass_5,cclass_7,cclass_8,cclass_4,cclass_3,cclass_9 = rawdata.quality.value_counts()



# Divide by class

df_class_6 = rawdata[rawdata['quality'] == 6]

df_class_5 = rawdata[rawdata['quality'] == 5]

df_class_7 = rawdata[rawdata['quality'] == 7]

df_class_8 = rawdata[rawdata['quality'] == 8]

df_class_4 = rawdata[rawdata['quality'] == 4]

df_class_3 = rawdata[rawdata['quality'] == 3]

df_class_9 = rawdata[rawdata['quality'] == 9]
df_class_5_over = df_class_5.sample(cclass_6, replace=True)

df_test_over = pd.concat([df_class_6, df_class_5_over], axis=0)



df_class_7_over = df_class_7.sample(cclass_6, replace=True)

df_test_over = pd.concat([df_class_6, df_class_7_over], axis=0)



df_class_8_over = df_class_8.sample(cclass_6, replace=True)

df_test_over = pd.concat([df_class_6, df_class_8_over], axis=0)



df_class_4_over = df_class_4.sample(cclass_6, replace=True)

df_test_over = pd.concat([df_class_6, df_class_4_over], axis=0)



df_class_3_over = df_class_3.sample(cclass_6, replace=True)

df_test_over = pd.concat([df_class_6, df_class_3_over], axis=0)



df_class_9_over = df_class_9.sample(cclass_6, replace=True)

df_test_over = pd.concat([df_class_6, df_class_9_over], axis=0)



print('Random over-sampling:')

print(df_test_over.quality.value_counts())



df_test_over.quality.value_counts().plot(kind='bar', title='Count (target)');
df_test_over.corr()
rawdata['quality'].replace([3, 4, 5], 'buruk' , inplace=True)

rawdata['quality'].replace([6], 'sedang' , inplace=True)

rawdata['quality'].replace([7,8, 9], 'baik' , inplace=True)
rawdata.info()
rawdata['quality'].loc[5.87790935]
# MISSING VALUE PROCESS

# 1. Check Missing Values field in each Column

#    Pengecekan ini untuk mengetahui kolom mana yang perlu dilakukan aksi

def missing_values_table(df):

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(

    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[

        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

    '% of Total Values', ascending=False).round(1)

    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

        "There are " + str(mis_val_table_ren_columns.shape[0]) +

          " columns that have missing values.")

    return mis_val_table_ren_columns



missing_values_table(rawdata)



# # Untuk mengatasi missing value pada proses ini menggunakan nilai yang sering muncul

# rawdata['total_bedrooms'].value_counts()

# # Didapatkan 280 merupakan total kamar yang sering disebut maka nilai kosong akan diisi oleh 280

# rawdata['total_bedrooms'].fillna(280,inplace=True)

# rawdata.describe()
def check_outliers(column):

    print('Outliers on column ', column)

    data_mean, data_std = rawdata[column].mean(), rawdata[column].std()

    print('mean : ',data_mean,', std : ',data_std)

    # identify outliers

    cut_off = data_std * 3

    lower, upper = data_mean - cut_off, data_mean + cut_off

    outliers = [x for x in rawdata[column] if x < lower or x > upper]

    print('Identified outliers: %d' % len(outliers))

    # remove outliers

    outliers_removed = [x for x in rawdata[column] if x >= lower and x <= upper]

    print('Non-outlier observations: %d' % len(outliers_removed))

    print('')

    return outliers_removed



outliers_data = pd.DataFrame()

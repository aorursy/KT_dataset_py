# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
# Read Data Files

cardio_dataset_raw = pd.read_csv("../input/cardio_train.csv",sep=";")



print("Dataset memiliki " + str(len(cardio_dataset_raw)) + " baris dan " + str(len(cardio_dataset_raw.columns)) + " kolom")

print()

print(cardio_dataset_raw.head())
# Cek semua deskripsi label pada dataset

cardio_dataset_raw.describe()
# Cek semua label, lihat apakah terdapat NULL

cardio_dataset_raw.isnull().sum()
# Cek outlier menggunakan boxplot

import seaborn as sns

import matplotlib.pyplot as plt

for col in cardio_dataset_raw:

    sns.boxplot(x=cardio_dataset_raw[col])

    plt.show()
# Cek data outlier pada setiap label menggunakan IQR (InterQuartile Range)



copy_of_cardio_dataset_raw = cardio_dataset_raw.copy()



# # Menentukan label apa saja yang ingin dicek outliernya. Yang dicek hanyalah label yang bersifat numerical (bukan categorical maupun identifier)

# check_outlier_column = ['age','height','weight','ap_hi','ap_lo']



# row_with_outlier = []



# # Cek IQR pada semua label

Q1 = copy_of_cardio_dataset_raw.quantile(0.25)

Q3 = copy_of_cardio_dataset_raw.quantile(0.75)



IQR = Q3 - Q1

rangeBawah = (Q1 - 1.5 * IQR) // 1

rangeAtas = (Q3 + 1.5 * IQR) // 1



# for i in range(len(check_outlier_column)):

#     countOutlier = 0

#     for j in range(len(copy_of_cardio_dataset_raw)):

        

#         #Cek apakah data lebih kecil dari Q1 - 1.5 * IQR

#         if(copy_of_cardio_dataset_raw.loc[j][check_outlier_column[i]] < rangeBawah[check_outlier_column[i]]): 

#             countOutlier = countOutlier + 1

#             row_with_outlier.append(j)

        

#         #Cek apakah data lebih besar dari Q3 + 1.5 * IQR

#         elif(copy_of_cardio_dataset_raw.loc[j][check_outlier_column[i]] > rangeAtas[check_outlier_column[i]]):

#             countOutlier = countOutlier + 1

#             row_with_outlier.append(j)

            

#     print("Label " + check_outlier_column[i] + " memiliki " + str(countOutlier) + " data outlier")



print("Dataset's shape before cleansing the outlier", copy_of_cardio_dataset_raw.shape)

copy_of_cardio_dataset_raw = copy_of_cardio_dataset_raw[~((copy_of_cardio_dataset_raw < rangeBawah) | copy_of_cardio_dataset_raw > rangeAtas).any(axis=1)]

# copy_df.drop(copy_df[(copy_df['height'] > copy_df['height'].quantile(0.75)) | copy_df['height'] < copy_df['height'].quantile(0.25)].index, inplace=True)

# copy_df.drop(copy_df[(copy_df['weight'] > copy_df['weight'].quantile(0.75)) | copy_df['weight'] < copy_df['weight'].quantile(0.25)].index, inplace=True)

# copy_df.drop(copy_df[(copy_df['ap_hi'] > copy_df['ap_hi'].quantile(0.75)) | copy_df['ap_hi'] < copy_df['ap_hi'].quantile(0.25)].index, inplace=True)

# copy_df.drop(copy_df[(copy_df['ap_lo'] > copy_df['ap_lo'].quantile(0.75)) | copy_df['ap_lo'] < copy_df['ap_lo'].quantile(0.25)].index, inplace=True)

print("Dataset's shape after cleansing the outlier", copy_of_cardio_dataset_raw.shape)
#Logically, ap_hi (sistole) harus lebih tinggi dari ap_lo (diastole)

row_with_error = []



#Cek apakah ada tekanan darah yang error

for i in range(len(copy_of_cardio_dataset_raw)):

    if(copy_of_cardio_dataset_raw.loc[i]['ap_hi'] < copy_of_cardio_dataset_raw.loc[i]['ap_lo']):

        row_with_error.append(i)

    

print("Dataset memiliki " + str(len(row_with_error)) + " yang memiliki ap_lo yang lebih tinggi dari ap_hi")
# Menggabungkan row_with_outlier dengan row_with_error

row_with_errors = []

row_with_errors = row_with_errors + row_with_outlier

row_with_errors = row_with_errors + row_with_error



# Cek row yang memiliki outlier & error, hapus row duplikasi, kemudian sort row dari terbesar ke terkecil

row_with_errors = list(dict.fromkeys(row_with_errors))



# Sort row dari terbesar ke terkecil

for i in range(len(row_with_errors)):

    swap = i + np.argmax(row_with_errors[i:])

    (row_with_errors[i], row_with_errors[swap]) = (row_with_errors[swap], row_with_errors[i])



#print(row_with_errors)

print("Secara total, terdapat " + str(len(row_with_errors)) + " baris yang harus dibersihkan")
cleaned_cardio_dataset = copy_of_cardio_dataset_raw.copy().drop(row_with_errors, axis=0)



#Reset row index

cleaned_cardio_dataset = cleaned_cardio_dataset.reset_index()



print("Dataset yang baru memiliki " + str(len(cleaned_cardio_dataset)) + " baris dan " + str(len(cleaned_cardio_dataset.columns)) + " kolom")

print()

print(cleaned_cardio_dataset.head(5))

#Label height dan weight dapat digabung, menghasilkan BMI (Body Mass Index)

bmi = []



# BMI List (for 18 years older, male and female):                      

# Underweight = <18.5            -> 0

# Normal      = 18.5 - 24.9      -> 1

# Overweight  = 25 - 29.9        -> 2

# Obesity     = >30              -> 3



def convert_bmi_to_list(list_bmi):

    for i in range(len(list_bmi)):

        if(list_bmi[i] < 18.5):

            list_bmi[i] = 0 #Underweight

        elif(list_bmi[i] >= 18.5 and list_bmi[i] < 25):

            list_bmi[i] = 1 #Normal

        elif(list_bmi[i] >= 25 and list_bmi[i] < 30):

            list_bmi[i] = 2 #Overweight

        elif(list_bmi[i] >= 30):

            list_bmi[i] = 3 #Obesity

    return list_bmi



for i in range(len(cleaned_cardio_dataset)):

    bmi.append(cleaned_cardio_dataset.loc[i]['weight'] / (cleaned_cardio_dataset.loc[i]['height'] / 100) ** 2)

            

convert_bmi_to_list(bmi)



#print(bmi)



#Tambah kolom baru, yaitu bmi yang merupakan gabungan dari kolom weight dan kolom height

cleaned_cardio_dataset['bmi'] = bmi



print(cleaned_cardio_dataset.head(5))
#Label ap_hi dan ap_lo dapat digabung, menghasilkan blood pressure

bp = []



# Blood Pressure List: (based on bloodpressureUK)

# ap_hi < 90 or ap_lo < 60 : Low Blood Pressure                       -> 0

# ap_hi >= 90 and ap_hi < 140 and ap_lo >= 60 and ap_lo < 90 : Normal -> 1

# ap_hi >= 140 or ap_lo >= 90 : High Blood Pressure                   -> 2



for i in range(len(cleaned_cardio_dataset)):

    if(cleaned_cardio_dataset.loc[i]['ap_hi'] < 90 or cleaned_cardio_dataset.loc[i]['ap_lo'] < 60):

        bp.append(0) #Low BP

    elif(cleaned_cardio_dataset.loc[i]['ap_hi'] >= 90 and cleaned_cardio_dataset.loc[i]['ap_hi'] < 140 and cleaned_cardio_dataset.loc[i]['ap_lo'] >= 60 and cleaned_cardio_dataset.loc[i]['ap_lo'] < 90):

        bp.append(1) #Normal BP

    elif(cleaned_cardio_dataset.loc[i]['ap_hi'] >= 140 and cleaned_cardio_dataset.loc[i]['ap_lo'] >= 90):

        bp.append(2) #High BP

    else:

        bp.append(-1) #Error

        

cleaned_cardio_dataset['bp'] = bp



print(cleaned_cardio_dataset.head(5))
#Cek apakah terdapat bp yang error

row_with_bp_error = []



for i in range(len(cleaned_cardio_dataset)):

    if(cleaned_cardio_dataset.loc[i]['bp'] == -1):

        row_with_bp_error.append(i)

        

print("Terdapat " + str(len(row_with_bp_error)) + " baris yang memiliki bp error")
# Cek row yang memiliki bp error, hapus row duplikasi, kemudian sort row dari terbesar ke terkecil

row_with_bp_error = list(dict.fromkeys(row_with_bp_error))



# Sort row dari terbesar ke terkecil

for i in range(len(row_with_bp_error)):

    swap = i + np.argmax(row_with_bp_error[i:])

    (row_with_bp_error[i], row_with_bp_error[swap]) = (row_with_bp_error[swap], row_with_bp_error[i])



#print(row_with_bp_error)
cleaned_cardio_dataset = cleaned_cardio_dataset.drop(row_with_bp_error, axis=0)



#Reset row index

cleaned_cardio_dataset = cleaned_cardio_dataset.reset_index()



print("Dataset yang baru memiliki " + str(len(cleaned_cardio_dataset)) + " baris dan " + str(len(cleaned_cardio_dataset.columns)) + " kolom")

print()

print(cleaned_cardio_dataset.head(5))
sns.heatmap(cleaned_cardio_dataset.corr(),annot=True,cmap='YlGnBu')

fig=plt.gcf()

fig.set_size_inches(15,10)

plt.show
final_cardio = cleaned_cardio_dataset.copy().drop(columns=['level_0','index','id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo'])

final_cardio.head(5)
final_cardio.describe()
y_cardio = final_cardio['cardio']

x_cardio = final_cardio.drop('cardio', axis=1)
y_cardio.head(5)
x_cardio.head(5)
x_train, x_test, y_train, y_test = train_test_split(x_cardio, y_cardio, test_size=0.3)
print("Data training {}, data testing {}".format(x_train.shape, x_test.shape))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
for col in x_train.columns:

    x_train[col] = scaler.fit_transform(np.array(x_train[col]).reshape(-1,1))

x_train.head()
for col in x_train.columns:

    x_test[col] = scaler.fit_transform(np.array(x_test[col]).reshape(-1,1))

x_test.head()
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200, criterion="entropy", random_state=2, warm_start=True)

rfc.fit(x_train, y_train)



from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()

abc.fit(x_train, y_train)



from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(tol=1e-16)

gbc.fit(x_train, y_train)
print("Random Forest Classifier - Accuracy: ", rfc.score(x_test, y_test))

print("Ada Boost Classifier - Accuracy: ", abc.score(x_test, y_test))

print("Gradient Boosting Classifier - Accuracy: ", gbc.score(x_test, y_test))
np.unique(y_test)
((y_train == 0).sum() + (y_test == 0).sum()) / final_cardio.shape[0]
((y_train == 1).sum() + (y_test == 1).sum()) / final_cardio.shape[0]
cleaned_cardio_df = cleaned_cardio_dataset.copy().drop(columns=["level_0", "index", "id", "ap_hi", "ap_lo", "weight", "height"])

cleaned_cardio_df.head()
cleaned_cardio_df.describe()
sns.heatmap(cleaned_cardio_df.corr(),annot=True,cmap='YlGnBu',square=True)

fig=plt.gcf()

fig.set_size_inches(15,10)

plt.show
y_coba = cleaned_cardio_df["cardio"]

x_coba = cleaned_cardio_df.drop(columns=["cardio"])
x_coba.head()
y_coba.head()
x_train_coba, x_test_coba, y_train_coba, y_test_coba = train_test_split(x_coba, y_coba, test_size=0.3, random_state=1)
print(x_train_coba.shape)

print(x_test_coba.shape)
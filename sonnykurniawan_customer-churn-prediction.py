#Import library yang dibutuhkan 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix, classification_report

import pickle

pd.options.display.max_columns = 50





#import dataset

df_load = pd.read_csv('../input/telco_company.csv')



#Tampilkan jumlah baris dan kolom

print(df_load.shape)



#Tampilkan 5 data teratas

print(df_load.head(5))



#Jumlah ID yang unik

print(df_load.customerID.nunique())
#Membuat kolom bantuan 'valid_id',mencari customerID diawali dengan angka 45 2 digit pertama dengan str.match

df_load['valid_id'] = df_load['customerID'].astype(str).str.match(r'(45\d{9,10})')

#Mengambil valid_id bernilai True kemudian drop kolom valid_id

df_load = (df_load[df_load['valid_id'] == True]).drop('valid_id', axis = 1)

#Menghitung jumlah baris 'customerID' setelah difilter

print('Hasil jumlah ID Customer yang terfilter adalah',df_load['customerID'].count())

# Drop Duplicate Rows

df_load.drop_duplicates()

# Drop duplicate ID sorted by Periode

df_load = df_load.sort_values('UpdatedAt', ascending=False).drop_duplicates(['customerID'])

print('Hasil jumlah ID Customer yang sudah dihilangkan duplikasinya (distinct) adalah',df_load['customerID'].count())
print('Total missing values data dari kolom Churn',df_load['Churn'].isnull().sum())

# Dropping all Rows with spesific column (churn)

df_load.dropna(subset=['Churn'],inplace=True)

print('Total Rows dan kolom Data setelah dihapus data Missing Values adalah',df_load.shape)
print('Status Missing Values :',df_load.isnull().values.any())

print('\nJumlah Missing Values masing-masing kolom, adalah:')

print(df_load.isnull().sum().sort_values(ascending=False))



# handling missing values Tenure fill with 11

df_load['tenure'].fillna(11, inplace=True)



# Handling missing values num vars (except Tenure)

for col_name in list(['MonthlyCharges','TotalCharges']):

  median = df_load[col_name].median()

  df_load[col_name].fillna(median, inplace=True)



print('\nJumlah Missing Values setelah di imputer datanya, adalah:')

print(df_load.isnull().sum().sort_values(ascending=False))
print('\nPersebaran data sebelum ditangani Outlier: ')

print(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())



# Creating Box Plot

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure() # untuk membuat figure baru

sns.boxplot(x=df_load['tenure'])

plt.show()

plt.figure() # untuk membuat figure baru

sns.boxplot(x=df_load['MonthlyCharges'])

plt.show()

plt.figure() # untuk membuat figure baru

sns.boxplot(x=df_load['TotalCharges'])

plt.show()
# Handling with IQR

Q1 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.25)

Q3 = (df_load[['tenure','MonthlyCharges','TotalCharges']]).quantile(0.75)



IQR = Q3 - Q1

maximum = Q3 + (1.5*IQR)

print('Nilai Maximum dari masing-masing Variable adalah: ')

print(maximum)

minimum = Q1 - (1.5*IQR)

print('\nNilai Minimum dari masing-masing Variable adalah: ')

print(minimum)



more_than = (df_load > maximum)

lower_than = (df_load < minimum)

df_load = df_load.mask(more_than, maximum, axis=1)

df_load = df_load.mask(lower_than, minimum, axis=1)



print('\nPersebaran data setelah ditangani Outlier: ')

print(df_load[['tenure','MonthlyCharges','TotalCharges']].describe())
#Loop

for col_name in list(['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','Churn']):

  print('\nUnique Values Count \033[1m' + 'Before Standardized \033[0m Variable',col_name)



  print(df_load[col_name].value_counts())
#Replace unique values dari masing-masing variable

df_load = df_load.replace(['Wanita','Laki-Laki','Churn','Iya'],['Female','Male','Yes','Yes'])



#Memihat unique values setelah dilakukan standarisasi

for col_name in list(['gender','Dependents','Churn']):

  print('\nUnique Values Count \033[1m' + 'After Standardized \033[0mVariable',col_name)

  print(df_load[col_name].value_counts())
#Mengambil kolom yang diperlukan yang disimpan pada variabel df_telco

df_telco = df_load[['UpdatedAt','customerID', 'gender','SeniorCitizen','Partner','tenure','PhoneService','StreamingTV','InternetService','PaperlessBilling','MonthlyCharges','TotalCharges','Churn']]



#Tampilkan bentuk dari dataset

print(df_telco.shape)



#Tampilkan 5 data teratas

print(df_telco.head())



#Tampilkan jumlah ID yang unik

print(df_telco.customerID.nunique())
#Membuat figur plot

fig = plt.figure()

#Membuat sumbu plot

ax = fig.add_axes([0,0,1,1])

#Mengubah batas sumbu x atau y sehingga memiliki panjang yang sama

ax.axis('equal')

#Membuat Label pie chart

labels = ['Yes','No']

#Menghitung banyaknya unik dari kolom Churn dengan value_counts()

churn = df_telco.Churn.value_counts()

#Membuat bentuk pie chart

ax.pie(churn, labels=labels, autopct='%.0f%%')

plt.show()
#Membuat bin dalam chart

numerical_features =  ['MonthlyCharges','TotalCharges','tenure']

#Membuat subplot

fig, ax = plt.subplots(1, 3, figsize=(15, 6))

#Untuk memplot dua overlay histogram per numerical_features masing-masing

df_load[df_telco.Churn == 'No'][numerical_features].hist(bins=20, color='blue', alpha=0.5, ax=ax)

df_load[df_telco.Churn == 'Yes'][numerical_features].hist(bins=20, color='orange', alpha=0.5, ax=ax)

plt.show()
#Membuat subplot

fig, ax = plt.subplots(3, 3, figsize=(14, 12))

#membuat plot dengan jumlah pengamatan di setiap bin kategorik variable dengan countplot()

sns.countplot(data=df_telco, x='gender', hue='Churn', ax=ax[0][0])

sns.countplot(data=df_telco, x='Partner', hue='Churn', ax=ax[0][1])

sns.countplot(data=df_telco, x='SeniorCitizen', hue='Churn', ax=ax[0][2])

sns.countplot(data=df_telco, x='PhoneService', hue='Churn', ax=ax[1][0])

sns.countplot(data=df_telco, x='StreamingTV', hue='Churn', ax=ax[1][1])

sns.countplot(data=df_telco, x='InternetService', hue='Churn', ax=ax[1][2])

sns.countplot(data=df_telco, x='PaperlessBilling', hue='Churn', ax=ax[2][1])

plt.tight_layout()

plt.show()

#Menghapus kolom customerID & UpdatedAt

cleaned_df = df_telco.drop(['customerID','UpdatedAt'], axis=1)

print(cleaned_df.head())
#Mengubah semua tipe data non-numeric columns ke numerical

for column in cleaned_df.columns:

    if cleaned_df[column].dtype == np.number: continue

    #Melakukan encoding  untuk setiap kolom non-numeric dengan LabelEncoder()

    cleaned_df[column] = LabelEncoder().fit_transform(cleaned_df[column])

print(cleaned_df.describe())
# Predictor dan target

X = cleaned_df.drop('Churn', axis = 1)

y = cleaned_df['Churn']

# Splitting train and test

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print('Jumlah baris dan kolom dari x_train adalah:', x_train.shape,', sedangkan Jumlah baris dan kolom dari y_train adalah:', y_train.shape)

print('Prosentase Churn di data Training adalah:')

#Mengecek apakah pembagian sudah sama proporsinya

print(y_train.value_counts(normalize=True))
#Membuat model dengan menggunakan Algoritma Logistic Regression

log_model = LogisticRegression().fit(x_train, y_train)

print('Model Logistic Regression yang terbentuk adalah: \n',log_model)
# Predict

y_train_pred =  log_model.predict(x_train)

# Print classification report 

print('Classification Report Training Model (Logistic Regression) :')

print(classification_report(y_train, y_train_pred))
# Form confusion matrix as a DataFrame

confusion_matrix_df = pd.DataFrame((confusion_matrix(y_train, y_train_pred)), ('No churn', 'Churn'), ('No churn', 'Churn'))



# Plot confusion matrix

plt.figure()

heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)



plt.title('Confusion Matrix for Training Model\n(Logistic Regression)', fontsize=18, color='darkblue')

plt.ylabel('True label', fontsize=14)

plt.xlabel('Predicted label', fontsize=14)

plt.show()
# Predict

y_test_pred = log_model.predict(x_test)

# Print classification report

print('Classification Report Testing Model (Logistic Regression):')

print(classification_report(y_test, y_test_pred))
# Form confusion matrix as a DataFrame

confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, y_test_pred)), ('No churn', 'Churn'), ('No churn', 'Churn'))



# Plot confusion matrix

plt.figure()

heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)



plt.title('Confusion Matrix for Testing Model\n(Logistic Regression)\n', fontsize=18, color='darkblue')

plt.ylabel('True label', fontsize=14)

plt.xlabel('Predicted label', fontsize=14)

plt.show()
#Save Model

pickle.dump(log_model, open('best_model_churn.pkl', 'wb'))
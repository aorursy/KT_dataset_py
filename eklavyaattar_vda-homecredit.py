# import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#read data file
df_installments_payments = pd.read_csv("../input/installments_payments.csv")
#Viewing the records
print(df_installments_payments.head())
# Slicing the data such that working with 5 lac records and 6 columns by removing the first 2 ids, only for the VDA purpose
df_installments_payments_mod = df_installments_payments.iloc[0:500000,2:8]
df_installments_payments_mod.head(10)
# Check Column Names
print(df_installments_payments_mod.columns)
# Check datatypes of the columns
df_installments_payments_mod.dtypes
# Check summary
df_installments_payments_mod.describe()
#1. Checking fot NULLS
df_installments_payments_mod.isnull().sum()
#2. Checking fot Whitespaces
np.where(df_installments_payments_mod.applymap(lambda x: x == ' '))
# Detect outlier fucntion
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
#1.Outliers for NUM_INSTALMENT_VERSION
# Call outlier function and Print outlier' s indexed values
print(outliers_iqr(ys=df_installments_payments_mod['NUM_INSTALMENT_VERSION']))
a = np.array([outliers_iqr(ys=df_installments_payments_mod['NUM_INSTALMENT_VERSION'])])
print(a)
a.size
#2.Outliers for NUM_INSTALMENT_NUMBER
# Call outlier function and Print outlier' s indexed values
print(outliers_iqr(ys=df_installments_payments_mod['NUM_INSTALMENT_NUMBER']))
a = np.array([outliers_iqr(ys=df_installments_payments_mod['NUM_INSTALMENT_NUMBER'])])
print(a)
a.size
#3.Outliers for DAYS_INSTALMENT
# Call outlier function and Print outlier' s indexed values
print(outliers_iqr(ys=df_installments_payments_mod['DAYS_INSTALMENT']))
a = np.array([outliers_iqr(ys=df_installments_payments_mod['DAYS_INSTALMENT'])])
print(a)
a.size
#4.Outliers for DAYS_ENTRY_PAYMENT
# Call outlier function and Print outlier' s indexed values
print(outliers_iqr(ys=df_installments_payments_mod['DAYS_ENTRY_PAYMENT']))
a = np.array([outliers_iqr(ys=df_installments_payments_mod['DAYS_ENTRY_PAYMENT'])])
print(a)
a.size
#5.Outliers for AMT_INSTALMENT
# Call outlier function and Print outlier' s indexed values
print(outliers_iqr(ys=df_installments_payments_mod['AMT_INSTALMENT']))
a = np.array([outliers_iqr(ys=df_installments_payments_mod['AMT_INSTALMENT'])])
print(a)
a.size
#6.Outliers for AMT_PAYMENT
# Call outlier function and Print outlier' s indexed values
print(outliers_iqr(ys=df_installments_payments_mod['AMT_PAYMENT']))
a = np.array([outliers_iqr(ys=df_installments_payments_mod['AMT_PAYMENT'])])
print(a)
a.size
#1. To check realtionshiop between DAYS_INSTALMENT and DAYS_ENTRY_PAYMENT using Joint plot
sns.jointplot(x='DAYS_INSTALMENT', y='DAYS_ENTRY_PAYMENT', data=df_installments_payments_mod)
plt.show()
#1. To check realtionshiop between AMT_INSTALMENT and AMT_PAYMENT using Joint plot
sns.jointplot(x='AMT_INSTALMENT', y='AMT_PAYMENT', data=df_installments_payments_mod)
plt.show()
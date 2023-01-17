import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualization of the data
import seaborn as sns # 
import re
df_kiva = pd.read_csv("../input/kiva_loans.csv")
print(df_kiva.shape)# rows and columns



df_kiva.head(10)

print(type(df_kiva))# type of the data
total_missing=df_kiva.isnull().sum().sort_values(ascending=False)

print(total_missing)
# due to high frequency of missing data in tags- 171416 ,region-56800,funded_time-48331,partner_id-13507
#use-4228,borrower_genders -4221,disbursed_time-2396,country_code -8
# we can't anlayse these features 
# let's analyse remaining features


          
           
df_kiva.nunique()# unique data
plt.figure(figsize=(12,6))
df_kiva['sector'].value_counts().head(10).plot.bar()
plt.figure(figsize=(12,6))
df_kiva['country'].value_counts().head(10).sort_values(ascending=False).plot.bar()
df_kiva['repayment_interval'].value_counts().unique()
df_kiva['repayment_interval'].value_counts().head(10).plot.barh()
plt.title("Types of repayment intervals", fontsize=16)


df_kiva['activity'].sort_values().unique()

plt.figure(figsize=(12,6))
df_kiva['activity'].value_counts().head(10).plot.barh()

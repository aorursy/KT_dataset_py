#import libraries



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')



%matplotlib inline
df = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv",encoding = "ISO-8859-1")
df.head()
#check missing value

def see_lack(i):

    print("◆types")

    print(i.dtypes)

    print("")

    print("◆check columns which has missing values")

    print(i.isnull().any())

    print("")

    print("◆count missing values in each columns")

    print(i.isnull().sum())

    print("")

    print("◆data shape")

    print(i.shape)
see_lack(df)
df.describe()
#convert Portuguese to English 

month_convert={'Janeiro' : 'January',

                            "Fevereiro":"Feburary",

                            "Março":"March",

                            "Abril":"April",

                            "Maio" : "May",

                            "Junho" : "June",

                            "Julho":"July",

                            "Agosto":"August",

                            "Setembro":"September",

                            "Outubro":"October",

                            "Novembro" : "November",

                            "Dezembro":"December"}



df["month_en"] = df.month.replace(month_convert)
#change type of column"year" to object



df.year = df.year.astype("O")
#make pivot table

pivot_sum = df.pivot_table(values="number",index="year",columns="month_en", aggfunc=np.sum)



#arrange the order of month

pivot_sum = pivot_sum.loc[:,['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December']]



pivot_sum
plt.figure(figsize=(15, 8))

sns.heatmap(pivot_sum,annot=True,fmt="1.2f",cmap='Reds')



plt.title("Heatmap of forest fires by year and month", fontsize =10)
plt.figure(figsize=(15, 8))

sns.heatmap(pivot_sum,annot=True,fmt="1.2f",cmap='hot_r')



plt.title("Heatmap of forest fires by year and month", fontsize =10)
#pick up data of it occurs in 2017

df_2017 = df[df.year == 2017]



df_2017.head()
#make pivot table

pivot_sum_2017 = df_2017.pivot_table(values="number",index="state",columns="month_en", aggfunc=np.sum)



#arrange the order of month

pivot_sum_2017 = pivot_sum_2017.loc[:,['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December']]

plt.figure(figsize=(15, 8))

sns.heatmap(pivot_sum_2017,annot=True,fmt="1.2f",cmap='hot_r')



plt.title("Heatmap of forest fires by month and state in 2017", fontsize =10)
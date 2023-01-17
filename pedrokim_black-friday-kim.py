import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

import plotly.plotly as py

import plotly.graph_objs as go

from collections import Counter

from math import log
data_polluted = pd.read_csv('../input/blackfriday/BlackFriday.csv')



data_raw = data_polluted.dropna(subset=['Purchase'])


##############################################################

### Aplica a regra de Sturges para definição de bins

def bins_sturges(df):

    n = df.size

    k = round(1+ log(n)/log(2))

    return k



##############################################################



def pizza(labels, sizes):

    fig1, ax1 = plt.subplots()

    ax1.pie(sizes, labels=labels,shadow=True, autopct='%1.1f%%',startangle=90)

    plt.show()

    plt.clf()

    plt.cla()

    plt.close()

    

##############################################################

## Agrupar e contar todos os grupos de um data frame(df) de uma coluna(column) 

def df_group(df, column):

    name_list = []

    count_list = []

    ele_col_list = pd.unique(data_raw[column])

    ele_col_list.sort()

    for i in ele_col_list:

        discriminated = df.loc[data_raw[column] == i]

        count = discriminated.size

        name_list.append(i)

        count_list.append(count)

        

    return name_list, count_list

 

##############################################################

##Histograma de uma coluna de dataset

def df_hist(df, column):

    listed = pd.DataFrame.from_dict(Counter(sorted(df[column])), orient='index')

    listed.plot(kind='bar')

    

    

##############################################################

##Histograma baseado em valores



def df_hist_sturges_purchase(df, common_group, raw_column, product_id):

    x = df.loc[df[raw_column] == common_group]

    k = bins_sturges(x)

    x.loc[x['Product_ID'] == product_id,'Purchase'].hist(bins = k)

    
data_gender_M = data_raw.loc[data_raw['Gender'] == 'M']

data_gender_F = data_raw.loc[data_raw['Gender'] == 'F']

labels, counts = df_group(data_raw, 'Gender')

male_list = pd.unique(data_gender_M['User_ID'])

fem_list = pd.unique(data_gender_F['User_ID'])

pizza(labels, [fem_list.size, male_list.size])
plt.boxplot ([data_gender_F['Purchase'].values, data_gender_M['Purchase'].values], labels = labels)
labels, counts = df_group(data_raw[['User_ID','Marital_Status']].drop_duplicates(), 'Marital_Status')

pizza(labels, counts)
labels, counts = df_group(data_gender_M[['User_ID','Marital_Status']].drop_duplicates(), 'Marital_Status')

labels2, counts2 = df_group(data_gender_F[['User_ID','Marital_Status']].drop_duplicates(), 'Marital_Status')



fig, axs = plt.subplots(1,2)

fig.tight_layout(pad=5.0)

axs[0].pie(counts, labels=labels,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)



axs[1].pie(counts2, labels=labels2,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)



axs[0].set_title(label = 'Homens')

axs[1].set_title(label = 'Mulheres')

plt.show()

plt.clf()

plt.cla()

plt.close()
data_age_0_17 = data_raw.loc[data_raw['Age'] == '0-17']

data_age_18_25 = data_raw.loc[data_raw['Age'] == '18-25']

data_age_26_35 = data_raw.loc[data_raw['Age'] == '26-35']

data_age_36_45 = data_raw.loc[data_raw['Age'] == '36-45']

data_age_46_50 = data_raw.loc[data_raw['Age'] == '46-50']

data_age_51_55 = data_raw.loc[data_raw['Age'] == '51-55']

data_age_55 = data_raw.loc[data_raw['Age'] == '55+']
labels, counts = df_group(data_raw[['User_ID','Age']].drop_duplicates(), 'Age')

pizza(labels, counts)
labels, counts = df_group(data_gender_M[['User_ID','Age']].drop_duplicates(), 'Age')

labels2, counts2 = df_group(data_gender_F[['User_ID','Age']].drop_duplicates(), 'Age')



fig, axs = plt.subplots(1,2)

fig.tight_layout(pad=5.0)

axs[0].pie(counts, labels=labels,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)



axs[1].pie(counts2, labels=labels2,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)



axs[0].set_title(label = 'Homens')

axs[1].set_title(label = 'Mulheres')

plt.show()

plt.clf()

plt.cla()

plt.close()
fig, axs = plt.subplots(1,2)

fig.tight_layout(pad=5.0)

axs[0].pie(counts, labels=labels,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)



axs[1].pie(counts2, labels=labels2,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)



axs[0].set_title(label = 'Homens')

axs[1].set_title(label = 'Mulheres')

plt.show()

plt.clf()

plt.cla()

plt.close()
labels, counts = df_group(data_gender_M[['User_ID','Age']].drop_duplicates(), 'Age')

labels2, counts2 = df_group(data_gender_F[['User_ID','Age']].drop_duplicates(), 'Age')



fig, axs = plt.subplots(1,2)

fig.tight_layout(pad=3.0)

fig.set_size_inches(15, 15, forward=True)

#axs[0].pie(counts, labels=labels,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)

#axs[1].pie(counts2, labels=labels2,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)

axs[0].bar(x = labels, height = counts)

axs[1].bar(x = labels2, height = counts2)



axs[0].set_title(label = 'Homens')

axs[1].set_title(label = 'Mulheres')

plt.show()

plt.clf()

plt.cla()

plt.close()
labels, counts = df_group(data_raw, 'Age')

plt.boxplot ([data_age_0_17['Purchase'].values, data_age_18_25['Purchase'].values, data_age_26_35['Purchase'].values, data_age_36_45['Purchase'].values, data_age_46_50['Purchase'].values,  data_age_51_55['Purchase'].values, data_age_55['Purchase'].values], labels= labels)

data_geo_A = data_raw.loc[data_raw['City_Category'] == 'A']

data_geo_B = data_raw.loc[data_raw['City_Category'] == 'B']

data_geo_C = data_raw.loc[data_raw['City_Category'] == 'C']
labels, counts = df_group(data_raw[['User_ID','City_Category']].drop_duplicates(), 'City_Category')

pizza(labels, counts)
labels, counts = df_group(data_raw[['City_Category']], 'City_Category')

pizza(labels, counts)
labels, counts = df_group(data_raw, 'City_Category')

plt.boxplot([data_geo_A['Purchase'].values, data_geo_B['Purchase'].values, data_geo_C['Purchase'].values], labels=labels)

labels, counts = df_group(data_raw, 'Stay_In_Current_City_Years')



pizza(labels, counts)
plt.boxplot([data_raw.loc[data_raw['Stay_In_Current_City_Years'] == '0', 'Purchase'].values, data_raw.loc[data_raw['Stay_In_Current_City_Years'] == '1', 'Purchase'].values, data_raw.loc[data_raw['Stay_In_Current_City_Years'] == '2', 'Purchase'].values, data_raw.loc[data_raw['Stay_In_Current_City_Years'] == '3', 'Purchase'].values, data_raw.loc[data_raw['Stay_In_Current_City_Years'] == '4+', 'Purchase'].values], labels=labels, showcaps = True)
labels, counts = df_group(data_raw[['User_ID','Marital_Status']].drop_duplicates(), 'Marital_Status')

pizza(labels, counts)
labels, counts = df_group(data_gender_M[['User_ID','Marital_Status']].drop_duplicates(), 'Marital_Status')

labels2, counts2 = df_group(data_gender_F[['User_ID','Marital_Status']].drop_duplicates(), 'Marital_Status')



fig, axs = plt.subplots(1,2)

fig.tight_layout(pad=5.0)

axs[0].pie(counts, labels=labels,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)



axs[1].pie(counts2, labels=labels2,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)



axs[0].set_title(label = 'Homens')

axs[1].set_title(label = 'Mulheres')

plt.show()

plt.clf()

plt.cla()

plt.close()


fig, axs = plt.subplots(1,2)

fig.tight_layout(pad=3.0)

fig.set_size_inches(15, 15, forward=True)

#axs[0].pie(counts, labels=labels,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)

#axs[1].pie(counts2, labels=labels2,shadow=True, radius=2, autopct='%1.1f%%',startangle=90)

axs[0].bar(x = labels, height = counts)

axs[1].bar(x = labels2, height = counts2)



axs[0].set_title(label = 'Homens')

axs[1].set_title(label = 'Mulheres')

plt.show()

plt.clf()

plt.cla()

plt.close()
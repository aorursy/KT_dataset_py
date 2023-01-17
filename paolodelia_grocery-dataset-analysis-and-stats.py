import numpy as np 

import pandas as pd 

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
filepath = '../input/grocery-exprenses/spesa_dataset.csv'

df = pd.read_csv(filepath, delimiter=";", encoding = "cp1252")

df.head(-10)
df.dtypes
#fixme:to remove, it's just a temporary error

df.loc[df['giorno'] == 'Lidl24/08/2020']

df.loc[907, 'giorno'] = '24/08/2020'



df.loc[906, 'supermercato'] = 'Lidl'
df['giorno'] = pd.to_datetime(df['giorno'], infer_datetime_format=True)



print(df.giorno.dtypes)
supermarkets = df['supermercato'].unique()



supermarkets.sort()

supermarkets
df['supermercato'] = df['supermercato'].str.strip()
missing_types = df['tipo'].isnull().sum()



missing_types
df['tipo'] = df['tipo'].fillna('none')
types = df['tipo'].unique()



types.sort()

types
import fuzzywuzzy

from fuzzywuzzy import process

import chardet



incostencies = ["frutta secca", "passata pomodoro", "bevande","dolce","integratore","briosche","aceto","borsa spesa","gnocchi","crackers"]



matches_list = []



for el in incostencies:

    matches = fuzzywuzzy.process.extract(el, types, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    print(matches, end='\n\n')

    matches_list.append(matches)
def replace_second_match(df:pd.DataFrame, column:str, matches:list):

    close_matches = [matches[1][0]]

    row_with_matches = df[column].isin(close_matches)

    

    df.loc[row_with_matches, column] = matches[0][0] 

    

for el in matches_list:

    replace_second_match(df, 'tipo', el)
types = df['tipo'].unique()



types.sort()

types
row_with_matches = df['tipo'].isin(['arachidi'])

    

df.loc[row_with_matches, 'tipo'] = 'frutta secca'



row_with_matches = df['tipo'].isin(['bibite'])

    

df.loc[row_with_matches, 'tipo'] = 'bevande'



row_with_matches = df['tipo'].isin(['gnochetti'])

    

df.loc[row_with_matches, 'tipo'] = 'gnocchi'
# just a simple description of the dataset

df.describe(include=np.number)
# most frequent name in the grocery dataset

df['nome'].mode()
# most frequent type of grocery item in the dataset

df['tipo'].mode()
# most frequent supermarket on the dataset

df['supermercato'].mode()
def fromSeriesToLists(pd_serie, threshold=0):

    keys = []

    values = []

    

    for key, value in pd_serie.items():

        if value > threshold:

            keys.append(key)

            values.append(value)

            

    return keys, values
most_freq_items = df.nome.value_counts()



names, values = fromSeriesToLists(most_freq_items, 4)



freq_items_df = pd.DataFrame(

    data = {

        'Names': names, 

        'Values':values

    }

)



sns.barplot(x=freq_items_df['Names'], y=freq_items_df['Values'])

plt.xticks(rotation=70)
most_freq_types = df.tipo.value_counts()



names, values = fromSeriesToLists(most_freq_types, 15)





freq_types_df = pd.DataFrame(

    data = {

        'Names': names, 

        'Values':values

    }

)



sns.barplot(x=freq_types_df['Names'], y=freq_types_df['Values'])

plt.xticks(rotation=70)
most_freq_super = df.supermercato.value_counts()



names, values = fromSeriesToLists(most_freq_super)





freq_super_df = pd.DataFrame(

    data = {

        'Names': names, 

        'Values':values

    }

)



sns.barplot(x=freq_super_df['Names'], y=freq_super_df['Values'])

plt.xticks(rotation=70)
sns.distplot(a=df['prezzo'], kde=False)
# tot = df.groupby(['tipo']).sum()

# freq_type_name = freq_types_df.Names



# tot.reset_index(drop=True, inplace=True)



# # tot = tot[tot['tipo'].isin(freq_type_name)]



# tot

# # sns.barplot(x=tot[tot.isin(freq_type_name)], y=tot[tot['prezzo'].isin(freq_type_name)])

# # plt.xticks(rotation=70)
# total for now

df['prezzo'].sum()
df['giorno'] = pd.to_datetime(df['giorno'], infer_datetime_format=True)

weekly_gr = df.groupby(df.giorno.dt.strftime('%W'))



weekly = {

    'week':[],

    'weekly_shopping': [], 

    'amount_per_day': [], 

    'price_mean': [],

    'price_std': [],

    'most_freq_item': [],

    'most_freq_type':[]

}



for name, group in weekly_gr:

    if len(group) > 0:

        tot = group.prezzo.sum()

        tot_per_day = tot / 7

        mean = group.prezzo.mean()

        std = group.prezzo.std()

        weekly['week'].append(name)

        weekly['weekly_shopping'].append(tot)

        weekly['amount_per_day'].append(tot_per_day)

        weekly['price_mean'].append(mean)

        weekly['price_std'].append(std)

        weekly['most_freq_item'].append(group.nome.value_counts().idxmax())

        weekly['most_freq_type'].append(group.tipo.value_counts().idxmax())



weekly_df = pd.DataFrame(weekly)

weekly_df
weekly_df.describe(include=np.number)
fig, axs = plt.subplots(ncols=2)

sns.distplot(a=weekly_df['weekly_shopping'], kde=False, ax=axs[0])

sns.distplot(a=weekly_df['amount_per_day'], kde=False,bins=5, ax=axs[1])
monthly_gr = df.groupby(df.giorno.dt.strftime('%m'))



monthly = {

    'month':[],

    'monthly_shopping': [], 

    'amount_per_week': [], 

    'price_mean': [],

    'price_std': [],

    'most_freq_item': [],

    'most_freq_type':[]

}



for name, group in monthly_gr:

    if len(group) > 0:

        tot = group.prezzo.sum()

        tot_per_day = tot / 4

        mean = group.prezzo.mean()

        std = group.prezzo.std()

        monthly['month'].append(name)

        monthly['monthly_shopping'].append(tot)

        monthly['amount_per_week'].append(tot_per_day)

        monthly['price_mean'].append(mean)

        monthly['price_std'].append(std)

        monthly['most_freq_item'].append(group.nome.mode()[0])

        monthly['most_freq_type'].append(group.tipo.mode()[0])



monthly_df = pd.DataFrame(monthly)

monthly_df
monthly_df.describe(include=np.number)
sns.distplot(a=monthly_df['monthly_shopping'], kde=False)
yearly_gr = df.groupby(df.giorno.dt.strftime('%Y'))



yearly = {

    'year':[],

    'yearly_shopping': [], 

    'amount_per_month': [], 

    'price_mean': [],

    'price_std': [],

    'most_freq_item': [],

    'most_freq_type':[]

}



for name, group in yearly_gr:

    if len(group) > 0:

        tot = group.prezzo.sum()

        tot_per_day = tot / 4

        mean = group.prezzo.mean()

        std = group.prezzo.std()

        yearly['year'].append(name)

        yearly['yearly_shopping'].append(tot)

        yearly['amount_per_month'].append(tot_per_day)

        yearly['price_mean'].append(mean)

        yearly['price_std'].append(std)

        yearly['most_freq_item'].append(group.nome.mode()[0])

        yearly['most_freq_type'].append(group.tipo.mode()[0])



yearly_df = pd.DataFrame(yearly)

yearly_df
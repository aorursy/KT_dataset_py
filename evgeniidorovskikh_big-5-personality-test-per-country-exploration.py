import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sb

import pycountry
path = r'../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv'

df_full = pd.read_csv(path, sep='\t')

df_full.head()


df = df_full.loc[~(df_full[df_full.columns[:-1]] == 0).any(axis=1)] #removing nulls

df1 = df[(df['IPC'] == 1)] #cleaning repeaters (same IP)

df1 = df1[((df1['testelapse'] >= 150) & (df1['testelapse'] <= 2000))] # time filter

df1 = df1.drop(columns = ['screenw','screenh','lat_appx_lots_of_err','long_appx_lots_of_err']) # I'm not using that data

#Petar Luketina code

#Changing country names for more readability

country_dict = {i.alpha_2: i.alpha_3 for i in pycountry.countries}

df1['country'] = df1.country.map(country_dict)



# Dividing questions into positive and negative and then replace answers accordingly

pos_questions = [ # positive questions adding to the trait.

    'EXT1','EXT3','EXT5','EXT7','EXT9',                       # 5 Extroversion

    'EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10', # 8 Neuroticism

    'AGR2','AGR4','AGR6','AGR8','AGR9','AGR10',               # 6 Agreeableness

    'CSN1','CSN3','CSN5','CSN7','CSN9','CSN10',               # 6 Conscientiousness

    'OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10',        # 7 Openness

]

neg_questions = [ # negative (negating) questions subtracting from the trait.

    'EXT2','EXT4','EXT6','EXT8','EXT10', # 5 Extroversion

    'EST2','EST4',                       # 2 Neuroticism

    'AGR1','AGR3','AGR5','AGR7',         # 4 Agreeableness

    'CSN2','CSN4','CSN6','CSN8',         # 4 Conscientiousness

    'OPN2','OPN4','OPN6',                # 3 Openness

]

df2 = df1.copy()

df2[pos_questions] = df2[pos_questions].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})

df2[neg_questions] = df2[neg_questions].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})

df2.head()
#Create new columns in DF which represent sum of answers in each category (trait)

traits = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']

trait_labels = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

for trait in traits:

    new_col = str(trait+'_s')

    cols = [col for col in df2.columns if (trait in col) and ('_E' not in col)]

    df2[new_col] = df2[cols].sum(axis=1)



#Creating new DF which rows are top 50 countries (by number of respondents) and columns are means (of responces) for each trait.

count50 = list(df2['country'].value_counts().iloc[0:50].index)

traits_cols = list(df2.iloc[:,-5:].columns)

means_traits = pd.DataFrame(columns = traits_cols)

for country in count50:    

    means = {}

    for trait in traits_cols:

        means[trait] = (df2[df2['country'] == country])[trait].mean()

    means_traits.loc[country] = means
df_norm_col=(means_traits-means_traits.mean())/means_traits.std() #Normalization



for i in range(0,40,10):

    plt.subplots(figsize=(10,10)) 

    hmap = sb.heatmap(df_norm_col.iloc[i:i+10], cmap='viridis',annot=True)

    hmap.set_xticklabels(trait_labels)

    plt.show()
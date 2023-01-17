import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
%matplotlib inline
# *** Following lines if using Kaggle ***

data_new = pd.read_csv('../input/who-national-life-expectancy/who_life_exp.csv', skipinitialspace=True)
data_old = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv', skipinitialspace=True).rename(
    columns = {'Country':'country', 'Year':'year', 'Life expectancy ':'kag_life',
               'Adult Mortality':'kag_adult', 'Alcohol':'kag_alcohol',
               'BMI ':'kag_bmi', 'Polio':'kag_polio', 'Population':'kag_pop'})

# *** otherwise read from local version of file ***
#data_new = pd.read_csv('who_life_exp.csv', skipinitialspace=True)
#data_old = pd.read_csv('Life_Expectancy_Data.csv', skipinitialspace=True).rename(
#    columns = {'Country':'country', 'Year':'year', 'Life expectancy ':'kag_life',
#               'Adult Mortality':'kag_adult', 'Alcohol':'kag_alcohol',
#               'BMI ':'kag_bmi', 'Polio':'kag_polio', 'Population':'kag_pop'})

# Replace two of the country names, which changed since the Kaggle set was made
data_old['country'] = data_old['country'].replace(['Swaziland'],'Eswatini')
data_old['country'] = data_old['country'].replace(['The former Yugoslav republic of Macedonia'],'Republic of North Macedonia')

#
# make a new dataframe with some overlapping features

data_old2 = data_old[{'country', 'year', 'kag_life', 'kag_pop', 'kag_adult', 'kag_alcohol', 'kag_bmi', 'kag_polio'}]

data_new2 = data_new[{'country', 'year', 'life_expect', 'une_pop', 'adult_mortality', 'alcohol', 'bmi', 'polio'}]
# the newer data has population in thousands; the older set does not
data_new2['population'] = 1000.0 * data_new2['une_pop']

# merge tables, then remove any rows with missing values

data_new2 = data_new2.merge(data_old2, how='left')
clean_df = data_new2.dropna(axis=0)
print(clean_df.info())
list_features = [['life_expect', 'kag_life'], ['population', 'kag_pop'], ['adult_mortality', 'kag_adult'],
                 ['alcohol', 'kag_alcohol'], ['polio', 'kag_polio'], ['bmi', 'kag_bmi']]

print(list_features)

for feat1, feat2 in list_features:
    print("Plotting features:",feat1, feat2)
    plt.figure(figsize=(12,4))
    plt.subplot(1, 3, 1)
    plt.hist(clean_df[feat1])
    plt.xlabel(feat1)
    plt.subplot(1, 3, 2)
    plt.hist(clean_df[feat2])
    plt.xlabel(feat2)

    plt.subplot(1, 3, 3)
    plt.scatter(clean_df[feat1], clean_df[feat2])
    plt.plot(clean_df[feat1], clean_df[feat1], color="red")
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    
    plt.tight_layout()
    plt.show()
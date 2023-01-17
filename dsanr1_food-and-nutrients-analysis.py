import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
food = pd.read_csv('../input/food-and-nutrient-data/food.csv')
food_nutrient = pd.read_csv('../input/food-and-nutrient-data/food_nutrient.csv')
nutrient = pd.read_csv('../input/food-and-nutrient-data/nutrient.csv')
food.head()
food.info()
f'number of unique food items {pd.unique(food.iloc[:, 2]).size}'
food_nutrient.head()
food_nutrient.info()
food_nutrient['footnote'].unique()
food_nutrient[food_nutrient['footnote']=='Trace amount']
f'number of unique nutrients {pd.unique(food_nutrient.iloc[:, 2]).size}'
nutrient.head()
nutrient.info()
sns.countplot(x='unit_name', data=nutrient, order=nutrient['unit_name'].value_counts().index)
nutrients_with_names = nutrient.merge(food_nutrient.drop('id', axis=1), left_on='id', right_on='nutrient_id')
nutrients_with_names.head()
nutrients_with_names.shape
nutrients_with_names.isnull().sum()
nutrients_with_names = nutrients_with_names[['nutrient_id', 'name', 'amount', 'unit_name', 'fdc_id']]
df = pd.merge(food, nutrients_with_names, on='fdc_id')
df
df.groupby('description').count().sort_values(by='amount', ascending=False).head(15)
df.groupby(['name']).count().sort_values(by='amount', ascending=False).head(15)
nutrients = pd.unique(df['name'])
sorted(nutrients)
f'number of unique nutrient names {nutrients.size}'
vitamins = ['Vitamin A, RAE', 'Thiamin', 'Riboflavin', 'Niacin', 'Pantothenic acid', 'Vitamin B-6', 'Biotin', 'Folate, total', 'Vitamin B-12',
 'Vitamin C, total ascorbic acid', 'Vitamin D (D2 + D3)', 'Vitamin D2 (ergocalciferol)', 'Vitamin D3 (cholecalciferol)', 'Vitamin E (alpha-tocopherol)', 
 'Vitamin K (Dihydrophylloquinone)', 'Vitamin K (Menaquinone-4)', 'Vitamin K (phylloquinone)']
minerals = ['Potassium, K', 'Choline, total', 'Choline, free', 'Sodium, Na', 'Calcium, Ca', 'Phosphorus, P', 'Magnesium, Mg', 'Iron, Fe', 'Zinc, Zn', 'Manganese, Mn',
             'Copper, Cu', 'Iodine, I', 'Molybdenum, Mo', 'Selenium, Se', 'Cobalt, Co', 'Nickel, Ni', 'Boron, B']
for i in vitamins:
    print(df[df['name']==i].iloc[0]['unit_name'], 'is measured unit of ', i )
for i in minerals:
    print(df[df['name']==i].iloc[0]['unit_name'], 'is measured unit of ', i )
import difflib
matches = difflib.get_close_matches('Protein', nutrients, n=15, cutoff=.4)
matches
df[df['name']=='Protein'].sort_values(by='amount', ascending=False).head(10)
df[df['description'].str.startswith('Oil, ')]
def filter_description(df, unique_words_in_description):
    return df.loc[(df['description'].str.split(r",|;", expand=True)).drop_duplicates(subset=[i for i in range(unique_words_in_description)]).index]
filter_description(df[df['name']=='Protein'].sort_values(by='amount', ascending=False), 3)
nutrient['name'][nutrient['name'].duplicated()]
nutrient[nutrient['name']=='Energy']
(df[(df['name']=='Energy') & (df['unit_name']=='KCAL')].reset_index()['fdc_id'] == df[(df['name']=='Energy') & (df['unit_name']=='kJ')].reset_index()['fdc_id']).all()
df[df['nutrient_id']==1062].index
df_1 = df.drop(df[df['nutrient_id']==1062].index)
df_1
df_1.columns
df_1.drop(['data_type', 'food_category_id','publication_date'], axis=1, inplace=True)
df_1[df_1['description']=='Hummus, commercial'].set_index(['fdc_id', 'description', 'name'])['amount'].unstack().reset_index().rename_axis(None, axis=1)
wide_df = df_1.set_index(['fdc_id', 'description', 'name'])['amount'].unstack().reset_index().rename_axis(None, axis=1)
wide_df
wide_df.count()[vitamins]
wide_df.count()[vitamins].plot(kind='bar', figsize=(10,5))
wide_df.count()[minerals]
wide_df.count()[minerals].plot(kind='bar', figsize=(10,5))
wide_df.set_index('description').loc[:, vitamins].dropna(how='all').rank(na_option='bottom', ascending=False).sum(axis=1).sort_values()
wide_df.set_index('description').loc[:, minerals].dropna(how='all').rank(na_option='bottom', ascending=False).sum(axis=1).sort_values()
wide_df[wide_df['description']=='Cheese, swiss'].loc[:,vitamins]
wide_df.sort_values(by='Vitamin B-6', ascending=False)[['description', 'Vitamin B-6']]
wide_df.sort_values(by='Vitamin C, total ascorbic acid', ascending=False)['description']
wide_df.loc[(wide_df['Protein']/wide_df['Carbohydrate, by summation']).sort_values(ascending=False).index].head(10)
matches = difflib.get_close_matches('beans', df['description'], n=15, cutoff=.4)
matches
wide_df.iloc[wide_df.rank(numeric_only=True, ascending=False, na_option='bottom').sort_values('Zinc, Zn').index]
def foods_by_nutrient_rank(nutrient, unique_words_in_description):
    fnr = wide_df.iloc[wide_df.rank(numeric_only=True, ascending=False).sort_values(nutrient).index]
    return fnr.loc[(fnr['description'].str.split(r",|;", expand=True)).drop_duplicates(subset=[i for i in range(unique_words_in_description)]).index]
foods_by_nutrient_rank('Calcium, Ca', 1)[['description', 'Calcium, Ca']].head(10)
foods_with_high_vitamins = wide_df.rank(numeric_only=True, ascending=False, na_option='bottom').loc[:, vitamins].sum(axis=1).sort_values()
wide_df.loc[foods_with_high_vitamins.index, ['description'] + vitamins].head(10)
filter_description(wide_df.loc[foods_with_high_vitamins.index, ['description'] + vitamins], 1)
foods_with_high_minerals = wide_df.rank(numeric_only=True, ascending=False, na_option='bottom').loc[:, minerals].sum(axis=1).sort_values()
wide_df.loc[foods_with_high_minerals.index, ['description'] + minerals].head(10)
filter_description(wide_df.loc[foods_with_high_minerals.index, ['description'] + minerals], 1).head(10)
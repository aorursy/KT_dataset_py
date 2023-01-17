import numpy as np 
import pandas as pd 
from matplotlib import pyplot as pp
%matplotlib inline
from category_encoders import CountEncoder
import itertools
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
population = pd.read_csv('/kaggle/input/world-bank-data-for-nigeria/Data Science Project/Population_Stats.csv',
                         engine = 'python')
population.head()
population = population.drop(population.index[:3])
population.columns = population.iloc[0,:]
population = population.drop(population.index[0])
population.head()
population.reset_index(drop = True, inplace = True)
population.head()
values = population.columns[4:]
X = pd.melt(population, id_vars = 'Country Name',value_vars = values, var_name = 'Year', 
            value_name = 'Population' )
X.head()
population_ng = X.loc[X['Country Name'] == 'Nigeria']
population_ng.reset_index(drop = True, inplace = True)
population_ng.head()
labour_force = pd.read_csv('/kaggle/input/world-bank-data-for-nigeria/Data Science Project/Labour_Force_Stats.csv', 
                           engine = 'python')
labour_force.head()
labour_force = labour_force.drop(labour_force.index[:3])
labour_force.columns = labour_force.iloc[0,:]
labour_force = labour_force.drop(labour_force.index[0])
labour_force.head()
labour_force.reset_index(drop = True, inplace = True)
labour_force.head()
Y = pd.melt(labour_force, id_vars = 'Country Name',value_vars = values, var_name = 'Year', 
            value_name = 'Labour Force' )
Y.head()
labour_force_ng = Y.loc[Y['Country Name'] == 'Nigeria']
labour_force_ng.reset_index(drop = True, inplace = True)
labour_force_ng.head()
literacy = pd.read_csv('/kaggle/input/world-bank-data-for-nigeria/Data Science Project/Adult_Literacy_Stats.csv',
                       engine = 'python')
literacy.head()
literacy = literacy.drop(literacy.index[:3])
literacy.columns = literacy.iloc[0,:]
literacy = literacy.drop(literacy.index[0])
literacy.head()
literacy.reset_index(drop = True, inplace = True)
literacy.head()
Z = pd.melt(literacy, id_vars = 'Country Name',value_vars = values, var_name = 'Year', 
            value_name = 'Literacy')
Z.head()
literacy_ng = Z.loc[Z['Country Name'] == 'Nigeria']
literacy_ng.reset_index(drop = True, inplace = True)
literacy_ng.head()
manufacturing = pd.read_csv('/kaggle/input/world-bank-data-for-nigeria/Data Science Project/Manufacturing_Stats.csv',
                            engine = 'python')
manufacturing.head()
manufacturing = manufacturing.drop(manufacturing.index[:3])
manufacturing.columns = manufacturing.iloc[0,:]
manufacturing = manufacturing.drop(manufacturing.index[0])
manufacturing.head()
manufacturing.reset_index(drop = True, inplace = True)
manufacturing.head()
A = pd.melt(manufacturing, id_vars = 'Country Name',value_vars = values, var_name = 'Year', 
            value_name = '% of Manufacturing_Value_Added')
A.head()
manufacturing_ng = A.loc[A['Country Name'] == 'Nigeria']
manufacturing_ng.reset_index(drop = True, inplace = True)
manufacturing_ng.head()
unemployment = pd.read_csv('/kaggle/input/world-bank-data-for-nigeria/Data Science Project/Unemployment_Stats.csv', 
                           engine = 'python')
unemployment.head()
unemployment = unemployment.drop(unemployment.index[:3])
unemployment.columns = unemployment.iloc[0,:]
unemployment = unemployment.drop(unemployment.index[0])
unemployment.head()
unemployment.reset_index(drop = True, inplace = True)
unemployment.head()
B = pd.melt(unemployment, id_vars = 'Country Name',value_vars = values, var_name = 'Year', 
            value_name = 'Unemployment Rate')
B.head()
unemployment_ng = B.loc[B['Country Name'] == 'Nigeria']
unemployment_ng.reset_index(drop = True, inplace = True)
unemployment_ng.head()
first_merge = pd.merge(population_ng, labour_force_ng, how = 'outer', on = 'Year')
second_merge = pd.merge(first_merge, literacy_ng, how = 'outer', on = 'Year')
third_merge = pd.merge(second_merge, manufacturing_ng, how = 'outer', on = 'Year')
final_merge = pd.merge(third_merge, unemployment_ng, how = 'outer', on = 'Year')
ng_df = final_merge.drop(['Country Name_x', 'Country Name_x', 'Country Name_y', 
                         'Country Name_y', 'Country Name'], axis = 1)
ng_df.head()
ng_df.dtypes
ng_df.head()
ng_df.isna().sum()
pp.figure(figsize = (10, 8))
pp.plot(ng_df['Year'][-10:], ng_df['Population'][-10:])
pp.xlabel('Year')
pp.ylabel('Population')
pp.title("Population Growth Over the last decade")
pp.figure(figsize = (10, 8))
pp.plot(ng_df['Year'][-10:], ng_df['Labour Force'][-10:])
pp.xlabel('Year')
pp.ylabel('Labour Force')
pp.title("Labour Force Growth Over the last decade")
pp.figure(figsize=(10.5, 8))
pp.plot(ng_df['Year'][30:58:], ng_df['% of Manufacturing_Value_Added'][30:58])
pp.xlabel('Year')
pp.ylabel('Manufacturing Value Added')
pp.title("Manufacturing value created from 1992 to 2016")
updated_unemployment_rate = [3.638, 3.673, 3.743, 3.756, 3.759, 3.77, 3.761, 3.758, 3.793, 3.78,3.778,3.817,3.821,3.786,
         3.74,3.646, 3.565, 3.539,3.722,3.767,3.77,3.735,3.703,4.562,4.311,7.06,8.389,8.243,8.096]
ng_df['Unemployment Rate'][31:60] = updated_unemployment_rate
pp.figure(figsize = (10, 8))
Years = ['2013', '2014', '2015', '2016', '2017', '2018', '2019']
y_pos = np.arange(len(Years))
pp.bar(y_pos,ng_df['Unemployment Rate'][53:60])
pp.xticks(y_pos, Years)
pp.title('Unemployment Rate from 2013 to 2019')
pp.show()
cols = ['Year', '% of Manufacturing_Value_Added', 'Labour Force', 'Unemployment Rate']


ng_final_df = ng_df[cols][31:60]
ng_final_df.head()
import itertools

def next_row(data):
    
    a, b = itertools.tee(data)
    next(b, None)
    return list(zip(a,b))

def change(data):
    nextrow = next_row(data['% of Manufacturing_Value_Added'])
    new_list = [data.iloc[0]['% of Manufacturing_Value_Added']]
    for i in nextrow:
        answer = (i[1] - i[0])/i[0]
        new_list.append(answer)
    return new_list
ng_final_df['% increase in Manufacturing_Value_Added'] = change(ng_final_df)
ng_final_df['% increase in Manufacturing_Value_Added'] = ng_final_df['% increase in Manufacturing_Value_Added'].fillna(0)
ng_final_df.drop(['% of Manufacturing_Value_Added'], axis = 1, inplace = True)
ng_final_df = ng_final_df.drop(ng_final_df.index[0])
interactions_list = list(itertools.combinations(ng_final_df.columns,2))
for i in interactions_list:
    if not 'Unemployment Rate' in i:
        ng_final_df[i[0] + '_' + i[1]] = ng_final_df[i[0]].astype(str) + '_' + ng_final_df[i[1]].astype(str)
ng_final_df.head()
valid_fraction = 0.1
ng_final_df = ng_final_df.sort_values('Year')
valid_rows = int(len(ng_final_df) * valid_fraction)
train = ng_final_df[:-valid_rows]
# valid size == test size, last two sections of the data
valid = ng_final_df[-valid_rows:]

cat_cols = [col for col in ng_final_df.columns if ng_final_df[col].dtype == 'object']
print('The categorical columns: \t')
print(cat_cols)
print(train)
print('-'*40)
print(valid)
count_enc = CountEncoder(cols = cat_cols)


train_encoded = count_enc.fit_transform(train[cat_cols])
train = train.join(train_encoded.add_suffix('_count'))
train.drop(cat_cols, axis=1, inplace = True)


valid_encoded = count_enc.fit_transform(valid[cat_cols])
valid = valid.join(valid_encoded.add_suffix('_count'))
valid.drop(cat_cols, axis=1, inplace = True)

features = [col for col in train.columns if col != 'Unemployment Rate']

model = DecisionTreeRegressor(max_depth = 1, random_state = 0)
model.fit(train[features], train['Unemployment Rate'])

preds = model.predict(valid[features])
error = mean_absolute_error(preds,valid['Unemployment Rate'])
error
pp.figure(figsize=(10.5, 8))
pp.plot(ng_final_df['Year'], ng_final_df['% increase in Manufacturing_Value_Added'])
pp.xlabel('Year')
pp.ylabel('Increase Manufacturing Value Added')
pp.title("Change in Manufacturing value created from 1992 to 2019")
ng_experiment = pd.DataFrame({'Year': [2020,2021, 2022, 2023, 2024], 'Labour Force':
                            [61174048,62502776,63860366,65246432,66663615], 
                             '% increase in Manufacturing_Value_Added': [1,1,1,1,1]})
new_interactions_list = list(itertools.combinations(ng_experiment.columns,2))
for i in new_interactions_list:
    ng_experiment[i[0] + '_' + i[1]] = ng_experiment[i[0]].astype(str) + '_' + ng_experiment[i[1]].astype(str)
ng_experiment.head()
X_test = ng_experiment
X_test_encoded = count_enc.fit_transform(X_test[cat_cols])
X_test = X_test.join(X_test_encoded.add_suffix('_count'))
X_test.drop(cat_cols, axis = 1, inplace = True)

preds = model.predict(X_test)
preds
final_df = pd.DataFrame({'Year': [2019,2020,2021, 2022, 2023, 2024], 'Labour Force':
                            [59873566, 61174048, 62502776,63860366,65246432,66663615], 
                             '% increase in Manufacturing_Value_Added': [0,1,1,1,1,1]})
final_df['Unemployment Rate'] = [8.096, 7.7245, 7.7245, 7.7245, 7.7245, 7.7245]
final_df
pp.figure(figsize = (10, 8))
new_Years = ['2019', '2020', '2021', '2022', '2023', '2024']
new_y_pos = np.arange(len(new_Years))
pp.bar(new_y_pos,final_df['Unemployment Rate'])
pp.xticks(new_y_pos, new_Years)
pp.title('Predicted Unemployment Rate with increase in Manufacturing Value created in Tech Industry')
pp.show()
power_bi_df = ng_final_df.copy()
power_bi_df.columns
power_bi_df.drop(['Year_Labour Force', 'Year_% increase in Manufacturing_Value_Added',
                 'Labour Force_% increase in Manufacturing_Value_Added'], axis = 1, inplace = True)
power_bi_df.to_csv('power_bi_df.csv')
final_df.to_csv('predicted_df.csv')

# Data engineering.
import pandas as pd
import numpy as np

# Regular expressions module.
import re

# Data visualization and frame's visualization options.
import missingno as msno # Copyright (c) 2016 Aleksey Bilogur
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
sns.set_style("whitegrid")
# Load data
df = pd.read_csv('../input/food_coded.csv')
# Check data frame's shape
df.shape
df.head() 
# Data features
df.columns
# Explore the dataset
df.describe(include='all').T
from collections import Counter
# Explore datatypes
col_dtypes = np.array([df[x].dtype for x in df.columns])

# Quick features dtype counter
for i, j in Counter(col_dtypes).items():
    print('dtype: ', i, ' --- ', 'value: ', j)
# Quick look at features list with object datatype
df_obj = pd.DataFrame({'dtype_': col_dtypes}, index=df.columns)
# Slice the dtype
df_obj = df_obj[df_obj['dtype_'] == object]
df_obj
types = {}

for feature in df_obj.index.values:
    feat_dict = {}
    
    for value in df[feature].values:
        # Take out dtype from a string with regex
        dtype = str(type(value))
        match = re.search("int|float|str", dtype)

        # Create a dict with number of dtypes for particular feature
        if match.group() not in feat_dict.keys():
            feat_dict[match.group()] = 1
        else:
            feat_dict[match.group()] += 1
    types[feature] = feat_dict
    # Clean up the dict before next iteration
    feat_dict = {}
# Create transposed data frame with dtypes counter for each object feature
df_type = pd.DataFrame.from_dict(types).T
# Fill missing data with zeros
df_type.fillna(value=0)
# Check how many features have missing data
df.isnull().any().value_counts()
# Amount of NaN values for each feature
total = df.isnull().sum().sort_values(ascending=False)
# Percentage part of total
percent = (df.isnull().sum()/df.isnull().count()*100).round(1).sort_values(ascending=False)
# Merge series
nan_data = pd.concat({"# of NaN's": total, '% of Total': percent}, axis=1)
nan_data.head(10)
# Use missingno module for NaN's distribution
msno.matrix(df[df.columns[df.isnull().any()]])
(df['comfort_food_reasons_coded'] == df['comfort_food_reasons_coded.1']).value_counts()
df[['comfort_food_reasons', 'comfort_food_reasons_coded.1']].tail(20)
df.drop(['comfort_food_reasons', 'comfort_food_reasons_coded'], axis=1, inplace=True)
df.rename(columns={'comfort_food_reasons_coded.1': 'comfort_food_reasons'}, inplace=True)
df['GPA'].unique()
# Take the most common value to fill. Assume that it will not oversimplify model (<5% of columns data to replace).
df['GPA'].value_counts().head()
# Use regex to clean blended data, fill missing values and set up dtype
df['GPA'] = df['GPA'].str.replace(r'[^\d\.\d+]', '').replace((np.nan, ''), '3.5').astype(float).round(2)
# Values after changes
df['GPA'].unique()
# Boxplot to visualize and check results
fig, ax = plt.subplots(figsize=[12,6])
sns.boxplot(df['GPA'])
ax.set_title("'GPA' distribution")
# Some strings, blended and missing values
df['weight'].unique()
# Clean blended values, non-numeric become NaN
df['weight'] = df['weight'].str.replace(r'[^\d\d\d]', '').replace('', np.nan).astype(float)
df['weight'].unique()
# Dict contains mean values of weights for both genders
weight_mean = {}
# Dict contains std values of weights for both genders
weight_std = {}

# Set plot size
fig, ax = plt.subplots(figsize=[16,4])

# Create two distributions for both genders
for gen, frame in df[['Gender', 'weight']].dropna().groupby('Gender'):
    weight_mean[gen] = frame['weight'].values.mean()
    weight_std[gen] = frame['weight'].values.std()
    sex_dict = {1: 0, 2: 1}
    sns.distplot(frame['weight'], ax=ax, label=['Female', 'Male'][sex_dict[gen]])

ax.set_title('weight distribution, hue=Gender')
ax.legend()
# Let's check rows with NaN weight values
df[df['weight'].isnull()]
# Let's check how many NaN's these 3 rows have
inv_data = {} # key - observation index; values - (actual value, # of row's NaNs, gender)

for index, row in df[df['weight'].isnull()].iterrows():
    # Variable for printing results and getting # of NaNs
    temp = row.isnull().value_counts()
    print('Index: ', temp.name, " --- # of NaN's: ", temp.values[1])
    
    # Adds to dict 3 values by data frame's index - (actual value, # of row's NaNs, gender)
    inv_data[str(index)] = (row['weight'], temp.values[1], row['Gender'])        
# Condition for dropping a row
drop_cond = df.shape[1]/2

for df_index, tuple_ in inv_data.items():
    # Row with # of NaNs > 'drop_cond' will be dropped.
    if tuple_[1] > drop_cond:
        df.drop(int(df_index), inplace=True)
    # Weight's NaN will be replaced with random number based on mean value 
    # in regards to the standard deviation.
    else:
        # Mean value of weights set in regards to gender
        mean_val = weight_mean[tuple_[2]]
        # The standard deviation value of weights set in regards to gender
        std_val = weight_std[tuple_[2]]
                
        # Random value creator in range defined by mean and the standard deviation 
        # value in regards to gender
        rand_val = np.random.randint(mean_val - std_val, mean_val + std_val)
        
        # Replacing NaN's with prepared value
        df['weight'].values[int(df_index)] = rand_val
# Finally we can change column dtype.
df['weight'] = df['weight'].astype(int)
# Slice binary features and their values amount without NaNs
x = df.describe().T
y = pd.Series(x[x['max'] == 2]['count'], index=x[x['max'] == 2].index)

# Percent values of 0/1 for each feature
zero_list = []
one_list = []

# Convert into percentages
for ind, col in y.iteritems():
    zero_list.append(((df[ind]==1).sum()*100)/col)
    one_list.append(((df[ind]==2).sum()*100)/col)
# Plot preparation
plt.figure()
fig, ax = plt.subplots(figsize=(6,6))

# Create barplots
sns.barplot(ax=ax, x=x[x['max'] == 2].index, y=zero_list, color="blue")
sns.barplot(ax=ax, x=x[x['max'] == 2].index, y=one_list, bottom= zero_list, color="red")

# Plot labels
plt.ylabel('Percent of zero/one [%]', fontsize=16)
plt.xlabel('Binary features', fontsize=16)

# Plot's font settings
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.tick_params(axis='both', which='major', labelsize=16)
def cat_bin(data, x_var, hue_var, hue_class, x_names=[], hue_names=[]):
    # Prepare dichotomous and/or ordinal variable/s to graphic representation
    
    # Create data frame
    df = pd.DataFrame(index=x_names)
    
    # Prepare converted values for each hue
    for i, j in [(name, ind+1) for ind, name in enumerate(hue_names)]:
        df[i] = data[x_var][data[hue_var] == j].value_counts().sort_index().values
        df[i] = ((df[i]/df[i].sum())*100).round(1)
        
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'categories'}, inplace=True)
    
    return pd.melt(df, id_vars="categories", var_name=hue_class, value_name="percent")
# New category names
objects = ['daily', '3 times/w', 'rarely','holidays', 'never']
# Data preparation for plotting
df_cook_sex = cat_bin(df, 'cook', 'Gender', 'sex', objects, hue_names=['Female', 'Male'])

with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's barplot creator
    sns.factorplot(x='categories', y='percent', hue='sex', data=df_cook_sex, 
                   kind='bar', palette="muted", size=8)
    # Plot labels
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Percent of Female/Male [%]', fontsize=18)
    plt.title('Cooking frequency', y=1.01, size=25)
# New category names
objects = ['very low', 'low', 'average','above-aver.', 'high', 'very high']
# Data preparation for plotting
df_inc = cat_bin(df, 'income', 'Gender', 'sex', objects, hue_names=['Female', 'Male'])

with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's barplot creator
    sns.factorplot(x='categories', y='percent', hue='sex', data=df_inc, 
                   kind='bar', palette='inferno', size=8)
    # Plot labels
    plt.xlabel('Income', fontsize=18)
    plt.ylabel('Percent of Female/Male [%]', fontsize=18)
    plt.title('Income distribution', y=1.01, size=25) 
# New category names
objects = ['daily', '3 times/w', 'rarely','holidays', 'never']
# Create data frame
df_camp = pd.DataFrame(index=objects)

# Add new columns in df_camp with data
df_camp['on_campus'] = df['cook'][df['on_off_campus'] == 1].value_counts().sort_index().values
# Values != 1 are different types of accommodation outside the campus
df_camp['off_campus'] = df['cook'][df['on_off_campus'] > 1].value_counts().sort_index().values

# Prepare converted values for each hue
df_camp['on_campus'] = ((df_camp['on_campus']/df_camp['on_campus'].sum())*100).round(1)
df_camp['off_campus'] = ((df_camp['off_campus']/df_camp['off_campus'].sum())*100).round(1)

df_camp.reset_index(inplace=True)
df_camp.rename(columns={'index': 'categories'}, inplace=True)

# Reshaping data frame
df_camp = pd.melt(df_camp, id_vars="categories", var_name='on/off campus', value_name="percent")  
with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's barplot creator
    sns.factorplot(x='categories', y='percent', hue='on/off campus', data=df_camp, 
                   kind='bar', palette="dark", size=8)
    # Plot labels
    plt.xlabel('Frequency', fontsize=18)
    plt.ylabel('Percent of on/off campus [%]', fontsize=18)
    plt.title('Cooking frequency', y=1.01, size=25)  
df_parent = df[['parents_cook', 'cook']]
df_parent.corr(method = 'spearman')
# New category names
frequency = ['never', 'rare', 'sometimes', 'often', 'always']
# Data preparation for plotting
df_sport = cat_bin(df, 'nutritional_check', 'sports', 'activity', 
                   frequency, hue_names=['no_sport', 'sport'])

with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's barplot creator
    sns.factorplot(x='percent', y="categories", hue="activity", data=df_sport, 
                   kind='bar', palette="Greens_d", size=8)
    # Plot labels
    plt.xlabel('Percent dependent of activity [%]', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title('Nutritional check frequency', y=1.01, size=25)
sport_corr = df[['nutritional_check', 'sports']]
sport_corr.corr(method = 'kendall')
# New category names
food = ['greek_food', 'indian_food', 'italian_food', 'thai_food', 'persian_food', 'ethnic_food']
# Set diverging palette
cmap = sns.diverging_palette(50, 10, as_cmap=True)
# Plot preparation
plt.figure(figsize=(12,12))
plt.title('Correlation between food preferences', y=1.01, size=20)

with sns.plotting_context(context='notebook', font_scale=1.5):
    # Seaborn's heatmap creator
    g = sns.heatmap(df[food].corr(method='spearman'),linewidths=0.5,vmax=1.0, square=True, center=0,
                    cmap=cmap, annot=True, cbar_kws={"shrink": .75, "orientation": "horizontal"})
    # Plot labels
    loc, labels = plt.xticks()
    g.set_xticklabels(labels, rotation=45)
    g.set_yticklabels(labels, rotation=45)
# Slice of data frame
df_weight = df[['veggies_day', 'fruit_day', 'weight']]

# Data preparation for plotting
df_weight = pd.melt(df_weight, id_vars='weight', var_name='day', value_name="frequency")

# Seaborn's barplot creator
g = sns.factorplot(x="frequency", y="weight", col="day", data=df_weight, 
                   kind='box', palette="YlGnBu_d", size=8, aspect=.75)
g.despine(left=True)
sns.factorplot(x="frequency", y="weight", col="day", data=df_weight, 
               kind='swarm', palette="YlGnBu_d", size=8, aspect=.75)
g
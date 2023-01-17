import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import scipy
df_2015 = pd.read_csv("../input/world-happiness/2015.csv")  
df_2016 = pd.read_csv("../input/world-happiness/2016.csv") 
df_2017 = pd.read_csv("../input/world-happiness/2017.csv") 
df_2015.head()
#Final set odf common variable matched in 3 dataset
df_2015 = df_2015[['Country','Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
                       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)']]
df_2016 = df_2016[['Country','Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',
                       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)']]
df_2017 = df_2017[['Country', 'Happiness.Rank', 'Happiness.Score', 'Economy..GDP.per.Capita.', 'Family',
                       'Health..Life.Expectancy.', 'Freedom', 'Generosity', 'Trust..Government.Corruption.']]

#Final Names given to each variable
new_names = ['Country or region', 'Happiness Rank', 'Happiness Score', 'GDP per Capita', 'Social support',
                       'Healthy Life Expectancy', 'Freedom', 'Generosity', 'Perceptions of corruption']
df_2015.columns = new_names
df_2016.columns = new_names
df_2017.columns = new_names
#A column named Year is added
df_2015['Year'] = 2015
df_2016['Year'] = 2016
df_2017['Year'] = 2017
#concatenate the dataset
data = pd.concat([df_2015, df_2016, df_2017], axis=0)
data.head(3)
#2018 and 19 dataset are different compare to other 3 year 
# New data
df_2018 = pd.read_csv("../input/world-happiness/2018.csv")
df_2019 = pd.read_csv("../input/world-happiness/2019.csv")
# Concatenate data
df_2018['Year'] = 2018
df_2019['Year'] = 2019
new_data = pd.concat([df_2018, df_2019], axis=0)
# Switching overall rank column with country/ region
columns_titles = ['Country or region', 'Overall rank', 'Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Year']
new_data = new_data.reindex(columns=columns_titles)
# Renaming old data columns i.e. the 3 year dataset
old_data = data[['Country or region', 'Happiness Rank', 'Happiness Score', 'GDP per Capita', 'Social support',
                       'Healthy Life Expectancy', 'Freedom', 'Generosity', 'Perceptions of corruption','Year']]
old_data.columns = ['Country or region', 'Overall rank', 'Score', 'GDP per capita',
       'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Year']
final.head(3)
#Extract final dataset
final.to_csv (r'C:\Users\Pritha Roy Choudhary\Downloads\Kaggle\final.csv', index = False, header=True)
# Double check to see if there are any missing values left
plt.figure(figsize = (16,6))
sns.heatmap(data = data.isna(), cmap = 'Blues')

plt.xticks(fontsize = 13.5);
data[data['Perceptions of corruption'].isna()]
final.describe()
final.info()
#Columns overview
#setting color palette and style for the rest of the graphs
sns.set_palette("GnBu_d")
sns.set_style('darkgrid')
sns.jointplot(x = 'GDP per capita', y = 'Score', data = final)
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(final.corr(), annot = True, linewidth = 0.5, fmt='.2f', ax=ax)
plt.show()
#Important check - Normal Distribution
sns.distplot(final['Score'])
#Consider X and Y as per correlation plot
#Before running in pycaret checked by doing MLR
X = final[['Perceptions of corruption', 'Freedom to make life choices', 'Social support','GDP per capita','Generosity']]
y = final['Score']
from pycaret.regression import *
exp_reg = setup(df, target = 'Score')
ridge_reg = create_model('ridge')
tuned_ridge = tune_model('ridge')
plot_model(estimator=tuned_ridge)
X = df[['Country or region','Overall rank','GDP per capita', 'Social support', 'Healthy life expectancy','Freedom to make life choices','Generosity','Perceptions of corruption','Year']]
pred = predict_model(tuned_ridge, data=X)

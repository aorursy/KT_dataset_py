# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import warnings
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 15, 7
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
wine_df = pd.read_csv('/kaggle/input/wine-reviews/winemag-data-130k-v2.csv')
wine_df.head()
num_rows = wine_df.shape[0]
num_cols = wine_df.shape[1]
print('Number of rows: ', num_rows)
print('Number of columns: ', num_cols)
wine_df.drop(columns='Unnamed: 0', inplace = True)
wine_df.info()
# The columns with categorical data type
cat_cols_name = wine_df.columns[wine_df.dtypes=='object']
cat_cols_nunique = wine_df[cat_cols_name].nunique()
cat_cols_unique = [wine_df[cols_name].unique() for cols_name in cat_cols_name]
data = {'cat_cols_nunique': cat_cols_nunique, 'cat_cols_unique':cat_cols_unique}
# Create dataframe
df_cat_cols = pd.DataFrame(data, columns = ['cat_cols_nunique', 'cat_cols_unique'])
df_cat_cols
wine_df.isnull().sum()
wine_df.drop(columns=['designation', 'taster_twitter_handle'], inplace = True)
wine_df[wine_df['country']=='Egypt']
wine_df = wine_df[wine_df['country']!='Egypt'].reset_index(drop = True)
num_rows -= 1 # update value
# Handle missing in categorical variable
cat_name = ['country', 'region_1', 'region_2', 'taster_name', 'province', 'variety']
for name in cat_name:
    wine_df[name].fillna('unknown', inplace = True)
#Handle missing in numerical variable

isnull_price = wine_df['price'].isnull()
mean_price_wine_byCountry = wine_df.groupby('country')['price'].agg('mean')
wine_df.loc[isnull_price,'price'] = [mean_price_wine_byCountry[row[0]] for row in wine_df[isnull_price].values]
# Check again
wine_df.isnull().sum()
# Create new attribute is provinceOfCountry -  it's useful for visualization
wine_df['provinceOfCountry'] = '( ' + wine_df['country'] + ' ) ' + wine_df['province']
wine_df['provinceOfCountry'].head()
## Some simple descriptive statistic
wine_df.describe()
plt.figure(figsize = (15, 8))
sns.distplot(wine_df['price'])
plt.title("Price distribution", fontsize = 30)
# Creat a new log_price property
wine_df['price_log'] = np.log(wine_df['price'])
plt.figure(figsize = (15, 8))
sns.distplot(wine_df['price_log'])
plt.xlabel('Price ( e^x)', fontsize = 20)
plt.title("Price log distribution", fontsize = 30)
plt.figure(figsize = (15, 8))
sns.distplot(wine_df['points'])
plt.title("Distribution of points", fontsize = 30)
plt.figure(figsize = (10, 5))
plot = sns.regplot(x = 'points', y = 'price_log', data = wine_df, line_kws={'color':'red'}, 
                  x_jitter=True, fit_reg=True, color='darkgreen')
plot.set_title("Corr between price and points")
plt.show()
plt.figure(figsize = (15, 7))
country = wine_df['country'].value_counts().sort_values(ascending = False)
country[:20].plot.bar(color = 'darkgreen')
plt.xticks(rotation = 45)
plt.show()

def barchart_plot(distinct_val, attr_gr, attr_agg, x_label, y_label, title):
    '''
    func: barchart drawing
    ---------
    Parameter:
    
    distinct_val: list 1-d

    attr_gr: string
             the property for groupby
    attr_agg: string 
              the property for operation of aggregation
    x_label : string
              label for x coordinate in chart
    y_label : string
              label for y coordinate in chart
    title   : string
              title for chart
    
    '''
    df_top = wine_df.loc[wine_df[attr_gr].isin(distinct_val)]
    df = df_top.groupby(attr_gr)[attr_agg].nunique().sort_values(ascending = False)
    df.plot.bar(use_index = True, color = 'darkgreen')
    plt.xlabel(x_label, fontsize = 20)
    plt.ylabel(y_label,fontsize = 20)
    plt.xticks(rotation = 40, fontsize = 10)
    plt.title(title, fontsize = 20)
    plt.show()
    
    
def boxplot_chart(distinct_val, x, y, title):
    '''
    func:  boxplot
    ---------
    Parameter:
    
    distinct_val: list 1-d
                  
    x: string
    y: string 
    title   : string
              title for chart
    '''
    plt.figure(figsize = (15, 7))
    df_top = wine_df.loc[wine_df[x].isin(distinct_val)]
    sns.boxplot(x = x , y = y, data = df_top, color = 'darkgreen')
    plt.xticks(rotation = 45, fontsize = 10)
    plt.title(title, fontsize = 20)
    plt.show()
    
province = wine_df['provinceOfCountry'].value_counts()[:20]
top_province = province.index.values
barchart_plot(top_province ,'provinceOfCountry', 'region_1', 'Province', 'Number of region1' ,'Province with the most region_1')
boxplot_chart(top_province, 'provinceOfCountry', 'price', 'The regions that produce the first-class wines' )
barchart_plot(top_province, 'provinceOfCountry', 'title', 'Province', 'Count', 'The province owns the most wines' )
barchart_plot(top_province, 'provinceOfCountry', 'variety', 'Province', 'Variety of grapes',\
                                                  'The regions that have the most type of grapes' )
wine_df[wine_df['province']=='California']['variety'].value_counts(ascending=False).index[:10]
boxplot_chart(top_province, 'provinceOfCountry', 'points', 'Point by Province')
top_country = country.index.values[:20]
barchart_plot(top_country, 'country', 'province', 'Country', 'Number of provinces', 'The country that have the most provinces')
boxplot_chart(top_country, 'country', 'price_log', "Price by Country's")
boxplot_chart(top_country, 'country', 'points', "Point by Country's")
top_manufacture_of_wine = wine_df.groupby(['winery', 'country']).agg({'points':'mean', 
                                        'price': ['mean', 'size']}).sort_values(by=[('price', 'size')], ascending = False)
top_manufacture_of_wine[:10]
winery = wine_df['winery'].value_counts()[:20]
top_winery = winery.index.values
boxplot_chart(top_winery, 'winery', 'price_log', "Price by winery's" )
boxplot_chart(top_winery, 'winery', 'points', "Point by winery's") 
barchart_plot(top_country, 'country', 'winery', 'Count', 'Country', 'The country that have the most winery')
variety = wine_df['variety'].value_counts()[:20]
top_variety = variety.index.values
variety.plot.bar(use_index = True, color = 'darkgreen')
plt.title("Grapes for wine", fontsize = 30)
plt.ylabel("Count", fontsize = 20)
plt.xlabel("Grape", fontsize = 20)
plt.xticks(rotation= 60)
plt.show()
df_top_variety = wine_df.loc[wine_df['variety'].isin(top_variety)]
grape_in_country_nunique = df_top_variety.groupby('variety')['country'].nunique()
grape_in_country_unique = df_top_variety.groupby('variety')['country'].unique()
data = {'grape_in_country_nunique': grape_in_country_nunique, 'grape_in_country_unique': grape_in_country_unique}
df_variety = pd.DataFrame(data)
df_variety = df_variety.sort_values(by='grape_in_country_nunique', ascending = False)
df_variety
df_variety.plot.bar(use_index = True, y = 'grape_in_country_nunique', color = 'darkgreen')
plt.title("Number of countries where grapes are grown", fontsize = 20)
plt.ylabel("Count", fontsize = 15)
plt.xlabel("Grape", fontsize = 15)
plt.xticks(rotation = 45, fontsize = 10)
plt.show()
grape_for_price = df_top_variety.groupby(['variety', 'provinceOfCountry'])['price'].agg('mean')

grape_for_price = grape_for_price.unstack('variety')
grape_for_price
grape_in_best_province = grape_for_price.idxmax(axis = 0)
print(" Loại nho Red Blend cho giá trị rượu tốt nhất ở: ", grape_in_best_province['Red Blend'])
print(" Loại nho Pinot Noir cho giá trị rượu tốt nhất ở: ", grape_in_best_province['Pinot Noir']) 
print(" Loại nho Chardonnay cho giá trị rượu tốt nhất ở: ", grape_in_best_province['Chardonnay']) 
print(" Loại nho Cabernet Sauvignon cho giá trị rượu tốt nhất ở: ", grape_in_best_province['Cabernet Sauvignon']) 
print(" Loại nho Merlot cho giá trị rượu tốt nhất ở: ", grape_in_best_province['Merlot']) 
boxplot_chart(top_variety, 'variety', 'price_log', 'Price by variety of grapes')
boxplot_chart(top_variety, 'variety', 'points', 'Points by variety of grapes')
df = wine_df[wine_df['taster_name']!='unknown']
print( df['taster_name'].nunique())
sns.countplot(x = 'taster_name', data = df, color = 'darkgreen')
plt.xlabel('Taster name', fontsize=20)
plt.ylabel('Count', fontsize = 20)
plt.xticks(rotation = 45, fontsize = 10)
plt.title('Taster name count', fontsize = 20)
plt.show()
boxplot_chart( df['taster_name'].unique(), 'taster_name', 'points', "Point by taster's" )
boxplot_chart( df['taster_name'].unique(), 'taster_name', 'price_log', "price by taster's" )
wine_df.iloc[68586, 8:11]
wine_df[wine_df['title'].str.contains('1105', '2014')]
wine_df.iloc[63, 8:11]
number_in_title = wine_df['title'].apply(lambda x: re.findall(r'\d{4}', x))
number_in_winery = wine_df['winery'].apply(lambda x: re.findall(r'\d{4}', x))
act_year = [list(set(number_in_title[i]) - set(number_in_winery[i])) if number_in_winery[i]!=None
                                                                     else number_in_title[i] for i in range(num_rows)]
year_generate = []
for idx, v in enumerate(act_year):
    if len(v)==1:
        year_generate.append(v[0])
        continue
    if len(v)>1:
        arr = []
        for y in v:
            if y<='2020':
                arr.append(y)
        year_generate.append(max(arr))
        continue
    if len(v)==0:
        year_generate.append(year_generate[idx-1])
wine_df = wine_df.assign(year = year_generate)
wine_df['year'] = wine_df['year'].astype(int)
print("Tổng số mẫu Null của thuộc tính year: ", wine_df['year'].isnull().sum())
print("\nMiền giá trị trong year: ",  wine_df['year'].unique())
wine_df[wine_df['title'].str.contains(r'1607|1503')]
wine_df = wine_df[~((wine_df['year']==1503) | (wine_df['year']==1607))]
wine_df = wine_df.assign(age = wine_df['year'].apply(lambda x: 2019 - x ))

# TCreate age property
wine_df['age'].describe()
plot = sns.countplot(x = 'age', data=wine_df, color = 'darkgreen')
plot.set_xticklabels(plot.get_xticklabels(), rotation = 100)
plt.show()
plt.figure(figsize = (15, 10))
plot = sns.regplot(x = 'age', y = 'price_log', data = wine_df, line_kws={'color':'red'}, 
                  x_jitter=True, fit_reg=True, color='darkgreen')
plot.set_title("Corr between price and points")
plt.show()
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from pylab import rcParams

from sklearn.model_selection import train_test_split

import lightgbm as lgb

%matplotlib inline
prod_df = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')

cat_df = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv')

cat_sorted_df = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')
prod_df.info()
prod_df.head()
prod_df.drop_duplicates(inplace=True)
def find_null(dataFrame):

    total = len(dataFrame.columns)

    n = 0

    for col in dataFrame.columns:

        null_sum = dataFrame[col].isna().sum()

        if null_sum > 0:

            n+=1

            print(f'"{col}": {null_sum} null values')

            

    print('---------------------------------------')

    print(f'\n{n}/{total} columns have null values')
find_null(prod_df)
prod_df['product_color'].unique()
tmp_df = prod_df['product_color'].value_counts()

tmp_df = tmp_df.reset_index()

tmp_df['percentage'] = tmp_df['product_color'] / prod_df.shape[0]

tmp_df.head(15)
color_convert_dict = {

          'white':'white', 'green':'green','leopardprint':'other',

          'black':'black','yellow':'yellow', 'blue':'blue', 'navyblue':'blue',

          'grey':'grey','orange':'orange', 'red':'red','beige':'other',

          'lightblue':'blue','Black':'black','armygreen':'green',

          'khaki':'green', 'red & blue':'multicolor', 'blue & pink':'multicolor',

          'purple':'purple','RED':'red', 'white & green':'multicolor',

          'winered':'red', 'black & green':'multicolor','whitefloral':'white',

          'floral':'other', 'fluorescentgreen':'green', 'orange & camouflage':'orange',

          'lightyellow':'yellow', 'coolblack':'black','camouflage':'other','pink':'pink',

          'lightpink':'pink', 'pink & black':'multicolor', 'silver':'other',

          'Army green':'green', 'lightgreen':'green','mintgreen':'green',

          'pink & grey':'multicolor', 'gray':'grey', 'coffee':'other', 'rose':'red',

          'light green':'green', 'leopard':'other', 'black & white':'multicolor',

          'orange-red':'red','dustypink':'pink', 'White':'white', 'star':'other',

          'white & black':'multicolor', 'apricot':'orange','skyblue':'blue',

          'burgundy':'red', 'claret':'purple', 'pink & white':'multicolor',

          'rosered':'red', 'tan':'other','navy blue':'blue', 'wine red':'red',

          'lightred':'red', 'Pink':'pink', 'coralred':'red', 'brown':'other',

          'lakeblue':'blue', 'darkblue':'blue', 'camel':'other', 'pink & blue':'multicolor',

          'nude':'other', 'Blue':'blue','lightpurple':'purple', 'army':'other',

          'black & stripe':'multicolor', 'Rose red':'red','greysnakeskinprint':'other',

          'denimblue':'blue', 'applegreen':'green', 'offwhite':'white','lightgray':'grey',

          'navy':'blue', 'gray & white':'multicolor', 'brown & yellow':'multicolor',

          'winered & yellow':'multicolor', 'army green':'green', 'whitestripe':'white',

          'rainbow':'other','lightgrey':'grey', 'watermelonred':'green',

          'prussianblue':'blue', 'navyblue & white':'multicolor','white & red':'multicolor',

          'wine':'red', 'ivory':'white', 'black & yellow':'multicolor', 'jasper':'other',

          'lightkhaki':'green', 'offblack':'black', 'violet':'purple',

          'black & blue':'multicolor', 'blackwhite':'other','darkgreen':'green', 'rosegold':'pink',0:'other'

}
prod_df['product_color'] = prod_df['product_color'].map(color_convert_dict)

prod_df['product_color'] = prod_df['product_color'].fillna('other')

prod_df['product_color'].unique()



tmp_df = prod_df['product_color'].value_counts()

tmp_df = tmp_df.reset_index()

tmp_df['percentage'] = tmp_df['product_color'] / prod_df.shape[0]

print('--- After converting product color ---')

tmp_df
prod_df['origin_country'] = prod_df['origin_country'].fillna('unknown')

prod_df['origin_country'].unique()
prod_df["product_variation_size_id"].unique()
tmp_df = prod_df['product_variation_size_id'].value_counts()

tmp_df = tmp_df.reset_index()

tmp_df['percentage'] = tmp_df['product_variation_size_id'] / prod_df.shape[0]

tmp_df.head(15)
product_size_convert_dict = {

         'M':'M', 'XS':'XS', 'S':'S', 'Size-XS':'XS', 'M.':'M',

         'XXS':'XXS', 'L':'L', 'XXL':'XXL','S.':'S',

         's':'S','choose a size':'other', 'XS.':'XS', '32/L':'L',

         'Suit-S':'S', 'XXXXXL':'other', 'EU 35':'S',

         '4':'other','Size S.':'S', '1m by 3m':'other', '3XL':'other',

         'Size S':'S', 'XL':'XL', 'Women Size 36':'S',

         'US 6.5 (EU 37)':'M', 'XXXS':'other', 'SIZE XS':'XS',

         '26(Waist 72cm 28inch)':'M','Size XXS':'XXS',

         '29':'other', '1pc':'other', '100 cm':'other',

         'One Size':'other', 'SIZE-4XL':'other', '1':'other',

         'S/M(child)':'other', '2pcs':'other', 'XXXL':'other',

         'S..':'S', '30 cm':'L', '5XL':'other', '33':'S',

         'Size M':'M', '100 x 100cm(39.3 x 39.3inch)':'other',

         '100pcs':'other', '2XL':'XXL', '4XL':'other',

         'SizeL':'L', 'SIZE XXS':'XXL', 'XXXXL':'other',

         'Base & Top & Matte Top Coat':'other','size S':'S',

         '35':'S', '34':'S', 'SIZE-XXS':'XXS', 'S(bust 88cm)':'S',

         'S (waist58-62cm)':'S', 'S(Pink & Black)':'S', '20pcs':'other', 'US-S':'S',

         'Size -XXS':'XXS', 'X   L':'XL', 'White':'other',

         '25':'other', 'Size-S':'S', 'Round':'other',

         'Pack of 1':'other', '1 pc.':'other', 'S Diameter 30cm':'S', '6XL':'other',

         'AU plug Low quality':'other', '5PAIRS':'other',

         '25-S':'S', 'Size/S':'S', 'S Pink':'S',

         'Size-5XL':'other', 'daughter 24M':'other', '2':'other',

         'Baby Float Boat':'other', '10 ml':'other', '60':'other',

         'Size-L':'L', 'US5.5-EU35':'S', '10pcs':'other',

         '17':'other', 'Size-XXS':'XXS', 'Women Size 37':'M',

         '3 layered anklet':'other', '4-5 Years':'other',

         'Size4XL':'other', 'first  generation':'other',

         '80 X 200 CM':'other', 'EU39(US8)':'L', 'L.':'L',

         'Base Coat':'other', '36':'M', '04-3XL':'other',

         'pants-S':'S', 'Floating Chair for Kid':'other',

         '20PCS-10PAIRS':'other', 'B':'other',

         'Size--S':'S', '5':'other', '1 PC - XL':'XL',

         'H01':'other', '40 cm':'other', 'SIZE S':'S'

}
prod_df['product_variation_size_id'] = prod_df['product_variation_size_id'].map(product_size_convert_dict)

prod_df["product_variation_size_id"] = prod_df["product_variation_size_id"].fillna('other')

prod_df['product_variation_size_id'].unique()



tmp_df = prod_df['product_variation_size_id'].value_counts()

tmp_df = tmp_df.reset_index()

tmp_df['percentage'] = tmp_df['product_variation_size_id'] / prod_df.shape[0]



tmp_df
rating_cols = [

    'rating_five_count',

    'rating_four_count',

    'rating_three_count',

    'rating_two_count',

    'rating_one_count'

]



for col in rating_cols:

    prod_df[col] = prod_df[col].fillna(0)
prod_df['discount_amt'] = prod_df['retail_price'] - prod_df['price']



prod_df['discount_percentage'] = (prod_df['retail_price'] - prod_df['price'])/prod_df['retail_price']*100



prod_df['price_range'] = pd.cut(prod_df['price'], bins=np.arange(0, 60, 10), right=False)



prod_df['discount_amt_range'] = pd.cut(prod_df['discount_amt'], bins=np.arange(-10, 260, 10), right=False)



prod_df['discount_prct_range'] = pd.cut(prod_df['discount_percentage'], bins=np.arange(-20, 110, 10), right=False)



# create rating_percentage columns which indicates rating_x_count / rating_counts

prod_df['five_rating_prct'] = prod_df['rating_five_count']/prod_df['rating_count']

prod_df['four_rating_prct'] = prod_df['rating_four_count']/prod_df['rating_count']

prod_df['three_rating_prct'] = prod_df['rating_three_count']/prod_df['rating_count']

prod_df['two_rating_prct'] = prod_df['rating_two_count']/prod_df['rating_count']

prod_df['one_rating_prct'] = prod_df['rating_one_count']/prod_df['rating_count']
rating_prct_cols = [

    'five_rating_prct',

    'four_rating_prct',

    'three_rating_prct',

    'two_rating_prct',

    'one_rating_prct'

]



for col in rating_prct_cols:

    prod_df[col] = prod_df[col].fillna(0)
prod_df["has_urgency_banner"].unique()
prod_df["has_urgency_banner"] = prod_df["has_urgency_banner"].fillna(0)
trgt_columns = ['product_color',

                'units_sold',

]



tmp_df = prod_df[trgt_columns]

tmp_df = tmp_df.groupby('product_color').mean()

tmp_df = tmp_df.sort_values(by='units_sold', ascending=False)

tmp_df = tmp_df.reset_index()



colors = ['#f68741', '#a3acb1', '#491d88', '#b48464', 

          '#151c15','#d5dadd','#43b5a0','#1b96f3',

          '#ac0e28','#446b04','#f2b0a5','#fcdf87'

]



plt.subplots(figsize=(10,5))

plt.bar(tmp_df['product_color'], tmp_df['units_sold'], color=colors)



plt.title('Average sales by product color', fontsize=15)

plt.xlabel('Product color', fontsize=12)

plt.ylabel('Sales', fontsize=12)

plt.show()
trgt_columns = ['product_color',

                'units_sold',

]



tmp_df = prod_df[trgt_columns]

tmp_df = tmp_df.groupby('product_color').count()

tmp_df = tmp_df.sort_values(by='units_sold', ascending=False)

tmp_df = tmp_df.reset_index()

tmp_df = tmp_df.rename(columns={'units_sold':'count'})



colors = ['#151c15','#d5dadd', '#1b96f3','#43b5a0',

           '#ac0e28','#b48464','#f2b0a5','#fcdf87',

           '#a3acb1','#491d88', '#446b04','#f68741'

]



plt.subplots(figsize=(10,5))

plt.bar(tmp_df['product_color'], tmp_df['count'], color=colors)



plt.title('Number of product by color', fontsize=15)

plt.xlabel('Product color', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.show()
trgt_columns = ['product_variation_size_id',

                'units_sold',

]



tmp_df = prod_df[trgt_columns]

tmp_df = tmp_df.groupby('product_variation_size_id').mean()

tmp_df = tmp_df.sort_values(by='units_sold', ascending=False)

tmp_df = tmp_df.reset_index()





fig = px.bar(tmp_df, x='product_variation_size_id', y='units_sold', title='Average sales by product size')

fig.show()
trgt_columns = ['product_variation_size_id',

                'units_sold',

]



tmp_df = prod_df[trgt_columns]

tmp_df = tmp_df.groupby('product_variation_size_id').count()

tmp_df = tmp_df.sort_values(by='units_sold', ascending=False)

tmp_df = tmp_df.reset_index()

tmp_df = tmp_df.rename(columns={'units_sold':'count'})







fig = px.bar(tmp_df, x='product_variation_size_id', y='count', title='Number of product by product size')

fig.show()
def plot_multiple_hist(df):

    rcParams['figure.figsize'] = 10, 5

    df.hist(bins=20)

    plt.tight_layout()

    plt.show()



target_columns = ['price', 'retail_price']

target_df = prod_df[target_columns]



plot_multiple_hist(target_df)
def plot_hist_together(df,label1, label2, bins):

    plt.hist([df[label1], df[label2]], bins, label=[label1, label2])

    title = label1 + ' vs. ' + label2

    plt.title(title, fontsize=15)

    plt.legend()

    plt.show()



bins = np.linspace(0, 250, 40)

plot_hist_together(prod_df, 'price', 'retail_price', bins)
def count_plot(df, label, color, figsize=(10,5), rotation=0):

    plt.figure(figsize=figsize)

    df[label].value_counts().plot(kind='bar', color=color)

    plt.xlabel(label)

    plt.ylabel('count')

    

    title = label + ' distribution'

    plt.title(title, fontsize=15)

    plt.xticks(rotation=rotation)

    plt.show()    



count_plot(prod_df, 'origin_country', '#ff6d69')
figsize=(10,5)

sns.distplot(prod_df['rating'])

plt.title('rating distribution', fontsize=15)

plt.show()
fig = px.scatter(prod_df, x='retail_price', y='rating')

fig.update_layout(title_text="Relationship between retail_price and rating")

fig.show()
trgt_columns = ['uses_ad_boosts',

                'units_sold',

                'retail_price',

                'rating']



tmp_df = prod_df[trgt_columns]

tmp_df.groupby('uses_ad_boosts').mean()
plt.subplots(figsize=(10,5))

sns.countplot(x='discount_prct_range', data=prod_df, order=sorted(prod_df['discount_prct_range'].unique()),palette= ["#013766"])

plt.title('Number of data by discount_percentage_range', fontsize=15)

plt.show()
plt.subplots(figsize=(10,5))

sns.countplot(x='discount_amt_range', data=prod_df, order=sorted(prod_df['discount_amt_range'].unique()),palette= ["#ff6d69"])

plt.xticks(rotation=60)

plt.title('Number of data by discount_amount_range', fontsize=15)

plt.show()
trgt_columns = ['discount_amt_range',

                'units_sold'

               ]



tmp_df = prod_df[trgt_columns]

tmp_df = tmp_df.groupby('discount_amt_range').mean()

tmp_df = tmp_df.reset_index()



plt.subplots(figsize=(10,5))

sns.barplot(x='discount_amt_range', y='units_sold', data=tmp_df, palette='ocean_r')

plt.xticks(rotation=90)

plt.title('units_sold by discount_amount_range', fontsize=15)

plt.show()
plt.subplots(figsize=(20,12))



prod_df_corr = prod_df.corr()



sns.heatmap(prod_df_corr, cmap="YlGnBu", annot=True)

plt.show()
def get_pairs(df, trgt_col):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        pairs_to_drop.add((trgt_col, cols[i]))

    return pairs_to_drop



def get_strong_correlations(df, n=10):

    au_corr = df.corr().unstack()

    labels_to_drop = get_pairs(df, 'units_sold')

    au_corr_desc = au_corr['units_sold'].sort_values(ascending=False)

    au_corr_asc = au_corr['units_sold'].sort_values(ascending=True)

    return au_corr_desc[0:n], au_corr_asc[0:n]



au_corr_desc, au_corr_asc = get_strong_correlations(prod_df, 15)



print("Strong Correlations")

print("Positive Correlations")

print(au_corr_desc)

print(au_corr_desc.index)

print('-----------------------------')

print("Negative Correlations")

print(au_corr_asc)
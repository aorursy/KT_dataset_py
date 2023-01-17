import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pd.options.mode.chained_assignment = None
#Read in the dataset
trans = pd.read_csv('/kaggle/input/quantium-data-analytics-virtual-experience-program/Transactions.csv')
#Check first 5 rows
trans.head()
#Change column headers to lower cases and rename columns
trans.columns = map(str.lower, trans.columns)
trans = trans.rename(columns={'lylty_card_nbr':'card_nbr'})
#Check general info
trans.info()
#Convert Excel dates to dates
import datetime
import xlrd
date_tuple = [xlrd.xldate.xldate_as_datetime(x, 0)
              for x in trans['date'].tolist()]
trans['date'] = [d.strftime('%m-%d-%Y') for d in date_tuple]
trans['date'] = pd.to_datetime(trans['date'])
#Change id variables to categorical values
obj_type = ['store_nbr','card_nbr','txn_id','prod_nbr']
trans[obj_type] = trans[obj_type].astype(str)

#Create price column
trans['prod_price'] = trans['tot_sales']/trans['prod_qty']
#Numerical data
trans.describe()
#Draw boxplots
trans.plot.box(y=['prod_qty','prod_price','tot_sales'],
               figsize=(10,6))
plt.title('Boxplots of product quantity, price, and total sales', fontsize=16)
#Check which rows have product quantity of 100 and above
trans[trans['prod_qty'] > 100][['card_nbr','prod_qty','tot_sales']]
#Doulbe ckeck these rows before removing them
trans[trans['card_nbr'] == '226000']
#Keep rows that have product quantity that is less than 100
trans = trans[trans['prod_qty'] < 100]
#Check maximum values of prod_qty and tot_sales
trans[['prod_qty','tot_sales']].max()
#Check datetime values
trans.describe(include=np.datetime64)
#Create list of date from 07/01/2018 to 06/30/2019
date_range = pd.DataFrame(pd.date_range('2018-07-01', '2019-06-30').tolist(),
                          columns=['date'])

#Merge this list to column date, showing indicator
date_merge = pd.merge(trans['date'],
                      date_range,
                      on='date',
                      how='outer',
                      indicator=True)

#Row that has indicator (_merge column) as right_only contains the missing date
date_merge[date_merge['_merge'] == 'right_only']
#Create new columns for year, month, and weekday
trans['year'], trans['month'], trans['weekday'] = \
trans['date'].dt.year, trans['date'].dt.month, trans['date'].dt.day_name()
#Check categorical values
trans.describe(include=object)
#Drop duplicates if available
trans = trans.drop_duplicates(ignore_index=True)

#Print data length
print(f'There are {len(trans)} unique data rows.')
#Check how many rows that have the same information of date, store_nbr, card_nbr, and txn_id
print(f"There are {trans.duplicated(['date','store_nbr','card_nbr','txn_id']).sum()} \
rows that have the same information of date, store number, customer, and transaction id.")
#Create dataframe including date, store_nbr, card_nbr, txn_id, and prod_nbr that have same txn_id
dup_trans = trans[trans.duplicated('txn_id', keep=False)][['date','store_nbr','card_nbr','txn_id','prod_nbr']]

#Group this dataframe by txn_id, counting number of unique values of other columns
count_df = dup_trans.groupby(['txn_id']).nunique()
#Print out data that have more than one unique value of date, store_nbr, or card_nbr
count_df[(count_df['date'] != 1) |
         (count_df['store_nbr'] != 1) |
         (count_df['card_nbr'] != 1)]
#Print out data of these transaction id from transaction data
trans[(trans['txn_id'] == '155468') | (trans['txn_id'] == '155469') |
      (trans['txn_id'] == '156002') | (trans['txn_id'] == '50042')].sort_values(['date','card_nbr'])
#Check if there're duplicated transactions and product id
(count_df['prod_nbr'] == 1).any()
#Remove potential duplicated transactions
trans = trans[(trans['txn_id'] != '155468') & (trans['txn_id'] != '155469') &
              (trans['txn_id'] != '156002') & (trans['txn_id'] != '50042')]
#Extract size of the product
trans['prod_size'] = trans['prod_name'].str.extract(r'(\d+)').astype(np.int64)

#Extract size unit of the product
trans['prod_unit'] = trans['prod_name'].str.extract(r'(\D+$)')
#Check statistic of size
trans['prod_size'].describe()
print(f"Number of unique package sizes: {trans['prod_size'].nunique()}")
print(f"Unique values of size units:\n{trans['prod_unit'].value_counts()}")
#Print out data that have unit 'g Swt Pot Sea Salt'
trans[trans['prod_unit'] == 'g Swt Pot Sea Salt']['prod_name'].sample(5)
#Drop prod_unit
trans = trans.drop('prod_unit', axis=1)

#Remove package size and unit out of the product name
trans['prod_name'] = trans['prod_name'].str.replace(r'\d+.', '').str.rstrip(' ')
print('Unique product name:')
np.sort(trans['prod_name'].unique()).tolist()
#Create brand name dictionary
brand_dict = {'Burger':'Burger Rings', 'CCs':'CCs', 'Cheetos':'Cheetos',
              'Cheezels':'Cheezels','Cobs':'Cobs', 'Dorito':'Doritos', 
              'Doritos':'Doritos', 'French':'French Fries','Infuzions':'Infuzions',
              'Infzns':'Infuzions', 'Kettle':'Kettle', 'Natural':'Natural Chip Co',
              'NCC':'Natural Chip Co', 'Old':'Old El Paso', 'Pringles':'Pringles', 
              'RRD':'Red Rock Deli','Red':'Red Rock Deli', 'Smith':'Smiths', 
              'Smiths':'Smiths','Grain':'Sunbites', 'GrnWves':'Sunbites',
              'Snbts':'Sunbites', 'Sunbites':'Sunbites', 'Thins':'Thins',
              'Tostitos':'Tostitos', 'Twisties':'Twisties', 'Tyrrells':'Tyrrells', 
              'Woolworths':'Woolworths', 'WW':'Woolworths'}
#Create brand column
trans['brand'] = trans['prod_name'].str.extract(r'(^\w+)')
trans['brand'] = trans['brand'].map(brand_dict)
#Print out unique brand name
print(f"Unique brand names: \n{trans['brand'].unique()}")
#Remove spaces in prod_name column
trans['prod_name'] = trans['prod_name'].apply(lambda x: ' '.join(x.split())) 
#Display full data
pd.set_option('display.max_colwidth', None)

#Group product names by brands
trans[['brand', 'prod_name']].groupby('brand').agg(lambda x: '; '.join(set(x))).reset_index()
#Create product name dictionary
prod_name_dict = {'Cheetos Chs & Bacon Balls':'Cheetos Cheese & Bacon Balls',
                  'Cobs Popd Swt/Chlli &Sr/Cream Chips':'Cobs Popd Sweet Chili & Sour Cream Chips',
                  'Cobs Popd Sour Crm &Chives Chips':'Cobs Popd Sour Cream & Chives Chips',
                  'Dorito Corn Chp Supreme':'Doritos Corn Chips Supreme',
                  'Doritos Corn Chip Mexican Jalapeno':'Doritos Corn Chips Mexican Jalapeno',
                  'Doritos Corn Chip Southern Chicken':'Doritos Corn Chips Southern Chicken',
                  'Infuzions SourCream&Herbs Veg Strws':'Infuzions Sour Cream & Herb Veggie Straws',
                  'Infuzions Mango Chutny Papadums':'Infuzions Mango Chutney Papadams',
                  'Infuzions Thai SweetChili PotatoMix':'Infuzions Thai Sweet Chili Potato Mix',
                  'Infzns Crn Crnchers Tangy Gcamole':'Infuzions Corn Crunchers Tangy Guacamole',
                  'Kettle Tortilla ChpsBtroot&Ricotta':'Kettle Tortilla Chips Beetroot & Ricotta',
                  'Kettle Tortilla ChpsFeta&Garlic':'Kettle Tortilla Chips Feta & Garlic',
                  'Kettle Tortilla ChpsHny&Jlpno Chili':'Kettle Tortilla Chips Honey & Jalapeno Chili',
                  'Kettle Sensations BBQ&Maple':'Kettle Sensations BBQ Maple',
                  'Kettle Swt Pot Sea Salt':'Kettle Sweet Pot Sea Salt',
                  'Natural Chip Compny SeaSalt':'Natural Chip Co Sea Salt',
                  'Natural ChipCo Hony Soy Chckn':'Natural Chip Co Honey Soy Chicken',
                  'NCC Sour Cream & Garden Chives':'Natural Chip Co Sour Cream & Garden Chives',
                  'Natural ChipCo Sea Salt & Vinegr':'Natural Chip Co Sea Salt & Vinegar',
                  'Natural Chip Co Tmato Hrb&Spce':'Natural Chip Co Tomato Herbs & Spices',
                  'Pringles Sthrn FriedChicken':'Pringles Southern Fried Chicken',
                  'Pringles Barbeque':'Pringles BBQ',
                  'Pringles SourCream Onion':'Pringles Sour Cream Onion',
                  'Pringles Chicken Salt Crips':'Pringles Chicken Salt Chips',
                  'Pringles Slt Vingar':'Pringles Salt Vinegar',
                  'Pringles Sweet&Spcy BBQ':'Pringles Sweet & Spicy BBQ',
                  'RRD Honey Soy Chicken':'Red Rock Deli Honey Soy Chicken',
                  'Red Rock Deli SR Salsa & Mzzrlla':'Red Rock Deli Salsa & Mozzarella',
                  'RRD Lime & Pepper':'Red Rock Deli Lime & Pepper',
                  'Red Rock Deli Chikn&Garlic Aioli':'Red Rock Deli Chicken & Garlic Aioli',
                  'Red Rock Deli Sp Salt & Truffle':'Red Rock Deli Sea Salt & Truffle',
                  'RRD Salt & Vinegar':'Red Rock Deli Sea Salt & Vinegar',
                  'RRD Sweet Chilli & Sour Cream':'Red Rock Deli Sweet Chilli & Sour Cream',
                  'RRD Pc Sea Salt':'Red Rock Deli Sea Salt',
                  'Red Rock Deli Thai Chilli&Lime':'Red Rock Deli Thai Chilli & Lime',
                  'RRD SR Slow Rst Pork Belly':'Red Rock Deli Slow Roast Pork Belly',
                  'RRD Steak & Chimuchurri':'Red Rock Deli Steak & Chimuchurri',
                  'RRD Chilli& Coconut':'Red Rock Deli Chilli & Coconut',
                  'Smith Crinkle Cut Bolognese':'Smiths Crinkle Cut Bolognese',
                  'Smiths Crinkle Cut French OnionDip':'Smiths Crinkle Cut French Onion Dip',
                  'Smiths Crinkle Cut Chips Barbecue':'Smiths Crinkle Cut Chips BBQ',
                  'Smiths Thinly Swt Chli&S/Cream':'Smiths Thinly Sweet Chili & Sour Cream',
                  'Smiths Crinkle Cut Chips Chs&Onion':'Smiths Crinkle Cut Chips Cheese & Onion',
                  'Smiths Crnkle Chip Orgnl Big Bag':'Smiths Crinkle Chips Original Big Bag',
                  'Smiths Chip Thinly CutSalt/Vinegr':'Smiths Chips Thinly Cut Salt Vinegar',
                  'Smiths Crinkle Cut Snag&Sauce':'Smiths Crinkle Cut Snag & Sauce',
                  'Smith Crinkle Cut Mac N Cheese':'Smiths Crinkle Cut Mac N Cheese',
                  'Smiths Chip Thinly S/Cream&Onion':'Smiths Chips Thinly Sour Cream & Onion',
                  'Smiths Chip Thinly Cut Original':'Smiths Chips Thinly Cut Original',
                  'Snbts Whlgrn Crisps Cheddr&Mstrd':'Sunbites Wholegrain Crisps Cheddar & Mustard',
                  'Sunbites Whlegrn Crisps Frch/Onin':'Sunbites Wholegrain Crisps French Onion',
                  'GrnWves Plus Btroot & Chilli Jam':'Sunbites Grain Waves Plus Beetroot & Chilli Jam',
                  'Grain Waves Sweet Chilli':'Sunbites Grain Waves Sweet Chilli',
                  'Grain Waves Sour Cream&Chives':'Sunbites Grain Waves Sour Cream & Chives',
                  'Thins Chips Light& Tangy':'Thins Chips Light & Tangy',
                  'Thins Chips Originl saltd':'Thins Chips Originl Salted',
                  'Thins Chips Seasonedchicken':'Thins Chips Seasoned Chicken',
                  'Tyrrells Crisps Ched & Chives':'Tyrrells Crisps Cheddar & Chives',
                  'WW Sour Cream &OnionStacked Chip':'Woolworths Sour Cream & Onion Stacked Chips',
                  'WW Crinkle Cut Original':'Woolworths Crinkle Cut Original',
                  'WW Crinkle Cut Chicken':'Woolworths Crinkle Cut Chicken',
                  'WW Original Stacked Chips':'Woolworths Original Stacked Chips',
                  'WW Original Corn Chips':'Woolworths Original Corn Chips',
                  'WW D/Style Chip Sea Salt':'Woolworths Deli Style Chips Sea Salt',
                  'WW Supreme Cheese Corn Chips':'Woolworths Supreme Cheese Corn Chips'}
#Create another column of correct product names and replace Nan values in this column with old values
trans['prod_name_correct'] = trans['prod_name'].map(prod_name_dict)
trans['prod_name_correct'] = np.where(trans['prod_name_correct'].isna(),
                                      trans['prod_name'], trans['prod_name_correct'])

#Replace prod_name with the correct one and remove one of them
trans['prod_name'] = trans['prod_name_correct']
trans = trans.drop(['prod_name_correct'],axis=1)
#Remove Old El Paso brand
trans = trans[(trans['brand'] != 'Old El Paso') &
              (trans['prod_name'] != 'Tostitos Splash Of Lime')]

#Print out product names that have "salsa"
print(f"Product names that have 'salsa':\n{trans[trans['prod_name'].str.contains('Salsa', regex=False)]['prod_name'].unique()}")
#Select data that have non-salsa products
trans = trans[(trans['prod_name'] != 'Doritos Salsa Medium') &
              (trans['prod_name'] != 'Woolworths Mild Salsa') &
              (trans['prod_name'] != 'Woolworths Medium Salsa') &
              (trans['prod_name'] != 'Doritos Salsa Mild')]
#Print out data sample
trans.sample(5)
#Create lists of numerical variables and their names
num_col = ['tot_sales', 'prod_price', 'prod_size', 'prod_qty']
num_name = ['Total sales', 'Product price', 'Product size', 'Product quantity']

#Loop through 4 graphs
fig, axes = plt.subplots(2,2, figsize=(10,8))
for i,ax in enumerate(axes.flatten()):
    trans[num_col[i]].plot.hist(title=(num_name[i] + ' distribution'),
                                ax=ax)
    ax.set_ylabel(None)   
plt.tight_layout()
#Draw four agrregated graphs separately
fig, axes = plt.subplots(2,2, figsize=(10,8))
trans.pivot_table(index='txn_id',
                  values='tot_sales',
                  aggfunc='sum').plot.hist(ax=axes[0,0],
                                           title='Total Sales per transaction Distribution',
                                           legend=None)
trans.pivot_table(index='txn_id',
                  values='prod_price',
                  aggfunc='mean').plot.hist(ax=axes[0,1],
                                            title='Average Product Price per transaction Distribution',
                                            legend=None)
trans.pivot_table(index='txn_id',
                  values='prod_size',
                  aggfunc='mean').plot.hist(ax=axes[1,0],
                                            title='Average Product Size per transaction Distribution',
                                            legend=None)
trans.pivot_table(index='txn_id',
                  values='prod_qty',
                  aggfunc='sum').plot.hist(ax=axes[1,1],
                                           title='Total Product Quantities per transaction Distribution',
                                           legend=None)
axes[0,0].set_ylabel(None)
axes[0,1].set_ylabel(None)
axes[1,0].set_ylabel(None)
axes[1,1].set_ylabel(None)
plt.tight_layout()
fig, axes = plt.subplots(2,1, figsize=(12,6))

#Total sales by date
trans.pivot_table(index='date',
                  values='tot_sales',
                  aggfunc='sum').plot(ax=axes[0],
                                      title='Total Sales',
                                      legend=None,
                                      color='cornflowerblue')

#Total transactions by date
trans.pivot_table(index='date',
                  values='txn_id',
                  aggfunc=lambda x: x.nunique()).plot(ax=axes[1],
                                                      title='Total Transactions',
                                                      legend=None,
                                                      color='salmon',
                                                      sharex=True)

plt.xlabel('Period')
#Draw a count plot of total transactions by weekdays
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                 'Friday', 'Saturday', 'Sunday']

trans_by_weekday = trans.pivot_table(index='weekday',
                                     values='txn_id',
                                     aggfunc=lambda x: x.nunique())

trans_by_weekday.loc[weekday_order].plot.bar(figsize=(10,5),
                                             color='salmon',
                                             legend=None)

plt.title('Total transactions by Weekday', fontsize=16)
plt.xlabel('Weekday')
plt.ylabel('Total Transactions')
#Aggregate total sales by store number and time
sales_by_store = trans.pivot_table(index=['year', 'month'], columns=['store_nbr'],
                                   values='tot_sales', aggfunc='sum')
#Draw graphs of top five stores over time
period = sales_by_store.index.tolist()
fig, axes = plt.subplots(4, 3, figsize=(12,8), sharex=True)
fig.suptitle('Top five stores with highest total sales from July 2018 to June 2019', fontsize=16)
fig.text(0.5, -0.01, 'Total Sales', ha='center')
fig.text(-0.01, 0.5, 'Store Number', va='center', rotation='vertical')

for i, ax in enumerate(axes.flatten()):
    sales_by_store.iloc[i].nlargest(5).sort_values().plot.barh( \
        title=(str(period[i][0]) + '-' + str(period[i][1])),
        color='cornflowerblue', ax=ax)
    ax.set_ylabel(None)
       
plt.tight_layout(pad=3)
#Draw heatmap of total sales by brand and time
sales_by_brand = trans.pivot_table(columns=['year', 'month'],
                                   index='brand',
                                   values='tot_sales',
                                   aggfunc='sum')

fig, ax = plt.subplots(figsize=(12,8))
sns.heatmap(sales_by_brand,
            cmap='Greys',
            linewidths=.5,
            ax=ax)

plt.title('Compare total sales among brands', fontsize=16)
plt.xlabel('Period')
plt.ylabel('Brand')
plt.tight_layout()
#Loop over four brands
for i in ['Kettle', 'Smiths', 'Doritos', 'Pringles']:
    print('Most popular flavour of ' + str(i) +
          f": \t{trans[trans['brand'] == i]['prod_name'].value_counts().idxmax()}")
from wordcloud import WordCloud, STOPWORDS
text = ' '.join(trans['prod_name'])
stopwords = list(STOPWORDS) + ' '.join(trans['brand'].unique()).split() + ['Chips']
wc = WordCloud(stopwords=stopwords,
               collocations=False,
               background_color='white',
               random_state=0).generate(text=text)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
#Read in the dataset
customer = pd.read_csv('/kaggle/input/quantium-data-analytics-virtual-experience-program/PurchaseBehaviour.csv')
#View the first 5 rows
customer.head()
#Change column headers to lower cases and rename columns
customer.columns = map(str.lower, customer.columns)
customer = customer.rename(columns={'lylty_card_nbr':'card_nbr','premium_customer':'customer_type'})
#Check general info
customer.info()
#Change card_nbr to category
customer['card_nbr'] = customer['card_nbr'].astype(str)
#Check other info
customer.describe()
print('Customer lifestage:')
customer['lifestage'].unique().tolist()
print('Customer type:')
customer['customer_type'].unique().tolist()
print('Number of customers by segments:')
customer['customer_type'].value_counts()
#Count plot of number of customers in each segment
order = ['Budget', 'Mainstream', 'Premium']
customer['customer_type'].value_counts().loc[order].plot.bar(figsize=(8,5),
                                                             color='mediumseagreen')
plt.title('Number of customers by segment', fontsize=16)
plt.xlabel('Customer Type')
plt.ylabel('Number of customers')
col_order = ['YOUNG SINGLES/COUPLES', 'MIDAGE SINGLES/COUPLES', 'OLDER SINGLES/COUPLES',
             'NEW FAMILIES', 'YOUNG FAMILIES', 'OLDER FAMILIES', 'RETIREES']

#Create a function to make stacked bar plot with percentage
def draw_bar(data):
    #Count number of unique customers by segment and lifestage
    pivot_df = data.pivot_table(index='customer_type',
                                values='card_nbr',
                                columns='lifestage',
                                aggfunc=lambda x: x.nunique()).reindex(columns=col_order)
    
    #Convert to percentage
    percentage_dist = 100*pivot_df.divide(pivot_df.sum(axis=1), axis=0)
    #Stacked bar plot
    ax = percentage_dist.plot(kind='barh',
                              stacked=True,
                              figsize=(13,8),
                              colormap='Set2')

    #Insert percentage labels to the graph
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x + width / 2,
                y + height / 2,
                '{:.2f}'.format(width),
                horizontalalignment='center',
                verticalalignment='center')
        
    #Complete the graph
    plt.legend(bbox_to_anchor=(1,0.5), loc='center left')
    plt.title('Customer lifestage proportion by Customer type', fontsize=16)
    plt.xlabel('Percentage of each lifestage')
    plt.ylabel('Customer Type')
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
#Apply draw_bar function on customer data
draw_bar(customer)
#Check number of records
print(f'There are {len(trans)} rows in transaction dataset.')
print(f'There are {len(customer)} rows in customer dataset.')
#Merge transaction data and customer data
df = pd.merge(customer, trans, on='card_nbr', how='right')
#Check general info
df.info()
#Check number of customers
print(f"There are {df['card_nbr'].nunique()} customers.")
#Aggregate total sales by segment and lifestage
sales_by_segment = df.pivot_table(index='customer_type',
                                  columns='lifestage',
                                  values='tot_sales',
                                  aggfunc='sum').reindex(columns=col_order)
#Total sales by segment
sales_by_segment.sum(axis=1).plot.bar(figsize=(8,5),
                                      color='cornflowerblue',
                                      legend=None)

plt.title('Total Sales by Segment', fontsize=16)
plt.xlabel(None)
#Apply draw_bar function on the merged data
draw_bar(df)
#Total sales by segment and lifestage
sales_by_segment.plot(kind='bar',
                      figsize=(13,8),
                      colormap='Set2')

plt.legend(bbox_to_anchor=(1,0.5), loc='center left')
plt.title('Total Sales by Segment and Lifestage', fontsize=16)
plt.xlabel(None)
#Separate pointplots of average product quantity, size, and price per
fig, axes = plt.subplots(3,1, figsize=(10,15))
sns.pointplot(ax=axes[0],
              x='customer_type',
              y='prod_qty',
              hue='lifestage',
              hue_order=col_order,
              order=order,
              palette='Set2',
              data=df)
axes[0].legend(bbox_to_anchor=(1,1), loc='upper left')
axes[0].set_title('Average Product Quantity by Segment and Lifestage')
axes[0].set_ylabel('Average Product Quantity')
axes[0].set_xlabel(None)

sns.pointplot(ax=axes[1],
              x='customer_type',
              y='prod_size',
              hue='lifestage',
              hue_order=col_order,
              order=order,
              palette='Set2',
              data=df)
axes[1].get_legend().remove()
axes[1].set_title('Average Product Size by Segment and Lifestage')
axes[1].set_ylabel('Average Product Size')
axes[1].set_xlabel(None)

sns.pointplot(ax=axes[2],
              x='customer_type',
              y='prod_price',
              hue='lifestage',
              hue_order=col_order,
              order=order,
              palette='Set2',
              data=df)
axes[2].get_legend().remove()
axes[2].set_title('Average Product Price by Segment and Lifestage')
axes[2].set_ylabel('Average Product Price')
axes[2].set_xlabel(None)
plt.tight_layout()
#Apriori algorithm and association rule for basket analysis
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
#Extract target data
data = df[(df['customer_type'] == 'Mainstream') &
          (df['lifestage'] == 'YOUNG SINGLES/COUPLES')][['txn_id', 'brand', 'prod_qty']].reset_index(drop=True)
#Convert to dummies
basket = data.groupby(['txn_id', 'brand'])['prod_qty'].sum().unstack().fillna(0)
#Make sure there are only ones and zeroes
basket[basket > 1] = 1
basket.head()
#Get the frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
frequent_itemsets
#Create a function to run Apriori with support and lift metrics
def basket(col, segment, lifestage, support, metric, mmin):
    #Extract target data
    data = df[(df['customer_type'] == segment) &
              (df['lifestage'] == lifestage)][['txn_id', col, 'prod_qty']].reset_index(drop=True)
    
    #Convert to dummies
    basket = data.groupby(['txn_id', col])['prod_qty'].sum().unstack().fillna(0)
    basket[basket > 1] = 1 #Make sure there are only ones and zeroes
    
    #Get the frequent itemsets
    frequent_itemsets = apriori(basket, min_support=support, use_colnames=True)
    
    #Print out result and table of combination if available
    print(f"There are {len(basket[basket.sum(axis=1) > 1])} (out of {data['txn_id'].nunique()}) transactions")
    print(f'that have more than one '+ col + '\n')
    print(f'There are maximum {basket.sum(axis=1).max()} ' + col + 's per transaction\n')
    print(f'There are {len(frequent_itemsets)} ' + col + 's that are likely to be bought')
    print('in at least ' + str(support*100) + '% of total transactions')
    if len(frequent_itemsets) == 0:
        pass
    else:
        print(f"\n{frequent_itemsets.sort_values(['support'], ascending=False).reset_index(drop=True)}\n")
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=mmin)
        if len(rules) == 0:
            print('But these ' + col + 's are not likely to be bought together')
        else:
            print(col + f's that are likely to be bought together\n{rules}')
#Print out number of unique transactions, brands, and product names in the target dataset
print('Number of unique transactions, brands, and product names:')
df[(df['customer_type'] == 'Mainstream') &
   (df['lifestage'] == 'YOUNG SINGLES/COUPLES')][['txn_id', 'brand', 'prod_name']].nunique()
#Apply basket function on brands
basket('brand', 'Mainstream', 'YOUNG SINGLES/COUPLES', 0.05, 'lift', 0.7)
#Apply basket function on product names
basket('prod_name', 'Mainstream', 'YOUNG SINGLES/COUPLES', 0.01, 'lift', 0.7)
#Create a function to run Apriori and compare results between target and other groups
def compare(col, segment, lifestage, mmin):    
    #Extract data
    target_df = df[(df['customer_type'] == segment) &
                   (df['lifestage'] == lifestage)][['txn_id', col, 'prod_qty']].reset_index(drop=True)
    others_df = df[(df['customer_type'] != segment) &
                   (df['lifestage'] != lifestage)][['txn_id', col, 'prod_qty']].reset_index(drop=True)
    
    #Convert values into dummies
    target_basket = target_df.groupby(['txn_id', col])['prod_qty'].sum().unstack().fillna(0)
    others_basket = others_df.groupby(['txn_id', col])['prod_qty'].sum().unstack().fillna(0)
    
    #Make sure there are only ones and zeroes
    target_basket[target_basket > 1] = 1
    others_basket[others_basket > 1] = 1

    #Get the frequent itemsets
    target_itemsets = apriori(target_basket, min_support=mmin, use_colnames=True).set_index(['itemsets'])
    others_itemsets = apriori(others_basket, min_support=mmin, use_colnames=True).set_index(['itemsets'])
    
    #Join these two itemsets and return result
    compare = target_itemsets.join(others_itemsets, lsuffix='_target', rsuffix='_others', how='outer').reset_index()
    compare['affinity'] = compare['support_target']/compare['support_others']
    
    return compare.sort_values(['affinity'], ascending=False).reset_index(drop=True)
#Apply compare function on brands
compare('brand', 'Mainstream', 'YOUNG SINGLES/COUPLES', 0.001)
#Apply compare function on product sizes
compare('prod_size', 'Mainstream', 'YOUNG SINGLES/COUPLES', 0.001)
#Print out product names that have size of 270g or above
df[['prod_size', 'prod_name']].groupby('prod_size').\
agg(lambda x: '; '.join(set(x))).reset_index().nlargest(3, 'prod_size')
#Apply compare function on product names
compare('prod_name', 'Mainstream', 'YOUNG SINGLES/COUPLES', 0.01)
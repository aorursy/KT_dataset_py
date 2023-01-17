# imports and fill our 5 tables that instacart provided as dataframes into a dictionary called data

# note: only taking order_products__prior product purchases for this analysis



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline



files = ['aisles.csv','departments.csv', 'orders.csv','products.csv', 'order_products__prior.csv']

data = {}



for f in files:

    d = pd.read_csv('../input/{0}'.format(f))

    data[f.replace('.csv','')] = d

    

# rename each df for easier coding and readability

products= data['products']

order_products_prior = data['order_products__prior']

departments = data['departments']

orders = data['orders']

ailes = data['aisles']



OPPsample = order_products_prior.sample(n=3000000)



merged = products.merge(OPPsample,on='product_id',how='inner')

merged = departments.merge(merged,on='department_id',how='inner')

merged = orders.merge(merged,on='order_id',how='inner')



merged.head()
department_list = list(departments.department)



department_data = {}



for n in department_list:

    d = merged.loc[merged['department']=='{0}'.format(n)]

    department_data['{0}'.format(n)] = d

    

list(department_data)
department_product_data = {}



for n in department_list:

    d = department_data['{0}'.format(n)].groupby(['product_name']).count().reset_index()

    department_product_data['{0}'.format(n)] = d

    department_product_data['{0}'.format(n)] = department_product_data['{0}'.format(n)].iloc[:,0:2]

    department_product_data['{0}'.format(n)].columns = ['product_name','quantity']

    department_product_data['{0}'.format(n)] = department_product_data['{0}'.format(n)].sort_values('quantity',ascending=False)

    department_product_data['{0}'.format(n)].reset_index(inplace=True)

    department_product_data['{0}'.format(n)] = department_product_data['{0}'.format(n)].iloc[:,1:4]
# sanity check random table in department_product_data

department_product_data['babies'].head()
# define the columns we are interested in from our big merged table and make a new merged table with only those columns

time_columns = ['order_id','order_hour_of_day','department','product_name']



time_merged = pd.DataFrame(merged,columns=time_columns)

time_merged.head()
columns=['Order hour of day','Quantity'] #columns to rename to



department_time_data = {}



for n in department_list:

    d = time_merged.loc[time_merged['department']=='{0}'.format(n)] # Insert data from time_merged table into our new dictionary per each department

    department_time_data['{0}'.format(n)] = d

    department_time_data['{0}'.format(n)] = department_time_data['{0}'.format(n)].groupby('order_hour_of_day').count()

    department_time_data['{0}'.format(n)].reset_index(inplace=True)

    department_time_data['{0}'.format(n)] = department_time_data['{0}'.format(n)][department_time_data['{0}'.format(n)].columns[0:2]]

    department_time_data['{0}'.format(n)].columns = columns
# sanity check random table in department_time_data

department_time_data['alcohol']
orders.head()
pd.isnull(orders['days_since_prior_order']).sum()
# check unique values in that column

orders['days_since_prior_order'].unique()
orders = orders.dropna()
reorder_columns = ['order_id','product_id','days_since_prior_order','department']

reorder_merged = pd.DataFrame(merged,columns=reorder_columns)
columns2 = ['Days since prior order','Quantity'] # columns to rename to



department_reorder_data = {}



for n in department_list:

    d = reorder_merged.loc[reorder_merged['department']=='{0}'.format(n)]

    department_reorder_data['{0}'.format(n)] = d

    department_reorder_data['{0}'.format(n)] = department_reorder_data['{0}'.format(n)].groupby('days_since_prior_order').count()

    department_reorder_data['{0}'.format(n)].reset_index(inplace=True)

    department_reorder_data['{0}'.format(n)] = department_reorder_data['{0}'.format(n)][department_reorder_data['{0}'.format(n)].columns[0:2]]

    department_reorder_data['{0}'.format(n)].columns = columns2
# sanity check table in department_reorder_data

department_reorder_data['produce']
def toptenplot(name):

    p = sns.cubehelix_palette(10, start=0.6,dark=0.5, rot=1,light=0.8,reverse=True)

    plot = sns.barplot(palette = p,y='product_name',x='quantity',data=department_product_data['{0}'.format(name)].head(n=10))

    sns.plt.title('{0}'.format(name))

    plot.set(xlabel='Quantity',ylabel='Product Name')
toptenplot('produce')
toptenplot('dairy eggs')
toptenplot('alcohol')
for n in department_list:

    sns.pointplot(x='Order hour of day',y='Quantity',data=department_time_data['{0}'.format(n)],markers='',linestyles='-')
department_time_norm = {}



for n in department_list:

    #calculate normalized quantity

    q = department_time_data['{0}'.format(n)]['Quantity']

    q_norm = (q-q.mean())/(q.max()-q.min())

    

    #copy "department_time_data" to "department_time_norm"

    d = department_time_data['{0}'.format(n)]

    department_time_norm['{0}'.format(n)] = d

    

    #replace the quantity with our new normalized quantity "q_norm"

    department_time_norm['{0}'.format(n)]['Quantity']=q_norm
paper_rc = {'lines.linewidth': 1}                  

sns.set_context("paper", rc = paper_rc)

plt.figure(figsize=(12, 6))

plt.ylabel('Normalized Quantity')



for n in department_list: 

    sns.pointplot(x='Order hour of day',y='Quantity',data=department_time_norm['{0}'.format(n)],markers='',linestyles='-')



sns.pointplot(x='Order hour of day',y='Quantity',data=department_time_norm['alcohol'],markers='',linestyles='-',color = 'g')

sns.pointplot(x='Order hour of day',y='Quantity',data=department_time_norm['babies'],markers='',linestyles='-',color = 'r')
department_reorder_norm = {}



for n in department_list:

    #calculate normalized quantity

    q = department_reorder_data['{0}'.format(n)]['Quantity']

    q_norm = (q-q.mean())/(q.max()-q.min())

    

    #copy "department_data" to "department_data_norm"

    d = department_reorder_data['{0}'.format(n)]

    department_reorder_norm['{0}'.format(n)] = d

    

    #replace the quantity with our new normalized quantity "q_norm"

    department_reorder_norm['{0}'.format(n)]['Quantity']=q_norm
# check random department

department_reorder_norm['produce']
paper_rc = {'lines.linewidth': 1}                  

sns.set_context("paper", rc = paper_rc)

plt.figure(figsize=(12, 6))

plt.ylabel('Normalized Quantity')



for n in department_list:

    sns.pointplot(x='Days since prior order',y='Quantity',data=department_reorder_norm['{0}'.format(n)],markers='',linestyles='-')

    

sns.pointplot(x='Days since prior order',y='Quantity',data=department_reorder_norm['alcohol'],markers='',linestyles='-',color = 'r')

sns.pointplot(x='Days since prior order',y='Quantity',data=department_reorder_norm['babies'],markers='',linestyles='-',color = 'g')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
sns.set_style('whitegrid'); sns.set_palette('deep')
trans = pd.read_excel(r'../input/KPMG_data.xlsx', sheet_name = 'Transactions')
trans.head()
trans.columns
trans.dtypes
# check validation of year
trans.product_first_sold_date.dt.year.unique()
# check missing values
trans.isnull().sum() / trans.count() *100
trans = trans.dropna(subset=['brand'])
trans.isnull().sum()
# check uniqueness of transactions
len(trans.transaction_id.unique()) /len(trans)
trans.describe()
cusd = pd.read_excel(r'../input/KPMG_data.xlsx', sheet_name = 'CustomerDemographic')
cusd.head()
cusd.columns
cusd.dtypes
# delete unuse column
cusd = cusd.drop(columns = ['default'])
# check missing values
cusd.isnull().sum() / cusd.count() *100
cusd = cusd.dropna(subset=['DOB']).reset_index(drop=True)
len(cusd.customer_id.unique()) / len(cusd)
# multiple formatted of gender
cusd.gender.unique()
gender=[]
for i in cusd.gender:
    if i.startswith('F'):
        i = i.replace(i,'Female')
    elif i.startswith('M'):
        i = i.replace(i,'Male')
    else:
        i = i.replace(i,'Unknown')
    gender.append(i)
    
cusd['gender'] = pd.Series(gender)
# validity of DOB
cusd.DOB.sort_values()
cusd = cusd[cusd['DOB'].dt.to_period('Y').astype(str).astype(int) >= 1931]
cusd['age'] = pd.Series([2020]*3912) - pd.Series(cusd['DOB'].dt.to_period('Y').astype(str).astype(int))
cusd.describe()
add = pd.read_excel(r'../input/KPMG_data.xlsx', sheet_name='CustomerAddress')
add.head()
add.columns
add.dtypes
len(add.customer_id.unique())/len(add)
add.state.unique()
add.state = add.state.str.replace('New South Wales','NSW')
add.state = add.state.str.replace('Victoria','VIC')
add.isnull().sum()
add.describe()
new = pd.read_excel(r'../input/KPMG_data.xlsx', sheet_name='NewCustomerList')
new.head()
new.columns
new.dtypes
new.isnull().sum()
new=new.drop(columns=['Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20','country'])
new.describe()
info = cusd.merge(add, how='left', left_on='customer_id', right_on='customer_id')
info.head()
info.shape
info.isnull().sum()
total_customer = trans.merge(info, how='left',right_on='customer_id', left_on='customer_id')
total_customer.head()
# delete missing values
total_customer = total_customer.dropna(subset=['first_name']).reset_index(drop=True)
# create some new columns to explore the data
total_customer['profit'] = total_customer['list_price'] - total_customer['standard_cost']
total_customer['transaction_month']=total_customer['transaction_date'].dt.to_period('M')
total_customer['recency']=total_customer.transaction_date.max() - total_customer.transaction_date
total_customer['recency'] = total_customer['recency'].dt.days.astype(int)
trans[trans['order_status'] == 'Approved'].count()/trans.count() *100
total_customer.groupby(['product_line'])\
            .count()['transaction_id']\
            .reset_index()
total_customer.groupby(['product_line','product_class'])\
            .agg({'transaction_id':'count',
                  'list_price': ['sum','mean'],
                  'profit':'mean'})
high_product = total_customer.groupby(['product_id'])\
            .agg({'transaction_id':'count',
                  'list_price': 'sum',
                  'standard_cost':'sum'})\
            .reset_index()
high_product.head()
high_product.describe()
high_product_id = high_product[high_product['list_price']>=275228.8]['product_id'].reset_index(drop=True)
def id_map(product_id):
    if product_id in high_product_id:
        classify = 'High margin product'
    else:
        classify = 'Low margin product'
    return classify
total_customer['classify'] = total_customer.product_id.apply(id_map)
total_customer.groupby(['classify']).count()['customer_id']
trans_by_cus = total_customer.groupby(['customer_id'])\
                            .agg({'transaction_id':'count',
                                  'list_price':'sum',
                                  'profit':'sum'})\
                            .reset_index()
trans_by_cus.head()
trans_by_cus.groupby(['transaction_id'])\
            .agg({'customer_id':'count',
                  'list_price':'mean',
                  'profit':'mean'})\
            .reset_index()
total_customer['group_age'] = pd.cut(x = total_customer.age,
                                  bins=[10,40,63,100],
                                  labels=['youth','middle','old']) 
data_t = trans_by_cus.merge(total_customer.groupby(['customer_id']).head(1).reset_index(drop=True),
                   how='left',on='customer_id')
data_t = data_t.drop(columns=['transaction_id_y', 'product_id', 'transaction_date', 
                     'online_order','order_status', 'brand', 'product_line', 
                     'product_class', 'product_size', 'list_price_y', 'standard_cost',
                     'product_first_sold_date', 'first_name', 'last_name','DOB',
                     'job_title','deceased_indicator','postcode',
                     'country', 'profit_y','recency' ])
data_t.columns
data_t.head()
data_t.groupby(['group_age']).agg({'customer_id':'count',
                                 'profit_x':'mean'})
data_t.groupby(['gender']).agg({'customer_id':'count',
                              'profit_x':'mean'})
data_t.groupby(['wealth_segment']).agg({'customer_id':'count',
                                        'profit_x':'mean'})
data_t.groupby(['state']).agg({'customer_id':'count',
                             'profit_x':'mean'})
data_t.groupby(['job_industry_category']).agg({'customer_id':'count',
                                        'profit_x':'mean'})
age_tenure = data_t[['age','tenure']]
# correlation between age and tenure is average possitive
age_tenure.corr()
# there is approximately not correlate between age and property valuation
age_prop = data_t[['age','property_valuation']]
age_prop.corr()
age_3 = data_t[['age','past_3_years_bike_related_purchases']]
age_3.corr()
data_t.describe()
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
stats.ttest_1samp(data_t.list_price_x, popmean=6269)
stats.ttest_1samp(data_t.profit_x, popmean=3120)
male = data_t[data_t.gender =='Male']
female = data_t[data_t.gender =='Female']
stats.ttest_ind(male.profit_x, female.profit_x, equal_var=False)
youth = data_t[data_t.group_age =='youth']
mid = data_t[data_t.group_age =='middle']
old = data_t[data_t.group_age =='old']
# variance testing: is equal
stats.levene(youth.profit_x,mid.profit_x,old.profit_x)
stats.f_oneway(youth.profit_x, mid.profit_x, old.profit_x)
#there is no difference between group age
print(pairwise_tukeyhsd(data_t.profit_x,data_t.group_age))
total_customer.head()
rfm = total_customer.groupby(['customer_id'])\
                    .agg({'transaction_id':'count','recency':'min', 'profit':'sum'})\
                    .reset_index()
rfm = rfm.rename(columns=({
    'transaction_id' : 'frequency',
    'profit':'monetary'
}))

rfm.head()
rfm.describe()
def f_map(frequency):
    if frequency <= 4:
        f_score = 1
    elif frequency <=6:
        f_score = 2
    elif frequency <=7:
        f_score = 3
    else:
        f_score = 4
    return f_score
def r_map(recency):
    if recency <= 18:
        r_score = 4
    elif recency <=44:
        r_score = 3
    elif recency <=86:
        r_score = 2
    else:
        r_score = 1
    return r_score
def m_map(monetary):
    if monetary <= 1834.8925:
        m_score = 1
    elif monetary <= 2848.865:
        m_score = 2
    elif monetary <= 4170.79:
        m_score = 3
    else:
        m_score = 4
    return m_score
rfm['r_score']=rfm.recency.apply(r_map)
rfm['f_score']=rfm.frequency.apply(f_map)
rfm['m_score']=rfm.monetary.apply(m_map)
rfm['RFM_values'] = rfm.r_score*100 + rfm.f_score*10 + rfm.m_score
rfm.head()
rfm['RFM_values'].quantile([0.25,0.5,0.75])
def rank(RFM_values):
    if RFM_values <= 211:
        rank = 'Bronze'
    elif RFM_values <= 311:
        rank = 'Silver'
    elif RFM_values <= 411:
        rank = 'Gold'
    else:
        rank = 'Platinum'
    return rank
rfm['customer_rank'] = rfm['RFM_values'].apply(rank)
rfm.head()
rfm.groupby(['customer_rank']).count()['customer_id']
rfm['RFM_values'].quantile(np.arange(0,1.1,0.1))
def segment(RFM_values):
    if RFM_values == 444:
        segment = 'Champions'
    elif RFM_values >= 433:
        segment = 'Loyal Customers'
    elif RFM_values >= 421:
        segment = 'Potential Loyalist'
    elif RFM_values >= 344:
        segment = 'Recent Customers'
    elif RFM_values >= 323:
        segment = 'Promising'
    elif RFM_values >= 311 :
        segment = 'Customers Needing Attention'
    elif RFM_values >= 224 :
        segment = 'About To Sleep'
    elif RFM_values >= 212 :
        segment = 'At Risk'
    elif RFM_values >= 124 :
        segment = 'Can’t Lose Them'
    elif RFM_values >= 112 :
        segment = 'Hibernating'
    else:
        segment='Lost'
    return segment
rfm['customer_segment']=rfm['RFM_values'].apply(segment)
rfm['customer_segment']=pd.Categorical(rfm['customer_segment'],
                                       ['Champions','Loyal Customers',
                                        'Potential Loyalist','Recent Customers',
                                        'Promising','Customers Needing Attention',
                                        'About To Sleep','At Risk',
                                        'Can’t Lose Them','Hibernating','Lost'])
rfm.head()
customer_segment = rfm.groupby(['customer_segment']).agg({
    'customer_id':'count',
    'monetary':'mean'
}).reset_index()
customer_segment['cumulative'] = customer_segment.customer_id.cumsum()
customer_segment
data = data_t.merge(rfm, how='left', right_on='customer_id',left_on='customer_id')
data = data.rename(columns=({
   'transaction_id_x':'transaction_count',
    'list_price_x':'total_sales',
    'profit_x':'total_profit'
  
}))
highv_customer = data[(data['customer_segment']=='Champions') |
                      (data['customer_segment']=='Loyal Customers') |
                      (data['customer_segment']=='Potential Loyalist') |
                      (data['customer_segment']=='Recent Customers')]\
                .merge(info[['first_name','customer_id']], how='left',on='customer_id')

highv_customer.head()
data.head()
data.columns
px.bar(data.groupby(['wealth_segment', 'customer_segment','gender','group_age']).count().reset_index(), 
       x='customer_segment',y='customer_id',
       color='wealth_segment',facet_col='gender', facet_row='group_age',
       orientation='v',height=1000, width=800)
px.bar(data.groupby(['wealth_segment', 'customer_segment','gender']).mean().reset_index(), 
       x='customer_segment',y='total_profit',
       color='wealth_segment',facet_row='gender',
       orientation='v', barmode='group')
px.bar(data.groupby(['wealth_segment', 'customer_segment','group_age']).mean().reset_index(), 
       x='customer_segment',y='total_profit',
       color='wealth_segment',facet_row='group_age',
       orientation='v', barmode='group',height=1000)
px.bar(data.groupby(['state', 'customer_segment','wealth_segment']).mean().reset_index(), 
       x='customer_segment',y='total_profit',
       color='wealth_segment',facet_row='state',
       orientation='v', barmode='group',height=800)
px.treemap(data.groupby(['customer_segment']).count().reset_index(),
       values='customer_id',path=['customer_segment'])
px.pie(data.groupby(['job_industry_category']).count().reset_index(),
       values='customer_id',names='job_industry_category')
data.transaction_month=data.transaction_month.astype(str)
px.line(data.groupby(['transaction_month','customer_segment']).count().reset_index(), 
       x='transaction_month',y='customer_id',color='customer_segment')
px.bar(data.groupby(['state','wealth_segment']).mean().reset_index(), 
       x='state',y='RFM_values',color='wealth_segment',
       orientation='v', barmode='group')
px.bar(data.groupby(['classify','customer_segment']).count().reset_index(), 
       x='classify',y='customer_id',color='customer_segment',
       orientation='v', barmode='group')
px.scatter(data, x='recency',y='monetary')
px.scatter(data, x='frequency',y='monetary')
px.scatter(data, x='recency',y='frequency')
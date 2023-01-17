import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import re
import seaborn as sns
from subprocess import check_output
from sklearn.preprocessing import LabelEncoder
plt.style.use('ggplot')

pd.options.display.max_columns=999
pd.options.display.max_rows = 999
print(check_output(["ls", "../input"]).decode("utf8"))
train_data = pd.read_csv('../input/train.tsv', low_memory='False',sep='\t')
train_data = train_data.set_index('train_id')
train_data.head()
test_data = pd.read_csv('../input/test.tsv', low_memory='False',sep='\t')
test_data = test_data.set_index('test_id')
test_data.head()
def data_overview(df):
    print('Data information:')
    df_info = df.info(verbose=False)
    df_describe = df.describe()
    df_missing = df.isnull().sum()[df.isnull().sum()>0]
    print('Data description : ')
    print(np.round(df_describe,2))
    print('Missing Data values:')
    print(df_missing)
#train_data.describe()
data_overview(train_data)
data_overview(test_data)
train_data['brand_name']= train_data['brand_name'].fillna('missing')
test_data['brand_name'] = test_data['brand_name'].fillna('missing')
train_data['item_description'] = train_data['item_description'].fillna('missing')
from pandas.plotting import table
fig,ax= plt.subplots(figsize=(12,8))
table(ax, train_data['price'].to_frame().describe(),loc='upper right', colWidths=[0.2, 0.2, 0.2]);
sns.distplot(train_data['price'],hist=False,kde=True);
train_data['log_price']=np.log(train_data['price']+1)
from pandas.plotting import table
fig,ax= plt.subplots(figsize=(12,8))
#table(ax, train_data['log_price'].to_frame().describe(),loc='upper right', colWidths=[0.2, 0.2, 0.2]);
sns.distplot(train_data['log_price'],hist=False,kde=True);
ax.set_xlabel('log(price+1)');
fig,ax = plt.subplots(figsize=(12,8))
train_data['price'].plot();
ax.set_ylabel('price');
train_data.category_name.value_counts().iloc[0:10]
train_data['category_name']=train_data['category_name'].fillna('NaN/NaN/NaN')
test_data['category_name'] = test_data['category_name'].fillna('NaN/NaN/NaN')
def create_cats(df):
    ## df - dataframe input: test_data or train_data
    first_cat = []
    second_cat = []
    third_cat = []
    
    for i in range(len(df)):
        cat_names = df.category_name[i].split('/')
        cat1 = cat_names[0]
        cat2 = cat_names[1]
        cat3 = cat_names[2]
        first_cat.append(cat1)
        second_cat.append(cat2)
        third_cat.append(cat3)
    return first_cat, second_cat, third_cat

first_cat_train, second_cat_train, third_cat_train = create_cats(train_data)
first_cat_test, second_cat_test, third_cat_test = create_cats(test_data)
train_data['first_cat']=first_cat_train
train_data['second_cat']=second_cat_train
train_data['third_cat']=third_cat_train
test_data['first_cat']=first_cat_test
test_data['second_cat']=second_cat_test
test_data['third_cat']=third_cat_test
train_data.head()
test_data.head()
print("Numer of unique categories = %s" %len(train_data['category_name'].unique()))
fig,ax =plt.subplots()
train_data['category_name'].value_counts()[:30].plot(kind='barh',figsize=(12,8));
ax.set_xlabel('count');
fig,ax = plt.subplots()
train_data.groupby(by='category_name').mean()['price'].sort_values(ascending=False).iloc[0:30].plot(kind='barh', figsize=(12,8));
ax.set_xlabel('price');
print("Number of first_cat=%s" %len(train_data['first_cat'].unique()))
fig,ax = plt.subplots()
train_data['first_cat'].value_counts().plot(kind='barh',figsize=(12,8));
ax.set_xlabel('count');
fig,ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='price',y='first_cat', data=train_data, jitter=True);
fig,ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='price',y='first_cat', data=train_data[train_data.price>=1000], jitter=True);
print("Number of second_cat = %s" %len(train_data['second_cat'].unique()))
train_sc_df = train_data.groupby(by='second_cat').filter(lambda x: len(x)>20000)
fig,ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='price',y='second_cat', data=train_sc_df, jitter=True);
ax.set_title('Most common second categories');
print("Number of third_cat = %s" %len(train_data['third_cat'].unique()))
train_tc_df = train_data.groupby(by='third_cat').filter(lambda x: len(x)>10000)
fig,ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='price',y='third_cat', data=train_tc_df, jitter=True);
ax.set_title('Most common third categories');
train_data['name'].iloc[0:10]
train_data['name_len']= train_data['name'].apply(lambda x: len(x.split(' ')))
test_data['name_len']= test_data['name'].apply(lambda x: len(x.split(' ')))
print("Number of unique name_len = %s" %len(train_data['name_len'].unique()))
print(train_data['name_len'].value_counts())
sns.factorplot(x='name_len',y='price',data=train_data,size=8,aspect=1.5);
sns.factorplot(x='name_len',y='price',data=train_data[train_data.name_len<=10],size=8,aspect=1.5);
train_data.shipping.value_counts()
fig,ax = plt.subplots(figsize=(12,8))
sns.distplot(train_data['log_price'][train_data.shipping==0], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['log_price'][train_data.shipping==1], hist=False, kde=True,label='shipping paid by seller');
ax.set_xlabel('log(price+1)');
plt.figure(figsize=(12,12))
ax1 = plt.subplot(321)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price<=20)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price<=20)], hist=False, kde=True,label='shipping paid by seller');
ax2 = plt.subplot(322)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price>20)& (train_data.price<=100)], hist=False, kde=True,label='shipping paid by seller');
ax3 = plt.subplot(323)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price>100)& (train_data.price<=500)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price>100)& (train_data.price<=500)], hist=False, kde=True,label='shipping paid by seller');
ax4 = plt.subplot(324)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price>500)& (train_data.price<=1500)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price>500)& (train_data.price<=1500)], hist=False, kde=True,label='shipping paid by seller');
ax5 = plt.subplot(325)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price>1500)& (train_data.price<=2500)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price>1500)& (train_data.price<=2500)], hist=False, kde=True,label='shipping paid by seller');
g =sns.factorplot(x='first_cat',y='price',hue='shipping', data=train_data, size=8, aspect=1.5, kind='bar');
g.set_xticklabels(rotation=75);
sns.factorplot('shipping',col='first_cat',col_wrap=4, data=train_data, size=8, aspect=.5,sharey=False, sharex=False, kind='count');
train_data.item_condition_id.value_counts()
fig,ax = plt.subplots(figsize=(12,8))
sns.distplot(train_data['log_price'][(train_data.item_condition_id==1)], hist=False,kde=True, label='1');
sns.distplot(train_data['log_price'][(train_data.item_condition_id==2)], hist=False,kde=True, label='2');
sns.distplot(train_data['log_price'][(train_data.item_condition_id==3)], hist=False,kde=True, label='3');
sns.distplot(train_data['log_price'][(train_data.item_condition_id==4)], hist=False,kde=True, label='4');
sns.distplot(train_data['log_price'][(train_data.item_condition_id==5)], hist=False,kde=True, label='5');
ax.set_xlabel('log(price+1)');
plt.subplots(figsize=(12,12))
ax1=plt.subplot(321)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price<=20)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price<=20)], hist=False,kde=True, label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price<=20)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price<=20)], hist=False,kde=True, label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price<=20)], hist=False,kde=True, label='5');

ax2 = plt.subplot(322)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='5');

ax3 = plt.subplot(323)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price>100)& (train_data.price<=500)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price>100)& (train_data.price<=500)], hist=False, kde=True,label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price>100)& (train_data.price<=500)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price>100)& (train_data.price<=500)], hist=False, kde=True,label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price>100)& (train_data.price<=500)], hist=False, kde=True,label='5');

ax4 = plt.subplot(324)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price>500)& (train_data.price<=1500)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price>500)& (train_data.price<=1500)], hist=False, kde=True,label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price>500)& (train_data.price<=1500)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price>500)& (train_data.price<=1500)], hist=False, kde=True,label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price>500)& (train_data.price<=1500)], hist=False,kde=True, label='5');

ax5 = plt.subplot(325)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price>1500)& (train_data.price<=2500)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price>1500)& (train_data.price<=2500)], hist=False, kde=True,label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price>1500)& (train_data.price<=2500)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price>1500)& (train_data.price<=2500)], hist=False, kde=True,label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price>1500)& (train_data.price<=2500)], hist=False,kde=True, label='5');
g =sns.factorplot(x='first_cat',y='price',hue='item_condition_id', data=train_data, size=8, aspect=1.5, kind='bar');
g.set_xticklabels(rotation=75);
train_data.groupby(by='brand_name').mean()['price'].sort_values(ascending=False).iloc[0:10]
print('Total number of brands used in train_data set= %s' %len(train_data['brand_name'].value_counts()))
print('Total number of brands used in test_data set= %s' %len(test_data['brand_name'].value_counts()))
train_data['branded'] = train_data['brand_name'].apply(lambda x: 0 if x =='missing' else 1)
test_data['branded'] = test_data['brand_name'].apply(lambda x: 0 if x =='missing' else 1)
fig,ax= plt.subplots(figsize=(12,8))
sns.distplot(train_data['log_price'][train_data.branded==0], hist=False, kde=True,label='Not branded');
sns.distplot(train_data['log_price'][train_data.branded==1], hist=False, kde=True, label='Branded');
ax.set_xlabel('log(price+1)');
g =sns.factorplot(x='first_cat',y='price',hue='branded', data=train_data, size=8, aspect=1.5, kind='bar');
g.set_xticklabels(rotation=75);
train_data['desc_len']=train_data['item_description'].apply(lambda x: len(x.split(' ')))
test_data['desc_len']=test_data['item_description'].apply(lambda x: len(x.split(' ')))
fig,ax= plt.subplots(figsize=(12,8))
train_data.groupby('desc_len').count()['price'][0:50].plot(kind='bar');
ax.set_ylabel('count');
fig,ax= plt.subplots(figsize=(12,8))
train_data.groupby('desc_len').mean()['price'].iloc[0:150].plot();
ax.set_ylabel('average price');
fig,ax= plt.subplots(figsize=(12,12))
train_data_gb_fc_dl = train_data.groupby(['first_cat','desc_len']).mean()['price']
plt.plot(train_data_gb_fc_dl.xs('Beauty')[0:50],'-bo')
plt.plot(train_data_gb_fc_dl.xs('Women')[0:50],'-ro')
plt.plot(train_data_gb_fc_dl.xs('Men')[0:50],'-go')
plt.plot(train_data_gb_fc_dl.xs('Electronics')[0:50],'-ko')
plt.plot(train_data_gb_fc_dl.xs('Kids')[0:50],'-yo')
plt.plot(train_data_gb_fc_dl.xs('Home')[0:50],'-mo')
plt.plot(train_data_gb_fc_dl.xs('Sports & Outdoors')[0:50],'-co')
plt.plot(train_data_gb_fc_dl.xs('Vintage & Collectibles')[0:50],'-k*')
plt.plot(train_data_gb_fc_dl.xs('Handmade')[0:50],'-g*')
#plt.plot(train_data_gb_fc_dl.xs('Other')[0:50],'--y*');
#plt.plot(train_data_gb_fc_dl.xs('NaN')[0:50],'-mo');
plt.legend(['Beauty','Women','Men','Electronics','Kids','Home','Sports & Outdoors','Vintage & Collectibles',
           'Handmade']);
ax.set_xlabel('desc_len')
ax.set_ylabel('average price');
desc_len_range = [(0,20),(21,50),(51,100),(101,150),(151,250)]
item_description_len = []
def cont_to_range(range_list,series):
    #range_list: list of range we want to generate
    #series: pandas series whose values are converted as range
    Series_col_range=[]
    for j in range(len(series)):
        for i in range(len(range_list)):
            if series[j] in range(range_list[i][0], range_list[i][1]):
                series_range = range_list[i]
            else:
                pass
        Series_col_range.append(series_range)
    return Series_col_range
train_data['item_description_len']=cont_to_range(desc_len_range,train_data.desc_len)
test_data['item_description_len']=cont_to_range(desc_len_range,test_data.desc_len)
g =sns.factorplot(x='first_cat',y='price',col='item_description_len',col_wrap=3, data=train_data, size=12, aspect=.5,kind='bar');
g.set_xticklabels(rotation=75);
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
def count_significant_words(desc):
    try:
        desc =  desc.lower()
        desc_reg = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        desc_txt = desc_reg.sub(" ", desc)

        words = [w for w in desc_txt.split(" ") if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]
        return len(words)
    except: 
        return 0
train_data['item_desc_word_count'] = train_data['item_description'].apply(lambda x: count_significant_words(x))
test_data['item_desc_word_count'] = test_data['item_description'].apply(lambda x: count_significant_words(x))

train_data.head()
fig,ax= plt.subplots(figsize=(12,8))
train_data.groupby('item_desc_word_count').mean()['price'].iloc[0:150].plot();
ax.set_ylabel('average price');
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenized(desc):
    desc = desc.lower()
    desc_reg = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    desc_txt = desc_reg.sub(" ", desc)
    tokenized_words = word_tokenize(desc_txt) 
    tokens = list(filter(lambda t: t.lower() not in stop, tokenized_words))
    filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
    filtered_tokens = set([w.lower() for w in filtered_tokens if len(w)>=3])
    filtered_tokens = list(filtered_tokens)
    return filtered_tokens
    
train_data['item_desc_tokenized'] =train_data['item_description'].map(tokenized)
test_data['item_desc_tokenized'] = test_data['item_description'].map(tokenized)
for description, tokens in zip(train_data['item_description'].head(),
                              train_data['item_desc_tokenized'].head()):
    print('description:', description)
    print('tokens:', tokens)
    print()

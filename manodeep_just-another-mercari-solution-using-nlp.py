# import initial libraries
import pandas as pd
import numpy as np
from numpy import NaN
import seaborn as sns
from scipy import stats
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
train = pd.read_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep = '\t', low_memory=True)
train.head()
#function to get all info in one go
def full_info(df):
    df_column=[]
    df_dtype=[]
    df_null=[]
    df_nullc=[]
    df_mean=[]
    df_median=[]
    df_std=[]
    df_min=[]
    df_max=[]
    df_uniq=[]
    df_count=[]
    for col in df.columns: 
        df_column.append(  col )
        df_dtype.append( df[col].dtype)
        df_null.append( round(100 * df[col].isnull().sum(axis=0)/len(df[col]),2))
        df_nullc.append( df[col].isnull().sum(axis=0))
        df_uniq.append( df[col].nunique()) if df[col].dtype == 'object' else df_uniq.append( NaN)
        df_mean.append(  '{0:.2f}'.format(df[col].mean())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_mean.append( NaN)
        df_median.append( '{0:.2f}'.format(df[col].median())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_median.append( NaN)
        df_std.append( '{0:.2f}'.format(df[col].std())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_std.append( NaN)
        df_max.append( '{0:.2f}'.format(df[col].max())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_max.append( NaN)
        df_min.append( '{0:.2f}'.format(df[col].min())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_min.append( NaN)
        df_count.append(df[col].count())
    return pd.DataFrame(data = {'ColName':  df_column, 'ColType': df_dtype, 'NullCnt': df_nullc, 'NullCntPrcntg': df_null,  'Min': df_min, 'Max': df_max, 'Mean': df_mean, 'Med': df_median, 'Std': df_std, 'UniqCnt': df_uniq, 'ColValCnt': df_count})

# lets get full desciption of the data
full_info(train)
print(train['category_name'].str.count('/').min())
print(train['category_name'].str.count('/').max())
# lets split the category into category splits
train_sp = train.join(train['category_name'].str.split('/', expand=True).add_prefix('category_split_'))

# and lets see how the data looks
train_sp.head()
# lets get full desciption of the data again
train_sp_info= full_info(train_sp)
train_sp_info
# lets visualize the Null Count percentage graphically
train_sp_info.plot.bar(x = 'ColName', y = 'NullCntPrcntg', figsize=(20, 5),rot=90, title='Missing (null) Feature Values')
plt.show()
# lets drop the category_split_3 and category_split_4 as they have most nulls. lets keep the brand for now.
train_sp_trim=train_sp.drop(['category_split_3', 'category_split_4'],axis=1)
# lets see how the data looks like now
train_sp_trim
# lets remove the items with price of $0 as well as they are of no use in price prediction
train_sp_trim = train_sp_trim[train_sp_trim.price != 0]

# Create a function to impute missing values
def fill_missing_value(df):
    df['category_split_0'].fillna(value = 'unknown', inplace=True)
    df['category_split_1'].fillna(value = 'unknown', inplace=True)
    df['category_split_2'].fillna(value = 'unknown', inplace=True)
    df['brand_name'].fillna(value = 'unknown', inplace=True)
    df['category_name'].fillna(value = 'unknown', inplace=True)
    df['item_description'].fillna(value = 'No description yet', inplace=True)
    
    return df
# lets apply the fill_missing_value function on the data to fill the nulls
train_fill = fill_missing_value(train_sp_trim)

# lets get full desciption of the data again
full_info(train_fill)
# lets see the price distribution visually
sns.set()
sns.distplot(train_fill['price'], bins = 50)
plt.title('Price Distribution', fontsize=12);
plt.figure(figsize=(20, 6))
plt.hist(train_fill['price'], bins=50, range=[0,2010], label='price')
plt.title('Price Distribution', fontsize=15)
plt.xlabel('Price', fontsize=15)
plt.ylabel('Samples', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.show()
plt.figure(figsize=(20, 6))
sns.distplot(np.log(train_fill['price']+1), fit = stats.norm)
plt.xlabel('log(price+1)', fontsize=12)
plt.title('Log Price Distribution', fontsize=12);
# lets see by brand name how its priced
brand = train_fill.groupby('brand_name').price.agg(['count','mean'])
brand = brand[brand['count']>1000].sort_values(by='mean', ascending=False)
brand.head(30)
# lets visualize the count by brand name
brand = train_fill['brand_name'].value_counts()
fig = go.Figure([go.Pie(labels=brand.keys(), values=brand)])
fig.update_traces( hoverinfo="label+percent")
fig.update_layout(title_text="% by Brand")
fig.show()
# lets visualize Top 75 Expensive Brands By Mean Price
plt.figure(figsize=(25, 6))
top_brands = train_fill.groupby('brand_name', axis=0).mean()
df_expPrice = pd.DataFrame(top_brands.sort_values('price', ascending = False)['price'][0:75].reset_index())


ax = sns.barplot(x="brand_name", y="price", data=df_expPrice)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=15)
ax.set_title('Top Expensive Brands', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# lets see by main category how its priced
maincat = train_fill.groupby('category_split_0').price.agg(['count','mean'])
maincat = maincat[maincat['count']>1000].sort_values(by='mean', ascending=False)
maincat.head(30)
# lets visualize the count by main category or category_split_0
categories = train_fill['category_split_0'].value_counts()
fig = go.Figure([go.Pie(labels=categories.keys(), values=categories)])
fig.update_traces( hoverinfo="label+percent")
fig.update_layout(title_text="% by Main Category")
fig.show()
# lets visualize Top 75 category_split_0 By Mean Price
plt.figure(figsize=(25, 6))
category = train_fill.groupby('category_split_0', axis=0).mean()
df_expPrice = pd.DataFrame(category.sort_values('price', ascending = False)['price'][0:75].reset_index())
result = df_expPrice.groupby(["category_split_0"])['price'].aggregate(np.median).reset_index().sort_values('price')

ax = sns.barplot(x="category_split_0", y="price", data=df_expPrice)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=15)
ax.set_title('Top Expensive Main Sub Category', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
#price comparison by item condition across 5 most frequent Main Categories
plt.figure(figsize=(20, 6))
sns.barplot(x='item_condition_id', y="price", hue='category_split_0', data=train_fill[(train_fill['category_split_0'] == train_fill['category_split_0'].value_counts().index[0]) | (train_fill['category_split_0'] == train_fill['category_split_0'].value_counts().index[1]) | (train_fill['category_split_0'] == train_fill['category_split_0'].value_counts().index[2]) | (train_fill['category_split_0'] == train_fill['category_split_0'].value_counts().index[3]) | (train_fill['category_split_0'] == train_fill['category_split_0'].value_counts().index[4])])
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handles[4], handles[3], handles[2], handles[0], handles[1]]
labels = [labels[4], labels[3], labels[2], labels[0], labels[1]]
plt.legend(handles, labels, title='Top 5 categories1', loc='upper right');
# lets see by second main category or category_split_1 how its priced
seccat = train_fill.groupby('category_split_1').price.agg(['count','mean'])
seccat = seccat[seccat['count']>1000].sort_values(by='mean', ascending=False)
seccat.head(30)
# Display Top 75 category_split_1 By Mean Price
plt.figure(figsize=(25, 6))
category = train_fill.groupby('category_split_1', axis=0).mean()
df_expPrice = pd.DataFrame(category.sort_values('price', ascending = False)['price'][0:75].reset_index())

ax = sns.barplot(x="category_split_1", y="price", data=df_expPrice)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=15)
ax.set_title('Top Expensive category1', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
#price comparison by item condition across 5 most frequent Second Main Categories
plt.figure(figsize=(20, 6))
sns.barplot(x='item_condition_id', y="price", hue='category_split_1', data=train_fill[(train_fill['category_split_1'] == train_fill['category_split_1'].value_counts().index[0]) | (train_fill['category_split_1'] == train_fill['category_split_1'].value_counts().index[1]) | (train_fill['category_split_1'] == train_fill['category_split_1'].value_counts().index[2]) | (train_fill['category_split_1'] == train_fill['category_split_1'].value_counts().index[3]) | (train_fill['category_split_1'] == train_fill['category_split_1'].value_counts().index[4])])
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handles[4], handles[3], handles[2], handles[0], handles[1]]
labels = [labels[4], labels[3], labels[2], labels[0], labels[1]]
plt.legend(handles, labels, title='Top 5 categories2', loc='upper right');
# lets see by third main category or category_split_2 how its priced
thrdcat = train_fill.groupby('category_split_2').price.agg(['count','mean'])
thrdcat = thrdcat[thrdcat['count']>1000].sort_values(by='mean', ascending=False)
thrdcat.head(30)
# Display Top 75 category_split_2 By Mean Price
plt.figure(figsize=(25, 6))
top_category2 = train_fill.groupby('category_split_2', axis=0).mean()
df_expPrice = pd.DataFrame(top_category2.sort_values('price', ascending = False)['price'][0:75].reset_index())

ax = sns.barplot(x="category_split_2", y="price", data=df_expPrice)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90, fontsize=15)
ax.set_title('Top Expensive category2', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
#price comparison by item condition across 5 most frequent Third Main Categories
plt.figure(figsize=(20, 6))
sns.barplot(x='item_condition_id', y="price", hue='category_split_2', data=train_fill[(train_fill['category_split_2'] == train_fill['category_split_2'].value_counts().index[0]) | (train_fill['category_split_2'] == train_fill['category_split_2'].value_counts().index[1]) | (train_fill['category_split_2'] == train_fill['category_split_2'].value_counts().index[2]) | (train_fill['category_split_2'] == train_fill['category_split_2'].value_counts().index[3]) | (train_fill['category_split_2'] == train_fill['category_split_2'].value_counts().index[4])])
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handles[4], handles[3], handles[2], handles[0], handles[1]]
labels = [labels[4], labels[3], labels[2], labels[0], labels[1]]
plt.legend(handles, labels, title='Top 5 categories3', loc='upper right');
# lets visualize the count by item_condition_id
conditioncnt = train_fill['item_condition_id'].value_counts()
fig = go.Figure([go.Pie(labels=conditioncnt.keys(), values=conditioncnt)])
fig.update_traces( hoverinfo="label+percent")
fig.update_layout(title_text="% by Item Condition")
fig.show()
# visualizing the price distribution by Item Condition
plt.figure(figsize=(25, 8))

sns.boxplot(x='item_condition_id', y="price", data=train_fill)
plt.ylim(0, 200);
#price comparison by item condition across 5 most frequent Category Names
plt.figure(figsize=(20, 6))
sns.barplot(x='item_condition_id', y="price", hue='category_name', data=train_fill[(train_fill['category_name'] == train_fill['category_name'].value_counts().index[0]) | (train_fill['category_name'] == train_fill['category_name'].value_counts().index[1]) | (train_fill['category_name'] == train_fill['category_name'].value_counts().index[2]) | (train_fill['category_name'] == train_fill['category_name'].value_counts().index[3]) | (train_fill['category_name'] == train_fill['category_name'].value_counts().index[4])])
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handles[4], handles[3], handles[2], handles[0], handles[1]]
labels = [labels[4], labels[3], labels[2], labels[0], labels[1]]
plt.legend(handles, labels, title='Top 5 categories', loc='upper right');
# define function for text normalization
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stopwords
from nltk.stem.porter import PorterStemmer

def text_normalization(text):
    # lowercase words
    text = text.lower()
    # remove stopwords
    text = ' '.join([i for i in text.split(' ') if i not in stopwords])
    #remove digits
    text = ''.join([i for i in text if not i.isdigit()])
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text
%%time

train_fill['item_description_normalized'] = train_fill['item_description'].apply(text_normalization).copy()
%%time

train_fill['name_normalized'] = train_fill['name'].apply(text_normalization).copy()
pd.set_option('display.width', 1000)

# check first item description
train_fill['item_description'][:8]
# check first item description after normalization and compare with previous result
train_fill['item_description_normalized'][:8]
train_fill['name'][:8]
train_fill['name_normalized'][:8]
# Generate a word cloud image for name frequency
wordcloud = WordCloud().generate((train_fill['name_normalized'].sample(100000) + ' ').sum())
plt.figure(figsize=(20,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# Generate a word cloud image for description word frequency
wordcloud = WordCloud().generate((train_fill['item_description_normalized'].sample(100000) + ' ').sum())
plt.figure(figsize=(20,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer()
item_description_bow = vectorizer.fit_transform(train_fill['item_description_normalized'])
item_description_bow
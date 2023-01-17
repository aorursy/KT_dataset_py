import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline
import string
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.simplefilter("ignore")
#exclamatory mark runs the command in terminal
!ls ../input/
#read_csv reads csv file as dataframe.
#here files are tsv, so we use tab delimiter/separator.
train_df = pd.read_csv("../input/train.tsv",sep='\t')
test_df = pd.read_csv("../input/test.tsv",sep='\t')
#return 5 data points sampled randomly from dataset.
train_df.sample(5)
print("Length of Training Data: " + str(len(train_df)/100000) + " lac.")
print("Length of Testing Data: " + str(len(test_df)/100000)+ " lac.")
train_df.train_id.nunique()/len(train_df)
train_df.isnull().sum()
train_df.dtypes
plt.figure(figsize=(10,8))
plt.hist((train_df['price']), bins=30, range=[0,250], label='price')
plt.title('Price Distribution')
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()
train_df.price.describe()
len(train_df[train_df.price==0])
plt.figure(figsize=(10,8))
plt.hist(np.log(train_df['price']), bins=30, range=[0,8], label='price')
plt.title('Log Price Distribution')
plt.xlabel("Log Price")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(10,8))
sns.boxplot(x=train_df.price)
plt.xlim(-10,100)
plt.xlabel("Price")
plt.title("Box Plot of Price Variable")
plt.show()
train_df.sample(5)
item_condn_percent = train_df.item_condition_id.value_counts()/len(train_df)*100
plt.figure(figsize=(10,8))
sns.barplot(x=item_condn_percent.index,y=item_condn_percent.values)
plt.xlabel("Item Condition ID")
plt.ylabel("Percentage of products")
plt.title("Bar Plot of Item Condition with Percentage of Products")
plt.show()
pd.concat([item_condn_percent],keys=["percentage"],axis=1)
shipping_percent = train_df.shipping.value_counts()/len(train_df)*100
plt.figure(figsize=(5,8))
sns.barplot(y=shipping_percent.values,x=shipping_percent.index)
plt.title("Bar Plot of Shipping with Percentage of Products")
plt.xlabel("Shipping Variable")
plt.ylabel("Percentage")
plt.show()
pd.concat([shipping_percent],keys=["percentage"],axis=1)
train_df.sample(5)
train_df = train_df[train_df.category_name.notnull()]
train_df['general_cat'], train_df['sub_cat1'], train_df['sub_cat2'] = \
zip(*train_df.category_name.apply(lambda x: x.split('/')))
train_df.sample(5)
train_df.general_cat.nunique()
general_cat_percent = train_df.general_cat.value_counts().sort_values(ascending=False)/len(train_df)*100
plt.figure(figsize=(15,8))
sns.barplot(x=general_cat_percent.index,y=general_cat_percent.values)
plt.show()
pd.concat([general_cat_percent],keys=["percent"],axis=1)
sub_cat1_percent = train_df[train_df.general_cat=="Women"]['sub_cat1'].value_counts().\
    sort_values(ascending=False)[:10]/len(train_df[train_df.general_cat=="Women"])*100
plt.figure(figsize=(20,10))
sns.barplot(x=sub_cat1_percent.index,y=sub_cat1_percent.values)
plt.title("Percentage of various sub-categories INSIDE the WOMEN category")
plt.xlabel("Sub-category Name")
plt.ylabel("Percentage")
plt.show()
pd.concat([sub_cat1_percent],keys=["percentage"],axis=1)
sub_cat1_percent = train_df[train_df.general_cat=="Beauty"]['sub_cat1'].value_counts().\
    sort_values(ascending=False)/len(train_df[train_df.general_cat=="Beauty"])*100
plt.figure(figsize=(20,10))
sns.barplot(x=sub_cat1_percent.index,y=sub_cat1_percent.values)
plt.title("Percentage of various sub-categories INSIDE the BEAUTY category")
plt.xlabel("Sub-category Name")
plt.ylabel("Percentage")
plt.show()
pd.concat([sub_cat1_percent],keys=["percentage"],axis=1)
sub_cat1_percent = train_df[train_df.general_cat=="Kids"]['sub_cat1'].value_counts().\
    sort_values(ascending=False)[:10]/len(train_df[train_df.general_cat=="Kids"])*100
plt.figure(figsize=(20,10))
sns.barplot(x=sub_cat1_percent.index,y=sub_cat1_percent.values)
plt.title("Percentage of various sub-categories INSIDE the KIDS category")
plt.xlabel("Sub-category Name")
plt.ylabel("Percentage")
plt.show()
pd.concat([sub_cat1_percent],keys=["percentage"],axis=1)
train_df.sample(5)
import wordcloud as wc
desc_word_cloud = wc.WordCloud(width=1000,height=1000).generate(" ".join(train_df.sample(10000).item_description.astype(str)))
plt.figure(figsize = (20, 15))
plt.imshow(desc_word_cloud)
plt.show()
brand_name_percent = train_df['brand_name'].value_counts().\
    sort_values(ascending=False)[:10]/len(train_df)*100
plt.figure(figsize=(20,10))
sns.barplot(x=brand_name_percent.index,y=brand_name_percent.values)
plt.title("Bar plot of Brand Name")
plt.xlabel("Brand Name")
plt.ylabel("Percentage of Products")
plt.show()
pd.concat([brand_name_percent],keys=["percentage"],axis=1)
train_df.sample(5)
plt.figure(figsize=(10,8))
plt.hist(np.log(train_df[train_df.shipping==1]['price']), bins=30, range=[0,8],color='red',alpha=0.5,label="shipping=1")
plt.hist(np.log(train_df[train_df.shipping==0]['price']),bins=30,range=[0,8],color='blue',alpha=0.5,label="shipping=0")
plt.legend(loc='upper right')
plt.title("Log Price Distribution for different shipping categories.")
plt.ylabel("Count")
plt.xlabel("Log Price")
plt.show()
plt.figure(figsize=(10,8))
train_df['log_price'] = np.log(train_df['price'])
sns.boxplot(x="shipping",y="log_price",data=train_df)
plt.ylim(0,6)
plt.title("Box Plot of Log Price for different shipping categories.")
plt.ylabel("Log Price")
plt.xlabel("Shipping")
plt.show()
train_df.sample(5)
plt.figure(figsize=(20,10))
train_df['log_price'] = np.log(train_df['price'])
sns.boxplot(x="general_cat",y="log_price",data=train_df)
plt.ylim(1,6)
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(x="item_condition_id",y="log_price",data=train_df)
plt.ylim(1,6)
plt.show()

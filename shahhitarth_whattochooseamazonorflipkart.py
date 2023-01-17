!pip install pyforest
from pyforest import *
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn import metrics
ama = pd.read_csv('/kaggle/input/Amazon.csv')
flip = pd.read_csv('/kaggle/input/Flipkart.csv')
ama.head()
ama.describe()
ama.info()
print('-----------------------------------------------------------------')
flip.info()
print(ama.shape)
print(flip.shape)
print(ama.isnull().sum().sum())
print(flip.isnull().sum().sum())
flip.head()
flip.describe()
ama_flip = pd.concat([ama, flip],axis=1)
ama_flip.head()
ama_flip.info()
ama_flip.drop(columns=['amazon_author','amazon_title','amazon_rating','amazon_reviews count','amazon_isbn-10','flipkart_ratings count','flipkart_stars'], inplace=True)

ama_flip.head()   #isbn10 is same for both amazon and flipkart. So we can drop a column either from flipkart or amazon. 
ama_flip.info()
ama_flip.rename(columns={'flipkart_author':'Author', 'flipkart_isbn10':'BN', 'flipkart_title':'Title'}, inplace=True)
ama_flip.head()
ama_flip.isnull().sum()
ama_flip.replace(np.NaN,'Anonymous',inplace=True)
ama_flip.isnull().sum()
print(ama_flip.amazon_price.max())
print(ama_flip.amazon_price.min())
print(ama_flip.flipkart_price.max())
print(ama_flip.flipkart_price.min())

ama_flip.loc[ama_flip.flipkart_price == 5201]
ama_flip.loc[ama_flip.flipkart_price == 30]
ama_flip.loc[ama_flip.amazon_price == 895]
ama_flip.loc[ama_flip.amazon_price == 1]
ama_flip['price_diff'] = ama_flip['flipkart_price'] - ama_flip['amazon_price']
ama_flip.head()
ama_flip.loc[ama_flip.price_diff == 0]
same_price = print('Same price:',ama_flip[ama_flip.price_diff == 0].count()[0]/ama_flip.shape[0] * 100)
diff_price = print('Variation in price:',ama_flip[ama_flip.price_diff != 0].count()[0]/ama_flip.shape[0] * 100)
ama_flip.amazon_price.sum()
ama_flip.flipkart_price.sum()
ama_flip["flipkart_price"] = np.where(ama_flip["flipkart_price"] <=975, 975,ama_flip['flipkart_price'])
print(ama_flip['flipkart_price'].skew())
outliers = ama_flip[ama_flip.flipkart_price > 1000].index
ama_flip = ama_flip.drop(outliers)
plt.figure(figsize=(10,6))
sns.distplot(flip.flipkart_price,color='orange',bins=30,kde=False)
sns.distplot(ama.amazon_price,color='blue',bins=30,kde=False)
plt.legend()
plt.xlabel('Price')
plt.ylabel('Number of books')
plt.figure(figsize=(10,6))
sns.distplot(abs(ama_flip['price_diff']),color='red',bins=30,kde=False)
plt.legend()
plt.xlabel('Price')
plt.ylabel('Difference')
print(ama_flip.price_diff.max())
ama_flip.loc[ama_flip.price_diff == 367]
plt.figure(figsize=(10,6))
sns.distplot(ama_flip.price_diff,color='grey',bins=30,kde=False)
#sns.distplot(ama.amazon_price,color='blue',bins=30,kde=False)
plt.legend()
plt.xlabel('Price')
plt.ylabel('Price Difference')

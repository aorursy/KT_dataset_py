import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth 
from PIL import Image
sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%matplotlib inline
dataset = pd.read_csv('../input/market-basket-optimization/Market_Basket_Optimisation.csv', header = None)
dataset.head()
dataset.shape
dataset.describe()
mask = np.array(Image.open('../input/hamb1png/Hamb1.png'))
plt.subplots(figsize=(12,8))
wordcloud = WordCloud(
                          background_color='White',max_words = 60,
                          mask = mask, contour_color='orange', contour_width=4, 
                          width=1500, margin=10,
                          height=1080
                         ).generate(" ".join(dataset[0]))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
Products = pd.DataFrame(dataset[0].value_counts())
Twenty_Products = pd.DataFrame(dataset[0].value_counts()).head(20)

sns.barplot(x = Twenty_Products.index, y = Twenty_Products[0])

labels =Twenty_Products.index.tolist()
plt.gcf().set_size_inches(15, 7)

plt.title('20 most popular products vs. amount', fontsize = 20)
plt.xlabel('Most popular products', fontsize = 15)
plt.ylabel('Amount', fontsize = 15)

plt.xticks(ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] , labels = labels, rotation = '45')
plt.show()
Twenty_Products.columns = ['Amount']
Twenty_Products.head(10).style.background_gradient(cmap='plasma')
Twenty_Products["Item"] = Twenty_Products.index
Twenty_Products["20 most popular items"] = "20 most popular items"
Twenty_Products['index'] = list(range(len(Twenty_Products)))
Twenty_Products.set_index('index')
Twenty_Products = Twenty_Products[['Item','Amount',"20 most popular items"]]
fig = px.treemap(Twenty_Products, path=["20 most popular items", 'Item'], values='Amount',
                  color=Twenty_Products["Amount"], hover_data=['Item'],
                  color_continuous_scale='plasma',
                  )
fig.show()
tickets = []
for i in range(dataset.shape[0]):
    tickets.append([str(dataset.values[i,j]) for j in range(dataset.shape[1])])
    
tickets = np.array(tickets)
TE = TransactionEncoder()
dataset2 = TE.fit_transform(tickets)
dataset2 = pd.DataFrame(dataset2, columns = TE.columns_)
dataset2.head(3)
dataset_20 = dataset2.copy()
dataset_20 = dataset_20[Twenty_Products["Item"]] # Using the previous DF.
def encode_units(x):
    if x == False:
        return 0 
    if x == True:
        return 1
dataset_20 = dataset_20.applymap(encode_units) # Element wise function in DF.
dataset_20.head(3)
frequent_itemsets = apriori(dataset_20, min_support = 0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
frequent_itemsets[ (frequent_itemsets['length'] >= 2) &
                   (frequent_itemsets['support'] >= 0.04) ]
frequent_itemsets[ (frequent_itemsets['length'] == 3)].head(3)
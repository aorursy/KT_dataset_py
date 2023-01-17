import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/flavors_of_cacao.csv', encoding='utf-8')
# df.head()
df.keys()
df = df.fillna('-')
df.columns = ['company_name', 'bar_name', 'ref', 'review_data', 'cocoa_percent', 'company_location', 'rating', 'bean_type', 'broad_bean_origin']
sns.set(rc={'figure.figsize':(16,8.5)})
temp = df.groupby('company_location')['rating'].mean().reset_index()
sns.barplot(x=temp['company_location'], y=df['rating'])
plt.xticks(rotation=90)
plt.show()
temp = df.groupby('broad_bean_origin')['rating'].mean().reset_index()
sns.barplot(x=temp['broad_bean_origin'], y=df['rating'])
plt.xticks(rotation=90)
sns.set(rc={'figure.figsize':(16,8.5)})
plt.show()
temp = df.copy()
temp = temp.groupby('rating')['cocoa_percent'].count().reset_index()
temp
sns.barplot(x=temp['rating'], y=temp['cocoa_percent'])
plt.show()

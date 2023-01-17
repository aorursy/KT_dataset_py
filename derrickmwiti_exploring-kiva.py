import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
kiva_loan = pd.read_csv('../input/kiva_loans.csv')
pd.isnull(kiva_loan).any()
kiva_loan.head()
kiva_loan.describe()
sns.countplot(data=kiva_loan, y='sector')
kiva_loan['country'].nunique()
kiva_loan['country'].unique()
loan_amount_by_nation = kiva_loan.groupby('country')['funded_amount'].sum().reset_index()
loan_amount_by_nation.columns = ['Country','Total Amount']
loan_df = loan_amount_by_nation.sort_values(by='Total Amount',ascending=False)
loan_df.head(10)
plt.figure(figsize=(12,6))
sns.barplot(data=loan_df.head(10),x='Country',y='Total Amount')
plt.figure(figsize=(12,6))
sns.barplot(data=loan_df.tail(10),x='Country',y='Total Amount')
phil = kiva_loan[kiva_loan['country'] == 'Philippines']
sns.countplot(data=phil,y='sector')
ke = kiva_loan[kiva_loan['country'] == 'Kenya']
sns.countplot(data=ke,y='sector')
region_count = ke.groupby('region')['funded_amount'].sum().reset_index()
region_count.columns= ['Region','Total Amount Funded']
region_count.sort_values(by='Total Amount Funded',ascending=False).head(10)

piv = ke.pivot_table(columns=['region','sector'],aggfunc='count')
piv
from wordcloud import WordCloud, STOPWORDS
corpus = ' '.join(kiva_loan['activity'])
corpus = corpus.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)
plt.figure(figsize=(12,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
from wordcloud import WordCloud, STOPWORDS
corpus = ' '.join(kiva_loan['use'].astype(str))
corpus = corpus.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)
plt.figure(figsize=(12,15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

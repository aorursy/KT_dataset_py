#import files and Print 'Setup Complete' to verify import
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pandas import Series, DataFrame

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
print("Setup Complete")
#Import file from Directory and check that it loaded properly
df = pd.read_csv('../input/chocolate-bar-ratings/flavors_of_cacao.csv')
df.head()
#rename columns for easier data manipulation
df.columns = ['company', 'origin', 'ref', 'review_date', 'cocoa_percent', 'company_location', 'rating', 'bean_type', 'bean_origin']
#check to confirm
df.head()
#remove percentage symbol from 'cocoa_percent' for easier python syntax and change the column to a float
df['cocoa_percent']=(df['cocoa_percent']).str.replace('%', '')
df['cocoa_percent']=(df['cocoa_percent']).astype(float)
#check to confirm
df.head()
df.head()
#Let's see a summary of of all the columns in the table
print("Table 1: Summary of Statistical Measurements")
df.describe(include='all').T
#Let's analyze some of these findings with visualizations
#To compare the number of users that rated chocolate bars with their ratings
sns.countplot(x='rating', data=df)
plt.xlabel('Rating')
plt.ylabel('Number of consumers')
plt.title('Number of consumers that rated Chocolate Bars')
print('Fig 1: Chocolate bar ratings')
#Now, Let's find out how the percentage of cocoa affects the rating of a chocolate bar
sns.lmplot(x='rating', y='cocoa_percent', fit_reg=False,scatter_kws={"alpha":0.3,"s":100}, data=df)
plt.xlabel('Rating')
plt.ylabel('Percentage of Cocoa in Chocolate')
plt.title('Relationship between percentage of Cocoa and Chocolate rating')
print('Fig 2: Chocolate bar ratings and Cocoa Percentage')
#Let's see how the chocolate bars are spread based on their cocoa percentage.
#sns.countplot(x='cocoa_percent', data=df)
sns.distplot(a=df['cocoa_percent'], hist=True, kde=False)
plt.xlabel('Percentage of Cocoa')
plt.ylabel('Number of chocolate bars')
plt.title('Count of chocolate bars and their percentage of cocoa')
print('Fig 3: Chocolate bar cocoa percentage')
# Let's see the top 5 companies in terms of reviews
companies=df['company'].value_counts().index.tolist()[:5]
satisfactory={} # empty dictionary
for j in companies:
    c=0
    b=df[df['company']==j]
    br=b[b['rating']>=3] # rating more than 4
    for i in br['rating']:
        c+=1
        satisfactory[j]=c    
print(satisfactory)
#Visualizing the data above
li=satisfactory.keys()
plt.figure(figsize=(10,5))
plt.bar(range(len(satisfactory)), satisfactory.values())
plt.xticks(range(len(satisfactory)), list(li))
plt.xlabel('Company')
plt.ylabel('Number of chocolate bars')
plt.title("Top 5 Companies with Chocolate Bars Rating above 3.0")
print('Fig 4: Best chocolate companies')
d2 = df.sort_values('cocoa_percent', ascending=False).head(6)
plt.figure(figsize=(15, 4))
sns.barplot(x='company', y='cocoa_percent', data=d2)
plt.xlabel("Chocolate Company")
plt.ylabel("Cocoa Percentage")
plt.title("Top 5 Companies in terms of Cocoa Percentage")
print('Fig 5: Top companies with Cocoa Percentage')
plt.figure(figsize=(15, 4))
ax = sns.lineplot(x='review_date', y='cocoa_percent', data=df)
plt.xlabel("Year of Review")
plt.ylabel("Cocoa Percentage")
plt.title("Cocoa Percentage patterns over the years")
print('Fig 6: Pattern of Cocoa percentage over the years')
plt.figure(figsize=(15, 4))
ax = sns.lineplot(x='review_date', y='rating', data=df)
plt.xlabel("Year of Review")
plt.ylabel("Ratings")
plt.title("Rating patterns over the years")
print('Fig 7: Pattern of Chocolate bar ratings over the years')
#To confirm this speculation, let's see
cocoa_seventy=df[df['cocoa_percent' ] == 70.0]
cocoa_one_hundred=df[df['cocoa_percent' ] == 100.0] 
cocoa_seventy.count()
cocoa_one_hundred.count()
sns.countplot(x='rating', data=cocoa_seventy, color='orange')
sns.countplot(x='rating', data=cocoa_one_hundred, color='red')
print('Fig 8: Ratings of Chocolate Bars with 70% & 100% Cocoa')

print ('Top Chocolate Producing Countries in the World\n')
country=list(df['company_location'].value_counts().head(10).index)
choco_bars=list(df['company_location'].value_counts().head(10))
prod_ctry=dict(zip(country,choco_bars))
print(df['company_location'].value_counts().head(10))
#Let's visualize this

plt.figure(figsize=(10,5))
plt.hlines(y=country,xmin=0,xmax=choco_bars)
plt.plot(choco_bars, country)
plt.xlabel('Number of chocolate bars')
plt.ylabel('Country')
plt.title("Top Chocolate Producing Countries in the World")
print('Fig 8: Countries with highest chocolate producing companies')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
from pandas import Series, DataFrame

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
df = pd.read_csv('../input/flavors_of_cacao.csv')
df.head()
df.columns = ['Company', 'Origin', 'REF', 'Review_Date', 'Cocoa_Percent', 'Company Location', 
              'Rating', 'Bean_Type', 'Bean_Org']
# I will be changing the column names in order to simplify data manipulation 
df['Company']=df['Company'].astype('category')
df['Cocoa_Percent']=(df['Cocoa_Percent']).str.replace('%', ' ')
df['Cocoa_Percent']=(df['Cocoa_Percent']).astype(float)
df.head()
print("Table 1: Summary of Statistical Measurements")
df.describe(include='all').T
sns.countplot(x='Rating', data=df)
plt.xlabel('Rating')
plt.ylabel('Count of Users')
plt.title('Number of Users that Rated Chocolate Bars')
print('Fig 1: Count of Chocolate Bar Ratings')
sns.distplot(a=df['Cocoa_Percent'], hist=True,kde=False,rug=False, color='darkgreen') 
#count of average percentage of chocolate bars 
plt.xlabel('Cocoa Percent')
plt.ylabel('Count')
plt.title('Count of Percentage of Cooca in Chocolate Bars')
print('Fig 2: Count of Cocoa Percentage')
plot_data = [go.Scatter(x=df['Cocoa_Percent'].tolist(),y=df["Rating"], mode='markers', marker=dict(color='lightblue',
        line=dict(color='darkblue', width=1,),symbol='circle',size=16,))]

plot_layout = go.Layout(title="Rating of Chocolate Bars by Cocoa Percent", xaxis=dict(title='Cocoa Percent'), 
                        yaxis=dict(title='Rating'))

fig = go.Figure(data=plot_data, layout=plot_layout)

iplot(fig)
print("Fig 3: Chocolate Bar Rating Based on Cocoa Percent")
origin_max = df.groupby(['Bean_Org'])['Rating'].max()
max_desc=origin_max.sort_values(ascending=False)
top_20_bean=max_desc[:21]
top_20_bean.head(21)
data = top_20_bean.head(21)
data2 = data = {'Bean Origin': ['Venezuela ', 'Venezuela', 'Guatemala', 'Sao Tome & Principe', 'Sao Tome', 
                                'Peru, Dom. Rep', 'Peru', 'Papua New Guinea', 'Nicaragua', 
                                'Madagascar', 'Jamaica', 'Indonesia', 'Haiti', 'Guat., D.R., Peru, Mad., PNG', 
                                'St. Lucia', 'Gre., PNG, Haw., Haiti, Mad', 'Ghana', 'Ecuador', 
                                'Dominican Republic', 'Dom. Rep., Madagascar'],
        'Rating': [ 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                   4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]}
table = pd.DataFrame(data2)
table = pd.DataFrame(data, columns=['Bean Origin', 'Rating'],
                      index=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
print("Table 1: Top 20 Countries Where Beans Originate")
table
df['Company Location'].value_counts().head(15)
df['Company Location'].value_counts().head(15).plot('barh')
plt.xlabel('Count of Chocolate Bars')
plt.ylabel('Countries of Distribution')
print("Fig 4: 15 Companies with the Highest Chocolate Vendors")
cocoa_one_hundred=df[df['Cocoa_Percent' ] == 100.0] 
#how many chocolate bars have a 100 percent rating 
cocoa_one_hundred.count()
sns.countplot(x='Rating', data=cocoa_one_hundred, color='red')
print("Fig 5: Ratings of Chocolate Bars with 100% Cocoa")
cocoa_seventy=df[df['Cocoa_Percent' ] == 70.0]
cocoa_seventy.count()
sns.countplot(x='Rating', data=cocoa_seventy, color='orange')
print("Fig 6: Ratings of Chocolate Bars with 70% Cocoa")

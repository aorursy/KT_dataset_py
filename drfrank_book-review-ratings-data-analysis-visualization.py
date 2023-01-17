import numpy as np 
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt
# Plotly Libraris
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Users
u_cols = ['user_id', 'location', 'age']
users = pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1',low_memory=False)

#Books
i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']
items = pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep=';', names=i_cols, encoding='latin-1',)

#Ratings
r_cols = ['user_id', 'isbn', 'rating']
ratings = pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1',low_memory=False)
users.head(2)
items.head(2)
ratings.head(2)
users = users.drop(users.index[0])
items = items.drop(items.index[0])
ratings = ratings.drop(ratings.index[0])

df = pd.merge(users, ratings, on='user_id')
df = pd.merge(df, items, on='isbn')

df.head(2)
df.info()
df.columns
df.shape
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df[df.duplicated() == True]
location = df.location.str.split(', ', n=2, expand=True)
location.columns=['city', 'state', 'country']

df['city'] = location['city']
df['state'] = location['state']
df['country'] = location['country']
df['rating'] = df['rating'].astype('int32')
df['user_id'] = df['user_id'].astype('int32')

location = users.location.str.split(', ', n=2, expand=True)
location.columns=['city', 'state', 'country']

users['city'] = location['city']
users['state'] = location['state']
users['country'] = location['country']

df = df.drop(['img_s', 'img_m', 'img_l'], axis=1)

df.head(2)
df.describe().T
users.head(2)
ax = sns.countplot(x="rating", data=df)
fig = go.Figure(data=[go.Histogram(x=df['rating'],  # To get Horizontal plot ,change axis - 
                                  marker_color="Crimson",
                       xbins=dict(
                      start=0, #start range of bin
                      end=10,  #end range of bin
                      size=1   #size of bin
                      ))])
fig.update_layout(title="Distribution Of Rating",xaxis_title="Rating",yaxis_title="Counts",title_x=0.5)
fig.show()
indexs=df[(df['rating']==0)]['user_id'].index
df_no_0=df.drop(indexs)
df_no_0.head(2)
ds = df['rating'].value_counts().to_frame().reset_index()
ds.columns = ['value', 'count']
ds=ds.drop([0])

fig = go.Figure(go.Bar(
    y=ds['value'],x=ds['count'],orientation="h",
    marker={'color': ds['count'], 
    'colorscale': 'sunsetdark'},  
    text=ds['count'],
    textposition = "outside",
))
fig.update_layout(title_text='Rating Count',xaxis_title="Value",yaxis_title="Count",title_x=0.5)
fig.show()
ds = df['rating'].value_counts().to_frame().reset_index()
ds.columns = ['value', 'count']
ds=ds.drop([0])

fig = go.Figure(go.Bar(
    x=ds['value'],y=ds['count'],
    marker={'color': ds['count'], 
    'colorscale': 'sunsetdark'},  
    text=ds['count'],
    textposition = "outside",
))
fig.update_layout(title_text='Rating Distribution',xaxis_title="Value",yaxis_title="Count",title_x=0.5)
fig.show()
ds = df['rating'].value_counts().to_frame().reset_index()
ds.columns = ['value', 'count']
ds=ds.drop([0])

ds['value'] = ds['value'].astype('int32')
ds['multiplication']= ds['value']* ds['count']
x=ds.apply(np.sum, axis=0).reset_index() 
x=x.drop(['index'],axis=1)
y = x[1:3].values
rate=y[1]/y[0]
df_rate = pd.DataFrame(columns=['Count', 'Total Rating', 'AVG Rate'])
df_rate = df_rate.append({'Total Count': 383842, 'Total Rating': 2927454, 'AVG Rate': 7.62}, ignore_index=True)

colors=['DarkKhaki','Coral','LightSalmon']
    
fig = go.Figure(data=[go.Table(header=dict(values=['Total Rating Count', 'Total Rating Sum ','Average Rating Of Books '],
                                          line_color='white', fill_color='gray',
                                  align='center',font=dict(color='white', size=12)
                                          ),
                               
                 cells=dict( values=[df_rate['Total Count'], df_rate['Total Rating'],df_rate['AVG Rate']],
                           line_color=colors, fill_color=colors,
                           align='center', font=dict(color='black', size=13))
                              )])
                      

fig.update_layout(
    autosize=False,
    width=500,
    height=300,
   )    

fig.show()

Count_Rating=383842
Total_Rating=2927454
avg_rate_rating=7.62
fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  Count_Rating,
    domain = {'x': [0.25, 1], 'y': [0.4, 0.6]},
    title = {'text': "Count Ratings",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 400000]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = Total_Rating,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.9]},
    title = {'text': "Sum Of Ratings",'font':{'color': 'black','size':15}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,3000000]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_rate_rating,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.3]},
    title = {'text' :"Average Rating Of Books",'font':{'color': 'black','size':15}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,10]},
        'bar': {'color': "Gold"}}
))
fig.update_layout(title=" Ratings Of Books ",title_x=0.5)
fig.show()
users_city=users.city.value_counts()[0:20].reset_index().rename(columns={'index':'city','city':'count'})

fig = go.Figure(go.Bar(
    x=users_city['city'],y=users_city['count'],
    marker={'color': users_city['count'], 
    'colorscale': 'Viridis'},  
    text=users_city['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 20 Users City',xaxis_title="City",yaxis_title="Count",title_x=0.5)
fig.show()
users_country=users.country.value_counts()[0:10].reset_index().rename(columns={'index':'country','country':'count'})

fig = go.Figure(go.Bar(
    x=users_country['country'],y=users_country['count'],
    marker={'color': users_country['count'], 
    'colorscale': 'inferno'},  
    text=users_country['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10 Country',xaxis_title="Country",yaxis_title="Count",title_x=0.5)
fig.show()
fig = go.Figure(go.Box(y=users['age'],name="Age")) # to get Horizonal plot change axis 
fig.update_layout(title="Distribution Of Age ",title_x=0.5)
fig.show()
fig = px.histogram(
    df, 
    "age", 
    nbins=100, 
    title='Age distribution', 
    width=700,
    height=600
)
fig.show()
users2=users.copy()
users2=users2.dropna()
users2['age'] = users2['age'].astype('int32')
users2.isnull().sum()


users2['age_category']=np.where((users2['age']<19),"below 19",
                                 np.where((users2['age']>18)&(users2['age']<=30),"19-30",
                                    np.where((users2['age']>30)&(users2['age']<=50),"31-50",
                                                np.where(users2['age']>50,"Above 50","NULL"))))

age=users2['age_category'].value_counts().to_frame().reset_index().rename(columns={'index':'age_category','age_category':'Count'})


fig = go.Figure(data=[go.Scatter(
    x=age['age_category'], y=age['Count'],
    mode='markers',
    marker=dict(
        color=age['Count'],
        size=age['Count']*0.002,
        showscale=True
    ))])

fig.update_layout(title='Age Distribution',xaxis_title="Age Category",yaxis_title="Number Of People",title_x=0.5)
fig.show()
df_book_name=df_no_0.book_title.value_counts()[0:10].reset_index().rename(columns={'index':'book_title','book_title':'count'})


colors=['cyan','royalblue','blue','darkblue',"darkcyan",'Brown','Coral','OrangeRed','SaddleBrown','Tomato']
fig = go.Figure([go.Pie(labels=df_book_name['book_title'], values=df_book_name['count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Most Reviewed Books ",title_x=0.3)
fig.show()
df_book_name=df_no_0.book_title.value_counts()[0:10].reset_index().rename(columns={'index':'book_title','book_title':'count'})
df_book_name

fig = go.Figure(go.Bar(
    y=df_book_name['book_title'],x=df_book_name['count'],orientation="h",
    marker={'color': df_book_name['count'], 
    'colorscale': 'darkmint'},  
    text=df_book_name['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10  Reviewed Books',xaxis_title=" Rating Count",yaxis_title="Books Name",title_x=0.6)

fig.update_layout(
    autosize=False,
    width=920,
    height=700,
   )
fig.show()
df_book_name_2=df_no_0.book_title.value_counts()[0:5].reset_index().rename(columns={'index':'book_title','book_title':'count'})

book_rate1 = df_no_0[df_no_0['book_title']=='The Lovely Bones: A Novel']['rating'].sum()
book_rate1=book_rate1/707

book_rate2 = df_no_0[df_no_0['book_title']=='Wild Animus']['rating'].sum()
book_rate2=book_rate2/581

book_rate3 = df_no_0[df_no_0['book_title']=='The Da Vinci Code']['rating'].sum()
book_rate3=book_rate3/494

book_rate4 = df_no_0[df_no_0['book_title']=='The Secret Life of Bees']['rating'].sum()
book_rate4=book_rate4/406


book_rate5 = df_no_0[df_no_0['book_title']=='The Nanny Diaries: A Novel']['rating'].sum()
book_rate5=book_rate5/393

df_rate = pd.DataFrame(columns=['book_title', 'Rating', ])
df_rate = df_rate.append({'book_title':'The Lovely Bones: A Novel' , 'Rating': book_rate1}, ignore_index=True)
df_rate = df_rate.append({'book_title':'Wild Animus' , 'Rating': book_rate2}, ignore_index=True)
df_rate = df_rate.append({'book_title':'The Da Vinci Code' , 'Rating': book_rate3}, ignore_index=True)
df_rate = df_rate.append({'book_title':'The Secret Life of Bees' , 'Rating': book_rate4}, ignore_index=True)
df_rate = df_rate.append({'book_title':'The Nanny Diaries: A Novel' , 'Rating': book_rate5}, ignore_index=True)


fig = make_subplots(rows=2, cols=1,
                   subplot_titles=("Top 5 Reviewed Books",
                                   "Average Rate Of Top 5 Reviewed Books ",))  # Subplot titles

fig.add_trace(go.Bar(
    y=df_book_name_2['book_title'],x=df_book_name_2['count'],orientation="h",
    marker={'color': df_book_name_2['count'], 
    'colorscale': 'darkmint'},  
    text=df_book_name_2['count'],
    textposition = "outside"),
    row=1, col=1           
)
fig.add_trace(go.Bar(
    y=df_rate['book_title'],x=df_rate['Rating'],orientation="h",
    marker={'color': df_rate['Rating'], 
    'colorscale': 'solar'},  
    text=df_rate['Rating'],
    textposition = "outside"),
    row=2, col=1           
)


fig.update_layout(height=900, width=900,title ="Books Ratings",title_x=0.5)
fig.show()

df_books_sum=df_no_0.groupby(by =['book_title'])['rating'].sum().to_frame().reset_index().rename(columns={'book_title':'book_title','rating':'total_rating'})
df_books_sum=df_books_sum.sort_values(by='total_rating', ascending=False)
#df_books_sum
df_books_count=df_no_0.groupby(by =['book_title'])['rating'].count().to_frame().reset_index().rename(columns={'book_title':'book_title','rating':'count_rating'})
df_books_count=df_books_count.sort_values(by='count_rating', ascending=False)
#df_books_count


df_books_merge = pd.merge(df_books_sum, df_books_count, on='book_title')[0:50]
df_books_merge['rate']=df_books_merge['total_rating']/df_books_merge['count_rating']
df_books_merge=df_books_merge.sort_values(by='rate', ascending=False)
#df_books_merge
df_books_merge1=df_books_merge[df_books_merge['count_rating']>75]
df_books_merge1=df_books_merge1.sort_values(by='rate', ascending=False)[0:10]
df_books_merge1


fig = make_subplots(rows=2, cols=1,
                   subplot_titles=(" Best Rated Books",
                                   " Best Rated Books Count Rating ",))  # Subplot titles

fig.add_trace(go.Bar(
    y=df_books_merge1['book_title'],x=df_books_merge1['count_rating'],orientation="h",
    marker={'color': df_books_merge1['count_rating'], 
    'colorscale': 'thermal'},  
    text=df_books_merge1['count_rating'],
    textposition = "outside"),
    row=2, col=1           
)
fig.add_trace(go.Bar(
    y=df_books_merge1['book_title'],x=df_books_merge1['rate'],orientation="h",
    marker={'color': df_books_merge1['rate'], 
    'colorscale': 'jet'},  
    text=df_books_merge1['rate'],
    textposition = "outside"),
    row=1, col=1           
)


fig.update_layout(height=1100, width=1100,title ="Best Books Ratings",title_x=0.5)
fig.show()
df_books_year=items.year_of_publication.value_counts()[0:10].reset_index().rename(columns={'index':'year_of_publication','year_of_publication':'count'})

fig = go.Figure(go.Bar(
    x=df_books_year['year_of_publication'],y=df_books_year['count'],
    marker={'color': df_books_year['count'], 
    'colorscale': 'inferno'},  
    text=df_books_year['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10 Number Of Book Works Published By Years ',xaxis_title="Years",yaxis_title="Count",title_x=0.5)
fig.show()
df_books_publisher=items.publisher.value_counts()[0:10].reset_index().rename(columns={'index':'publisher','publisher':'count'})

fig = go.Figure(go.Bar(
    x=df_books_publisher['publisher'],y=df_books_publisher['count'],
    marker={'color': df_books_publisher['count'], 
    'colorscale': 'inferno'},  
    text=df_books_publisher['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10 Publisher',xaxis_title="Publisher",yaxis_title="Count",title_x=0.5)
fig.show()
df_sk=df_no_0[(df_no_0['book_author']=='Stephen King')]

df_SK_sum=df_sk.groupby(by =['book_title'])['rating'].sum().to_frame().reset_index().rename(columns={'book_title':'book_title','rating':'total_rating'})
df_SK_sum=df_SK_sum.sort_values(by='total_rating', ascending=False)
df_SK_count=df_sk.groupby(by =['book_title'])['rating'].count().to_frame().reset_index().rename(columns={'book_title':'book_title','rating':'count_rating'})
df_SK_count=df_SK_count.sort_values(by='count_rating', ascending=False)

#df_SK_sum
#df_SK_count
df_SK_merge = pd.merge(df_SK_sum, df_SK_count, on='book_title')[0:5]
df_SK_merge['rate']=df_SK_merge['total_rating']/df_SK_merge['count_rating']
df_SK_merge

fig = make_subplots(rows=2, cols=1,
                   subplot_titles=("Top 5 Reviewed Stephen King Books",
                                   "Average Rate Of Top 5 Reviewed Stephen King Books ",))  # Subplot titles

fig.add_trace(go.Bar(
    y=df_SK_merge['book_title'],x=df_SK_merge['count_rating'],orientation="h",
    marker={'color': df_SK_merge['count_rating'], 
    'colorscale': 'thermal'},  
    text=df_SK_merge['count_rating'],
    textposition = "outside"),
    row=1, col=1           
)
fig.add_trace(go.Bar(
    y=df_SK_merge['book_title'],x=df_SK_merge['rate'],orientation="h",
    marker={'color': df_SK_merge['rate'], 
    'colorscale': 'jet'},  
    text=df_SK_merge['rate'],
    textposition = "outside"),
    row=2, col=1           
)


fig.update_layout(height=900, width=900,title ="Stephen King Books Ratings",title_x=0.5)
fig.show()
df_sk=df_no_0[(df_no_0['book_author']=='Stephen King')]

df_SK_sum=df_sk.groupby(by =['book_title'])['rating'].sum().to_frame().reset_index().rename(columns={'book_title':'book_title','rating':'total_rating'})
df_SK_sum=df_SK_sum.sort_values(by='total_rating', ascending=False)
df_SK_count=df_sk.groupby(by =['book_title'])['rating'].count().to_frame().reset_index().rename(columns={'book_title':'book_title','rating':'count_rating'})
df_SK_count=df_SK_count.sort_values(by='count_rating', ascending=False)


df_SK_merge = pd.merge(df_SK_sum, df_SK_count, on='book_title')
df_SK_merge['rate']=df_SK_merge['total_rating']/df_SK_merge['count_rating']
df_SK_merge=df_SK_merge.sort_values(by='rate', ascending=False)
df_SK_merge1=df_SK_merge[df_SK_merge['count_rating']>15]
df_SK_merge1=df_SK_merge1.sort_values(by='rate', ascending=False)[0:5]


fig = make_subplots(rows=2, cols=1,
                   subplot_titles=("Stephen King's 5 Best Rated Books",
                                   "Stephen King's 5 Best Rated Books Count Rating ",))  # Subplot titles

fig.add_trace(go.Bar(
    y=df_SK_merge1['book_title'],x=df_SK_merge1['count_rating'],orientation="h",
    marker={'color': df_SK_merge1['count_rating'], 
    'colorscale': 'thermal'},  
    text=df_SK_merge1['count_rating'],
    textposition = "outside"),
    row=2, col=1           
)
fig.add_trace(go.Bar(
    y=df_SK_merge1['book_title'],x=df_SK_merge1['rate'],orientation="h",
    marker={'color': df_SK_merge1['rate'], 
    'colorscale': 'jet'},  
    text=df_SK_merge1['rate'],
    textposition = "outside"),
    row=1, col=1           
)


fig.update_layout(height=900, width=900,title ="Stephen King Books Ratings",title_x=0.5)
fig.show()
df_sk=df_no_0[(df_no_0['book_author']=='Stephen King')]

df_SK_country=df_sk.groupby(by =['country'])['rating'].count().to_frame().reset_index().rename(columns={'country':'country','rating':'count'})

df_SK_country=df_SK_country.sort_values(by='count', ascending=False)[0:5]


fig = go.Figure(go.Bar(
    x=df_SK_country['country'],y=df_SK_country['count'],
    marker={'color': df_SK_country['count'], 
    'colorscale': 'haline'},  
    text=df_SK_country['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 5 Stephen King Fan Country',xaxis_title="Country",yaxis_title="Count",title_x=0.5)
fig.show()
df_sk=df_no_0[(df_no_0['book_author']=='Stephen King')]

df_SK_city=df_sk.groupby(by =['city'])['rating'].count().to_frame().reset_index().rename(columns={'city':'city','rating':'count'})

df_SK_city=df_SK_city.sort_values(by='count', ascending=False)[0:5]


fig = go.Figure(go.Bar(
    x=df_SK_city['city'],y=df_SK_city['count'],
    marker={'color': df_SK_city['count'], 
    'colorscale': 'haline'},  
    text=df_SK_city['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 5 Stephen King Fan City',xaxis_title="City",yaxis_title="Count",title_x=0.5)
fig.show()
df_sk=items[(items['book_author']=='Stephen King')]


df_sk_years=df_sk.year_of_publication.value_counts()[0:10].reset_index().rename(columns={'index':'year_of_publication','year_of_publication':'count'})

fig = go.Figure(go.Bar(
    x=df_sk_years['year_of_publication'],y=df_sk_years['count'],
    marker={'color': df_sk_years['count'], 
    'colorscale': 'speed'},  
    text=df_sk_years['count'],
    textposition = "outside",
))
fig.update_layout(title_text='Stephen King Top 10 Number Of Book Works Published By Years',xaxis_title="Publisher",yaxis_title="Count")
fig.show()

df_sk=items[(items['book_author']=='Stephen King')]
df_sk

df_sk_publisher=df_sk.publisher.value_counts()[0:10].reset_index().rename(columns={'index':'publisher','publisher':'count'})

fig = go.Figure(go.Bar(
    x=df_sk_publisher['publisher'],y=df_sk_publisher['count'],
    marker={'color': df_sk_publisher['count'], 
    'colorscale': 'balance'},  
    text=df_sk_publisher['count'],
    textposition = "outside",
))
fig.update_layout(title_text='Stephen King Top 10 Publisher',xaxis_title="Publisher",yaxis_title="Count",title_x=0.5)
fig.show()

df_sk=df_no_0[(df_no_0['book_author']=='Stephen King')]

df_SK_sum=df_sk.groupby(by =['book_title'])['rating'].sum().to_frame().reset_index().rename(columns={'book_title':'book_title','rating':'total_rating'})
df_SK_sum=df_SK_sum.sort_values(by='total_rating', ascending=False)
df_SK_count=df_sk.groupby(by =['book_title'])['rating'].count().to_frame().reset_index().rename(columns={'book_title':'book_title','rating':'count_rating'})
df_SK_count=df_SK_count.sort_values(by='count_rating', ascending=False)

df_SK_merge = pd.merge(df_SK_sum, df_SK_count, on='book_title')
df_SK_merge['rate']=df_SK_merge['total_rating']/df_SK_merge['count_rating']

Count_Rating=df_SK_merge['count_rating'].sum()     
Total_Rating=df_SK_merge['total_rating'].sum()     
avg_rate_rating=Total_Rating/Count_Rating

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  Count_Rating,
    domain = {'x': [0.25, 1], 'y': [0.4, 0.6]},
    title = {'text': "Stephen King Sum Of Count Ratings",'font':{'color': 'black','size':12}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 5000]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = Total_Rating,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.9]},
    title = {'text': "Stephen King Sum Of Ratings",'font':{'color': 'black','size':12}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,40000]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_rate_rating,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.3]},
    title = {'text' :"Stephen King Average Rating Of Books",'font':{'color': 'black','size':12}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,10]},
        'bar': {'color': "Gold"}}
))
fig.update_layout(title="Stephen King Ratings Of Books ",title_x=0.5)
fig.show()
df_author_name=df_no_0.book_author.value_counts()[0:10].reset_index().rename(columns={'index':'book_author','book_author':'count'})

fig = go.Figure(go.Bar(
    x=df_author_name['book_author'],y=df_author_name['count'],
    marker={'color': df_author_name['count'], 
    'colorscale': 'cividis'},  
    text=df_author_name['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10 Reviewed Book Author',xaxis_title="Author",yaxis_title="Count",title_x=0.5)
fig.show()
df_user=df_no_0.user_id.value_counts()[0:10].reset_index().rename(columns={'index':'user_id','user_id':'count'})
df_user['user_id']='User '+ df_user['user_id'].astype('str')

fig = go.Figure(go.Bar(
    x=df_user['user_id'],y=df_user['count'],
    marker={'color': df_user['count'], 
    'colorscale': 'blackbody'},  
    text=df_user['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 10 Bookworms Users',xaxis_title="Users",yaxis_title="Count",title_x=0.5)
fig.show()
df_user=df_no_0.user_id.value_counts()[0:10].reset_index().rename(columns={'index':'user_id','user_id':'count'})
df_user['user_id']='User '+df_user['user_id'].astype('str') 

df_user_11676=df_no_0[(df_no_0['user_id']==11676)]

Count_Rating=df_user_11676['rating'].count()    
Total_Rating=df_user_11676['rating'].sum()    
avg_rate_rating=Total_Rating/Count_Rating

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value =  Count_Rating,
    domain = {'x': [0.25, 1], 'y': [0.4, 0.6]},
    title = {'text': "User 11676 Sum Of Count Ratings",'font':{'color': 'black','size':12}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 8000]},
        'bar': {'color': "blue"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = Total_Rating,
    domain = {'x': [0.25, 1], 'y': [0.7, 0.9]},
    title = {'text': "User 11676 Sum Of Ratings",'font':{'color': 'black','size':12}},
    number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,60000]},
        'bar': {'color': "cyan"}}))

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = avg_rate_rating,
    domain = {'x': [0.25, 1], 'y': [0.1, 0.3]},
    title = {'text' :"User 11676 Average Rating Of Books",'font':{'color': 'black','size':12}},
     number={'font':{'color': 'black'}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None,10]},
        'bar': {'color': "Gold"}}
))
fig.update_layout(title="User 11676 Ratings Of Books ",title_x=0.5)
fig.show()
df_book_name=items.book_title.value_counts()[0:25].reset_index().rename(columns={'index':'book_title','book_title':'count'})

fig = go.Figure(go.Bar(
    x=df_book_name['book_title'],y=df_book_name['count'],
    marker={'color': df_book_name['count'], 
    'colorscale': 'earth'},  
    text=df_book_name['count'],
    textposition = "outside",
))
fig.update_layout(title_text=' Top 25 Same Book Title ',xaxis_title="Books Name",yaxis_title="Count",title_x=0.5)
fig.show()
df_Dracula=items[items['book_title']=='Dracula']

colors=['DarkKhaki','Coral','DarkSalmon','FireBrick']
    
fig = go.Figure(data=[go.Table(header=dict(values=['Book Title', 'Book Author','Year Of Publication','Publisher'],
                                          line_color='white', fill_color='gray',
                                  align='center',font=dict(color='white', size=12)
                                          ),
                               
                 cells=dict( values=[df_Dracula['book_title'], df_Dracula['book_author'],df_Dracula['year_of_publication'],df_Dracula['publisher']],
                           line_color=colors, fill_color=colors,
                           align='center', font=dict(color='black', size=13))
                              )])
                      

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
   )    

fig.show()
df_author_name=items.book_author.value_counts()[0:20].reset_index().rename(columns={'index':'book_author','book_author':'count'})
df_author_name=items.book_author.value_counts()[0:20].reset_index().rename(columns={'index':'book_author','book_author':'count'})
df_author_name

fig = go.Figure(go.Bar(
    x=df_author_name['book_author'],y=df_author_name['count'],
    marker={'color': df_author_name['count'], 
    'colorscale': 'temps'},  
    text=df_author_name['count'],
    textposition = "outside",
))
fig.update_layout(title_text='Number Of Works By 20 Authors With The Most Works Published ',xaxis_title="Author Name",yaxis_title="Count")
fig.show()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import grangercausalitytests
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
games = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
games
sns.set_style('whitegrid')
fig,(ax1) = plt.subplots(figsize=(20,11))
plt.title('Number of different games per game platforms')
sns.countplot(x='Platform', data=games, ax=ax1)
plt.show()
top_10_publishers = games.Publisher.value_counts().sort_values(ascending=False).head(10)
fig = px.pie(top_10_publishers,
             values= top_10_publishers.values,
             names= top_10_publishers.index,
             title='Top 10 Games Publishers')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
def top_sales(df,region):
    if region == 'JP_Sales':
        japan = games.groupby('Name')['JP_Sales'].sum().reset_index().sort_values('JP_Sales', ascending=False).head(10)
        return japan
    elif region == 'EU_Sales':
        eu = games.groupby('Name')['EU_Sales'].sum().reset_index().sort_values('EU_Sales', ascending=False).head(10)
        return eu
    elif region == 'NA_Sales':
        na = games.groupby('Name')['NA_Sales'].sum().reset_index().sort_values('NA_Sales', ascending=False).head(10)
        return na
    elif region == 'Global_Sales':
        globe = games.groupby('Name')['Global_Sales'].sum().reset_index().sort_values('Global_Sales', ascending=False).head(10)
        return globe
    else:
        other = games.groupby('Name')['Other_Sales'].sum().reset_index().sort_values('Other_Sales', ascending=False).head(10)
        return other
    
top10_JP_sales = top_sales(games, 'JP_Sales')
top10_EU_sales = top_sales(games, 'EU_Sales')
top10_NA_sales = top_sales(games, 'NA_Sales')
top10_Global_sales = top_sales(games, 'Global_Sales')
top10_Other_sales = top_sales(games, 'Other_Sales')
fig = go.Figure(
    data=[go.Bar(x=top10_JP_sales.Name, y=top10_JP_sales.JP_Sales)],
    layout_title_text = 'Top 10 Sales Games In Japan'
)
fig.show()
fig = go.Figure(
    data = [go.Bar(x=top10_EU_sales.Name, y=top10_EU_sales.EU_Sales)],
    layout_title_text = 'Top 10 Sales Games In EU'
)
fig.show()
fig = go.Figure(
    data = [go.Bar(x=top10_NA_sales.Name, y=top10_NA_sales.NA_Sales)],
    layout_title_text = 'Top 10 Sales Games In NA'
)
fig.show()
fig = go.Figure(
    data = [go.Bar(x=top10_Global_sales.Name, y=top10_Global_sales.Global_Sales)],
    layout_title_text = 'Top 10 Sales Games Global'
)
fig.show()
fig = go.Figure(
    data = [go.Bar(x=top10_Other_sales.Name, y=top10_Other_sales.Other_Sales)],
    layout_title_text = 'Top 10 Sales Games Other'
)
fig.show()
publication_distr = games.Year.value_counts().sort_values(ascending=False)
fig = px.bar(publication_distr, 
             x=publication_distr.index, 
             y=publication_distr.values,
             title= "Year of Games Releasing & Distribution")
fig.show()
top10_sales_games = games.groupby(['Name','Year'])['Global_Sales'].sum().reset_index().sort_values('Global_Sales', ascending=False).head(10)
fig = px.pie(top10_sales_games,
             values = top10_sales_games.Year,
             names = top10_sales_games.Name,
             title= 'Top 10 Worldwide Sales Games Globally')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
def genre_sales(df,region):
    if region == 'JP_Sales':
        japan = games.groupby('Genre')['JP_Sales'].sum().reset_index().sort_values('JP_Sales', ascending=False)
        return japan
    elif region == 'EU_Sales':
        eu = games.groupby('Genre')['EU_Sales'].sum().reset_index().sort_values('EU_Sales', ascending=False)
        return eu
    elif region == 'NA_Sales':
        na = games.groupby('Genre')['NA_Sales'].sum().reset_index().sort_values('NA_Sales', ascending=False)
        return na
    elif region == 'Global_Sales':
        globe = games.groupby('Genre')['Global_Sales'].sum().reset_index().sort_values('Global_Sales', ascending=False)
        return globe
    else:
        other = games.groupby('Genre')['Other_Sales'].sum().reset_index().sort_values('Other_Sales', ascending=False)
        return other
genre_sales_jp = genre_sales(games, 'JP_Sales')
genre_sales_eu = genre_sales(games, 'EU_Sales')
genre_sales_na = genre_sales(games, 'NA_Sales')
genre_sales_global = genre_sales(games, 'Global_Sales')
genre_sales_other = genre_sales(games, 'Other_Sales')
fig = go.Figure(data=[
    go.Bar(name='JP', x=genre_sales_jp.Genre, y=genre_sales_jp.JP_Sales),
    go.Bar(name='EU', x=genre_sales_eu.Genre, y=genre_sales_eu.EU_Sales),
    go.Bar(name='NA', x=genre_sales_na.Genre, y=genre_sales_na.NA_Sales),
    go.Bar(name='Rest of the world', x=genre_sales_other.Genre, y=genre_sales_other.Other_Sales)
])

fig.update_layout(barmode='group')
fig.show()
fig = go.Figure(data=[go.Pie(
     labels = genre_sales_global.Genre,
     values = genre_sales_global.Global_Sales,
     textinfo = 'percent+label',
     insidetextorientation='radial')
])
fig.show()
new_games_PC = games[(games.Year >= 2010) & (games.Platform == 'PC')].head(10)
new_games_PC
new_games_PS4 = games[(games.Year >= 2010) & (games.Platform == 'PS4')].head(10)
new_games_PS4
new_games_XOne = games[(games.Year >= 2010) & (games.Platform == 'XOne')].head(10)
new_games_XOne
fig = go.Figure(data=[
    go.Bar(name='Xbox One', x=new_games_XOne.Name, y=new_games_XOne.Global_Sales),
    go.Bar(name='PC', x=new_games_PC.Name, y=new_games_PC.Global_Sales),
    go.Bar(name='PS4', x=new_games_PS4.Name, y=new_games_PS4.Global_Sales)
])

fig.update_layout(barmode='group', title_text='Xbox / PC / PS4 TOP 10 Games Each Game Platform')
fig.show()
games
X = np.array(games.values[:,6:10], dtype=np.float64)
y = np.array(games.values[:,10], dtype=np.float64)
means, stds = np.mean(X, axis=0), np.std(X, axis=0)
X = (X - means) / stds
n = np.shape(X)[0]
ones = np.reshape(np.ones(n),(n,1))

X = np.hstack((X,ones))
def mserror(y,y_pred):
    return np.mean((y-y_pred)**2)
answer1 = mserror(y, np.median(y))
print(round(answer1,3))
def normal_equation(X, y):
    trans = np.transpose(X)
    inv = np.linalg.inv(trans.dot(X))
    pinv = inv.dot(trans)
    return pinv.dot(y)
norm_eq_weights = normal_equation(X, y)
print(norm_eq_weights)
column2 = np.ones((n, 1)) * norm_eq_weights[0]
answer2 = mserror(column2, y)
print(round(answer2,3))
def linear_prediction(X, w):
    return np.dot(X,w)
lin_pred = linear_prediction(X,norm_eq_weights)
answer3 = mserror(y,lin_pred)
print(answer3)
def stochastic_gradient_step(X,y,w,train_ind, eta=0.01):
    return w + 2 * eta/X.shape[0] * X[train_ind] * (y[train_ind] - linear_prediction(X[train_ind],w))
def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    weight_dist = np.inf
    
    w = w_init
    
    errors=[]
    
    iter_num=0
    
    np.random.seed(seed)
    
    while weight_dist > min_weight_dist and iter_num < max_iter:
        random_ind = np.random.randint(X.shape[0])
        w_new = stochastic_gradient_step(X,y,w,random_ind, eta)
        weight_dist = np.linalg.norm(w-w_new)
        w = w_new
        errors.append(mserror(y, linear_prediction(X,w)))
        iter_num += 1
        
    return w,errors
%%time
stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, np.zeros(X.shape[1]),max_iter=1e5)
%pylab inline
plot(range(len(stoch_errors_by_iter)), stoch_errors_by_iter)
xlabel('Iteration number')
ylabel('MSE')
plt.show()
stoch_grad_desc_weights
stoch_errors_by_iter[-1]
answer4 = mserror(y, linear_prediction(X, stoch_grad_desc_weights))
print(round(answer4, 3))
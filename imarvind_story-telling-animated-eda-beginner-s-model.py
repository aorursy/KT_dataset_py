import pandas as pd 
import numpy as np
import os

path = '/kaggle/input/ntt-data-global-ai-challenge-06-2020/'
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

%matplotlib inline

from matplotlib import animation, rc
from IPython.display import HTML, Image
rc('animation', html='html5')

!pip install bar_chart_race
import bar_chart_race as bcr
# Load Crude Oil data
data_1 = pd.read_csv(path+"Crude_oil_trend_From1986-10-16_To2020-03-31.csv")
print('Number of data points : ', data_1.shape[0])
print('Number of features : ', data_1.shape[1])
print('Features : ', data_1.columns.values)
data_1.head() # to print first 5 rows
# Converting Date format
data_1['Date'] = pd.to_datetime(data_1['Date'])
data_1['Date'].dtype
# Year wise data
# mean price 
data_1_year = data_1.groupby(data_1.Date.dt.year)['Price'].agg('mean').reset_index()
data_1_year.head()
# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim((1986, 2020))
ax.set_ylim(np.min(data_1_year.Price), np.max(data_1_year.Price)+1)
ax.set_xlabel('Year',fontsize = 14)
ax.set_ylabel('Price',fontsize = 14)
ax.set_title('Crude Oil Price Over the Years',fontsize = 18)
ax.xaxis.grid()
ax.yaxis.grid()
ax.set_facecolor('#000000') 
line, = ax.plot([], [], lw=4,color='green')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)


# animation function. This is called sequentially
def animate(i):
    d = data_1_year.iloc[:int(i+1)] #select data range
    x = d.Date
    y = d.Price
    line.set_data(x, y)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=40, repeat=True)
anim
# Week wise data 2020 Jan to April
mask = (data_1['Date'] > '2019-12-31') & (data_1['Date'] <= '2020-03-31')
data_2020 = data_1[mask]
# mean price 
data_2020_weekly = data_2020.set_index('Date').resample('W').mean().reset_index()
data_2020_weekly.head()
# First set up the figure, the axis, and the plot element we want to animate
import datetime
fig, ax = plt.subplots(figsize=(8,6))

ax.set_xlim([datetime.date(2020, 1, 2), datetime.date(2020, 3, 31)])
ax.set_ylim(np.min(data_2020_weekly.Price), np.max(data_2020_weekly.Price)+1)
ax.set_xlabel('Date',fontsize = 14)
ax.set_ylabel('Price',fontsize = 14)
ax.set_title('Crude Oil Price Per Week 2020 Jan - Mar',fontsize = 18)
ax.xaxis.grid()
ax.yaxis.grid()
ax.set_facecolor('#000000') 
line, = ax.plot([], [], lw=4,color='green')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)


# animation function. This is called sequentially
def animate(i):
    d = data_2020_weekly.iloc[:int(i+1)] #select data range
    x = d.Date
    y = d.Price
    line.set_data(x, y)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=14, repeat=True)
anim
# Load dataset
data_2 = pd.read_csv(path+"COVID-19_train.csv")
print('Number of data points : ', data_2.shape[0])
print('Number of features : ', data_2.shape[1])
data_2.head()
# Lets take only few countries
cols = ['Date','China_total_deaths','Germany_total_deaths','Spain_total_deaths',
        'France_total_deaths','UnitedKingdom_total_deaths','India_total_deaths',
       'Italy_total_deaths','SouthKorea_total_deaths','UnitedStates_total_deaths','Russia_total_deaths']
data_deaths = data_2[cols]
data_deaths.set_index("Date", inplace = True) 
data_deaths.head()
bcr.bar_chart_race(df=data_deaths, filename=None, figsize = (3.5,3),title='COVID-19 Deaths by Country')
# Modifying data
data_total_cases = data_2.filter(regex="total_cases|Date|Price")
# Drop countries with 0 cases
data_total_cases = data_total_cases.loc[:, (data_total_cases != data_total_cases.iloc[0]).any()] 
# countries = data_total_cases.columns.values[1:-1]
# countries = list(set([i.split('_')[0] for i in countries]))
data_total_cases.head()
# data transformation
dates = []
countries_ls = []
total_cases = []
prices = []
for index, row in data_total_cases.iterrows():
    df = pd.DataFrame(row).T
    c_ls = (df.iloc[:,1:-2].apply(lambda x: x.index[x.astype(bool)].tolist(), 1)[index])
    dates.extend([[df['Date'][index]]*len(c_ls)][0])
    prices.extend([[df['Price'][index]]*len(c_ls)][0])
    countries_ls.extend([col.split('_')[0] for col in c_ls])
    total_cases.extend([df[col][index] for col in c_ls])
    
data_2_mod = pd.DataFrame({'Date':dates,'Country':countries_ls,'Total_Cases':total_cases,'Price':prices})
data_2_mod.head()
from IPython.display import Image
sns.set(style="darkgrid", palette="pastel", color_codes=True)
sns.set_context("paper")
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "seaborn"
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot,plot
init_notebook_mode(connected=True)

fig = px.choropleth(
    data_2_mod, #Data
    locations= 'Country', #To get Lat and Lon of each country
    locationmode= 'country names', 
    color= 'Total_Cases', #color scales
    hover_name= 'Country', #Label while hovering
    hover_data= ['Country','Price'], #Data while hovering
    animation_frame= 'Date', #animate for each day
    color_continuous_scale=px.colors.sequential.Reds
)

fig.update_layout(
    title_text = "<b>COVID-19 Spread in the World up to Mar 31, 2020</b>",
    title_x = 0.5,
    geo= dict(
        bgcolor = 'black',
        showframe= False,
        showcoastlines= False,
        projection_type = 'equirectangular'
        
        
    )
)
iplot(fig)


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=data_2.Date, y=data_2.Price, name="Price"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=data_2.Date, y=data_2.World_total_cases, name="World Total Cases",line = dict(color = 'orangered')),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
#     title_text="Total cases vs Price"
    title='<b>Total cases vs Price</b>',
    plot_bgcolor='linen',
#     paper_bgcolor = 'grey',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=2,
                     label='2m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

# Set x-axis title
fig.update_xaxes(title_text="<b>Date</b>")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Price</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>World Total Cases</b>", secondary_y=True)

iplot(fig)

# Top countries impacted as of Mar 31, 2020.
cols = ['World_total_cases','World_total_deaths','China_total_cases','Italy_total_cases','Germany_total_cases',
        'Spain_total_cases','Iran_total_cases','France_total_cases','Price']

cordata = pd.DataFrame(data_2[cols].corr(method ='pearson'))

fig = go.Figure(data=go.Heatmap(z=cordata,x=cols,y=cols,colorscale='burgyl'))

iplot(fig)
cordata
data_2 = data_2.set_index(data_2['Date'])
train = data_2[:'2020-03-19']
cv = data_2['2020-03-20':]
train.tail()
features = ['World_total_cases','World_total_deaths','China_total_cases','Italy_total_cases','Germany_total_cases',
        'Spain_total_cases','Iran_total_cases','France_total_cases'] # columns to be used for training
y_train = train['Price'].values # target column
X_train = train[features] # Let's only consider few columns.
print('Number of X_train data points : ', X_train.shape[0])
print('Number of features train: ', X_train.shape[1])
print('Features : ', X_train.columns.values)
X_train = X_train.values
y_cv = cv['Price'].values
X_cv = cv[features]
print('Number of X_cv data points : ', X_cv.shape[0])
print('Number of features cv: ', X_cv.shape[1])
print('Features : ', X_cv.columns.values)
X_cv = X_cv.values
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from matplotlib.legend_handler import HandlerLine2D
from tqdm import tqdm


lr = LinearRegression()
lr.fit(X_train,y_train)

y_train_hat = lr.predict(X_train)
y_train_rmse = sqrt(mean_squared_error(y_train,y_train_hat))

y_cv_hat = lr.predict(X_cv)
y_cv_rmse = sqrt(mean_squared_error(y_cv,y_cv_hat))
print('Linear Regression Model trained!')
# train predictions
line1, = plt.plot(y_train, color="r", label="Actual Price")
line2, = plt.plot(y_train_hat, color="g", label="Predicted Price")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.title('Train: Actual Vs Predicted Price')
plt.xlabel('Number of rows')
plt.ylabel('Price')

# cv predictions
line1, = plt.plot(y_cv, color="r", label="Actual Price")
line2, = plt.plot(y_cv_hat, color="g", label="Predicted Price")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.title('CV: Actual Vs Predicted Price')
plt.xlabel('Number of rows')
plt.ylabel('Price')
test = pd.read_csv(path+'COVID-19_test.csv')
X_test = test[features]
print('Number of X_train data points : ', X_test.shape[0])
print('Number of features : ', X_test.shape[1])
test.head()
y_test_hat = lr.predict(X_test)
#print("Predicted Price from April 01 - May 22, 2020: \n")
# Predicted Price - First week of april
print("Predicted - First week of april \n")
for i in range(0,7):
    print("Date: {}, Predicted Price: {}".format(test['Date'][i],y_test_hat[i]))
submission_df = pd.DataFrame({'Date':test.Date,'Price':y_test_hat})
#submission_df.to_csv(path+'Submission.csv',index = False)
from IPython.display import Image
Image(filename='/kaggle/input/realitycovid19/Reality-covid.PNG') 
import pandas as pd 
import datetime as dt
import random #for random color function
import warnings

from bokeh.io import show, output_notebook
from bokeh.models import DatetimeTickFormatter
from bokeh.plotting import figure
from bokeh.palettes import brewer
from bokeh.transform import factor_cmap
from bokeh.models import HoverTool,ColumnDataSource
from bokeh.transform import dodge
from bokeh.core.properties import value

warnings.filterwarnings("ignore")
raw = pd.read_csv('../input/CryptocoinsHistoricalPrices.csv') #read the raw dataset
clean=raw.dropna(axis=0, inplace=False, thresh=7)  
clean=clean.drop(['Unnamed: 0','Open.','Close..'], axis=1, inplace=False)
clean= clean[clean['Market.Cap']!='-']
clean['Market.Cap'] = clean['Market.Cap'].str.replace(',', '')
clean['Volume'] = clean['Volume'].str.replace(',', '')

cols= [ 'coin','Date','Open','Close','High','Low','Volume','Market.Cap','Delta']
clean = clean[cols]

obj=clean.select_dtypes(include=object).columns.tolist()
obj.remove('coin')
obj.remove('Date')
clean[obj]=clean[obj].convert_objects(convert_numeric=True)

clean['Market.Cap']=clean['Market.Cap'].astype(float)
#Calculate 5 years trading volume, sort and store in the dataframe named 'Volume'
Volume=clean[['Volume','coin']].groupby('coin').sum().sort_values(by='Volume',ascending=False)
#select top 5 coins by volume 
TopVolume=Volume.head(5)
#show top coins and their 5 years volume
TopVolume['Volume']
#Generate top 5 coins' volume graph

#list the names and volumes of top 5 coins
coin = list(TopVolume.index)
vol = list(TopVolume['Volume'])
#generate source for graph
source = ColumnDataSource(data=dict(coin=coin, vol=vol))

p = figure(plot_height = 400,
           toolbar_location = 'right', 
           toolbar_sticky = True,
           x_range = coin,
           y_axis_type = 'log',
           y_range = [10**(11), 10**14],
           x_axis_label = 'Coin', 
           y_axis_label = 'Volume (USD)',
          )

p.vbar(x='coin', top='vol', width=0.7,source=source, legend= 'Coin',line_color='white',
       fill_color=factor_cmap('coin', palette=brewer['YlGnBu'][5], factors=coin),bottom=0.01)

p.xgrid.grid_line_color = None
p.title.text = 'Top Volume Cryptocoin'
p.title.align = 'center'
p.legend.location = 'top_right'
p.legend.orientation = 'vertical'
p.legend.border_line_color = 'gray'
p.border_fill_color = 'whitesmoke'
output_notebook()
show(p)
#list the name of top coins
Name=list(TopVolume.index)

#Separate and store data of top coin and store in the new dataframe names as the name of the coin
gbl = globals()
for i in Name:
    gbl[i]=clean[clean['coin']==i]
USDT
#random color function for time series graph
#def random_color(j):
#    shade = '#'+''.join([random.choice('abcdef0123456789') for n in range(6)])
#    return shade

#color function
color_list={}
color_list[Name[0]]= 'firebrick'
color_list[Name[1]]= 'darkorange'
color_list[Name[2]]= 'mediumseagreen'
color_list[Name[3]]= 'royalblue'
color_list[Name[4]]= 'indigo'

def check_color(str):
    shade = color_list[str]
    return shade

#create time series graph
p = figure(plot_width = 700, plot_height = 500, 
           y_axis_type='log',y_range=[10**(0), 10**13],
           x_axis_label = 'Date', y_axis_label = 'Volume (USD)')

p.title.text = '5 Years Trading Volume'
p.title.align = 'center'

p.xaxis.formatter=DatetimeTickFormatter()

#Because BTC has the longest history, the x-axis was set as the time scale of BTC data
x = [dt.datetime.strptime(date,'%Y-%m-%d').date() for date in BTC['Date']] 
    
for i in Name:
    #set_color=random_color(i)
    p.line(x, y=gbl[i]['Volume'], color = check_color(i), alpha=1.0,line_width=1.2, legend=i)
    p.add_tools(HoverTool(show_arrow=True, line_policy='next', 
                          tooltips=[('Date','@x{%F}'),('Volume','$y USD')],formatters={'x': 'datetime'}
                         )
                )
p.legend.location = 'top_left'
p.legend.border_line_color = 'black'

p.border_fill_color = 'whitesmoke'

output_notebook()

show(p)
#list the column of BTC dataframe and exclude 'Volume' column
cols = BTC.select_dtypes(include=float).columns.tolist()
cols.remove('Volume')
for i in Name:
    #split 'Date' column into 'Year' column
    gbl[i]['Year']=gbl[i]['Date'].apply(lambda x: x.split('-')[0])
    #calculate annual volume
    gbl['Annual_volume_'+i]=gbl[i].groupby('Year').sum()
    #delete irrelated columns
    gbl['Annual_volume_'+i].drop(cols, axis=1, inplace=True)
Annual_volume_BTC
Annual_volume = []
for i in Name:
    temp = gbl['Annual_volume_'+i].rename(index=str, columns={'Volume': i})
    Annual_volume.append(temp)
Annual_volume = pd.concat(Annual_volume, axis=1,sort=False)

Annual_volume
#function for preparing trading volume data for visualization
def pre_plot(dataframe,time):
    gbl[dataframe].reset_index(level=0, inplace=True)
    gbl[dataframe] = gbl[dataframe].rename(index=str, columns={time: 'index'})

#function for calculating possition on x-axis of bar graph
def possition(num,j):
    x = -(num)/2
    x = x+(1.5*j)
    x = x*0.1
    return x

#bar graph function for trading volume data
def bar(df,str):
    source = ColumnDataSource(data=df)
    num = len(df)
    
    p = figure(x_range = df['index'].tolist(), 
               plot_height = 400,
               toolbar_location = 'right', 
               toolbar_sticky = False,
               y_axis_type = 'log',
               y_range = [10**(5), 10**13],
               x_axis_label = 'Year', 
               y_axis_label = 'Volume (USD)'
              )
    
    for i in Name:
        #set_color = random_color(i)
        p.vbar(x = dodge('Time', 
               possition(num,Name.index(i)), 
               range = p.x_range), 
               top = i, 
               width = 0.1, 
               source = source,
               #color = set_color, 
               color = check_color(i),
               line_color = 'black', 
               legend = value(i),
               bottom = 0.01,
               alpha = 0.25
              )
        p.line(x = df['Time'], 
               y = df[i], 
               color = check_color(i), 
               line_width = 3,
               legend = value(i))
        p.circle(x = df['Time'], 
                 y = df[i],
                 fill_color = check_color(i), 
                 line_color = check_color(i),
                 size = 10,
                 )
        p.add_tools(HoverTool(show_arrow = True, 
                              line_policy = 'next', 
                              tooltips = [('Volume','$y $')]
                              )
                   )
        
    p.title.text = str
    p.title.align = 'center'
    p.xgrid.grid_line_color = None
    p.legend.location = 'top_center'
    p.legend.orientation = 'horizontal'
    p.legend.border_line_color = 'gray'
    p.border_fill_color = 'whitesmoke'

    output_notebook()
        
    show(p)
    return;
#call pre_plot function for Annual_volume dataframe
pre_plot('Annual_volume','Year')
#Additional modification, create Time column and set it as string column
Annual_volume['Time']=Annual_volume['index'].astype(str)
#plot bar graph for annual trading volumes of top coins
bar(Annual_volume,str='Top 5 Crypocoins: Annual Trading Volume')
#separate data of 2018 year
for i in Name:
    gbl['Y2018_volume_'+i]=gbl[i][gbl[i]['Year']=='2018']
    gbl['Y2018_volume_'+i]['Month']=gbl['Y2018_volume_'+i]['Date'].apply(lambda x: x.split('-')[1])
    gbl['Y2018_volume_'+i]=gbl['Y2018_volume_'+i].groupby('Month').sum()    
    gbl['Y2018_volume_'+i].drop(cols, axis=1, inplace=True)

#create monthly report
Monthly_volume = []
for i in Name:
    temp = gbl['Y2018_volume_'+i].rename(index=str, columns={'Volume': i})
    Monthly_volume.append(temp)
Monthly_volume = pd.concat(Monthly_volume, axis=1,sort=False)
Monthly_volume
#call pre_plot function for Monthly_volume dataframe
pre_plot('Monthly_volume','Month')
#Additional modification
#Convert month code to month name
Month_num = ['01', '02', '03', '04', '05', '06', '07','08', '09', '10', '11', '12']
Month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec']
Monthly_volume['index']=Monthly_volume['index'].replace(Month_num, Month_name)
#create Time column and set it as string column
Monthly_volume['Time']=Monthly_volume['index'].astype(str)
#plot bar graph of 2018's monthly trading volume
bar(Monthly_volume,'Top 5 Crypocoins: Monthly Trading Volume')
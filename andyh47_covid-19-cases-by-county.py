%matplotlib inline
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
plt.style.use('ggplot')
nyt_pull = pd.read_html('https://github.com/nytimes/covid-19-data/blob/master/us-counties.csv')[0]
nyt_pull.drop(0,inplace=True,axis=1)

# build data frame, set datatypes and calculate fields
nyt_df = pd.DataFrame((nyt_pull.loc[1:,1].apply(lambda x: x.split(','))).tolist())
nyt_df.columns = nyt_pull.loc[0].to_list()[0].split(',')
nyt_df['cases'] = nyt_df['cases'].astype(float)
nyt_df['log_cases'] = nyt_df.cases.apply(lambda x: np.log10(x))
nyt_df.deaths = nyt_df['deaths'].astype(float)
nyt_df['log_deaths'] = nyt_df.deaths.apply(lambda x: np.log(x))
nyt_df.date = nyt_df['date'].apply(lambda x: dt.datetime.strptime(str(x),'%Y-%m-%d'))
nyt_df.tail()
display_date = [np.max(nyt_df.date).month,np.max(nyt_df.date).day,np.max(nyt_df.date).year]
display_date = map(str,display_date)
display_date = ('-').join(display_date)
display(HTML(f'Data through {display_date}'))
def prepare_county(c,state):
    county = nyt_df[(nyt_df.county == c) & (nyt_df.state == state)].sort_values('date')
    county.set_index('date',inplace=True)
    return county

def prepare_us(nyt_df):
    df_all = nyt_df.groupby('date').apply(lambda df: df.cases.sum())
    df_all = df_all.to_frame()
    df_all.columns = ['cases']
    df_all['log_cases'] = df_all.cases.apply(lambda x: np.log10(x))
    return df_all
def plot_df(df,lower_bound=10):
    # plot time series cases, log cases and phase plot
    fig, (ax0,ax1,ax2) = plt.subplots(3,1,figsize = (8,12))
    ax0.plot(df.cases)
    #ax0.set_title('Cases')
    ax0.set_ylabel('Cases')
    Y = df.log_cases
    ax1.plot(Y)
    #ax1.set_title('Log10 Cases')
    ax1.set_ylabel('Log10 Cases')
    # plot change in cases against cases (x-axis)
    df1 = df[df.cases > lower_bound].copy()
    df1['dif'] = df1.loc[:,'cases'].diff()
    df1['dif_smooth'] = df1.dif.rolling(7).mean()
    df1['dif_smooth'] = df1['dif_smooth'].apply(lambda x: np.nan if x <=0 else x)
    ax2.scatter(np.log10(df1['cases']),np.log10(df1['dif_smooth']))
    ax2.set_xlabel('Log10 Total Cases')
    ax2.set_ylabel('Log10 Change in Cases')
    return fig,df
fig,tu = plot_df(prepare_county('Pima','Arizona'))
plt.savefig('./example.png')
napa = plot_df(prepare_county('Napa','California'))
mont = plot_df(prepare_county('Monterey','California'))

sc = plot_df(prepare_county('Santa Clara','California'))
fig,sf = plot_df(prepare_county('San Francisco','California'))
sm = plot_df(prepare_county('San Mateo','California'))
barn = plot_df(prepare_county('Barnstable','Massachusetts'))
bos = plot_df(prepare_county('Suffolk','Massachusetts'))
nyc = plot_df(prepare_county('New York City','New York'))
king = plot_df(prepare_county('King','Washington'))
fig,us = plot_df(prepare_us(nyt_df))

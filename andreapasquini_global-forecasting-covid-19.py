!pip install lmfit

# libraries to manage number
import numpy as np
import pandas as pd
# libraries to plotting
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# libraries to model

# Libraries to metrics
from sklearn import metrics


# libraries to optimize model
from lmfit import Model
def sigmoid(x, b, r, t):
    z = (t * (b + x))
    sig = 1/(1 + np.exp( -z ))*r
    return sig
#http://www.edscave.com/forecasting---time-series-metrics.html
def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df = pd.read_csv('../input/Global_Covid-19-05-04-2020.csv',index_col=0)
df = df.groupby(by='Country/Region',).sum()
df.reset_index(inplace=True)
days = len(df.columns[3:])
days = np.arange(days)
df.columns[days.max()+3]
df.sort_values(df.columns[days.max()+3],ascending=False)['Country/Region']
country_list = df.sort_values(df.columns[days.max()+3],ascending=False)['Country/Region']
fig, ax = plt.subplots(nrows=16, ncols=2, figsize=(16,80))
n = 0
c1 = ['Italy', 'Spain', 'US', 'Germany', 'Belgium', 'Switzerland', 'Iran',
      'Turkey', 'Brazil', 'Sweden', 'Portugal', 'Indonesia', 'Austria',
      'Korea, South', 'Ecuador', 'Romania']
df_coeff = pd.DataFrame()
for reg in c1:
    y = df.loc[df['Country/Region']== reg,'1/22/20':]
    ax[n][0].set_title('Current situation {}'.format(reg))
    sns.scatterplot(x=days, y=y.iloc[0,:],ax=ax[n][0])    
           
    model = Model(sigmoid)
    pred = model.fit(y.iloc[0,:], x=days, b= 500, r=y.iloc[0,:].max()/4, t= 0.0001)  # b= 0, r=, t= 0.001

    x_ideal = np.linspace(np.min(days), np.max(days)*2)
    ideal = pred.eval(x=x_ideal)
    sns.scatterplot(x_ideal, ideal, label='forecasting model', ax=ax[n][1])
    

    sns.scatterplot(x=days, y=y.iloc[0,:], label='deaths in {} until now'.format(reg), s=100, color='red', ax=ax[n][1])

    ax[n][1].grid()
    ax[n][1].set_title('{} model'.format(reg))
    ax[n][1].set_xlim(0)
    
    df_coeff = df_coeff.append(pred.values, ignore_index=True)
    #df_coeff.loc[n, 'MAPE'] = mean_absolute_percentage_error(ideal[:len(y)],y)
    df_coeff.loc[n,'Regions'] = reg
    n+=1
fig, ax = plt.subplots(nrows=16, ncols=2, figsize=(16,80))
n = 0
c1 = ['Philippines', 'Ireland', 'India','Japan', 'Peru', 'Egypt', 'Greece', 'Dominican Republic', 'Norway', 'Czechia', 'Morocco', 'Malaysia', 'Iraq',
       'Israel', 'Serbia', 'Germany']
for reg in c1:
    y = df.loc[df['Country/Region']== reg,'1/22/20':]
    ax[n][0].set_title('Current situation {}'.format(reg))
    sns.scatterplot(x=days, y=y.iloc[0,:],ax=ax[n][0])    
           
    model = Model(sigmoid)
    pred = model.fit(y.iloc[0,:], x=days, b= 500, r=y.iloc[0,:].max()/4, t= 0.0001)  # b= 0, r=, t= 0.001

    x_ideal = np.linspace(np.min(days), np.max(days)*2)
    ideal = pred.eval(x=x_ideal)
    sns.scatterplot(x_ideal, ideal, label='forecasting model', ax=ax[n][1])
    

    sns.scatterplot(x=days, y=y.iloc[0,:], label='deaths in {} until now'.format(reg), s=100, color='red', ax=ax[n][1])

    ax[n][1].grid()
    ax[n][1].set_title('{} model'.format(reg))
    ax[n][1].set_xlim(0)
    
    df_coeff = df_coeff.append(pred.values, ignore_index=True)
    #df_coeff.loc[n, 'MAPE'] = mean_absolute_percentage_error(ideal[:len(y)],y)
    df_coeff.loc[16+n,'Regions'] = reg
    n+=1
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16,20))
n = 0
c1 = ['Mexico','Netherlands', 'Denmark']
for reg in c1:
    y = df.loc[df['Country/Region']== reg,'1/22/20':]
    ax[n][0].set_title('Current situation {}'.format(reg))
    sns.scatterplot(x=days, y=y.iloc[0,:],ax=ax[n][0])    
           
    model = Model(sigmoid)
    pred = model.fit(y.iloc[0,:], x=days, b=0, r=y.iloc[0,:].max()/4, t= -0.05)  # b= 0, r=, t= 0.001 r=y.iloc[0,:].max()/4

    x_ideal = np.linspace(np.min(days), np.max(days)*2)
    ideal = pred.eval(x=x_ideal)
    sns.scatterplot(x_ideal, ideal, label='forecasting model', ax=ax[n][1])
    

    sns.scatterplot(x=days, y=y.iloc[0,:], label='deaths in {} until now'.format(reg), s=100, color='red', ax=ax[n][1])

    ax[n][1].grid()
    ax[n][1].set_title('{} model'.format(reg))
    ax[n][1].set_xlim(0)
    
    df_coeff = df_coeff.append(pred.values, ignore_index=True)
    #df_coeff.loc[n, 'MAPE'] = mean_absolute_percentage_error(ideal[:len(y)],y)
    df_coeff.loc[32+n,'Regions'] = reg
    n+=1

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
n = 0
c1 = ['United Kingdom','France']
for reg in c1:
    y = df.loc[df['Country/Region']== reg,'1/22/20':]
    ax[n][0].set_title('Current situation {}'.format(reg))
    sns.scatterplot(x=days, y=y.iloc[0,:],ax=ax[n][0])    
           
    model = Model(sigmoid)
    pred = model.fit(y.iloc[0,:], x=days, b=-1, r=y.iloc[0,:].max()/2, t= -0.005,)  # b= 0, r=, t= 0.001 r=y.iloc[0,:].max()/4

    x_ideal = np.linspace(np.min(days), np.max(days)*2)
    ideal = pred.eval(x=x_ideal)
    sns.scatterplot(x_ideal, ideal, label='forecasting model', ax=ax[n][1])
    

    sns.scatterplot(x=days, y=y.iloc[0,:], label='deaths in {} until now'.format(reg), s=100, color='red', ax=ax[n][1])

    ax[n][1].grid()
    ax[n][1].set_title('{} model'.format(reg))
    ax[n][1].set_xlim(0)

    df_coeff = df_coeff.append(pred.values, ignore_index=True)
    df_coeff.loc[35+n,'Regions'] = reg
    n+=1
df_coeff
df_coeff.columns=['beta', 'cap', 'theta', 'Regions']
df_coeff['cap'] = df_coeff['cap'].apply(lambda x: int(x))
df_coeff[['cap','Regions']].rename(columns={'cap':'Assumed number of deaths','Regions':'Regions'})
from scipy.spatial import ConvexHull
def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)
plt.figure(figsize=(10,10))
sns.scatterplot(data=df_coeff, x='beta', y='theta', s=100,)
n = list(np.concatenate( (df_coeff.sort_values('theta', ascending=False).head(3).index.values,df_coeff[df_coeff.beta<-77].index.values,df_coeff[df_coeff.beta>-65].index.values), axis=0))

for i in n:
    plt.annotate(df_coeff.Regions[i], (df_coeff.beta[i], df_coeff.theta[i]*1.01),size=20)

N = list(df_coeff.index.values)
N = [x for x in N if x not in n]

encircle(df_coeff[df_coeff.index.isin(N)].beta, df_coeff[df_coeff.index.isin(N)].theta, ec="orange", fc="none")
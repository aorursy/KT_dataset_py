# We will import all the necessary library modules used for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sns
df=pd.read_csv('../input/cancer2017.csv',engine='python')
df.head()
df.replace({r'[^\x00-\x7F]+':np.nan}, regex=True, inplace=True)
df.head()
df.columns = [c.strip() for c in df.columns.values.tolist()]
df.columns = [c.replace(' ','') for c in df.columns.values.tolist()]
df.columns
#removing commas
for i in range(0,df.shape[0]):
    for j in range(1,df.shape[1]):
        if ',' in str(df.iloc[i][j]):
            df.iloc[i][j]=df.iloc[i][j].replace(',','')
df.head()
df.info()
df=df.apply(pd.to_numeric, errors='ignore')
df.info()
df.head()
y=list(df.columns)
bdf=df.copy()
for col in range(1,len(y)):
    bdf[y[col]].fillna((bdf[y[col]].mean()), inplace=True)
bdf.head()
x='State'
i=1
z=["prostate","brain","breast","colon","leukemia","liver","lung","lymphoma","ovary","pancreas"]
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))
fig.suptitle('Incomplete Data Set')
for row in ax:
    for col in row:
        col.plot(df[x],df[y[i]])
        i=i+1
i=0
for ax in fig.axes:
    plt.xlabel('States')
    plt.ylabel("no of people affected")
    plt.title(z[i])
    i=i+1
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
i=1
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))
fig.suptitle('NaN Filled Data Set')

for row in ax:
    for col in row:
        col.plot(bdf[x],bdf[y[i]])
        i=i+1
i=0
for ax in fig.axes:
    plt.xlabel('States')
    plt.ylabel("no of people affected")
    plt.title(z[i])
    i=i+1
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
i=1
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))
fig.suptitle('Incomplete Data Set')

for row in ax:
    for col in row:
        col.bar(df[x],df[y[i]])
        i=i+1
i=0
for ax in fig.axes:
    plt.xlabel('States')
    plt.ylabel("no of people affected")
    plt.title(z[i])
    i=i+1
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
i=1
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))
fig.suptitle('NaN Filled Data Set')

for row in ax:
    for col in row:
        col.bar(bdf[x],bdf[y[i]])
        i=i+1
i=0
for ax in fig.axes:
    plt.xlabel('States')
    plt.ylabel("no of people affected")
    plt.title(z[i])
    i=i+1
    plt.sca(ax)
    plt.xticks(rotation=90)
    plt.grid()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
init_notebook_mode(connected=True)
usa=df.copy()
for col in range(1,len(y)):
    usa[y[col]].fillna((usa[y[col]].mean()), inplace=True)

usa['total']=usa.sum(axis=1)
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
usa['code']=states
for col in usa.columns:
    usa[col] = usa[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

usa['text'] =' State: '+ usa['State'] + '<br>' +\
    ' Brain/nervoussystem: '+usa['Brain/nervoussystem']+'<br>'+' Femalebreast: '+usa['Femalebreast']+'<br>'+\
    ' Colon&rectum: '+usa['Colon&rectum']+'<br>'+ ' Leukemia: ' + usa['Leukemia']+'<br>'+\
    ' Liver: '+usa['Liver']+'<br>'+' Lungs&bronchus: ' + usa['Lung&bronchus']+'<br>'+\
    ' Non-HodgkinLymphoma: '+usa['Non-HodgkinLymphoma']+'<br>'+' Ovary: ' + usa['Ovary']+'<br>'+\
    ' Pancreas: '+usa['Pancreas']+'<br>'+' Prostate: ' + usa['Prostate']
    

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = usa['code'],
        z = usa['total'].astype(float),
        locationmode = 'USA-states',
        text = usa['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "No of People")
        ) ]

layout = dict(
        title = '2017 Cancer Statistics of U.S.A<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )
cancertypes=list(df.columns[1:df.shape[1]])
corr = df[cancertypes].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.3f',annot_kws={'size': 12},
           xticklabels= cancertypes, yticklabels= cancertypes,
           cmap= 'coolwarm')
usa['canpop']=usa['total'].astype(float)
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(20, 15))

sns.barplot(x=usa['canpop'],y=usa['State'],
            label="Cancer Affected", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="",
       xlabel="Cancer Statistics 2017")
sns.despine(left=True, bottom=True)
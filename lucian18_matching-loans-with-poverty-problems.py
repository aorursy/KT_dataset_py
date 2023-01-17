import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

import colorlover as cl
from IPython.display import HTML

import warnings
warnings.filterwarnings("ignore")
df = pd.read_excel('../input/mpi-on-regions/mpi_on_regions.xlsx', encoding='utf-8')
df_loan_theme = pd.read_excel('../input/mpi-on-regions/all_loan_theme_merged_with_geo_mpi_regions.xlsx', encoding = 'utf-8')
kiva_mpi_reg_loc = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
df.shape
print('Missing values: ')
df.isnull().sum()
print('Statistics of the database: ')
df.describe()
for x in df.loc[:, df.dtypes == 'object']:
    y = len(list(df[x].unique()))
    print('For column %s we have %d individual values' %(x,y))
for x in df.loc[:, df.dtypes == np.float64]:
    y = df[x].mean()
    print('For column %s the average is %.2f' % (x,y))
w_reg = df['World region'].unique()
mpi_mean = df.groupby('World region')['country MPI'].mean()

trace = go.Scatter(x=mpi_mean.round(3), y=mpi_mean.round(3),
                   mode = 'markers',
                   marker = dict(size=mpi_mean.values*200, color = mpi_mean.values, 
                                 colorscale='YlOrRd', showscale=True, reversescale=True),
                   text = w_reg, line=dict(width = 2, color='k'),
                  )

axis_template = dict(showgrid=False, zeroline=False, nticks=10,
                    showline=True, title='MPI Scale', mirror='all')


layout = go.Layout(title="Average MPI of the World's Regions analyzed",
                  hovermode = 'closest', xaxis=axis_template, yaxis=axis_template)


data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# need 51 colors or shades for my plot
cs12 = cl.scales['12']['qual']['Paired']
col = cl.interp(cs12, 60)

coun_mpi = df[['Country', 'country MPI']].drop_duplicates().reset_index(drop=True)

trace = go.Bar(x = coun_mpi['country MPI'].sort_values(ascending=True),
               y = coun_mpi['Country'], 
               orientation = 'h',
               marker = dict(color = col),
              )

layout = go.Layout(title='MPI of the Countries in the dataset',
                   width = 800, height = 1000,
                   margin = dict(l = 175),
                   yaxis=dict(#tickangle=45,
                              tickfont=dict(family='Old Standard TT, serif', size=13, color='black')
                             ),
                  )


data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='MPI of the Countries')
data = [ dict(type = 'scattergeo',
              lat = kiva_mpi_reg_loc['lat'],
              lon = kiva_mpi_reg_loc['lon'],
              text = kiva_mpi_reg_loc['LocationName'],
              marker = dict(size = 8,
                            line = dict(width=0.5, color='k'),
                            color = kiva_mpi_reg_loc['MPI'],
                            colorscale = 'YlOrRd',
                            reversescale = True,
                            colorbar=dict(title="MPI")
                            )
             )
       ]
layout = dict(title = 'Regional MPI across the globe',
             geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
             )
fig = dict( data=data, layout=layout )
py.iplot(fig)
spec = cl.scales['10']['div']['Spectral']
spectral = cl.interp(spec, 20)

pc = df.groupby(['Field Partner Name'])['number'].sum().sort_values(ascending=False).head(20)
info = []
for i in pc.index:
    cn = str(df['Country'][df['Field Partner Name'] == i].unique())
    cn = cn.strip("['']").replace("' '", ', ')
    info.append(cn)

trace = go.Bar(x = pc.index,
               y = pc.values,
               text = info,
               orientation = 'v',
               marker = dict(color = spectral)
               )

layout = go.Layout(title='Number of loans facilitated by top 20 financial partners',
                   width = 800, height = 500,
                   margin = dict(b = 175),
                   )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
pc2 = df.groupby(['Field Partner Name'])['amount'].sum().sort_values(ascending=False).head(20)
info2 = []
for i in pc2.index:
    cn = str(df['Country'][df['Field Partner Name'] == i].unique())
    cn = cn.strip("['']").replace("' '", ', ')
    info2.append(cn)
trace = go.Bar(x = pc2.index,
               y = pc2.values,
               text = info2,
               orientation = 'v',
               marker = dict(color = spectral),
               )

layout = go.Layout(title='Gross amount of the loans facilitated by financial partners (in $)',
                   width = 800, height = 500,
                   margin = dict(b = 175),
                   )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
gb2 = df.groupby(['Loan Theme Type', 'World region', 'sector'])
num2 = gb2['number'].agg(np.sum)
amo2 = gb2['amount'].agg(np.sum)
sumsdf2 = pd.DataFrame({'amount': gb2['amount'].agg(np.sum), 'number': gb2['number'].agg(np.sum)}).reset_index()

hover_text = []
for index, row in sumsdf2.iterrows():
    hover_text.append(('Number of loans: {a}<br>' + 
                       'Amount of loans: {b}<br>' +
                       'Loan theme type: {c}<br>' +
                       #'Country: {d}<br>' +
                       'World region: {e}<br>').format(a = row['number'],
                                                       b = row['amount'],
                                                       c = row['Loan Theme Type'],
                                                       #d = row['Country'],
                                                       e = row['World region']
                                                  )
                     )    

sumsdf2['text'] = hover_text

sectors = ['General Financial Inclusion', 'Other', 'Water and Sanitation', 'Mobile Money and ICT', 'Clean Energy', 'Education',
           'DSE Direct', 'Artisan', 'SME Financial Inclusion', 'Agriculture', 'Health']

data = []

for s in sorted(sectors):
    trace = go.Scatter(x = sumsdf2['amount'][sumsdf2['sector'] == s], 
                   y = sumsdf2['number'][sumsdf2['sector'] == s],
                   name = s,
                   mode = 'markers',
                   text = sumsdf2['text'][sumsdf2['sector'] == s],
                   hoverinfo = 'text',
                   hoveron = 'points+fills',    
                   marker = dict(size = np.sqrt(sumsdf2['amount'][sumsdf2['sector'] == s]),
                                 sizemode = 'area', 
                                 line=dict(width = 2),
                                 ),
                   )
    data.append(trace)

layout = go.Layout(title="Type of loans grouped on sectors and world regions",
                   hovermode = 'closest', 
                   xaxis=dict(title='Total amount of loans', type='log'),
                   yaxis=dict(title='Total number of loans', type='log'),
                   paper_bgcolor='rgb(243, 243, 243)',
                   plot_bgcolor='rgb(243, 243, 243)',
                   )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace = []
for name, group in df_loan_theme.groupby(['sector']):
    trace.append(go.Box( x = group['amount'].values,
                         name = name
                       )
                )
layout = go.Layout( title = 'Amount of loans per sectors', 
                  margin = dict( l = 170 ),
                   xaxis = dict( type = 'log' )
                  )
fig = go.Figure(data = trace, layout = layout)
py.iplot(fig)
gb = df.groupby(['Sub-national region', 'Country', 'World region'])
num = gb['number'].agg(np.sum)
amo = gb['amount'].agg(np.sum)
sumsdf = pd.DataFrame({'amount': gb['amount'].agg(np.sum), 'number': gb['number'].agg(np.sum)}).reset_index()
sumsdf[30:35]
hover_text = []
for index, row in sumsdf.iterrows():
    hover_text.append(('Number of loans: {a}<br>' + 
                       'Amount of loans: {b}<br>' +
                       'Sub-national region: {c}<br>' +
                       'Country: {d}<br>').format(a = row['number'],
                                                  b = row['amount'],
                                                  c = row['Sub-national region'],
                                                  d = row['Country']
                                                  )
                     )    

sumsdf['text'] = hover_text
    
world = ['East Asia and the Pacific', 'Sub-Saharan Africa', 'Arab States', 'Latin America and Caribbean', 
         'Central Asia', 'South Asia']

data = []

for w in sorted(world):
    trace = go.Scatter(x = sumsdf['amount'][sumsdf['World region'] == w], 
                   y = sumsdf['number'][sumsdf['World region'] == w],
                   name = w,
                   mode = 'markers',
                   text = sumsdf['text'][sumsdf['World region'] == w],
                   hoverinfo = 'text',
                   marker = dict(size = np.sqrt(sumsdf['amount'][sumsdf['World region'] == w]),
                                 sizemode = 'area',                                 
                                 line=dict(width = 2),                                 
                                 ),
                   )
    data.append(trace)

layout = go.Layout(title="Loans across regions",
                   hovermode = 'closest', 
                   xaxis=dict(title='Total amount of loans', type='log'),
                   yaxis=dict(title='Total number of loans', type='log'),
                   paper_bgcolor='rgb(243, 243, 243)',
                   plot_bgcolor='rgb(243, 243, 243)',
                   )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
top30 = df[['Country', 'Sub-national region', 'region MPI']].drop_duplicates().reset_index(drop=True)
small_mpi = top30.sort_values('region MPI').reset_index(drop=True)

c11s3 = cl.scales['11']['qual']['Set3']
col2 = cl.interp(c11s3, 30)

x = small_mpi['region MPI'].head(30)
y = small_mpi['Sub-national region'].head(30)
country = small_mpi['Country'].head(30)

trace = go.Bar(x = x[::-1],
               y = y[::-1],
               text = country[::-1],
               orientation = 'h',
               marker=dict(color = col2),
               )

layout = go.Layout(title='The 30 least poor regions of the globe',
                   width = 800, height = 800,
                   margin = dict(l = 195),
                   xaxis=dict(title='MPI'),
                   yaxis=dict(#tickangle=45, 
                              tickfont=dict(family='Old Standard TT, serif', size=13, color='black'),
                             )
                   )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
t10_reg = df.sort_values('region MPI')
t10_reg.drop_duplicates(subset=['region MPI', 'Sub-national region'], inplace=True)
t10_reg = t10_reg.reset_index(drop=True)

labels = ['Education', 'Health', 'Living standards']
colors = ['rgb(11, 133, 215)', 'rgb(51,160,44)', 'rgb(240, 88, 0)']

for x in range(len(t10_reg[:10])):
    values = t10_reg.iloc[x, 17:20]
    trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent', marker=dict(colors=colors), sort=False)
    layout = go.Layout(title='Contribution to overall poverty for %s, %s' % (
                        t10_reg['Sub-national region'][x], t10_reg['Country'][x]))
    fig = go.Figure(data=[trace], layout=layout)
    py.iplot(fig)
labels = ['Schooling', 'Child school attendance', 'Child mortality', 'Nutrition', 'Electricity', 'Improved sanitation', 
          'Drinking water', 'Floor', 'Cooking fuel', 'Asset ownership']

x = list(t10_reg['Sub-national region'][:10])
text = list(t10_reg['Country'][:10])
data = []

for i in range(10):
    trace = go.Bar(x=x, y=t10_reg.iloc[:10, (20+i)], name=labels[i], text=text)
    data.append(trace)

    
layout = go.Layout(barmode='stack', title='Poverty decomposition for the 10 least poor regions', showlegend=True, 
                   margin = dict(b = 125),
                  xaxis=dict(title='Region', tickangle = 45), yaxis=dict(title='Percent %'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
top30 = df[['Country', 'Sub-national region', 'region MPI']].drop_duplicates().reset_index(drop=True)
high_mpi = top30.sort_values('region MPI').reset_index(drop=True)

trace = go.Bar(x = high_mpi['region MPI'].tail(30),
               y = high_mpi.index[:30],
               text = high_mpi['Sub-national region'].tail(30) + ', ' + high_mpi['Country'].tail(30),
               orientation = 'h',
               marker = dict(color = col2),
               )

layout = go.Layout(title='The 30 poorest regions of the globe',
                   width = 800, height = 800,
                   xaxis=dict(title='MPI'),
                   yaxis=dict(tickangle=45, 
                              tickfont=dict(family='Old Standard TT, serif', size=11, color='black'),
                             )
                   )

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
t10_hi = df.sort_values('region MPI', ascending=False)
t10_hi.drop_duplicates(subset=['region MPI', 'Sub-national region'], inplace=True)
t10_hi = t10_hi.reset_index(drop=True)

labels = ['Education', 'Health', 'Living standards']
colors = ['rgb(11, 133, 215)', 'rgb(51,160,44)', 'rgb(240, 88, 0)']

x = list(t10_hi['Sub-national region'][:10])
text = list(t10_hi['Country'][:10])
data = []

for i in range(3):
    trace = go.Bar(x=x, y=t10_hi.iloc[:10, (17+i)], name=labels[i], text=text)
    data.append(trace)

    
layout = go.Layout(barmode='group', title='Contribution to overall poverty for the 10 poorest regions', showlegend=True, 
                   xaxis=dict(title='Region'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
labels = ['Schooling', 'Child school attendance', 'Child mortality', 'Nutrition', 'Electricity', 'Improved sanitation', 
          'Drinking water', 'Floor', 'Cooking fuel', 'Asset ownership']

x = list(t10_hi['Sub-national region'][:10])
text = list(t10_hi['Country'][:10])
data = []

for i in range(10):
    trace = go.Bar(x=x, y=t10_hi.iloc[:10, (20+i)], name=labels[i], text=text)
    data.append(trace)

    
layout = go.Layout(barmode='stack', title='Poverty decomposition for top 10 poorest regions', showlegend=True, 
                  xaxis=dict(title='Region'), yaxis=dict(title='Percent %'))

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
x = t10_hi['Population Share of the Region (%)'].head(50)
y = t10_hi['region MPI'].head(50)

hover_text = []
for index, row in t10_hi.head(50).iterrows():
    hover_text.append(('Population Share of the Region: {a}<br>' + 
                       'Region MPI: {b}<br>' +
                       'Sub-national region: {c}<br>' +
                       'Country: {d}<br>').format(a = str(row['Population Share of the Region (%)']*100)+'%',
                                                  b = row['region MPI'],
                                                  c = row['Sub-national region'],
                                                  d = row['Country']
                                                  )
                     )    


trace = go.Scatter(x=x, y=y, mode = 'markers',
                   text = hover_text,
                   hoverinfo = 'text',
                   line=dict(width = 2, color='k'),
                   marker = dict(size=x*200,
                                 color = x,
                                 colorscale='Rainbow', showscale=True, reversescale=False,
                                 ),
                   )

layout = go.Layout(title="Percent of the country's population living in the 50 poorest areas",
                   hovermode = 'closest',                  
                   xaxis=dict(title='Population Share of the Region (%)'),
                   yaxis=dict(title='MPI of the region')
                   )


data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_mpi = df.drop_duplicates(subset=['Sub-national region', 'region MPI']).reset_index(drop=True)
df_mpi.sort_values(['region MPI'], ascending=False, inplace=True)
df_mpi = df_mpi.iloc[:, 10:].reset_index(drop=True)

x = list(df_mpi.columns.values)[10:20]
y = df_mpi['Sub-national region'].head(20)
z = df_mpi.iloc[:, 10:20].head(20).values

trace = go.Heatmap(x = x, y = y, z = z,
                  colorscale = 'Jet',
                  colorbar = dict(title = 'IN %', x = 'center')
                  )

layout = go.Layout(title='Decomposition of problems for 20 poorest regions',
                   margin = dict(l = 155, b = 100),
                    xaxis = dict(tickfont = dict(size=11)),
                    #yaxis = dict(tickangle = 45)
                  )

fig = dict(data = [trace], layout = layout)
py.iplot(fig)
df_loan = df.groupby(['Country', 'Sub-national region', 'sector', 'Loan Theme Type', 'region MPI'])
df_loan = pd.DataFrame({'number': df_loan['number'].sum(), 'amount': df_loan['amount'].sum()}).reset_index()
mycolors = ['#F81106','#FA726C','#F8C1BE',
            '#137503','#54B644','#B2F5A7',
            '#051E9B','#4358C0','#A6B3F9',
           '#9C06A0','#C34BC6','#F3A1F6',
           '#A07709','#CDA742','#F4DC9D',
           '#08A59E','#4DD5CE','#AAF7F3']
hover_text = []
for index, row in df_loan.iterrows():
    hover_text.append(('Loan type: {a}<br>' + 'Sector: {b}<br>' + 'Amount: {c}<br>').format(
                        a = row['Loan Theme Type'], b = row['sector'], 
                        c = '$' + str("{:,}".format(row['amount'])))
                     )
df_loan['text'] = hover_text
countries = ['Bolivia']
for c in countries:
    creg = df_loan[df_loan['Country'] == c]
    regions = pd.Series(creg['Sub-national region'].unique())
    for r in regions:
        selector = df_loan[(df_loan['Country'] == c) & (df_loan['Sub-national region'] == r)]
        trace = go.Pie(values = selector['amount'],
              labels = selector['Loan Theme Type'],
               text = selector['text'],
               hoverinfo = 'text',
               textinfo = 'percent',
               textfont = dict(size=15),
               marker = dict(colors = mycolors,
                            line = dict(color='k', width=0.75)),           
              )
        layout = go.Layout(title = 'Amount and type of loans for {}, {} (MPI: {})'.format(r,c, selector['region MPI'].median())
                          )
        fig = go.Figure(data = [trace], layout = layout)
        py.iplot(fig)
countries = ['Philippines']

for c in countries:
    creg = df_loan[df_loan['Country'] == c]
    regions = pd.Series(creg['Sub-national region'].unique())
    for r in regions:
        selector = df_loan[(df_loan['Country'] == c) & (df_loan['Sub-national region'] == r)]
        trace = go.Pie(values = selector['amount'],
              labels = selector['Loan Theme Type'],
               text = selector['text'],
               hoverinfo = 'text',
               textinfo = 'percent',
               textfont = dict(size=15),
               marker = dict(colors = mycolors,
                            line = dict(color='k', width=0.75)),           
              )
        layout = go.Layout(title = 'Amount and type of loans for {}, {} (MPI: {})'.format(r,c, selector['region MPI'].median())
                          )
        fig = go.Figure(data = [trace], layout = layout)
        py.iplot(fig)
lt = df_loan_theme.groupby(['Loan Theme Type', 'forkiva'])['number'].sum()
lt = lt.to_frame().reset_index()
lt = lt.pivot(index = 'Loan Theme Type', columns = 'forkiva', values= 'number')
lt['No'] = lt['No'].fillna(0)
lt['Yes'] = lt['Yes'].fillna(0)
lt['total'] = lt['No'] + lt['Yes']
# get rid of General loan theme as is skewing the chart
lt = lt.loc[~(lt['No'] > 300000)]
lt = lt.sort_values('total', ascending = False).head(40)
trace0 = go.Bar(x = lt.No[::-1], y = lt.index[::-1], name = 'No',
              orientation = 'h')
trace1 = go.Bar(x = lt.Yes[::-1], y = lt.index[::-1], name = 'Yes',
              orientation = 'h')

data = [trace0, trace1]

layout = go.Layout(barmode = 'stack', title = 'Kiva influence on loan themes',
                   height = 900,
                   margin = dict(l = 155, t = 100),
                   xaxis = dict(tickfont = dict(size = 11)),
                   yaxis = dict(tickfont = dict(size = 10),
                               )
                  )

fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
avg_am = df.groupby(['Sub-national region', 'Country', 'region MPI'])
lsum = avg_am['amount'].sum()
lcount = avg_am['number'].sum()
lavg = (lsum / lcount).round(2)
avg_am = pd.DataFrame({'loan amounts': lsum, 'total loans': lcount, 'average loan amount': lavg}).reset_index()
avg_am.head(10)
hover_text = []
for index, row in avg_am.iterrows():
    hover_text.append(('Total loans: {a}<br>' + 
                       'Average loan amount: {b}<br>' +
                       'Region MPI: {c}<br>' +
                       'Country: {d}<br>').format(a = row['total loans'], 
                                                  b = '$' + str("{:,}".format(row['average loan amount'])),
                                                  c = row['region MPI'],
                                                  d = row['Country'])
                     )

avg_am['info'] = hover_text
l100 = avg_am[avg_am['total loans'] <= 100].sort_values(['average loan amount'], ascending = False).head(20)
trace = go.Bar( x = l100['Sub-national region'],
                y = l100['average loan amount'],
               text = l100['total loans'],
                textposition = 'outside',
                hoverinfo = 'text',
                hovertext = l100['info'],
                marker=dict(color='#B2F5A7',
                            line=dict(color='rgb(8,48,107)',
                                      width=1.5)
                           )
              )
        
layout = go.Layout( title = 'Average amount per region (number of loans <= 100)')
fig = go.Figure(data = [trace], layout = layout)
py.iplot(fig)
l1k = avg_am[(avg_am['total loans'] > 100) & (avg_am['total loans'] <= 1000)].sort_values(['average loan amount'], ascending = False).head(20)
trace = go.Bar( x = l1k['Sub-national region'],
                y = l1k['average loan amount'],
                text = l1k['total loans'],
                textposition = 'outside',
                hoverinfo = 'text',
                hovertext = l1k['info'],
                marker=dict(color='#54B644',
                            line=dict(color='rgb(8,48,107)',
                                      width=1.5)
                           )
              )
        
layout = go.Layout( title = 'Average amount per region (number of loans <= 1000)',
                  margin = dict(b = 135)
                  )
fig = go.Figure(data = [trace], layout = layout)
py.iplot(fig)
l10k = avg_am[avg_am['total loans'] > 1000].sort_values(['average loan amount'], ascending = False).head(20)
trace = go.Bar( x = l10k['Sub-national region'],
                y = l10k['average loan amount'],
                text = l10k['total loans'],
                textposition = 'outside',
                hoverinfo = 'text',
                hovertext = l10k['info'],
                marker=dict(color='#137503',
                            line=dict(color='rgb(8,48,107)',
                                      width=1.5)
                           )
              )
        
layout = go.Layout( title = 'Average amount per region (number of loans > 1000)',
                   height = 600,
                   margin = dict(b = 155, r = 100)                  
                  )
fig = go.Figure(data = [trace], layout = layout)
py.iplot(fig)
coun_stats = pd.read_csv('../input/mpi-on-regions/country_stats.csv') # from beluga 
kl = pd.read_csv('../input/mpi-on-regions/all_kiva_loans.csv') # the 1.4 Mil entries downloaded from Kiva
kl['not_funded'] = kl['loan_amount'] - kl['funded_amount']
nf = kl[kl['not_funded'] != 0].reset_index(drop=True)
nf.describe()
nf[nf.status == 'funded']
nf[nf.not_funded < 0]
nf['not_funded_percent'] = nf['not_funded'] / nf['loan_amount'] * 100
nf['not_funded_percent'] = round(nf['not_funded_percent'], 2)
nf['borrower_genders']=[elem if elem in ['female','male'] else 'group' for elem in nf['borrower_genders'] ]
borr = nf['borrower_genders'].value_counts()

pie1 = go.Pie( labels = ['Funded amount', 'Not funded amount'],
             values = [44457755, 45720020],
             hoverinfo = 'label+value+percent',
             textfont=dict(size=18, color='#000000'),
             name = "Loans that didn't get the funds",
             domain = dict( x=[0, 0.5] )
            )
pie2 = go.Pie( labels = borr.index,
             values = borr.values,
             hoverinfo = 'label+value+percent',
             textfont=dict(size=18, color='#000000'),
             text = "Distribution of genders",
             domain = dict( x=[0.5, 1] )
            )

layout = go.Layout(showlegend=True, title = 'Amount and gender distribution for expired loans')

fig = go.Figure(data=[pie1, pie2], layout=layout)
py.iplot(fig)
sta = nf['status'].value_counts()
rep = nf['repayment_interval'].value_counts()

trace0 = go.Bar(x = sta.index, y = sta.values,
               marker = dict(color = sta.values,
                            colorscale = 'Viridis'
                            ),
                name = 'Status'
               )

trace1 = go.Bar(x = rep.index, y = rep.values,
               marker = dict(color = rep.values,
                            colorscale = 'Viridis'
                            ),
                name = 'Repayment interval'
               )

fig = tls.make_subplots(rows=1, cols=2, subplot_titles=( 'Status distribution for<br>partialy funded loans', 'Repayment interval for<br>partialy funded loans'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(showlegend = False, width = 800)

py.iplot(fig)
nf_exp = nf[nf['status'] == 'expired']
country_sum = nf_exp.groupby(['country_name'])['not_funded'].sum().sort_values().tail(30)

trace = go.Bar(x = country_sum.values, 
               y = country_sum.index,
               orientation = 'h',
               marker = dict(color = country_sum.values,
                             colorscale = 'Viridis',
                             reversescale = True
                            )
              )
layout = go.Layout(title = 'Amount of unfunded loans per country',
                   margin = dict(l = 150), height = 750
                  )
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
act = nf_exp['activity_name'].value_counts()[:30]
sec = nf_exp['sector_name'].value_counts()

activity = go.Bar(x = act.index, y = act.values,
               marker = dict(color = act.values,
                            colorscale = 'Portland'
                            ),
               )

sector = go.Bar(x = sec.index, y = sec.values,
               marker = dict(color = sec.values,
                            colorscale = 'Portland'
                            ),
               )

fig = tls.make_subplots(rows=2, cols=1, subplot_titles=( "Top 30 activities that didn't get loans", "Sectors that didn't get loans"))
fig.append_trace(activity, 1, 1)
fig.append_trace(sector, 2, 1)

fig['layout'].update(showlegend = False, height=900)

py.iplot(fig)
S = nf_exp['sector_name'].unique()
N = len(list(S))
C = [ 'hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, N) ] # create your own 'rainbow' palette (source: plotly website) 

trace = []
for s,c in zip(S,C):
    trace.append(go.Box(y = nf_exp[nf_exp['sector_name'] == s]['not_funded'],
                        marker = dict(color = c),
                        name = s
                        )
                 )
layout = go.Layout(title = 'Amount of unfunded loans per sectors',
                   yaxis = dict(type = 'log')
                  )

fig = go.Figure(data=trace, layout=layout)
py.iplot(fig)
sec_f = nf_exp.groupby(['sector_name'])['funded_amount'].sum()
sec_nf = nf_exp.groupby(['sector_name'])['not_funded'].sum()

first = go.Bar( x = sec_f.index, y = sec_f.values,
               name = 'Funded amount of loan'              
              )

second = go.Bar( x = sec_nf.index, y = sec_nf.values,
               name = 'Not funded amount of loan'              
              )

data = [first, second]
layout = go.Layout( barmode = 'group', 
                  title = 'Funded vs not funded amounts per sector'
                  )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

big_act = nf_exp.groupby(['activity_name'])['loan_amount'].sum().sort_values(ascending=False).head(50)
act_f = [ nf_exp[nf_exp['activity_name'] == i]['funded_amount'].sum() for i in big_act.index ]
act_nf = [ nf_exp[nf_exp['activity_name'] == i]['not_funded'].sum() for i in big_act.index ]

first = go.Bar( x = big_act[:30].index, y = act_f[:30],
               name = 'Funded amount of loan'              
              )

second = go.Bar( x = big_act[:30].index, y = act_nf[:30],
               name = 'Not funded amount of loan'              
              )

data = [first, second]
layout = go.Layout( barmode = 'group', 
                  title = 'Funded vs not funded amounts per activity<br>(top 30 by amount of loans requested)',
                   showlegend=False,
                   margin = dict(b=175),
                   #bargap = 0.35
                  )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
fund = np.log(nf_exp['funded_amount']) + 1
nfund = np.log(nf_exp['not_funded']) + 1

trace1 = go.Histogram(x=nf_exp['funded_amount'], nbinsx=50, opacity=0.75, name='Funded amount')
trace2 = go.Histogram(x=nf_exp['not_funded'], nbinsx=50, opacity=0.75, name='Not funded amount')

trace3 = go.Histogram(x=fund, nbinsx=50, opacity=0.75, name='Funded amount')
trace4 = go.Histogram(x=nfund, nbinsx=50, opacity=0.75, name='Not funded amount')

fig = tls.make_subplots(rows=1, cols=2, subplot_titles = ('Normal distribution', 'Logarithmic distribution'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(title='Funded vs Unfunded amount', showlegend=False)
py.iplot(fig)

#nf_exp['not_funded'].describe()
fun = np.array(nf_exp['funded_amount'])
mean = np.mean(fun, axis=0)
sd = np.std(fun, axis=0)

fun_noq = [x for x in fun if (x > mean - 2*sd)]
fun_noq = [x for x in fun_noq if (x < mean + 2*sd)]
nofun = np.array(nf_exp['not_funded'])
mean = np.mean(nofun, axis=0)
sd = np.std(nofun, axis=0)

nofun_noq = [x for x in nofun if (x > mean - 2*sd)]
nofun_noq = [x for x in nofun_noq if (x < mean + 2*sd)]
fun_noq_log = np.log(fun_noq) + 1
nofun_noq_log = np.log(nofun_noq) + 1

trace1 = go.Histogram(x=fun_noq, nbinsx=50, opacity=0.75, name='Funded amount')
trace2 = go.Histogram(x=nofun_noq, nbinsx=50, opacity=0.75, name='Not funded amount')

trace3 = go.Histogram(x=fun_noq_log, nbinsx=50, opacity=0.75, name='Funded amount')
trace4 = go.Histogram(x=nofun_noq_log, nbinsx=50, opacity=0.75, name='Not funded amount')

fig = tls.make_subplots(rows=1, cols=2, subplot_titles = ('Normal distribution', 'Logarithmic distribution'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig['layout'].update(title='Funded vs Unfunded amount<br>(expired loans with removed quartiles)', showlegend=False)
py.iplot(fig)
# sns.axes_style('white')
sns.jointplot(x = nf_exp['not_funded'], y = nf_exp['not_funded_percent'], data=nf_exp, kind='hex', color='blue').set_axis_labels(
    'Amount not funded', 'Percent of the unfunded loan')
sns.jointplot(x = pd.Series(nofun_noq), y = nf_exp['not_funded_percent'], kind='hex', color='red').set_axis_labels(
    'Amount not funded (data without quartiles)', 'Percent of the unfunded loan')




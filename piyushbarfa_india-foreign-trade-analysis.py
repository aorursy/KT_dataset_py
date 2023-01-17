from IPython.display import Image

Image('/kaggle/input/foreign-trade/foreign trade.png')


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# charts

import seaborn as sns 

import matplotlib.pyplot as plt

import squarify #TreeMap



# import graph objects as "go"



import plotly.graph_objs as go

%matplotlib inline



#ignore warning 

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Display markdown formatted output like bold, italic bold etc.

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))

    

# Required to use plotly offline in jupyter notebook    

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True)
'''Read in export and import data from CSV file'''



%time data_import = pd.read_csv('/kaggle/input/indian-export-and-import-data-since-1997/Import_Clean')

%time data_export = pd.read_csv('/kaggle/input/indian-export-and-import-data-since-1997/Export_Clean')
bold('**Preview of Export Data:**')

display(data_export.sample(n=5))

display('Export DataSet',data_export.shape)

bold('**Preview of Import Data:**')

display(data_import.sample(n=5))

display('Import DataSet',data_import.shape)
'''Variable Description'''

def description(df):

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values

    return summary
bold('**Variable Description of export dataset:**')

display(description(data_export))



bold('**Variable Description of import dataset:**')

display(description(data_import))
data_import['Year'] = data_import.Year.apply(lambda x: x[:4])

data_export['Year'] = data_export.Year.apply(lambda x: x[:4])

data_import['Year'] = pd.to_numeric(data_import['Year'])

data_export['Year'] = pd.to_numeric(data_export['Year'])
"""Let's see if export and import data contain the zero and NAN values """

bold('**Export Data with zeros:**')

display(data_export[data_export.Value_of_Export == 0].head(3))

bold('**Import Data with zeros:**')

display(data_import[data_import.Value_of_Import == 0].head(3))

bold('**Export Data with NAN:**')

display(data_export.isnull().sum())

bold('**Import Data with NAN:**')

display(data_import.isnull().sum())
'''Imputing the missing valriable'''

print('Size of export data before cleaning',data_export.shape)

print('Size of import data before cleaning',data_import.shape)

data_import = data_import.dropna()

data_import['Country'] = data_import['Country'].replace({'U S A': 'USA'})

data_import = data_import.reset_index(drop=True)



df_export = data_export.dropna()

data_export['Country'] = data_export['Country'].replace({'U S A': 'USA'})

df_export = data_export.reset_index(drop=True)

print('Size of export data after cleaning',data_export.shape)

print('Size of import data after cleaning',data_import.shape)
print("Import Commodity Count : "+str(len(data_import['2HSCode'].unique())))

print("Export Commodity Count : "+str(len(data_export['2HSCode'].unique())))
print("No of Country were we are importing Comodities are "+str(len(data_import['Country'].unique())))

print("No of Country were we are Exporting Comodities are "+str(len(data_export['Country'].unique())))
df3 = data_import.groupby('Year').agg({"Value_of_Import":'sum'})

df4 = data_export.groupby('Year').agg({"Value_of_Export":'sum'})

df3['Growth_Import']=df3.pct_change()

df4['Growth_Export']=df4.pct_change()

'''Calculating trade deficit'''

df3['deficit'] = df4.Value_of_Export - df3.Value_of_Import

total_year = pd.concat([df4, df3], axis = 1)



bold('**Export/Import and Trade Balance of India**')

display(total_year)

bold('**Descriptive statistics**')

display(total_year.describe())
# create trace1 

trace1 = go.Bar(

                x = df3.index,

                y = df3.Value_of_Import,

                name = "Import",

                marker = dict(color = 'rgba(0,191,255, 1)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df3.Value_of_Import)

# create trace2 

trace2 = go.Bar(

                x = df4.index,

                y = df4.Value_of_Export,

                name = "Export",

                marker = dict(color = 'rgba(1, 255, 130, 1)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df4.Value_of_Export)



trace3 = go.Bar(

                x = df3.index,

                y = df3.deficit,

                name = "Trade Deficit",

                marker = dict(color = 'rgba(220, 20, 60, 1)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df3.deficit)





data = [trace1, trace2, trace3]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

fig.update_layout(

    title=go.layout.Title(

        text="Yearwise Import/Export/Trade deficit",

        xref="paper",

        x=0

    ),

    xaxis=go.layout.XAxis(

        title=go.layout.xaxis.Title(

            text="Year",

            font=dict(

                family="Courier New, monospace",

                size=18,

                color="#7f7f7f"

            )

        )

    ),

    yaxis=go.layout.YAxis(

        title=go.layout.yaxis.Title(

            text="Value",

            font=dict(

                family="Courier New, monospace",

                size=18,

                color="#7f7f7f"

            )

        )

    )

)









fig.show()
df5 = data_import.groupby('Country').agg({'Value_of_Import':'sum'})

df5 = df5.sort_values(by='Value_of_Import', ascending = False)

df5 = df5[:10]



df6 = data_export.groupby('Country').agg({'Value_of_Export':'sum'})

df6 = df6.sort_values(by='Value_of_Export', ascending = False)

df6 = df6[:10]
sns.set(rc={'figure.figsize':(15,6)})

ax1 = plt.subplot(121)



sns.barplot(df5.Value_of_Import,df5.index).set_title('Country Wise Import')



ax2 = plt.subplot(122)

sns.barplot(df6.Value_of_Export,df6.index).set_title('Country Wise Export')

plt.tight_layout()

plt.show()
fig = go.Figure()

# Create and style traces

fig.add_trace(go.Scatter(x=df3.index, y=df3.Value_of_Import, name='Import',mode='lines+markers',

                         line=dict(color='firebrick', width=4)))

fig.add_trace(go.Scatter(x=df4.index, y=df4.Value_of_Export, name = 'Export',mode='lines+markers',

                         line=dict(color='royalblue', width=4)))

fig.update_layout(

    title=go.layout.Title(

        text="Yearwise Import/Export",

        xref="paper",

        x=0

    ),

    xaxis=go.layout.XAxis(

        title=go.layout.xaxis.Title(

            text="Year",

            font=dict(

                family="Courier New, monospace",

                size=18,

                color="#7f7f7f"

            )

        )

    ),

    yaxis=go.layout.YAxis(

        title=go.layout.yaxis.Title(

            text="Value",

            font=dict(

                family="Courier New, monospace",

                size=18,

                color="#7f7f7f"

            )

        )

    )

)



fig.show()
''' creating a new dataframe on Sections of HSCode'''

HSCode = pd.DataFrame() 

HSCode['Start']=[1,6,15,16,25,28,39,41,44,47,50,64,68,71,72,84,86,90,93,94,97]

HSCode['End']=[5,14,15,24,27,38,40,43,46,49,63,67,70,71,83,85,89,92,93,96,99]

HSCode['Sections Name']=['Animals & Animal Products',

'Vegetable Products',

'Animal Or Vegetable Fats',

'Prepared Foodstuffs',

'Mineral Products',

'Chemical Products',

'Plastics & Rubber',

'Hides & Skins',

'Wood & Wood Products',

'Wood Pulp Products',

'Textiles & Textile Articles',

'Footwear, Headgear',

'Articles Of Stone, Plaster',

'Pearls, Precious Stones, Metals',

'Base Metals & Articles Thereof',

'Machinery & Mechanical Parts',

'Transportation Equipment',

'Instruments-Measuring, Musical',

'Arms & Ammunition',

'Miscellaneous',

'Works Of Art',]

HSCode.index += 1

HSCode.index.name = 'Section'

bold('**List Of indian HS Classification is based on the HS Code:**')

display(HSCode)
data_export['Sections Name'] = data_export['2HSCode']

data_import['Sections Name'] = data_import['2HSCode']

for i in range(1,22):

    data_export.loc[(data_export['Sections Name'] >= HSCode['Start'][i]) & (data_export['Sections Name'] <= HSCode['End'][i]),'Sections Name']=i

    data_import.loc[(data_import['Sections Name'] >= HSCode['Start'][i]) & (data_import['Sections Name'] <= HSCode['End'][i]),'Sections Name']=i



exp_hscode = data_export.groupby(['Sections Name']).agg({'Value_of_Export':'sum'})

exp_hscode['Sections Name'] = HSCode['Sections Name']

exp_hscode = exp_hscode.sort_values(['Value_of_Export']).reset_index(drop=True)

imp_hscode = data_import.groupby(['Sections Name']).agg({'Value_of_Import':'sum'})

imp_hscode = imp_hscode.sort_values(['Value_of_Import']).reset_index(drop=True)

imp_hscode['Sections Name'] = HSCode['Sections Name']
'''Visualization of Export/Import HS Classification is based'''

def bar_plot(y,x, label, color):

    global ax

    font_size = 20

    title_size =30

    plt.rcParams['figure.figsize'] = (15,9)

    ax = sns.barplot(x, y, palette = color)

    ax.set_xlabel(xlabel = 'USD (millions)', fontsize = font_size)

    ax.set_ylabel(ylabel = 'Sections Name', fontsize = font_size)

    ax.set_title(label = label, fontsize = title_size)

    plt.xticks(rotation = 90, fontsize=15)

    plt.yticks(fontsize=15)

    plt.show()



bar_plot(exp_hscode['Sections Name'], exp_hscode.Value_of_Export, 'Export of India (HS Classification is based)', 'Paired')

bar_plot(imp_hscode['Sections Name'], imp_hscode.Value_of_Import, 'Import of India (HS Classification is based)', 'Set1')
expensive_import = data_import[data_import.Value_of_Import>100000]

df =data_import.groupby(['2HSCode']).agg({'Value_of_Import': 'sum'})

df = df.sort_values(by='Value_of_Import')
value=np.array(df)

commodityCode=df.index

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (10.0, 10.0)

squarify.plot(sizes=value, label=commodityCode, alpha=.7 )

plt.axis('off')

plt.title("Imports HsCode Share")

plt.show()
df1 = data_import.groupby(['Country']).agg({'Value_of_Import': 'sum'})

df1 = df1.sort_values(by='Value_of_Import')
value=np.array(df1)

country=df1.index

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (10.0, 10.0)

squarify.plot(sizes=value, label=country, alpha=.7 )

plt.title("Imports Countrywise Share")

plt.axis('off')

plt.show()
export_map = pd.DataFrame(df_export.groupby(['Country'])['Value_of_Export'].sum().reset_index())

count = pd.DataFrame(export_map.groupby('Country')['Value_of_Export'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'algae',

            locationmode = 'country names',

            locations = count['Country'],

            text = count['Country'],

            z = count['Value_of_Export'],

            reversescale=True)]



layout = go.Layout(title = 'India Export to Other Country')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)

import_map = pd.DataFrame(data_import.groupby(['Country'])['Value_of_Import'].sum().reset_index())

count = pd.DataFrame(import_map.groupby('Country')['Value_of_Import'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'amp',

            locationmode = 'country names',

            locations = count['Country'],

            text = count['Country'],

            z = count['Value_of_Import'],

            reversescale=True)]



layout = go.Layout(title = 'India Import from Other Country')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)
df_final_trade = pd.concat([data_export, data_import])

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (12, 12)

wordcloud = WordCloud(background_color = 'gray', width = 1200,  height = 1200, max_words = 100).generate(' '.join(df_final_trade['Country']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Country in Trade With India',fontsize = 30)

plt.show()
def country_graph (Country, title):

    print(title)

    df_e = data_export[data_export.Country == Country]

    df_i = data_import[data_import.Country == Country]

    df_i = df_i.groupby('Year').agg({"Value_of_Import":'sum'})

    df_e = df_e.groupby('Year').agg({"Value_of_Export":'sum'})

    df_i['Growth_Import']=df_i.pct_change()

    df_e['Growth_Export']=df_e.pct_change()

    '''Calculating trade deficit'''

    df_i['deficit'] = df_e.Value_of_Export - df_i.Value_of_Import

    total_year = pd.concat([df_e, df_i], axis = 1)



    #print('Export/Import and Trade Balance of India')

    #display(total_year)

    #bold('**Descriptive statistics**')

    #display(total_year.describe())

    # create trace1 

    trace1 = go.Bar(x = df_i.index, y = df_i.Value_of_Import, name = "Import", 

                    marker = dict(color = 'rgba(0,191,255, 1)', line=dict(color='rgb(0,0,0)',width=1.5)),

                    text = df_i.Value_of_Import)

    # create trace2 

    trace2 = go.Bar(x = df_e.index, y = df_e.Value_of_Export, name = "Export",marker = dict(color = 'rgba(1, 255, 130, 1)',

                    line=dict(color='rgb(0,0,0)',width=1.5)),text = df_e.Value_of_Export)



    trace3 = go.Bar( x = df_i.index, y = df_i.deficit, name = "Trade Deficit", marker = dict(color = 'rgba(220, 20, 60, 1)',

                    line=dict(color='rgb(0,0,0)',width=1.5)), text = df_i.deficit)



    data = [trace1, trace2, trace3]

    layout = go.Layout(barmode = "group")

    fig = go.Figure(data = data, layout = layout)

    fig.update_layout(title=go.layout.Title( text="Yearwise Trade (Import/Export/Trade deficit)", xref="paper", x=0 ),

        xaxis=go.layout.XAxis( title=go.layout.xaxis.Title( text="Year", font=dict( family="Courier New, monospace",size=18, color="#7f7f7f"))),

        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Value",font=dict(family="Courier New, monospace",size=18, color="#7f7f7f")))

    )

    fig.show()

    fig = go.Figure()

    '''

        # Create and style traces

    fig.add_trace(go.Scatter(x=df_i.index, y=df_i.Value_of_Import, name='Import',mode='lines+markers', line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Scatter(x=df_e.index, y=df_e.Value_of_Export, name = 'Export',mode='lines+markers',line=dict(color='royalblue', width=4)))

    fig.update_layout(title=go.layout.Title(text="Yearwise Trade Import/Export", xref="paper",x=0),

    xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Year",font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))),

    yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Value", font=dict( family="Courier New, monospace",size=18,color="#7f7f7f"))))

    fig.show()'''
def bar_plot(Country, title):

    print(title)

    df_e = data_export[data_export.Country == Country]

    df_i = data_import[data_import.Country == Country]

    df_e['Sections Name'] = df_e['2HSCode']

    df_i['Sections Name'] = df_i['2HSCode']

    for i in range(1,22):

        df_e.loc[(df_e['Sections Name'] >= HSCode['Start'][i]) & (df_e['Sections Name'] <= HSCode['End'][i]),'Sections Name']=i

        df_i.loc[(df_i['Sections Name'] >= HSCode['Start'][i]) & (df_i['Sections Name'] <= HSCode['End'][i]),'Sections Name']=i

    exp_hscode = df_e.groupby(['Sections Name']).agg({'Value_of_Export':'sum'})

    exp_hscode['Sections Name'] = HSCode['Sections Name']

    exp_hscode = exp_hscode.sort_values(['Value_of_Export']).reset_index(drop=True)

    imp_hscode = df_i.groupby(['Sections Name']).agg({'Value_of_Import':'sum'})

    imp_hscode = imp_hscode.sort_values(['Value_of_Import']).reset_index(drop=True)

    imp_hscode['Sections Name'] = HSCode['Sections Name']

    global ax

    font_size = 20

    title_size =30

    plt.rcParams['figure.figsize'] = (15,9)

    ax = sns.barplot(exp_hscode.Value_of_Export, exp_hscode['Sections Name'], palette = 'Paired')

    ax.set_xlabel(xlabel = 'USD (millions)', fontsize = font_size)

    ax.set_ylabel(ylabel = 'Sections Name', fontsize = font_size)

    ax.set_title(label = 'Export of India (HS Classification is based)', fontsize = title_size)

    plt.xticks(rotation = 90, fontsize=15)

    plt.yticks(fontsize=15)

    plt.show()

    

    plt.rcParams['figure.figsize'] = (15,9)

    ax = sns.barplot(imp_hscode.Value_of_Import, imp_hscode['Sections Name'], palette = 'Set1')

    ax.set_xlabel(xlabel = 'USD (millions)', fontsize = font_size)

    ax.set_ylabel(ylabel = 'Sections Name', fontsize = font_size)

    ax.set_title(label = 'Import of India (HS Classification is based)', fontsize = title_size)

    plt.xticks(rotation = 90, fontsize=15)

    plt.yticks(fontsize=15)

    plt.show()

    

Country = 'CHINA P RP'

title = 'China'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'USA'

title = 'USA'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'U ARAB EMTS'

title = 'U ARAB EMTS'

country_graph(Country, title)
Country = 'SAUDI ARAB'

title = 'Saudi Arab'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'SWITZERLAND'

title = 'SWITZERLAND'

country_graph(Country, title)
Country = 'FRANCE'

title = 'FRANCE'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'U ARAB EMTS'

title = 'U ARAB EMTS'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'U K'

title = 'U K'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'AUSTRALIA'

title = 'AUSTRALIA'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'INDONESIA'

title = 'INDONESIA'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'SINGAPORE'

title = 'SINGAPORE'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'HONG KONG'

title = 'HONG KONG'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'IRAQ'

title = 'IRAQ'

country_graph(Country, title)

bar_plot(Country, title)
Country = 'KOREA RP'

title = 'KOREA RP'

country_graph(Country, title)

bar_plot(Country, title)
gdp = pd.read_csv('/kaggle/input/india-gdp-growth-world-bank-1961-to-2017/India GDP from 1961 to 2017.csv', index_col = 0)

gdp.rename({'1961':'Year','3.722742533':'gdp'}, inplace=True, axis = 1)

#gdp.set_index('Year')

gdp.drop(gdp.head(35).index, inplace=True)

gdp.info()

gdp
gdp_year= pd.concat((df3,df4,gdp),axis=1)
gdp_year
#sns.scatterplot(x='gdp', y='Value_of_Export', data = gdp_year)



#sns.lineplot(y=gdp_year.Value_of_Import, x=gdp_year.index)

#sns.lineplot(y = gdp_year.gdp, x = gdp_year.index)
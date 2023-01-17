%matplotlib inline
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')

import pandas as pd
#from pandas_datareader import wb
pd.set_option("display.max_colwidth",200)

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff

import numpy as np

import nltk
#nltk.download('popular')
import pandas as pd

import pickle

from bs4 import BeautifulSoup
import re
#import os
#import codecs
from sklearn import feature_extraction
#import mpld3
from nltk.stem.snowball import SnowballStemmer

from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
#%% define function to plot world maps from series

def plot_world_map_from_series(df, filename = None):
    
    df = df.reset_index()
    # by default, first columns is country, second columns is value
    
    data = [ dict(
        type = 'choropleth',
        locations = df.ix[:,0],
        locationmode = 'country names',
        z = df.ix[:,1].astype('float'),
#        text = df.ix[:,0].str.cat( df.ix[:,1].astype('str'), sep = ' '),
        text = df.ix[:,0], 
#        colorscale = 'Blues',
        autocolorscale = True,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            tickprefix = '',
            #title = df.columns[1]
        )
      ) ]

    layout = dict(
        title = df.columns[1],
        geo = dict(
            showframe = False,
            showcoastlines = False,
            projection = dict(
                type = 'Mercator'
            )
        )
    )

    if filename == None:
        filename = df.columns[1]

    fig = dict( data=data, layout=layout )
    return py.iplot( fig, validate = True, filename = filename)
    
#%% plot horizontal bars from series
def plot_barh_from_series(df, filename = None):
    
    df = df.reset_index()
    # by default, first columns is country, second columns is value
    
    trace = go.Bar(
        y= df.ix[:,0],
        x=df.ix[:,1],
        orientation = 'h',
        marker=dict(
            color=df.ix[:,1],
            autocolorscale = True,
            reversescale = False
        ),
    )
    
    layout = go.Layout(
        title= df.columns[1],
        width=800,
        height=1200,
        )
    data = [trace]
    
    fig = go.Figure(data=data, layout=layout)
    
    if filename == None:
        filename = df.columns[1]
    
    return py.iplot(fig, filename= filename)

#%% plot correlation matrix from pandas frame
def plot_correlation_matrix(corr, xcols = None, ycols = None, filename = None, title = None):
    # corr is the correlation matrix obtained from a dataframe using pandas
    
    if xcols == None:
        xcols = corr.columns.tolist()
    if ycols == None:
        ycols = corr.columns.tolist()
    
    layout = dict(
        title = title,
        width = 800,
        height = 800,
#        margin=go.Margin(l=100, r=10, b=50, t=50, pad=5),
        margin=go.Margin(l=250, r=50, b=50, t=250, pad=4),
        yaxis= dict(tickangle=-30,
                    side = 'left',
                    ),
        xaxis= dict(tickangle=-30,
                    side = 'top',
                    ),
    )
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x= xcols,
        y= ycols,
        colorscale='Portland',
        reversescale=True,
        showscale=True,
        font_colors = ['#efecee', '#3c3636'])
    fig['layout'].update(layout)
    
    if filename == None:
        filename = 'correlation matrix'
    return py.iplot(fig, filename= filename)
#%% load data from kiva
data_kvloans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
# using mpi from OPHI. it has been shown that mpi data from kiva is prone to inaccuracy, in particular in GPS locations
# https://www.kaggle.com/marcomarchetti/ophi-subnat-decomp-headcount-mpi-201718/data
data_mpi = pd.read_csv('../input/ophi-subnat-decomp-headcount-mpi-201718/OPHI_SubNational_Decomposition_HeadCount_MPI_2017-18.csv', index_col = None, nrows =993, encoding = 'latin1', usecols = ['ISO_Country_Code', 'Country', 'Sub_national_region', 'World_region', 'Survey', 'Year', 'Population_Share', 'Country_MPI', 'Region_MPI', 'Schooling', 'Child_school_attendance', 'Child_mortality ', 'Nutrition', 'Electricity', 'Improved_sanitation', 'Drinking_water', 'Floor', 'Cooking _fuel', 'Asset_ownership', 'Num_of_indic', 'Indic_missing'])
data_kvmpi = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
data_kvmpi.dropna(axis= 0, thresh = 2, inplace = True)
# column of year when loan funded
data_kvloans['year']  = pd.to_datetime(data_kvloans['date']).dt.year.astype(str)

# change name of Cote d'Ivoire
data_kvloans['country'] = data_kvloans['country'].str.replace("Cote D'Ivoire","Cote d'Ivoire")
data_wb = pd.read_csv('../input/datawb/data_wb.csv')
data_wb['year'] = data_wb['year'].astype(str)
data_wb.set_index(['country','year'], inplace = True)
# data poverty
data_poverty = pd.read_csv('../input/datawb/data_poverty.csv')
data_poverty['year'] = data_poverty['year'].astype(str)
data_poverty.set_index(['country','year'], inplace = True)
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
#%% put all necessary data into a single frame
df1 = data_kvloans.groupby(by = ['country', 'year'])['loan_amount'].mean()
df1.name = 'Mean loan amount'

df2 = data_kvloans.groupby(by = ['country', 'year'])['loan_amount'].sum()
df2.name = 'Total loan amount'

# number of loans in each country in each years
df3 = data_kvloans.groupby(by = ['country', 'year'])['loan_amount'].size()
df3.name = '# loans'

df = pd.concat( [data_wb, df1, df2, df3 ], axis = 1, join = 'outer' )

df['# loans in 10000 inhabitants'] = 10000.*df['# loans'] /df['Total population']

df['Mean loan amount / GDP per capita (current US$) (%)'] = 100.* df['Mean loan amount'] / df['GDP per capita (current US$)']

plot_correlation_matrix( df.corr().round(2))
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
dfaux = df.reset_index().groupby('country')['# loans'].mean()
dfaux = dfaux.fillna(value = 0)
plot_barh_from_series( dfaux.sort_values(ascending = False).head(40) )

l1 = dfaux.sort_values(ascending = False).index.tolist()
df.ix[ l1[:5]]
plot_world_map_from_series( dfaux )
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
dfaux = df.reset_index().groupby('country')['Total population'].mean().loc[l1]

plot_barh_from_series( dfaux.head(30) )
dfaux = df.reset_index().groupby('country')['# loans in 10000 inhabitants'].mean().loc[l1]

plot_barh_from_series( dfaux.head(30) )
dfaux = df.reset_index().groupby('country')['# loans in 10000 inhabitants'].mean()
# remove Samoa (too different)
dfaux = dfaux[ ~dfaux.index.isin([ 'Samoa', 'El Salvador'] ) ]
# replace nan with 0 (no loan)
dfaux = dfaux.fillna(value = 0)
plot_world_map_from_series( dfaux )
df.ix[ ['Samoa', 'El Salvador']]
# join mean values of two dataframes (over different periods, assumption that the trend stays the same for poverty data)
df_pov = pd.concat([df.reset_index().groupby('country').mean(), data_poverty.reset_index().groupby('country').mean(), data_kvmpi.groupby('country')['MPI'].mean() ], axis = 1, join = 'outer')
df_pov[u'# loans in 10000  poor inhabitants'] = 10000. * df_pov[u'# loans'] / (df_pov[u'Total population'] * df_pov[u'Poverty headcount ratio at national poverty lines (% of population)'] / 100.)
plot_barh_from_series( df_pov.loc[l1[:30],u'# loans in 10000  poor inhabitants'])
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
dfaux = df.reset_index().groupby('country')['Mean loan amount'].mean()
dfaux = dfaux.fillna(value = 0.)
dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
plot_barh_from_series( dfaux.ix[l1[:30] ] )
plot_world_map_from_series( dfaux.loc[ ~(dfaux.index == "Cote d'Ivoire")]) # remove Cote d'Ivoire
data_kvloans.loc[  data_kvloans['country'] == "Cote d'Ivoire"]
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
dfaux = df.reset_index().groupby('country')['Mean loan amount / GDP per capita (current US$) (%)'].mean().loc[l1]
dfaux = dfaux.fillna(value = 0)

plot_barh_from_series( dfaux.ix[l1[:70]] , filename = 'mean loan vs. gdp per capita barh')
plot_world_map_from_series( dfaux, filename = 'mean loan vs. gdp per capita map')
xcols = ['Total population','GDP per capita (current US$)',u'Income share held by lowest 10%',u'Income share held by lowest 20%',u'Poverty gap at $1.90 a day (2011 PPP) (%)',u'Poverty gap at national poverty lines (%)',u'Poverty gap at $5.50 a day (2011 PPP) (% of population)',u'Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)',u'Poverty headcount ratio at $3.20 a day (2011 PPP) (% of population)',u'Poverty headcount ratio at $5.50 a day (2011 PPP) (% of population)',u'Poverty headcount ratio at national poverty lines (% of population)','MPI']
ycols = ['Total loan amount', '# loans','# loans in 10000 inhabitants', '# loans in 10000  poor inhabitants', 'Mean loan amount', 'Mean loan amount / GDP per capita (current US$) (%)']
plot_correlation_matrix( df_pov.corr().loc[ycols, xcols].round(2), xcols = xcols, ycols = ycols, filename = 'correlation matrix poverty' )
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
dfkv = data_kvloans[ [u'country', u'year', u'activity', u'sector', u'loan_amount']]
trace = go.Pie(sort = False,labels=dfkv.groupby('sector').size().index.tolist(), values=list(dfkv.groupby('sector').size().values))
fig = {
    'data': [trace],
    'layout': {'title': '# loans by sector'}
     }

py.iplot(fig)
trace = go.Pie(sort = False,labels=dfkv.groupby('sector')['loan_amount'].sum().index.tolist(), values=list(dfkv.groupby('sector')['loan_amount'].sum().values))
fig = {
    'data': [trace],
    'layout': {'title': 'Total loan amount by sector'}
     }

py.iplot(fig)
trace = go.Pie(sort = False, labels=dfkv.groupby('sector')['loan_amount'].mean().index.tolist(), values=list(dfkv.groupby('sector')['loan_amount'].mean().values))
fig = {
    'data': [trace],
    'layout': {'title': 'Mean loan amount by sector'}
     }

py.iplot(fig)
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
#%% % of each sector in each country

dfaux = dfkv.groupby(['country', 'sector']).sum()
dfaux = dfaux.unstack(level = -1)
dfaux.fillna(value = 0., inplace = True)


dfa = 100 * dfaux / dfaux.sum(axis = 0)

sectors = ['Agriculture', 'Arts', 'Clothing', 'Construction', 'Education', 'Entertainment', 'Food', 'Health', 'Housing', 'Manufacturing', 'Personal Use', 'Retail', 'Services', 'Transportation', 'Wholesale']

sectors = ['Agriculture', 'Clothing', 'Construction', 'Education', 'Entertainment', 'Food', 'Health', 'Housing',  'Services', 'Personal Use', 'Retail']


for sector in sectors:
    print('\n')
    print('Sector ' + sector + ' : Loan amount per country (% of worldwide total loan amount in this sector) ')

    dfaux = dfa.xs(sector, axis = 1, level = 1)
    dfaux.columns = [ 'Loan amount (% of worldwide total loan amount in the sector)']
    
    country_list = data_wb.index.get_level_values('country').unique().tolist()
    
    for i in country_list:
        if not(i in dfaux.index.tolist()):
            dfaux = dfaux.append( pd.DataFrame( index = [i], data = [0.0], columns = dfaux.columns ) )
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_world_map_from_series( dfaux )
    print(dfaux.sort_values(by =[ 'Loan amount (% of worldwide total loan amount in the sector)'], ascending = False).head(10))
    
    print(dfaux.sort_values(by =[ 'Loan amount (% of worldwide total loan amount in the sector)'], ascending = False).head(10).sum())    
    
    print('- - - - - - - - - - - - - - - - - -  - - - - - - - - - - - -  - - - - - - - - - - - -')
    print('\n')
#sectors = ['Agriculture', 'Arts', 'Clothing', 'Construction', 'Education', 'Food', 'Health', 'Housing', 'Manufacturing', 'Retail', 'Services', 'Transportation', 'Wholesale']
sectors = ['Agriculture',  'Food']


for sector in sectors:

    print('Sector ' + sector + ' : Distribution of loans in terms of number of loans, mean loan amount, mean amount / GDP per capita')

    # total loan amount per country per sector
    df1 = dfkv[ dfkv['sector'] == sector ].groupby(by = ['country', 'year'])['loan_amount'].sum()
    df1.name = 'Total loan amount'
    # mean loan amount per country per sector
    df2 = dfkv[ dfkv['sector'] == sector ].groupby(by = ['country', 'year'])['loan_amount'].mean()
    df2.name = 'Mean loan amount'
    
    # number of loans per country per sector
    df3 = dfkv[ dfkv['sector'] == sector ].groupby(by = ['country', 'year'])['loan_amount'].size()
    df3.name = '# loans'
    
    
    
    df = pd.concat([data_wb, df1, df2, df3 ], axis = 1, join = 'outer')

    df['# loans in 10000 inhabitants'] = 10000.*df['# loans'] /df['Total population']
    
    df['Mean loan amount / GDP per capita (current US$) (%)'] = 100.* df['Mean loan amount'] / df['GDP per capita (current US$)']
    

    # Mean loan amount /year
    dfaux = df.reset_index().groupby('country')['Mean loan amount'].mean()
    dfaux = dfaux.fillna(value = 0.)
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    #plot_barh_from_series( dfaux.sort_values(ascending = False).head(10) )
    plot_world_map_from_series( dfaux)
    print( dfaux.sort_values(ascending = False).head(10) )
    print('- - - - - - - - - - - - - - - ')
    print('\n')
    
    # Mean loan amount / gdp / year
    dfaux = df.reset_index().groupby('country')['Mean loan amount / GDP per capita (current US$) (%)'].mean()
    dfaux = dfaux.fillna(value = 0)
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_world_map_from_series( dfaux, filename = 'Mean loan vs. gdp per capita')
    print( dfaux.sort_values(ascending = False).head(10) )
    print('- - - - - - - - - - - - - - - ')
    print('\n')
    # mean number of loans per year
    dfaux = df.reset_index().groupby('country')['# loans'].mean()
    dfaux = dfaux.fillna(value = 0)
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_world_map_from_series( dfaux, filename = 'Mean number of loans per year' )
    print( dfaux.sort_values(ascending = False).head(10) )
    print('- - - - - - - - - - - - - - - ')
    print('\n')
    
    # total number of loans / 10000 habitants in all years
    dfaux = df.reset_index().groupby('country')['# loans in 10000 inhabitants'].mean()
    # remove Samoa (too different)
    dfaux = dfaux[ ~dfaux.index.isin([ 'Samoa'] ) ]
    # replace nan with 0 (no loan)
    dfaux = dfaux.fillna(value = 0)
    dfaux.drop(['Virgin Islands (U.S.)'], inplace = True)
    plot_world_map_from_series( dfaux , filename = 'N# loans in 10000 inhabitants')
    print( dfaux.sort_values(ascending = False).head(10) )
    
    print('- - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - ')
    print('\n')
dfaux = data_kvloans[ [u'country', u'year', u'activity', u'sector', u'loan_amount', u'borrower_genders']]
# replace 4221 missing values in gender by female (the most popular)
dfaux.loc[ dfaux['borrower_genders'].isnull() , 'borrower_genders']  = 'female'
dfaux.loc[ dfaux['borrower_genders'].str.contains('female'), 'borrower_genders' ] = 'female'
dfaux.loc[ ~dfaux['borrower_genders'].str.contains('female') , 'borrower_genders'] = 'male'
trace = go.Pie(sort = False,labels=dfaux.groupby('borrower_genders').size().index.tolist(), values=list(dfaux.groupby('borrower_genders').size().values))
fig = {
    'data': [trace],
    'layout': {'title': '# loans by gender'}
     }

py.iplot(fig)
trace = go.Pie(sort = False,labels=dfaux.groupby('borrower_genders')['loan_amount'].sum().index.tolist(), values=list(dfaux.groupby('borrower_genders')['loan_amount'].sum().values))
fig = {
    'data': [trace],
    'layout': {'title': 'Total loan amount by gender'}
     }

py.iplot(fig)
# frequency of borrowers by gender
dfaux2 = dfaux.groupby(by='country')['borrower_genders'].value_counts(normalize=True)
dfaux2.head(5)
plt.figure()
dfaux2.loc[:,'female'].plot.hist()
plt.xlabel('Female borrower (% of total number)')
plt.show()
plt.figure()
dfaux2.loc[:,'female'].plot(kind = 'box')
plt.ylabel('Female borrower (% of total number)')
plt.title('')
plt.show()
countrylist = data_wb.reset_index().set_index('country').index.unique()
dfaux3 = pd.concat( [ pd.DataFrame(index = countrylist), dfaux2.loc[:,'female'] ], axis = 1, join = 'outer' )
dfaux3.columns = ['Female borrower (% of total number)']
plot_world_map_from_series( dfaux3)
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
df = data_kvloans[ [ 'loan_amount', 'activity', 'sector', 'use', 'country_code', 'country', 'region', 'term_in_months', 'borrower_genders', 'repayment_interval','year'] ]
# add GDP per capita to the dataframe
df = pd.concat( [df, pd.DataFrame(columns = ['GDP per capita (current US$)'],index = df.index, data = [ np.nan ] * len(df) ) ], axis = 1, join = 'outer' )

      
for country, year in data_wb.reset_index()[[ 'country', 'year' ]].values:
    df.loc[(df['country'] == country) & (df['year'] == year), 'GDP per capita (current US$)'] = data_wb.loc[ country, year ]['GDP per capita (current US$)']


df['Loan amount / GDP per capita (%)' ]  = 100* df['loan_amount'] / df['GDP per capita (current US$)']

# MPI country
df = pd.concat( [df, pd.DataFrame(columns = ['MPI country'],index = df.index, data = [ np.nan ] * len(df) ) ], axis = 1, join = 'outer' )
for country in data_mpi[ 'Country'].unique().tolist():
    df.loc[ (df['country'] == country) , 'MPI country'] = data_mpi.loc[  (data_mpi['Country'] == country) , 'Country_MPI' ].unique()[0]
labels_sector = preprocessing.LabelEncoder()
labels_sector.fit( df['sector'].unique().tolist() )
df.loc[:,'sector label'] = labels_sector.transform( df['sector'] )
pd.DataFrame.from_dict( { 'Sector': df['sector'].unique().tolist(), 'Sector label': list(labels_sector.transform( df['sector'].unique().tolist() ) ) }) 
labels_repayment = preprocessing.LabelEncoder()
labels_repayment.fit( df['repayment_interval'].unique().tolist() )
df.loc[:,'repayment interval label'] = labels_repayment.transform( df['repayment_interval'] )
pd.DataFrame.from_dict( { 'Repayment interval': df['repayment_interval'].unique().tolist(), 'Repayment interval label': list(labels_repayment.transform( df['repayment_interval'].unique().tolist() ) ) }) 
# replace missing values by female (the most popular)
df.loc[ df['borrower_genders'].isnull() , 'borrower_genders']  = 'female'
df.loc[ df['borrower_genders'].str.contains('female'), 'borrower_genders' ] = 'female'
df.loc[ ~df['borrower_genders'].str.contains('female') , 'borrower_genders'] = 'male'
labels_gender = preprocessing.LabelEncoder()
labels_gender.fit( df['borrower_genders'].unique().tolist() )
df.loc[:,'gender label'] = labels_gender.transform( df['borrower_genders'] )
pd.DataFrame.from_dict( { 'Gender': df['borrower_genders'].unique().tolist(), 'Gender label': list(labels_gender.transform( df['borrower_genders'].unique().tolist() ) ) }) 
# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")
#print(stemmer)


# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens
#df['use'] = df['use'].str.replace('buy','purchase')
text = df['activity'].values.astype(str).tolist() # convert nan cell to string
# create a dictionary of tokens and stems
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in text:
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
totalvocab_tokenized = list( set( totalvocab_tokenized  ) )

totalvocab_stemmed = [ stemmer.stem(t) for t in totalvocab_tokenized ]
#pickle.dump(totalvocab_stemmed, open("../input/cm-kiva-nlp/totalvocab_stemmed.p","wb"))
#totalvocab_stemmed = pickle.load(open("../input/cm-kiva-nlp/totalvocab_stemmed.p","rb"))

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

#vocab_frame.info()
# term frequency - inverse document frequency :  is a numerical statistic reflecting how important a word is to a document in a collection or corpus
# take into account only tokens that appear in more than 5% less than 95% of documents. Those appearing more than 95% of docs tend to be irrelevant
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features= 20,\
                                 min_df=0.01, stop_words='english', norm = 'l2',\
                                 use_idf=True, analyzer = 'word', tokenizer=tokenize_and_stem, ngram_range= (1,1) )

#This operation takes around 20 minutes of computational time, therefore, I computed the matrix offline and use it as a data source
#tfidf_matrix = tfidf_vectorizer.fit_transform(text)
#pickle.dump(tfidf_vectorizer, open("../input/cm-kiva-nlp/tfidf_vectorizer.p","wb"))
#pickle.dump(tfidf_matrix, open("../input/cm-kiva-nlp/tfidf_matrix.p","wb"))

tfidf_vectorizer = pickle.load( open("../input/cm-kiva-nlp/tfidf_vectorizer.p","rb") )
tfidf_matrix = pickle.load( open("../input/cm-kiva-nlp/tfidf_matrix.p","rb") )
terms = tfidf_vectorizer.get_feature_names()
len(terms)
print(terms[:])
#vocab_frame.loc[ terms]
#vocab_frame.loc[ 'like']
#%% CLUSTERING by loan activity

num_clusters = 50

km = KMeans(n_clusters=num_clusters, random_state = 1)

#This operation takes several minutes of computational time, therefore, I computed the matrix offline and use it as a data source

#km.fit(tfidf_matrix)
#pickle.dump(km, open("../input/cm-kiva-nlp/km.p","wb"))

km = pickle.load( open("../input/cm-kiva-nlp/km.p","rb") )

clusters = km.labels_.tolist()
df['activity label']  = clusters
#df['activity label'].value_counts()
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
df.head(2)
# prediction: assume that loanuses_predict contains the loan uses of new demands
loanact_predict = df.loc[0:1,'activity'].values.astype(str).tolist() # convert nan cell to string
tfidf_matrix_predict = tfidf_vectorizer.transform(loanact_predict)
# loan uses classes for the new demand are predicted by:
loanact_classes = km.predict( tfidf_matrix_predict )
print(loanact_classes )
for i in df['activity label'].unique().tolist():
    print('Activity cluster %d is characterized by:'%i)
    print(df[ df['activity label'] == i][ 'activity' ].unique().tolist() )
    
    print('\n')
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
col_list = [ 'MPI country', 'Loan amount / GDP per capita (%)', 'sector label', 'activity label', 'gender label' ,  'term_in_months', 'repayment interval label']
print(col_list)
plot_correlation_matrix( df[col_list].corr().round(2), filename = 'correlation matrix classfication' )
# fill missing values with median
fill_missing_values = preprocessing.Imputer(strategy = 'median').fit( df[ col_list ]  )

fill_missing_values.transform( df[ col_list ]  )


num_loan_clusters = 50

loan_clusters = KMeans(n_clusters=num_loan_clusters , random_state = 1)

#loan_clusters.fit( fill_missing_values.transform( df[ col_list ]  ) )
#pickle.dump(loan_clusters, open("../input/cm-kiva-nlp/loan_clusters.p","wb"))

loan_clusters = pickle.load( open("../input/cm-kiva-nlp/loan_clusters.p","rb") )

clusters = loan_clusters.labels_.tolist()

df['Loan class'] = clusters
#%% plot loans in each loan use group

#cl_list = df['Loan class'].unique().tolist()
# plot 3 largest loan classes
cl_list = df['Loan class'].value_counts().index[0:3].tolist()

for i in cl_list :
    print('Loan class %d'%i)
    print( df[ df['Loan class'] == i][ col_list ].describe() )
    print('- - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - ')
    print('\n')

for y in [ 'MPI country', 'Loan amount / GDP per capita (%)', 'sector label', 'activity label' ,'term_in_months']:
    plt.figure()
    sns.boxplot(x="Loan class", y= y,  data=df[ df['Loan class'].isin(df['Loan class'].value_counts().index[0:5].tolist())]);
plt.figure()
sns.countplot(y="Loan class", hue = 'repayment_interval',  data=df[ df['Loan class'].isin(df['Loan class'].value_counts().index[0:5].tolist())]);
plt.figure()
sns.countplot(y="Loan class", hue = 'gender label',  data=df[ df['Loan class'].isin(df['Loan class'].value_counts().index[0:5].tolist())]);
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
df.loc[ 0:1, : ]
loan_classes_predict = loan_clusters.predict( fill_missing_values.transform( df.loc[ 0:1, col_list ]  ) )
print(loan_classes_predict)
for i in loan_classes_predict:
    print('Loan class %d'%i)
    print(df[ df['Loan class']== i][ col_list ].describe())
    print('\n')
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
def kmeans_func(df, nclusters = 10):
    fill_missing_values = preprocessing.Imputer(strategy = 'median').fit( df )
    
    clusters = KMeans(n_clusters=num_loan_clusters , random_state = 1)
    clusters.fit( preprocessing.Imputer(strategy = 'median').fit_transform( fill_missing_values.transform(df) ))
    
    return clusters.labels_.tolist()
# IDEAL CASE; MPI OF REGIONS should be included.
col_list = [ 'Loan amount / GDP per capita (%)', 'sector label', 'activity label', 'gender label',  'term_in_months', 'repayment interval label']
country = 'Samoa'

dfaux =  df.loc[ df['country'] == country ]
dfaux = dfaux.drop(['Loan class'], axis = 1)

cl_predict = kmeans_func( dfaux.loc[:,col_list] , nclusters = 5 )
dfaux = pd.concat([dfaux, pd.DataFrame(data= cl_predict, index = dfaux.index, columns = ['Loan class'] ) ], axis = 1, join = 'outer' )
#%% plot loans in each loan use group

#cl_list = dfaux['Loan class'].unique().tolist()
cl_list = dfaux['Loan class'].value_counts().index[0:3].tolist()

#%%
for i in cl_list :
    print('Loan class %d'%i)
    print(dfaux[ dfaux['Loan class'] == i][ col_list ].describe())
    print('- - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - ')
    print('\n')
    
for y in [ 'Loan amount / GDP per capita (%)', 'sector label', 'activity label' ,  'term_in_months', 'repayment interval label']:
    plt.figure()
    sns.boxplot(x="Loan class", y= y,  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);
    
plt.figure()
sns.countplot(y="Loan class", hue = 'gender label',  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
country = 'Philippines'

dfaux =  df.loc[ df['country'] == country ]
dfaux = dfaux.drop(['Loan class'], axis = 1)

cl_predict = kmeans_func( dfaux.loc[:,col_list] , nclusters = 5 )
dfaux = pd.concat([dfaux, pd.DataFrame(data= cl_predict, index = dfaux.index, columns = ['Loan class'] ) ], axis = 1, join = 'outer' )

cl_list = dfaux['Loan class'].value_counts().index[0:3].tolist()

#%%
for i in cl_list :
    print('Loan class %d'%i)
    print(dfaux[ dfaux['Loan class'] == i][ col_list ].describe())
    print('- - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - ')
    print('\n')
    
for y in [ 'Loan amount / GDP per capita (%)', 'sector label', 'activity label' ,  'term_in_months', 'repayment interval label']:
    plt.figure()
    sns.boxplot(x="Loan class", y= y,  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);
    
plt.figure()
sns.countplot(y="Loan class", hue = 'gender label',  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
country = 'Kenya'

dfaux =  df.loc[ df['country'] == country ]
dfaux = dfaux.drop(['Loan class'], axis = 1)

cl_predict = kmeans_func( dfaux.loc[:,col_list] , nclusters = 5 )
dfaux = pd.concat([dfaux, pd.DataFrame(data= cl_predict, index = dfaux.index, columns = ['Loan class'] ) ], axis = 1, join = 'outer' )

cl_list = dfaux['Loan class'].value_counts().index[0:3].tolist()

#%%
for i in cl_list :
    print('Loan class %d'%i)
    print(dfaux[ dfaux['Loan class'] == i][ col_list ].describe())
    print('- - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - ')
    print('\n')
    
for y in [ 'Loan amount / GDP per capita (%)', 'sector label', 'activity label' ,  'term_in_months', 'repayment interval label']:
    plt.figure()
    sns.boxplot(x="Loan class", y= y,  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);
    
plt.figure()
sns.countplot(y="Loan class", hue = 'gender label',  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
# IDEAL CASE; MPI OF REGIONS should be included.
col_list = [ 'Loan amount / GDP per capita (%)', 'activity label', 'gender label',  'term_in_months', 'repayment interval label']

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
country = 'India'
sector = 'Agriculture'

dfaux =  df.loc[ (df['country'] == country ) & (df['sector'] == sector) ]
dfaux = dfaux.drop(['Loan class'], axis = 1)

cl_predict = kmeans_func( dfaux.loc[:,col_list] , nclusters = 5 )
dfaux = pd.concat([dfaux, pd.DataFrame(data= cl_predict, index = dfaux.index, columns = ['Loan class'] ) ], axis = 1, join = 'outer' )

cl_list = dfaux['Loan class'].value_counts().index[0:3].tolist()

#%%
for i in cl_list :
    print('Loan class %d'%i)
    print(dfaux[ dfaux['Loan class'] == i][ col_list ].describe())
    print('- - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - ')
    print('\n')
    
    
for y in [ 'Loan amount / GDP per capita (%)' ,  'term_in_months', 'repayment interval label']:
    plt.figure()
    sns.boxplot(x="Loan class", y= y,  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);

plt.figure()
sns.countplot(y="Loan class", hue = 'activity label',  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);

plt.figure()
sns.countplot(y="Loan class", hue = 'gender label',  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
dfaux =  df.loc[ (df['country'] == country ) & (df['sector'] == sector) ]
dfaux = dfaux.drop(['Loan class'], axis = 1)
# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")
#print(stemmer)


dfaux['use'] = dfaux['use'].str.replace('buy','purchase')
text = dfaux['use'].values.astype(str).tolist() # convert nan cell to string

# create a dictionary of tokens and stems
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in text:
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
totalvocab_tokenized = list( set( totalvocab_tokenized  ) )

totalvocab_stemmed = [ stemmer.stem(t) for t in totalvocab_tokenized ]

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

#vocab_frame.info()

# term frequency - inverse document frequency :  is a numerical statistic reflecting how important a word is to a document in a collection or corpus
# take into account only tokens that appear in more than 5% less than 95% of documents. Those appearing more than 95% of docs tend to be irrelevant
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features= 20,\
                                 min_df=0.01, stop_words='english', norm = 'l2',\
                                 use_idf=True, analyzer = 'word', tokenizer=tokenize_and_stem, ngram_range= (1,1) )

#This operation takes minutes of computational time, therefore, I computed the matrix offline and use it as a data source
tfidf_matrix = tfidf_vectorizer.fit_transform(text)

terms = tfidf_vectorizer.get_feature_names()
#len(terms)
#print(terms[:])

#%% CLUSTERING by loan use

num_clusters = 20

km = KMeans(n_clusters=num_clusters, random_state = 1)

#This operation takes several minutes of computational time, therefore, I computed the matrix offline and use it as a data source

km.fit(tfidf_matrix)


clusters = km.labels_.tolist()

dfaux['loan use label']  = clusters

# IDEAL CASE; MPI OF REGIONS should be included.
col_list = [ 'Loan amount / GDP per capita (%)', 'activity label', 'gender label', 'loan use label',  'term_in_months', 'repayment interval label']

cl_predict = kmeans_func( dfaux.loc[:,col_list] , nclusters = 5 )
dfaux = pd.concat([dfaux, pd.DataFrame(data= cl_predict, index = dfaux.index, columns = ['Loan class'] ) ], axis = 1, join = 'outer' )

cl_list = dfaux['Loan class'].value_counts().index[0:3].tolist()

#%%
for i in cl_list :
    print('Loan class %d'%i)
    print(dfaux[ dfaux['Loan class'] == i][ col_list ].describe())
    print('- - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - ')
    print('\n')
    
    
for y in [ 'Loan amount / GDP per capita (%)' ,  'term_in_months', 'repayment interval label']:
    plt.figure()
    sns.boxplot(x="Loan class", y= y,  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);

plt.figure()
sns.countplot(y="Loan class", hue = 'activity label',  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);

plt.figure()
sns.countplot(y="Loan class", hue = 'loan use label',  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);


plt.figure()
sns.countplot(y="Loan class", hue = 'gender label',  data=dfaux[ dfaux['Loan class'].isin(dfaux['Loan class'].value_counts().index[0:5].tolist())]);
ind = dfaux['loan use label'] == 7
textpr = dfaux.loc[ind, 'use'].values.astype(str).tolist() # convert nan cell to string
tfidf_vectorizer.fit(textpr)
#print(tfidf_vectorizer.get_feature_names())
# get the common terms with the collection of terms from all documents
imp_terms = vocab_frame.loc[ set(terms) & set( tfidf_vectorizer.get_feature_names() )]
imp_terms.reset_index().drop_duplicates(subset='index', keep='first', inplace=False).set_index('index')
ind = dfaux['loan use label'] == 10
textpr = dfaux.loc[ind, 'use'].values.astype(str).tolist() # convert nan cell to string
tfidf_vectorizer.fit(textpr)
#print(tfidf_vectorizer.get_feature_names())
# get the common terms with the collection of terms from all documents
imp_terms = vocab_frame.loc[ set(terms) & set( tfidf_vectorizer.get_feature_names() )]
imp_terms.reset_index().drop_duplicates(subset='index', keep='first', inplace=False).set_index('index')
ind = dfaux['loan use label'] == 18
textpr = dfaux.loc[ind, 'use'].values.astype(str).tolist() # convert nan cell to string
tfidf_vectorizer.fit(textpr)
#print(tfidf_vectorizer.get_feature_names())
# get the common terms with the collection of terms from all documents
imp_terms = vocab_frame.loc[ set(terms) & set( tfidf_vectorizer.get_feature_names() )]
imp_terms.reset_index().drop_duplicates(subset='index', keep='first', inplace=False).set_index('index')
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
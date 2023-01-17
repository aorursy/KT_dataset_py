# Let's import the libraries
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import time
import seaborn as sns
import statsmodels.formula.api as smf
import os
import matplotlib.cm as cm
#import plotly.plotly as py
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
from geopy.geocoders import Nominatim
import folium
from folium.features import DivIcon
from sklearn.feature_extraction.text import TfidfVectorizer
import missingno as msno
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import networkx as nx
from plotly.graph_objs import *
from os import path
from PIL import Image
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from pandas import DataFrame, scatter_matrix
import warnings


%matplotlib inline
mpl.rcParams['axes.facecolor'] = 'w'
mpl.rcParams['figure.facecolor'] = 'w'
# Let's import the datasets
kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
kiva_ids=pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
kiva_ids= kiva_ids.dropna()
kiva_themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
warnings.filterwarnings("ignore")
# We will create a copy of the dataset, reducing it to columns & data relevant to the initial analysis
kiva_full = kiva_loans.copy()

# Calculate percentage of missing values
missing_vals = kiva_full.isnull().sum()[kiva_full.isnull().sum() > 0]
total_cells = np.product(kiva_full.shape)
total_missing_vals = missing_vals.sum()
per_missing_vals = (total_missing_vals/total_cells) * 100

# Look at missing value
#print ("Let's visualize Missing values in the dataset")
msno.matrix(kiva_full)
# Now let's work on missing values
# Lets check what are the countries for which country code is missing
kiva_full.loc[kiva_full.country_code.isnull(), 'country'].unique()

# Country code is missing for only Namibia - lets replace the missing values with NAM as country code
kiva_full['country_code'] = kiva_full['country_code'].fillna('NAM')

# disbursed_time , funded_time - substitute missing values with the corresponding value from posted_time
kiva_full['disbursed_time'] = kiva_full['disbursed_time'].fillna(kiva_full['posted_time'])
kiva_full['funded_time'] = kiva_full['funded_time'].fillna(kiva_full['posted_time'])

# borrower_genders - substitute missing values
kiva_full['borrower_genders'] = kiva_full['borrower_genders'].fillna(method='pad')

# log amount
kiva_full['log_amount'] = np.log(kiva_full['funded_amount'] + 1)

# Format date column to date format
kiva_full['disbursed_time'] = pd.to_datetime(kiva_full['disbursed_time'])
kiva_full['date'] = pd.to_datetime(kiva_full['date'])
kiva_full['posted_time'] = pd.to_datetime(kiva_full['posted_time'])
kiva_full['funded_time'] = pd.to_datetime(kiva_full['funded_time'])

# Add 2 columns containing count of males & females according to contents of the gender column
for i, row in kiva_full.iterrows():
    g = row['borrower_genders']         
    
    female = g.count('female')
    male = g.count('male')
    
    kiva_full.set_value(i, 'borrower_female', female)
    kiva_full.set_value(i, 'borrower_male', male)
    
kiva_full.borrower_male = kiva_full.borrower_male.astype(int)
kiva_full.borrower_female = kiva_full.borrower_female.astype(int)
kiva_full.head()
# View distribution of the 3 variables
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

kiva_full['borrower_male'].plot(kind='hist', color = '#D35400', histtype='step', ax=ax[0], linewidth=2)
ax[0].get_xaxis().set_ticklabels([])
ax[0].get_xaxis().set_label([])
ax[0].get_yaxis().set_ticklabels([])
ax[0].get_yaxis().get_label().set_visible(False)
ax[0].set_title("Male borrowers", fontsize=18, color = '#D35400')
ax[0].grid(color="slateblue", which="both", linestyle=':', linewidth=0.5)

kiva_full['log_amount'].plot(kind='hist', color = '#D35400', histtype='step', ax=ax[1], linewidth=2)
ax[1].get_xaxis().set_ticklabels([])
ax[1].get_xaxis().set_label([])
ax[1].get_yaxis().set_ticklabels([])
ax[1].get_yaxis().get_label().set_visible(False)
ax[1].set_title("Loan Amount", fontsize=18, color = '#D35400')
ax[1].grid(color="slateblue", which="both", linestyle=':', linewidth=0.5)

kiva_full['borrower_female'].plot(kind='hist', color = '#D35400', histtype='step', ax=ax[2], linewidth=2)
ax[2].get_xaxis().set_ticklabels([])
ax[2].get_xaxis().set_label([])
ax[2].get_yaxis().set_ticklabels([])
ax[2].get_yaxis().get_label().set_visible(False)
ax[2].set_title("Female borrowers", fontsize=18, color = '#D35400')
ax[2].grid(color="slateblue", which="both", linestyle=':', linewidth=0.5)

#plt.title("View distribution of the 3 variables")
plt.suptitle('View distribution of the 3 variables', fontsize=20, y=1.05, color = '#D35400')
plt.tight_layout()
plt.show()
#fig.tight_layout()
colors = cm.YlOrRd(np.linspace(0,1,len(kiva_full)))
f, ax = plt.subplots(figsize=(15, 8))
plt.scatter(kiva_full['country'], kiva_full['log_amount'], s=kiva_full['term_in_months'], color=colors, alpha=0.5)
ax.set_title("Amounts based on term", fontsize=18)
ax.set_xlabel("Country", fontsize=18)
ax.set_ylabel("Amount", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.show()
kt2 = kiva_themes.groupby(['Field Partner Name', 'Loan Theme Type', 'region'])['number'].sum().reset_index().sort_values('number', ascending = False).reset_index(drop=True)

kt3=kt2[:300]
partners = kt3["Field Partner Name"].unique()
partners=partners.tolist()
themes = kt3["Loan Theme Type"].unique()
themes=themes.tolist()

G = nx.Graph()
G.add_nodes_from(kt3['Field Partner Name'], bipartite=0)
G.add_nodes_from(kt3['Loan Theme Type'], bipartite=1)
G.add_nodes_from(kt3['region'], bipartite=2)
G.add_weighted_edges_from(
    [(row['Field Partner Name'], row['Loan Theme Type'], row['number']) for idx, row in kt3.iterrows()], 
    weight='weight')
G.add_weighted_edges_from(
    [(row['Loan Theme Type'], row['region'], row['number']) for idx, row in kt3.iterrows()], 
    weight='weight')
G.add_weighted_edges_from(
    [(row['Field Partner Name'], row['region'], row['number']) for idx, row in kt3.iterrows()], 
    weight='weight')

pos=nx.random_layout(G)

dmin=1
ncenter=0
for n in pos:
    x,y=pos[n]
    d=(x-0.5)**2+(y-0.5)**2
    if d<dmin:
        ncenter=n
        dmin=d

p=nx.single_source_shortest_path_length(G,ncenter)

edge_trace = Scatter(
    x=[],
    y=[],
    line=Line(width=0.5,color='#888'),
    #line=Line(width=[],color='#888'),
    hoverinfo='text',
    text=[],
    mode='lines')

for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += [x0, x1, None]
    edge_trace['y'] += [y0, y1, None]
        
node_trace = Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=Marker(
        showscale=False,
        # colorscale options
        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
        #colorscale='RdBu',
        #reversescale=True,
        color=[],
        size=[],
        line=dict(width=2, color='black')
    )
    )

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'].append(x)
    node_trace['y'].append(y) 
 
    if (node in partners):
        col = "#2096BA"
        node_info = 'Partner: '+ str(node)
        s = 30
    elif (node in themes):
        col = "#842f82"
        node_info = 'Theme: '+ str(node)
        s = 36
    else:
        col = "#DF6E21"
        node_info = 'Region: '+ str(node)
        s = 24
    
    node_trace['text'].append(node_info)
    node_trace['marker']['color'].append(col) 
    node_trace['marker']['size'].append(s)
    
    
fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>Network of Partners, Themes & Regions (Blue = Partners, Purple = Themes, Orange = Regions)',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=XAxis(showgrid=True, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=True, zeroline=False, showticklabels=False)))

py.iplot(fig, filename='networkx')
counts = dict()

words = str(kiva_loans['use'].dropna()).split()

for word in words:
    if word in counts:
        counts[word] += 1
    else:
        counts[word] = 1

counts = {k: v for k, v in counts.items() if not k.isdigit()}

counts['kiva'] = 120
counts['loans'] = 110

d = "data/"
mask = np.array(Image.open("../input/kiva-self-data/k6.jpg"))

stopwords=set(STOPWORDS)
stopwords.add('[True')
stopwords.add('''farm.]''')

wc = WordCloud(background_color="white", mask=mask, colormap="Greens",stopwords=stopwords)

# generate word cloud
wc.generate_from_frequencies(counts)

#image_colors = ImageColorGenerator(mask)

# show
fig,ax=plt.subplots(figsize=(25,12))
plt.imshow(wc,  interpolation='bilinear')
plt.axis("off")
plt.show()
# .recolor(color_func=image_colors)
# We will keep the above dataset for future reference & create another one for current (Country-wise) analysis
kiva_df = kiva_full.copy()
kiva_df = kiva_df.drop(labels=['id', 'use', 'tags', 'region', 'country_code', 'currency', 'partner_id', 'disbursed_time', 'funded_time', 'posted_time', 'borrower_genders', 'loan_amount'], axis=1)
kiva_df.columns = ['Amount', 'Activity', 'Sector', 'Country', 'Term', 'Lenders', 'Repayment_Interval', 'Date', 'Log_Amount', 'Females',  'Males']
kiva_df = kiva_df [['Country', 'Sector', 'Activity', 'Date', 'Repayment_Interval', 'Term', 'Amount', 'Log_Amount', 'Lenders', 'Females', 'Males']]
kiva_df.head()
# Get a list of all Country names arranged by decreasing number of loans
countries = kiva_df['Country'].value_counts().to_frame()
countries.reset_index(inplace=True)
countries = countries.rename(columns = {'index':'Country', 'Country':'NumberOfLoans'})

# List of top 20 countries & their geographic coordinates
top20_list = countries[:20].copy()

top20_geo = top20_list.copy()
top20_geo = top20_geo.drop('NumberOfLoans', axis = 1)


# Top countries with the hightest number of loans
colors = cm.YlGnBu(np.linspace(0,1,len(top20_list)))
fig, ax = plt.subplots(figsize=(15,8))
  
sns.barplot(top20_list.NumberOfLoans, top20_list.Country, ax = ax, palette=colors)
ax.set_title('Top countries with highest number of loans')

for i, v in enumerate(top20_list.NumberOfLoans):
    plt.text(0.8,i,v,color='k',fontsize=10)

plt.show()
cnt_s = kiva_df['Sector'].value_counts()

labels = cnt_s.index
values = cnt_s.values
trace = go.Pie(labels=labels, values=values)
py.iplot([trace], filename='basic_pie_chart')
fig, ax = plt.subplots(figsize=(15,5))

sns.boxplot(x="Sector", y="Log_Amount", data=kiva_df, ax = ax)

ax.set_title("Sector vs Amount", fontsize=15)
ax.set_xlabel("Sector", fontsize=12)
ax.set_ylabel("Amount", fontsize=12)

plt.tight_layout()
cnt_a = kiva_df['Activity'].value_counts()

fig = {
  "data": [
    {
      "values": cnt_a.values,
      "labels": cnt_a.index,
      "text":"Activities",
      "textposition":"inside",
      "hoverinfo":"label+percent",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Activity Wise distributions",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Activities"
            }
        ]
    }
}
py.iplot(fig, filename='donut')
kt4=kiva_loans.groupby(['sector','activity'])['loan_amount'].sum().reset_index(name='amount').sort_values('amount', ascending = False).reset_index(drop=True)
kt4 = kt4.drop_duplicates()
kt4 = kt4.dropna()

kt4.loc[(kt4['activity'] == 'Retail'), 'activity'] = 'Activity_Retail'
kt4.loc[(kt4['activity'] == 'Agriculture'), 'activity'] = 'Activity_Agriculture'
kt4.loc[(kt4['activity'] == 'Clothing'), 'activity'] = 'Activity_Clothing'
kt4.loc[(kt4['activity'] == 'Food'), 'activity'] = 'Activity_Food'
kt4.loc[(kt4['activity'] == 'Health'), 'activity'] = 'Activity_Health'
kt4.loc[(kt4['activity'] == 'Services'), 'activity'] = 'Activity_Services'
kt4.loc[(kt4['activity'] == 'Manufacturing'), 'activity'] = 'Activity_Manufacturing'
kt4.loc[(kt4['activity'] == 'Transportation'), 'activity'] = 'Activity_Transportation'
kt4.loc[(kt4['activity'] == 'Arts'), 'activity'] = 'Activity_Arts'
kt4.loc[(kt4['activity'] == 'Construction'), 'activity'] = 'Activity_Construction'
kt4.loc[(kt4['activity'] == 'Wholesale'), 'activity'] = 'Activity_Wholesale'
kt4.loc[(kt4['activity'] == 'Entertainment'), 'activity'] = 'Activity_Entertainment'

sectors = kt4["sector"].unique()
sectors=sectors.tolist()

activities = kt4["activity"].unique()
activities=activities.tolist()

kt5 = kt4[:50]

G1 = nx.Graph()
G1.add_nodes_from(kt4['sector'], bipartite=0)
G1.add_nodes_from(kt4['activity'], bipartite=1)
G1.add_weighted_edges_from(
    [(row['sector'], row['activity'], row['amount']) for idx, row in kt4.iterrows()], 
    weight='weight')

pos1=nx.circular_layout(G1)

dmin1=1
ncenter1=0
for n in pos1:
    x,y=pos1[n]
    d=(x-0.5)**2+(y-0.5)**2
    if d<dmin1:
        ncenter1=n
        dmin1=d

p1=nx.single_source_shortest_path_length(G1,ncenter1)

edge_trace1 = Scatter(
    x=[],
    y=[],
    line=Line(width=1.5,color='#47112a'),
    #line=Line(width=[],color='#888'),
    hoverinfo='text',
    text=[],
    mode='lines')

for edge in G1.edges(data=True):
    x0, y0 = pos1[edge[0]]
    x1, y1 = pos1[edge[1]]
    edge_trace1['x'] += [x0, x1, None]
    edge_trace1['y'] += [y0, y1, None]
        
node_trace1 = Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=Marker(
        showscale=False,
        opacity= 0.9,
        # colorscale options
        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
        #colorscale='RdBu',
        #reversescale=True,
        color=[],
        size=[],
        line=dict(width=2, color='#47112a')
    )
    )

for node in G1.nodes():
    x, y = pos1[node]
    node_trace1['x'].append(x)
    node_trace1['y'].append(y) 
 
    if (node in sectors):
        col = "#bf9a18"
        node_info = 'Sector: '+ str(node)
        s=60
    elif (node in activities):
        col = "#84174a"
        node_info = 'Activity: '+ str(node)
        s=60
    
    node_trace1['text'].append(node_info)
    node_trace1['marker']['color'].append(col) 
    node_trace1['marker']['size'].append(s)
    
fig1 = Figure(data=Data([edge_trace1, node_trace1]),
             layout=Layout(
                title='<br>Top Sectors <-> Activity contributors (Yellow = Sector, Red = Activity)',
                titlefont=dict(size=20),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=XAxis(showgrid=True, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=True, zeroline=False, showticklabels=False)))

py.iplot(fig1, filename='networkx1')
cnt_s5 = cnt_s[:5]
sectors = ['Agriculture', 'Food', 'Retail', 'Services', 'Personal Use']
cnt_a2 = kiva_df.groupby(['Sector'])['Activity'].value_counts().groupby(level=0).head(3).reset_index(name='count')
cnt_a2 = cnt_a2.groupby(['Sector','Activity'], as_index=False)['count'].sum()
cnt_a2 = pd.DataFrame(cnt_a2)
cnt_a2 = cnt_a2.loc[cnt_a2['Sector'].isin(sectors)].reset_index(drop=True)

color_list = ['#D98880', '#D7DBDD', '#F4D03F', '#45B39D', '#9A7D0A', '#884EA0', '#34495E', '#EAF2F8', '#1F618D', '#76448A', '#99A3A4', '#27AE60', '#935116', '#85C1E9', '#6C3483']
fig, ax = plt.subplots(figsize=(14,8))
cnt_p = cnt_a2.pivot_table(index="Activity", columns="Sector", values="count").T.fillna(0)
cnt_p.plot(kind='barh', ax=ax, color=color_list, stacked=True, legend=True).invert_yaxis()

ax.set_title("Top 3 Activities per Sector", fontsize=20)
ax.set_xlabel("Activities", fontsize=15)
ax.set_ylabel("Sectors", fontsize=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)


labels = []
for j in cnt_p.columns:
    for i in cnt_p.index:
        label = str(j)
        if (label=="Fruits & Vegetables"):
            label = "Fruits&Vegetables"
        if (label=="Home Energy"):
            label = "Energy"
        if (label=="Personal Expenses"):
            label = "Personal"
        if (label=="Cosmetics Sales"):
            label = "C.Sales"
        labels.append(label)

                
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., label, ha='center', va='center', fontsize="small")
plt.legend(bbox_to_anchor=(1.2,1), fontsize=10)

kiva_time = kiva_df.copy()
kiva_time['Year'] = kiva_time['Date'].dt.year
kiva_time = kiva_time[kiva_time.Year != 2017]

kiva_growth = pd.DataFrame(kiva_time.groupby('Country').size()).reset_index().rename(columns={'Country': 'Country', 0: 'Count'}).sort_values(by = 'Count', ascending = False).reset_index(drop=True)
kiva_temp = pd.DataFrame(kiva_time.groupby(['Country', 'Year']).size()).reset_index().rename(columns={'Country': 'Country', 0: 'Count'}).sort_values(by = 'Count', ascending = False).reset_index(drop=True)

for i, row in kiva_growth.iterrows():
    Counts_2014 = kiva_temp.loc[(kiva_temp.Country==row.Country) & (kiva_temp.Year == 2014)].Count
    Counts_2015 = kiva_temp.loc[(kiva_temp.Country==row.Country) & (kiva_temp.Year == 2015)].Count
    Counts_2016 = kiva_temp.loc[(kiva_temp.Country==row.Country) & (kiva_temp.Year == 2016)].Count
    Counts_2017 = kiva_temp.loc[(kiva_temp.Country==row.Country) & (kiva_temp.Year == 2017)].Count
    
    if (len(Counts_2014) == 0):
        kiva_growth.set_value(i, 'Counts_2014', 0)
    else:
        kiva_growth.set_value(i, 'Counts_2014', Counts_2014.item())
        
    if (len(Counts_2015) == 0):
        kiva_growth.set_value(i, 'Counts_2015', 0)
    else:
        kiva_growth.set_value(i, 'Counts_2015', Counts_2015.item())
        
    if (len(Counts_2016) == 0):
        kiva_growth.set_value(i, 'Counts_2016', 0)
    else:
        kiva_growth.set_value(i, 'Counts_2016', Counts_2016.item())
        
    if (len(Counts_2017) == 0):
        kiva_growth.set_value(i, 'Counts_2017', 0)
    else:
        kiva_growth.set_value(i, 'Counts_2017', Counts_2017.item())        

        
increase = kiva_growth.loc[kiva_growth.Counts_2016 > kiva_growth.Counts_2014].Country.value_counts().sum()
decrease = kiva_growth.loc[kiva_growth.Counts_2016 < kiva_growth.Counts_2014].Country.value_counts().sum()
same = kiva_growth.loc[kiva_growth.Counts_2016 == kiva_growth.Counts_2014].Country.value_counts().sum()
total = kiva_growth.Country.value_counts().sum()
inc_per = increase/total * 100
dec_per = decrease/total * 100

color_list = ['#8a4da0', '#2f8477', '#e08011']
fig, ax = plt.subplots(figsize=(15,5))
kg = kiva_growth[:20]

kg.plot(x="Country", y=["Counts_2014", "Counts_2015", "Counts_2016"], 
        kind="bar", ax = ax, width=0.95, color=color_list)
ax.set_title("Number of Loans per Year for Top 20 Countries", fontsize=20)
ax.set_xlabel("Country", fontsize=20)
ax.set_ylabel("Number of Loans per Year", fontsize=20)
ax.set_xticklabels(labels = ax.get_xticklabels(), fontsize=15)
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
plt.show()
color_list = ['#b2b2a2' , '#A1BBD0', '#85929E' , '#61C0BF' ]

fig, ax = plt.subplots(figsize=(20,6))
#ax.set_facecolor('#E5DACE')
kiva_time.groupby(['Repayment_Interval', 'Year']).size().reset_index().rename(columns={0: 'Count'}).pivot_table(index="Repayment_Interval", columns="Year", values="Count").T.plot(kind='bar', ax=ax, color=color_list)
ax.set_title("Changes in Repayment Intervals over the Years", fontsize=20)
ax.set_xlabel("Year", fontsize=20)
ax.set_ylabel("Growth in Repayment Interval", fontsize=20)
ax.set_xticklabels(labels = ax.get_xticklabels(), fontsize=15, rotation=0)
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., fontsize=15)
#ax.grid(True)
for p in ax.patches:
    ax.annotate((p.get_height()).astype(int), (p.get_x() * 1.01, p.get_height() * 1.01))
    
plt.show()
fig, ax = plt.subplots(figsize=(24,10))
kiva_full.groupby(['date'])['funded_amount'].sum().reset_index().sort_values(by = 'date', ascending = True).plot(x="date", ax=ax, c='orange')
kiva_full.groupby(['date'])['loan_amount'].sum().reset_index().sort_values(by = 'date', ascending = True).plot(x="date", ax=ax, c='c')
ax.set_title("Funded(orange) vs Loan(blue) Amount over Time", fontsize=20)
ax.set_xlabel("Time", fontsize=20)
ax.set_ylabel("Amount", fontsize=20)
ax.legend_.remove()
plt.show()
# The data for GDP, GDP per Capita & GNI per Capita is taken from UNHD Reports
gdp = pd.read_excel('../input/kiva-additional-data/GDP_GNI.xlsx', 'Data', index_col=None)

# The data for Income Group classifications is taken from The World Bank's dataset
inc = pd.read_excel('../input/kiva-additional-data/Income_classification.xls', 'Data', index_col=None)

# Gowth in GDP (percent increase) - World Bank data
pgdp = pd.read_excel('../input/kiva-additional-data/GDP_growth.xls', 'Data', index_col=None)

# Bank Lending Rates (Source: The World Bank)
lend = pd.read_csv('../input/kiva-additional-data/Bank Lending Rates.csv', 'Data', index_col=None, engine='python', delimiter=",")
# The data for HDI is taken from UNHD Reports
hdi = pd.read_excel('../input/kiva-additional-data/HDI_IHDI.xlsx', 'Data', index_col=None)
data = [ dict(
        type = 'choropleth',
        locations = hdi['Country'],
        locationmode='country names',
        z = hdi['HDI_Value'],
        text = hdi['HDI_Value'],
        colorscale = [[0,"rgb(1, 73, 99)"],[0.35,"rgb(42, 126, 168)"],[0.5,"rgb(102, 177, 214)"],\
            [0.6,"rgb(131, 219, 226)"],[0.7,"rgb(222, 247, 249)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,        
        hoverinfo="location+text",
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
       showscale=False,
      ) ]

layout = dict(
    title = 'Human Development Index',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout)
py.iplot( fig, validate=False)
d1 = hdi.merge(countries, how="left", on="Country")
d1['NumberOfLoans']=d1['NumberOfLoans'].fillna(0)

data = [ dict(
        type = 'choropleth',
        locations = d1['Country'],
        locationmode='country names',
        z = np.log(d1['NumberOfLoans'] + 1),
        text = d1['NumberOfLoans'],
        colorscale = [[0,"rgb(122, 3, 21)"],[0.1,"rgb(155, 3, 41)"],[0.2,"rgb(183, 14, 73)"],\
            [0.3,"rgb(219, 41, 124)"],[0.4,"rgb(219, 41, 166)"],[0.5,"rgb(214, 64, 204)"],\
            [0.6,"rgb(212, 79, 221))"],[0.7,"rgb(224, 106, 242)"],[0.8,"rgb(220, 125, 242)"],\
                      [0.9,"rgb(213, 184, 242)"],[1.0,"rgb(249, 222, 249)"]
                     ],
        autocolorscale = False,
        reversescale = True,
        hoverinfo="location+text",
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        showscale=False,
      ) ]

layout = dict(
    title = 'Number of Loans',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout)
py.iplot( fig, validate=False)
metrics = countries.merge(gdp, how="left", on = "Country")
metrics = metrics.merge(hdi, how="left", on = "Country")
metrics = metrics.merge(inc, how="left", on = "Country")
metrics = metrics.merge(lend, how="left", on = "Country")
metrics = metrics.merge(pgdp, how="left", on = "Country")
df = kiva_full.merge(metrics, how="left", left_on = "country", right_on="Country")

df.loc[((df.HDI_Scale == "Very High") & (df.IncomeCategory == "High income")), 'Group'] = 'G8'
df.loc[((df.HDI_Scale == "High") & (df.IncomeCategory == "Upper middle income")), 'Group'] = 'G7'
df.loc[((df.HDI_Scale == "High") & (df.IncomeCategory == "Lower middle income")), 'Group'] = 'G6'
df.loc[((df.HDI_Scale == "Medium") & (df.IncomeCategory == "Upper middle income")), 'Group'] = 'G5'
df.loc[((df.HDI_Scale == "Medium") & (df.IncomeCategory == "Lower middle income")), 'Group'] = 'G4'
df.loc[((df.HDI_Scale == "Medium") & (df.IncomeCategory == "Low income")), 'Group'] = 'G3'
df.loc[((df.HDI_Scale == "Low") & (df.IncomeCategory == "Lower middle income")), 'Group'] = 'G2'
df.loc[((df.HDI_Scale == "Low") & (df.IncomeCategory == "Low income")), 'Group'] = 'G1'

# Plot of Income Groups vs Loan Amount
fig, ax = plt.subplots(figsize=(10,6))

sns.violinplot("Group","log_amount",data=df, ax = ax)

ax.set_title("HDI/Income Group vs Amount", fontsize=20)
ax.set_xlabel("HDI/Income Group Composite", fontsize=15)
ax.set_ylabel("Amount", fontsize=15)

plt.tight_layout()
# We will group the rates of growth in categories
bins = [-7,0,3,7,11]
group_names = ['Low','Moderate','High','Very High']
metrics["GDP_growth"] = pd.cut(metrics["GDP_growthPercent"], bins, labels=group_names)

# We will group the Bank Lending rates in categories
bins1 = [2,5,8,20,53]
group_names1 = ['Low','Medium','High','Very High']
metrics["BankRate"] = pd.cut(metrics["Bank Rate"], bins1, labels=group_names1)

df_m = metrics.groupby(['HDI_Scale', 'IncomeCategory', 'GDP_growth', 'BankRate'])['NumberOfLoans'].sum().reset_index(name='NumberOfLoans').sort_values('NumberOfLoans', ascending = False).reset_index(drop=True)
tot = df_m["NumberOfLoans"].sum()
df_m["PercentLoans"] = df_m["NumberOfLoans"]/tot * 100

df_m1 = df_m[['HDI_Scale', 'IncomeCategory']].drop_duplicates().merge(df.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).sector.first())
df_m1 = df_m1.merge(df.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).activity.first())
df_m1 = df_m1.merge(df.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).repayment_interval.first())
df_m1 = df_m1.merge(df_m.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).BankRate.first())
df_m1 = df_m1.merge(df_m.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).GDP_growth.first())
df_m1 = df_m1.merge(df_m.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).NumberOfLoans.sum())
df_m1 = df_m1.merge(df_m.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).PercentLoans.sum())
df_m1 = df_m1.merge(df.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).borrower_female.sum())
df_m1 = df_m1.merge(df.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).borrower_male.sum())
df_m1 = df_m1.merge(df.groupby(['HDI_Scale', 'IncomeCategory'],as_index=False).loan_amount.sum())

df_m1.style.background_gradient(cmap="Set3_r")
metrics = countries.merge(gdp, how="left", on = "Country")
metrics = metrics.merge(hdi, how="left", on = "Country")
metrics = metrics.merge(inc, how="left", on = "Country")
metrics = metrics.merge(lend, how="left", on = "Country")
metrics = metrics.merge(pgdp, how="left", on = "Country")

# We will group the rates of growth in categories
bins = [-7,0,3,7,11]
group_names = ['Low','Moderate','High','Very High']
metrics["GDP_growth"] = pd.cut(metrics["GDP_growthPercent"], bins, labels=group_names)

# We will group the Bank Lending rates in categories
bins1 = [2,5,8,20,53]
group_names1 = ['Low','Medium','High','Very High']
metrics["BankRate"] = pd.cut(metrics["Bank Rate"], bins1, labels=group_names1)

fig, ax = plt.subplots(2,2,figsize=(20, 10))

a = sns.barplot(x="HDI_Scale", y="NumberOfLoans", data=metrics, palette="YlGn_r", ax=ax[0][0])
b = sns.barplot(x="IncomeCategory", y="NumberOfLoans", data=metrics, palette="GnBu_r", ax = ax[0][1])
c = sns.barplot(x="GDP_growth", y="NumberOfLoans", data=metrics, palette="Oranges", ax = ax[1][0])
d = sns.barplot(x="BankRate", y="NumberOfLoans", data=metrics, palette="RdPu_r", ax = ax[1][1])

ax[0][0].set_title('HDI Scale vs Number of Loans', fontsize=20)
ax[0][0].get_yaxis().set_visible(False)
ax[0][0].get_xaxis().get_label().set_visible(False)
ax[0][0].set_xticklabels(ax[0][0].get_xticklabels(), fontsize=15)

ax[0][1].set_title('Income Category vs Number of Loans', fontsize=20)
ax[0][1].get_yaxis().set_visible(False)
ax[0][1].get_xaxis().get_label().set_visible(False)
ax[0][1].set_xticklabels(ax[0][1].get_xticklabels(), fontsize=15)

ax[1][0].set_title('GDP Growth Category vs Number of Loans', fontsize=20)
ax[1][0].get_yaxis().set_visible(False)
ax[1][0].get_xaxis().get_label().set_visible(False)
ax[1][0].set_xticklabels(ax[1][0].get_xticklabels(), fontsize=15)

ax[1][1].set_title('Bank Interest Rate Category vs Number of Loans', fontsize=20)
ax[1][1].get_yaxis().set_visible(False)
ax[1][1].get_xaxis().get_label().set_visible(False)
ax[1][1].set_xticklabels(ax[1][1].get_xticklabels(), fontsize=15)

plt.suptitle('Number of Loans according to various aspects', fontsize=25, y=1.05)
plt.tight_layout()
plt.show()
df_m = metrics.groupby(['HDI_Scale', 'IncomeCategory', 'GDP_growth', 'BankRate'])['NumberOfLoans'].sum().reset_index(name='NumberOfLoans').sort_values('NumberOfLoans', ascending = False).reset_index(drop=True)
tot = df_m["NumberOfLoans"].sum()
df_m["PercentLoans"] = df_m["NumberOfLoans"]/tot * 100

df_m.style.background_gradient(cmap="Pastel1_r")
mid_loans = metrics[(metrics["HDI_Scale"]=='Medium') & (metrics["IncomeCategory"]=='Lower middle income')].NumberOfLoans.sum()
high_loans = metrics[(metrics["GDP_growth"]=='High')].NumberOfLoans.sum()
br_loans = metrics[(metrics["BankRate"]=='High')].NumberOfLoans.sum()
mix_loans = metrics[(metrics["HDI_Scale"]=='Medium') & (metrics["IncomeCategory"]=='Lower middle income') & (metrics["GDP_growth"]=='High')].NumberOfLoans.sum()
total_loans = metrics['NumberOfLoans'].sum()
per_mid_loans = (mid_loans/total_loans) * 100
per_br_loans = (br_loans/total_loans) * 100
per_high_loans = (high_loans/total_loans) * 100
per_mix_loans = (mix_loans/total_loans) * 100
v1 = TfidfVectorizer(stop_words='english', max_df=0.1)
x1 = v1.fit_transform(df[(df.HDI_Scale=='Low') & (df.IncomeCategory=='Low income')].use.dropna())
weights1 = np.asarray(x1.mean(axis=0)).ravel().tolist()
weights_df1 = pd.DataFrame({'term': v1.get_feature_names(), 'weight': weights1})

v2 = TfidfVectorizer(stop_words='english', max_df=0.1)
x2 = v2.fit_transform(df[(df.HDI_Scale=='Medium') & (df.IncomeCategory=='Lower middle income')].use.dropna())
weights2 = np.asarray(x2.mean(axis=0)).ravel().tolist()
weights_df2 = pd.DataFrame({'term': v2.get_feature_names(), 'weight': weights2})

v3 = TfidfVectorizer(stop_words='english', max_df=0.1)
x3 = v3.fit_transform(df[(df.HDI_Scale=='High') & (df.IncomeCategory=='Upper middle income')].use.dropna())
weights3 = np.asarray(x3.mean(axis=0)).ravel().tolist()
weights_df3 = pd.DataFrame({'term': v3.get_feature_names(), 'weight': weights3})

v4 = TfidfVectorizer(stop_words='english', max_df=0.1)
x4 = v4.fit_transform(df[(df.HDI_Scale=='Very High') & (df.IncomeCategory=='High income')].use.dropna())
weights4 = np.asarray(x4.mean(axis=0)).ravel().tolist()
weights_df4 = pd.DataFrame({'term': v4.get_feature_names(), 'weight': weights4})


w1 = weights_df1.sort_values(by='weight', ascending=False)[:10].reset_index(drop=True)
w2 = weights_df2.sort_values(by='weight', ascending=False)[:10].reset_index(drop=True)
w3 = weights_df3.sort_values(by='weight', ascending=False)[:10].reset_index(drop=True)
w4 = weights_df4.sort_values(by='weight', ascending=False)[:10].reset_index(drop=True)
fig, ax = plt.subplots(2,2,figsize=(20, 8))

a = sns.barplot(w1['term'], w1['weight'], palette="copper", ax=ax[0][0], order=w1["term"].tolist(), linewidth=1, edgecolor='k')
b = sns.barplot(w2['term'], w2['weight'], palette="bone", ax=ax[0][1], order=w2["term"].tolist(), linewidth=1, edgecolor='k')
c = sns.barplot(w3['term'], w3['weight'], palette="Purples_r", ax=ax[1][0], order=w3["term"].tolist(), linewidth=1, edgecolor='k')
d = sns.barplot(w4['term'], w4['weight'], palette="gist_heat", ax=ax[1][1], order=w4["term"].tolist(), linewidth=1, edgecolor='k')

ax[0][0].set_title('Low HDI & Low income (Food : Butcher Shop)', fontsize=20)
ax[0][0].get_yaxis().set_visible(False)
ax[0][0].get_xaxis().get_label().set_visible(False)
ax[0][0].set_xticklabels(ax[0][0].get_xticklabels(), fontsize=15, rotation=45)

ax[0][1].set_title('Medium HDI & Lower middle income (Food : Fuits & Vegetables)', fontsize=20)
ax[0][1].get_yaxis().set_visible(False)
ax[0][1].get_xaxis().get_label().set_visible(False)
ax[0][1].set_xticklabels(ax[0][1].get_xticklabels(), fontsize=15, rotation=45)

ax[1][0].set_title('High HDI & Upper middle income (Personal Use : Personal Expenses)', fontsize=20)
ax[1][0].get_yaxis().set_visible(False)
ax[1][0].get_xaxis().get_label().set_visible(False)
ax[1][0].set_xticklabels(ax[1][0].get_xticklabels(), fontsize=15, rotation=45)

ax[1][1].set_title('Very High HDI & High income (Food : Food Production/Sales)', fontsize=20)
ax[1][1].get_yaxis().set_visible(False)
ax[1][1].get_xaxis().get_label().set_visible(False)
ax[1][1].set_xticklabels(ax[1][1].get_xticklabels(), fontsize=15, rotation=45)

plt.suptitle('Most frequent use for loans per HDI/Income group (incl. Top Sector : Top Activity)', fontsize=25, y=1.08)
plt.tight_layout()
plt.show()
# Source: World Bank datasets
# International Bank for Reconstruction and Development Loans to countries in need
ibrd = pd.read_csv('../input/kiva-additional-data/IBRD_Loans.csv', 'Data', index_col=None, engine='python', delimiter=",")

# Inflation
infl = pd.read_excel('../input/kiva-additional-data/Inflation.xls', 'Data', index_col=None)

# GINI Index (Inequality)
gini = pd.read_excel('../input/kiva-additional-data/GINI Index.xls', 'Data', index_col=None)

# Growth in sectors
sectors = pd.read_excel('../input/kiva-additional-data/By_Sector.xlsx', 'Data', index_col=None)

# Literacy Rate Female
lit_f = pd.read_excel('../input/kiva-additional-data/Literacy_Rate_F.xls', 'Data', index_col=None)

# Literacy Rate Male
lit_m = pd.read_excel('../input/kiva-additional-data/Literacy_Rate_M.xls', 'Data', index_col=None)

# Population Density
popu_den = pd.read_excel('../input/kiva-additional-data/Population Density.xls', 'Data', index_col=None)
# MPI of countries
mpi_c = pd.read_excel('../input/kiva-additional-data/MPI_National.xlsx', 'Data', index_col=None)

#MPI of Regions
mpi_r = pd.read_excel('../input/kiva-additional-data/MPI_Regional.xlsx', 'Data', index_col=None)

mpi1 = mpi_r[['Country', 'Region','MPI_Region']].merge(mpi_c, how="right", on = "Country")
mpi1 = mpi1[['Country_Code', 'World_Region', 'Country', 'Region', 'MPI_Country', 'MPI_Region', 'Population_in_Med_MPI_p', 'Population_in_High_MPI_p', 'Destitutes_p', 'Inequality', 'Population']]
mpi1['MPI_Region'] = mpi1['MPI_Region'].fillna(0)

# Merge MPI with GDP & HDI
mpi2 = mpi1.merge(countries[['Country']], how="outer", on = "Country")
mpi2 = mpi2.merge(gdp, how="left", on = "Country")
mpi2 = mpi2.merge(hdi, how="left", on = "Country")
mpi2.loc[(mpi2.HDI_Scale=='Very High') & (mpi2['MPI_Country'].isnull()), 'MPI_Country'] = 0
mpi2.loc[(mpi2.HDI_Scale=='Very High') & (mpi2['MPI_Country'].isnull()), 'MPI_Region'] = 0
mpi2.loc[(mpi2.HDI_Scale=='High') & (mpi2['MPI_Country'].isnull()), 'MPI_Country'] = 0.001
mpi2.loc[(mpi2.HDI_Scale=='High') & (mpi2['MPI_Country'].isnull()), 'MPI_Region'] = 0.001

# Merge with other metrics
mpi3 = mpi2.copy()
mpi3 = mpi3.merge(lend, how="left", on = "Country")
mpi3 = mpi3.merge(pgdp, how="left", on = "Country")
mpi3 = mpi3.merge(ibrd, how="left", on = "Country")
mpi3 = mpi3.merge(infl, how="left", on = "Country")
mpi3 = mpi3.merge(gini, how="left", on = "Country")
mpi3 = mpi3.merge(sectors, how="left", on = "Country")
mpi3 = mpi3.merge(lit_f, how="left", on = "Country")
mpi3 = mpi3.merge(lit_m, how="left", on = "Country")
mpi3 = mpi3.merge(popu_den, how="left", on = "Country")
mpi3 = mpi3.merge(inc, how="left", on = "Country")

bins = [0,0.05,0.35,0.7]
group_names = ['Low','Medium','High']
mpi3["MPI_Category"] = pd.cut(mpi3["MPI_Country"], bins, labels=group_names)

mpi3["Sector_total"] = mpi3["Industry"] + mpi3["Manu"] + mpi3["Services"] - mpi3["Agri"]
mpi3["Literacy"] = (mpi3["Lit_Rate_F"] + mpi3["Lit_Rate_M"])/2

fig, ax = plt.subplots(figsize=(25,18))
c = mpi3.corr()
sns.heatmap(c, annot=True, ax=ax, square=True, cmap="ocean")
plt.show()
#mpi3.loc[((mpi3.HDI_Scale == "Very High") & (mpi3.IncomeCategory == "High income") & (mpi3.MPI_Category == "Low")), 'Score'] = 'P15'
mpi3.loc[((mpi3.HDI_Scale == "High") & (mpi3.IncomeCategory == "Upper middle income") & (mpi3.MPI_Category == "Low")), 'Score'] = 'P14'
mpi3.loc[((mpi3.HDI_Scale == "High") & (mpi3.IncomeCategory == "Lower middle income") & (mpi3.MPI_Category == "Low")), 'Score'] = 'P13'
mpi3.loc[((mpi3.HDI_Scale == "Medium") & (mpi3.IncomeCategory == "Lower middle income") & (mpi3.MPI_Category == "Low")), 'Score'] = 'P12'
#mpi3.loc[((mpi3.HDI_Scale == "Medium") & (mpi3.IncomeCategory == "Upper middle income") & (mpi3.MPI_Category == "Low")), 'Score'] = 'P11'
#mpi3.loc[((mpi3.HDI_Scale == "Medium") & (mpi3.IncomeCategory == "Low income") & (mpi3.MPI_Category == "Low")), 'Score'] = 'P10'
mpi3.loc[((mpi3.HDI_Scale == "Medium") & (mpi3.IncomeCategory == "Lower middle income") & (mpi3.MPI_Category == "Medium")), 'Score'] = 'P9'
mpi3.loc[((mpi3.HDI_Scale == "Medium") & (mpi3.IncomeCategory == "Upper middle income") & (mpi3.MPI_Category == "Medium")), 'Score'] = 'P8'
mpi3.loc[((mpi3.HDI_Scale == "Medium") & (mpi3.IncomeCategory == "Low income") & (mpi3.MPI_Category == "Medium")), 'Score'] = 'P7'
#mpi3.loc[((mpi3.HDI_Scale == "Low") & (mpi3.IncomeCategory == "Lower middle income") & (mpi3.MPI_Category == "Low")), 'Score'] = 'P6'
#mpi3.loc[((mpi3.HDI_Scale == "Low") & (mpi3.IncomeCategory == "Low income") & (mpi3.MPI_Category == "Low")), 'Score'] = 'P5'
mpi3.loc[((mpi3.HDI_Scale == "Low") & (mpi3.IncomeCategory == "Lower middle income") & (mpi3.MPI_Category == "Medium")), 'Score'] = 'P4'
mpi3.loc[((mpi3.HDI_Scale == "Low") & (mpi3.IncomeCategory == "Low income") & (mpi3.MPI_Category == "Medium")), 'Score'] = 'P3'
#mpi3.loc[((mpi3.HDI_Scale == "Low") & (mpi3.IncomeCategory == "Lower middle income") & (mpi3.MPI_Category == "High")), 'Score'] = 'P2'
mpi3.loc[((mpi3.HDI_Scale == "Low") & (mpi3.IncomeCategory == "Low income") & (mpi3.MPI_Category == "High")), 'Score'] = 'P1'
mpi4=mpi3[['GDP_per_Capita', 'GNI_per_Capita',
           'HDI_Value', 'IHDI_Value',
           'GINI Index', 'IBRD_loan', 
           'Literacy', 'Sector_total', 
           'MPI_Country','MPI_Region', 'Population_in_High_MPI_p', 'Destitutes_p', 'Inequality',
           'Score']]
# 'GDP_per_Capita', 'GNI_per_Capita', 'HDI_Rank', 'HDI_Value', 'IHDI_Value'
# 'Amount_bl', 'GINI Index', 
# 'Agri', 'Industry', 'Manu', 'Services', 'Sector_total', 
# 'Lit_Rate_F', 'Lit_Rate_M', 'Literacy', 
# 'MPI_Country','MPI_Region', 'Population_in_High_MPI_p', 'Destitutes_p', 'Inequality', 

mpi4=mpi4.dropna().reset_index(drop=True)

fig,ax=plt.subplots(figsize=(20,10))

sm = pd.plotting.scatter_matrix(mpi4, diagonal="kde", c='#41caf4',ax=ax)
for subaxis in sm:
        for ax in subaxis:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])            
            ax.set_ylabel("")
            ax.set_xlabel("")

plt.tight_layout()
plt.show()
X = mpi4.loc[:, mpi4.columns != 'Score']
y=mpi4.Score

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors =1)
model_k = knn.fit(Xtrain, ytrain)
preds_k = model_k.predict(Xtest)

#print(classification_report(ytest, preds_k))

Xtest.to_csv('predictions.csv')
p=pd.read_csv('predictions.csv')
p['Results'] = preds_k
p.to_csv('predictions.csv')

print ('Poverty Score Grouping Completed')
top20_data = pd.read_csv("../input/kiva-self-data/Top20_Countries_geo.csv")
top20_data = top20_data.merge(metrics, how="left", on = "Country")
top20_data = top20_data.merge(kiva_df.groupby(['Country'],as_index=False).Amount.sum())
top20_data = top20_data.merge(kiva_df.groupby(['Country'],as_index=False).Lenders.sum())
top20_data = top20_data.merge(kiva_df.groupby(['Country'],as_index=False).Females.sum())
top20_data = top20_data.merge(kiva_df.groupby(['Country'],as_index=False).Males.sum())
top20_data = top20_data.merge(kiva_df.groupby('Country',as_index=False).Sector.first())
top20_data = top20_data.merge(kiva_df.groupby('Country',as_index=False).Activity.first())
top20_data = top20_data.merge(kiva_df.groupby('Country',as_index=False).Repayment_Interval.first())
top20_data = top20_data.merge(kiva_df.groupby('Country',as_index=False).Term.first())
top20_data = top20_data.drop(labels=['GDP_Total', 'GDP_per_Capita', 'HDI_Value'], axis=1)

m = folium.Map(location=[0,0], tiles="Mapbox Control Room",  zoom_start=2)

for i in range(0,len(top20_data)):
    sr = round(top20_data.iloc[i]["Males"]/top20_data.iloc[i]["Females"], 2)
    popup_text = "Country: {}<br> Rank: {}<br> Loans: {}<br> Sector: {}<br> Sex Ratio: {}<br> Total Amount: {}<br> Repayment Interval: {}<br> HDI Scale: {}<br> Income Category: {} "
    popup_text = popup_text.format(top20_data.iloc[i]["Country"],
                                   i+1,
                                   top20_data.iloc[i]["NumberOfLoans"],
                                   top20_data.iloc[i]["Sector"],
                                   sr,
                                   top20_data.iloc[i]['Amount'],                                  
                                  top20_data.iloc[i]['Repayment_Interval'],
                                  top20_data.iloc[i]['HDI_Scale'],
                                  top20_data.iloc[i]['IncomeCategory'])
    
    
    folium.Marker([top20_data.iloc[i]['c_lat'],
                   top20_data.iloc[i]['c_lon']],
                  popup=popup_text,
                 icon = folium.Icon(color='darkgreen', icon='address-card')).add_to(m)
    
display(m)
# We will keep the above dataset for future reference & create another one for current (Country-wise) analysis
kiva_reg = kiva_full.copy()
kiva_reg = kiva_reg.drop(labels=['id', 'use', 'tags', 'country_code', 'currency', 'partner_id', 'disbursed_time', 'funded_time', 'posted_time', 'borrower_genders', 'funded_amount'], axis=1)
kiva_reg = kiva_reg.dropna(subset = ['region'], axis=0)

# Region
for i, row in kiva_reg.iterrows():
    g = list()
    val = row.loc['region']
    g.extend(l.strip() for l in val.split(","))
    
    if (len(g) > 2):
        reg = g[-1:]
    else:
        reg = g
    
    reg = ' '.join(str(r) for r in reg)    
    
    kiva_reg.set_value(i, 'reg', reg)    
    
kiva_reg.columns = ['Amount', 'Activity', 'Sector', 'Country', 'SubRegion', 'Term', 'Lenders', 'Repayment_Interval', 'Date', 'Females', 'Males', 'Log_Amount', 'Region']
kiva_reg = kiva_reg [['Country', 'Region', 'Sector', 'Activity', 'Date', 'Repayment_Interval', 'Term', 'Amount', 'Log_Amount', 'Lenders', 'Females', 'Males', 'SubRegion']]

# We will make a copy of the dataset for the current analysis
df_reg_all = kiva_reg.groupby(['Country','Region']).size().reset_index(name="NumberOfLoans").sort_values('NumberOfLoans', ascending=False).reset_index(drop=True)
# List of top 20 regions & their geographic coordinates
top20_reg_geo = df_reg_all[:20].copy()

top20_reg_data = pd.read_csv("../input/kiva-self-data/Top20_Regions_geo.csv", engine="python")

#top20_reg_data = top20_reg_geo.copy()
top20_reg_data = top20_reg_data.merge(kiva_reg.groupby(['Region'],as_index=False).Amount.sum())
top20_reg_data = top20_reg_data.merge(kiva_reg.groupby(['Region'],as_index=False).Lenders.sum())
top20_reg_data = top20_reg_data.merge(kiva_reg.groupby(['Region'],as_index=False).Females.sum())
top20_reg_data = top20_reg_data.merge(kiva_reg.groupby(['Region'],as_index=False).Males.sum())
top20_reg_data = top20_reg_data.merge(kiva_reg.groupby('Region',as_index=False).Sector.first())
top20_reg_data = top20_reg_data.merge(kiva_reg.groupby('Region',as_index=False).Activity.first())
top20_reg_data = top20_reg_data.merge(kiva_reg.groupby('Region',as_index=False).Repayment_Interval.first())
top20_reg_data = top20_reg_data.merge(kiva_reg.groupby('Region',as_index=False).Term.first())
m1 = folium.Map(location=[0,0], tiles="Stamen Toner",  zoom_start=2)

for i in range(0,len(top20_reg_data)):
    sr1 = round(top20_reg_data.iloc[i]["Males"]/top20_reg_data.iloc[i]["Females"], 2)
    popup_text1 = "Country: {}<br> Region: {}<br> Rank: {}<br> Loans: {}<br> Sector: {}<br> Sex Ratio: {}<br> Total Loan: {}<br> Repayment Interval: {}"
    popup_text1 = popup_text1.format(top20_reg_data.iloc[i]["Country"],
                                   top20_reg_data.iloc[i]["Region"],
                                   i+1,
                                   top20_reg_data.iloc[i]["NumberOfLoans"],
                                   top20_reg_data.iloc[i]["Sector"],
                                   sr1,
                                  top20_reg_data.iloc[i]['Amount'],
                                  top20_reg_data.iloc[i]['Repayment_Interval'])
    
    
    folium.Marker([top20_reg_data.iloc[i]['r_lat'],
                   top20_reg_data.iloc[i]['r_lon']],
                  popup=popup_text1,
                 icon = folium.Icon(color='purple', icon='address-card')).add_to(m1)
    
display(m1)
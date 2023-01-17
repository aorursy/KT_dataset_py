import numpy as np
import pandas as pd 
import os
import datetime as dt
import gc
from sklearn import cluster
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.offline.init_notebook_mode(connected=True)
%matplotlib inline
print(os.listdir("../input/part-1-preprocessing-feature-engineering"))
print(os.listdir("../input/io"))
# define columns to import
projectCols = ['Project ID', 'School ID', 'Teacher ID',
               'Teacher Project Posted Sequence', 'Project Type',
               'Project Subject Category Tree', 'Project Subject Subcategory Tree',
               'Project Grade Level Category', 'Project Resource Category',
               'Project Cost', 'Project Posted Date', 'Project Expiration Date',
               'Project Current Status', 'Project Fully Funded Date']

resourcesCols = ['Project ID','Resource Quantity','Resource Unit Price', 'Resource Vendor Name']

# import files
donations = pd.read_csv('../input/io/Donations.csv', dtype = {'Donation Amount': np.float32, 'Donor Cart Sequence': np.int32})
donors = pd.read_csv('../input/io/Donors.csv', dtype = {'Donor Zip':'str'})
projects = pd.read_csv('../input/io/Projects.csv', usecols = projectCols, dtype = {'Teacher Project Posted Sequence': np.float32, 'Project Cost': np.float32})
resources = pd.read_csv('../input/io/Resources.csv', usecols = resourcesCols, dtype = {'Resource Quantity': np.float32,'Resource Unit Price': np.float32})
schools = pd.read_csv('../input/io/Schools.csv', dtype = {'School Zip': 'str'})
teachers = pd.read_csv('../input/io/Teachers.csv')

# These are files from Part I:
donorFeatureMatrixNoAdj = pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorFeatureMatrixNoAdj.csv')
donorFeatureMatrix =  pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorFeatureMatrix.csv')
donorsMapping = pd.read_csv('../input/part-1-preprocessing-feature-engineering/donorsMapping.csv') 
schoolsMapping = pd.read_csv('../input/part-1-preprocessing-feature-engineering/schoolsMapping.csv')
projFeatures = pd.read_csv('../input/part-1-preprocessing-feature-engineering/projFeatures.csv')
distFeatures = pd.read_csv('../input/part-1-preprocessing-feature-engineering/distFeatures.csv')
# donations
donations['Donation Received Date'] = pd.to_datetime(donations['Donation Received Date'])
donations['Donation Included Optional Donation'].replace(('Yes', 'No'), (1, 0), inplace=True)
donations['Donation Included Optional Donation'] = donations['Donation Included Optional Donation'].astype('bool')
donations['Donation_Received_Year'] = donations['Donation Received Date'].dt.year
donations['Donation_Received_Month'] = donations['Donation Received Date'].dt.month
donations['Donation_Received_Day'] = donations['Donation Received Date'].dt.day

# donors
donors['Donor Is Teacher'].replace(('Yes', 'No'), (1, 0), inplace=True)
donors['Donor Is Teacher'] = donors['Donor Is Teacher'].astype('bool')

# projects
cols = ['Project Posted Date', 'Project Fully Funded Date']
projects.loc[:, cols] = projects.loc[:, cols].apply(pd.to_datetime)
projects['Days_to_Fullyfunded'] = projects['Project Fully Funded Date'] - projects['Project Posted Date']

# teachers
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])

##
# Name the dataframes
##
def name_dataframes(dfList, dfNames):
    '''
    give names to a list of dataframes. 
    Argument:
        dfList = list of dataframes,
        dfNames = list of names for the dataframes
    Return:
        None
    '''
    for df, name in zip(dfList, dfNames):
        df.name = name
    
    return

dfList = [donations, donors, projects, resources, schools, teachers]
dfNames = ['donations', 'donors', 'projects', 'resources', 'schools', 'teachers']
name_dataframes(dfList, dfNames)

##
#  Remove rows in the datasets that cannot be mapped
##
projects = projects.loc[projects['School ID'].isin(schools['School ID'])]
projects = projects.loc[projects['Project ID'].isin(resources['Project ID'])]
donations = donations.loc[donations['Project ID'].isin(projects['Project ID'])]
donations = donations.loc[donations['Donor ID'].isin(donors['Donor ID'])]
donors = donors.loc[donors['Donor ID'].isin(donations['Donor ID'])]

donors = donors.merge(donorsMapping, left_on = 'Donor ID', right_on = 'Donor ID', how = 'left')
donors = donors.drop(['Unnamed: 0'], axis=1)
schools = schools.merge(schoolsMapping.filter(items = ['School ID', 'School_Lon', 'School_Lat']), left_on = 'School ID', right_on = 'School ID', how = 'left')
allDonors = set(donations['Donor ID'])
totalDonors = len(allDonors)
print("There are " + "{:,}".format(totalDonors) + " unique donors in the donations dataset")

multipleCartDonors = set(donations[donations['Donor Cart Sequence'] > 1]['Donor ID'])
print("% of donors donated multiple times (multiple carts): ", "{0:.2f}".format(len(multipleCartDonors)/totalDonors*100))

singleCartDonors = allDonors.difference(multipleCartDonors) # new set with elements in s but not in t
print("% of donors donated once: ", "{0:.2f}".format(len(singleCartDonors)/totalDonors*100))

# Since a donor can donate to multiple projects in a single Cart, 
#let us find the donors that only donated once so far, but in that cart, donated to multiple projects.
temp = donations.groupby(['Donor ID']).agg({'Project ID': 'nunique'})
multipleProject = temp[temp['Project ID'] > 1]
multipleProjectDonors = set(multipleProject.index)
print("% of donors donated to multiple projects: ", "{0:.2f}".format(len(multipleProjectDonors)/totalDonors*100))

singleCartmultipleProjectDonors = singleCartDonors.intersection(multipleProjectDonors)
print("% of donors donated to multiple projects but only donated once: ", "{0:.2f}".format(len(singleCartmultipleProjectDonors)/totalDonors*100))

donorTotal = donations.groupby(['Donor ID']).agg({'Donation Amount': 'sum',  'Donor ID': 'count', 'Project ID':'nunique'}).sort_values(by = ['Donation Amount'], ascending=False)
donorTotal = donorTotal.rename(columns={'Donation Amount':'Total Donation Amount', 'Donor ID':'Total Donation Occurances', 'Project ID':'Total Projects'}).reset_index()
donorTotal['% Total Donation'] = donorTotal['Total Donation Amount']/sum(donorTotal['Total Donation Amount'])*100
donorTotal['cumulative % of total donations'] = donorTotal['% Total Donation'].cumsum()
donorTotal = donorTotal.reset_index()
donorTotal['cumulative % of total donors'] =  donorTotal['index']/totalDonors*100
donorTotal.loc[donorTotal['cumulative % of total donors'] > 10].head(1)

plt.plot(donorTotal['cumulative % of total donors'], donorTotal['cumulative % of total donations'])
plt.xlabel('% of total donors')
plt.ylabel('% of funding')
plt.title('Top 25% of donors provided 80% of the fundings ')
plt.plot()
# Prepare Chart Data
chartData = donations[['Donor ID', 'Project ID', 'Donation Amount']].merge(distFeatures[['Donor ID', 'Project ID', 'dist']], left_on = ['Donor ID', 'Project ID'], right_on = ['Donor ID', 'Project ID'], how = 'left')
chartData = chartData.merge(donors[['Donor ID', 'no_mismatch']], left_on = 'Donor ID', right_on = 'Donor ID', how = 'left')
chartData = chartData.loc[chartData['no_mismatch'] == 1]
chartData['dist_cut'] = pd.cut(chartData['dist'], bins = [-1, 0, 5, 10, 20, 50, 100, 15000], labels = ['0', '1-5', '6-10', '11-20', '21-50', '51-100', '>100'])
chartData['Total Amount'] = chartData.groupby(['Donor ID', 'Project ID'])['Donation Amount'].transform('sum')
chartData = chartData.drop_duplicates(subset=['Project ID', 'Donor ID'], keep='first')
chart = pd.DataFrame(chartData.groupby('dist_cut')['Total Amount'].agg('sum')/chartData['Donation Amount'].sum()*100).reset_index(drop = False)

# Plot chart
plot = sns.barplot(x = 'dist_cut', y = 'Total Amount', data = chart)
plot.set_title("% of donations relative to distances between donor and schools")
plot.set_ylabel('% donations')
plot.set_xlabel('distances in miles')

print("The median distance in miles between project/donor pair is:", chartData['dist'].median())
print(chart)
def merge_data(idList):
    ''' 
    Filter data based on a list of Donor ID.  Merge all data together into one dataframe.  
    Arguments: list of 'Donor ID'
    Returns: dataframe 
    '''
    temp = donations[donations['Donor ID'].isin(idList)].reset_index(drop = True)
    temp = temp.merge(donors, on = 'Donor ID', how='left')
    temp = temp.merge(projects, on = 'Project ID', how = 'left')
    temp = temp.merge(resources, on = 'Project ID', how = 'left')
    temp = temp.merge(schools, on = 'School ID', how = 'left')
    temp = temp.merge(teachers, on = 'Teacher ID', how = 'left')
    
    return temp


def summarize_by_city(df):
    ''' 
    Calculate donation amount in each city and scale to be used in the plot.
    '''
    df['cityTotal'] = df['Donation Amount'].groupby(df['School City']).transform('sum')
    chartData = df.groupby('School City').first().sort_values(by = 'cityTotal', ascending=False).reset_index()
    chartData = chartData[['School City','cityTotal', 'School_Lon', 'School_Lat', 'Donor City', 'Donor_Lat', 'Donor_Lon']].copy(deep = True)
    chartData['text'] = chartData['School City'] + ': $' + chartData['cityTotal'].apply('{0:,.0f}'.format).astype(str)

    # define cuts for plot
    top01 = chartData['cityTotal'].quantile(0.999)
    top1 = chartData['cityTotal'].quantile(0.99)
    top10 = chartData['cityTotal'].quantile(0.9)
    top50 = chartData['cityTotal'].quantile(0.5)
    topMax = chartData['cityTotal'].max()+1
    
    # bin donation
    chartData['group'] = pd.cut(chartData['cityTotal'],np.array([-0.1, top50, top10, top1, top01, topMax]), 3,
                           labels = ['Bottom 50%','11-50%', 'Top 2-10%', 'Top 1%', 'Top 0.01%'])
    
    # calculate scale
    scale = chartData['cityTotal'].median()
    return chartData, scale

def get_chart_index(chartData):
    ''' 
    Since the bubble map has various groupings according to the percentile of the donation, this function
    identifies the row numbers of the data to plot in each grouping as well as the colors for each group.
    '''
    x = chartData['group'].value_counts()
    x = x.reindex(index = ['Top 0.01%', 'Top 1%', 'Top 2-10%', '11-50%', 'Bottom 50%'])
    x = x.loc[x>0]
    numX = len(x.loc[x>0])
    limx = 0
    limits = []
    colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)", "rgb(232,255,184)"]
    
    # get colors:
    colors = colors[0:numX]
    colors.append('lightgrey')
    
    # get limits 
    # Not all datasets have value in every group, so we loop through the available values only
    for i in range(numX):
        limy = limx+x[i]
        limitEntry = (limx, limy)
        limx = limy
        limits.append(limitEntry)

    return limits, colors

def prepare_cities_donation(chartData, scale, limits, colors):
    ''' 
    prepare data into the format that plotly used for bubble map
    this applies to cities that the donor donated to
    '''
    cities = []
    for i in range(len(limits)):
        lim = limits[i]
        df_sub = chartData[lim[0]:lim[1]]
        city = dict(
            type = 'scattergeo',
            locationmode = 'USA-states',
            lon = df_sub['School_Lon'],
            lat = df_sub['School_Lat'],
            text = df_sub['text'],
            marker = dict(
                size = df_sub['cityTotal']/scale,
                color = colors[i],
                line = dict(width=0.5, color='rgb(40,40,40)'),
                sizemode = 'area'),
            name = df_sub.iloc[0]['group'])
            #name = '{0} - {1}'.format(lim[0],lim[1]) )
        cities.append(city)
    return cities

def donor_city_total(chartData):
    '''
    Calculate the amount that the donor donates to his/her home city
    '''
    donorHome = chartData['Donor City'][0]
    if (chartData['School City'] == donorHome).sum()>0:
        donorCityTotal = chartData[chartData['School City'] == donorHome]['cityTotal'].apply('{0:,.0f}'.format).astype(str)
        donorTotal = chartData[chartData['School City'] == donorHome]['cityTotal']
    else:
        donorCityTotal = '0'
        donorTotal = 0
    return donorCityTotal, donorTotal, donorHome 

def prepare_donor_donation(chartData, donorTotal, donorCityTotal, scale, donorHome):
    ''' 
    prepare data into the format that plotly used for bubble map
    this applies to donation in the donor's home town
    '''
    donorCity = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = max(chartData['Donor_Lon']),
        lat = max(chartData['Donor_Lat']),
        text = "Donor's City: " + donorHome + " $" + donorCityTotal,
        marker = dict(
            size = donorTotal/scale,
            color = 'rgb(40,40,40)',
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'),
        name = "Donor's City")
    return donorCity

def plot_bubble_map(idList):
    '''
    Run list of data preparation steps in order to plot the bubble map
    '''
    df = merge_data(idList)
    chartData, scale = summarize_by_city(df)
    limits, colors = get_chart_index(chartData)
    cities = prepare_cities_donation(chartData, scale, limits, colors)
    donorCityTotal, donorTotal, donorHome  = donor_city_total(chartData)
    donorCity = prepare_donor_donation(chartData, donorTotal, donorCityTotal, scale, donorHome)

    cities.append(donorCity)

    layout = dict(
            title = 'Donation by City<br>(Click legend to toggle traces)',
            showlegend = True,
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showland = True,
                landcolor = 'rgb(217, 217, 217)',
                subunitwidth=1,
                countrywidth=1,
                subunitcolor="rgb(255, 255, 255)",
                countrycolor="rgb(255, 255, 255)"
            ),
        )
    fig = dict( data=cities, layout=layout )
    
    return py.offline.iplot(fig, validate=False, filename='donation map' )
    
idList = ['4416745560343f14a74dedcda4ec03b0']
plot_bubble_map(idList)
def stacked_chart_data(dframe, index, values, columns, orderBy, xAxisOrder = None):
    chartData = pd.pivot_table(dframe,
                               index = index,
                               values = values,
                               columns = columns,
                               aggfunc=[np.sum],
                               fill_value = 0 )
    chartData.columns = chartData.columns.droplevel()
    
    # if want to re-order x-axis
    if xAxisOrder is not None:
        chartData = chartData.reindex(xAxisOrder)
        
    else: 
        # Re-order columns in the data (re-order index in chartData).  
        temp = pd.pivot_table(dframe,
                              index = index,
                              values = values,
                              aggfunc=[np.sum],
                              fill_value = 0 )
        temp.columns = temp.columns.droplevel()
        temp = temp.reindex(temp.sort_values(by = values, ascending=False).index)
        newXAxisOrder = list(temp.index)
        chartData = chartData.reindex(newXAxisOrder)


    # Re-order legends in the data. Highest in the fill should be Project Resource Category with highest donation amount.
    temp = pd.pivot_table(dframe,
                              index = orderBy,
                              values = values,
                              aggfunc=[np.sum],
                              fill_value = 0 )
    temp.columns = temp.columns.droplevel()
    temp = temp.reindex(temp.sort_values(by = values, ascending=True).index)
    newColOrder = list(temp.index)
    chartData = chartData[newColOrder]

    return chartData
def stacked_bar(chartData, title):
    
    # create empty list to store data
    data = []

    # each category is a separate trace on the stacked chart
    for colName in chartData.columns:
        data.append({
            'type' : 'bar',
            'x' : chartData.index,
            'y' : chartData[colName],
            'name' : colName
        })

    # Set layout
    layout = go.Layout(
        barmode='stack', 
        title = "Total Donation by "+ title
    )

    # Plot figure
    fig = go.Figure(data=data, layout=layout)
    # py.offline.iplot(fig, filename='stacked-bar')
    
    return py.offline.iplot(fig, filename='stacked-bar')
def stacked_horiz_bar(chartData, title):
    
    # reverse chartData for horizontal chart.  Highest number on top
    chartData = chartData.reindex(index=chartData.index[::-1])
    
    # create empty list to store data
    data = []

    # each category is a separate trace on the stacked chart
    for colName in chartData.columns:
        data.append({
            'type' : 'bar',
            'y' : chartData.index,
            'x' : chartData[colName],
            'name' : colName,
            'orientation' : 'h'
        })

    # Set layout
    layout = go.Layout(
        barmode='stack', 
        title = "Total Donation by "+ title,
        font= dict(size=12),
        margin=go.Margin(l=200)
    )


    # Plot figure
    fig = go.Figure(data=data, layout=layout)
    # py.offline.iplot(fig, filename='stacked-bar')
    
    return py.offline.iplot(fig, filename='stacked-bar')
df = merge_data(idList)
index = 'Project Grade Level Category'
values = 'Donation Amount'
columns = 'Project Resource Category'
xAxisOrder = ['Grades PreK-2', 'Grades 3-5', 'Grades 6-8', 'Grades 9-12']

orderBy = 'Project Resource Category'

chart = stacked_chart_data(df, index, values, columns, orderBy, xAxisOrder)
stacked_bar(chart, index)
index = 'School Metro Type'
values = 'Donation Amount'
columns = 'Project Resource Category'
xAxisOrder = None
orderBy = 'Project Resource Category'
chart = stacked_chart_data(df, index, values, columns, orderBy, xAxisOrder)
stacked_bar(chart, index)
index =  'School Name'
values = 'Donation Amount'
columns = 'Project Resource Category'
xAxisOrder = None
orderBy = 'Project Resource Category'

chart = stacked_chart_data(df, index, values, columns, orderBy, xAxisOrder)

chart_1 = chart[0:10].copy(deep = True)
chart_1.loc[len(chart_1)] = chart[10:].sum(axis = 0)
chart_1 = chart_1.rename(index={10: 'Others'})
stacked_horiz_bar(chart_1, 'Top 10 Schools')

index = 'School Percentage Free Lunch'
values = 'Donation Amount'
columns = 'Project Resource Category'
xAxisOrder = None
orderBy = 'Project Resource Category'

chart = stacked_chart_data(df, index, values, columns, orderBy, xAxisOrder)
stacked_horiz_bar(chart, index)
index = 'Project Resource Category'
values = 'Donation Amount'
columns = 'Project Subject Subcategory Tree'
xAxisOrder = None
orderBy = 'Project Subject Subcategory Tree'

chart = stacked_chart_data(df, index, values, columns, orderBy, xAxisOrder)
stacked_bar(chart, index+' & '+columns)
index = 'Project Resource Category'
values = 'Donation Amount'
columns = 'Project Subject Category Tree'
xAxisOrder = None
orderBy = 'Project Subject Category Tree'

chart = stacked_chart_data(df, index, values, columns, orderBy, xAxisOrder)
stacked_bar(chart, index+' & '+columns)
df['date'] = df['Donation Received Date'].dt.date
chartData = df.groupby('date')['Donation Amount'].sum().to_frame(name = 'donation on date').reset_index()
chartData['cumulative donation'] = chartData['donation on date'].cumsum()
index = 'date'
values = 'Donation Amount'
columns = 'Project Resource Category'
xAxisOrder = None
orderBy = 'Project Resource Category'

chart = stacked_chart_data(df, index, values, columns, orderBy, xAxisOrder)
stacked_bar(chart, index)
trace_1 = go.Scatter(
    x=chartData['date'],
    y=chartData['donation on date'],
    name = "donation on date",
    line = dict(color = '#17BECF'),
    opacity = 0.8)

trace_2 = go.Scatter(
    x=chartData['date'],
    y=chartData['cumulative donation'],
    name = "cumulative donations",
    line = dict(color = '#D97CBB'),
    opacity = 0.8)

data = [trace_1,trace_2]

layout = dict(
    title='Donation through time',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=12,
                     label='1y',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = dict(data=data, layout=layout)

py.offline.iplot(fig, filename = "Time Series with Rangeslider")
# Run K-means model on donor FeatureMatrix 
k_means = cluster.KMeans(n_clusters=5)

# Group using project categories
colsInclude = list(donorFeatureMatrix.loc[:,'ProjCat_Applied Learning': 'ProjCat_Warmth, Care & Hunger'].columns)
result = k_means.fit(donorFeatureMatrix[colsInclude])

# Get the k-means grouping label
clusterLabel = result.labels_

pd.DataFrame(clusterLabel)[0].value_counts(normalize=True)
def plot_cluster_traits(donorFeatureMatrix, col_category, clusterLabel):
    '''
    col_category are the filters for the column names in the donorFeatureMatrix
    values could be: 
    'Project Type', 'School Metro Type', 'Project Grade Level Category',
    'Project Resource Category', 'lunchAid', 'ProjCat', 'Dist', 'Percentile'
    
    clusterLabel is labels from the output of k-means
    '''
    
    # get columns to chart
    chart = donorFeatureMatrix.filter(regex='^'+col_category, axis=1).copy()
    chart['label'] = clusterLabel
    
    # for each column, get mean of each cluster
    chart = chart.groupby(['label']).mean().reset_index()
    chart_melt = pd.melt(chart, id_vars = ['label'], value_vars = chart.columns[1:], var_name='category', value_name = 'mean')
    chart_melt['color'] = np.where(chart_melt['mean']<0, 'orange', 'pink')
    chart_melt = chart_melt.sort_values(by = ['label', 'category']).reset_index(drop = True)
    
    # delete the col_category from column names for the chart
    chart_melt['category'] = chart_melt['category'].str.replace(col_category+'_','')
    
    # plot chart using Seaborn
    if chart_melt['category'].nunique()>8:
        g = sns.FacetGrid(chart_melt, row = 'label', size=1.5, aspect=8, sharex=True, sharey=True)  # size: height, # aspect * size gives the width
        g.map(sns.barplot, 'category', 'mean', palette="Set1")
        g.set_xticklabels(rotation=90)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Cluster Preferences- ' + col_category)
    else:
        g = sns.FacetGrid(chart_melt, row = 'label', size=1.5, aspect=4)
        g.map(sns.barplot, 'category', 'mean', palette="Set2")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Cluster Preferences- ' + col_category)
    return g

def plot_cluster_traits_plotly(donorFeatureMatrix, col_category, clusterLabel):
    '''
    col_category are the filters for the column names in the donorFeatureMatrix
    values could be: 
    'Project Type', 'School Metro Type', 'Project Grade Level',
    'Project Resource Category', 'lunchAid', 'ProjCat', 'Dist', 'percentile'
    
    clusterLabel is labels from the output of k-means
    '''
    
    # append cluster labeling to donorFeatureMatrix
    donorFeatureMatrix['label'] = clusterLabel
    
    # get columns to chart
    cols = [col for col in donorFeatureMatrix.columns if  col_category in col]
    cols.append('label')
    
    # for each column, get mean of each cluster
    chart = donorFeatureMatrix[cols].groupby(['label']).mean().reset_index()
    chart_melt = pd.melt(chart, id_vars = ['label'], value_vars = chart.columns[1:], var_name='category', value_name = 'mean')
    
    # delete the col_category from column names for the chart
    chart_melt['category'] = chart_melt['category'].str.replace(col_category+'_','')
    
    # make chart
    fig = ff.create_facet_grid(chart_melt, 
                  x='category', 
                  y='mean', 
                  facet_row='label', 
                  facet_col=None,
                  color_name=None, 
                  colormap=None, 
                  color_is_cat=False,
                  facet_row_labels=None, 
                  facet_col_labels=None,
                  height=700,
                  width=1000, 
                  trace_type='bar', #  The options are 'scatter', 'scattergl', 'histogram','bar', and 'box'.
                  scales='fixed', 
                  dtick_x=None, 
                  dtick_y=[0.10],
                  show_boxes=True, 
                  ggplot2=False, 
                  binsize=1)

    fig['layout'].update(title='Interest Level by K-Means Grouping')
    
    return py.offline.iplot(fig, validate=False, filename='kmeans groups')
plot_cluster_traits(donorFeatureMatrixNoAdj, 'ProjCat', clusterLabel)
plot_cluster_traits(donorFeatureMatrix, 'ProjCat', clusterLabel)
# Run K-means model on donor FeatureMatrix 
k_means = cluster.KMeans(n_clusters=5)

# Group using project categories
colsInclude = list(donorFeatureMatrix.loc[:,'Project Type_Professional Development': 'Percentile_Dist_max'].columns)
result = k_means.fit(donorFeatureMatrix[colsInclude])

# Get the k-means grouping label
clusterLabel_1 = result.labels_

plot_cluster_traits(donorFeatureMatrix, 'ProjCat', clusterLabel_1)
plot_cluster_traits(donorFeatureMatrix, 'Project Grade Level Category', clusterLabel_1)
plot_cluster_traits(donorFeatureMatrixNoAdj, 'School Metro Type', clusterLabel_1)
plot_cluster_traits(donorFeatureMatrixNoAdj, 'Project Grade Level Category', clusterLabel_1)
plot_cluster_traits(donorFeatureMatrix, 'Project Resource Category', clusterLabel_1)
plot_cluster_traits(donorFeatureMatrix, 'lunchAid', clusterLabel_1)
plot_cluster_traits(donorFeatureMatrix, 'Dist', clusterLabel_1)
plot_cluster_traits(donorFeatureMatrix, 'Percentile', clusterLabel_1)
chart = donorFeatureMatrixNoAdj.filter(regex=r'Concentration|ProjCat')
chart['label'] = clusterLabel_1
chart = chart.groupby(['label']).mean().reset_index()
chart_melt = pd.melt(chart, id_vars = ['label'], value_vars = chart.columns[1:], var_name='category', value_name = 'mean')
chart_melt = chart_melt.sort_values(by = ['label', 'category']).reset_index(drop = True)

g = sns.FacetGrid(chart_melt, row = 'label', size=1.5, aspect=8)  # size: height, # aspect * size gives the width
g.map(sns.barplot, 'category', 'mean', palette="Set1")
g.set_xticklabels(rotation=90)
g.fig.subplots_adjust(top=0.9)

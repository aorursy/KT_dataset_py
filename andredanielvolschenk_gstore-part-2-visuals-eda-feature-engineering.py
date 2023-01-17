import numpy as np
import pandas as pd
import statistics as ss

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot') # if error, use plt.style.use('ggplot') instead

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

import datetime as dt

import plotly as ply
import plotly.offline as plyo
plyo.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import seaborn as sns


import json
import pandas.io.json as pdjson
import ast

import gc
gc.enable()

import os
print(os.listdir("../input"))
json_vars = ['device', 'geoNetwork', 'totals', 'trafficSource', 'hits', 'customDimensions']

final_vars = ['channelGrouping','customDimensions_index','customDimensions_value','date',
'device_browser','device_deviceCategory','device_isMobile','device_operatingSystem',
'fullVisitorId','geoNetwork_city','geoNetwork_continent','geoNetwork_country',
'geoNetwork_metro','geoNetwork_networkDomain','geoNetwork_region','geoNetwork_subContinent',
'hits_appInfo.exitScreenName','hits_appInfo.landingScreenName','hits_appInfo.screenDepth',
'hits_appInfo.screenName','hits_contentGroup.contentGroup1','hits_contentGroup.contentGroup2',
'hits_contentGroup.contentGroup3','hits_contentGroup.contentGroup4','hits_contentGroup.contentGroup5',
'hits_contentGroup.contentGroupUniqueViews1','hits_contentGroup.contentGroupUniqueViews2',
'hits_contentGroup.contentGroupUniqueViews3','hits_contentGroup.previousContentGroup1',
'hits_contentGroup.previousContentGroup2','hits_contentGroup.previousContentGroup3',
'hits_contentGroup.previousContentGroup4','hits_contentGroup.previousContentGroup5',
'hits_customDimensions','hits_customMetrics','hits_customVariables','hits_dataSource',
'hits_eCommerceAction.action_type','hits_eCommerceAction.option','hits_eCommerceAction.step',
'hits_eventInfo.eventAction','hits_eventInfo.eventCategory','hits_eventInfo.eventLabel',
'hits_exceptionInfo.isFatal','hits_experiment','hits_hitNumber','hits_hour','hits_isEntrance',
'hits_isExit','hits_isInteraction','hits_item.currencyCode','hits_item.transactionId',
'hits_latencyTracking.domContentLoadedTime','hits_latencyTracking.domInteractiveTime',
'hits_latencyTracking.domLatencyMetricsSample','hits_latencyTracking.domainLookupTime',
'hits_latencyTracking.pageDownloadTime','hits_latencyTracking.pageLoadSample',
'hits_latencyTracking.pageLoadTime','hits_latencyTracking.redirectionTime',
'hits_latencyTracking.serverConnectionTime','hits_latencyTracking.serverResponseTime',
'hits_latencyTracking.speedMetricsSample','hits_minute','hits_page.hostname','hits_page.pagePath',
'hits_page.pagePathLevel1','hits_page.pagePathLevel2','hits_page.pagePathLevel3',
'hits_page.pagePathLevel4','hits_page.pageTitle','hits_page.searchCategory','hits_page.searchKeyword',
'hits_promotionActionInfo.promoIsClick','hits_promotionActionInfo.promoIsView','hits_publisher_infos',
'hits_referer','hits_social.hasSocialSourceReferral','hits_social.socialInteractionNetworkAction',
'hits_social.socialNetwork','hits_time','hits_transaction.affiliation','hits_transaction.currencyCode',
'hits_transaction.localTransactionRevenue','hits_transaction.localTransactionShipping',
'hits_transaction.localTransactionTax','hits_transaction.transactionId',
'hits_transaction.transactionRevenue','hits_transaction.transactionShipping',
'hits_transaction.transactionTax','hits_type','totals_bounces','totals_hits','totals_newVisits',
'totals_pageviews','totals_sessionQualityDim','totals_timeOnSite','totals_totalTransactionRevenue',
'totals_transactionRevenue','totals_transactions','trafficSource_adContent',
'trafficSource_adwordsClickInfo.adNetworkType','trafficSource_adwordsClickInfo.gclId',
'trafficSource_adwordsClickInfo.isVideoAd','trafficSource_adwordsClickInfo.page',
'trafficSource_adwordsClickInfo.slot','trafficSource_campaign','trafficSource_isTrueDirect',
'trafficSource_keyword','trafficSource_medium','trafficSource_referralPath','trafficSource_source',
'visitId','visitNumber','visitStartTime']

print('created json_var and final_var')
# lets append json_vars with final_vars, because we still need to import the json vars before expanding them
all_vars  = json_vars + final_vars # the master list of columns to import

def load_df(csv_path, usecols=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    
    dfs = pd.read_csv(csv_path, sep=',',
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                    chunksize = 100000,   # 100000
                     nrows=500000  # TODO: take out
                     )
    
    for df in dfs:
        df.reset_index(drop = True,inplace = True)
        
        device_list=df['device'].tolist()
        #deleting unwanted columns before normalizing
        for device in device_list:
            del device['browserVersion'],device['browserSize'],device['flashVersion'],device['mobileInputSelector'],device['operatingSystemVersion'],device['screenResolution'],device['screenColors']
        df['device']=pd.Series(device_list)
        
        geoNetwork_list=df['geoNetwork'].tolist()
        for network in geoNetwork_list:
            del network['latitude'],network['longitude'],network['networkLocation'],network['cityId']
        df['geoNetwork']=pd.Series(geoNetwork_list)
        
        df['hits']=df['hits'].apply(ast.literal_eval)
        df['hits']=df['hits'].str[0]
        df['hits']=df['hits'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    
        df['customDimensions']=df['customDimensions'].apply(ast.literal_eval)
        df['customDimensions']=df['customDimensions'].str[0]
        df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    
        JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource','hits','customDimensions']

        for column in JSON_COLUMNS:
            column_as_df = pdjson.json_normalize(df[column])
            column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
        
        print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        
        # we wont see each and every column in each chunk that we load, so we need to find where our master list intersects with the actual data
        usecols = set(usecols).intersection(df.columns)
        usecols = list(usecols)
        use_df = df[usecols]
        del df
        gc.collect()
        ans = pd.concat([ans, use_df], axis = 0).reset_index(drop = True)
        print('Stored shape:', ans.shape)
        
    return ans



data1 = load_df('../input/train_v2.csv', usecols=final_vars)
#data2 = load_df("../input/test_v2.csv", usecols=final_vars)

print('data1 shape: ', data1.shape)
#print('data2 shape: ', data2.shape)

print("data1 loaded")
#print("data2 loaded")
data1['totals_transactionRevenue'].fillna(0, inplace=True)
data1['totals_transactionRevenue'] = data1['totals_transactionRevenue'].astype('float')/1000000

data1[['totals_transactionRevenue']].head()
def bar_plots(cstr, data, n_bars=None):
    
    
    if n_bars:
        counts=data[cstr].value_counts()
        counts = counts.iloc[:n_bars]
        data = data[data[cstr].isin(counts.index)]
    #
    
    
    df = data.copy()
    df[cstr] = df[cstr].astype('str')
    
    fig = plt.figure(figsize=(14,10))   # define (new) plot area
    ax = fig.gca()   # define axis
    plt.suptitle(str('Plots for '+cstr))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # FIGURE 1
    ax = fig.add_subplot(221)
    ax.set_xlabel('Count of instances')   # Set text for the x axis
    counts = df[cstr].value_counts() # This is important to get the full nr of instances
    counts.plot.barh(ax = ax)   # Use the plot.bar method on the counts data frame
    
    for i, v in enumerate( counts ):  # give each 3 decimal points
        d=''
        if v>1000000000:
            v=v/1000000000
            d='B'
        elif v>1000000:
            v=v/1000000
            d='M'
        elif v>1000:
            v=v/1000
            d='k'
        ax.text(v + 0.5, i + .25, str(round(v,3))+d, color='black', fontweight='bold')
    
    
    # FIGURE 2
    ax = fig.add_subplot(222)
    ax.set_xlabel('Count of purchases')   # Set text for the x axis
    counts[:] = 0  # set all to 0
    counts2=df[cstr].where(df.totals_transactionRevenue>0).value_counts()
    counts.update(counts2)
    counts.plot.barh(ax = ax)   # Use the plot.bar method on the counts data frame
    
    for i, v in enumerate( counts ):  # give each 3 decimal points
        d=''
        if v>1000000000:
            v=v/1000000000
            d='B'
        elif v>1000000:
            v=v/1000000
            d='M'
        elif v>1000:
            v=v/1000
            d='k'
        ax.text(v + 0.5, i + .25, str(round(v,3))+d, color='black', fontweight='bold')
    
    
    # FIGURE 3
    ax = fig.add_subplot(223)
    ax.set_xlabel('Mean of purchases')   # Set text for the x axis
    counts[:] = 0  # set all to 0
    counts2 = df[df.totals_transactionRevenue > 0].groupby(cstr)['totals_transactionRevenue'].agg(['mean'])
    idx=counts2.index
    counts2=pd.Series(counts2.values.reshape(-1,))
    counts2.index = idx
    counts.update(counts2)
    counts = counts.astype('int64')
    counts.plot.barh(ax = ax)   # Use the plot.bar method on the counts data frame
    
    for i, v in enumerate( counts ):  # give each 3 decimal points
        d=''
        if v>1000000000:
            v=v/1000000000
            d='B'
        elif v>1000000:
            v=v/1000000
            d='M'
        elif v>1000:
            v=v/1000
            d='k'
        ax.text(v + 0.5, i + .25, str(round(v,3))+d, color='black', fontweight='bold')
    
    
    # FIGURE 4
    ax = fig.add_subplot(224)
    ax.set_xlabel('Sum of purchases')   # Set text for the x axis
    counts[:] = 0  # set all to 0
    counts2 = df.groupby(cstr)['totals_transactionRevenue'].agg(['sum'])
    idx=counts2.index
    counts2=pd.Series(counts2.values.reshape(-1,))
    counts2.index = idx
    counts.update(counts2)
    counts = counts.astype('int64')
    counts.plot.barh(ax = ax)   # Use the plot.bar method on the counts data frame
    
    for i, v in enumerate( counts ):  # give each 3 decimal points
        d=''
        if v>1000000000:
            v=v/1000000000
            d='B'
        elif v>1000000:
            v=v/1000000
            d='M'
        elif v>1000:
            v=v/1000
            d='k'
        ax.text(v + 0.5, i + .25, str(round(v,3))+d, color='black', fontweight='bold')
#

print("'bar_plots' function declared")
def scatter_plots(train_df, test_df):
    
    
    def scatter_plot(cnt_srs, color):
        trace = go.Scatter(
            x=cnt_srs.index[::-1],
            y=cnt_srs.values[::-1],
            showlegend=False,
            marker=dict(
                color=color,
            ),
        )
        return trace
    
    train_df['date'] = train_df['date'].apply(lambda x: dt.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    cnt_srs = train_df.groupby('date')['totals_transactionRevenue'].agg(['size', 'count', 'mean', 'sum'])
    cnt_srs.columns = ["count", "count of non-zero revenue", "mean", "sum"]
    cnt_srs = cnt_srs.sort_index()
    #cnt_srs.index = cnt_srs.index.astype('str')
    trace1 = scatter_plot(cnt_srs["count"], 'blue')
    trace2 = scatter_plot(cnt_srs["count of non-zero revenue"], 'blue')
    trace3 = scatter_plot(cnt_srs["mean"], 'blue')
    trace4 = scatter_plot(cnt_srs["sum"], 'blue')
    
    fig = ply.tools.make_subplots(rows=4, cols=1, vertical_spacing=0.08,
                              subplot_titles=["Date - Count", "Date - Non-zero Revenue count",
                                              "Date - Mean of purchases", "Date - Sum of purchases"])
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 3, 1)
    fig.append_trace(trace4, 4, 1)
    fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
    plyo.iplot(fig, filename='date-plots')
    
    
    # test set
    test_df['date'] = test_df['date'].apply(lambda x: dt.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    cnt_srs = test_df.groupby('date')['fullVisitorId'].size()
    
    
    trace = scatter_plot(cnt_srs, 'red')
    
    layout = go.Layout(
        height=400,
        width=800,
        paper_bgcolor='rgb(233,233,233)',
        title='Dates in Test set'
    )
    
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    plyo.iplot(fig, filename="ActivationDate")
    
    
#
print("'scatter_plots' function declared")
def kde_scatter_plots(cstr, train):
    
    # Figure 1
    plt.figure(figsize=(15,5))
    plt.title(str(cstr)+" distribution")
    ax1 = sns.kdeplot(train[cstr].astype('float64'), color="#006633", shade=True)
    
    # Figure 2
    plt.figure(figsize=(15,5))
    plt.title(str(cstr)+" distribution")
    ax2 = sns.kdeplot(train[np.isnan(train['totals_transactionRevenue'])][cstr].astype('float64'),
                      label='No revenue', color="#0000ff")
    ax2 = sns.kdeplot(train[train['totals_transactionRevenue'] >0][cstr].astype('float64'),
                      label='Has revenue', color="#ff6600")
    
    # Figure 3
    temp = train.groupby(cstr, as_index=False)['totals_transactionRevenue'].mean()
    
    plt.figure(figsize=(15,5))
    plt.title(str(cstr)+" distribution")
    ax3 = sns.scatterplot(x=temp[cstr], y=temp.totals_transactionRevenue, label='Mean of purchases', color="#ff6600")
    
    # Figure 3
    temp = train.groupby(cstr, as_index=False)['totals_transactionRevenue'].sum()
    
    plt.figure(figsize=(15,5))
    plt.title(str(cstr)+" distribution")
    ax4 = sns.scatterplot(x=temp[cstr], y=temp.totals_transactionRevenue, label='Sum of purchases', color="#ff6600")
    
#
print("'kde_scatter_plots' function declared")
def globe_plot(train):
    
    def globe(tmp, title, var2=None):
        
        if var2 != None:
            locations = tmp.geoNetwork_country
            z = tmp.totals_transactionRevenue
            text = tmp.totals_transactionRevenue
        else:
            locations = tmp.index
            z = tmp.values
            text = tmp.values
        
        # plotly globe credits - https://www.kaggle.com/arthurtok/generation-unemployed-interactive-plotly-visuals
        colorscale = [[0, 'rgb(102,194,165)'], [0.005, 'rgb(102,194,165)'], 
                      [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 
                      [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 
                      [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
        
        data = [ dict(
                type = 'choropleth',
                autocolorscale = False,
                colorscale = colorscale,
                showscale = True,
                locations = locations,
                z = z,
                locationmode = 'country names',
                text = text,
                marker = dict(
                    line = dict(color = '#fff', width = 2)) )           ]
        
        layout = dict(
            height=500,
            title = title,
            geo = dict(
                showframe = True,
                showocean = True,
                oceancolor = '#222',
                projection = dict(
                type = 'orthographic',
                    rotation = dict(
                            lon = 60,
                            lat = 10),
                ),
                lonaxis =  dict(
                        showgrid = False,
                        gridcolor = 'rgb(102, 102, 102)'
                    ),
                lataxis = dict(
                        showgrid = False,
                        gridcolor = 'rgb(102, 102, 102)'
                        )
                    ),
                )
        fig = dict(data=data, layout=layout)
        plyo.iplot(fig)
    #
    
    
    
    tmp = train["geoNetwork_country"].value_counts()
    title = 'Visits by Country'
    globe(tmp, title, var2=None)
    
    tmp = train[train.totals_transactionRevenue > 0]["geoNetwork_country"].value_counts()
    title = 'Number of purchases by Country'
    globe(tmp, title, var2=None)
    
    tmp = train.groupby("geoNetwork_country").agg({"totals_transactionRevenue" : "mean"}).reset_index()
    var2='totals_transactionRevenue'
    title = 'Mean Revenue by Countries'
    globe(tmp, title, var2)
    
    tmp = train.groupby("geoNetwork_country").agg({"totals_transactionRevenue" : "sum"}).reset_index()
    var2='totals_transactionRevenue'
    title = 'Sum Revenue by Countries'
    globe(tmp, title, var2)
    
#
print("'globe_plot' function defined")

data1 = data1.reindex_axis(sorted(data1.columns), axis=1)
print('Number of columns:', len(data1.columns))
for col in data1.columns:
    print("'"+col+"',")
print(data1.channelGrouping.unique())
bar_plots('channelGrouping', data1)
print('Number of unique values (incl. nan):', data1.customDimensions_index.nunique(dropna=False))
bar_plots('customDimensions_index', data1)
print('Number of unique values (incl. nan):', data1.customDimensions_value.nunique(dropna=False))

bar_plots('customDimensions_value', data1)
data2 = load_df("../input/test_v2.csv", usecols=['date', 'fullVisitorId'])

scatter_plots(data1, data2)
data1['date'] = data1['date'].apply(lambda x: dt.date(int(str(x)[:4]), int(str(x)[5:7]), int(str(x)[8:])))

#% feature representation
data1.date = pd.to_datetime(data1.date, errors='coerce')

#% feature extraction - time and date features

# Get the month value from date
data1['month'] = data1['date'].dt.month
# Get the week value from date
data1['week'] = data1['date'].dt.week
# Get the weekday value from date
data1['weekday'] = data1['date'].dt.weekday

data1 = data1.drop(labels=['date'], axis=1)
data1.head()
kde_scatter_plots('month', data1)
kde_scatter_plots('week', data1)
kde_scatter_plots('weekday', data1)
print('Number of unique values (incl. nans) is:', data1.device_browser.nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('device_browser', data1, n_bars=10)
bar_plots('device_deviceCategory', data1)
bar_plots('device_isMobile', data1)
del(data1['device_isMobile']) # remove

print('Number of unique values (incl. nans) is:', data1.device_operatingSystem.nunique(dropna=False))
bar_plots('device_operatingSystem', data1)
print("Number of unique visitors in train set : ",data1.fullVisitorId.nunique(), " out of rows : ",data1.shape[0])
print("Number of unique visitors in test set : ",data2.fullVisitorId.nunique(), " out of rows : ",data2.shape[0])
print("Number of common visitors in train and test set : ",len(set(data1.fullVisitorId.unique()).intersection(set(data2.fullVisitorId.unique())) ))

del(data2)
print('Number of unique values (incl. nans) is:', data1.geoNetwork_city.nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('geoNetwork_city', data1, n_bars=25)
def make_countsum(df, dfstr):
    df['totals_hits']=df['totals_hits'].fillna(0).astype('int')
    df['totals_pageviews']=df['totals_pageviews'].fillna(0).astype('int')
    
    df[str(dfstr+'_count')] = df[dfstr]
    df[str(dfstr+'_count')]=df.groupby(dfstr).transform('count')
    
    df[str(dfstr+'_hitssum')] = df.groupby(dfstr)['totals_hits'].transform('sum')
    df[str(dfstr+'_viewssum')] = df.groupby(dfstr)['totals_pageviews'].transform('sum')
    #del(df[dfstr])
    return df

print('make_countsum function created')

data1 = make_countsum(data1, 'geoNetwork_city')

print( 'created geoNetwork_city_count,geoNetwork_city_hitssum,and geoNetwork_city_viewssum features')
bar_plots('geoNetwork_city_count', data1, n_bars=10)
bar_plots('geoNetwork_city_hitssum', data1, n_bars=10)
bar_plots('geoNetwork_city_viewssum', data1, n_bars=10)
bar_plots('geoNetwork_metro', data1, n_bars=20)
bar_plots('geoNetwork_region', data1, 20)
bar_plots('geoNetwork_continent', data1)
bar_plots('geoNetwork_subContinent', data1)
print('Number of unique values (incl. nans) is:', data1.geoNetwork_country.nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('geoNetwork_country', data1, n_bars=20)
globe_plot(data1)
print('Number of unique values (incl. nans) is:', data1.geoNetwork_networkDomain.nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('geoNetwork_networkDomain', data1, n_bars=20)
fig = plt.figure(figsize=(8,6))   # define (new) plot area
ax = fig.gca()   # define axis
plt.suptitle('Boxplot for comcastbusiness.net')
temp = data1[data1.geoNetwork_networkDomain=='comcastbusiness.net'][['totals_transactionRevenue']]
print(temp.describe())
temp.boxplot(ax=ax)
del(temp)
print('Number of unique values (incl. nans) is:', data1['hits_appInfo.exitScreenName'].nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('hits_appInfo.exitScreenName', data1, n_bars=10)
print('Number of unique values (incl. nans) is:', data1['hits_appInfo.landingScreenName'].nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('hits_appInfo.landingScreenName', data1, n_bars=10)
print('Number of unique values (incl. nans) is:', data1['hits_appInfo.screenDepth'].nunique(dropna=False), 'out of', data1.shape[0])
data1['hits_appInfo.screenDepth']=data1['hits_appInfo.screenDepth'].astype('str')
bar_plots('hits_appInfo.screenDepth', data1, n_bars=10)
del(data1['hits_appInfo.screenDepth'])

print('Number of unique values (incl. nans) is:', data1['hits_appInfo.screenName'].nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('hits_appInfo.screenName', data1, n_bars=10)
print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup1'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup1', data1)
del(data1['hits_contentGroup.contentGroup1'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup2'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup2', data1)
print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup3'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup3', data1)
del(data1['hits_contentGroup.contentGroup3'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup4'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup4', data1)
del(data1['hits_contentGroup.contentGroup4'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup5'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup5', data1)
del(data1['hits_contentGroup.contentGroup5'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroupUniqueViews1'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroupUniqueViews1', data1)
del(data1['hits_contentGroup.contentGroupUniqueViews1'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroupUniqueViews2'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroupUniqueViews2', data1)
print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroupUniqueViews3'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroupUniqueViews3', data1)
del(data1['hits_contentGroup.contentGroupUniqueViews3'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup1'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup1', data1)
del(data1['hits_contentGroup.previousContentGroup1'])


print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup2'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup2', data1)
del(data1['hits_contentGroup.previousContentGroup2'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup3'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup3', data1)
del(data1['hits_contentGroup.previousContentGroup3'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup4'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup4', data1)
del(data1['hits_contentGroup.previousContentGroup4'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup5'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup5', data1)
del(data1['hits_contentGroup.previousContentGroup5'])

print('Number of unique values (incl. nan):', data1['hits_customDimensions'].astype('str').nunique(dropna=False)) # convert to str otherwise error
bar_plots('hits_customDimensions', data1)
del(data1['hits_customDimensions'])

print('Number of unique values (incl. nan):', data1['hits_customMetrics'].astype('str').nunique(dropna=False)) # convert to str otherwise error
bar_plots('hits_customMetrics', data1)
del(data1['hits_customMetrics'])

print('Number of unique values (incl. nan):', data1['hits_customVariables'].astype('str').nunique(dropna=False)) # convert to str otherwise error
bar_plots('hits_customVariables', data1)
del(data1['hits_customVariables'])

print('Number of unique values (incl. nan):', data1['hits_dataSource'].nunique(dropna=False))
bar_plots('hits_dataSource', data1)
print('Number of unique values (incl. nan):', data1['hits_eCommerceAction.action_type'].nunique(dropna=False))
bar_plots('hits_eCommerceAction.action_type', data1)
del(data1['hits_eCommerceAction.action_type'])

print('Number of unique values (incl. nan):', data1['hits_eCommerceAction.step'].nunique(dropna=False))
bar_plots('hits_eCommerceAction.step', data1)
del(data1['hits_eCommerceAction.step'])

print('Number of unique values (incl. nan):', data1['hits_eventInfo.eventAction'].nunique(dropna=False))
bar_plots('hits_eventInfo.eventAction', data1)
del(data1['hits_eventInfo.eventAction'])

print('Number of unique values (incl. nan):', data1['hits_eventInfo.eventCategory'].nunique(dropna=False))
bar_plots('hits_eventInfo.eventCategory', data1)
print('Number of unique values (incl. nan):', data1['hits_eventInfo.eventLabel'].nunique(dropna=False))
bar_plots('hits_eventInfo.eventLabel', data1, n_bars=20)
print('Number of unique values (incl. nan):', data1['hits_exceptionInfo.isFatal'].nunique(dropna=False))
bar_plots('hits_exceptionInfo.isFatal', data1)
del(data1['hits_exceptionInfo.isFatal'])

print('Number of unique values (incl. nan):', data1['hits_experiment'].astype('str').nunique(dropna=False))   # convert to 'str' otherwise error
bar_plots('hits_experiment', data1)
del(data1['hits_experiment'])

#print('Number of unique values (incl. nan):', data1['hits_hitNumber'].nunique(dropna=False))
kde_scatter_plots('hits_hitNumber', data1)
print('Unique values (incl. nan):', data1['hits_hour'].unique())
print('Number of nans:', data1.hits_hour.isnull().sum() )
print('The mode is:', ss.mode(data1.hits_hour) )
print("Replace nans with mode")
data1.hits_hour.fillna(ss.mode(data1.hits_hour))
kde_scatter_plots('hits_hour', data1)
print('Number of unique values (incl. nan):', data1['hits_isEntrance'].nunique(dropna=False))
bar_plots('hits_isEntrance', data1)
print('Number of unique values (incl. nan):', data1['hits_isExit'].nunique(dropna=False))
bar_plots('hits_isExit', data1)
print('Number of unique values (incl. nan):', data1['hits_isInteraction'].nunique(dropna=False))
bar_plots('hits_isInteraction', data1)
del(data1['hits_isInteraction'])

print('Number of unique values (incl. nan):', data1['hits_item.currencyCode'].nunique(dropna=False))
bar_plots('hits_item.currencyCode', data1)
kde_scatter_plots('hits_minute', data1)
del(data1['hits_minute'])  # drop immediately

print('Number of unique values (incl. nan):', data1['hits_page.hostname'].nunique(dropna=False))
bar_plots('hits_page.hostname', data1)
print('Number of unique values (incl. nan):', data1['hits_page.pagePath'].nunique(dropna=False))
#bar_plots('hits_page.pagePath', data1)
print('Number of unique values (incl. nan):', data1['hits_page.pagePathLevel1'].nunique(dropna=False))
#bar_plots('hits_page.pagePathLevel1', data1)
print('Number of unique values (incl. nan):', data1['hits_page.pagePathLevel2'].nunique(dropna=False))
#bar_plots('hits_page.pagePathLevel2', data1)
print('Number of unique values (incl. nan):', data1['hits_page.pagePathLevel3'].nunique(dropna=False))
#bar_plots('hits_page.pagePathLevel3', data1)
print('Number of unique values (incl. nan):', data1['hits_page.pagePathLevel4'].nunique(dropna=False))
#bar_plots('hits_page.pagePathLevel4', data1)
print('Number of unique values (incl. nan):', data1['hits_page.pageTitle'].nunique(dropna=False))
#bar_plots('hits_page.pageTitle', data1)
print('Number of unique values (incl. nan):', data1['hits_page.searchCategory'].nunique(dropna=False))
bar_plots('hits_page.searchCategory', data1)
del(data1['hits_page.searchCategory'])

print('Number of unique values (incl. nan):', data1['hits_page.searchKeyword'].nunique(dropna=False))
bar_plots('hits_page.searchKeyword', data1)
del(data1['hits_page.searchKeyword'])

#print('Number of unique values (incl. nan):', data1['hits_promotionActionInfo.promoIsClick'].nunique(dropna=False))
bar_plots('hits_promotionActionInfo.promoIsClick', data1)
del(data1['hits_promotionActionInfo.promoIsClick'])

#print('Number of unique values (incl. nan):', data1['hits_promotionActionInfo.promoIsView'].nunique(dropna=False))
bar_plots('hits_promotionActionInfo.promoIsView', data1)
print('Number of unique values (incl. nan):', data1['hits_publisher_infos'].astype('str').nunique(dropna=False))
bar_plots('hits_publisher_infos', data1)
del(data1['hits_publisher_infos'])

print('Number of unique values (incl. nan):', data1['hits_referer'].nunique(dropna=False))
#bar_plots('hits_referer', data1, n_bars=15)
print('Number of unique values (incl. nan):', data1['hits_social.hasSocialSourceReferral'].nunique(dropna=False))
data1['hits_social.hasSocialSourceReferral']=data1['hits_social.hasSocialSourceReferral'].astype('str')
bar_plots('hits_social.hasSocialSourceReferral', data1, n_bars=15)
print('Number of unique values (incl. nan):', data1['hits_social.socialInteractionNetworkAction'].nunique(dropna=False))
bar_plots('hits_social.socialInteractionNetworkAction', data1)
del(data1['hits_social.socialInteractionNetworkAction'])

print('Number of unique values (incl. nan):', data1['hits_social.socialNetwork'].nunique(dropna=False))
bar_plots('hits_social.socialNetwork', data1)
print('Number of unique values (incl. nan):', data1['hits_time'].nunique(dropna=False))
bar_plots('hits_time', data1)
del(data1['hits_time'])

print('Number of unique values (incl. nan):', data1['hits_transaction.currencyCode'].nunique(dropna=False))
bar_plots('hits_transaction.currencyCode', data1)
print('Number of unique values (incl. nan):', data1['hits_type'].nunique(dropna=False))
bar_plots('hits_type', data1)
print('Number of unique values (incl. nan):', data1['totals_bounces'].nunique(dropna=False))
bar_plots('totals_bounces', data1)
kde_scatter_plots('totals_hits', data1)
bar_plots('totals_newVisits', data1)
kde_scatter_plots('totals_pageviews', data1)
kde_scatter_plots('totals_sessionQualityDim', data1)
kde_scatter_plots('totals_timeOnSite', data1)
data1['totals_transactionRevenue'].describe()
data1["totals_transactionRevenue"] = data1["totals_transactionRevenue"].astype('float')


gdf = data1.groupby("fullVisitorId")["totals_transactionRevenue"].sum().reset_index()    # sum each user's revenue
plt.figure(figsize=(8,6))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals_transactionRevenue"].values)))    # take log of revenue
plt.xlabel('index', fontsize=12)
plt.ylabel('totals_transactionRevenue', fontsize=12)
plt.show()

# NOTE
# Why are using natural log plus 1 (np.log1p) instead of natural log only (np.log) ?  
# Answer: np.log1p is used to deal with log(0). np.log1p(0) = np.log(0+1) = 0
del(data1['totals_totalTransactionRevenue'])
print('totals_totalTransactionRevenue deleted')
#kde_scatter_plots('totals_transactions', data1)
bar_plots('totals_transactions', data1)
print('Number of unique values (incl. nan):', data1['trafficSource_adContent'].nunique(dropna=False))
bar_plots('trafficSource_adContent', data1, n_bars=15)
print('Number of unique values (incl. nan):', data1['trafficSource_adwordsClickInfo.adNetworkType'].nunique(dropna=False))
bar_plots('trafficSource_adwordsClickInfo.adNetworkType', data1)
print('Number of unique values (incl. nan):', data1['trafficSource_adwordsClickInfo.gclId'].nunique(dropna=False))
#bar_plots('trafficSource_adwordsClickInfo.adNetworkType', data1)
print('Number of unique values (incl. nan):', data1['trafficSource_adwordsClickInfo.page'].nunique(dropna=False))
bar_plots('trafficSource_adwordsClickInfo.page', data1)
del(data1['trafficSource_adwordsClickInfo.page'])

print('Number of unique values (incl. nan):', data1['trafficSource_adwordsClickInfo.slot'].nunique(dropna=False))
bar_plots('trafficSource_adwordsClickInfo.slot', data1)
print('Number of unique values (incl. nan):', data1['trafficSource_campaign'].nunique(dropna=False))
bar_plots('trafficSource_campaign', data1, n_bars=15)
del(data1['trafficSource_campaign'])

print('Number of unique values (incl. nan):', data1['trafficSource_isTrueDirect'].nunique(dropna=False))
bar_plots('trafficSource_isTrueDirect', data1)
print('Number of unique values (incl. nan):', data1['trafficSource_keyword'].nunique(dropna=False))
bar_plots('trafficSource_keyword', data1, n_bars=20)
del(data1['trafficSource_keyword'])

print('Number of unique values (incl. nan):', data1['trafficSource_medium'].nunique(dropna=False))
bar_plots('trafficSource_medium', data1)
print('Number of unique values (incl. nan):', data1['trafficSource_referralPath'].nunique(dropna=False))
bar_plots('trafficSource_referralPath', data1, n_bars=20)
print('Number of unique values (incl. nan):', data1['trafficSource_source'].nunique(dropna=False))
bar_plots('trafficSource_source', data1, n_bars=20)
print('For data1:')
print("Number of unique 'fullVisitorId' entries", data1.fullVisitorId.nunique(dropna=False))
print("Number of unique 'visitId' entries", data1.visitId.nunique(dropna=False))
full_Vis = data1.fullVisitorId + data1.visitId.astype('str')
print('Number of unique combinations (out of ',data1.shape[0],' possible):',full_Vis.nunique(dropna=False))
del(full_Vis)
print('Number of unique values (incl. nan):', data1['visitNumber'].nunique(dropna=False))
kde_scatter_plots('visitNumber', data1)
data1['visitStartTime'] = pd.to_datetime(data1['visitStartTime'], unit='s')
data1['hour'] = data1['visitStartTime'].dt.hour
kde_scatter_plots('hour', data1)

data1 = data1.drop(labels=['visitStartTime'], axis=1)
print('Number of columns:', len(data1.columns))
for col in data1.columns:
    print("'"+col+"',")
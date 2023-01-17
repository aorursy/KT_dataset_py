import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Download the data from Google Mobility Site
#https://www.google.com/covid19/mobility/
import pandas as pd
validata=pd.read_csv('/kaggle/input/temp-data-set-mobility/Global_Mobility_Report.csv')

validata.head(10)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
validata.columns.tolist()
validata_aus=validata[validata.country_region == 'Australia']
display_all(validata_aus.head().transpose())
validata_aus.head(10)
import numpy as np
testdata = validata_aus.groupby(['date']).agg(
    {
    "retail_and_recreation_percent_change_from_baseline": [np.median],
    "grocery_and_pharmacy_percent_change_from_baseline": [np.median],
    "parks_percent_change_from_baseline": [np.median],
    "transit_stations_percent_change_from_baseline": [np.median],
    "workplaces_percent_change_from_baseline": [np.median],
    "residential_percent_change_from_baseline": [np.median]
        
}).reset_index()
testdata.columns = ["_".join(x) for x in testdata.columns.ravel()]
d = pd.to_datetime({'year':[2020], 'month':[3], 'day':[22]})
testdata.loc[:,'countofdays'] = (pd.to_datetime(testdata['date_']) - d[0]).dt.days
testdata.dtypes
testdata_1 = testdata.copy()
testdata_1.rename(columns={
       'retail_and_recreation_percent_change_from_baseline_median' :'Retail and Recreation Median Change',
 'grocery_and_pharmacy_percent_change_from_baseline_median':'Grocery and Pharmacy Median Change',
 'parks_percent_change_from_baseline_median':'Parks Median Change',
 'transit_stations_percent_change_from_baseline_median':'Transit Station Median Change',
 'workplaces_percent_change_from_baseline_median':'Workplaces Median Change',
 'residential_percent_change_from_baseline_median':'Residential Median Change',
 'date_': 'date'
}, inplace=True)
import plotly.express as px

import pandas as pd


# fig = px.line(testdata, x='date', y='vals')
# fig.show()

df_long=pd.melt(testdata_1, id_vars=['date'], value_vars=['Retail and Recreation Median Change',
                                                       'Grocery and Pharmacy Median Change',
                                                       'Parks Median Change',
                                                        'Transit Station Median Change',
                                                        'Workplaces Median Change',
                                                        'Residential Median Change'
                                                        
                                                        
                                                       ])
df_long.rename(columns={
    'value':'Median % Change',
    'variable':'areas',
    'date':'Date'
}, inplace=True)
fig = px.line(df_long, x='Date', y='Median % Change', color='areas')
fig.update_layout(shapes=[
    dict(
      type= 'line',
      yref= 'paper', y0= 0, y1= 1,
      xref= 'x', x0= '2020-03-22', x1= '2020-03-22',
      name ='test'
    )
])
# Show plot 
fig.show()
area_map_dict={
    'retail_and_recreation_percent_change_from_baseline_median' :'Retail and Recreation',
 'grocery_and_pharmacy_percent_change_from_baseline_median':'Grocery and Pharmacy',
 'parks_percent_change_from_baseline_median':'Parks',
 'transit_stations_percent_change_from_baseline_median':'Transit Station',
 'workplaces_percent_change_from_baseline_median':'Workplaces',
 'residential_percent_change_from_baseline_median':'Residential',
    
}
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
%matplotlib notebook
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
required_cols = [cols for cols in testdata.columns.tolist() if 'median' in cols]
def create_plot(sample_data,title):
    fig = plt.figure(figsize=(30,6))
    plt.ylim(np.min(sample_data['columnvalue']), np.max(sample_data['columnvalue']))
    plt.xlabel('Year',fontsize=20)
    plt.ylabel('Percentage Change Compared to Median for a Pervious Period',fontsize=10)
    title = '{title} From Google Mobility Analysis report'.format(title=title)
    plt.title(title,fontsize=20)
    ax = plt.axes()
    txt = ax.text(0.1,0.9,'Day to/from lockdown =0', transform=ax.transAxes)
    return plt,txt,fig,ax

def animate(i,dataframe,plt,txt,ax):
    data = dataframe.iloc[:int(i+1)]
    p=sns.lineplot(x=data['date_'], y=data['columnvalue'], data=data, color='red')
    p.tick_params(labelsize=17)
    test=data['countofdays'][i]
    txt.set_text('Day to/from lockdown={:d}'.format(int(test)))
    plt.setp(ax.get_xticklabels(), rotation=80, horizontalalignment='right')
    plt.tight_layout()
    plt.xlabel('Date of Year',fontsize=10)
    plt.ylabel('Median % Changes',fontsize=10)
    return txt
for cols in required_cols:
    try:
        dataframe='{data}_cols'.format(data=cols)
        dataframe = testdata[['date_','countofdays',cols]]
        title = area_map_dict.get(cols)
        dataframe.rename(columns={cols:'columnvalue'}, inplace=True)
        plt,txt,fig,ax=create_plot(dataframe,title)
        ani=matplotlib.animation.FuncAnimation(fig, animate,fargs=[dataframe, plt,txt,ax],interval=100000, repeat=False)
        plt.show(block=True)
        filname ='{cols}_mobility1.mp4'.format(cols=cols)
        #anim.save(filname, writer='imagemagick', fps=5)
        ani.save(filname, writer=writer)
    except KeyError:
        pass
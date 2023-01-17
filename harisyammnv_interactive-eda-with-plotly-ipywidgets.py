import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
%matplotlib inline

import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True,  theme='pearl')
import folium
import altair as alt
import missingno as msg
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from ipywidgets import interact, interactive, fixed
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
donations=pd.read_csv('../input/io/Donations.csv',low_memory=False)
donors=pd.read_csv('../input/io/Donors.csv',low_memory=False)
projects=pd.read_csv('../input/io/Projects.csv',low_memory=False,error_bad_lines=False,warn_bad_lines=False,parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources=pd.read_csv('../input/io/Resources.csv',low_memory=False,error_bad_lines=False,warn_bad_lines=False)
teachers=pd.read_csv('../input/io/Teachers.csv',low_memory=False,error_bad_lines=False)
schools=pd.read_csv('../input/io/Schools.csv',low_memory=False,error_bad_lines=False)
donations.head()
print("The number of records in the Donations.csv file: ",donations.shape)
plt.style.use('ggplot')
msg.bar(donations,figsize=(15,8),fontsize=12);
diod=donations['Donation Included Optional Donation'].value_counts().reset_index()
diod.iplot(kind='pie',labels='index',values='Donation Included Optional Donation',title = 'Whether the Donation is included with Optional Donation?',pull=.05,colors = ["dimgrey","orange"],
           textposition='inside',textinfo='value+percent')
plt.figure(figsize = (16, 8))
donations['Donation_log']=np.log(donations['Donation Amount']+1)

plt.subplot(1,2,1)
plt.style.use('ggplot')
plt.scatter(range(donations.shape[0]), np.sort(donations['Donation Amount'].values))
plt.xlabel('Donation Index')
plt.ylabel('Donation Amount', fontsize=12)
plt.title("Distribution of Donation Amount")

plt.subplot(1,2,2)
sns.distplot(donations['Donation_log'].dropna())
plt.xlabel('Donation Amount', fontsize=12)
plt.title("Histogram of Donation Amount")

plt.show()
donors.head()
msg.bar(donors,figsize=(15,8),fontsize=12);
msg.matrix(donors,figsize=(15,8),fontsize=12);
# Merge donation data with donor data 
donors_donations = donations.merge(donors, on='Donor ID', how='inner')
donors_donations.head()
d_crts=donors_donations.groupby(['Donor Cart Sequence','Donation Included Optional Donation']).size().reset_index(name='counts')
top_15_donor_crt =donors_donations['Donor Cart Sequence'].value_counts()[:15].sort_values(ascending=False).reset_index()
top_15_donor_crt=top_15_donor_crt['index'].values.tolist()

@interact(Select_DonorCart = top_15_donor_crt )
def f(Select_DonorCart):
    tmp=d_crts[d_crts['Donor Cart Sequence']==Select_DonorCart]
    tmp.iplot(kind='pie',labels='Donation Included Optional Donation',values='counts',title = 'Percentage of optional Donation in Donor Cart Sequence: '+str(Select_DonorCart),pull=.05,colors = ["orange","dimgrey"],
           textposition='outside',textinfo='value+percent+label')
do_donations=donors_donations['Donor Cart Sequence'].value_counts()[:15].sort_values(ascending=False).reset_index()
do_donations.rename({'index':'# Cart Sequence'},axis=1,inplace=True)

trace=go.Bar(x=do_donations['# Cart Sequence'],y=do_donations['Donor Cart Sequence'],marker=dict(
        color = do_donations['Donor Cart Sequence'].values,))

cart_donors=donors_donations.groupby(['Donor Cart Sequence','Donor Is Teacher']).size().reset_index(name='counts')
cart_donors=cart_donors[:30]
dt_y=cart_donors[cart_donors['Donor Is Teacher']=='Yes']
dt_n=cart_donors[cart_donors['Donor Is Teacher']=='No']

trace1=go.Bar(x=dt_y['Donor Cart Sequence'],y=dt_y['counts'],name='Yes')
trace2=go.Bar(x=dt_n['Donor Cart Sequence'],y=dt_n['counts'],name='No')

data = [trace,trace1,trace2]

updatemenus = list([
    dict(buttons=list([  
            dict(label = 'None',
                 method = 'update',
                 args = [{'visible': [False, False, False]},{'title': 'Top 15 Donor Cart Sequences',}]),
            dict(label = 'Grouped',
                 method = 'update',
                 args = [{'visible': [False, True, True]},{'title': 'Top 15 Donor Cart Sequences Grouped by Donor being a Teacher or Not',}]),
            dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, False, False]},{'title': 'Top 15 Donor Cart Sequences',}])
        ]),direction = 'down',
        showactive = True, 
    )
])

layout = dict(title = 'Donor Cart Sequences (Select the option in dropdown)',yaxis=dict(title='Count',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),
               xaxis= dict(title= '# Donor Cart Sequence',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),margin=go.Margin(l=50,r=20),paper_bgcolor='rgb(105,105,105)',
               plot_bgcolor='rgb(105,105,105)',barmode='stack',font= {'color': '#FFFFFF'},updatemenus=updatemenus,showlegend=True)


fig = dict(data=data, layout=layout)

py.iplot(fig, filename='relayout_option_dropdown')
donor_count = donors_donations['Donor State'].value_counts().reset_index()
donor_count.columns = ['state', 'counts']
for col in donor_count.columns:
    donor_count[col] = donor_count[col].astype(str)
donor_count['text'] = donor_count['state'] + '<br>' + '# of Donors: ' + donor_count['counts']

state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

donor_count['code'] = donor_count['state'].map(state_codes)  

# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        colorscale = 'Portland',
        locations = donor_count['code'], # The variable identifying state
        z = donor_count['counts'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = donor_count['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "# of Donors")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Number of Donors in different states',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
py.iplot(fig)
do_states=donors_donations['Donor State'].value_counts()[:20].sort_values(ascending=False)
trace=go.Scatter(x=do_states.index.tolist(),y=do_states.values,mode="markers",marker=dict(symbol='hexagram',sizemode = 'diameter',
        sizeref = 1,
        size = 30,
        colorscale='Portland',
        reversescale=True,
        showscale=True,
        color = do_states.values,))
layout = dict(title = 'Top 20 Donor States',yaxis=dict(title='Count',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),
               xaxis= dict(title= 'Donor States',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),margin=go.Margin(l=50,r=30),paper_bgcolor='rgb(105,105,105)',
               plot_bgcolor='rgb(105,105,105)',font= {'color': '#FFFFFF'})
data=[trace]


updatemenus=list([
    dict(
        buttons=list([   
            dict(
                args=[{'marker.symbol':'hexagram'}],
                label='Hexagram',
                method='restyle'
            ),
            dict(
                args=[{'marker.symbol':'Circle'}],
                label='Circle',
                method='restyle'
            ),
            dict(
                args=[{'marker.symbol':'triangle-up'}],
                label='Triangle',
                method='restyle'
            ) 
        ]),
        direction = 'down',
        pad = {'r': 10, 't': 10},
        showactive = True,
        x = 0.01,
        xanchor = 'left',
        y = 1.15,
        yanchor = 'top' 
    ),
])

annotations = list([
    dict(text='Select Marker type:', x=0.1, y=1.2, yref='paper', align='center', showarrow=False)
])
layout['updatemenus'] = updatemenus
layout['annotations'] = annotations

fig=go.Figure(data=data,layout=layout)

py.iplot(fig)
do_city=donors_donations['Donor City'].value_counts()[:20].sort_values(ascending=False)
do_city.iplot(kind='bar',values='Donor City',xTitle='Donor City',yTitle='Frequency',title='Top 20 Donor Cities')
labels1=donors['Donor Is Teacher'].value_counts().index
sizes1=donors['Donor Is Teacher'].value_counts().values

fig = {
  "data": [
    {
      "values": sizes1,
      "labels": labels1,
      "domain": {"x": [0, 1]},
      "name": "",
      "textinfo":'value+percent',
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie",
    },     
    ],
  "layout": {
        "title":"Percentage of Donors being a Teacher",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "Teacher or Not?",
                "x": 0.5,
                "y": 0.5
            },],'paper_bgcolor':'rgb(105, 105, 105)',
              'plot_bgcolor':'rgb(105, 105, 105)','font': {'color': '#FFFFFF'}}}
py.iplot(fig, filename='donut')
teacher_donors=donors_donations.groupby(['Donor State','Donor Is Teacher']).size().reset_index(name='counts')

total_donors_state=donors_donations['Donor State'].value_counts().sort_values(ascending=False).to_frame().reset_index()
total_donors_state.rename({'index':'States','Donor State':'Number of Donations'},axis=1,inplace=True)

dt_y=teacher_donors[teacher_donors['Donor Is Teacher']=='Yes']
dt_n=teacher_donors[teacher_donors['Donor Is Teacher']=='No']

trace1=go.Bar(x=dt_y['Donor State'],y=dt_y['counts'],name='Yes')
trace2=go.Bar(x=dt_n['Donor State'],y=dt_n['counts'],name='No')

data=[trace1,trace2]

layout = dict(title = 'State wise Distribution of Teacher being a Donor or Not?',yaxis=dict(title='Count',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),
               xaxis= dict(title= 'Donor States',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),margin=go.Margin(l=50,r=20,b=170),paper_bgcolor='rgb(105,105,105)',
               plot_bgcolor='rgb(105,105,105)',barmode='stack',font= {'color': '#FFFFFF'})

fig=go.Figure(data=data,layout=layout)
py.iplot(fig)
projects.head(3)
msg.bar(projects,figsize=(15,8),fontsize=12);
trace1= go.Bar(x=projects['Project Type'].value_counts().index, y=projects['Project Type'].value_counts().values,
               name='Project-Type',hoverinfo = 'label+percent',marker=dict(color=projects['Project Type'].value_counts().values))
trace2=go.Bar(x=projects['Project Grade Level Category'].value_counts().index, y=projects['Project Grade Level Category'].value_counts().values,
               name='Grade-Level',hoverinfo = 'label+percent',marker=dict(color=projects['Project Grade Level Category'].value_counts().values))
#data=[trace1]
trace3=go.Bar(x=projects['Project Resource Category'].value_counts().index, y=projects['Project Resource Category'].value_counts().values,
               name='Project Resource Category',hoverinfo = 'label+percent',marker=dict(color=projects['Project Resource Category'].value_counts().values))

layout=dict(title = 'Count-plots for Categorical Data',yaxis1=dict(title='# Count',linecolor='rgba(255,255,255, 0.8)',showgrid=False,gridcolor='rgba(255,255,255,0.2)'),
            margin=go.Margin(l=50,r=30,b=150),paper_bgcolor='rgb(105, 105, 105)',yaxis2=dict(title='# Count',linecolor='rgba(255,255,255, 0.8)',showgrid=False,gridcolor='rgba(255,255,255,0.2)'),
              plot_bgcolor='rgb(105,105,105)',font= {'color': '#FFFFFF'},showlegend=False,)


fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                          subplot_titles=("Project Type Distribuition ",
                                          "Project Grade Level Category Distribuition", 
                                          "Project Resource Category Distribuition"))

#setting the figs
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)


fig['layout'].update(layout,showlegend=False)
py.iplot(fig)
def mosaic_plot(df, dic_color_row, row_labels=None, col_labels=None, alpha_label=None, top_label="Size",
                x_label=None, y_label=None, pad=0.01, color_ylabel=False, ax=None, order="Size",fig_sz=(20,12)):
    """ 

    From a contingency table NxM, plot a mosaic plot with the values inside. There should be a double-index for rows
    e.g.
                                         3   4   1   0   2  5
        Index_1          Index_2                       
        AA               C               0   0   0   2   3  0
                         P               6   0   0  13   0  0
        BB               C               0   2   0   0   0  0
                         P              45   1  10  10   1  0
        CC               C               0   6  35  15  29  0
                         P               1   1   0   2   0  0
        DD               C               0  56   0   3   0  0
                         P              30   4   2   0   1  9

    order: how columns are order, by default, from the biggest to the smallest in term of category. Possible values are 
        - "Size" [default]
        - "Normal" : as the columns are order in the input df
        - list of column names to reorder the column
    top_label: Size of each columns. The label can be changed to adapt to your value. 
               If `False`, nothing is displayed and the secondary legend is set on top instead of on right.  
    """

    is_multi = len(df.index.names) == 2

    if ax == None:
        fig, ax = plt.subplots(1,1, figsize=fig_sz)
        ax.grid(False)

    size_col = df.sum().sort_values(ascending=True)
    prop_com = size_col.div(size_col.sum())
    cols=df.columns.tolist()
    df = pd.DataFrame(df.sort_values(by = cols, ascending = [False,False,False,False]))

    if order == "Size":
        df = df[size_col.index.values]
    elif order == "Normal":
        prop_com = prop_com[df.columns]
        size_col = size_col[df.columns]
    else:
        df = df[order]
        prop_com = prop_com[order]
        size_col = size_col[order]

    if is_multi:
        inner_index = df.index.get_level_values(1).unique()
        prop_ii0 = (df.swaplevel().loc[inner_index[0]]/(df.swaplevel().loc[inner_index[0]]+df.swaplevel().loc[inner_index[1]])).fillna(0)
        alpha_ii = 0.5
        true_y_labels = df.index.levels[0]
    else:
        alpha_ii = 1
        true_y_labels = df.index

    Yt = (df.groupby(level=0).sum().iloc[:,0].div(df.groupby(level=0).sum().iloc[:,0].sum())+pad).cumsum() - pad
    Ytt = df.groupby(level=0).sum().iloc[:,0].div(df.groupby(level=0).sum().iloc[:,0].sum())

    x = 0    
    for j in df.groupby(level=0).sum().iteritems():
        bot = 0
        S = float(j[1].sum())
        for lab, k in j[1].iteritems():
            bars = []
            ax.bar(x, k/S, width=prop_com[j[0]], bottom=bot, color=dic_color_row[lab], alpha=alpha_ii, lw=0, align="edge")
            if is_multi:
                ax.bar(x, k/S, width=prop_com[j[0]]*prop_ii0.loc[lab, j[0]], bottom=bot, color=dic_color_row[lab], lw=0, alpha=1, align="edge")
            bot += k/S + pad
        x += prop_com[j[0]] + pad

    ## Aesthetic of the plot and ticks
    # Y-axis
    if row_labels == None:
        row_labels = Yt.index
    ax.set_yticks(Yt - Ytt/2)
    ax.set_yticklabels(row_labels)

    ax.set_ylim(0, 1 + (len(j[1]) - 1) * pad)
    if y_label == None:
        y_label = df.index.names[0]
    ax.set_ylabel(y_label)

    # X-axis
    if col_labels == None:
        col_labels = prop_com.index
    xticks = (prop_com + pad).cumsum() - pad - prop_com/2.
    ax.set_xticks(xticks)
    ax.set_xticklabels(col_labels)
    ax.set_xlim(0, prop_com.sum() + pad * (len(prop_com)-1))

    if x_label == None:
        x_label = df.columns.name
    ax.set_xlabel(x_label)

    # Top label
    if top_label:
        ax2 = ax.twiny()
        ax2.set_xlim(*ax.get_xlim())
        ax2.set_xticks(xticks) 
        ax2.set_xticklabels(size_col.values.astype(int))
        ax2.set_xlabel(top_label)
        ax2.tick_params(top=False, right=False, pad=0, length=0)

    # Ticks and axis settings

    ax.tick_params(top=False, right=False, pad=5)
    sns.despine(left=0, bottom=False, right=0, top=0, offset=3)

    # Legend
    if is_multi: 
        if alpha_label == None:
            alpha_label = inner_index
        bars = [ax.bar(np.nan, np.nan, color="0.2", alpha=[1, 0.5][b]) for b in range(2)]
        if top_label:
            plt.legend(bars, alpha_label, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, )
        else:
            plt.legend(bars, alpha_label, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    plt.tight_layout(rect=[0, 0, .9, 0.95])
    if color_ylabel:
        for tick, label in zip(ax.get_yticklabels(), true_y_labels):
            tick.set_bbox(dict( pad=5, facecolor=dic_color_row[label]))
            tick.set_color("w")
            tick.set_fontweight("bold")
    
    ax.grid(False)

    return ax
import matplotlib
df_ctb=pd.crosstab(projects['Project Type'],projects['Project Grade Level Category'])
df_ctb.drop('unknown',axis=1,inplace=True)
cmap = matplotlib.cm.Accent
my_values=[i**3 for i in range(1,4)]
mini=min(my_values)
maxi=max(my_values)
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in my_values]

keys=df_ctb.index.tolist()
dictionary = dict(zip(keys, colors))

plt.style.use('ggplot')
mosaic_plot(df_ctb,dictionary,pad=0.005,fig_sz=(15,7),top_label='Mosaic Plot of Project Type & Project Grade Level Category');
df_ctb=pd.crosstab(projects['Project Resource Category'],projects['Project Grade Level Category'])

df_ctb.drop('unknown',axis=1,inplace=True)

cmap = matplotlib.cm.BrBG
my_values=[i**3 for i in range(1,18)]
mini=min(my_values)
maxi=max(my_values)
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in my_values]

keys=df_ctb.index.tolist()
dictionary = dict(zip(keys, colors))

plt.style.use('ggplot')
mosaic_plot(df_ctb,dictionary,pad=0.005,fig_sz=(22,15),top_label='Mosaic Plot of Project Resource Category & Project Grade Level Category');
import squarify

x = 0.
y = 0.
width = 100.
height = 100.

values = projects['Project Resource Category'].value_counts().values
li_values=projects['Project Resource Category'].value_counts().values.tolist()
labels=projects['Project Resource Category'].value_counts().index
normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

# Choose colors from http://colorbrewer2.org/ under "Export"
# add more colors
color_brewer = ['rgb(166,206,227)','rgb(31,120,180)','rgb(178,223,138)',
                'rgb(51,160,44)','rgb(251,154,153)','rgb(227,26,28)','rgb(255,247,251)','rgb(236,231,242)','rgb(236,231,242)','rgb(208,209,230)','rgb(166,189,219)','rgb(116,169,207)','rgb(54,144,192)',
'rgb(5,112,176)','rgb(4,90,141)','rgb(2,56,88)','rgb(255,247,251)','rgb(208,209,230)']
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 2 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = labels[counter]+'<br>'+str(values[counter]),
            showarrow = False,
          font = dict(
          color = "black",
          size = 9
        )
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in labels ],
    mode = '',
)
layout = dict(title='Treemap for Project Resource Category',
    height=700, 
    width=1200,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest'
)
# With hovertext
figure = dict(data=[trace0], layout=layout)
py.iplot(figure, filename='squarify-treemap')
resources.head()
msg.bar(resources,figsize=(15,8),fontsize=12);
rs_vendors=resources['Resource Vendor Name'].value_counts()[:20].sort_values(ascending=False)
trace=go.Scatter(x=rs_vendors.index.tolist(),y=rs_vendors.values,mode="markers",marker=dict(symbol='hexagram',sizemode = 'diameter',
        sizeref = 1,
        size = 30,
        colorscale='Portland',
        reversescale=True,
        showscale=True,
        color = rs_vendors.values,))
layout = dict(title = 'Top 20 Resource Vendors',yaxis=dict(title='Count',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),
               xaxis= dict(title= 'Vendors',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),margin=go.Margin(l=50,r=30,b=150),paper_bgcolor='rgb(105,105,105)',
               plot_bgcolor='rgb(105,105,105)',font= {'color': '#FFFFFF'})
data=[trace]


updatemenus=list([
    dict(
        buttons=list([   
            dict(
                args=[{'marker.symbol':'hexagon-dot'}],
                label='Hexagon-Dot',
                method='restyle'
            ),
            dict(
                args=[{'marker.symbol':'Circle'}],
                label='Circle',
                method='restyle'
            ),
            dict(
                args=[{'marker.symbol':'triangle-up'}],
                label='Triangle',
                method='restyle'
            ) 
        ]),
        direction = 'down',
        pad = {'r': 10, 't': 10},
        showactive = True,
        x = 0.01,
        xanchor = 'left',
        y = 1.15,
        yanchor = 'top' 
    ),
])

annotations = list([
    dict(text='Select Marker type:', x=0.1, y=1.2, yref='paper', align='center', showarrow=False)
])
layout['updatemenus'] = updatemenus
layout['annotations'] = annotations

fig=go.Figure(data=data,layout=layout)

py.iplot(fig)
teachers.head()
msg.bar(teachers,figsize=(15,8),fontsize=12);
teacher_qualifications=teachers['Teacher Prefix'].value_counts().reset_index()
fig = {
  "data": [
    {
      "values": teacher_qualifications["Teacher Prefix"],
      "labels": teacher_qualifications["index"],
      "domain": {"x": [0, 1]},
      "name": "Teachers Prefix",
      "hoverinfo":"label+value+name",
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of Teachers Prefix",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "Teachers Prefix",
                "x": 0.5,
                "y": 0.5
            }
            
        ],'paper_bgcolor':'rgb(105, 105, 105)',
              'plot_bgcolor':'rgb(105, 105, 105)','font': {'color': '#FFFFFF'}
    }
}
py.iplot(fig, filename='donut')
schools.head()
msg.bar(schools,figsize=(15,8),fontsize=12);
school_metro=schools['School Metro Type'].value_counts().reset_index()
fig = {
  "data": [
    {
      "values": school_metro["School Metro Type"],
      "labels": school_metro["index"],
      "domain": {"x": [0, 1]},
      "name": "Teachers Prefix",
      "hoverinfo":"label+value+name",
      "hole": .6,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of School Metro Type",
        "annotations": [
            {
                "font": {
                    "size": 15
                },
                "showarrow": False,
                "text": "School Metro Type",
                "x": 0.5,
                "y": 0.5
            }
            
        ],'paper_bgcolor':'rgb(105, 105, 105)',
              'plot_bgcolor':'rgb(105, 105, 105)','font': {'color': '#FFFFFF'}
    }
}
py.iplot(fig, filename='donut')
school_count = schools['School State'].value_counts().reset_index()
school_count.columns = ['state', 'schools']
for col in school_count.columns:
    school_count[col] = school_count[col].astype(str)
school_count['text'] = school_count['state'] + '<br>' + '# of schools: ' + school_count['schools']

state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

school_count['code'] = school_count['state'].map(state_codes)  

# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        colorscale = 'Portland',
        locations = school_count['code'], # The variable identifying state
        z = school_count['schools'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_count['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "# of Schools")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Number of schools in different states',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
py.iplot(fig)
schools['School County']=schools['School County'].str.replace('De Soto','DeSoto')
schools['School County']=schools['School County'].str.replace('St Lucie','St. Lucie')
schools['School County']=schools['School County'].str.replace('St Johns','St. Johns')
import plotly.figure_factory as ff

df_sample = pd.read_csv('../input/countydata/minoritymajority.csv')
df_sample_r = df_sample[df_sample['STNAME'] == 'Florida']

df_sample_r['School County'] = df_sample_r['CTYNAME'].str[:-7]
florida_schools=schools[schools['School State']=='Florida']
florida_schools_by_county=florida_schools.groupby(['School County']).size().reset_index(name='counts')

df_sample_r.reset_index(inplace=True)

df_sample_r=df_sample_r.merge(florida_schools_by_county,on=['School County'],how='left')

values = df_sample_r['counts'].tolist()
fips = df_sample_r['FIPS'].tolist()

endpts = list(np.mgrid[min(values):max(values):9j])
colorscale = ["#030512","#1d1d3b","#323268","#3d4b94","#3e6ab0",
              "#4989bc","#60a7c7","#85c5d3","#b7e0e4","#eafcfd"]
fig = ff.create_choropleth(width=1000,height=500,
    fips=fips, values=values, scope=['Florida'], show_state_data=True,
    colorscale=colorscale, binning_endpoints=endpts, round_legend_values=True,
    plot_bgcolor='rgb(229,229,229)',
    paper_bgcolor='rgb(229,229,229)',
    legend_title='Number of Schools by County',
    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
    exponent_format=True,
)
py.iplot(fig, filename='choropleth_florida')
df_sample_r = df_sample[df_sample['STNAME'] == 'California']

df_sample_r['School County'] = df_sample_r['CTYNAME'].str[:-7]
calif_schools=schools[schools['School State']=='California']
calif_schools_by_county=calif_schools.groupby(['School County']).size().reset_index(name='counts')

df_sample_r.reset_index(inplace=True)

df_sample_r=df_sample_r.merge(calif_schools_by_county,on=['School County'],how='left')
df_sample_r['counts'].fillna(0.0,inplace=True)
values = df_sample_r['counts'].tolist()
fips = df_sample_r['FIPS'].tolist()

#endpts = list(np.mgrid[min(values):max(values):4j])
#colorscale = ["#030512","#1d1d3b","#323268","#3d4b94","#3e6ab0",
#              "#4989bc","#60a7c7","#85c5d3","#b7e0e4","#eafcfd"]
fig = ff.create_choropleth(width=1000,height=500,
    fips=fips, values=values, scope=['California'], show_state_data=True, round_legend_values=True,hovermode='closest',
    plot_bgcolor='rgb(229,229,229)',
    paper_bgcolor='rgb(229,229,229)',
    legend_title='Number of Schools by County in California',
    county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},
    exponent_format=True,
)
py.iplot(fig, filename='choropleth_california')
projects_schools = projects.merge(schools, on='School ID', how='inner')
projects_schools.head()
stats=projects_schools['Project Current Status'].value_counts().reset_index()
stats.iplot(kind='pie',labels='index',values='Project Current Status',title = 'Total Status of the Projects in the Database',pull=.05,colors = ["blue","dimgrey","green"],
           textposition='outside',textinfo='value+percent+label')
prj_sls=projects_schools['School Name'].value_counts()[:15].sort_values(ascending=False).reset_index()
trace=go.Bar(x=prj_sls['index'],y=prj_sls['School Name'],marker=dict(
        color = prj_sls['School Name'].values,))
data = [trace]

layout = dict(title = 'Top 15 Schools with most projects',yaxis=dict(title='Total Number of Projects',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),
               xaxis= dict(title= 'School Name',linecolor='rgba(255,255,255, 0.8)',showgrid=True,gridcolor='rgba(255,255,255,0.2)'),margin=go.Margin(l=70,r=50,b=150),paper_bgcolor='rgb(105,105,105)',
               plot_bgcolor='rgb(105,105,105)',barmode='stack',font= {'color': '#FFFFFF'},showlegend=False)
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='relayout_option_dropdown')
ind_prj_sls=projects_schools.groupby(['School Name','Project Current Status']).size().reset_index(name='counts')
from ipywidgets import interact, interactive, fixed
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
top_15_schools = prj_sls['index'].values.tolist()

@interact(Select_School = top_15_schools )
def f(Select_School):
    tmp=ind_prj_sls[ind_prj_sls['School Name']==Select_School]
    tmp.iplot(kind='pie',labels='Project Current Status',values='counts',title = 'Status of the Projects in: '+Select_School,pull=.05,colors = ["blue","dimgrey","green"],
           textposition='outside',textinfo='value+percent+label')

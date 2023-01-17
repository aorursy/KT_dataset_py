#'''Importing Data Manipulation Modules'''
import numpy as np                 # Linear Algebra
import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)

#'''Seaborn and Matplotlib Visualization'''
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
import seaborn as sns              # Python Data Visualization Library based on matplotlib
import geopandas as gpd            # Python Geospatial Data Library
plt.style.use('fivethirtyeight')
%matplotlib inline

#'''Plotly Visualizations'''
import plotly as plotly                # Interactive Graphing Library for Python
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

#'''Spatial Visualizations'''
import folium
import folium.plugins

#'''NLP - WordCloud'''
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from plotly import tools

from PIL import Image
# reading the dataset
df = pd.read_csv('../input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv')
print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())
df.isnull().sum()
# filling the missing value in PdDistrict using the mode function
df['PdDistrict'].fillna(df['PdDistrict'].mode()[0],inplace = True)
df.head()
df['Category'].value_counts()
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('seaborn')
sns.countplot(x = 'Category', data = df, order = df['Category'].value_counts().index)
plt.title('Crime Categories in Sanfrancisco', fontweight = 40, fontsize = 20)
plt.xticks(rotation = 90)
plt.show()

df['PdDistrict'].value_counts()
xSOUTHERN = df[df['PdDistrict'] == 'SOUTHERN']
xNORTHERN = df[df['PdDistrict'] == 'NORTHERN']
xMISSION = df[df['PdDistrict'] == 'MISSION']
xCENTRAL = df[df['PdDistrict'] == 'CENTRAL']
xBAYVIEW = df[df['PdDistrict'] == 'BAYVIEW']
xINGLESIDE = df[df['PdDistrict'] == 'INGLESIDE']
xTARAVAL  = df[df['PdDistrict'] == 'TARAVAL ']
xTENDERLOIN = df[df['PdDistrict'] == 'TENDERLOIN']
xRICHMOND = df[df['PdDistrict'] == 'RICHMOND']
xPARK = df[df['PdDistrict'] == 'PARK']
trace1 = go.Histogram(
    x=xSOUTHERN['Category'],
    opacity=0.75,
    name = "SOUTHERN",
    marker=dict(color='rgb(165,0,38)'))
trace2 = go.Histogram(
    x=xNORTHERN['Category'],
    opacity=0.75,
    name = "NORTHERN",
    marker=dict(color='rgb(215,48,39)'))
trace3 = go.Histogram(
    x=xMISSION['Category'],
    opacity=0.75,
    name = "MISSION",
    marker=dict(color='rgb(244,109,67)'))
trace4 = go.Histogram(
    x=xCENTRAL['Category'],
    opacity=0.75,
    name = "CENTRAL",
    marker=dict(color='rgb(253,174,97)'))
trace5 = go.Histogram(
    x=xBAYVIEW['Category'],
    opacity=0.75,
    name = "BAYVIEW",
    marker=dict(color='rgb(254,224,144)'))
trace6 = go.Histogram(
    x=xINGLESIDE['Category'],
    opacity=0.75,
    name = "INGLESIDE",
    marker=dict(color='rgb(170,253,87)'))
trace7 = go.Histogram(
    x=xTARAVAL['Category'],
    opacity=0.75,
    name = "TARAVAL",
    marker=dict(color='rgb(171,217,233)'))
trace8 = go.Histogram(
    x=xTENDERLOIN['Category'],
    opacity=0.75,
    name = "TENDERLOIN",
    marker=dict(color='rgb(116,173,209)'))
trace9 = go.Histogram(
    x=xRICHMOND['Category'],
    opacity=0.75,
    name = "RICHMOND",
    marker=dict(color='rgb(69,117,180)'))
trace10 = go.Histogram(
    x=xPARK['Category'],
    opacity=0.75,
    name = "PARK",
    marker=dict(color='rgb(49,54,149)'))

data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10]
layout = go.Layout(barmode='stack',
                   title='District counts according to Crime genres',
                   xaxis=dict(title='Crime genres'),
                   yaxis=dict( title='Count'),
                   paper_bgcolor='beige',
                   plot_bgcolor='beige'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
wave_mask= np.array(Image.open('../input/policemanjpg/policeman-silhouette-silhouette-policeman-white-background-135347935.jpg'))

stopwords = set(STOPWORDS)
stopwords.update(["II", "III"])
plt.subplots(figsize=(15,15))
wordcloud = WordCloud(mask=wave_mask,background_color="lavenderblush",colormap="hsv" ,contour_width=2, contour_color="black",
                      width=950,stopwords=stopwords,
                          height=950
                         ).generate(" ".join(df.Descript))

plt.imshow(wordcloud ,interpolation='bilinear')
plt.axis('off')
plt.savefig('graph.png')

plt.show()
plt.rcParams['figure.figsize'] = (18, 8)
plt.style.use('dark_background')
sns.countplot(x = 'DayOfWeek', data = df)
plt.title('Crimes in each day of the week', fontweight = 40, fontsize = 30)
plt.xticks(rotation = 45)
plt.show()


df['Date'] = pd.to_datetime(df['Date'])
def get_month(x):
    return x.month
df['month'] = df['Date'].apply(get_month)
df['month'].unique()
plt.rcParams['figure.figsize'] = (18,8)

plt.style.use('dark_background')

sns.countplot(x = 'month', data = df)

plt.title('Crimes in each month')

plt.show()
df['Resolution'].value_counts()
none = df[df['Resolution'] == 'NONE']
arrest = df[df['Resolution'] == 'ARREST, BOOKED']
unfounded = df[df['Resolution'] == 'UNFOUNDED']
juvenile = df[df['Resolution'] == 'JUVENILE BOOKED ']
exceptional = df[df['Resolution'] == 'EXCEPTIONAL CLEARANCE']
none_category = pd.DataFrame(none['Category'].value_counts())
arrest_category = pd.DataFrame(arrest['Category'].value_counts())
unfounded_category = pd.DataFrame(unfounded['Category'].value_counts())
juvenile_category = pd.DataFrame(juvenile['Category'].value_counts())
exceptional_category = pd.DataFrame(exceptional['Category'].value_counts())
trace0 = go.Scatter(
    x = none_category.index,
    y = none_category["Category"],
    mode = 'markers',
    name = 'none',
    marker= dict(size= 14,
                    line= dict(width=1),
                    color= "lime",
                    opacity= 0.7
                   )
)
trace1 = go.Scatter(
    x = arrest_category.index,
    y = arrest_category["Category"],
    mode = 'markers',
    name = 'arrest',
       marker= dict(size= 14,
                    line= dict(width=1),
                    color= "firebrick",
                    opacity= 0.7,
                   symbol=220
                   )
)
trace2 = go.Scatter(
    x = unfounded_category.index,
    y = unfounded_category["Category"],
    mode = 'markers',
    name = 'unfounded',
    marker= dict(size= 14,
                    line= dict(width=1),
                    color='rgba(150, 26, 80, 0.8)',
                    opacity= 0.7
                   )
)
trace3 = go.Scatter(
    x = juvenile_category.index,
    y = juvenile_category["Category"],
    mode = 'markers',
    name = 'juvenile',
       marker= dict(size= 14,
                    line= dict(width=1),
                    color= 'rgba(28, 149, 249, 0.8)',
                    opacity= 0.7,
                    symbol=220
                   )
)

trace4 = go.Scatter(
    x = exceptional_category.index,
    y = exceptional_category["Category"],
    mode = 'markers',
    name = 'exceptional',
    marker= dict(size= 14,
                    line= dict(width=1),
                    color= 'rgba(249, 94, 28, 0.8)',
                    opacity= 0.7
                   )
)


fig = tools.make_subplots(rows=2, cols=2, 
                          subplot_titles=('Category','Category','Category'))


fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 1, 2)


fig['layout'].update(showlegend=False,height=800, width=800, title="Resolutions by Categories " ,paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor="moccasin")
iplot(fig)
none_PdDistrict = pd.DataFrame(none['PdDistrict'].value_counts())
arrest_PdDistrict = pd.DataFrame(arrest['PdDistrict'].value_counts())
unfounded_PdDistrict = pd.DataFrame(unfounded['PdDistrict'].value_counts())
juvenile_PdDistrict = pd.DataFrame(juvenile['PdDistrict'].value_counts())
exceptional_PdDistricty = pd.DataFrame(exceptional['PdDistrict'].value_counts())
data8=[go.Scatterpolar(
      r = list(none_PdDistrict["PdDistrict"].values),
      theta = none_PdDistrict.index,
      fill = 'toself',
      name = "none",
    thetaunit = "radians",
    ),
 go.Scatterpolar(
      r = list(arrest_PdDistrict["PdDistrict"].values),
      theta = arrest_PdDistrict.index,
      fill = 'toself',
      name = 'arrest',
    thetaunit = "radians"
    ),
go.Scatterpolar(
      r = unfounded_PdDistrict["PdDistrict"].values,
      theta = unfounded_PdDistrict.index,
      fill = 'toself',
      name = "unfounded",
    thetaunit = "radians",
     subplot = "polar2"
    ),
go.Scatterpolar(
      r = exceptional_PdDistricty["PdDistrict"].values,
      theta =exceptional_PdDistricty.index,
      fill = 'toself',
      name = 'exceptional',
    subplot = "polar2",
    thetaunit = "radians"
    )]
layout = go.Layout(
    showlegend=False,
    paper_bgcolor='moccasin',
    title="Resolutions by Districts",
    font=dict(family='Gravitas One',size=20,color='darkred'),
     
    
    polar = dict(
      bgcolor="linen",
      domain = dict(
        y = [0.60, 0.90],
        x = [0, 0.48]
      ),
      radialaxis = dict(
             visible = False,
        angle = 45
      ),
      angularaxis = dict(
        direction = "clockwise",
        period = 6,
          gridwidth=3,
          tickfont=dict(size=11,color="black"),
      )
    ),
    polar2 = dict(
        bgcolor="linen",
      domain = dict(
        y = [0.60, 0.90],
        x = [0.52, 1]
      ),
      radialaxis = dict(
             visible = False,
        angle = 45
      ),
      angularaxis = dict(
        direction = "clockwise",
        period = 5,
           gridwidth=3,
          tickfont=dict(size=11,color="black"),
      )),
    
     annotations=[dict(showarrow=False,text="District",x=0.18,y=1.05,xref="paper",yref="paper",font=dict(size=15,color="midnightblue"),bgcolor="lightyellow",borderwidth=5),
                                  dict(showarrow=False,text="District",x=0.83,y=1.05,xref="paper",yref="paper",font=dict(size=15,color="midnightblue"),bgcolor="lightyellow",borderwidth=5)
                                  ]
)

fig = go.Figure(data=data8,layout=layout)
iplot(fig)
import folium
from folium.plugins import HeatMap
m=folium.Map([37.7749,-122.4194],zoom_start=11)
HeatMap(df[['Y','X']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)
display(m)
plt.rcParams['figure.figsize'] = (18,8)

plt.style.use('dark_background')

plt.title('Crimes by time',fontsize = 25)

df['Time'].value_counts()[:15].plot.bar()
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly libraries
import plotly
import plotly.express as px
import plotly.graph_objs as go
#import chart_studio.plotly as py

import cufflinks as cf
from plotly.offline import iplot, init_notebook_mode, plot
cf.go_offline()

import warnings
warnings.filterwarnings('ignore')
consumer_data= pd.read_csv('../input/consumercomplaintsdata/Consumer_Complaints.csv')
consumer_data.head(2)
consumer_data.columns = consumer_data.columns.str.title()
mode_value= consumer_data['Consumer Disputed?'].mode()
mode_value ='No'
consumer_data['Consumer Disputed?'].fillna(mode_value, inplace=True)
consumer_data['Consumer Disputed?'].isnull().fillna(mode_value,inplace =True)
consumer_data.isnull().mean().round(4)*100
# getting the sum of null values and ordering.
total = consumer_data.isnull().sum().sort_values(ascending = False)  

#getting the percent and order of null.
percent = (consumer_data.isnull().sum()/consumer_data.isnull().count()*100).sort_values(ascending =False)

# Concatenating the total and percent
df = pd.concat([total , percent],axis =1,keys=['Total' ,'Percent'])

# Returning values of nulls different of 0
(df[~(df['Total'] == 0)])
consumer_data[['Issue','Date Received','Product','Sub-Issue','Consumer Complaint Narrative','Company',
               'Company Public Response','Consumer Consent Provided?',
               'Company Response To Consumer','Submitted Via']].describe().transpose()
sns.set(style='white')
consumer_data['Issue'].str.strip("'").value_counts()[0:15].iplot(kind='bar',title='Top 15 issues',fontsize=14,color='orange')
consumer_data['Sub-Issue'].str.strip("'").value_counts()[0:15].iplot(kind ='bar',
                                                                     title='Top 15 Sub Issues',fontsize=14,color='#9370DB')
consumer_data['Company'].str.strip("'").value_counts()[0:15].iplot(kind='bar',
                                                          title='Top 15 Company',fontsize=14,color='purple')
from datetime import datetime
consumer_data['Date'] =pd.to_datetime(consumer_data['Date Received'])

#Extracting Year.
consumer_data['Year'] =consumer_data['Date'].dt.year

#Extracting Month.
consumer_data['Month'] =consumer_data['Date'].dt.month_name()

#Extracting Weekdays

consumer_data['Week_Days'] = consumer_data['Date'].dt.day_name()

consumer_data.head()
consumer_data['Week_Days'].value_counts().iplot(kind ='barh',title ='Number of Complaints per Weekday')
pd.crosstab(consumer_data['Year'],consumer_data['Month']).iplot(kind='bar',barmode='stack',
                                                        title='Number of Complaints per Month')
grouped = consumer_data.groupby(['Company Response To Consumer']).size()
pie_chart = go.Pie(labels=grouped.index,values=grouped,
                  title='Company Response to the Customer')
iplot([pie_chart])
states = consumer_data['State'].value_counts()

scl = [
    [0.0, 'rgb(242,240,247)'],
    [0.2, 'rgb(218,218,235)'],
    [0.4, 'rgb(188,189,220)'],
    [0.6, 'rgb(158,154,200)'],
    [0.8, 'rgb(117,107,177)'],
    [1.0, 'rgb(84,39,143)']
]

data = [go.Choropleth(
    colorscale = scl,
    autocolorscale = False,
    locations = states.index,
    z = states.values,
    locationmode = 'USA-states',
    text = states.index,
    marker = go.choropleth.Marker(
        line = go.choropleth.marker.Line(
            color = 'rgb(254,254,254)',
            width = 2
        )),
    colorbar = go.choropleth.ColorBar(
        title = "Complaints")
)]

layout = go.Layout(
    title = go.layout.Title(
        text = 'Complaints by State'
    ),
    geo = go.layout.Geo(
        scope = 'usa',
        projection = go.layout.geo.Projection(type = 'albers usa'),
        showlakes = True,
        lakecolor = 'rgb(100,149,237)'),
)

fig = go.Figure(data = data, layout = layout)
iplot(fig)
pd.crosstab(consumer_data['Timely Response?'],consumer_data['Submitted Via']).iplot(kind='bar',
                                                                                    title='Company Response to the Customer')
pd.crosstab(consumer_data['Timely Response?'], consumer_data['Consumer Disputed?']).iplot(kind='bar',
                                                                    title ='Timely Response vs Consumer Disputed' )
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder =OneHotEncoder()
# Label Encoding the Consumer Disputed? column
consumer_data['Consumer_encode']= labelencoder.fit_transform(consumer_data['Consumer Disputed?'])
enc = OneHotEncoder(handle_unknown='ignore')
consumer_data1 = pd.DataFrame(enc.fit_transform(consumer_data[['Product']]).toarray())
df = consumer_data.join(consumer_data1)
x = df.iloc[:,24:41].values
y = df['Consumer_encode'].values
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y, test_size =0.25, random_state =10)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy', random_state =10)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
from sklearn.metrics import accuracy_score
print('Accuracy Score:',accuracy_score(y_test,y_pred))
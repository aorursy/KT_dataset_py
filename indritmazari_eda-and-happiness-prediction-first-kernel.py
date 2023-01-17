
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


%matplotlib inline
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
import os
import matplotlib.pyplot as plt#visualization
from PIL import  Image
%matplotlib inline
import pandas as pd
import seaborn as sns#visualization
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
import matplotlib.pyplot as plt

df_2015=pd.read_csv('../input/2015.csv')
df_2016=pd.read_csv('../input/2016.csv')
df_2017=pd.read_csv('../input/2017.csv')

df_2017.rename(columns={'Happiness.Rank':'Happiness Rank','Happiness.Score':'Happiness Score','Economy..GDP.per.Capita':'Economy (GDP per Capita)','Health..Life.Expectancy.': 'Health (Life Expectancy)','Whisker.high':'Upper Confidence Interval','Whisker.low':'Lower Confidence Interval', 'Trust..Government.Corruption.':'Trust (Government Corruption)','Economy..GDP.per.Capita.':'Economy (GDP per Capita)','Dystopia.Residual':'Dystopia Residual'},inplace=True)

df_2015['year_id']=2015
df_2016['year_id']=2016
df_2017['year_id']=2017

df=pd.concat([df_2015,df_2016,df_2017])
df.groupby('year_id').apply(lambda x:x.isnull().sum())
df.sort_values(by=['Country','year_id'],inplace=True)
df['Happiness_change']=df['Happiness Rank'].diff()

df['Happiness_change']=-df['Happiness_change']

df.loc[df['Country']!=df['Country'].shift(),'Happiness_change']=None

happiness_deltas=df.groupby('Country')['Happiness_change'].mean().reset_index().sort_values(by='Happiness_change',ascending=False)

happiness_deltas['Happiness_change'].describe()

happiness_deltas['Happiness Change Range']=pd.cut(happiness_deltas['Happiness_change'],bins=[-50,-20,-10,-5,-2,0,2,5,10,20,50])


sns.barplot(x='Happiness Change Range',y='index',data=happiness_deltas['Happiness Change Range'].value_counts().reset_index(),orient='x')
df[df.year_id==2017].corr()['Happiness Score'].reset_index().round(2).sort_values(by='Happiness Score',ascending=False)[3:]
#Prepare Data
X=df[df.year_id==2017].drop(['Happiness Rank','Happiness Score','Upper Confidence Interval','Lower Confidence Interval','year_id','Happiness_change','Country','Region','Standard Error'],axis=1)
y=df[df.year_id==2017]['Happiness Score']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
#Run Model
rf=RandomForestRegressor(max_depth=4,n_estimators=50)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
#Evaluation and Vizualisation
print(np.sqrt(mse(y_pred,y_test)))

plt.scatter(y_pred,y_test)
plt.xlabel('Predicted')
plt.ylabel('True')
importance_df=pd.DataFrame(index=X.columns,data=rf.feature_importances_).sort_values(by=0,ascending=False)
sns.barplot(y=importance_df.index,x=importance_df[0],data=importance_df,orient='h')
df_2=df[df.year_id==2016]
pivot_features=[
    'Freedom',
    'Generosity',
    'Health (Life Expectancy)',
    'Economy (GDP per Capita)',
    'Trust (Government Corruption)',
]
#Create a continent attribute
df_2['Continent']=df_2['Region'].map({'Sub-Saharan Africa':'Africa','Middle East and Northern Africa':'Africa','Central and Eastern Europe':'Europe','Western Europe':'Europe','Southern Asia':'Asia','Eastern Asia':'Asia','Southeastern Asia':'Asia'})
df_2['Continent']=np.where(df_2['Continent'].isnull(),df_2['Region'],df_2['Continent'])
#Preparation of df
scaled_df=scaler.fit_transform(df_2[pivot_features])
scaled_df=pd.DataFrame(scaled_df,columns=pivot_features)
#plotting radar chart 
def plot_radar(df,title) :
    data_frame = df
    #Prepare attributes
    data_frame_x = data_frame.groupby('Continent').mean()[pivot_features].reset_index()
    data_frame_x=data_frame_x.T
    #data_frame_x.columns  = ["cluster_id","feature","yes"]
    #data_frame_x["no"]    = data_frame.shape[0]  - data_frame_x["yes"]
    #data_frame_x  = data_frame_x[data_frame_x["feature"] != "cluster_id"]
    
    #average 
    trace1 = go.Scatterpolar(r = data_frame_x.drop('Continent',axis=0)[0].values.tolist(),
                             theta = data_frame_x.drop('Continent',axis=0).index.tolist(),
                             fill  = "toself",
                             name = "Africa",
                             mode = "markers+lines",
                             marker = dict(size = 10)
                            )
    trace2 = go.Scatterpolar(r = data_frame_x.drop('Continent',axis=0)[1].values.tolist(),
                            theta = data_frame_x.drop('Continent',axis=0).index.tolist(),
                             fill  = "toself",
                             name = "Asia",
                             mode = "markers+lines",
                             marker = dict(size = 10)
                            )
    
    trace3 = go.Scatterpolar(r = data_frame_x.drop('Continent',axis=0)[2].values.tolist(),
                              theta = data_frame_x.drop('Continent',axis=0).index.tolist(),
                             #fill  = "toself",
                             name = "Australia and New Zealand",
                             mode = "markers+lines",
                             marker = dict(size = 10)
                            )
       
    trace4 = go.Scatterpolar(r = data_frame_x.drop('Continent',axis=0)[3].values.tolist(),
                            theta = data_frame_x.drop('Continent',axis=0).index.tolist(),
                             fill  = "toself",
                             name = "Europe",
                             mode = "markers+lines",
                             marker = dict(size = 10)
                            )
          
    trace5 = go.Scatterpolar(r = data_frame_x[4].drop('Continent',axis=0).values.tolist(),
                             theta = data_frame_x.drop('Continent',axis=0).index.tolist(),
                             #fill  = "toself",
                             name = "Latin America",
                             mode = "markers+lines",
                             marker = dict(size = 10),
                            
                            )
    
          
    trace6 = go.Scatterpolar(r = data_frame_x[5].drop('Continent',axis=0).values.tolist(),
                             theta = data_frame_x.drop('Continent',axis=0).index.tolist(),
                             fill  = "toself",
                             name = "North America",
                             mode = "markers+lines",
                             marker = dict(size = 10),
                            
                            )
    
    layout = go.Layout(dict(polar = dict(radialaxis = dict(visible = False,
                                                           side = "counterclockwise",
                                                           showline = True,linewidth = 2,
                                                           tickwidth = 2,gridcolor = "white",
                                                           gridwidth = 2),
                                         angularaxis = dict(tickfont = dict(size = 10),
                                                            layer = "below traces"
                                                           ),
                                         bgcolor  = "white",
                                        ),
                            font=dict(family='Times New Roman', size=12, color='#7f7f7f'),
                            paper_bgcolor = "white",
                            title = title,height = 700))
    
    data = [trace2,trace4,trace3,trace5,trace1,trace6]
    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig)

#plot
plot_radar(df_2,'Continents')

plt.figure(figsize=(14,10))
sns.boxplot(y='Region',x='Happiness Score',hue='Continent',data=df_2,orient='h')
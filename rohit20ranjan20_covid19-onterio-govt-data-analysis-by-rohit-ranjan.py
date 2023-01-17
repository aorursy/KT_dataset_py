#Author Rohit Ranjan
#Analysis on ontario_government data on covid infection

%reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.executable
import plotly.express as px
import datetime
import plotly.graph_objects as go 
%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.options.display.max_columns=100


#reading input file
os.chdir("/kaggle/input/uncover")
#os.listdir() 
onterio_ds = pd.read_csv("/kaggle/input/uncover/ontario_government/confirmed-positive-cases-of-covid-19-in-ontario.csv")
#Histogram to visulaise the relation between age and recovery of covid 19 patient
# Find out whether young people immune system are better against covid 19 ?


fig1 = go.Figure()
sub_df=onterio_ds[onterio_ds['outcome1']=='Resolved']

fig1=px.histogram(sub_df,x='age_group',title='age wise recovery pattern')
fig1.update_layout(    
    xaxis_title="Age class",
    yaxis_title="No of Recoveries",
    title = {
             'y':0.9,
        'x':0.5,
            'xanchor': 'center'
            ,'yanchor': 'top'
        }
)


# Below data visualization suggest that people in there 40's and 50's have better immune system to 
#fight this infections and have recoverd .
# Below use Scatter plot to show distribution of covid infection as per onterio infection trace data
df= pd.DataFrame(onterio_ds,columns=['reporting_phu_latitude','reporting_phu_longitude'])

fig=go.Figure(data=go.Scattergeo(lon=df['reporting_phu_longitude'],lat=df['reporting_phu_latitude'],mode='markers'))
fig.update_layout(title='Infection Distribution',geo_scope='world',)

#It is showing that eastern coast has more cases , and western coast is almost isolated from covid 19
# source of  infection as per age group 

df3= pd.DataFrame(onterio_ds,columns=['row_id','age_group','case_acquisitioninfo','client_gender'])

df3['no_of_cases']=df3.groupby(['age_group','case_acquisitioninfo','client_gender']).row_id.transform('count')

c=sns.catplot(x = 'age_group',
            y = 'no_of_cases', 
             col='case_acquisitioninfo',
              estimator=np.sum,
              kind='point',
            data =df3 )
c.fig.set_size_inches(17,10)




#As per below visualisation, It is clear that source of transmission (for all ages) is not traceable , which
#could mean that either community spread has taken place or people are not diclosing proper contact history



# Location wise covid infection growth trejectory

df3= pd.DataFrame(onterio_ds,columns=['row_id','accurate_episode_date','reporting_phu_city'])
# Adding month of detection field , extrcated from accurate_episode_date
df3['year'] = pd.DatetimeIndex(df3['accurate_episode_date']).year
#df3['month'] = pd.DatetimeIndex(df3['accurate_episode_date']).month


df3["month"] = pd.DatetimeIndex(df3['accurate_episode_date']).map(lambda x: x.month)
df3["month_class"]=df3['month'].astype(str)+"_"+df3['year'].astype(str)


df3['no_of_cases']=df3.groupby(['accurate_episode_date','reporting_phu_city','year','month','month_class']).row_id.transform('count')

px.scatter(df3,x="month_class" ,y="no_of_cases" ,color="reporting_phu_city")

#Below visualisation shows that toronto has seen most rapid increase in cases ,where as port hope curve has been almost
#flat throughout the outbreak
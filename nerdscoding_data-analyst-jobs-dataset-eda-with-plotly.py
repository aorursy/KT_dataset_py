import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns



import plotly.graph_objs as go

import plotly.express as px

import re



from wordcloud import WordCloud,STOPWORDS

%matplotlib inline
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')

data.drop("Unnamed: 0",1,inplace = True)
data.shape
data.head(10)
data.tail(10)
data.info()
data.replace(to_replace =-1 , value=np.nan,inplace=True)

data.replace(to_replace ='-1' , value=np.nan,inplace=True)

data.replace(to_replace =-1.0 , value=np.nan,inplace=True)
def FindingMissingValues(dataFrame):

    for col in dataFrame.columns:

        print('{0:.2f}% or {1} values are Missing in {2} Column'.format(dataFrame[col].isna().sum()/len(dataFrame)*100,dataFrame[col].isna().sum(),col),end='\n\n')



FindingMissingValues(data)
data.drop(['Easy Apply','Competitors'],1,inplace = True)
data['Job Domain'] = data['Job Title'].apply(lambda x: re.search(r',.*',x).group().replace(',','') if(bool(re.search(r',.*',x))) else x )

data['Job Role'] = data['Job Title'].apply(lambda x: re.search(r'.*,',x).group().replace(',','') if(bool(re.search(r',.*',x))) else x )


data['Min Salary'] = 0

data['Max Salary'] = 0



for x in range(len(data)):

    

    if(type(data.iloc[x,1])==float):

        data.iloc[x,15] = np.nan

        data.iloc[x,16] = np.nan

    else:

        cleanSal = data.iloc[x,1].replace('(Glassdoor est.)','').strip().split('-')

    

    if('K' in cleanSal[0]):

        data.iloc[x,15] = float(cleanSal[0].replace('$','').replace('K',''))*1000

    

        

    if('K' in cleanSal[1]):

        data.iloc[x,16]= float(cleanSal[1].replace('$','').replace('K',''))*1000

    
data.drop('Job Description',1,inplace=True)
data['Company Name'] = data['Company Name'].apply(lambda x: re.sub(r'\n.*','',str(x)))


data['MaxEmpSize'] = 0



for x in range(len(data)):

    emp = data.iloc[x,6]

    

    try:

        if(type(emp)==float or emp == 'Unknown'): #type(np.nan)== float

            data.iloc[x,16] =  np.nan

        elif('+' in emp):

            data.iloc[x,16] = float(emp.replace('+','').replace('employees','').strip())

        elif('employees' in emp):

            data.iloc[x,16] = float(emp.replace('employees','').strip().split('to')[1])

    except(Exception)as e:

        print(e,emp)



data['MaxRevenue'] = 0



for x in range(len(data)):

    rev = data.iloc[x,11]

    

    if(rev == 'Unknown / Non-Applicable' or type(rev)==float):

        data.iloc[x,17] = np.nan

    elif(('million' in rev) and ('billion' not in rev)):

        maxRev = rev.replace('(USD)','').replace("million",'').replace('$','').strip().split('to')

        if('Less than' in maxRev[0]):

            data.iloc[x,17] = float(maxRev[0].replace('Less than','').strip())*100000000

        else:

            if(len(maxRev)==2):

                data.iloc[x,17] = float(maxRev[1])*100000000

            elif(len(maxRev)<2):

                data.iloc[x,17] = float(maxRev[0])*100000000

    elif(('billion'in rev)):

        maxRev = rev.replace('(USD)','').replace("billion",'').replace('$','').strip().split('to')

        if('+' in maxRev[0]):

            data.iloc[x,17] = float(maxRev[0].replace('+','').strip())*1000000000

        else:

            if(len(maxRev)==2):

                data.iloc[x,17] = float(maxRev[1])*1000000000

            elif(len(maxRev)<2):

                data.iloc[x,17] = float(maxRev[0])*1000000000

        
data.drop(['Job Title','Salary Estimate','Size','Revenue'],1,inplace = True)

data.head(10)
data.describe().transpose()
data.describe(include='object').transpose()
dataAnalyst = data[data['Job Role']=='Data Analyst']



fig = go.Figure()



fig.add_trace(go.Box(y=dataAnalyst['Min Salary'],name='Min Salary',boxmean='sd'))

fig.add_trace(go.Box(y=dataAnalyst['Max Salary'],name='Max Salary',boxmean='sd'))



fig.update_layout(title='Minimum and Maximum Salary of Data Analyst',height=800)



fig.show()
industry = data.groupby(['Industry'])['MaxRevenue'].mean()

fig = go.Figure()



fig.add_trace(go.Scatter(x=industry.index,y=industry,mode='lines+markers',marker = dict(color = 'rgba(255,0, 0, 1.0)',size=12)))



fig.show()
industry = industry[industry>20000000000]



fig = go.Figure()



fig.add_trace(go.Scatter(x=industry.index,y=industry,mode='lines+markers',marker=dict(color='rgba(255,0,0,1.0)',size=12)))



fig.update_layout(title='Industry Type Which have Mean Revenue Greater than 20Billion',

                 yaxis=dict(title='Billion Dollars'),

                  xaxis=dict(title='Industry Type')

                 )



fig.show()
sector = data.groupby(['Sector']).mean()



fig = go.Figure()



fig.add_trace(go.Bar(x=sector.index,y=sector['Min Salary'],name='Mininum Mean Salary',text=sector['Min Salary'],textposition='auto' ))

fig.add_trace(go.Bar(x=sector.index,y=sector['Max Salary'],name='Maximum Mean Salary',text=sector['Min Salary'],textposition='auto'))



fig.update_layout(title='Mean Salary',barmode='stack')



fig.show()
fig = go.Figure()



fig.add_trace(go.Scatter(x=sector.index,y=sector['Rating'],name='Rating',text=sector['Rating'],

                         mode='markers',

                         marker=dict(size=sector['Rating']*8,color=sector['Rating'])

                        ))



fig.update_layout(title='Sectors with Average Rating',

                 xaxis=dict(title='Sectors'),

                 yaxis=dict(title='Average Rating'))



fig.show()
fig = go.Figure()



sector.dropna(inplace=True)

fig.add_trace(go.Scatter(x=sector.index,y=sector['MaxRevenue'],name='MaxRevenue',text=sector['Rating'],

                         mode='markers',

                         marker=dict(size=sector['MaxRevenue']**0.15,color=sector['MaxRevenue'])

                        ))



fig.update_layout(title='Sector with Average Revenue',

                 xaxis=dict(title='Sectors'),

                 yaxis=dict(title='Average MaxRevenue'))



fig.show()
data2 = data.dropna()

fig = px.sunburst(data2, path=['Sector','Location','Job Role'], values='Min Salary',height=800)

fig.update_layout(title='Sector -> Location -> JobRole -> Min Salary')

fig.show()
from wordcloud import WordCloud



def WordCloudMaking(data,col):

    invester = data[col][~pd.isnull(data[col])]



    wordCloud = WordCloud(width=500,height= 300).generate(' '.join(invester))



    plt.figure(figsize=(19,9))



    plt.axis('off')

    plt.title(data[col].name,fontsize=20)

    plt.imshow(wordCloud)

    plt.show()

WordCloudMaking(data,'Job Role')
WordCloudMaking(data,'Job Domain')
WordCloudMaking(data,'Industry')
WordCloudMaking(data,'Sector')
WordCloudMaking(data,'Company Name')
WordCloudMaking(data,'Type of ownership')
location = data[data['Job Role']=='Data Analyst']['Location'].value_counts()

fig = px.pie(location,names=location.index,values=location,height=800)

fig.update_traces(textposition='inside',textinfo='label+percent',hole=.4)



fig.update_layout(annotations=[dict(text='Locations with Maximum Data Analyst',showarrow=False)])



fig.show()
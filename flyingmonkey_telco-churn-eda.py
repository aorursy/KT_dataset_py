import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from IPython.display import HTML
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools
init_notebook_mode(connected=True)
from plotly.offline import iplot
import seaborn as sns
import plotly.figure_factory as ff
telco = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv',  header='infer')
telco.head()
display(telco.head(5))
print('Overall {} Rows and {} columns'.format(telco.shape[0], telco.shape[1]))
display(telco.nunique())
num_cols = ['MonthlyCharges','TotalCharges', 'tenure']
telco[num_cols] = telco[num_cols].apply(pd.to_numeric, errors='cource')
missingData = telco.isnull().sum()
missingData
churn = telco["Churn"].value_counts(dropna=False)
churn = churn.to_frame().reset_index()
total = telco.shape[0]
churn = churn[churn['index'] == 'Yes'] 
churn = churn['Churn']/total*100
print('Overall churn rate is {}%'.format(churn.values[0]))
Gender = telco["gender"].value_counts(dropna=False)
print('Number of male is {}'.format(Gender.values[0]),'and number of female is {}'.format(Gender.values[1]))

sex = telco.groupby(['gender','Churn']).size()
sex = sex.to_frame('count').reset_index()
sex = sex[sex['Churn'] == 'Yes'] 
sex = sex[['gender','count']]
labels = sex['gender']
values = sex['count']
trace = go.Pie(labels= labels, values=values)
iplot([trace],filename='basic_pie_chart')
seniorCitizen = telco.groupby(['gender','SeniorCitizen']).size().to_frame('count').reset_index()
seniorCitizen = seniorCitizen[seniorCitizen['SeniorCitizen'] == 1]
print('Number of male Senior Citizen is {}'.format(seniorCitizen['count'].values[1])
     ,'and Number of male Senior Citizen is {}'.format(seniorCitizen['count'].values[0]))
seniorchurn = telco.groupby(['gender','SeniorCitizen', 'Churn']).size().to_frame('count').reset_index()
seniorchurn = seniorchurn[seniorchurn['Churn'] == 'Yes'] 
seniorchurn = seniorchurn[seniorchurn['SeniorCitizen'] == 1]
seniorchurn

Gender = telco["gender"].value_counts(dropna=False)
Gender = Gender.to_frame().reset_index()
dfF = Gender.rename(columns={'gender': 'total','index':'gender'})
df4 = pd.merge(dfF,sex, on="gender")
df4['Result'] = df4['count']/df4['total']*100

df2 = sex.rename(columns={'count': 'total'})
df3 = pd.merge(df2,seniorchurn, on="gender")
df3['Result'] = df3['count']/df3['total']*100

data1 = go.Bar(
            x= df3['gender'],
            y= df3['Result'],
            name = 'Churn'
)
data2 = go.Bar(
            x= df4['gender'],
            y= df4['Result'],
            name = 'SeniorCitizen churn'
)

data = [data1, data2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='grouped-bar')
def functionPartener(df, col):
    part = df.groupby([col]).size()
    part = part.to_frame('count').reset_index()
    part = part.rename(columns={'count':'total'})
    partChurn = df.groupby([col,'Churn']).size()
    partChurn = partChurn.to_frame('count').reset_index()
    partChurn = partChurn[partChurn['Churn'] == 'Yes']
    final = pd.merge(part, partChurn, on=col)
    final['Result'] = final['count']/final['total']*100
    result = final.sort_values(by='Result', ascending=False)
    return result

def plotfunction(df, col2):
    col = df[[col2,'Result']]
    data = go.Bar(
            x= col[col2],
            y= col['Result'],
            name='Churn rate based on {}'.format(col2)
    )
    return data
    
ge = functionPartener(telco, 'gender')
gen = plotfunction(ge,'gender')

pt = functionPartener(telco, 'Partner')
ptr = plotfunction(pt,'Partner')

de = functionPartener(telco,'Dependents')
dep = plotfunction(de,'Dependents')

sh = tools.make_subplots(rows=1, cols=3)
sh.append_trace(gen, 1, 1)
sh.append_trace(ptr, 1, 2)
sh.append_trace(dep, 1, 3)

sh['layout'].update(height=500, width=1000, title='Gender, Partner and Dependent based churn rate')
iplot(sh)
ph = functionPartener(telco,'PhoneService')
phn = plotfunction(ph,'PhoneService')

mu = functionPartener(telco,'MultipleLines')
mul = plotfunction(mu,'MultipleLines')

In = functionPartener(telco,'InternetService')
Int = plotfunction(In,'InternetService')

on = functionPartener(telco,'OnlineSecurity')
onl = plotfunction(on,'OnlineSecurity')

onb = functionPartener(telco,'OnlineBackup')
onbk = plotfunction(onb,'OnlineBackup')

depr = functionPartener(telco,'DeviceProtection')
depro = plotfunction(depr,'DeviceProtection')



sh = tools.make_subplots(rows=2, cols=3)
sh.append_trace(phn, 1, 1)
sh.append_trace(mul, 1, 2)
sh.append_trace(Int, 1, 3)
sh.append_trace(onl, 2, 1)
sh.append_trace(onbk,2, 2)
sh.append_trace(depro,2, 3)

sh['layout'].update(height=1000, width=1000, title='Phone and Internet based churn rate')
iplot(sh)
tech = functionPartener(telco,'TechSupport')
techs = plotfunction(tech,'TechSupport')

stream = functionPartener(telco,'StreamingTV')
streamtv = plotfunction(stream,'StreamingTV')

stre = functionPartener(telco,'StreamingMovies')
streamMv = plotfunction(stre ,'StreamingMovies')

con = functionPartener(telco,'Contract')
cont = plotfunction(con ,'Contract')

pp = functionPartener(telco,'PaperlessBilling')
ppbil = plotfunction(pp ,'PaperlessBilling')

pay = functionPartener(telco,'PaymentMethod')
paym = plotfunction(pay ,'PaymentMethod')

sh = tools.make_subplots(rows=2, cols=3)

sh.append_trace(techs, 1, 1)
sh.append_trace(streamtv, 1, 2)
sh.append_trace(streamMv, 1, 3)
sh.append_trace(cont, 2, 1)
sh.append_trace(ppbil, 2, 2)
sh.append_trace(paym, 2, 3)
sh['layout'].update(height=1000, width=1000, title='Streaming and Payment mode based churn rate')
iplot(sh)
numeric_cols = telco[['tenure','MonthlyCharges','TotalCharges'] ]
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix,
            xticklabels=correlation_matrix.columns.values,
            yticklabels=correlation_matrix.columns.values)
ten = telco[['Churn','tenure']]
tenure1 = ten[ten['Churn'] == 'No']
tenure1 = tenure1.rename(columns={'Churn':'No churn'})
tenure2 = ten[ten['Churn'] == 'Yes']

trace0 = go.Box(
    x= tenure1['tenure'],
    name = 'No Churn',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    x=tenure2['tenure'],
    name = 'churn',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
mn = telco[['Churn','MonthlyCharges']]
mn1 = mn[mn['Churn'] == 'No']
mn1 = mn1.rename(columns={'Churn':'No churn', 'MonthlyCharges':'Monthcharge'})
mn2 = mn[mn['Churn'] == 'Yes']

trace0 = go.Box(
    x= mn1['Monthcharge'],
    name = 'No Churn',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    x=mn2['MonthlyCharges'],
    name = 'churn',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
tc = telco.groupby(['Churn','TotalCharges']).size().to_frame().reset_index()
tc1 = tc[tc['Churn'] == 'No']
tc1 = tc1.rename(columns={'Churn':'No churn', 'TotalCharges':'Totalcharge'})
tc2 = tc[tc['Churn'] == 'Yes']
trace0 = go.Box(
    x = tc1['Totalcharge'],
    name = 'No Churn',
   marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    x = tc2['TotalCharges'],
    name = 'churn',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)

data1 = [trace0, trace1]
iplot(data1)
# Basic Libraries

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

sb.set() # set the default Seaborn style for graphics

from sklearn.model_selection import train_test_split

import chart_studio 

chart_studio.tools.set_credentials_file(username='LIANGJING', api_key='KeSQYwdxfNybb8zzzMj9')



import chart_studio.plotly as py

import plotly.io as pio

pio.templates.default = "none"



import plotly.graph_objects as go
townName = pd.read_csv('../input/frenchemployment/name_geographic_information.csv')

townName=townName.drop(['latitude','longitude','European Union Circonscription','region code','region','administrative center','department number','department','prefecture','num of the circumpscription','postal code','distance'], axis = 1)

townName.sort_values(by=['town code'])

townName['town code'] = townName['town code'].astype(int)

townName=townName.set_index('town code')
townDemographics = pd.read_csv('../input/frenchemployment/population.csv')

townSexPopulation=townDemographics.drop(['age category','cohabitation mode','geographic level'], axis = 1)

townSexPopulation=townSexPopulation.pivot_table(index='town code',columns='sex',aggfunc=sum)

townSexPopulation.loc['Total',:]= townSexPopulation.sum(axis=0) #Total row

townSexPopulation
highestWomenRatio_col= ["men","women",'men percentage','women percentage']

highestWomenRatioTable=pd.DataFrame(columns =highestWomenRatio_col)

highestWomenRatioTable['women percentage']= (townSexPopulation['num of people','women']/(townSexPopulation['num of people','women']+townSexPopulation['num of people','men']))

highestWomenRatioTable['men percentage']= (townSexPopulation['num of people','men']/(townSexPopulation['num of people','women']+townSexPopulation['num of people','men']))

highestWomenRatioTable['men']= townSexPopulation['num of people','men']

highestWomenRatioTable['women']= townSexPopulation['num of people','women']

highestWomenRatioTable=highestWomenRatioTable.nlargest(10, 'women percentage')

highestWomenRatioTable=highestWomenRatioTable.join(townName)

highestWomenRatioTable=highestWomenRatioTable.sort_values('women percentage', ascending=False)

highestWomenRatioTable
highestMenRatio_col= ["men","women",'men percentage','women percentage']

highestMenRatioTable=pd.DataFrame(columns =highestMenRatio_col)

highestMenRatioTable['women percentage']= (townSexPopulation['num of people','women']/(townSexPopulation['num of people','women']+townSexPopulation['num of people','men']))

highestMenRatioTable['men percentage']= (townSexPopulation['num of people','men']/(townSexPopulation['num of people','women']+townSexPopulation['num of people','men']))

highestMenRatioTable['men']= townSexPopulation['num of people','men']

highestMenRatioTable['women']= townSexPopulation['num of people','women']

highestMenRatioTable=highestMenRatioTable.nlargest(10, 'men percentage')

highestMenRatioTable=highestMenRatioTable.join(townName)

highestMenRatioTable=highestMenRatioTable.sort_values('men percentage', ascending=False)

highestMenRatioTable
cohabitationTable=townDemographics.drop(['age category','sex','geographic level','town name'], axis = 1)

cohabitationTable['num of people'] = cohabitationTable['num of people'].astype(int)

cohabitationTable=cohabitationTable.pivot_table(index='town code',columns='cohabitation mode',aggfunc=sum)

cohabitationTable.loc['Total',:]= cohabitationTable.sum(axis=0) #Total row

cohabitationTable
AgeTable=townDemographics.drop(['cohabitation mode','sex','geographic level','town name'], axis = 1)

AgeTable['num of people'] = AgeTable['num of people'].astype(int)

AgeTable=AgeTable.pivot_table(index='town code',columns='age category',aggfunc=sum)

AgeTable.loc['Total',:]= AgeTable.sum(axis=0) #Total row

AgeTable
salary = pd.read_csv('../input/frenchemployment/net_salary_per_town_categories.csv')

salaryGenderType=salary.drop(['town name','18-25 yo','26-50 yo','>50 years old','women 18-25 yo','women 26-50 yo','women >50 yo','men 18-25 yo','men 26-50 yo','men >50 yo'], axis = 1)

salaryGenderType=pd.merge(salaryGenderType, townName, on='town code', how='inner')

salaryGenderType
Salary_col = ['town code','town name','mean net salary','women','man','pay disparity']

Salary=pd.DataFrame(columns =Salary_col)

Salary['town code']=salaryGenderType['town code']

Salary['town name']=salaryGenderType['town name']

Salary['mean net salary']=salaryGenderType['mean net salary']

Salary['women']=salaryGenderType['women']

Salary['man']=salaryGenderType['man']

Salary['pay disparity']=Salary['man']-Salary['women']

highestSalary=Salary.nlargest(15, 'pay disparity')

highestSalary=highestSalary.set_index('town code')

highestSalary
SalaryManager_col = ['town code','town name','manager','manager (w)','manager (m)','pay disparity']

SalaryManager=pd.DataFrame(columns =SalaryManager_col)

SalaryManager['town code']=salaryGenderType['town code']

SalaryManager['town name']=salaryGenderType['town name']

SalaryManager['manager']=salaryGenderType['manager']

SalaryManager['manager (w)']=salaryGenderType['manager (w)']

SalaryManager['manager (m)']=salaryGenderType['manager (m)']

SalaryManager['pay disparity']=SalaryManager['manager (m)']-SalaryManager['manager (w)']

highestSalaryManager=SalaryManager.nlargest(15, 'pay disparity')

highestSalaryManager=highestSalaryManager.set_index('town code')

highestSalaryManager
SalaryEmployee_col = ['town code','town name','employee','employee (w)','employee (m)','pay disparity']

SalaryEmployee=pd.DataFrame(columns =SalaryEmployee_col)

SalaryEmployee['town code']=salaryGenderType['town code']

SalaryEmployee['town name']=salaryGenderType['town name']

SalaryEmployee['employee']=salaryGenderType['employee']

SalaryEmployee['employee (w)']=salaryGenderType['employee (w)']

SalaryEmployee['employee (m)']=salaryGenderType['employee (m)']

SalaryEmployee['pay disparity']=SalaryEmployee['employee (m)']-SalaryEmployee['employee (w)']

highestSalaryEmployee=SalaryEmployee.nlargest(15, 'pay disparity')

highestSalaryEmployee=highestSalaryEmployee.set_index('town code')

highestSalaryEmployee
SalaryExecutive_col = ['town code','town name','executive','executive (w)','executive (m)','pay disparity']

SalaryExecutive=pd.DataFrame(columns=SalaryExecutive_col)

SalaryExecutive['town code']=salaryGenderType['town code']

SalaryExecutive['town name']=salaryGenderType['town name']

SalaryExecutive['executive']=salaryGenderType['executive']

SalaryExecutive['executive (w)']=salaryGenderType['executive (w)']

SalaryExecutive['executive (m)']=salaryGenderType['executive (m)']

SalaryExecutive['pay disparity']=SalaryExecutive['executive (m)']-SalaryExecutive['executive (w)']

highestSalaryExecutive=SalaryExecutive.nlargest(15, 'pay disparity')

highestSalaryExecutive=highestSalaryExecutive.set_index('town code')

highestSalaryExecutive
SalaryWorker_col = ['town code','town name','worker','worker (w)','worker (m)','pay disparity']

SalaryWorker=pd.DataFrame(columns=SalaryWorker_col)

SalaryWorker['town code']=salaryGenderType['town code']

SalaryWorker['town name']=salaryGenderType['town name']

SalaryWorker['worker']=salaryGenderType['worker']

SalaryWorker['worker (w)']=salaryGenderType['worker (w)']

SalaryWorker['worker (m)']=salaryGenderType['worker (m)']

SalaryWorker['pay disparity']=(SalaryWorker['worker (m)']-SalaryWorker['worker (w)'])

highestSalaryWorker=SalaryWorker.nlargest(15, 'pay disparity')

highestSalaryWorker=highestSalaryWorker.set_index('town code')

highestSalaryWorker
total = townSexPopulation.loc['Total'].tolist()

import matplotlib.pyplot as plt

# Pie chart

labels = ['Men', 'Women']

sizes = total

#colors

colors = ['#66b3ff','#ff9999']

 

fig1, ax1 = plt.subplots()

patches, texts, autotexts = ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)

for text in texts:

    text.set_color('grey')

for autotext in autotexts:

    autotext.set_color('grey')

# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
import plotly.graph_objects as go



town=highestWomenRatioTable['town name'].values.tolist()



fig = go.Figure()

fig.add_trace(go.Bar(

    x=town,

    y=highestWomenRatioTable['men'].values.tolist(),

    name='men',

    marker_color='lightblue'

))

fig.add_trace(go.Bar(

    x=town,

    y=highestWomenRatioTable['women'].values.tolist(),

    name='women',

    marker_color='pink'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(title="Top 10 towns with highest ratio of female vs male", barmode='group', xaxis_tickangle=-45)

fig.show()
import plotly.graph_objects as go



town=highestMenRatioTable['town name'].values.tolist()



fig = go.Figure()

fig.add_trace(go.Bar(

    x=town,

    y=highestMenRatioTable['men'].values.tolist(),

    name='men',

    marker_color='lightblue'

))

fig.add_trace(go.Bar(

    x=town,

    y=highestMenRatioTable['women'].values.tolist(),

    name='women',

    marker_color='pink'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(title="Top 10 towns with highest ratio of male vs female", barmode='group', xaxis_tickangle=-45)

fig.show()
import plotly.graph_objects as go

x = ['couple w/o children','couple with children','children with parents','living alone','children with single parent','single parent with children','not living with family']

y = cohabitationTable.loc['Total'].tolist()

y.sort(reverse = True)



# Use the hovertext kw argument for hover text

fig = go.Figure(data=[go.Bar(x=x, y=y)])

# Customize aspect

fig.update_traces(marker_color='#8ac6d1', marker_line_color='#8ac6d1',

                  marker_line_width=1.5, opacity=0.6)

fig.update_layout(title_text='Overall demographic: cohabitation mode')

fig.show()
import plotly.graph_objects as go

y=['0-4 ','5-9 ','10-14 ','15-19 ','20-24 ','25-29 ','30-34 ','35-39 ','40-44 ','45-49 ','50-54 ','55-59 ','60-64 ','65-69 ','70-74 ','75-79 ']

x = AgeTable.loc['Total'].tolist()



fig = go.Figure()

fig.add_trace(go.Bar(

    y=y,

    x=x,

    name='overall cohabitation mode',

    orientation='h',

    marker=dict(

        color='#beebe9',

        line=dict(color='#beebe9', width=1)

    )

))



fig.update_layout(barmode='stack')

fig.update_layout(title_text='Overall demographic: age category')

fig.show()
import plotly.graph_objects as go

import numpy as np



mean = Salary['women'].values.tolist()

executive = SalaryExecutive['executive (w)'].values.tolist()

manager = SalaryManager['manager (w)'].values.tolist()

employee = SalaryEmployee['employee (w)'].values.tolist()

worker = SalaryWorker['worker (w)'].values.tolist()



fig = go.Figure()

fig.add_trace(go.Box(y=mean, name='mean',

                marker_color = '#f7d695'))

fig.add_trace(go.Box(y=executive, name = 'executive',

                marker_color = '#ff80b0'))

fig.add_trace(go.Box(y=manager, name = 'manager',

                marker_color = '#ff80b0'))

fig.add_trace(go.Box(y=employee, name = 'employee',

                marker_color = '#ff80b0'))

fig.add_trace(go.Box(y=worker, name = 'worker',

                marker_color = '#ff80b0'))



fig.update_layout(title="Overview distribution on female salary")



fig.show()
import plotly.graph_objects as go

import numpy as np



mean = Salary['man'].values.tolist()

executive = SalaryExecutive['executive (m)'].values.tolist()

manager = SalaryManager['manager (m)'].values.tolist()

employee = SalaryEmployee['employee (m)'].values.tolist()

worker = SalaryWorker['worker (m)'].values.tolist()



fig = go.Figure()

fig.add_trace(go.Box(y=mean, name='mean',

                marker_color = '#f7d695'))

fig.add_trace(go.Box(y=executive, name = 'executive',

                marker_color = '#88e1f2'))

fig.add_trace(go.Box(y=manager, name = 'manager',

                marker_color = '#88e1f2'))

fig.add_trace(go.Box(y=employee, name = 'employee',

                marker_color = '#88e1f2'))

fig.add_trace(go.Box(y=worker, name = 'worker',

                marker_color = '#88e1f2'))



fig.update_layout(title="Overview distribution on male salary")



fig.show()
highestTownName = highestSalary['town name'].values.tolist()

highestMean = highestSalary['mean net salary'].values.tolist()

highestWomen = highestSalary['women'].values.tolist()

highestMen = highestSalary['man'].values.tolist()



import plotly.express as px

data = px.data.gapminder()



fig = go.Figure()

fig.add_trace(go.Scatter(

    x=highestWomen,

    y=highestTownName,

    marker=dict(color="pink", size=12),

    mode="markers",

    name="Women",

))



fig.add_trace(go.Scatter(

    x=highestMen,

    y=highestTownName,

    marker=dict(color="lightblue", size=12),

    mode="markers",

    name="Men",

))



fig.add_trace(go.Scatter(

    x=highestMean,

    y=highestTownName,

    marker=dict(color="beige", size=12),

    mode="markers",

    name="Town's average",

))





fig.update_layout(title="Top 15 Gender Earnings Disparity",

                  xaxis_title="Mean Net Salary")



fig.show()
highestTownName = highestSalaryExecutive['town name'].values.tolist()

highestMean = highestSalaryExecutive['executive'].values.tolist()

highestWomen = highestSalaryExecutive['executive (w)'].values.tolist()

highestMen = highestSalaryExecutive['executive (m)'].values.tolist()



fig = go.Figure()

fig.add_trace(go.Scatter(

    x=highestWomen,

    y=highestTownName,

    marker=dict(color="pink", size=12),

    mode="markers",

    name="female executive",

))



fig.add_trace(go.Scatter(

    x=highestMen,

    y=highestTownName,

    marker=dict(color="lightblue", size=12),

    mode="markers",

    name="male executive",

))



fig.add_trace(go.Scatter(

    x=highestMean,

    y=highestTownName,

    marker=dict(color="beige", size=12),

    mode="markers",

    name="Town's average",

))





fig.update_layout(title="Top 15 Gender Earnings Disparity (executives)",

                  xaxis_title="Executive Salary")



fig.show()
highestTownName = highestSalaryWorker['town name'].values.tolist()

highestMean = highestSalaryWorker['worker'].values.tolist()

highestWomen = highestSalaryWorker['worker (w)'].values.tolist()

highestMen = highestSalaryWorker['worker (m)'].values.tolist()



fig = go.Figure()

fig.add_trace(go.Scatter(

    x=highestWomen,

    y=highestTownName,

    marker=dict(color="pink", size=12),

    mode="markers",

    name="female worker",

))



fig.add_trace(go.Scatter(

    x=highestMen,

    y=highestTownName,

    marker=dict(color="lightblue", size=12),

    mode="markers",

    name="male worker",

))



fig.add_trace(go.Scatter(

    x=highestMean,

    y=highestTownName,

    marker=dict(color="beige", size=12),

    mode="markers",

    name="Town's average",

))





fig.update_layout(title="Top 15 Gender Earnings Disparity (workers)",

                  xaxis_title="Worker Salary")



fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(

    y=['mean','worker','executive'],

    x=[23.4,18.0,31.0],

    name='female',

    orientation='h',

    marker=dict(

        color='pink',

        line=dict(color='rgba(246, 78, 139, 1.0)', width=1)

    )

))

fig.add_trace(go.Bar(

    y=['mean','worker','executive'],

    x=[46.9,53.2,51.4],

    name='male',

    orientation='h',

    marker=dict(

        color='lightblue',

        line=dict(color='rgba(58, 71, 80, 1.0)', width=1)

    )

))

fig.update_layout(title="Fourqueux: Gender Earnings Disparity")

fig.show()
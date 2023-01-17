# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
replace=lambda x:(x.replace(".",""))
column_list=['Confirmed cases New', 'Confirmed cases Total',
        'Deaths New', 'Deaths Total',
       'Recoveries New', 'Recoveries Total', 'Severe cases Intubated',
       'Severe cases ICU', 'Number of tests New', 'Number of tests Total']

converter={i:replace for i in column_list}



data_covid = pd.read_csv('/kaggle/input/covid19inturkey/COVID-19_in_Turkey.csv',converters=converter)
data_covid=data_covid.replace('-',0)

for i in column_list:
       data_covid[i]=data_covid[i].astype(int)

data_covid.dtypes
data_covid.columns
data_covid.rename(columns={'Week':"Week_No",
'Confirmed cases New':"Confirmed_New_Case",
'Confirmed cases Total':'Confirmed_Total_Case',
'Deaths New':'Deaths_New',
'Deaths Total':'Deaths_Total',
'Recoveries New':"Recoveries_New", 
'Recoveries Total':'Recoveries_Total',
'Severe cases Intubated':'Intubated_Case',
'Severe cases ICU':"Intensive_Care_Case",
'Number of tests New':"New_Test",
'Number of tests Total':"Total_Test"
 },inplace=True)
data_covid.index=[i for i in range(1,data_covid.Date.size+1)]
data_covid["Active_Case"]=data_covid["Confirmed_Total_Case"]-data_covid["Recoveries_Total"]-data_covid["Deaths_Total"]
data_covid.head(5)
data_covid.tail()
data_covid.info()
data_covid.dtypes
data_covid.describe().T
data_covid.corr()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data_covid.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
# ploty kutuphanesinden bar grafik turu kullanilarak gorsellestirildi
fig = go.Figure(data=[
    go.Bar(name='Confirmed Total Case', x=data_covid['Date'], y=data_covid['Confirmed_Total_Case'], marker_color='rgba(135, 206, 250, 0.8)'),
    go.Bar(name='Deaths Total', x=data_covid['Date'], y = data_covid['Deaths_Total'], marker_color='rgba(255, 0, 0, 0.8)'),
    go.Bar(name='Recoveries Total', x=data_covid['Date'] , y=data_covid['Recoveries_Total'], marker_color='rgba(0, 255, 0, 0.8)',)
])
fig.update_layout(barmode='group', 
                title_text='COVID-19 TURKEY TOTAL CONFIRMED CASE / DEATH / RECOVERIES COUNTS',
                xaxis_tickangle=-45,
                bargap=0.15, 
                bargroupgap=0.1,
                width=2000,
                height=500 )
fig.show()
# Bar gosterimi
# Virusun 11/4/2020 tarihinde zirve yaptigini ve dususe gectigini gosterir. 4. Hafta zirvede
fig = go.Figure(data=[
    go.Bar(name='New cases per day', x=data_covid['Date'], y = data_covid['Confirmed_New_Case'], marker_color='rgba(255, 0, 0, 0.8)'),
    
])
fig.update_layout(barmode='group', title_text='COVID-19 TURKEY New Cases Per Day', xaxis_tickangle=-45,width=1000)
fig.show()
# ilk ölümlü vaka 15/3/2020 ve 19/4 dan sonra olum oranlarinda dusus gozlenmektedir.
fig = go.Figure(data=[
    go.Bar(name='Deaths New', x=data_covid['Date'], y=data_covid['Deaths_New'], marker_color='rgba(0, 255, 0, 0.8)'),   
])
fig.update_layout(barmode='group', title_text='COVID-19 TURKEY  Deaths Per Day', xaxis_tickangle=-45,width=1000)
fig.show()
def plot_plot(variable):
    plt.figure(figsize= (9,4))
    plt.plot(data_covid[variable])
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(" {} cases distribution".format(variable))
    plt.show()
    
numericVar = ["Confirmed_Total_Case", "Recoveries_Total", "Deaths_Total", "Active_Case"]
for n in numericVar:
    plot_plot(n)
# korelasyonla arasinda eksi baglanti oldugundan dolayi tablolastirildi
data_covid.plot(kind="scatter", x="Confirmed_Total_Case", y="Recoveries_Total",alpha = 1,color = "green")
plt.xlabel("Confirmed")              
plt.ylabel("Recovered")
plt.title("Confirmed_Total_Case & Recoveries_Total")     
plt.show()
data_covid.plot(kind="scatter", x="Confirmed_Total_Case", y="Deaths_Total",alpha = 1,color = "red")
plt.xlabel("Confirmed")              
plt.ylabel("Deaths")
plt.title("Confirmed_Total_Case & Deaths_Total")     
plt.show()
data_covid.plot(kind="scatter", x="Confirmed_Total_Case", y="Active_Case",alpha = 1,color = "green")
plt.xlabel("Confirmed")              
plt.ylabel("Active")
plt.title("Confirmed_Total_Case & Active")     
plt.show()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x=data_covid.Date,y=data_covid.Deaths_New,color='green',alpha=0.5)
sns.pointplot(x=data_covid.Date,y=data_covid.Confirmed_New_Case,color='red',alpha=0.5)
plt.text(0,3500,'Confirmed',color='red',fontsize =10,style = 'italic')
plt.text(0,1000,' Deaths',color='green',fontsize = 10,style = 'italic')
plt.xlabel('Date',fontsize = 10,color='black')
plt.ylabel('Values',fontsize = 10,color='black')
plt.xticks(rotation= 90)
plt.title('New Case for Deaths  VS   Confirmed',fontsize = 20,color='blue')
plt.grid()
# aktif vakalara ve olum sayilari-- vakalarin artmasina ragmen olum oranlari dusuk seviyede tutulabilmistir
fig = go.Figure(data=[
    go.Bar(name='Aktif Case', x=data_covid['Date'], y=data_covid['Active_Case'], marker_color='rgba(135, 206, 250, 0.8)'),
    go.Bar(name='Deaths Total', x=data_covid['Date'] , y=data_covid['Deaths_Total'], marker_color='rgba(0, 255, 0, 0.8)')
])
fig.update_layout(barmode='group', title_text='COVID-19 TURKEY TOTAL Death / Active Case', xaxis_tickangle=-45,width=1000)
fig.show()

labels = 'death', 'health', 'intensivecare', 'totalcase',
sizes = [2, 35, 13, 50]
explode = (0.2, 0.2, 0.2, 0.4)  

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', 
        pctdistance=0.6, labeldistance=1.1, 
        textprops={'fontsize': 13,'color':"g"},
        radius=1.5, shadow=True, startangle=270, rotatelabels=False
       )

ax1.set_title("Total",color='c',size=26, x = 0.4, y = 1.3);
plt.legend(loc="center right",bbox_to_anchor=(1.3, 0, 0.5, 3))
plt.show()
case_rate = [0] #ilk değerden önce bir değer olmadığı için ilk değeri 0 verdik
test_rate = [0]

for i in range(1,len(data_covid)):
    if (data_covid["Total_Test"][i]==0):
        testRate=0
    else:
        testRate =  round((data_covid["Total_Test"][i+1] - data_covid["Total_Test"][i]) / data_covid["Total_Test"][i], 2)

    if data_covid["Confirmed_Total_Case"][i]==0:
        caseRate=0
    else:    
        caseRate = round((data_covid["Confirmed_Total_Case"][i+1] - data_covid["Confirmed_Total_Case"][i]) / data_covid["Confirmed_Total_Case"][i], 2)
    
    test_rate.append(testRate)
    case_rate.append(caseRate)


data_covid["Test_Increase_Rate"] = test_rate
data_covid["Case_Increase_Rate"] = case_rate
# scatter gosterimi

plt.figure(figsize=(21,7))


plt.scatter(data_covid.Date, data_covid.Test_Increase_Rate, s=150, color="m", edgecolors='black', linewidths=1.5, alpha=0.6)
plt.scatter(data_covid.Date, data_covid.Case_Increase_Rate, s=100, color="y", edgecolors='black', linewidths=3.5, alpha=0.6)


plt.xticks(rotation=45)
plt.title('Test', color = 'g', size = 13)
plt.xlabel('Date', color = "c");
plt.ylabel('Index', color = "r");
plt.show()
# scatter gosterimi
c=data_covid.tail(10)

plt.figure(figsize=(21,7))


plt.scatter(c.Date,c.Test_Increase_Rate, s=150, color="m", edgecolors='black', linewidths=1.5, alpha=0.6)
plt.scatter(c.Date, c.Case_Increase_Rate, s=100, color="y", edgecolors='black', linewidths=3.5, alpha=0.6)


plt.xticks(rotation=45)
plt.title('Test', color = 'g', size = 13)
plt.xlabel('Date', color = "c");
plt.ylabel('Index', color = "r");
plt.show()
# birden fazla cizgi grafigi birlikte gosterme
c=data_covid.tail(10)

plt.figure(figsize=(7,4))


plt.plot(c.Date, c.Intubated_Case, color = "b", 
         linewidth = 2, linestyle="solid", 
         marker="8", markersize = 14, markerfacecolor='r')

plt.plot(c.Date, c.	Intensive_Care_Case, color = "m", 
         linewidth = 4, linestyle="dotted", 
         marker="8", markersize = 14, markerfacecolor='y')

plt.xticks(rotation=45)
plt.title('INTUBA - INTENSIVE Index', color = 'g', size = 13)
plt.xlabel('Date', color = "c");
plt.ylabel('Index', color = "r");
plt.show()

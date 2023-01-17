import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

from collections import namedtuple
df = pd.read_csv('../input/veri.csv',parse_dates=['Date'])

df.rename(columns={'Date':'Date', 'Country_Region':'Country'}, inplace=True)
df.head()
df.describe()
df.groupby('Date').sum()
df.info()
df.isnull().sum()
start_date = df.Date.min()

end_date = df.Date.max()

print('Dataset information:\n 1. Start date = {}\n 2. End date = {}'.format(start_date, end_date))
print(df['ConfirmedCases'].describe())

print(df['Fatalities'].describe())
print(df['new_tests'].describe())
df.columns
mask = df['Date'].max()

cum_confirmed = sum(df[df['Date'] == mask].ConfirmedCases)

cum_fatal = sum(df[df['Date'] == mask].Fatalities)



###################################################################



print('Number of Countries are: ', len(df['Country'].unique()))

print('Training dataset ends at: ', mask)

print('Number of cumulative confirmed cases are: ', cum_confirmed)

print('Number of cumulative fatal cases are: ', cum_fatal)
total = df.groupby('Country').sum()

total = total.sort_values(by=['ConfirmedCases'], ascending=False)

total.style.background_gradient(cmap='OrRd')
cum_per_country = df[df['Date'] == mask].groupby(['Date','Country']).sum().sort_values(['ConfirmedCases'], ascending=False)

cum_per_country[:5]
confirmed = df.groupby('Date').sum()['ConfirmedCases'].reset_index()

deaths = df.groupby('Date').sum()['Fatalities'].reset_index()

tests = df.groupby('Date').sum()['new_tests'].reset_index()
fig = go.Figure()

fig.add_trace(go.Bar(x=confirmed['Date'],

                y=confirmed['ConfirmedCases'],

                name='Vaka',

                marker_color='blue'

                ))

fig.add_trace(go.Bar(x=deaths['Date'],

                y=deaths['Fatalities'],

                name='Ölüm',

                marker_color='Red'

                ))

fig.add_trace(go.Bar(x=tests['Date'],

                y=tests['new_tests'],

                name='Test',

                marker_color='Green'

                ))



fig.update_layout(

    title='5 Ülkenin Corona Virüs Vaka, Ölüm, Test Sayıları (Bar Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Vaka Sayısı',

        titlefont_size=15,

        tickfont_size=15,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.1, # gap between bars of adjacent location coordinates.

    bargroupgap=0.05 # gap between bars of the same location coordinate.

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed['Date'], 

                         y=confirmed['ConfirmedCases'],

                         mode='lines+markers',

                         name='Vaka',

                         line=dict(color='blue', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths['Date'], 

                         y=deaths['Fatalities'],

                         mode='lines+markers',

                         name='Ölüm',

                         line=dict(color='Red', width=2)

                        ))

fig.add_trace(go.Scatter(x=tests['Date'], 

                         y=tests['new_tests'],

                         mode='lines+markers',

                         name='Test',

                         line=dict(color='Green', width=2)

                        ))

fig.update_layout(

    title='5 Ülkenin Corona Virüs Vaka, Ölüm, Test Sayıları (Line Chart)',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Vaka Sayıları',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
#TODO: optimize code

date = df['Date']

cc_ir = df[df['Country'] == 'Iran'].groupby(['Date']).sum().ConfirmedCases

ft_ir = df[df['Country'] == 'Iran'].groupby(['Date']).sum().Fatalities

cc_tr = df[df['Country'] == 'Turkey'].groupby(['Date']).sum().ConfirmedCases

ft_tr = df[df['Country'] == 'Turkey'].groupby(['Date']).sum().Fatalities

cc_ity = df[df['Country'] == 'Italy'].groupby(['Date']).sum().ConfirmedCases

ft_ity = df[df['Country'] == 'Italy'].groupby(['Date']).sum().Fatalities

cc_gmn = df[df['Country'] == 'Germany'].groupby(['Date']).sum().ConfirmedCases

ft_gmn = df[df['Country'] == 'Germany'].groupby(['Date']).sum().Fatalities

cc_frc = df[df['Country'] == 'France'].groupby(['Date']).sum().ConfirmedCases

ft_frc = df[df['Country'] == 'France'].groupby(['Date']).sum().Fatalities



fig = go.Figure()

# add trace

fig.add_trace(go.Scatter(x=date, y=cc_ir, name='Iran'))

fig.add_trace(go.Scatter(x=date, y=cc_tr, name='Turkey'))

fig.add_trace(go.Scatter(x=date, y=cc_ity, name='Italy'))

fig.add_trace(go.Scatter(x=date, y=cc_gmn, name='Germany'))

fig.add_trace(go.Scatter(x=date, y=cc_frc, name='France'))

fig.update_layout(title="5 Ülke için Kümülatif Vaka Dağılımı",

    xaxis_title="Tarih",

    yaxis_title="Vaka")

fig.update_xaxes(nticks=50)



fig.show()
fig = go.Figure()

# add traces

fig.add_trace(go.Scatter(x=date, y=ft_ir, name='Iran'))

fig.add_trace(go.Scatter(x=date, y=ft_tr, name='Turkey'))

fig.add_trace(go.Scatter(x=date, y=ft_ity, name='Italy'))

fig.add_trace(go.Scatter(x=date, y=ft_gmn, name='Germany'))

fig.add_trace(go.Scatter(x=date, y=ft_frc, name='France'))

fig.update_layout(title="5 Ülke için Kümülatif Ölüm Sayısı Dağılımı",

    xaxis_title="Tarih",

    yaxis_title="Ölüm")

fig.update_xaxes(nticks=30)



fig.show()
turkey=df[df['Country'] == 'Turkey']

iran=df[df['Country'] == 'Iran']

france=df[df['Country'] == 'France']

germany=df[df['Country'] == 'Germany']

italy=df[df['Country'] == 'Italy']
df.boxplot(column='Daily Cases',by = 'new_deaths',figsize = (20,10))
turkey.boxplot( column='Daily Cases',by = 'new_deaths',figsize = (20,10))
iran.boxplot( column='Daily Cases',by = 'new_deaths',figsize = (20,10))
germany.boxplot( column='Daily Cases',by = 'new_deaths',figsize = (20,10))
france.boxplot( column='Daily Cases',by = 'new_deaths',figsize = (20,10))
italy.boxplot( column='Daily Cases',by = 'new_deaths',figsize = (20,10))
turkey1 = turkey.loc[:,["ConfirmedCases","Fatalities"]]

turkey1.plot().set_title("Turkey")

france1 = france.loc[:,["ConfirmedCases","Fatalities"]]

france1.plot().set_title("France")
iran1 = iran.loc[:,["ConfirmedCases","Fatalities"]]

iran1.plot().set_title("Iran")
germany1 = germany.loc[:,["ConfirmedCases","Fatalities"]]

germany1.plot().set_title("Germany")
italy1 = italy.loc[:,["ConfirmedCases","Fatalities"]]

italy1.plot().set_title("Italy")
Country=pd.DataFrame()

temp = df.loc[df["Date"]==df["Date"][len(df)-1]].groupby(['Country'])["ConfirmedCases"].sum().reset_index()

Country['Name']=temp["Country"]

Country['Values']=temp["ConfirmedCases"]



fig = px.choropleth(Country, locations='Name',

                    locationmode='country names',

                    color="Values")

fig.update_layout(title="Corona spread on 09-07-2020")

fig.show()
# Korelasyon katsayısı negatif ise iki değişken arasında ters ilişki vardır, yani "değişkenlerden biri artarken diğeri azalmaktadır" denir. 

# Korelasyon katsayısı pozitif ise "değişkenlerden biri artarken diğeride artmaktadır" yorumu yapılır.



# İki değişken arasında hesaplanan korelasyon (r) değeri:

# • r<0.20 ve sıfıra yakın değerler ilişkinin olmadığı ya da çok zayıf ilişkiyi işaret eder.

# • 0.20-0.39 arasında ise zayıf ilişki

# • 0.40-0.59 arasında ise orta düzeyde ilişki

# • 0.60-0.79 arasında ise yüksek düzeyde ilişki

# • 0.80-1.0 ise çok yüksek ilişki olduğu yorumu yapılır.
# Korelasyon katsayısını yorumlarken neden-sonuç ilişkisinden bahsetmek doğru değildir. 

# Çünkü korelasyon bize iki değişken arasındaki ilişkinin büyüklüğünü gösterirken neden-sonuç ilişkisine dair bir şey söylememektedir. 

# A değişkeni B değişkeni etkiliyor olabilir ya da B değişkeni A değişkenini etkiliyor olabilir. 

# Başka biralternatif de iki A ile B değişkenleri arasında neden-sonuç ilişkisi olmayabilir. 

# Korelasyon değeri neden-sonuç ilişkisinin yönünü vermemektedir. 

# Korelasyon değerine bakarak neden-sonuç ilişkisinden bahsedemememizin başka sebebi de üçüncü bir değişkenin etkisidir.

# İki değişkenin arasındaki neden-sonuç ilişkisini diğer değişkenlerin etkisinden bağımsız düşünemeyiz.

f,ax = plt.subplots(figsize=(15, 5))

sns.heatmap(df.corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()



corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
f,ax = plt.subplots(figsize=(15, 5))

sns.heatmap(df[df['Country'] == 'Turkey'].corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()



corr_tr = df[df['Country'] == 'Turkey'].corr()

corr_tr.style.background_gradient(cmap='coolwarm')
f,ax = plt.subplots(figsize=(15, 5))

sns.heatmap(df[df['Country'] == 'Iran'].corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()



corr_ir = df[df['Country'] == 'Iran'].corr()

corr_ir.style.background_gradient(cmap='coolwarm')
f,ax = plt.subplots(figsize=(15, 5))

sns.heatmap(df[df['Country'] == 'Italy'].corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()



corr_it = df[df['Country'] == 'Italy'].corr()

corr_it.style.background_gradient(cmap='coolwarm')
f,ax = plt.subplots(figsize=(15, 5))

sns.heatmap(df[df['Country'] == 'Germany'].corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()



corr_ger = df[df['Country'] == 'Germany'].corr()

corr_ger.style.background_gradient(cmap='coolwarm')
f,ax = plt.subplots(figsize=(15, 5))

sns.heatmap(df[df['Country'] == 'France'].corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()



corr_fr = df[df['Country'] == 'France'].corr()

corr_fr.style.background_gradient(cmap='coolwarm')
# Almanya

corr_tr.corrwith(corr_ger)
# İtalya

corr_tr.corrwith(corr_it)
# Fransa

corr_tr.corrwith(corr_fr)
# İran

corr_tr.corrwith(corr_ir)
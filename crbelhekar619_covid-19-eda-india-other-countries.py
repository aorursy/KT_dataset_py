# Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.style as style

style.use('fivethirtyeight')

# Load data
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)
print ('Last Updated: ' + str(df.Date.max()))
#Selecting only the required columns for analysis and droping others
df = df.drop(['Province/State'], axis = 'columns')
df.head(3)
# Group df dataset by 'Date' with sum parameter and analyse the 'Confirmed','Deaths' values.
cases = df.groupby('Date').sum()[['Confirmed', 'Recovered', 'Deaths']]
cases.plot(kind='line', figsize = (15,7) , marker='o',linewidth=2)
plt.bar(cases.index, cases['Confirmed'],alpha=0.3,color='c')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Number of cases', fontsize=15)
plt.title('Worldwide Covid-19 cases - Confirmed, Recovered & Deaths',fontsize=24)
plt.grid(True)
# The signature bar
plt.text(x = 18275.0, y = -400000, s = ' ©Chaitanya                                                                                                      Source: Novel Coronavirus (COVID-19) Cases, provided by JHU CSSE',fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')

plt.legend()
plt.savefig('worldwide.png')
# Checking after first 100 cases
ncases=15000

# India
df_india = df[(df['Country']=='India') & (df['Confirmed']>ncases)].reset_index()
df_india['Days'] = [i+1 for i in range(len(df_india))]
df_india = df_india[df_india['Days']<=len(df_india)]

# Italy
df_italy = df[(df['Country']=='Italy') & (df['Confirmed']>ncases)].reset_index()
df_italy['Days'] = [i+1 for i in range(len(df_italy))]
df_italy = df_italy[df_italy['Days']<=len(df_india)]

# Iran
df_iran = df[(df['Country']=="Iran") & (df['Confirmed']>ncases)].reset_index()
df_iran['Days'] = [i+1 for i in range(len(df_iran))]
df_iran = df_iran[df_iran['Days']<=len(df_india)]

# Japan
df_japan = df[(df['Country']=="Japan") & (df['Confirmed']>ncases)].reset_index()
df_japan['Days'] = [i+1 for i in range(len(df_japan))]
df_japan = df_japan[df_japan['Days']<=len(df_india)]
plt.figure(figsize=(15,6))
fig, ax = plt.subplots(2,2, sharey= True, figsize=(15,10))
  
first = ax[0,0].bar(df_iran.index, df_iran['Confirmed'],alpha=0.3,color='b', tick_label=df_iran.Days)
second = ax[0,1].bar(df_italy.index, df_italy['Confirmed'],alpha=0.3,color='g', tick_label=df_italy.Days)
third = ax[1,0].bar(df_japan.index, df_japan['Confirmed'],alpha=0.3,color='r', tick_label=df_japan.Days)
fourth = ax[1,1].bar(df_india.index, df_india['Confirmed'],alpha=0.3,color='m', tick_label=df_india.Days)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = int(rect.get_height())
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(first, ax[0,0])
autolabel(second, ax[0,1])
autolabel(third, ax[1,0])
autolabel(fourth, ax[1,1])

ax[0,0].set_title('No. of confirmed cases in Iran',fontsize=14)
ax[0,1].set_title('No. of confirmed cases in Italy',fontsize=14)
ax[1,0].set_title('No. of confirmed cases in Japan',fontsize=14)
ax[1,1].set_title('No. of confirmed cases in India',fontsize=14)

ax[0,0].set_xlabel('Days',fontsize=10)
ax[0,1].set_xlabel('Days',fontsize=10)
ax[1,0].set_xlabel('Days',fontsize=10)
ax[1,1].set_xlabel('Days',fontsize=10)

plt.style.use('ggplot')
plt.show()
fig.savefig('confirmed.png')
plt.figure(figsize=(15,6))
fig, ax = plt.subplots(2,2, sharey= True, figsize=(15,10))
  
first = ax[0,0].bar(df_iran.index, df_iran['Deaths'],alpha=0.3,color='b', tick_label=df_iran.Days)
second = ax[0,1].bar(df_italy.index, df_italy['Deaths'],alpha=0.3,color='g', tick_label=df_italy.Days)
third = ax[1,0].bar(df_japan.index, df_japan['Deaths'],alpha=0.3,color='r', tick_label=df_japan.Days)
fourth = ax[1,1].bar(df_india.index, df_india['Deaths'],alpha=0.3,color='m', tick_label=df_india.Days)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = int(rect.get_height())
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(first, ax[0,0])
autolabel(second, ax[0,1])
autolabel(third, ax[1,0])
autolabel(fourth, ax[1,1])

ax[0,0].set_title('No. of deaths in Iran',fontsize=14)
ax[0,1].set_title('No. of deaths in Italy',fontsize=14)
ax[1,0].set_title('No. of deaths in Japan',fontsize=14)
ax[1,1].set_title('No. of deaths in India',fontsize=14)

ax[0,0].set_xlabel('Days',fontsize=10)
ax[0,1].set_xlabel('Days',fontsize=10)
ax[1,0].set_xlabel('Days',fontsize=10)
ax[1,1].set_xlabel('Days',fontsize=10)

plt.style.use('ggplot')
plt.show()
fig.savefig('deaths.png')
ncases = 500
india = df[(df['Country'] == 'India') & (df['Confirmed']>ncases)].reset_index()
japan = df[(df['Country'] == 'Japan') & (df['Confirmed']>ncases)].reset_index()
italy = df[(df['Country']== 'Italy') & (df['Confirmed']>ncases)].reset_index()
iran = df[(df['Country']== 'Iran') & (df['Confirmed']>ncases)].reset_index()
spain = df[(df['Country']== 'Spain') & (df['Confirmed']>ncases)].reset_index()
skorea = df[(df['Country']== 'South Korea') & (df['Confirmed']>ncases)].reset_index()
usa = df[(df['Country']== 'US') & (df['Confirmed']>ncases)].reset_index()
usa = usa.groupby('Date').sum()

india['Days'] = [i+1 for i in range(len(india))]
japan['Days'] = [i+1 for i in range(len(japan))]
italy['Days'] = [i+1 for i in range(len(italy))]
iran['Days'] = [i+1 for i in range(len(iran))]
spain['Days'] = [i+1 for i in range(len(spain))]
skorea['Days'] = [i+1 for i in range(len(skorea))]
usa['Days'] = [i+1 for i in range(len(usa))]
# Group df dataset by 'Date' with sum parameter and analyse the 'Confirmed','Deaths' values.
plt.figure(figsize=(12,6))
plt.bar(india.Days, india['Confirmed'],alpha=0.6,color='c')
plt.bar(japan.Days, japan['Confirmed'],alpha=0.3,color='m')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Number of cases', fontsize=15)
plt.title('Comparison between India and Japan after first 10 cases',fontsize=20)
plt.grid(True)
plt.style.use('fivethirtyeight')
plt.legend(('India', 'Japan'))
plt.savefig('indvsjp.png')
plt.figure(figsize=(10, 6))
plt.plot(india.Days, india['Confirmed'], linewidth=4)
plt.plot(japan.Days, japan['Confirmed'], linewidth=4, alpha=0.3)
plt.plot(italy.Days, italy['Confirmed'], linewidth=4)
plt.plot(iran.Days, iran['Confirmed'], linewidth=4)
plt.plot(spain.Days, spain['Confirmed'],linewidth=4)
plt.plot(skorea.Days, skorea['Confirmed'], linewidth=4, alpha=0.3)
plt.plot(usa.Days, usa['Confirmed'], linewidth=4, alpha=0.8)
plt.legend(['India', 'Japan', 'Italy', 'Iran', 'Spain', 'South Korea', 'USA'], loc='upper right')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Number of cases', fontsize=15)
plt.title('Growth in the no. of cases after first 500 cases',fontsize=15)
plt.grid(True)
#plt.style.use('ggplot')
plt.savefig('growth.png')
# The signature bar
plt.text(x = -10, y = -150000, s = ' ©Chaitanya                                     Source: Novel Coronavirus (COVID-19) Cases, provided by JHU CSSE',fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')
plt.show()
plt.figure(figsize=(12, 7))
plt.plot(india.Days, india['Confirmed'], linewidth=4)
plt.plot(japan.Days, japan['Confirmed'], linewidth=4, alpha=0.3)
plt.plot(italy.Days, italy['Confirmed'], linewidth=4)
plt.plot(iran.Days, iran['Confirmed'], linewidth=4)
plt.plot(spain.Days, spain['Confirmed'],linewidth=4, color = 'c')
plt.plot(skorea.Days, skorea['Confirmed'], linewidth=4, alpha=0.3)
plt.plot(usa.Days, usa['Confirmed'], linewidth=4, alpha=0.6, color = 'm')
plt.yscale("log")
plt.legend(['India', 'Japan', 'Italy', 'Iran', 'Spain', 'South Korea', 'USA'], loc='upper right')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Number of cases', fontsize=15)
plt.title('Growth in the no. of cases after first 500 cases in log scale',fontsize=15)
plt.grid(True)
plt.style.use('fivethirtyeight')
plt.savefig('growth_log.png')
# The signature bar
plt.text(x = -8, y = 45, s = '©Chaitanya                                                                Source: Novel Coronavirus (COVID-19) Cases, provided by JHU CSSE',fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')
plt.show()
colors = [  [86/255,180/255,233/255], [0,158/255,115/255], [213/255,94/255,0], [0,114/255,178/255], [0,0,0], [230/255,159/255,0]]

plt1 = cases.plot( figsize = (12,6), color = colors, legend = False)
plt.yscale('log')
plt1.tick_params(axis = 'both', which = 'major', labelsize = 14)
plt1.axhline(y = 10, color = 'black', linewidth = 1.3, alpha = .7)

# The signature bar
plt1.text(x = 18280.0, y = 0.15, s = ' ©Chaitanya                                                Source: Novel Coronavirus (COVID-19) Cases, provided by JHU CSSE',fontsize = 14, color = '#f0f0f0', backgroundcolor = 'grey')

# Adding a title and a subtitle
plt1.text(x = 18284, y = 2000000, s = "Worldwide Covid-19 cases - Confirmed, Recovered & Deaths",color = colors[4],
               fontsize = 20, weight = 'bold', alpha = .75)

plt1.text(x = 18296, y = 90000, s = 'Confirmed', color = colors[0], weight = 'bold', rotation = 6,
              backgroundcolor = '#f0f0f0')
plt1.text(x = 18323, y = 24000, s = 'Recovered', color = colors[1], weight = 'bold', rotation = 4,
              backgroundcolor = '#f0f0f0')
plt1.text(x = 18328, y = 6900, s = 'Deaths', color = colors[2], weight = 'bold', rotation = 6,
               backgroundcolor = '#f0f0f0')
plt.show()
plt.savefig('538style.png')
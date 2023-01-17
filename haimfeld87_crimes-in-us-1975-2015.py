import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import operator

import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





df=pd.read_csv("../input/report.csv")
df.columns
df_total_US = df[df.agency_jurisdiction == 'United States']

df_no_total_US=df[df.agency_jurisdiction != 'United States']
ax=plt.figure(figsize=(10,22))

ax = sns.barplot(df_no_total_US["violent_crimes"]+df_no_total_US["homicides"]+

                 df_no_total_US["rapes"]+df_no_total_US["assaults"]+df_no_total_US["robberies"],                 

                 y=df_no_total_US["agency_jurisdiction"],estimator=sum ,ci=0)

ax.set_title('Total crimes in US during (1975-2015)')

ax.set(xlabel='Total number of crimes', ylabel='City')
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



a = set(df_no_total_US["agency_jurisdiction"])

a = list(a)



doubles = dict()

for i in range(0,len(a)):

    doubles[i] = df_no_total_US[df_no_total_US['agency_jurisdiction'].str.contains(a[i])]



trace = dict()

for i in range(0,len(a)):

    trace[i] = go.Scatter(x = doubles[i]['report_year'],y=doubles[i]['violent_crimes'],name = a[i],opacity = 0.8)



data = [trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9],

        trace[10],trace[11],trace[12],trace[13],trace[14],trace[15],trace[16],trace[17],trace[18],trace[19],

         trace[20],trace[21],trace[22],trace[23],trace[24],trace[25],trace[26],trace[27],trace[28],trace[29],

          trace[30],trace[31],trace[32],trace[33],trace[34],trace[35],trace[36],trace[37],trace[38],trace[39],

           trace[40],trace[41],trace[42],trace[43],trace[44],trace[45],trace[46],trace[47],trace[48],trace[49],

            trace[50],trace[51],trace[52],trace[53],trace[54],trace[55],trace[56],trace[57],trace[58],trace[59],

             trace[60],trace[61],trace[62],trace[63],trace[64],trace[65],trace[66],trace[67]]



layout = dict(title = "Total Crimes in US during (1975-2015)",

              xaxis = dict(title = 'Year'),

              yaxis = dict(title = 'Total Crimes'),)



fig = dict(data=data, layout=layout)

py.iplot(fig)
df_no_total_US.fillna(value=0)

violent_crimes=df_no_total_US["violent_crimes"]

total_number_of_crimes=np.nansum(violent_crimes)

print("Total number of crimes between 1975-2015 in US is:",total_number_of_crimes)

max_index, max_crimes_jur= max(enumerate(df_no_total_US["violent_crimes"]), key=operator.itemgetter(1))

max_crimes_jura=df_no_total_US.iloc[max_index] ["agency_jurisdiction"]



NYC = df_no_total_US[df_no_total_US.agency_jurisdiction == 'New York City, NY']

total_nyc=NYC['violent_crimes'].sum()

precent=total_nyc/total_number_of_crimes*100

print("The jurisdiction with the most crimes is:",max_crimes_jura, "and total number of crimes:",total_nyc, "that is %0.2f"% precent,"% of total crimes in USA between 1975-2015.")
plt.figure(figsize=(18,10))



ax1 = plt.subplot(511)

sns.barplot(NYC["report_year"], NYC["homicides"])

plt.ylabel("Number of Homicides")



ax2 = plt.subplot(512, sharex=ax1)

sns.barplot(NYC["report_year"], NYC["rapes"])

plt.ylabel("Number of Rapes")



ax3 = plt.subplot(513, sharex=ax1)

sns.barplot(NYC["report_year"], NYC["assaults"])

plt.ylabel("Number of Assaults")



ax4 = plt.subplot(514, sharex=ax1)

sns.barplot(NYC["report_year"], NYC["robberies"])

plt.ylabel("Number of Robberies")



ax1.set_title('Crimes in New York City in 1975-2015')

plt.xlabel("Report Year")

plt.show()
max_index, max_crimes = max(enumerate(NYC["violent_crimes"]), key=operator.itemgetter(1))

max_crimes_year=NYC.iloc[max_index] ["report_year"]

print("The higest number of crimes was",max_crimes,"in",max_crimes_year)
plt.figure(figsize=(18,4))

sns.barplot(NYC["report_year"], NYC["crimes_percapita"], palette="BuGn_d",)

plt.ylabel("Rate of crimes")

plt.title('Crimes rates in New York City in 1975-2015')

plt.xlabel("Report Year")

plt.show()
ax=plt.figure(5,figsize=(15,5))

ax = plt.plot([1975, 1975], [0, 193000], 'b-', lw=2)

ax = plt.text(1975, 193000, 'Abraham Beame',color='blue',horizontalalignment='center')



ax = plt.plot([1978, 1978], [0, 193000], 'b-', lw=2)

ax = plt.text(1978, 193000, 'Ed Koch',color='blue',horizontalalignment='left')



ax = plt.plot([1990, 1990], [0, 193000], 'b-', lw=2)

ax = plt.text(1990, 193000, 'David Dinkins',color='blue',horizontalalignment='center')



ax = plt.plot([1994, 1994], [0, 193000], 'r-', lw=2)

ax = plt.text(1994, 193000, 'Rudy Giuliani',color='red',horizontalalignment='center')



ax = plt.plot([2002, 2002], [0, 193000], 'r-', lw=2)

ax = plt.text(2002, 193000, 'Michael Bloomberg',color='red',horizontalalignment='center')



ax = plt.plot([2014, 2014], [0, 193000], 'b-', lw=2)

ax = plt.text(2014, 193000, 'Bill de Blasio',color='blue',horizontalalignment='center')



ax = plt.plot(NYC["report_year"],NYC["violent_crimes"])

plt.title('Total crimes in New York City during (1975-2015)')

plt.xlabel("Report Year")

plt.ylabel("Total number of crimes")

plt.ylim([0,200000])

plt.xlim([1974,2016])

plt.show()
plt.figure(figsize=(18,4))

sns.barplot(df_total_US["report_year"], df_total_US["violent_crimes"], palette="RdBu_r",)

plt.ylabel("Total crimes in USA")

plt.title('Crimes in USA in 1975-2015')

plt.xlabel("Report Year")

plt.show()
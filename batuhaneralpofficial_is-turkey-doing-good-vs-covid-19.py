# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
covid_19_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
covid_19_data.head()
covid_19_data = covid_19_data.drop(columns = ["Province/State","Lat","Long"])
covid_19_data.head()
#Start every country with a blank list

confirmed_dict = {}

for i in covid_19_data["Country/Region"]:

    confirmed_dict[i] = []

    

#Get dates to a list

dates = list(covid_19_data.columns)

dates.remove('Country/Region')



#Iterate through every row of the dataframe

for count,country in enumerate(covid_19_data["Country/Region"].values.tolist()):

    

    #Checking if a country has duplicate entries

    if confirmed_dict[country] == []:

        

        temp = []

        for date in dates:

            temp.append([date,covid_19_data[date][count]])

        confirmed_dict[country] = temp

    

    else:

        #Adding the duplicate value to the old value

        old = confirmed_dict[country]

        for entry in old:

            for date in dates:

                if date == entry[0]:

                    entry[1] += covid_19_data[date][count]
def createData(list,dict):

    result = {}

    for country in list:

        

        temp_X = []   #This will store total confirmed

        temp_Y = []   #This will store weekly difference by day

        

        for i in range(len(dict[country]) - 7):

            temp_X.append(dict[country][i + 7][1])   #Total Value

            temp_Y.append(dict[country][i + 7][1] - dict[country][i][1])   #Weekly Difference by Day

                

        result[country] = [temp_X,temp_Y]

    return result
plt.style.use("seaborn-dark")

for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:

    plt.rcParams[param] = '#212946'  # bluish dark grey

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:

    plt.rcParams[param] = '0.9'  # very light grey

size = (15,7.5)



fig, ax = plt.subplots(figsize=size)

ax.grid(color='#2A3459')

colors = ['#08F7FE','#FE53BB','#F5D300','#00ff41']

ax.set_title('Scalar Confirmed')



n_lines = 10

diff_linewidth = 1.05

alpha_value = 0.03



countries = ["US","Turkey","Italy","Korea, South"]

plot_data = createData(countries,confirmed_dict)





for idx,country in enumerate(countries):

    ax.plot(plot_data[country][0],plot_data[country][1],label = country,color = colors[idx])



    

#plt.yscale('log')

#plt.xscale('log')

plt.legend()

plt.show()
fig, ax = plt.subplots(figsize=size)

ax.grid(color='#2A3459')

colors = ['#08F7FE','#FE53BB','#F5D300','#00ff41']

ax.set_title('Logarithmic Confirmed')



countries = ["US","Turkey","Italy","Korea, South"]

plot_data = createData(countries,confirmed_dict)



for idx,country in enumerate(countries):

    ax.plot(plot_data[country][0],plot_data[country][1],label = country,color = colors[idx])



plt.yscale('log')

plt.xscale('log')

plt.legend()

plt.show()


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=size)



colors = ['#08F7FE','#FE53BB','#F5D300','#00ff41']

fig.suptitle('Logarithmic Confirmed by Country')







ax1.plot(plot_data[countries[0]][0],plot_data[countries[0]][1],label = countries[0],color = colors[0])

ax2.plot(plot_data[countries[1]][0],plot_data[countries[1]][1],label = countries[1],color = colors[1])

ax3.plot(plot_data[countries[2]][0],plot_data[countries[2]][1],label = countries[2],color = colors[2])

ax4.plot(plot_data[countries[3]][0],plot_data[countries[3]][1],label = countries[3],color = colors[3])

ax1.set_title(countries[0])

ax2.set_title(countries[1])

ax3.set_title(countries[2])

ax4.set_title(countries[3])



ax1.set_xscale("log")

ax2.set_xscale("log")

ax3.set_xscale("log")

ax4.set_xscale("log")



ax1.set_yscale("log")

ax2.set_yscale("log")

ax3.set_yscale("log")

ax4.set_yscale("log")





for ax in fig.get_axes():

    ax.label_outer()
plt.style.use("seaborn-dark")

for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:

    plt.rcParams[param] = '#212946'  # bluish dark grey

for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:

    plt.rcParams[param] = '0.9'  # very light grey



fig, ax = plt.subplots(figsize=size)

ax.grid(color='#2A3459')

colors = ['#08F7FE','#FE53BB','#F5D300','#00ff41']

ax.set_title('How is Turkey Doing? (Downwards Better)')



n_lines = 10

diff_linewidth = 1.05

alpha_value = 0.03



countries = ["Turkey"]

plot_data = createData(countries,confirmed_dict)



margin = 0.01



for idx,country in enumerate(countries):

    ax.plot(plot_data[country][0],plot_data[country][1],label = country,color = colors[idx])

    if np.log(plot_data[country][1])[-1] > np.log(plot_data[country][1])[-2] + margin:

        ax.fill_between(x=plot_data[country][0],y1=plot_data[country][1],y2=[0] * len(plot_data[country][0]),color=colors[1],alpha=0.1)

    elif np.log(plot_data[country][1])[-1] < np.log(plot_data[country][1])[-2] - margin:

        ax.fill_between(x=plot_data[country][0],y1=plot_data[country][1],y2=[0] * len(plot_data[country][0]),color=colors[3],alpha=0.1)

    elif np.log(plot_data[country][1])[-1] < np.log(plot_data[country][1])[-2] + margin and plot_data[country][1][-1] > plot_data[country][1][-2] - margin:

        ax.fill_between(x=plot_data[country][0],y1=plot_data[country][1],y2=[0] * len(plot_data[country][0]),color=colors[2],alpha=0.1)



    

plt.yscale('log')

plt.xscale('log')

plt.legend()

plt.show()
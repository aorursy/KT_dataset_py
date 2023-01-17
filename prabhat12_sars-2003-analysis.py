# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")

data.head()
for col in data.columns:

    print(data[col].value_counts)
data.isnull().sum()
list_country=data.Country.unique()

count_country=len(list_country)

print("Number of countries: "+str(count_country))

print("List of countries: "+str(list_country))

dict_country={}



print("Country name \t Cases \t Death \t Recovered \n")

for country in list_country:

    

    cases=data[(data["Country"]==country)]["Cumulative number of case(s)"].sum()

    death=data[(data["Country"]==country)]["Number of deaths"].sum()

    recover=data[(data["Country"]==country)]["Number recovered"].sum()

    

    dict_country[country]=[cases,death,recover]



    print(country + "\t" + str(cases) + "\t" + str(death) + "\t" + str(recover))    

sort_country={key: val for key, val in sorted(dict_country.items(), key=lambda item: item[1][0],reverse=True)}

sort_country
# defining the color and label which we will use frequently 



colors=["#58D68D",'#45B39D',"#138D75"]

label=["Number of Cases","Number of people Died","Number of people Recovered"]

X=np.arange(10)





fig, axs = plt.subplots(3, 1)

fig.set_size_inches(20, 15)



# plotting the graph

for count,country in enumerate(list(sort_country)[0:10]):

    axs[0].bar(X[count],sort_country[country][0],color=colors[0])

    axs[1].bar(X[count],sort_country[country][1],color=colors[1])

    axs[2].bar(X[count],sort_country[country][2],color=colors[2])



# changing the x axis labels to country name for all three graphs

for i in range(3):

    plt.sca(axs[i])

    plt.title(label[0])

    plt.xticks(X,list(sort_country)[0:10])



plt.show()
# grouping the date data to monthly data

data['Date']=pd.to_datetime(data['Date'])



date_group = data.groupby(pd.Grouper(key='Date', freq='1M')).sum() # groupby 1 month 

date_group.index =date_group.index.strftime('%B')



date_group
months=["March","April","May","June","July"]



X=np.arange(len(months))



fig, axs = plt.subplots(1, 3)

fig.set_size_inches(25, 10)



# plotting the graph

axs[0].bar(months,date_group["Cumulative number of case(s)"],color=colors[0])    

axs[1].bar(months,date_group["Number of deaths"],color=colors[1])    

axs[2].bar(months,date_group["Number recovered"],color=colors[2])    



# changing the x axis labels to country name for all three graphs

for i in range(3):

    plt.sca(axs[i])

    plt.title(label[i])



plt.show()
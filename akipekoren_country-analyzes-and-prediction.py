# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/master.csv")

data.head(25)
data.drop(columns = ["country-year"],inplace= True)

data.drop(columns = ["HDI for year"],inplace= True)





country_list = []

for country in data.country.unique():

    country_list.append(country)



new_count=[]

summ =0

count = 0

index = 0

suicide_num_list = []

year_count = 0



while index < len(data.country):

    if  data.country[index] == country_list[count]:

        summ = summ + data.suicides_no[index]

        index += 1

            

    else:

        suicide_num_list.append(summ)

        summ =0

        count += 1

suicide_num_list.append(summ)



suicide_num_list







iplot([go.Choropleth(

    locationmode='country names',

    locations=country_list,

    text=country_list,

    z=suicide_num_list

)])



year_list = []

for year in data.year.unique():

    year_list.append(year)



year_list = sorted(year_list) 





kill_by_year = []

for each in year_list:

    new_data = data[data.year == each]

    kill_by_year.append(sum(new_data.suicides_no))



kill_by_year   

    

        
plt.figure(figsize = (15,10))

sns.barplot(x = year_list, y =kill_by_year,)

plt.xticks(rotation = 45)

plt.xlabel("list of year")

plt.ylabel("count of suicide")

plt.title("year vs sucicide graph")
data.iloc[:,7] = [each.replace(",","") for each in data.iloc[:,7]]

data.iloc[:,7]= data.iloc[:,7].astype(float)

data.info()



    
liste =[]

sum_list=[]





      

index =0

temp = 0

year_List = []

country_List = []

for country in country_list:

    new_data = data[data.country == country]

    for year in year_list:

        year_data = new_data[new_data.year == year]

        if len(year_data.year) != 0:

            for i in range(len(year_data.year)):

                liste.append(data.population[index])

                index += 1

            sum_list.append(sum(liste[temp:index]))

            year_List.append(year)

            country_List.append(country)

            

            

            temp = index

        else:

            pass





sum_list


year_Data = pd.DataFrame(year_List)

country_Data = pd.DataFrame(country_List)

num_of_pop_Data = pd.DataFrame(sum_list)



pop_Data = pd.concat([country_Data,num_of_pop_Data,year_Data],axis =1)

names = ["Country", "Total_Population", "Year"]

pop_Data.columns=names



pop_Data





USA_data = pop_Data[pop_Data.Country == "United States"]





plt.figure(figsize=(15,10))

ax= sns.barplot(x=USA_data.Year, y=USA_data.Total_Population,

                palette = sns.cubehelix_palette(len(USA_data.Year)))

plt.xlabel("Year")

plt.ylabel("Population")

plt.title("Population of USA between 1985 to 2015")
import plotly.graph_objs as go



year_pop_data = pop_Data[pop_Data.Year == 2000]

trace =go.Scatter(

                    x = year_pop_data.Country,

                    y = year_pop_data.Total_Population,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= year_pop_data.Country)



data = [trace]



layout = dict(title = "Total population in 2000.",

              xaxis= dict(title= 'Country name',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Population',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
sum_pop_list = []

for year in year_list:  

    new_Data = pop_Data[pop_Data.Year == year]

    sum_pop_list.append(sum(new_Data.Total_Population))







sns.barplot(x=year_list,y = sum_pop_list)

plt.title('Population by years',color = 'blue',fontsize=15)

plt.xticks(rotation = 90)

    

data = pd.read_csv("../input/master.csv")

data.drop(columns = ["country-year"],inplace= True)

data.drop(columns = ["HDI for year"],inplace= True)





data.head(25)
y = data.sex

ax = sns.countplot(y, label = "Count")



male, female = y.value_counts()

print("Number of male suicide {}".format(male))

print("Number of female suicide {}".format(female))


y = data.sex



x = data.drop(["country","age", "generation","sex",], axis =1)

x = x[["year","suicides_no","population","suicides/100k pop", "gdp_per_capita ($)"]]

x_norm = (x - x.min())/(x.max()-x.min())



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)





from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))

# Importing libraries for visualization and reading the data

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
from IPython.display import HTML
sns.set_style("darkgrid") # You can choose any style among these : darkgrid, whitegrid, dark, white, ticks
os.listdir('../input/novel-corona-virus-2019-dataset') # Datasets is stored in ../input/
# Reading the csv files with pandas
deaths_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")
recoverd_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
confirmed_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
us_confirmed_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed_US.csv")
us_death_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths_US.csv")


deaths_data
recoverd_data
confirmed_data
us_confirmed_data
us_death_data
countries_death = deaths_data['Country/Region']
countries_cured = recoverd_data['Country/Region']
countries_confirmed = confirmed_data['Country/Region']
us_provinces_confirmed = us_confirmed_data['Province_State']
us_provinces_death = us_confirmed_data['Province_State']
unique_countries = [] 
for i in countries_death:
    if i  not in unique_countries:
        unique_countries.append(i)
        
print("DATASET CONTAINS INFORMATION ABOUT ",len(unique_countries)," COUNTRIES")
unique_provinces = []
for i in us_provinces_death:
    if i not in unique_provinces:
        unique_provinces.append(i)
        
print("DATASET CONTAINS INFORMATION ABOUT ",len(unique_provinces)," PROVINCES")

dates = list(deaths_data.keys())[4:]
dates_us_provinces = list(us_death_data.keys())[12:]
def get_data(name_of_country, datatype = 'death'): # Defining a function to get data based on country name
    
    if datatype == 'death':
        country_index = []
        for i in range(len(countries_death)):
            if countries_death[int(i)] == name_of_country:
                country_index.append(int(i))     

        data = np.zeros(len(dates))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates)):
                temp.append(deaths_data[dates[each_date_index]][i])
            data = data + np.asarray(temp)
        
        return data
    
    if datatype == 'recovered':
        country_index = []
        for i in range(len(countries_cured)):
            if countries_cured[int(i)] == name_of_country:
                country_index.append(int(i))     

        data = np.zeros(len(dates))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates)):
                temp.append(recoverd_data[dates[each_date_index]][i])
            data = data + np.asarray(temp)
        
        return data
    
    if datatype == 'confirmed':
        country_index = []
        for i in range(len(countries_confirmed)):
            if countries_confirmed[i] == name_of_country:
                country_index.append(i)     

        data = np.zeros(len(dates))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates)):
                temp.append(confirmed_data[dates[int(each_date_index)]][int(i)])
            data = data + np.asarray(temp)
        
        return data
        
def get_data_provinces(name_of_provinces, datatype = 'death'): # Defining a function to get data based on provinces name
    
    if datatype == 'death':
        country_index = []
        for i in range(len(us_provinces_death)):
            if us_provinces_death[int(i)] == name_of_provinces:
                country_index.append(int(i))     

        data = np.zeros(len(dates_us_provinces))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates_us_provinces)):
                temp.append(us_death_data[dates_us_provinces[each_date_index]][i])
            data = data + np.asarray(temp)
        
        return data
    

    
    if datatype == 'confirmed':
        country_index = []
        for i in range(len(us_provinces_death)):
            if us_provinces_confirmed[i] == name_of_provinces:
                country_index.append(i)     

        data = np.zeros(len(dates_us_provinces))
        for i in country_index:
            temp = []
            for each_date_index in range(len(dates_us_provinces)):
                temp.append(us_confirmed_data[dates_us_provinces[int(each_date_index)]][int(i)])
            data = data + np.asarray(temp)
        
        return data
        
plt.rc('figure', figsize=(15, 7)) # Setting graph size
death_global = np.zeros(len(dates))
recovered_global = np.zeros(len(dates))
confirmed_global = np.zeros(len(dates))

for country_names in unique_countries:
    # Adding the data
    death_global = get_data(country_names,'death') + death_global
    recovered_global = get_data(country_names,'recovered') + recovered_global
    confirmed_global = get_data(country_names,'confirmed') + confirmed_global
    
# Plotting the graph    
sns.lineplot(range(len(dates)),death_global , label = 'death')
sns.lineplot(range(len(dates)),recovered_global , label = 'recovered')
sns.lineplot(range(len(dates)),confirmed_global , label = 'confirmed')
    
# Shding the area
plt.fill_between(range(len(dates)),confirmed_global, color="b", alpha=0.4)   
plt.fill_between(range(len(dates)),recovered_global, color="g", alpha=0.5)
plt.fill_between(range(len(dates)),death_global, color="r", alpha=0.5)

death_global = list(death_global)
recovered_global = list(recovered_global)
confirmed_global = list(confirmed_global)

max_death = max(death_global)
date_of_max_death = death_global.index(max(death_global))
        
max_recovery = max(recovered_global)
date_of_max_recovery = recovered_global.index(max(recovered_global))
        
max_confirmation = max(confirmed_global)
date_of_max_confirmation = confirmed_global.index(max(confirmed_global))
 
# Highlighting the point at maxima
plt.scatter(date_of_max_death,max_death,color = 'r')
plt.scatter(date_of_max_recovery,max_recovery,color = 'g')
plt.scatter(date_of_max_confirmation,max_confirmation,color = 'b')

# Plotting value at maxima
plt.text(date_of_max_death, max_death,str(int(max_death)) , fontsize=12 , color = 'r')
plt.text(date_of_max_recovery, max_recovery,str(int(max_recovery)) , fontsize=12 , color = 'g')
plt.text(date_of_max_confirmation, max_confirmation,str(int(max_confirmation)) , fontsize=12 , color = 'b')

        
plt.legend(loc=2)
plt.title("Global Count")    
plt.xlabel("Time")
plt.ylabel("Count")

plt.show()

print("TOTAL DEATHS: ",sum(death_global))
print("TOTAL RECOVERED PATIENTS: ",sum(recovered_global))
print("TOTAL CONFIRMED CASES: ",sum(confirmed_global))
plt.rc('figure', figsize=(15, 7))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])


ax.bar(range(len(dates)),confirmed_global , label = 'confirmed',color = 'b')
ax.bar(range(len(dates)),recovered_global , label = 'recovered' , color = 'g')
ax.bar(range(len(dates)),death_global , label = 'death' , color = 'r')


ax.set_xlabel("Time")
ax.set_ylabel("Count")
ax.set_title('Global Count')
plt.legend(loc = 2)
plt.show()
plt.rc('figure', figsize=(15, 7))

for country_names in unique_countries:
    sns.lineplot(range(len(dates)),get_data(country_names,'death'))

        
plt.title("Time vs Death")    
plt.xlabel("Time")
plt.ylabel("Deaths")

plt.show()


plt.rc('figure', figsize=(15, 7))

for country_names in unique_countries:
    sns.lineplot(range(len(dates)),get_data(country_names,'recovered'))


        
plt.title("Time vs Recovery")    
plt.xlabel("Time")
plt.ylabel("Recovered patients")

plt.show()
plt.rc('figure', figsize=(15, 7))

for country_names in unique_countries:
    sns.lineplot(range(len(dates)),get_data(country_names,'confirmed'))


        
plt.title("Time vs Confirmed Cases")    
plt.xlabel("Time")
plt.ylabel("Confirmed")

plt.show()
total_death = []
total_confirmed = []
total_recovered = []

from more_itertools import sort_together


for country_names in unique_countries:
    total_death.append(int(sum(get_data(country_names,'death'))))
    
for country_names in unique_countries:
    total_confirmed.append(int(sum(get_data(country_names , 'confirmed'))))

for country_names in unique_countries:
    total_recovered.append(int(sum(get_data(country_names , 'recovered'))))
    
# Sorting the values based on number of deaths , recovery ,confirmed cases

total_death_sorted , countries_sorted_death = tuple(sort_together([total_death,unique_countries]))
total_recovered_sorted , countries_sorted_recovered = tuple(sort_together([total_recovered,unique_countries]))
total_confirmed_sorted , countries_sorted_confirmed = tuple(sort_together([total_confirmed,unique_countries]))

print("COUNTRIES BASED ON NUMBER OF DEATHS:")
for i in countries_sorted_death[:-11:-1]:
    print(" "+i)

print("----------------------------------------")

print("COUNTRIES BASED ON NUMBER OF PATIENTS RECOVERED:")
for i in countries_sorted_recovered[:-11:-1]:
    print(" "+i)
    
print("----------------------------------------")

print("COUNTRIES BASED ON NUMBER OF CONFIRMED CASES:")
for i in countries_sorted_confirmed[:-11:-1]:
    print(" "+i)
    
print("----------------------------------------")
death_each_country = pd.DataFrame(zip(countries_sorted_death,total_death_sorted) , columns = ['Country','Death'])
recovered_each_country = pd.DataFrame(zip(countries_sorted_recovered,total_recovered_sorted) , columns = ['Country','Recovered'])
confirmed_each_country = pd.DataFrame(zip(countries_sorted_confirmed,total_confirmed_sorted) , columns = ['Country','Confirmed'])

display(death_each_country[:-11:-1])

for country_names in countries_sorted_death[:-11:-1]:
    sns.lineplot(range(len(dates)),get_data(country_names,'death'),label = country_names)
        
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Death')
plt.legend(loc = 2)
plt.show()

# Plotting pie chart
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df'] # Choosing light colors
labels = list(countries_sorted_death[:-11:-1]) # Taking only 10 countries
labels.append('Others') 
sizes = list(total_death_sorted[:-11:-1])
sizes.append(sum(total_death_sorted[:-11])) #Adding remaining values
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Death Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()
display(recovered_each_country[:-11:-1])

for country_names in countries_sorted_recovered[:-11:-1]:
    sns.lineplot(range(len(dates)),get_data(country_names,'recovered'),label = country_names)
        
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Recovered')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(countries_sorted_recovered[:-11:-1])
labels.append('Others')
sizes = list(total_recovered_sorted[:-11:-1])
sizes.append(sum(total_recovered_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Recovered Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()
display(confirmed_each_country[:-11:-1])


for country_names in countries_sorted_confirmed[:-11:-1]:
    sns.lineplot(range(len(dates)),get_data(country_names,'confirmed'),label = country_names)

  
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Confirmed')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(countries_sorted_confirmed[:-11:-1])
labels.append('Others')
sizes = list(total_confirmed_sorted[:-11:-1])
sizes.append(sum(total_confirmed_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Confirmed Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()
display(confirmed_each_country[:-11:-1])


for country_names in countries_sorted_confirmed[:-11:-1]:
    sns.lineplot(range(len(dates)),get_data(country_names,'confirmed'),label = country_names)

  
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Confirmed')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(countries_sorted_confirmed[:-11:-1])
labels.append('Others')
sizes = list(total_confirmed_sorted[:-11:-1])
sizes.append(sum(total_confirmed_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Confirmed Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()
all_countries_data = pd.DataFrame(list(zip(unique_countries,total_death,total_recovered,total_confirmed)),columns = ['Country','Death','Recovered','Confirmed'])
# Displaying a table with values highlighted relative to other values with help of gradients

death_color = sns.light_palette("red", as_cmap=True) # Choosing gradient color pallete type
recovered_color = sns.light_palette("green", as_cmap=True)
confirmed_color = sns.light_palette("blue", as_cmap=True)

(all_countries_data.style
  .background_gradient(cmap=death_color, subset=['Death']) # Applying gradient
  .background_gradient(cmap=recovered_color, subset=['Recovered'])
  .background_gradient(cmap=confirmed_color, subset=['Confirmed'])

)

def country_info(name,graph_type): # Defining function to plot the graph for a specific country
    
    if graph_type == 'bar':
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
 
        ax.bar(dates,get_data(name , 'confirmed'),label = 'confirmed' , color = 'b')
        ax.bar(dates,get_data(name , 'recovered'),label = 'recovered' , color = 'g')
        ax.bar(dates,get_data(name,'death'),label = 'Death' , color = 'r')


        ax.set_xticklabels([])
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.set_title(name)
        plt.legend(loc = 2)
        plt.show()
        
    if graph_type == 'line':
        sns.lineplot(range(len(dates)),get_data(name,'confirmed'),label = 'confirmed',color = 'b')
        sns.lineplot(range(len(dates)),get_data(name,'recovered'),label = 'recovered' , color = 'g')
        sns.lineplot(range(len(dates)),get_data(name,'death'),label = 'death' , color = 'r')
        
        plt.fill_between(range(len(dates)),get_data(name,'confirmed'), color="b", alpha=0.2)
        plt.fill_between(range(len(dates)),get_data(name,'recovered'), color="g", alpha=0.2)
        plt.fill_between(range(len(dates)),get_data(name,'death'), color="r", alpha=0.2)
        
        max_death = max(list(get_data(name,'death')))
        date_of_max_death = list(get_data(name,'death')).index(max(list(get_data(name,'death'))))
        
        max_recovery = max(list(get_data(name,'recovered')))
        date_of_max_recovery = list(get_data(name,'recovered')).index(max(list(get_data(name,'recovered'))))
        
        max_confirmation = max(list(get_data(name,'confirmed')))
        date_of_max_confirmation = list(get_data(name,'confirmed')).index(max(list(get_data(name,'confirmed'))))
        
        plt.scatter(date_of_max_death,max_death,color = 'r')
        plt.scatter(date_of_max_recovery,max_recovery,color = 'g')
        plt.scatter(date_of_max_confirmation,max_confirmation,color = 'b')
        
        plt.text(date_of_max_death, max_death,str(int(max_death)) , fontsize=12 , color = 'r')
        plt.text(date_of_max_recovery, max_recovery,str(int(max_recovery)) , fontsize=12 , color = 'g')
        plt.text(date_of_max_confirmation, max_confirmation,str(int(max_confirmation)) , fontsize=12 , color = 'b')
        
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(name)
        plt.legend(loc = 2)
        plt.show()

        
country_info('US','line')
country_info('US','bar')



print("TOTAL DEATH IN US:",int(sum(get_data('US','death'))))
print("TOTAL PATIENTS RECOVERED IN US:",int(sum(get_data('US','recovered'))))
print("TOTAL CONFIRMED CASES IN US:",int(sum(get_data('US','confirmed'))))

country_info('China','line')
country_info('China','bar')

print("TOTAL DEATH IN China:",int(sum(get_data('China','death'))))
print("TOTAL PATIENTS RECOVERED IN China:",int(sum(get_data('China','recovered'))))
print("TOTAL CONFIRMED CASES IN China:",int(sum(get_data('China','confirmed'))))
country_info('Italy','line')
country_info('Italy','bar')

print("TOTAL DEATH IN Italy:",int(sum(get_data('Italy','death'))))
print("TOTAL PATIENTS RECOVERED IN Italy:",int(sum(get_data('Italy','recovered'))))
print("TOTAL CONFIRMED CASES IN Italy:",int(sum(get_data('Italy','confirmed'))))
country_info('Spain','line')
country_info('Spain','bar')

print("TOTAL DEATH IN Spain:",int(sum(get_data('Spain','death'))))
print("TOTAL PATIENTS RECOVERED IN Spain:",int(sum(get_data('Spain','recovered'))))
print("TOTAL CONFIRMED CASES IN Spain:",int(sum(get_data('Spain','confirmed'))))
country_info('Germany','line')
country_info('Germany','bar')

print("TOTAL DEATH IN Germany:",int(sum(get_data('Germany','death'))))
print("TOTAL PATIENTS RECOVERED IN Germany:",int(sum(get_data('Germany','recovered'))))
print("TOTAL CONFIRMED CASES IN Germany:",int(sum(get_data('Germany','confirmed'))))
country_info('France','line')
country_info('France','bar')

print("TOTAL DEATH IN France:",int(sum(get_data('France','death'))))
print("TOTAL PATIENTS RECOVERED IN France:",int(sum(get_data('France','recovered'))))
print("TOTAL CONFIRMED CASES IN France:",int(sum(get_data('France','confirmed'))))
country_info('United Kingdom','line')
country_info('United Kingdom','bar')

print("TOTAL DEATH IN United Kingdom:",int(sum(get_data('United Kingdom','death'))))
print("TOTAL PATIENTS RECOVERED IN United Kingdom:",int(sum(get_data('United Kingdom','recovered'))))
print("TOTAL CONFIRMED CASES IN United Kingdom:",int(sum(get_data('United Kingdom','confirmed'))))
country_info('Iran','line')
country_info('Iran','bar')

print("TOTAL DEATH IN Iran:",int(sum(get_data('Iran','death'))))
print("TOTAL PATIENTS RECOVERED IN Iran:",int(sum(get_data('Iran','recovered'))))
print("TOTAL CONFIRMED CASES IN Iran:",int(sum(get_data('Iran','confirmed'))))
country_info('Turkey','line')
country_info('Turkey','bar')

print("TOTAL DEATH IN Turkey:",int(sum(get_data('Turkey','death'))))
print("TOTAL PATIENTS RECOVERED IN Turkey:",int(sum(get_data('Turkey','recovered'))))
print("TOTAL CONFIRMED CASES IN Turkey:",int(sum(get_data('Turkey','confirmed'))))
country_info('Belgium','line')
country_info('Belgium','bar')

print("TOTAL DEATH IN Belgium:",int(sum(get_data('Belgium','death'))))
print("TOTAL PATIENTS RECOVERED IN Belgium:",int(sum(get_data('Belgium','recovered'))))
print("TOTAL CONFIRMED CASES IN Belgium:",int(sum(get_data('Belgium','confirmed'))))
country_info('India','line')
country_info('India','bar')

print("TOTAL DEATH IN India:",int(sum(get_data('India','death'))))
print("TOTAL PATIENTS RECOVERED IN India:",int(sum(get_data('India','recovered'))))
print("TOTAL CONFIRMED CASES IN India:",int(sum(get_data('India','confirmed'))))
plt.rc('figure', figsize=(15, 7))

for country_names in unique_provinces:
    sns.lineplot(range(len(dates_us_provinces)),get_data_provinces(country_names,'confirmed'))


        
plt.title("Time vs Confirmed")    
plt.xlabel("Time")
plt.ylabel("Confirmed count")

plt.show()
plt.rc('figure', figsize=(15, 7))

for country_names in unique_provinces:
    sns.lineplot(range(len(dates_us_provinces)),get_data_provinces(country_names,'death'))

        
plt.title("Time vs Death")    
plt.xlabel("Time")
plt.ylabel("Deaths")

plt.show()


total_death = []
total_confirmed = []
total_recovered = []

from more_itertools import sort_together


for country_names in unique_provinces:
    total_death.append(int(sum(get_data_provinces(country_names,'death'))))
    
for country_names in unique_provinces:
    total_confirmed.append(int(sum(get_data_provinces(country_names , 'confirmed'))))

    
# Sorting the values based on number of deaths , recovery ,confirmed cases

total_provinces_death_sorted , provinces_sorted_death = tuple(sort_together([total_death,unique_provinces]))
total_provinces_confirmed_sorted , provinces_sorted_confirmed = tuple(sort_together([total_confirmed,unique_provinces]))

print("PROVINCES BASED ON NUMBER OF DEATHS:")
for i in provinces_sorted_death[:-11:-1]:
    print(" "+i)

print("----------------------------------------")



print("PROVINCES BASED ON NUMBER OF CONFIRMED CASES:")
for i in provinces_sorted_confirmed[:-11:-1]:
    print(" "+i)
    
print("----------------------------------------")
provinces_sorted_death = pd.DataFrame(zip(provinces_sorted_death,total_provinces_death_sorted) , columns = ['Provinces','Death'])
provinces_sorted_confirmed = pd.DataFrame(zip(provinces_sorted_confirmed,total_provinces_confirmed_sorted) , columns = ['Provinces','Confirmed'])

display(provinces_sorted_death[:-11:-1])

for country_names in provinces_sorted_death['Provinces'][:-11:-1]:
    sns.lineplot(range(len(dates_us_provinces)),get_data_provinces(country_names,'death'),label = country_names)
        
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Deaths')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(provinces_sorted_death['Provinces'][:-11:-1])
labels.append('Others')
sizes = list(total_provinces_death_sorted[:-11:-1])
sizes.append(sum(total_provinces_death_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Deaths Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()
display(provinces_sorted_confirmed[:-11:-1])

for country_names in provinces_sorted_confirmed['Provinces'][:-11:-1]:
    sns.lineplot(range(len(dates_us_provinces)),get_data_provinces(country_names,'confirmed'),label = country_names)
        
plt.xlabel("Time")
plt.ylabel("Count")
plt.title('Confirmed Cases')
plt.legend(loc = 2)
plt.show()


colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#77d8d8','#4cbbb9','#efa8e4','#f2ed6f','#fb7b6b','#d7385e','#be79df']
labels = list(provinces_sorted_confirmed['Provinces'][:-11:-1])
labels.append('Others')
sizes = list(total_provinces_confirmed_sorted[:-11:-1])
sizes.append(sum(total_provinces_confirmed_sorted[:-11]))
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
         startangle=90 , colors = colors)
plt.legend(loc = 2)
plt.title('Confirmed Percentage')
ax1.axis('equal')  
plt.tight_layout()
plt.show()
# Displaying a table with values highlighted relative to other values with help of gradients
all_countries_data = pd.DataFrame(list(zip(unique_provinces,total_death,total_confirmed)),columns = ['Provinces','Death','Confirmed'])

death_color = sns.light_palette("red", as_cmap=True) # Choosing gradient color pallete type
confirmed_color = sns.light_palette("blue", as_cmap=True)

(all_countries_data.style
  .background_gradient(cmap=death_color, subset=['Death']) # Applying gradient
  .background_gradient(cmap=confirmed_color, subset=['Confirmed'])

)

import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv1D,MaxPooling1D,Dropout
INPUT_SIZE = 10
TARGET_SIZE = 1
PADDING = 40 # Most of the countries didn't have any confirmed cases in first 40 days (approx)
COUNTRIES_N = 50
X = []
Y=[]

for country_name in countries_sorted_death[:-1*COUNTRIES_N-1:-1]:

    a = get_data(country_name,'death')[PADDING:]
    b = get_data(country_name,'recovered')[PADDING:]
    c = get_data(country_name,'confirmed')[PADDING:]
    
    a = np.asarray(a).reshape(a.shape[0],1)
    b = np.asarray(b).reshape(1,b.shape[0])
    c = np.asarray(c).reshape(1,c.shape[0])


    for i in range(len(dates)-(INPUT_SIZE+TARGET_SIZE)-PADDING):

        temp = []
        x = np.concatenate((np.concatenate((a[i:i+INPUT_SIZE], b[0][i:i+INPUT_SIZE].reshape(1,INPUT_SIZE).T), axis=1),c[0][i:i+INPUT_SIZE].reshape(1,INPUT_SIZE).T),axis = 1)
        X.append(x)
        temp.append(a[i+INPUT_SIZE])
        temp.append(b[0][i+INPUT_SIZE])
        temp.append(c[0][i+INPUT_SIZE])
        
        Y.append(temp)


X = np.asarray(X)
Y = np.asarray(Y) 
print("Input: ",X[0])
print("Output: ",Y[0])
print(X.shape,Y.shape)
def plot_(n):
    death_ = []
    recovered_ = []
    confirmed_ = []
    for i in X[n]:
        death_.append(i[0])
        recovered_.append(i[1])
        confirmed_.append(i[2])
        
            
    plt.plot(death_,color = 'r')
    plt.plot(recovered_ , color = 'g')
    plt.plot(confirmed_ , color = 'b')
    
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+TARGET_SIZE),[death_[-1],Y[n][0]],color = 'r',linestyle = 'dashed')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+TARGET_SIZE),[recovered_[-1],Y[n][1]],color = 'g',linestyle = 'dashed')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+TARGET_SIZE),[confirmed_[-1],Y[n][2]],color = 'b',linestyle = 'dashed')
    
    plt.legend(labels = ['death','recovery','confirmed','actual death','actual confirmed cases','actual recovered patients'])
    plt.show()
import random
plot_(random.randrange(len(X)))

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(INPUT_SIZE, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3))

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mse',metrics=['mae', 'acc'])


model.summary()
history = model.fit(X, Y, epochs=20, verbose=2, validation_split=0.05,shuffle=True)
    
plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_loss")
plt.legend(loc="upper right")
plt.show()

plt.plot(history.history['acc'], label="accuracy")
plt.plot(history.history['val_acc'], label="val_accuracy")
plt.legend(loc="upper right")
plt.show()

def test_random(n):
    death_ = []
    recovered_ = []
    confirmed_ = []
    for i in X[n]:
        death_.append(i[0])
        recovered_.append(i[1])
        confirmed_.append(i[2])
        
            
    plt.plot(range(INPUT_SIZE),death_,color = 'r')
    plt.plot(range(INPUT_SIZE),recovered_ , color = 'g')
    plt.plot(range(INPUT_SIZE),confirmed_ , color = 'b')

    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[death_[-1],Y[n][0]],color = 'm')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[recovered_[-1],Y[n][1]],color = 'm')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[confirmed_[-1],Y[n][2]],color = 'm')

    out = model.predict(X[n].reshape((1,INPUT_SIZE,3)))

    print("PREDICTED: "+str(out[0][0]),"ACUTAL: "+str(Y[n][0][0]))
    print("PREDICTED: "+str(out[0][1]),"ACUTAL: "+str(Y[n][1]))
    print("PREDICTED: "+str(out[0][2]),"ACUTAL: "+str(Y[n][2]))
    
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[death_[-1],out[0][0]],color = 'r',linestyle = 'dashed')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[recovered_[-1],out[0][1]],color = 'g',linestyle = 'dashed')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+1),[confirmed_[-1],out[0][2]],color = 'b',linestyle = 'dashed')

    plt.legend(labels = ['death','crecovery','confirmed','actual death','actual confirmed cases','actual recovered patients','predicted death','predicted confirmed cases','predicted recovered patients'])
    plt.show()
for i in range(10):
    n = random.randrange(len(X))
    test_random(n)
def plot_pred_n_days(country_name,n_days,hide=False):
    predicted_d = []
    predicted_c = []
    predicted_r = []

    X_d = get_data(country_name,'death')[-INPUT_SIZE:]
    X_r = get_data(country_name,'recovered')[-INPUT_SIZE:]
    X_c = get_data(country_name,'confirmed')[-INPUT_SIZE:]
    
    plt.plot(X_d,color = '#ff6252',label = 'death')
    plt.plot(X_r,color = '#00e639',label = 'recovered')
    plt.plot(X_c,color = '#8a4eff',label = 'confirmed')
    
    X_input = []
    
    X_d = np.asarray(X_d).reshape(X_d.shape[0],1)
    X_r = np.asarray(X_r).reshape(1,X_r.shape[0])
    X_c = np.asarray(X_c).reshape(1,X_c.shape[0])
    x = np.concatenate((np.concatenate((X_d, X_r.reshape(1,INPUT_SIZE).T), axis=1),X_c.reshape(1,INPUT_SIZE).T),axis = 1)
    X_input.append(x)
    X_r = np.asarray(X_r).reshape(X_r.shape[1],1)
    X_c = np.asarray(X_c).reshape(X_c.shape[1],1)

    for i in range(n_days):
        predicted = model.predict(np.asarray(X_input).reshape(1,INPUT_SIZE,3))[0]
        predicted_d.append(predicted[0])
        predicted_r.append(predicted[1])
        predicted_c.append(predicted[2])
        X_input = np.concatenate((np.asarray(X_input).reshape(10,3), predicted.reshape(3,1).T), axis=0)[1:]
        
        if hide == False:
            plt.scatter(INPUT_SIZE+i,predicted[0],color = '#ff6252')
            plt.scatter(INPUT_SIZE+i,predicted[1],color = '#00e639')
            plt.scatter(INPUT_SIZE+i,predicted[2],color = '#8a4eff')
    
            plt.text(INPUT_SIZE+i, predicted[0]*102100,str(int(predicted[0])) , fontsize=12 , color = '#ff6252')
            plt.text(INPUT_SIZE+i, predicted[1]*102/100,str(int(predicted[1])) , fontsize=12 , color = '#00e639')
            plt.text(INPUT_SIZE+i, predicted[2]*102/100,str(int(predicted[2])) , fontsize=12 , color = '#8a4eff')
            
    if hide == True:
        plt.scatter(INPUT_SIZE,predicted_d[0],color = '#ff6252')
        plt.scatter(INPUT_SIZE,predicted_r[0],color = '#00e639')
        plt.scatter(INPUT_SIZE,predicted_c[0],color = '#8a4eff')
    
        plt.text(INPUT_SIZE, predicted_d[0]*102/100,str(int(predicted_d[0])) , fontsize=12 , color = '#ff6252')
        plt.text(INPUT_SIZE, predicted_r[0]*102/100,str(int(predicted_r[0])) , fontsize=12 , color = '#00e639')
        plt.text(INPUT_SIZE, predicted_c[0]*102/100,str(int(predicted_c[0])) , fontsize=12 , color = '#8a4eff')
        
        plt.scatter(INPUT_SIZE+n_days-1,predicted_d[-1],color = '#ff6252')
        plt.scatter(INPUT_SIZE+n_days-1,predicted_r[-1],color = '#00e639')
        plt.scatter(INPUT_SIZE+n_days-1,predicted_c[-1],color = '#8a4eff')
    
        plt.text(INPUT_SIZE+n_days-1, predicted_d[-1]*102/100,str(int(predicted_d[-1])) , fontsize=12 , color = '#ff6252')
        plt.text(INPUT_SIZE+n_days-1, predicted_r[-1]*102/100,str(int(predicted_r[-1])) , fontsize=12 , color = '#00e639')
        plt.text(INPUT_SIZE+n_days-1, predicted_c[-1]*102/100,str(int(predicted_c[-1])) , fontsize=12 , color = '#8a4eff')
    
    maximum_d_index = predicted_d.index(max(predicted_d))
    maximum_r_index = predicted_r.index(max(predicted_r))
    maximum_c_index = predicted_c.index(max(predicted_c))
    
    plt.scatter(INPUT_SIZE+maximum_d_index,predicted_d[maximum_d_index],color = 'r',label = 'maximum death')
    plt.scatter(INPUT_SIZE+maximum_r_index,predicted_r[maximum_r_index],color = 'g',label = 'maximum recovered patients')
    plt.scatter(INPUT_SIZE+maximum_c_index,predicted_c[maximum_c_index],color = 'b',label = 'maximum confirmed cases')
    
    plt.text(INPUT_SIZE+maximum_d_index, predicted_d[maximum_d_index]*102/100,str(int(predicted_d[maximum_d_index])) , fontsize=12 , color = 'r')
    plt.text(INPUT_SIZE+maximum_r_index, predicted_r[maximum_r_index]*102/100,str(int(predicted_r[maximum_r_index])) , fontsize=12 , color = 'g')
    plt.text(INPUT_SIZE+maximum_c_index, predicted_c[maximum_c_index]*102/100,str(int(predicted_c[maximum_c_index])) , fontsize=12 , color = 'b')
    

    
    predicted_d.insert(0,X_d[-1])
    predicted_r.insert(0,X_r[-1])
    predicted_c.insert(0,X_c[-1])
    
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+n_days),predicted_d,linestyle='dashed',color = '#ff6252',label = 'predicted death')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+n_days),predicted_r,linestyle='dashed',color = '#00e639',label = 'predicted recovered')
    plt.plot(range(INPUT_SIZE-1,INPUT_SIZE+n_days),predicted_c,linestyle='dashed',color = '#8a4eff',label = 'predicted confirmed')
    
    plt.legend(loc=2)
    plt.title("Predicted Death,Confirmed Cases and Recovered patients for "+country_name+" for "+str(n_days)+" days")

    plt.show()
    
    print("Total people died in " + str(n_days) + " days :",int(sum(predicted_d)))
    print("Total patients recovered in " + str(n_days) + " days :",int(sum(predicted_r)))
    print("Total confirmed cases in " + str(n_days) + " days :",int(sum(predicted_c)))
    
    print("Maximum Deaths in a day :",int((predicted_d[maximum_d_index])))
    print("Maximum Recovered patients in a day :",int(predicted_d[maximum_r_index]))
    print("Maximum Confired cases in a day :",int(predicted_d[maximum_c_index]))

for country_name in countries_sorted_death[:-15:-1]:
    plot_pred_n_days(country_name,30,hide=True)
plot_pred_n_days('India',30,hide=True)
plot_pred_n_days('US',1000,hide=True)
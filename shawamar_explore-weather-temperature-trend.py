import warnings #to disable warnings
warnings.filterwarnings('ignore') #disables warnings

#import numpy as np 
import pandas as pd #for data processing
import matplotlib.pyplot as plt #for plotting
import seaborn as sns #for plotting
#make plots visible inside the jupyter notebook
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'

from sklearn.linear_model import LinearRegression #calc the trend-data for later plots
import missingno as ms #visualize missing data

#improving the plot - quality (from: http://blog.rtwilson.com/how-to-get-nice-vector-graphics-in-your-exported-pdf-ipython-notebooks/)
from IPython.display import set_matplotlib_formats 
set_matplotlib_formats('png', 'pdf')

#load the datasets
df_city_data = pd.read_csv("../input/city_data.csv")
df_global_data = pd.read_csv("../input/global_data.csv")
df_city_list = pd.read_csv("../input/city_list.csv")
#checking the info of the city_data dataframe
df_city_data.info()
#checking the head of the city_data dataframe
df_city_data.head()
#plot missing values
ms.matrix(df_city_data)
#caluclate missing values in the "avg_temp" column
missing_data = df_city_data["avg_temp"].isna()
sum(missing_data)
#calculate the number of affected unique city names
len(df_city_data[missing_data]["city"].unique())
#checking the info of the global_data dataframe
df_global_data.info()
#checking the head of the global_data dataframe
df_global_data.head()
#calculate the mininmal & maximal values of the "year" and "avg_temp" column in the global_data dataframe
print("Begining Year:",df_global_data["year"].min())
print("Ending Year:",df_global_data["year"].max())
print("Min. Temp.:",df_global_data["avg_temp"].min())
print("Max. Temp.:",df_global_data["avg_temp"].max())
#select the cities from germany out of the city_list dataframe
df_city_list[df_city_list["country"] == "India"]
#create a new reference to the city_data dataframe for the city "Patna"
df_patna = df_city_data[df_city_data["city"] == "Patna"]
#checking the head of the Patna dataframe
df_patna.head()
#show the missing values in the Patna dataframe
missing_data=df_patna[df_patna["avg_temp"].isna()]
print(missing_data)
missing_years=(missing_data["year"])
print("Missing Years: ",missing_years)
print("No. of Missing Values: ",sum(df_patna["avg_temp"].isna()))
# droping data with missing values
df_patna.dropna()
df_patna.head()
print(df_global_data[df_global_data["avg_temp"].isna()])
print("No. of Missing Values:",sum(df_global_data["avg_temp"].isna()))
global_avg_temp=df_global_data["avg_temp"]
patna_avg_temp=df_patna["avg_temp"]
#Local Data is as same as Cairo
plt.plot(df_global_data['year'],global_avg_temp,label='Global')
plt.plot(df_patna['year'],patna_avg_temp,label='Patna')
plt.legend()
plt.xlabel("Years")
plt.ylabel("Temperature (°C)") 
plt.title("Patna City Average Temperature")
plt.show()
#create a copy of the city_data dataframe and the global_data dataframe for our needs
#df_patna = df_city_data[(df_city_data["city"] == "patna")].copy()
df_year=df_patna['year']
df_global_data_plot = df_global_data.merge(df_year, left_on='year',right_on='year')
print(df_global_data_plot)
#check if both dataframe have the same length
len(df_global_data_plot) == len(df_patna)
plt.plot(df_global_data_plot['year'],df_global_data_plot['avg_temp'],label='Global')
plt.plot(df_patna['year'],patna_avg_temp,label='Patna')
plt.legend()
plt.xlabel("Years")
plt.ylabel("Temperature (°C)") 
plt.title("Patna City Average Temperature")
plt.show()
#setting the window for the rolling mean
avg_num = 10

#calculate the rolling mean 
df_patna["rollingAverage"] = df_patna["avg_temp"].rolling(window = avg_num).mean()

#reset the index of the Patnain dataframe
df_patna.reset_index(inplace = True, drop = True)

#calculate the rolling mean for the global_data_plot dataframe
df_global_data_plot["rollingAverage"] = df_global_data_plot["avg_temp"].rolling(window = avg_num).mean()

#show the rows 10 to 20 of the Patna dataframe
df_patna
#show the rows 10 to 20 of the global_data_plot dataframe
df_global_data_plot[10:20]
with sns.plotting_context("notebook"):
    #create a matplotlib figure and axes
    fig, ax = plt.subplots(figsize = (10,6))

    #plot the global data with a lineplot
    sns.lineplot(x = "year", y = "avg_temp", data = df_global_data_plot, label = "Global average", alpha = 0.5,
                 color = "grey", lw = 1.2)
    sns.lineplot(x = "year", y = "rollingAverage", data = df_global_data_plot, label = "Global rolling average",
                 color = "steelblue", lw = 2)

    #show the plot-labels in the legend
    ax.legend()
    #set the y label of the plot
    ax.set_ylabel("Temperature in °C")
    #set the x label of the plot
    ax.set_xlabel("Year")
    #set the title of the plot
    ax.set_title("Temperature time-series GLOBAL")
    #disable the right and top spine for better look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #tighten the plot layout
    plt.tight_layout()

    #show the plot (basically not necessary for jupyter but I always put it at the end)
    plt.show()
#calculate the rolling standard deviaton for the global_data dataframe and plot it
df_global_data_plot["avg_temp"].rolling(window = avg_num).std().plot()

#set the title for the plot
plt.title("Rolling standard deviation for the global 'avg_temp' data")
def prepare_df(cities, avg_num = 10):
    """
    INPUT: cities - list of strings, city names
           avg_num - integer, window for the rolling mean

    OUTPUT: dictionary with preprocessed dataframes and citynames as key
    """    
    #create empty dict
    results = {}    
    
    #loop for all city names
    for x in cities:
        #selecting dataframe for the chosen city and years bigger or equal 1750 and smaller 2014 (for consistency)
        df_x = df_city_data[(df_city_data["city"] == x) & (df_city_data["year"] >= 1750) &
                            (df_city_data["year"] < 2014)].copy()
        
        #calculate rolling average with window = avg_num
        df_x["rollingAverage"] = df_x["avg_temp"].rolling(window = avg_num).mean()
        
        #calculating the trend
        try:
            #loading the model
            trend_model = LinearRegression()
            
            #define x and y values
            X = df_x[["year"]]
            y = df_x[["avg_temp"]]
            
            #fit the values to the model
            trend_model.fit(X, y)
            
            #predict the values with the X data to get the trend line and add it to the dataframe
            df_x["trend"] = trend_model.predict(X)
            print(f"{x} slope: {trend_model.coef_}")
                       
        except:
            #in case of NaN Values 
            print("Could not calculate trend.")
        
        finally:
            #reset the index and add the dataframe to the dictionary
            df_x.reset_index(inplace = True, drop = True)
            results[x] = df_x    
    
    for x in results:
        #print the average temperature from 1750 to 2013 for the chosen countries
        print(f'The average temperature in {x} (1750 - 2013) is: {results[x]["avg_temp"].mean():.2f} °C')
        print(f'More {x} values: Max: {results[x]["avg_temp"].max():.2f} °C; Min: {results[x]["avg_temp"].min():.2f} °C \n')
    #copy the global data for years < 2014
    df_global = df_global_data[df_global_data["year"] < 2014].copy()
    
    #caluclating the trend line as described above (I could also do this one time outside this function, but in case I 
    #would had to change the global data somehow I put it in here - also I started with the functionality to choose the timescale.)
    try:
        trend_model = LinearRegression()
        X = df_global["year"]
        trend_model.fit(X = df_global[["year"]], y = df_global[["avg_temp"]])
        trend_plot = trend_model.predict(df_global[["year"]])
        df_global["trend"] = trend_plot
        print(f"Global slope: {trend_model.coef_}")
        
    except:
        print("Error while calculating the global trend.")
        
    finally:
        
        df_global["rollingAverage"] = df_global["avg_temp"].rolling(window = avg_num).mean()    
        results["Global"] = df_global

        print(f'The global average temperature (1750 - 2013) is: {df_global["avg_temp"].mean():.2f} °C')
        print(f'More global values: Max: {df_global["avg_temp"].max():.2f} °C; Min: {df_global["avg_temp"].min():.2f} °C')
        #return the dict
        return results

def plot_results(result, fig_size = (10,6), global_plot = True, opac = 0.7, show_normal = False, trend = False):
    """
    INPUT: result - dictionary with dataframes from the function "prepare_df"
           fig_size - tuple to determine the size of the figure (width, height)
           global_plot - boolean, shows the plot of global data if set to true
           opac - float, sets the opacity for the city - data plots
           show_normal - boolean, plots also the avg_temp in addition to the rolling mean if set to true
           trend - boolean, shows the trend if set to true
    OUTPUT: plot
    """
    with sns.plotting_context("notebook"):
        #creating the figure and axes for the plot
        fig, ax = plt.subplots(figsize = fig_size)    

        #if global_plot = True shows the global data
        if global_plot:        
            sns.lineplot(x = "year", y = "rollingAverage", data = result["Global"], label = "Global", lw = 1.7)  
            #if global_plot = True shows the global trend
            if trend:
                sns.lineplot(x = "year", y = "trend", data = result["Global"], label = "Global trend")

            if show_normal:
                sns.lineplot(x = "year", y = "avg_temp", data = result["Global"], label = f"Global avg_temp", alpha = 0.5, 
                             color = "grey", lw = 1.2)

        #deletes the "Global" key-value pair in the dictionary to not double plot it    
        result.pop('Global', None)

        #plots the graph for every city-key in the dictionary
        for x in result:
            sns.lineplot(x = "year", y = "rollingAverage", data = result[x], label = x, alpha = opac)
            if show_normal:
                sns.lineplot(x = "year", y = "avg_temp", data = result[x], label = f"{x} avg_temp", alpha = 0.5, 
                             color = "grey", lw = 1.2)
            if trend:
                sns.lineplot(x = "year", y = "trend", data = result[x], label = f"{x} trend")

        #loads the legend
        ax.legend()

        #set x and y labels and the title
        ax.set_ylabel("Rolling average temperature in °C")
        ax.set_xlabel("Year")
        ax.set_title("Temperature time-series")

        #deactivate right and top spine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()

        #shows the plot
        plt.show()  
#creating an empty list
cities = []
#calling the newly created functions
plot_results(prepare_df(cities), global_plot=True, show_normal=True, opac = 1, trend = True)
cities = ["Patna"]
plot_results(prepare_df(cities), global_plot=False, show_normal=True, opac = 1, trend = True)
df_berlin["avg_temp"].rolling(window = avg_num).std().plot()
plt.title("Rolling standard deviation for the Berlin 'avg_temp' data")
cities = ["Berlin"]
plot_results(prepare_df(cities), global_plot=True, show_normal=False, opac = 0.8, trend = True)
#calculating the correlation between the berlin avg_temp and the global avg_temp
df_patna[["avg_temp"]].corrwith(df_global_data_plot["avg_temp"])
cities = "Berlin Hamburg Munich".split()
plot_results(prepare_df(cities), opac = 0.7, trend=True)
#getting the avg_temp data for Berlin and Munich in the year 2013
for x in ["Berlin", "Munich"]:
    print(f'{x}: {df_city_data[(df_city_data["city"] == x) & (df_city_data["year"] == 2013)]["avg_temp"]}')
cities = ["Hamburg"]
plot_results(prepare_df(cities), global_plot=True, show_normal=False, opac = 0.8, trend = True)
cities = ["Berlin", "Hamburg"]
plot_results(prepare_df(cities), global_plot=False, show_normal=False, opac = 1, trend = True)
#calculating the correlation between the Berlin avg_temp and the Hamburg avg_temp
prepare_df(["Berlin"])["Berlin"][["avg_temp"]].corrwith(prepare_df(["Hamburg"])["Hamburg"]["avg_temp"])
cities = ["Munich"]
plot_results(prepare_df(cities), global_plot=True, show_normal=False, opac = 0.8, trend = True)
cities = ["Berlin", "Munich"]
plot_results(prepare_df(cities), global_plot=False, show_normal=False, opac = 1, trend = True)
#calculating the correlation between the Berlin avg_temp and the Munich avg_temp
prepare_df(["Berlin"])["Berlin"][["avg_temp"]].corrwith(prepare_df(["Munich"])["Munich"]["avg_temp"])
#check for duplicates among the city column
cities = df_city_data.groupby(["city", "country"], as_index = False).count()["city"]
cities[cities.duplicated()]
#creating the key by joining the city and country column
df_city_data["key"] = df_city_data[["city", "country"]].apply(lambda x: " - ".join(x), axis=1)
df_city_data.head()
def calc_corr(cities, startYear):
    """
    INPUTS: cities - list of strings; city names
            startYear - integer; year to start calculating the correlation from
    OUTPUTS: dictionary with correlation values and the key as index
    """
    #creating an empty dictionary
    corr = {}    
    
    for x in cities:
        #selecting and copying the correct dataframe given different conditions
        df_x = df_city_data[(df_city_data["key"] == x) & (df_city_data["year"] >= startYear) & (df_city_data["year"] <= 2015)].copy()
        #reset the index to align the index of both dataframes
        df_x.reset_index(inplace = True)
        #calculate the correlation and add the result to the dictionary
        corr[x] = df_x[["avg_temp"]].corrwith(df_global_data["avg_temp"], axis = 0)
        
    #return the dictionary    
    return corr
corr_all = calc_corr(df_city_data["key"].unique(), 1750)
df_corr = pd.DataFrame(corr_all)
df_corr = df_corr.transpose().sort_values("avg_temp", ascending = False)
df_corr.reset_index(inplace = True)
df_corr["city"] = df_corr["index"].apply(lambda x: x.split(" - ")[0])
df_corr["country"] = df_corr["index"].apply(lambda x: x.split(" - ")[1])
df_corr.set_index("city", inplace = True)
df_corr.head()
#choosing the german cities out of the new dictionary
print(df_corr.loc[["Berlin", "Hamburg", "Munich"]]["avg_temp"])
#print(sum(df_city_data[df_city_data["city"] == "Munich"]["avg_temp"].isna()))
df_corr[["avg_temp", "country"]].head(5)
df_corr[["avg_temp", "country"]][df_corr["avg_temp"] < 0.01].head(5)
df_corr[["avg_temp", "country"]].tail(5)
cities = ["Ufa", "Maseru", "Kano"]
plot_results(prepare_df(cities), (13,10), show_normal=True)
cities = ["Ufa"]
plot_results(prepare_df(cities), show_normal=True)
cities = ["Maseru"]
plot_results(prepare_df(cities), show_normal=True)
cities = ["Kano"]
plot_results(prepare_df(cities), show_normal=True)

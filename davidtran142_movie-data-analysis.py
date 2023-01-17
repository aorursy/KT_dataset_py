import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import sqlite3 #import sqlite 3 module
file = "movies.db" #database file

connection = sqlite3.connect(file)

c = connection.cursor() #establish the connection the database
c.execute("drop table if exists movies") #drop the old table if exists

c.execute("""

            create table movies (

            number int,

            ID int primary key,

            Title varchar(255),

            Year int,

            Age char(3),

            IMDb float,

            Rotten_Tomatoes decimal(3,0),

            Netflix int,

            Hulu int,

            Prime_Video int,

            Disney int,

            Type int,

            Directors varchar(255),

            Genres varchar(255),

            Country varchar(255),

            Language varchar(255),

            Runtime int

            )""") #create table and define structure of table

connection.commit()
import csv #import csv module to use csv-related functions
delete_sql = "delete from movies" #delete old records from the table

c.execute(delete_sql)



f = open("/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv", encoding="utf8")

insert_sql = "insert into movies (Number, ID, Title, Year, Age,IMDb, Rotten_Tomatoes, Netflix, Hulu, Prime_Video, Disney, Type, Directors, Genres,Country,Language,Runtime) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)" #insert record into tables

for row in csv.reader(f): 

    c.execute(insert_sql, row)

connection.commit()
update_sql = "update movies set Title= Lower(Title)" #update value for processing

c.execute(update_sql)

connection.commit()
update_sql_netflix1 = "update movies set Netflix= 'Yes' where Netflix = 1" #update value

update_sql_netflix2 = "update movies set Netflix= 'No' where Netflix = 0"

c.execute(update_sql_netflix1)

c.execute(update_sql_netflix2)

connection.commit()
update_sql_hulu1 = "update movies set Hulu = 'Yes' where Hulu = 1" #update value

update_sql_hulu2 = "update movies set Hulu = 'No' where Hulu = 0"

c.execute(update_sql_hulu1)

c.execute(update_sql_hulu2)

connection.commit()
update_sql_primevideo1 = "update movies set prime_video = 'Yes' where prime_video = 1" #update value

update_sql_primevideo2 = "update movies set prime_video = 'No' where prime_video = 0"

c.execute(update_sql_primevideo1)

c.execute(update_sql_primevideo2)

connection.commit()
update_sql_disney1 = "update movies set disney = 'Yes' where disney = 1" #update value

update_sql_disney2 = "update movies set disney = 'No' where disney = 0"

c.execute(update_sql_disney1)

c.execute(update_sql_disney2)

connection.commit()
show_10_sql = "select * from movies limit 10" #show the first 10 movies

c.execute(show_10_sql)

rows = c.fetchall()

for row in rows: 

    print(row)

connection.commit()
movies_name = str(input("Please enter the movies name: ")) #Get input movie name

custom_sql = "select Title, Netflix, Hulu, Prime_Video, Disney from movies where Title = lower('{0}')".format(movies_name) 

#select movies with the input title

c.execute(custom_sql)

rows = c.fetchall()

if not rows: #If there is no result

    if movies_name == '': 

        print("Please enter a name to find") #Ask to input a movie name if it is blank

    else:

        print("Sorry, we don't have the movie name {} in the system".format(movies_name.capitalize())) 

        print("Here are some related results, try searching for these movies:")

        #Suggest 5 movies close to the movie name

        select_alternate = "select Title from movies where Title Like '%{}%' limit 5".format(movies_name)

        c.execute(select_alternate)

        records = c.fetchall()

        if not records:

            print("Sorry, we can't find any movies similar to the name to input. Please input another movie name")

        else:

            for record in records:

                print("- {}".format(record[0].title()))

for row in rows: #If the movie exists, show the information of that movie

    print("The movie name is {}".format(row[0].title()))

    print("Available on Netflix: {}".format(row[1]))

    print("Available on Hulu: {}".format(row[2]))

    print("Available on Prime Video: {}".format(row[3]))

    print("Available on Disney: {}".format(row[4]))

connection.commit()
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from scipy.stats import linregress

import statsmodels.formula.api as smf 
data = pd.read_csv("../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv", index_col=0)

data.head(10)

# data[['Title','Year']]

# data.loc[[2],["Title","Year","Netflix"]]

# plt.hist(data[['year']], bins = 3)

# plt.show()
data.describe() #Describe the data
data.pivot_table(values = "IMDb", index = "Genres", aggfunc= [np.mean, np.median]) #create pivot table with arregate function to calculate mean and median
data.groupby(["Genres", "Runtime"])[["IMDb"]].mean()
data[data["Year"]== 2020] #We still have a lot movies come out during the pandemic
ok = data.groupby("Genres")["Title"].nunique().sort_values(ascending=False).head(10)

ok.index
#create the pivot table and use nunique aggregate function

df = data.pivot_table(values = "Title", index = "Year", columns = "Genres", aggfunc= pd.Series.nunique, fill_value = 0).sort_values("Year", ascending= True)

df
dff = df[ok.index] #only show result of top 10 genres

dff
ab = dff.cumsum()

ab
def nice_axes(ax):

    ax.set_facecolor('.8') #set the face color of the axes

    ax.tick_params(labelsize=8, length=0) #Change the appearance of ticks

    ax.grid(True, axis='x', color='white') #Configure the grid lines

    ax.set_axisbelow(True)

    [spine.set_visible(False) for spine in ax.spines.values()]

def prepare_data(df, steps=5):

    abc = ab.loc[1900:2020].reset_index() #reset the index to add transitioning value

    abc.index = abc.index*5 #Add 5 transitioning values between each year

    last_idx = abc.index[-1] + 1

    df_expanded = abc.reindex(range(last_idx))

    df_expanded['Year'] = df_expanded['Year'].fillna(method='ffill').astype(int)

    df_expanded = df_expanded.set_index('Year') #Set Year column back to the index

    df_expanded = df_expanded.interpolate() #Interpolate data value value for more smoothly transitioning

    df_rank_expanded = df_expanded.rank(axis=1, method='first') #Rank the value to keep its color

    df_rank_expanded = df_rank_expanded.interpolate() #Interpolate rank value for more smoothly transitioning

    return df_expanded, df_rank_expanded



df_expanded, df_rank_expanded = prepare_data(df)

labels = df_expanded.columns

df_expanded.head()
from matplotlib.animation import FuncAnimation



def init():

    ax.clear()

    nice_axes(ax)

    ax.set_ylim(.2, 6.8) #Set the y-axis view limits



def update(i):

    for bar in ax.containers:

        bar.remove()

    y = df_rank_expanded.iloc[i] #The genres

    width = df_expanded.iloc[i] #Number of movies value

    ax.barh(y=y, width=width, color=colors, tick_label=labels) #Draw bar graph

    date_str = df_expanded.index[i] #Take the date in index column

    ax.set_title(f'Number of Movies by Genres - {date_str}')

    

fig = plt.Figure(figsize=(4, 2.5), dpi=144) #create figure

ax = fig.add_subplot()

colors = plt.cm.Dark2(range(6))

anim = FuncAnimation(fig=fig, func=update, init_func=init, frames=len(df_expanded), 

                     interval=200, repeat=False) #Configure the video
from IPython.display import HTML

html = anim.to_html5_video()

HTML(html)
anim.save('moviesbygenres.mp4') #Save the bar race video to local
df = data.groupby('Year')[['Title']].nunique() #Find number of unique movies every year

df
plt.plot(df.index, df[['Title']], color="blue") #Draw graph

plt.xlabel("Year") #Label x-axis

plt.ylabel("Number of movies") #Label y-axis

plt.title("Number of Movies over Years") #Label the Title

plt.show() #Show the graph
df = data[["Netflix","Hulu", "Prime Video", "Disney+"]].sum() #Calculate total movies of each platform

df
name = ["Netflix", "Hulu", "Prime Video", "Disney+"]

plt.bar(name, df, align="center", color="blue") #Draw bar graph

plt.xlabel("Streaming Platforms") #Label X-axis

plt.ylabel("Number of Movies") #label y-axis

plt.title("Number of Movies on Different Streaming Platforms") #Label the tile

plt.show() #Show the graph
#Find top 5 countries has the most movies

movies_by_country = data.groupby('Country')['Title'].count().reset_index().sort_values('Title',ascending = False).head(5).rename(columns = {'Title':'MovieCount'})

movies_by_country
explodeTuple = (0.1, 0.2, 0.3, 0.4, 0.5) #Making the wedge to explode out for easier to see

plt.pie(movies_by_country["MovieCount"], explode= explodeTuple, labels = movies_by_country["Country"], autopct='%5.0f%%',

        shadow=True, startangle=90) #Draw Pie chart

plt.axis('equal') #Ensure the pie chart is circle

plt.show() #Show the graph
runtime = data["Runtime"] + np.random.normal(0, 2, size = len(data)) #Define value and add random normal

IMDb = data["IMDb"] + np.random.normal(0,2, size = len(data)) #Define value and add random normal

plt.plot(IMDb, runtime, 'o', markersize = 0.2, alpha = 0.3) #Plot the Scatter graph with customized markersize and opacity

plt.xlabel("IMDb") #Label X-axis

plt.ylabel("Runtime (minutes)") #Label Y-axis

plt.axis([0, 10, 0, 200]) #Specify the range in the X-axis and Y-axis

plt.show() #Show the graph
#Draw the box plot to show relationship between IMDb score and Runtime with IMDb score > 8

sns.boxplot(x = data[data["IMDb"]>8]["IMDb"], y= data[data["IMDb"]>8]["Runtime"], data= data, whis = 10)

plt.show() #Show the plot
lg = pd.DataFrame(data.groupby("Year")["Title"].nunique()) #Number of movies by years

lg.plot() #Plot the line chart

plt.show() #Show the plot
Yearre = lg.index

Titlere = lg["Title"] 

totalre = linregress(Yearre, Titlere) #Find Linear Regression Indicators

totalre
lg["Year2"] = lg.index

lg.plot("Year2", "Title", kind= "scatter", alpha = 0.5) 

plt.show()
fx= np.array([lg["Year2"].min(), lg["Year2"].max()]) #Create an array contains the first and the last year in the dataset

fy = totalre.intercept + totalre.slope * fx #The Regression Line

lg.plot("Year2", "Title", kind= "scatter") #Plot the Scatter graph

plt.plot(fx, fy,'-') #Plot the Regression Line

plt.axis([1900, 2020, 0, 1400]) #Specify the range of axis-es

plt.legend([totalre.slope]) 

plt.show()
abcd= pd.DataFrame(data.groupby("Runtime")["IMDb"].mean()) #Find the mean IMDb score by Runtime

abcd = abcd[abcd.index < 500] #Specify the Runtime range since only a few movies has runtime > 500

plt.plot(abcd, 'o', alpha = 0.5) #Plot the Scatter

plt.xlabel('Runtime (Minutes)')

plt.ylabel('IMDb')

plt.show() #Show the plot
abcd["Runtime"] = abcd.index #Add a column

abcd["Runtime2"] = abcd["Runtime"] ** 2 #Add a quadractic term

abcd["IMDb2"] = abcd["IMDb"] ** 2 #Add a quadractic term

abcd.describe() #Describe the data
results = smf.ols('IMDb ~ Runtime + Runtime2', data= abcd).fit() #Run the multiple regression model

pred = results.predict(abcd) #Use the dataframe as the parameter to generate prediction

plt.plot(abcd['Runtime'], pred) #Plot the prediction generated by the model

plt.plot(abcd["Runtime"], abcd["IMDb"], 'o', alpha = 0.5) #Plot the Scatter

plt.legend(["predicted", "real"]) #Label the legends

plt.xlabel("Runtime") #Label the x-axis

plt.ylabel("IMDb Score") #Label the y-axis

plt.title("IMDb Score by Runtime") #Title the plot

plt.show() #Show the plot
opss = pd.DataFrame(data.groupby("Year")["Title"].nunique()) #Group the movies by year

opss.plot()

plt.show()
from statsmodels.tsa.statespace.sarimax import SARIMAX

opss.index = pd.to_datetime(opss.index, format = '%Y') #Format the datetime object
opss = opss.drop(opss.index[[0,1,2,3]]) #Drop the first few years because of interval inconsistency

opss = opss.drop(opss.index[-1]) #Drop the year of 2020 because it is not completed and it can cause error in data

opss
model = SARIMAX(opss, order = (1,0,0), trend = 'c') #Run the ARMA(1,0) model and constant model
results = model.fit() #Fit the model

print(results.summary()) #Print the Summary
forecast = results.get_prediction(start = -25) #Make predictions for the last 25 values
mean_forecast = forecast.predicted_mean  #Calculate the forecast mean

mean_forecast
confidence_intervals = forecast.conf_int() #Calculate the Confidence Interval (Include lower and upper limit)

confidence_intervals 
plt.plot(opss.index, opss, label ='observed') #Plot the real data

plt.plot(mean_forecast.index, mean_forecast, color = 'r', label='predicted') #Plot the predicted data

plt.fill_between(confidence_intervals.index,confidence_intervals['lower Title'], confidence_intervals['upper Title'], color='pink', label='interval') #Fill between the confidence intervals

plt.xlabel('Year') #Label the x-axis

plt.ylabel('Number of movies') #Label the y-axis

plt.legend() #Annotate the legends

plt.show() #Show the graph
forecast = results.get_forecast(steps = 16) #Forecast the next 16 years

mean_forecast = forecast.predicted_mean #Calculate the forecast mean

mean_forecast
confidence_interval = forecast.conf_int() #Calculate the confidence interval (include the upper and lower limits)

confidence_interval
plt.plot(opss.index, opss, label ='observed') #Plot the observed data the past years

plt.plot(mean_forecast.index, mean_forecast, color = 'r', label='predicted') #Plot the predicted data for the next 16 years

plt.fill_between(confidence_interval.index,confidence_interval['lower Title'], confidence_interval['upper Title'], color='pink', label = 'interval') #Fill between the confidence interval

plt.xlabel('Year') #Label the X-axis

plt.ylabel('Number of movies') #Label the Y-axis

plt.legend() #Show the legends

plt.show() #Show the graph
%matplotlib inline

#Basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

#Plots

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

#Format

from datetime import tzinfo, timedelta, datetime

#Models

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier
df_train = pd.read_csv("../input/train.csv",index_col=None)

df_train.dtypes
Crime_Categories = list(df_train.loc[:,"Category"].unique())

print("Number of crime categories: " + str(len(Crime_Categories)))

for crime in Crime_Categories:

    print(crime)
number_of_crimes = df_train.Category.value_counts()



_n_crime_plot = sns.barplot(x=number_of_crimes.index,y=number_of_crimes)

_n_crime_plot.set_xticklabels(number_of_crimes.index,rotation=90)
pareto_crime = number_of_crimes / sum(number_of_crimes)

pareto_crime = pareto_crime.cumsum()

_pareto_crime_plot = sns.tsplot(data=pareto_crime)

_pareto_crime_plot.set_xticklabels(pareto_crime.index,rotation=90)

_pareto_crime_plot.set_xticks(np.arange(len(pareto_crime)))

                              
Main_Crime_Categories = list(pareto_crime[0:8].index)

print("The following categories :")

print(Main_Crime_Categories)

print("make up to {:.2%} of the crimes".format(pareto_crime[8]))
df_train.DayOfWeek.value_counts()
df_train.head().Dates
origin_date = datetime.strptime('2003-01-01 00:00:00','%Y-%m-%d %H:%M:%S')



def delta_origin_date(dt):

    _ = datetime.strptime(dt,'%Y-%m-%d %H:%M:%S') - origin_date

    return(_.days+(_.seconds/86400))



delta_origin_date(df_train.loc[1,"Dates"])
tmp = df_train.loc[:,["Dates","Category"]]

tmp["RelativeDates"]=df_train.Dates.map(delta_origin_date)

tmp.head()
tmp["QuarterBucket"]=tmp.RelativeDates.map(lambda d: int(d/90.0))
pt = pd.pivot_table(tmp,index="QuarterBucket",columns="Category",aggfunc=len,fill_value=0)

pt = pt["Dates"]

pt[Main_Crime_Categories].iloc[:49,:].cumsum(1).plot()
pd.rolling_mean(pt[Main_Crime_Categories],3).iloc[2:49,:].plot()
tmp = df_train.loc[:,["Dates","Category"]]



tmp.loc[:,"Hour"] = df_train.Dates.map(lambda d: datetime.strptime(d,'%Y-%m-%d %H:%M:%S').hour)

tmp.describe()



pt = pd.pivot_table(tmp,index="Hour",columns="Category",aggfunc=len,fill_value=0)["Dates"]

pt.loc[:,Main_Crime_Categories].plot()
tmp = df_train.loc[:,["DayOfWeek","Category"]]

pt = pd.pivot_table(tmp,index="DayOfWeek",columns="Category",aggfunc=len,fill_value=0)

pt.loc[["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],

       Main_Crime_Categories].plot()
tmp = df_train.loc[:,["Dates","Category"]]



def assign_season(dt):

    _ = datetime.strptime(dt,'%Y-%m-%d %H:%M:%S')

    if _.month >= 11 or _.month <= 2:

        return("Winter")

    elif _.month > 2 and _.month < 6:

        return("Spring")

    elif _.month >= 6 and _.month < 9:

        return("Summer")

    else:

        return("Fall")

    

tmp.loc[:,"Season"] = tmp.Dates.map(assign_season)

pt = pd.pivot_table(tmp,index="Season",columns="Category",aggfunc=len,fill_value=0)["Dates"]

pt.loc[["Winter","Spring","Summer","Fall"],Main_Crime_Categories].plot()
tmp = df_train.loc[:,["Dates","Category"]]

tmp.loc[:,"Month"] = tmp.Dates.map(lambda dt: datetime.strptime(dt,'%Y-%m-%d %H:%M:%S').month)

pt = pd.pivot_table(tmp,index="Month",columns="Category",aggfunc=len,fill_value=0)["Dates"]

pt.loc[:,Main_Crime_Categories].plot(ylim=0)
tmp.loc[:,"Year"] = tmp.Dates.map(lambda dt: datetime.strptime(dt,'%Y-%m-%d %H:%M:%S').year)

pt = pd.pivot_table(tmp,index=["Year","Month"],columns="Category",aggfunc=len,fill_value=0)["Dates"]

pt[Main_Crime_Categories].plot()



#Remember to delete this, this is just for a test for FFT

y = pt
pt = pd.pivot_table(tmp,index="Month",columns="Year",aggfunc=len,fill_value=0)["Dates"]

pt.plot()
for cat in Main_Crime_Categories:

    spectrum = np.fft.fft(y[cat].iloc[:]-y[cat].iloc[:].mean())

    freq = np.fft.fftfreq(len(spectrum))

    plt.plot(freq, np.absolute(spectrum))

    plt.title(cat)

    plt.show()
most_dangerous_districts = df_train.PdDistrict.value_counts()

_n_crime_plot = sns.barplot(x=most_dangerous_districts.index,y=most_dangerous_districts)

_n_crime_plot.set_xticklabels(most_dangerous_districts.index,rotation=90)
pt = pd.pivot_table(df_train,index="PdDistrict",columns="Category",aggfunc=len,fill_value=0)["Dates"]

_ = pt.loc[most_dangerous_districts.index,number_of_crimes.index]

ax = sns.heatmap(_)

ax.set_title("Number of Crimes per District")
pt = pd.pivot_table(df_train,index="PdDistrict",columns="Category",aggfunc=len,fill_value=0)["Dates"]

pt = pt/pt.sum(axis = 0)

_ = pt.loc[most_dangerous_districts.index,number_of_crimes.index]

ax = sns.heatmap(_)

ax.set_title("Repartition of Crimes accross Districts")
pt = pd.pivot_table(df_train,index="PdDistrict",columns="Category",aggfunc=len,fill_value=0)["Dates"]

pt = pt.div(pt.sum(axis = 1),axis=0)

_ = pt.loc[most_dangerous_districts.index,number_of_crimes.index]

ax = sns.heatmap(_)

ax.set_title("Most Prevalent Crime per District")
def correct_coordinates(district):

    tmp = df_train[df_train["PdDistrict"] == district]

    x_avg = tmp.X.mean()

    y_avg = tmp.Y.mean()

    return([x_avg,y_avg])



index_to_correct = df_train[(df_train["X"] > -121)|(df_train["Y"] > 50)].index



for i in index_to_correct:

    tmp = correct_coordinates(df_train.loc[i,"PdDistrict"])

    x = tmp[0]

    y = tmp[1]

    df_train.loc[i,"X"] = x 

    df_train.loc[i,"Y"] = y
xy_box = np.zeros([2,2])



xy_box[0,0] = df_train["X"].min()

xy_box[0,1] = df_train["X"].max()

xy_box[1,0] = df_train["Y"].min()

xy_box[1,1] = df_train["Y"].max()





barycenter = [df_train["X"].sum()/len(df_train["X"]),df_train["Y"].sum()/len(df_train["Y"])]

print("Coordinates box: {}".format(xy_box))

print("Barycenter: {}".format(barycenter))



df_plot_coordinates = df_train[["X","Y"]].sample(frac=0.001)

sns.stripplot(df_plot_coordinates["X"],df_plot_coordinates["Y"])
for c in pareto_crime.index[:25]:

    df_tmp = df_train.loc[df_train.Category == c,["X","Y"]]

    with sns.axes_style("white"):

        sns.jointplot(x=df_tmp["X"] , y=df_tmp["Y"], stat_func = None,kind="hex", color="r");

        sns.plt.title(c)
df_test = pd.read_csv("../input/test.csv",index_col=None)

df_test.head()
model_null = df_test[["Id"]]

for crime in Crime_Categories:

    model_null[crime] = number_of_crimes[crime]/sum(number_of_crimes)

model_null.head()

#The output file seems to be to heavy

#model_null.to_csv("model_null.csv")
def is_weekend(d):

    if d in ["Friday","Saturday","Sunday"]:

        return(1)

    else:

        return(0)

    

def get_hour(d):

    h = datetime.strptime(d,'%Y-%m-%d %H:%M:%S').hour

    s = datetime.strptime(d,'%Y-%m-%d %H:%M:%S').second

    return((h+s/60.0)/24.0)

    



#df_train_time_only = df_train

#df_train_time_only["Year"] = df_train_time_only.Dates.map(lambda dt: datetime.strptime(dt,'%Y-%m-%d %H:%M:%S').year)

#df_train_time_only["Month"] = df_train_time_only.Dates.map(lambda dt: datetime.strptime(dt,'%Y-%m-%d %H:%M:%S').month/12.0)

#df_train_time_only["Hour"] = df_train_time_only.Dates.map(get_hour)

#df_train_time_only["IsWeekend"] = df_train_time_only.DayOfWeek.map(is_weekend)

#df_train_time_only.head()
def make_as_OTHER(crime):

    if crime in Main_Crime_Categories:

        return(crime)

    else:

        return("OTHER_MANUAL")



#df_tmp = df_train_time_only#[df_train_time_only['Year'].isin([2008,2009,2010,2011,2012,2013,2014])]

#df_tmp["Category"] = df_tmp.Category.map(make_as_OTHER)







#df_x = df_tmp[["Month","IsWeekend","Hour"]]

#df_y = df_tmp["Category"]



#scores = []

#for i in range(5):

#    knn = KNeighborsClassifier(n_neighbors = i + 2)

#    scores.append(cross_val_score(knn,df_x,df_y,cv=2))
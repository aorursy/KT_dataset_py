# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def simplify_armed(x):

    '''Simplify the armed column so that it only has three values'''

    if (pd.isnull(x) or x == "undetermined"):

        return "undetermined"

    elif x!="unarmed":

        return "armed"

    else:

        return "unarmed"



def impute_race(x):

    '''Impute the race column so that nan values return O'''

    if pd.isnull(x):

        return "O"

    else:

        return x

        

def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = round(rect.get_height(),2)

        axes.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')

        

def grouped_bars(fig, axes, labels,bar_data,xcoors,n_sub_bars,w,sub_labels,xlb,ylb,title):

    '''Create a grouped bar graph'''

    rects = []

    width = w / n_sub_bars

    n = len(bar_data)

    for i in range(n):

        rects.append(axes.bar(xcoors - 0.25 + (i * (w/n_sub_bars)), bar_data[i], width, label=sub_labels[i]))

    

    axes.set_xlabel(xlb)

    axes.set_ylabel(ylb)

    axes.set_title(title)

    axes.set_xticks(xcoors)

    axes.set_xticklabels(labels)

    axes.legend()

    for i in range(n):

        autolabel(rects[i])

    fig.set_size_inches(18,8)

    

def get_city_race_values(df,cities,race):

    values = np.zeros(len(cities), dtype=int)

    i=0

    for city in cities:

        tmp = df.query("city_state == @city and race == @race")["id"]

        if tmp.tolist() == []:

            values[i] = 0

        else:

            values[i] = int(tmp.tolist()[0])

        i = i+1

    return values

ps_df = pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv")

pd.Series(data=list(ps_df.shape),index=['Rows','Columns'])
pd.DataFrame({"Data_Type": ps_df.dtypes, "Percent_Null": ps_df.isnull().mean() * 100})
ps_df["race"] = list(map(impute_race,ps_df["race"])) #impute the race column
ps_df.head()
deaths_by_date = ps_df.groupby("date").agg("count").filter(["id"]).rename(columns={"id":"count"})

plt.figure(figsize=(18,8))

plt.plot(deaths_by_date) #time-series by date
import statsmodels

from statsmodels.tsa.stattools import adfuller



adf_test = adfuller(deaths_by_date, autolag='AIC')



pd.Series({"adf":adf_test[0],"p-value":adf_test[1],"usedlag":adf_test[2],"crit_1_pct":adf_test[4]["1%"],"crit_5_pct":adf_test[4]["5%"],"crit_10_pct":adf_test[4]["10%"]})

plt.figure(figsize=(18,8))

sns.distplot(ps_df["age"])
ps_df["age"].describe()
bar_data = ps_df.groupby("gender").agg("count").rename(columns={"id":"Deaths By Gender Count"})

plt.figure(figsize=(18,8))

sns.barplot(bar_data.index,bar_data["Deaths By Gender Count"])
bar_data = ps_df.groupby("race").agg("count").rename(columns={"id":"Deaths By Race Count"})

plt.figure(figsize=(18,8))

sns.barplot(bar_data.index,bar_data["Deaths By Race Count"])
ps_df["armed_simple"] = list(map(simplify_armed,ps_df["armed"])) #add a new column with simplified values from armed column.



#Add plot, same as above, broken down by armed_simple

bar_data = ps_df.groupby(["race","armed_simple"]).agg("count")["id"].reset_index().rename(columns={"id":"count"}) #Get the data for the plot

tots = ps_df.groupby(["race"]).agg("count")["id"].reset_index().rename(columns={"id":"count"})

labels = bar_data["race"].unique() 

labels = labels[pd.isnull(labels) == False] #Create x-tick labels

armed_pct = (bar_data.query("armed_simple == 'armed'")["count"].values / tots["count"]).values #Data for armed_count rectangles

unarmed_pct = (bar_data.query("armed_simple == 'unarmed'")["count"].values / tots["count"]).values #Data for unarmed_count rectangles

undetermined_pct = (bar_data.query("armed_simple == 'undetermined'")["count"].values / tots["count"]).values #Data for undetermined_count rectangles

xcoor = np.arange(1,len(bar_data["race"].unique()) - pd.isnull(bar_data["race"].unique()).sum()+1) #positions of x-tick marks

num_sub_bars = len(bar_data["armed_simple"].unique()) - pd.isnull(bar_data["armed_simple"].unique()).sum() #number of rectangles per category

width = 0.75/num_sub_bars #width of a rectangle

fig, axes = plt.subplots() #Create the plot

grouped_bars(fig,axes,labels,[armed_pct,unarmed_pct,undetermined_pct],xcoor,num_sub_bars,0.75,["armed","unarmed","undetermined"],"Race","Deaths Percent Armed and Race","Deaths by Race and Armed Simple")
ps_df["city_state"] = ps_df["city"] + "," + ps_df["state"]

bar_data = ps_df.groupby("city_state", as_index=False).agg("count").rename(columns={"id":"Deaths By City"}).sort_values(by="Deaths By City",ascending=False).head(10)

city_pops = [39,16,23,6.4,15,27,5.6,8.9,6.4,7.1]

fig, axes = plt.subplots()

fig.set_size_inches(18,8)

axes.plot(city_pops,color="red",label="population in hundreds of thousands")

sns.barplot(bar_data["city_state"],bar_data["Deaths By City"], ax=axes)

fig.legend()
#add a plot here that is the same as above except it splits by race.

highest_cities = bar_data["city_state"]

bar_data = ps_df.groupby(["city_state","race"], as_index=False).agg("count").filter(["city_state","race","id"]).query("city_state in @highest_cities")

bar_data["orig_order"] = [7,7,7,6,6,6,8,8,8,10,10,10,10,10,10,3,3,3,3,3,4,4,4,4,4,1,1,1,1,1,9,9,9,9,9,2,2,2,2,2,5,5,5,5]

bar_data = bar_data.sort_values(by="orig_order",ascending=True)

labels = bar_data["city_state"].unique().tolist()

A_data = get_city_race_values(bar_data,labels,"A")

B_data = get_city_race_values(bar_data,labels,"B")

H_data = get_city_race_values(bar_data,labels,"H")

N_data = get_city_race_values(bar_data,labels,"N")

O_data = get_city_race_values(bar_data,labels,"O")

W_data = get_city_race_values(bar_data,labels,"W")

xcoors = np.arange(1,len(labels)+1)

num_sub_bars = len(bar_data["race"].unique())

sub_bar_labels = ["A","B","H","N","O","W"]

xlb = "City and Race"

ylb = "Death Count"

title = "Deaths by Top 10 Cities and Race"



fig, axes = plt.subplots()

grouped_bars(fig,axes,labels,[A_data,B_data,H_data,N_data,O_data,W_data],xcoors,num_sub_bars,0.75,sub_bar_labels,xlb,ylb,title)
bar_data = ps_df.groupby("flee").agg("count")

plt.figure(figsize=(18,8))

sns.barplot(bar_data.index,bar_data["id"])
#add a plot here that is the same as above except it splits by manner of death

bar_data = ps_df.groupby(["flee","manner_of_death","armed_simple"],as_index=False).agg("count").filter(["flee","manner_of_death","armed_simple","id"]).rename(columns={"id":"count"})

labels = bar_data["flee"].unique()

shot_armed_data = bar_data.query("manner_of_death == 'shot' and armed_simple == 'armed'")["count"].values

shot_unarmed_data = bar_data.query("manner_of_death == 'shot' and armed_simple == 'unarmed'")["count"].values

shot_und_data = bar_data.query("manner_of_death == 'shot' and armed_simple == 'undetermined'")["count"].values

shot_and_tasered_armed_data = bar_data.query("manner_of_death == 'shot and Tasered' and armed_simple == 'armed'")["count"].values

shot_and_tasered_unarmed_data = bar_data.query("manner_of_death == 'shot and Tasered' and armed_simple == 'unarmed'")["count"].values

shot_and_tasered_ind_data = bar_data.query("manner_of_death == 'shot and Tasered' and armed_simple == 'undetermined'")["count"].values

xcoors = np.arange(1,len(labels)+1)

n_sub_bars = len(bar_data["manner_of_death"].unique())

sub_labels = ["shot armed","shot unarmed","shot undetermined","shot tasered armed","shot tasered unarmed","shot tasered undetermined"]

xlb = "Flee and Manner of Death"

ylb = "Deaths Count"

title = "Deaths by Flee and Manner of Death"

fig, axes = plt.subplots()

grouped_bars(fig,axes,labels,[shot_armed_data, shot_unarmed_data,shot_und_data,shot_and_tasered_armed_data, shot_and_tasered_unarmed_data,shot_and_tasered_ind_data],xcoors,n_sub_bars,0.25,sub_labels,xlb,ylb,title)
#as the title says. Just one plot.

bar_data = ps_df.groupby(["race","manner_of_death"],as_index=False).agg("count").rename(columns={"id":"count"}).filter(["race","manner_of_death","count"])

race_counts = ps_df.groupby("race").agg("count")["id"].values

labels = bar_data["race"].unique()

shot_data = ((bar_data.query("manner_of_death == 'shot'")["count"] / race_counts)*100).values

tasered_data = ((bar_data.query("manner_of_death == 'shot and Tasered'")["count"] / race_counts)*100).values

xcoors = np.arange(1,len(labels)+1)

n_sub_bars = len(bar_data["manner_of_death"].unique())

sub_labels = ["shot","shot and tasered"]

xlb = "Race and Manner of Death"

ylb = "Deaths Percent by Race"

title = "Deaths by Race and Manner"

fig, axes = plt.subplots()

grouped_bars(fig,axes,labels,[shot_data,tasered_data],xcoors,n_sub_bars,0.75,sub_labels,xlb,ylb,title)
bar_data = ps_df.groupby("signs_of_mental_illness").agg("count").rename(columns={"id":"count"})

fig = plt.figure(figsize=(18,8))

sns.barplot(["False","True"],bar_data["count"])

mi_factor = bar_data["count"][0] / bar_data["count"][1]
#add a plot here, same as above, splits on flee.

bar_data = ps_df.groupby(["signs_of_mental_illness","flee"],as_index=False).agg("count").rename(columns={"id":"count"}).filter(["signs_of_mental_illness","flee","count"])

labels = ["No Mental Illness","Has Signs of Mental Illness"]

car_data = (bar_data.query("flee == 'Car'")["count"] * [1,mi_factor]).values

foot_data = (bar_data.query("flee == 'Foot'")["count"] * [1,mi_factor]).values

not_flee_data = (bar_data.query("flee == 'Not fleeing'")["count"] * [1,mi_factor]).values

other_data = (bar_data.query("flee == 'Other'")["count"] * [1,mi_factor]).values

xcoors = np.arange(1,len(labels)+1)

n_sub_bars = len(bar_data["flee"].unique())

sub_labels = bar_data["flee"].unique().tolist()

xlb = "Mental Illness and Fleeing"

ylb = "Adj Deaths Count"

title = "Adjusted Deaths by Mental Illness and Fleeing"

fig, axes = plt.subplots()

grouped_bars(fig,axes,labels,[car_data,foot_data,not_flee_data,other_data],xcoors,n_sub_bars,0.75,sub_labels,xlb,ylb,title)
#add a plot here, same as above, splits on flee, only unarmed.

tmp = ps_df.query("armed == 'unarmed'").groupby(["signs_of_mental_illness"], as_index=False).agg("count").rename(columns={"id":"count"}).filter(["signs_of_mental_illness","count"])

mi_factor = tmp["count"][0]/tmp["count"][1]

bar_data = ps_df.query("armed_simple == 'unarmed'").groupby(["signs_of_mental_illness","flee"],as_index=False).agg("count").rename(columns={"id":"count"}).filter(["signs_of_mental_illness","flee","count"])

labels = ["No Mental Illness","Has Signs of Mental Illness"]

car_data = (bar_data.query("flee == 'Car'")["count"] * [1,mi_factor]).values

foot_data = (bar_data.query("flee == 'Foot'")["count"] * [1,mi_factor]).values

not_flee_data = (bar_data.query("flee == 'Not fleeing'")["count"] * [1,mi_factor]).values

other_data = (bar_data.query("flee == 'Other'")["count"] * [1,mi_factor]).values

xcoors = np.arange(1,len(labels)+1)

n_sub_bars = len(bar_data["flee"].unique())

sub_labels = bar_data["flee"].unique().tolist()

xlb = "Mental Illness and Fleeing Unarmed"

ylb = "Adj Deaths Count"

title = "Adjusted Deaths by Mental Illness and Fleeing Unarmed"

fig, axes = plt.subplots()

grouped_bars(fig,axes,labels,[car_data,foot_data,not_flee_data,other_data],xcoors,n_sub_bars,0.75,sub_labels,xlb,ylb,title)
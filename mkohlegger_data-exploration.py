import numpy as np

import pandas as pd

import datetime as dt

from matplotlib import pyplot as plt

import numpy as np

from sklearn.preprocessing import OneHotEncoder

import seaborn as sns

import matplotlib.colors as mcolors
# load data using pandas



root="/kaggle/input/coronavirusdataset/"



data_patient = pd.read_csv(

    root+"patient.csv",

    parse_dates=["confirmed_date", "released_date", "deceased_date"]

)



data_time = pd.read_csv(

    root+"time.csv",

    parse_dates=["date"]

)



data_route = pd.read_csv(

    root+"route.csv",

    parse_dates=["date"]

)





# change indexes after loading



data_time.index = data_time.date

data_time.drop("date", axis=1, inplace=True)



data_patient.index = data_patient.id

data_patient.drop("id", axis=1, inplace=True)



data_route.index = data_route.id

data_route.drop("id", axis=1, inplace=True)





# add columns



data_patient["age"] = dt.datetime.now().year - data_patient.birth_year

data_patient["age_binned"] = pd.cut(data_patient.age, bins=range(0,100,5))
data_time.head(3)
data_patient.head(3)
data_route.head(3)
plt.figure(figsize=(20,8))



for column in data_time.columns:

    if column[0:3] == "acc":

        plt.plot(data_time[column], label=column)



plt.legend(loc=0)

plt.title("Development of COVID-19 cases over time")

plt.xlabel("time")

plt.ylabel("cases")





plt.show()
plt.figure(figsize=(20,8))





plt.plot(data_time.new_negative, label="new negative", color="k", ls=":")

plt.plot(data_time.new_confirmed, label="new confirmed", color="k")       

plt.axvline(x=dt.datetime.strptime("2020-02-12", "%Y-%m-%d"), label="change of assessment method", color="red", ls=":")



plt.legend(loc=0)

plt.title("Development of new COVID-19 cases over time")

plt.xlabel("time")

plt.ylabel("cases")



plt.show()
fig, ax = plt.subplots(2,1, figsize=(20,8), sharex=True)



ax[0].hist(data_patient.age.dropna(), bins=10, color="k")

ax[0].axvline(x=np.median(data_patient.age.dropna()), label="median", color="red")

ax[0].axvline(x=np.mean(data_patient.age.dropna()), label="mean", color="orange")

ax[0].axvline(x=np.quantile(data_patient.age.dropna(), q=[.25]), label="quart 1", color="green")

ax[0].axvline(x=np.quantile(data_patient.age.dropna(), q=[.75]), label="quart 3", color="lightgreen")



ax[0].set_title("Distribution of Patient Age")

ax[0].set_ylabel("abs. freuquency")

ax[0].set_xlabel("age")

ax[0].legend(loc=0)



ax[1].boxplot(data_patient.age.dropna(), vert=False)

ax[1].set_xlabel("age")



plt.show()
bin_vector=range(0,100,5)



fig, ax = plt.subplots(1,3, figsize=(20,8), sharex=True, sharey=True)



ax[2].hist(data_patient.age.loc[data_patient.state=="deceased"].dropna(), color="k", bins=bin_vector)

ax[2].axvline(x=np.median(data_patient.age.loc[data_patient.state=="deceased"].dropna()), label="median", color="red")

ax[2].set_title("Deceased persons")

ax[2].legend(loc=0)



ax[0].hist(data_patient.age.dropna(), color="k", bins=bin_vector)

ax[0].axvline(x=np.median(data_patient.age.dropna()), label="median", color="red")

ax[0].set_title("Infected persons")

ax[0].legend(loc=0)



ax[1].hist(data_patient.age.loc[data_patient.state=="released"].dropna(), bins=bin_vector, color="k")

ax[1].axvline(x=np.median(data_patient.age.loc[data_patient.state=="released"].dropna()), label="median", color="red")

ax[1].set_title("Released persons")

ax[1].legend(loc=0)



plt.show()
bin_vector=range(0,100,5)



plt.figure(figsize=(20,8))



plt.hist(data_patient.age.dropna(), bins=bin_vector, color="black", label="infected")

plt.hist(data_patient.age.loc[data_patient.state=="released"].dropna(), bins=bin_vector, color="orange", label="released", alpha=0.7)

plt.hist(data_patient.age.loc[data_patient.state=="deceased"].dropna(), bins=bin_vector, color="red", label="deceased", alpha=0.7)



plt.xticks(ticks=bin_vector)

plt.title("Distribution of Patient Age")

plt.ylabel("abs. freuquency")

plt.xlabel("age")

plt.legend(loc=0)



plt.show()
import seaborn as sns



plt.figure(figsize=(20,8))



sns.distplot(data_patient.age.dropna(), hist=False, rug=False, label="infected")

sns.distplot(data_patient.age.loc[data_patient.state=="released"].dropna(), hist=False, rug=False, label="released")

sns.distplot(data_patient.age.loc[data_patient.state=="deceased"].dropna(), hist=False, rug=False, label="deceased")



plt.title("Relative Density of persons in group of infected/released/deceased")

plt.ylabel("rel density")

plt.xlabel("age")



plt.legend(loc=0)
plt.figure(figsize=(10,10))



plt.pie(

    data_patient.state.value_counts(),

    labels=data_patient.state.value_counts().index,

    colors=["k", "lightgreen", "red"],

    explode=(0.2, 0.2, 0.2)

)



plt.show()
plt.figure(figsize=(20,10))

sns.heatmap(

    data_patient.pivot_table(

        values="age",

        columns='region',

        index='age_binned',

        margins=False,

        aggfunc='count'

    ),

    annot=True,

    linewidths=0.1,

    linecolor="k",

    cbar=False

)



plt.title("Age group over Region (Counts)")
plt.figure(figsize=(10,6))

sns.stripplot(x=data_patient.country, y=data_patient.age, hue=data_patient.sex)

plt.title("Age of patients from different countries")

plt.show()
plt.figure(figsize=(20,8))



plt.scatter(data_patient.age, data_patient.contact_number, alpha=0.7, color="k", label="data")

plt.axvline(np.median(data_patient.age.dropna()), label="median age", color="orange")

plt.axhline(np.median(data_patient.contact_number.dropna()), label="median contact count", color="r")



plt.title("Age over Contacts")

plt.xlabel("age")

plt.ylabel("contacts")



plt.legend(loc=0)



plt.show()
from scipy.stats import ttest_ind

from itertools import combinations



released = ("released", data_patient.age.loc[data_patient.state=="released"].dropna())

deceased = ("deceased", data_patient.age.loc[data_patient.state=="deceased"].dropna())

infected = ("infected", data_patient.age.dropna())



for item in combinations([released, infected, deceased], r=2):

    

    result = ttest_ind(a=item[0][1], b=item[1][1])

    a_name = item[0][0]

    b_name = item[1][0]

    p = round(result[1], 4)

    t = round(result[0], 2)

    

    print(f"H0: age({a_name}) = age({b_name}) has a p-value of {p}")

    print(f"H0: age({a_name}) >= age({b_name}) has a p-value of {p/2} and a t of {t}\n")
data_route.head(10)
bounding_box = (

    np.floor(data_route.longitude.min()),

    np.ceil(data_route.longitude.max()),

    np.floor(data_route.latitude.min()),

    np.ceil(data_route.latitude.max())

)
plt.figure(figsize=(20,16))



plt.scatter(

    data_route.longitude,

    data_route.latitude,

    marker="o",

    zorder=1,

    c="k",

    alpha=0.6,

    s=120

)



plt.xlim(bounding_box[0], bounding_box[1])

plt.ylim(bounding_box[2], bounding_box[3])



plt.imshow(

    plt.imread("/kaggle/input/sk-map/map.png"),

    zorder=0,

    extent=bounding_box,

    aspect='equal',

    alpha=0.4

)



plt.title('Plotting Spatial data about COVID-19 patients in South Korea')

plt.legend(loc=0)



plt.show()
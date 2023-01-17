import numpy as np

import pandas as pd

import datetime



import copy



from py_stringmatching.tokenizer.alphanumeric_tokenizer import AlphanumericTokenizer as AT  #You will see what this can do in a while.

import math



import matplotlib.pyplot as plt

import seaborn as sns



import re
input_data = pd.read_csv("../input/googleplaystore.csv")
input_data.head()
input_data.shape
input_data.info()
category_dict = input_data["Category"].value_counts().to_dict()

genres_dict = input_data["Genres"].value_counts().to_dict()
print(len(category_dict))

print(len(genres_dict))
category_dict
genres_dict
# Dropping the Genres column



input_data = input_data.drop(["Genres"], axis=1)
input_data.loc[input_data["Category"] == "1.9"]
input_data.drop(input_data.index[10472], inplace=True)
input_data.reset_index(inplace=True)

input_data = input_data.drop(["index"], axis=1)  # dropping the old index which is now present as a separate row.
# Converting Installs to integers



pd.options.mode.chained_assignment = None    #to ignore the warning



for element in input_data["Installs"]:

    if type(element) != str:

        if math.isnan(element):

            pass

    else:

        new_element = ""

        element_int = AT().tokenize(element)

        for term in element_int:

            new_element += term

        input_data.Installs.replace([element], [new_element], inplace=True)

    

input_data["Installs"] = input_data["Installs"].apply(pd.to_numeric, errors='coerce')
# Converting reviews to integers



input_data["Reviews"] = input_data["Reviews"].apply(pd.to_numeric, errors='coerce')
f, ax = plt.subplots(figsize=[8,8])



sns.scatterplot(x = "Installs", y = "Reviews", data = input_data)

ax.set_xlim(1,10000000000)

ax.set_ylim(1,100000000)

ax.set(xscale="log", yscale="log")
input_data = input_data.drop(["Reviews"], axis=1)
input_data = input_data.drop(["Current Ver"], axis=1)
input_data["Last Updated"] = pd.to_datetime(input_data["Last Updated"], format="%B %d, %Y")
max(input_data["Last Updated"])
# Making a list of the years from "Last Updated" column

Update_Year = []



for i in range(len(input_data)):

    year = input_data["Last Updated"][i].year

    Update_Year.append(year)



# Making this list as a new column

input_data["Update_Year"] = Update_Year
# Dropping "Last Updated" column

#input_data = input_data.drop(["Last Updated"], axis=1)
input_data["Price"].value_counts()
# Converting Price to integers/floats



pd.options.mode.chained_assignment = None    #to ignore the warning



for element in input_data["Price"]:

    if type(element) != str:

        if math.isnan(element):

            pass

    else:

        new_element = ""

        element_int = AT().tokenize(element)

        for term in element_int:

            new_element += term

        input_data.Price.replace([element], [new_element], inplace=True)

    

input_data["Price"] = input_data["Price"].apply(pd.to_numeric, errors='coerce')
input_data["Price"] = input_data["Price"]/100
# Converting into kB



strg = "Varies with device"

pattern = re.compile(r'M')

pattern2 = re.compile(r'k')





for i in range(len(input_data)):

    a = pattern.sub(r'', input_data["Size"][i])

    if a in strg:

        pass

    else:

        if a != input_data["Size"][i]:

            a = float(a)

            input_data["Size"][i] = a * 1000

        else:

            b = pattern2.sub(r'', input_data["Size"][i])

            b = float(b)

            input_data["Size"][i] = b
# From kB to MB



for i in range(len(input_data)):

    if type(input_data["Size"][i]) == float:

        input_data["Size"][i] =  input_data["Size"][i]/1000
input_data.rename(columns = {"Content Rating":"Content_Rating", "Last Updated":"Last_Updated", "Android Ver":"Android_Ver"}, inplace=True)
f, ax = plt.subplots(2,1, figsize=[8,11])



sns.countplot(x = "Rating", data = input_data, ax = ax[0], palette="Reds")

sns.countplot(x = "Installs", data = input_data, ax = ax[1], palette="Blues")



for tick in ax[0].get_xticklabels():

    tick.set_rotation(70)

for tick in ax[1].get_xticklabels():

    tick.set_rotation(45)
f, ax = plt.subplots(figsize=[8,8])



sns.scatterplot(x="Installs", y="Rating", data=input_data)

ax.set_xlim(1,10000000000)

ax.set(xscale="log")
input_data["Rating"].corr(input_data["Installs"])
category_dict
del category_dict["1.9"]
len(category_dict)
r = [x for x in range(11)]*3

c = [y for y in range(3)]*11

i = 0

def coordinate():

    global r

    global c

    global i

    i += 1

    return r[i-1], c[i-1]

input_data["Installs"].median()

f, ax = plt.subplots(11,3, figsize=[18,90])



for keys in category_dict.keys():

    row, column = coordinate()

    category_df = input_data[(input_data.Category == keys)]

    sns.countplot(x = "Installs", data = category_df,hue = "Category", ax = ax[row, column])

    for tick in ax[row,column].get_xticklabels():

        tick.set_rotation(90)
count = []

mean = []

std = []

median = []



for keys in category_dict.keys():

    category_df = input_data[(input_data.Category == keys)]

    

    count.append(category_dict[keys])

    

    mean.append(category_df["Installs"].mean())

    

    std.append(category_df["Installs"].std())

    

    median.append(category_df["Installs"].median())
category_data = pd.DataFrame()
category_data["Category"] = category_dict.keys()

category_data["Count"] = count

category_data["Mean"] = mean

category_data["std"] = std

category_data["median"] = median
category_data
category_data.sort_values(by=["Mean"], ascending = False)
category_data.sort_values(by=["median", "Mean"], ascending = False)
input_data["Size"].value_counts()
size_list = input_data["Size"].tolist()
# Removing Varies with device

size_list = [x for x in size_list if x != "Varies with device"]
plt.hist(size_list, bins=20)
input_data["Type"].value_counts()
input_data["Content_Rating"].value_counts()
rating_dict = input_data["Content_Rating"].value_counts().to_dict()
rating_avg = {}

for key in rating_dict.keys():

    rating_df = input_data[(input_data.Content_Rating == key)]

    

    val = rating_df["Installs"].sum()/len(rating_df)

    

    rating_avg[key] = val
rating_avg
input_data["Type"].value_counts()
tp_dict = input_data["Type"].value_counts().to_dict()
tp_avg = {}

for key in tp_dict.keys():

    tp_df = input_data[(input_data.Type == key)]

    

    val = tp_df["Installs"].sum()/len(tp_df)

    

    tp_avg[key] = val
tp_avg
input_data
f, ax = plt.subplots(figsize=[16,7])



sns.countplot(input_data["Android_Ver"])

plt.xticks(rotation=90)
f, ax = plt.subplots()



sns.countplot(x = "Update_Year", data=input_data, palette='Reds')
data_2018 = input_data[(input_data.Update_Year == 2018)]
category_2018 = data_2018["Category"].value_counts().to_dict()
difference = {key: category_dict[key] - category_2018.get(key, 0) for key in category_dict.keys()}
difference_percent = {key: difference[key] /category_dict.get(key, 0) for key in category_dict.keys()}



# Rounding of to two decimal places

for x,y in difference_percent.items():

    y = round(y,2)

    difference_percent[x] = y
difference_percent
f, ax = plt.subplots(figsize=[16,7])



plt.bar(range(len(difference_percent)), list(difference_percent.values()), align='center')

plt.xticks(range(len(difference_percent)), list(difference_percent.keys()), rotation=90)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting data on charts(e.g. bar chart, pie chart)

%matplotlib inline

from collections import Counter 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/Suicides in India 2001-2012.csv') #Reading data

df = pd.DataFrame(data) #Creating dataframe of data



# Any results you write to the current directory are saved as output.


#Creating seperate dataframe of two columns "States" and "Total".

df1 = df[["State", "Total"]]

adder = 0

index_alert = 0

suicides = []

#Iterate over all the rows to sum up total suicides for every state.

for index in range(1,len(df1)):

    if df["State"][index] != df["State"][index-1]:

        adder = sum(np.array(df.loc[index_alert:index-1,"Total"]))

        suicides.append(adder)

        index_alert = index

        adder = 0

    elif df["State"][index] == "West Bengal":

        suicides.append(sum(np.array(df.loc[index:,"Total"])))

        break



states = list(set(np.array(df1["State"])))

states = sorted(states)



ind = np.arange(len(states))

width = 0.8

#Set figure size

plt.figure(figsize = (10,6))

plt.bar(ind, suicides, width, align = "edge")

plt.xticks(ind+width/2,states, rotation = "vertical")

plt.ylabel("No of suicides")



plt.show()



df2 = df["Gender"]

female_counter = 0

male_counter = 0

for i in range(0, len(df2)):

    if df2[i] == "Female":

        female_counter += 1

    else:

        male_counter += 1

gender_counter = [female_counter, male_counter]

#Set figure size

plt.figure(figsize = (10,6))



plt.pie(gender_counter,labels=["Female", "Male"], shadow = True,

        autopct = '%1.1f%%', explode= (0.08,0), colors = ["pink", "blue"],

        labeldistance = 0.75, startangle=90)

plt.title("Gender Count")

plt.show()

df3 = df[["Type_code", "Type", "Total"]]

df_causes = df3.loc[df3["Type_code"] == "Causes"]



ros = np.array(df_causes["Type"])

reason_of_suicide = np.unique(ros)

reason_of_suicide = np.sort(reason_of_suicide)

count_of_reasons = Counter(ros)



no_of_suicides = []

for i in reason_of_suicide:

    s = 0

    s = sum(df_causes.loc[df_causes["Type"] == i, "Total"])

    no_of_suicides.append(s + count_of_reasons[i])



ind = np.arange(len(reason_of_suicide))

width = 0.8

#Set figure size

plt.figure(figsize = (10,6))

plt.bar(ind, no_of_suicides, width, align = "edge")

plt.xticks(ind+width/2,reason_of_suicide, rotation = "vertical")

plt.ylabel("No of suicides")

plt.show()
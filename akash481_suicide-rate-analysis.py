import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

my_dataset = pd.read_csv("../input/master.csv")

















x = my_dataset.groupby("sex").agg({"suicides_no":"sum"})

xx = x.plot.bar(figsize = (8,6))

xx.set_title("Total Number of Suicides vs Sex")

plt.ylabel("Number of Suicides")


x1 = my_dataset.groupby(["sex","generation"]).agg({"suicides_no":"sum"})

xx1 = x1.plot.bar(figsize = (10,5))

xx1.set_title("Total Number of Suicides vs Sex and Generation)")

plt.ylabel("Number of Suicides")

x2 = my_dataset.groupby("country").agg({"suicides/100k pop":"mean"}).sort_values(by='suicides/100k pop')

f, ax = plt.subplots(figsize=(14, 26))

xx2 = x2.plot(kind='barh', ax=ax)

xx2.set_title("Country vs Suicides/100k population")

plt.ylabel("Country")

plt.xlabel("Suicides/100k population")

x3=my_dataset[my_dataset['country']=='Thailand'].groupby("gdp_per_capita ($)").agg({"suicides/100k pop":"mean"}).sort_values(by='gdp_per_capita ($)')                                                                                                                                                                                                                                                                                                                                                                                  

xx3 = x3.plot.bar(figsize = (20,10)) 

xx3.set_title("Suicides/100k population vs GDP of Thailand ")

plt.ylabel("Suicides/100k population")



x4=my_dataset[my_dataset['country']=='Iceland'].groupby("year").agg({"suicides_no":"sum"}).sort_values(by='year')

xx4 = x4.plot.bar(figsize = (20,10))  

xx4.set_title("Number of Suicides vs Year of Thailand ")

plt.ylabel("Number  of Suicides")



                                                                                                                                                                                                                                                     

x5 = my_dataset[my_dataset['sex']=='male'].groupby("age").agg({"suicides/100k pop":"mean"}).sort_values(by='suicides/100k pop')

xx5 = x5.plot.bar(figsize = (10,5))

xx5.set_title("Suicides/100k population vs Age Groups of Males")

plt.ylabel("Suicides/100k population")
x6 = my_dataset[my_dataset['sex']=='female'].groupby('age').agg({"suicides/100k pop":"mean"}).sort_values(by='suicides/100k pop')

xx6 = x6.plot.bar(figsize =(10,5))

xx6.set_title("Suicides/100k population vs Age Groups of Females")

plt.ylabel("Suicides/100k population")
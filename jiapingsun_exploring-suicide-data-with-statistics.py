# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# load the suicide data and the overview of first five rows

suicide = pd.read_csv("../input/master.csv")

suicide.head()
display(suicide.index) # row label of the df

display(suicide.columns) # column labels of the df, both are not callable

display(suicide.keys()) # columns for df, same as suicide.column

display(suicide.axes) # return a list representing all the axes of df, not callable
len(suicide.columns)
suicide.ndim # number of axes
display(suicide.size)

suicide.shape
suicide.info()
# so rename the labels as gdp_for_year to remove the space.

suicide.rename(columns={" gdp_for_year ($) ":"gdp_for_year","gdp_per_capita ($)":"gdp_per_capita"},inplace = True)
suicide.head(2)
suicide.dtypes
display(suicide.sex.nunique())

suicide.sex.unique()
display(suicide.age.nunique())

suicide.age.unique()
display(suicide.generation.nunique())

suicide.generation.unique()
suicide[["age","sex","generation"]] = suicide[["age","sex","generation"]].astype("category")

suicide.iloc[:,9] = suicide.iloc[:,9].str.replace(",","").astype("int")
suicide.info()

# memory usage decreased!
suicide.describe()
suicide.isna().head()
suicide.isna().sum()
display(suicide["HDI for year"].max())

suicide["HDI for year"].min()
suicide["HDI for year"].fillna(0, inplace=True)
suicide.info()
plt.figure(figsize=(15,5))

suicide.groupby("year")["country"].nunique().plot(kind = "bar")

plt.title('Number of Country by Year', fontsize=12)

plt.xlabel("")
# only keep the relevant columns

s1 = suicide.iloc[:,:7]

s1.head()
# focus first on the most recent data

# suicide data in 2016 (most recent year)

mask_year = s1["year"] == 2016

s1_2016 = s1[mask_year].drop("year",axis = 1).groupby("country").sum()

s1_2016
s1_2016.drop("Grenada",inplace=True)

s1_2016
s1_2016["country"] = s1_2016.index
fig,axes = plt.subplots(2,1,figsize = (10,10))

plt.subplots_adjust(wspace = 0,hspace = 0.2) # adjust the distance between subplots



ax1 = plt.subplot(2,1,1) 

# need to sort the data so that the output could be in the descending order

sns.barplot(x="suicides_no",y = "country", data = s1_2016.sort_values("suicides_no",ascending = False), palette="Blues_r")

# add text labels

locs1, labels1 = plt.yticks()

for a,b in zip(s1_2016["suicides_no"].sort_values(ascending=False),locs1):

    plt.text(a+20,b,'%.0f' % a, va="center",fontsize = 10)

# only keep the bottom left spines

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)

ax1.set_title("Suicide Numbers by Countries in 2016")

ax1.set_xlabel("")



ax2 = plt.subplot(2,1,2)

sns.barplot(x="suicides/100k pop",y = "country", data = s1_2016.sort_values("suicides/100k pop",ascending = False), palette="Blues_r")

locs2, labels2 = plt.yticks()

for a,b in zip(s1_2016["suicides/100k pop"].sort_values(ascending = False),locs2):

    plt.text(a+3,b,'%.0f' % a, va="center",fontsize = 10)

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)

ax2.set_title("Suicide Rates by Countries in 2016")
# count years

display(s1.year.max())

display(s1.year.min())

s1.year.max() - s1.year.min()
# count the number of data available in each year because we may only use the years with enough data.

country_num_year = suicide.groupby("year",as_index = False).agg({"country":"nunique"})



# draw the line graph would be much clearer that over 70% years have more than 65 data available.

# therefore, we select years from 1995 to 2014 to do the following examination.

x = country_num_year["year"]

y = country_num_year["country"]

quantile30 = country_num_year["country"].quantile(.3)

display(quantile30)



fig = plt.figure(figsize = (10,6))

plt.plot(x,y,"k-o",markersize = 3)

plt.axhline(quantile30,color = 'r',linestyle = '--')

plt.fill_between(x,y,quantile30,where=(y>=quantile30), facecolor='lightcoral',alpha = 0.8)

plt.fill_between(x,y,quantile30,where=(y<=quantile30), facecolor='lightgrey',alpha = 0.8)

plt.title("Number of Countries In Each Year")

plt.grid(True)

# color reference: https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
s2_time = suicide.groupby("year")["suicides/100k pop"].sum().loc["1995":"2014"].reset_index()

display(s2_time.head(),s2_time.tail())
# draw the line graph to see the general trends from 1995-2014

from matplotlib.ticker import MaxNLocator



fig,axes = plt.subplots(1, figsize=(9, 5))

ax3 = sns.lineplot(x="year",y = "suicides/100k pop", data = s2_time, marker = 'o')



ax3.set_xlabel("") # same as plt.xlabel("")

ax3.set_ylabel("") # same as plt.ylabel("")

ax3.xaxis.set_major_locator(MaxNLocator(integer=True)) # force the xtick label to be int for years

ax3.legend(["suicides per 100k population"])

# same as plt.legend(["suicides per 100k population"])

ax3.set_title("Suicides/100K population from 1995-2014")

# same as plt.set_title("Suicides/100K population from 1998-2014")
# to see different trend from age groups and sex groups

mask_end = suicide["year"] <= 2014

mask_start = suicide["year"] >= 1998

s2 = suicide[mask_start & mask_end]

s2_agetime = s2.groupby(["year","age"])["suicides/100k pop"].sum().reset_index()

s2_sextime = s2.groupby(["year","sex"])["suicides/100k pop"].sum().reset_index()



fig,axes = plt.subplots(2,1, figsize=(7, 7),sharex = True)

# set main title for all subplots

fig.suptitle("Different Trends for Age Groups and Gender Groups")

ax4 = plt.subplot(2,1,1)

sns.lineplot(x="year",y = "suicides/100k pop", data = s2_agetime, hue = "age",marker = "o",markersize = 5)

# adjust the legend position outside the graph

ax4.legend(loc = "upper right",bbox_to_anchor=(1.3,1))

ax4.set_xlabel("")



ax5 = plt.subplot(2,1,2)

sns.lineplot(x="year",y = "suicides/100k pop", data = s2_sextime, hue = "sex",marker = "*")

ax5.legend(loc = "upper right",bbox_to_anchor=(1.24,1))
suicide.head(3)
# show some statistic info on age groups over the years

s2_ageyear = s2.groupby(["year","age"])["suicides/100k pop"].agg(["size","mean","sum","std"])

s2_ageyear.head(10)
s2_ageyear.index.names
s2_ageyear.rename(index = {"5-14 years":"05-14 years"},inplace=True)

s2_ageyear.sort_index(ascending = [True,False],inplace = True)

s2_ageyear.head(10)
s2_ageyear.unstack()
ageyear_avg = s2_ageyear.unstack().loc[:,"mean"]

display(ageyear_avg.head())



y1 = ageyear_avg.iloc[:,0]

y2 = ageyear_avg.iloc[:,1]

y3 = ageyear_avg.iloc[:,2]

y4 = ageyear_avg.iloc[:,3]

y5 = ageyear_avg.iloc[:,4]

y6 = ageyear_avg.iloc[:,5]



fig = plt.figure(figsize = (10,5))

ax = plt.stackplot(ageyear_avg.index,y1,y2,y3,y4,y5,y6)

plt.legend(ageyear_avg.columns)

plt.title("The Average Suidice/100k Numbers in Different Age Groups from 1998-2014")



ageyear_avg.plot(kind = "bar",stacked = True)

plt.title("The Average Suidice/100k Numbers in Different Age Groups from 1998-2014")

plt.legend(loc = "upper right",bbox_to_anchor=(1.35,1))
ageyear_avg["total"] = ageyear_avg.apply(lambda row: row[0] + row[1] + row[2] + row[3] + row[4] + row[5], axis=1)



def getPercentage(row):

    for i in range(len(row)):

        row[i] = row[i]/row[6]

    return row



ageyear_avg_pct = ageyear_avg.apply(getPercentage, axis=1)

ageyear_avg_pct.drop(columns = ["total"], inplace = True)

ageyear_avg.drop(columns = ["total"], inplace = True)

display(ageyear_avg_pct.head())
fig = plt.figure()

ageyear_avg_pct.plot(kind = "bar",stacked = True)

plt.title("The Percentage of Average Suidice/100k in Different Age Groups from 1998-2014")

plt.legend(loc = "upper right",bbox_to_anchor=(1.35,1))
ageyear_sum = s2_ageyear.unstack().loc[:,"sum"]

ageyear_sum.describe()
xbar1 = ageyear_sum["15-24 years"].mean()

xbar2 = ageyear_sum["25-34 years"].mean()

var1 = ageyear_sum["15-24 years"].var()

var2 = ageyear_sum["25-34 years"].var()

df1 = df2 = 16

tscore = 2.12



var_sample = (var1 + var2)/2

ME = 2.12 * (var_sample*(1/8))**(0.5)

print("ME = ",ME)

print("xbar2 - xbar1 = ",xbar2-xbar1)

print("the confidence interval should be (",xbar2-xbar1-ME,",",xbar2-xbar1+ME,")")
s2_genderyear = s2.groupby(["year","sex"])["suicides/100k pop"].agg(["size","mean","sum","std"])

display(s2_genderyear.head(2))

s2_genderyear.tail(2)
meandiff = s2_genderyear.loc[(1998,"male"),"mean"]-s2_genderyear.loc[(2014,"male"),"mean"]

vardiff = s2_genderyear.loc[(1998,"male"),"std"]**2/474 + s2_genderyear.loc[(2014,"male"),"std"]**2/468

stddiff = vardiff ** 0.5



Zvalue = meandiff/stddiff

Zvalue > 1.645
suicide.head()
# only keep those valid records (HDI > 0)

mask_hdi = suicide["HDI for year"] > 0

s3_hdi = suicide[mask_hdi].groupby(["HDI for year","sex","age"])["suicides/100k pop"].sum().reset_index()

s3_hdi.head()
# see the distribute of HDI (a general picture)

fig,axes = plt.subplots()

display(s3_hdi["HDI for year"].max(),s3_hdi["HDI for year"].min())

bins = [0.45,0.55,0.65,0.75,0.85,0.95]

axes = plt.hist(s3_hdi["HDI for year"],bins = bins,rwidth = 0.9)
def f(row):

    if row['HDI for year'] < 0.55:

        text = "low"

    elif row['HDI for year'] < 0.7:

        text = "medium"

    elif row['HDI for year'] < 0.8:

        text = "high"

    else:

        text = "very high"

    return text



suicide["HDI level"] = suicide.apply(f,axis = 1)
# add a new column referencing HDI level to the data

s3_hdiLevel = suicide[mask_hdi].groupby(["HDI for year","sex","age","HDI level"])["suicides/100k pop"].sum().reset_index()

s3_hdiLevel.head()
fig,axes = plt.subplots(3,1,figsize=(8,10),sharex = True)

ax1 = plt.subplot(3,1,1)

sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel)

ax1.set_xlabel("")



ax2 = plt.subplot(3,1,2)

sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel,hue = "sex")

ax2.set_xlabel("")



ax3 = plt.subplot(3,1,3)

sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel,hue = "age")
s3_hdiLevel1 = suicide[mask_hdi].groupby("HDI level")["suicides/100k pop"].sum().reset_index()

s3_hdiLevel1.head()
s3_hdiLevel2 = suicide[mask_hdi].groupby(["HDI level","sex"])["suicides/100k pop"].sum().reset_index()

s3_hdiLevel2
s3_hdiLevel3 = suicide[mask_hdi].groupby(["HDI level","age"])["suicides/100k pop"].sum().reset_index()

s3_hdiLevel3
fig,axes = plt.subplots(3,1,figsize=(8,10),sharex = True)

ax1 = plt.subplot(3,1,1)

sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel1)

ax1.set_xlabel("")



ax2 = plt.subplot(3,1,2)

sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel2,hue = "sex")

ax2.set_xlabel("")



ax3 = plt.subplot(3,1,3)

sns.barplot("HDI level","suicides/100k pop",data = s3_hdiLevel3,hue = "age")
suicide.head()
s4_gdp = suicide.groupby(["gdp_per_capita"])["suicides/100k pop"].sum().reset_index()

s4_gdp.head()
# general picture of gdp_per_capita and suicides numbers

sns.scatterplot("gdp_per_capita","suicides/100k pop",data = s4_gdp,linewidth = 0)
# adding more groups criteria = sex and age

s4_gdp1 = suicide.groupby(["gdp_per_capita","sex"])["suicides/100k pop"].sum().reset_index()

s4_gdp1.head()
s4_gdp2 = suicide.groupby(["gdp_per_capita","age"])["suicides/100k pop"].sum().reset_index()

s4_gdp2.head()
fig,axes = plt.subplots(1,2,figsize=(12,6),sharey = True)

ax1 = plt.subplot(1,2,1)

sns.scatterplot("gdp_per_capita","suicides/100k pop",data = s4_gdp1,hue = "sex", style = "sex",linewidth = 0)



ax2 = plt.subplot(1,2,2)

sns.scatterplot("gdp_per_capita","suicides/100k pop",data = s4_gdp2,hue = "age", linewidth = 0)

ax2.set_ylabel("")
import numpy as np

suicide['gdp/10k_per_capita'] = np.floor(suicide["gdp_per_capita"].div(10000)).astype(int)

#s4_gdp = suicide.groupby(["gdp_per_capita"])["suicides/100k pop"].sum().reset_index()

suicide["gdp/10k_per_capita"].value_counts()
# general data about gdp group

s4_gdp3 = suicide[mask_year].groupby("gdp/10k_per_capita",as_index = False)["suicides/100k pop"].sum()

# multi-index = gdp group + sex group

s4_gdp4 = suicide[mask_year].groupby(["gdp/10k_per_capita","sex"])["suicides/100k pop"].sum()



fig = plt.subplots(2,1,figsize = (8,8))

plt.subplot(2,1,1)

sns.barplot("gdp/10k_per_capita","suicides/100k pop",data = s4_gdp3)

plt.subplot(2,1,2)

sns.barplot("gdp/10k_per_capita","suicides/100k pop",data = s4_gdp4.reset_index(),hue = "sex")
s4_gdp4.head()
s4_gdp4.index
sex_pct = s4_gdp4.groupby(level = 0).apply(lambda x: x / float(x.sum()))

sex_pct
sex_pct.unstack().head()
ax1 = sns.lineplot("gdp/10k_per_capita","suicides/100k pop",data = sex_pct.reset_index(),hue = "sex", style = "sex")

from matplotlib.ticker import FuncFormatter

ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
sex_pct.unstack().plot(kind = "bar", stacked = True)

plt.legend(loc = "upper right",bbox_to_anchor=(1.25,1))
# similar to the sex group, we then evaluate the age group

s4_gdp5 = suicide[mask_year].groupby(["gdp/10k_per_capita","age"])["suicides/100k pop"].sum()

age_pct = s4_gdp5.groupby(level = 0).apply(lambda x: x / float(x.sum()))

age_pct
ax2 = sns.lineplot("gdp/10k_per_capita","suicides/100k pop",data = age_pct.reset_index(),hue = "age", color = "age", marker = "o")



#the two methods below are same in adjusting percentage yticklabels

#reference: https://stackoverflow.com/questions/31357611/format-y-axis-as-percent

#vals = ax1.get_yticks()

#ax1.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

from matplotlib.ticker import FuncFormatter

ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

ax2.legend(loc = "upper right",bbox_to_anchor=(1.35,1))
age_pct.unstack().plot(kind = "bar", stacked = True)

plt.legend(loc = "upper right",bbox_to_anchor=(1.35,1))
# try to examine the correlations among different factors and suicides/100k population.

# using heatmap

# only keep relevant factors = suicides/100k pop & population & HDI for year & gdp_per_capita

suicide_corr = suicide[["suicides/100k pop","suicides_no","population", "HDI for year", "gdp_per_capita"]]

sns.heatmap(suicide_corr.corr(),annot = True)
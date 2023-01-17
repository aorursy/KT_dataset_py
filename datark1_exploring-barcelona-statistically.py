import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
population = pd.read_csv("../input/population.csv")

population.head()
population_total = population.groupby(["Year","Gender"])["Number"].sum()

population_total = population_total.reset_index()

plt.figure(figsize=(10, 6))

sns.barplot(x="Year", hue="Gender", y="Number", data=population_total)
pivot_popyr = population_total.pivot_table(values="Number",index="Year",columns="Gender", fill_value=0)

pivot_popyr["Total"] = pivot_popyr["Female"] + pivot_popyr["Male"]

pivot_popyr["F/M"] = pivot_popyr["Female"]/ pivot_popyr["Male"]

pivot_popyr
population_yr = population.groupby(["Year","District.Name"])["Number"].sum()

population_yr = population_yr.reset_index()

pivot_1 = population_yr.pivot_table(values="Number",index="Year",columns="District.Name", fill_value=0)

pivot_1
def age_groups(data):

    fixed_index = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54",

               "55-59","60-64","65-69","70-74","75-79","80-84","85-89","90-94",">=95"]

    data = data.reindex(fixed_index)

    data.reset_index(inplace=True)

    return data



def piramid(left_plot, right_plot, left_name="FEMALE", right_name="MALE",

                l_col ="Number", r_col ="Number"):

    '''

    Creates customised "age piramid" styled graph. By specifiying different columns it can display

    any stratified data on two horizontal graphs side by side.

    It also overlays the two groups showing excess of one group over another.

    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(10, 7))

    

    # left side

    ax1.barh(right_plot.index, right_plot[l_col], color="#7aafff")

    ax1.barh(left_plot.index, left_plot[l_col], tick_label=left_plot["Age"], color="#ed3bc3")

    ax1.set_title(left_name)

    ax1.grid(True)

    ax1.set_yticklabels([])

    ax1.set_xlim(ax1.set_xlim()[::-1])

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=70 )

    

    # right side

    ax2.barh(left_plot.index, left_plot[l_col], color="#f0a4fc")

    ax2.barh(right_plot.index, right_plot[r_col], tick_label=left_plot["Age"], color="#0e47f2")

    ax2.set_title(right_name)

    ax2.grid(True)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=70 )



    plt.tight_layout(pad=0.5, w_pad=0, h_pad=0)

    plt.show()
population2017 = population[population["Year"]==2017]

population2017.head()



pop2017 = population2017.loc[:,["Age","Number"]]

pop2017.head()



# Population grouped by age groups

grouped1 = pop2017.groupby("Age").agg(np.sum)

grouped1 = age_groups(grouped1)



#calculating total population 

pop_total = grouped1["Number"].sum()

#calculating percentage share of each group

grouped1["pct"] = [round((row*1.0)/pop_total*100,1) for row in grouped1['Number']]

#creating column of cumulative percentage sum

grouped1["cum_sum"] = grouped1["pct"].cumsum()



#fig, ax1 = plt.subplots(figsize=(10, 5))

#plt.bar(grouped1.index, grouped1["Number"], tick_label=grouped1["Age"])

#plt.xlabel("Age groups")

#plt.ylabel("Population")

#plt.xticks(rotation=45)

#plt.tight_layout()
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(15,12)) 



#population by age groups bar chart

plot1 = ax1.bar(grouped1.index, grouped1["Number"], tick_label=grouped1["Age"])

ax1.set_title("Population by age groups", fontweight="bold")

ax1.set_xlabel("Age groups")

ax1.set_ylabel("Population")

ax1.grid(True)

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

for i,h in enumerate(grouped1["Number"]):

    ax1.text(i-.4, h+1000, str(grouped1["pct"][i]) + " %", fontweight='bold')



#cumulative percentage chart

plot2 = ax2.bar(grouped1.index, grouped1["cum_sum"], tick_label=grouped1["Age"])

ax2.set_title("Cumulative percentage", fontweight="bold")

ax2.set_xlabel("Age groups")

ax2.set_ylabel("Cumulative pct")

ax2.grid(True)

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

for i,h in enumerate(grouped1["cum_sum"]):

    ax2.text(i-.4,h+1, str(round(h,1)) + " %", fontweight='bold')



plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

plt.show()
subset2 = population2017.loc[:,["Age","Number","Gender"]]



grouped2 = subset2.groupby(["Age","Gender"]).agg(np.sum)

grouped2.reset_index(inplace=True)

grouped2.set_index("Age", inplace=True)



male2017 = grouped2[grouped2["Gender"]=="Male"]

male2017 = age_groups(male2017)



female2017 = grouped2[grouped2["Gender"]=="Female"]

female2017 = age_groups(female2017)



#plotting the age piramid

piramid(female2017,male2017)
def grouping(df, col1 ="Immigrants", col2="Emigrants",col3=None):

    df2017 = df[df["Year"]==2017]

    subset = df2017.loc[:,["Age", col1, col2, col3]]

    subset = subset.dropna(axis=1, how='all', thresh=None, subset=None)

    grouped = subset.groupby("Age").agg(np.sum)

    grouped.reset_index(inplace=True)

    grouped.set_index("Age", inplace=True)

    #calling external function

    df_ready = age_groups(grouped)

    return df_ready



im_em_pop = pd.read_csv("../input/immigrants_emigrants_by_age.csv")

immigrants2017 = grouping(im_em_pop)

piramid(immigrants2017,immigrants2017, left_name="IMMIGRANTS", right_name="EMIGRANTS",l_col = "Immigrants", r_col = "Emigrants")
def grouping2(df,col):

    grouped = df.groupby(col).agg(np.sum)

    grouped.reset_index(inplace=True)

    grouped.set_index(col, inplace=True)

    return grouped



pop2017_districts = population2017.loc[:,["District.Name","Number"]]

pop2017_districts_grouped = grouping2(pop2017_districts,"District.Name")

pop2017_districts_grouped.index.names = ["District Name"]

pop2017_districts_grouped



pop2017_im_districts = im_em_pop.loc[:,["District Name","Immigrants","Emigrants"]]

pop2017_im_districts_grouped = grouping2(pop2017_im_districts,"District Name")

pop2017_im_districts_grouped



df1 = pop2017_districts_grouped

df2 = pop2017_im_districts_grouped

merged = df1.join(df2, how="outer")

merged.columns =["Total","Immigrants","Emigrants"]

merged.dropna(inplace=True)

merged["Percentage_im"] = merged["Immigrants"].div(merged["Total"]).mul(100)

merged["Percentage_em"] = merged["Emigrants"].div(merged["Total"]).mul(100)

merged
def age_categoriser(Age):

    if Age =="0-4" or Age == "5-9":

        return "Children"

    if Age =="10-14" or Age == "15-19":

        return "Teenagers"

    if Age =="20-24" or Age == "25-29"or Age == "30-34":

        return "Young adults"

    if Age =="35-39" or Age == "40-44" or Age == "45-49":

        return "Middle Adults"

    if Age =="50-54" or Age == "55-59" or Age == "60-64":

        return "Older Adults"

    else:

        return "Eldery people"



population= population.assign(age_category = lambda v: v.Age.apply(age_categoriser))



population_cat = population[population.loc[:,"Year"]==2017].groupby(["age_category"])["Number"].sum()

population_cat
population_cat.plot.bar()
births = pd.read_csv("../input/births.csv")

births.head()
births.describe()
births_grouped = births.groupby("Year").agg(np.sum)

births_grouped
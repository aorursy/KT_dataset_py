# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as srn

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/fifa19/data.csv")
data.head() #First 5 rows
data.tail() #Last 5 rows
data.info
data.columns
data.corr() #This shows us the table of correlation
fig,ax = plt.subplots(figsize=(30,30))

srn.heatmap(data.corr(),annot=True,linewidth=5,fmt = "0.1f",ax=ax)

plt.show()
#Dropping some columns

data2 = data.loc[:,["Name","Age","Overall","Potential","Position","Skill Moves","Weak Foot"]]



#Creating variables that contains mean values

by_skillmove = data2.groupby("Skill Moves").mean()

by_age = data2.groupby("Age").mean()

by_position = data2.groupby("Position").mean()



by_age
by_position
by_skillmove
def barplot(var,f1,f2,ax1,ax2):

    figure = plt.figure(figsize=(f1,f2))

    columns = [i for i in var.columns]

    for clmn in columns:

        ax = figure.add_subplot(ax1,ax2,columns.index(clmn)+1)

        plt.bar(var.index,var[clmn],label = clmn)

        plt.legend(loc="upper right")

        plt.xticks(rotation=45)

    plt.show()

    plt.tight_layout()

    

barplot(by_age,12,6,2,2)
barplot(by_position,25,14,3,2)
barplot(by_skillmove,12,8,2,2)
data.describe()
#And now,I am going to start with mean visualization with using boxplot



dscrb = data.corr()



ftr_list = [i for i in dscrb.columns]

ftr_list.remove("Unnamed: 0")

ftr_list.remove("ID")



def boxplot_creator(ftr_li):

    figure = plt.figure(figsize=(20,20))

    for i in range(0,len(ftr_li)):

        ax = figure.add_subplot(6,7,i+1)

        data.boxplot(column=ftr_li[i],grid=True)

    plt.show()

    plt.tight_layout()



boxplot_creator(ftr_list)

numerical_features = [i for i in data.describe().columns ]

numerical_features.remove("Unnamed: 0")

numerical_features.remove("ID")

colors = ["Red","Blue","Green","Brown","Cyan","Pink","Purple"]



def histogram_creator(ftr_list,color_li):

    

    figure = plt.figure(figsize=(25,25))   

    clr_nmb = 0

    for i in range(0,len(ftr_list)):       

        if clr_nmb == 6:

            clr_nmb = 0 

        ax = figure.add_subplot(6,7,i+1)            

        data[ftr_list[i]].plot(kind="hist",bins=30,label=ftr_list[i],color=color_li[clr_nmb])

        plt.legend(loc = "upper right")

        clr_nmb+=1

    plt.show()

    plt.tight_layout()



histogram_creator(numerical_features,colors)
#Overall Comparing by Countries

country_list = []

for i in data["Nationality"]:

    if i not in country_list:

        country_list.append(i)

        

        

#print(len(country_list)) Wow, there are 164 countries in the list 

""" There are too many countries in the list so I decided to divide the list into a few pieces """



part1 = country_list[:20]

part2 = country_list[20:40]

part3 = country_list[40:60]

part4 = country_list[60:80]

part5 = country_list[80:100]

part6 = country_list[100:120]

part7 = country_list[120:140]

part8 = country_list[140:]





part1_meanlist = []

part2_meanlist = []

part3_meanlist = []

part4_meanlist = []

part5_meanlist = []

part6_meanlist = []

part7_meanlist = []

part8_meanlist = []





""" Mean Calculating """

for ctr in part1:

    datafilter = data["Nationality"] == ctr

    part1_meanlist.append(np.mean(data[datafilter]["Overall"])) 



for ctr in part2:

    datafilter = data["Nationality"] == ctr

    part2_meanlist.append(np.mean(data[datafilter]["Overall"]))



for ctr in part3:

    datafilter = data["Nationality"] == ctr

    part3_meanlist.append(np.mean(data[datafilter]["Overall"])) 



for ctr in part4:

    datafilter = data["Nationality"] == ctr

    part4_meanlist.append(np.mean(data[datafilter]["Overall"])) 

    

for ctr in part5:

    datafilter = data["Nationality"] == ctr

    part5_meanlist.append(np.mean(data[datafilter]["Overall"])) 

    

for ctr in part6:

    datafilter = data["Nationality"] == ctr

    part6_meanlist.append(np.mean(data[datafilter]["Overall"])) 



for ctr in part7:

    datafilter = data["Nationality"] == ctr

    part7_meanlist.append(np.mean(data[datafilter]["Overall"]))



for ctr in part8:

    datafilter = data["Nationality"] == ctr

    part8_meanlist.append(np.mean(data[datafilter]["Overall"])) 



""" Plotting  """



_,ax = plt.subplots(figsize=(10,10))

plt.plot(part1,part1_meanlist,color="Green",linewidth=5)

plt.xticks(rotation=90)

plt.show()



_,ax = plt.subplots(figsize=(10,10))

plt.plot(part2,part2_meanlist,color="Blue",linewidth=5)

plt.xticks(rotation=90)

plt.show()



_,ax = plt.subplots(figsize=(10,10))

plt.plot(part3,part3_meanlist,color="Red",linewidth=5)

plt.xticks(rotation=90)

plt.show()



_,ax = plt.subplots(figsize=(10,10))

plt.plot(part4,part4_meanlist,color="Cyan",linewidth=5)

plt.xticks(rotation=90)

plt.show()



_,ax = plt.subplots(figsize=(10,10))

plt.plot(part5,part5_meanlist,color="Pink",linewidth=5)

plt.xticks(rotation=90)

plt.show()



_,ax = plt.subplots(figsize=(10,10))

plt.plot(part6,part6_meanlist,color="Purple",linewidth=5)

plt.xticks(rotation=90)

plt.show()



_,ax = plt.subplots(figsize=(10,10))

plt.plot(part7,part7_meanlist,color="Brown",linewidth=5)

plt.xticks(rotation=90)

plt.show()



_,ax = plt.subplots(figsize=(10,10))

plt.plot(part8,part8_meanlist,color="Black",linewidth=5)

plt.xticks(rotation=90)

plt.show()
#As we can saw the data is listed starting from high by overall.

players = data.head(10)

player_names = [i for i in players["Name"]]

player_ages = [i for i in players["Age"]]

player_overalls = [i for i in players["Overall"]]

player_potentials = [i for i in players["Potential"]]

player_positions = [i for i in players["Position"]]

ftr_names = ["Player Ages","Player Overalls","Player Potentials","Player Positions"]

colors = ["Blue","Pink","Green","Cyan"]



sum_list = [player_ages,player_overalls,player_potentials,player_positions]



def create_barplots(sum_li,ftr_li,color_li):

    for i in range(0,len(sum_li)):

        _,ax = plt.subplots(figsize=(10,10))

        ax.bar(player_names,sum_li[i],label=ftr_li[i],color=color_li[i])

        plt.legend(loc="upper right")

        plt.show()

create_barplots(sum_list,ftr_names,colors)
def sort(sort_li):

    first_five= list(np.zeros(5))

    while 0 in first_five:

        for i in sort_li:

            

            if i>first_five[0]:

                first_five[0] = i

                

            elif i>first_five[1] and i not in first_five[:1]:

                first_five[1] = i

                

            elif i>first_five[2] and i not in first_five[:2]:

                first_five[2] = i

                

            elif i>first_five[3] and i not in first_five[:3]:

                first_five[3]=i

                

            elif i>first_five[4] and i not in first_five[:4]:

                first_five[4]=i

    

    return first_five





height_data = data.Height #We can take the data but these data's height types are inches.

height_data.dropna(inplace=True)



def convert_to_cm(dt):

    converted_height_list = []

    for height in dt:

        height_spl = height.split("'")

        #A feet is equals 30.48 cm and an inch is equals 2.54 cm

        value1 = float(height_spl[0])*30

        value2 = float(height_spl[1])*2

        converted_height_list.append(value1+value2) 

    return converted_height_list



nm = convert_to_cm(height_data)

nm =sort(nm)



print(nm)
wh_list = (data.Weight) #We take weight data but we have to clear NaN values

wh_list.dropna(inplace=True)

wh_list = list(wh_list)





def delete_lbs(wh_li):

    wh_int = []

    for w in wh_li:

        w = w.replace("lbs","")

        wh_int.append(int(w))

    return wh_int



wh_list = delete_lbs(wh_list)

first_five = sort(wh_list)

print(first_five) #Let's identify the players
data = pd.read_csv("/kaggle/input/fifa19/data.csv")
import warnings

warnings.filterwarnings("ignore")





data_filter_w = (data["Weight"] == str(first_five[0])+"lbs") | (data["Weight"]==str(first_five[1])+"lbs") | (data["Weight"]==str(first_five[2])+"lbs") | (data["Weight"]==str(first_five[3])+"lbs") | (data["Weight"]==str(first_five[4])+"lbs")

players = data[data_filter_w]







players = players.loc[:,["Name","Age","Overall","Potential","Value","Release Clause","Weight"]]

players = players.dropna()





data_clause = [i for i in players["Release Clause"]]

data_value = [i for i in players["Value"]]





def currency_cleaner(data_li):

    clean_data_li = []

    for dt in data_li:

        dt = dt.replace("€","")

        if "K" in dt:

            dt_clean=dt.replace("K","000")

        

        elif "M" in dt:

            dt_clean = dt.replace("M","000000")

        

        if "." in dt_clean:

            dt_clean = dt_clean.replace(".","")

            dt_clean = int(dt_clean)

            dt_clean = dt_clean/10

        

        else:

            dt_clean = int(dt_clean)

        clean_data_li.append(dt_clean)

    return clean_data_li



data_clause = currency_cleaner(data_clause) #Release clause data is ready!



data_value = currency_cleaner(data_value) #Value data is ready!



players["Release Clause"] = data_clause

players["Value"] = data_value



#I had to group data by weight because in grp_data there are players more than five 

grp_data = players.groupby("Weight").mean()

grp_data

#And now, we are ready to visualization!

feature_names = [i for i in grp_data]

weights = ["214lbs","225lbs","234lbs","236lbs","243lbs"]

weight_count = [i for i in players["Weight"].value_counts()]

colors = ["Green","Red","Blue","Pink","Purple"]

feature_list = list(zip(feature_names,colors))



def line_plot_creator(ftr_names):

    figure = plt.figure(figsize = (13,13))

    for ftr,color in ftr_names:

        ax = figure.add_subplot(3,2,ftr_names.index((ftr,color))+1)

        plt.plot(grp_data.index,grp_data[ftr],color=color,label=ftr,linewidth=3)

        plt.legend(loc="upper right")

    ax = figure.add_subplot(3,2,6)

    plt.plot(weights,weight_count,color="Cyan",label="Weight Counts",linewidth=3)

    plt.legend(loc="upper right")

    plt.show()

    plt.tight_layout()

        

line_plot_creator(feature_list)
data = pd.read_csv("/kaggle/input/fifa19/data.csv")



df = data.loc[:,["Name","Age","Overall","Potential","Value","Release Clause","Preferred Foot"]]



df = df.dropna()



value_list = [i for i in df["Value"]]

release_list = [i for i in df["Release Clause"]]

value_list = currency_cleaner(value_list)

release_list = currency_cleaner(release_list)



df["Value"] = value_list

df["Release Clause"] = release_list



grp = df.groupby("Preferred Foot").mean()

grp
ftr_names = [i for i in grp.columns]

colors = ["Blue","Red","Pink","Brown","Green"]

ftr_list = list(zip(ftr_names,colors))

foot_counting = [i for i in df["Preferred Foot"].value_counts()]



figure = plt.figure(figsize=(10,10))

plt.pie(foot_counting,labels=["Right","Left"],autopct="%1.1f%%")

plt.title("Players' Foot Preferring")

plt.show()
plt.style.use("seaborn-whitegrid")

def create_line_plots(ftr_li):

    figure = plt.figure(figsize=(20,20))

    for ftr,clr in ftr_li:

        ax = figure.add_subplot(3,2,ftr_li.index((ftr,clr))+1)

        plt.title(ftr)

        plt.bar(grp.index,grp[ftr],color=clr,label=ftr,width=0.2)

        plt.legend(loc="upper right")

    plt.show()

    plt.tight_layout()



create_line_plots(ftr_list)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

import warnings

warnings.filterwarnings('ignore')

# Load the data set

df = pd.read_csv("../input/train.csv")
(df["Age"].describe())
df.head()
df["tot_rel"] = df["SibSp"]+df["Parch"]
pd.DataFrame(df.describe())
df_missing = df[df.isnull().any(axis=1)]

df_missing

print(df["Name"].isnull().sum())

print(df["Ticket"].isnull().sum())
fields = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"]

na_count = {}

for field in fields:

    na_count[field]=df[field].isnull().sum()

    

display(na_count)    
df.info()
%matplotlib inline

sns.boxplot(y=df["Age"])

plt.title("All Passengers")
df_male = df.groupby("Sex").get_group("male")

df_female = df.groupby("Sex").get_group("female")

fig = plt.figure()

plt.subplot(121)

sns.boxplot(y=df_male["Age"])

plt.title("Male Passengers")

plt.ylim(0,80)

plt.subplot(122)

sns.boxplot(y=df_female["Age"])

plt.title("Female Passengers")

plt.ylim(0,80)

fig.tight_layout()
fig = plt.figure()

fig.set_figheight(13)

fig.set_figwidth(13)

pclasses = [1,2,3]

plot_num_1 = 0

male_median_age_by_class = {}

female_median_age_by_class = {}

for pclass in pclasses:

 

    plot_num_1 = plot_num_1+1

    plt.subplot(2,3,plot_num_1)

    df_male_class = df_male.groupby("Pclass").get_group(pclass)

    sns.boxplot(y=df_male_class["Age"])

    title_string = "Class "+str(pclass)+" Male Passengers"

    plt.title(title_string)

    plt.ylim(0,80)

    male_median_age_by_class[pclass]=df_male_class["Age"].median()

    

    plot_num_2 = plot_num_1+3

    plt.subplot(2,3,plot_num_2)

    df_female_class = df_female.groupby("Pclass").get_group(pclass)

    sns.boxplot(y=df_female_class["Age"])

    title_string = "Class "+str(pclass)+" Female Passengers"

    plt.title(title_string)

    plt.ylim(0,80)

    female_median_age_by_class[pclass]=df_female_class["Age"].median()







fig.tight_layout()

display(male_median_age_by_class)

display(female_median_age_by_class)
for gender in ['male','female']:

    for pclass in [1,2,3]:

        if (gender=='male'):

            age_dict = male_median_age_by_class

        else:

            age_dict = female_median_age_by_class

        df.loc[(df['Sex']==gender)&(df['Pclass']==pclass)&(df['Age'].isnull()),'Age']=age_dict[pclass]

    
df["Age"].isnull().sum()
df["Embarked"].value_counts()/df.shape[0]
df["Embarked"].fillna("S",inplace=True)

df["Embarked"].isnull().sum()
%matplotlib inline

df["Sex"].value_counts().plot(kind="bar")
((df["Sex"].value_counts())*100/df.shape[0]).plot(kind="bar",title="All Passengers")
(df["Pclass"].value_counts()/df.shape[0]).plot(kind="Bar",title="Passenger Class (All Passengers)")
fig = plt.figure()

plt.subplot(121)

(df_female["Pclass"].value_counts()/df_female.shape[0]).plot(kind="bar",title="Passenger Class (Females)")

plt.subplot(122)

(df_male["Pclass"].value_counts()/df_male.shape[0]).plot(kind="bar",title="Passenger Class (Males)")
df_group_gender = df.groupby("Sex")

df_female  = df_group_gender.get_group("female")

df_male  = df_group_gender.get_group("male")

df_survived = df.groupby("Survived").get_group(1)

df_not_survived  = df.groupby("Survived").get_group(0)



((df_survived["Sex"].value_counts())/df_survived.shape[0]).plot(kind="bar",title="Survivors")

#ax.set_xticklabels(['Survived','Not Survived'])
df_survived_male = df_survived.groupby("Sex").get_group("male")

df_survived_female = df_survived.groupby("Sex").get_group("female")

                                                    

order = [1,0] #To ensure the order of plotting as survived, not survived in the bar chart

fig = plt.figure()

plt1 = plt.subplot(121)

plt1 = ((df_female["Survived"].value_counts())/df_female.shape[0]).ix[order].plot(kind="bar",title="Female Passengers")

plt1.set_xticklabels(['Survived','Not Survived'])

plt2 = plt.subplot(122)

plt2 = ((df_male["Survived"].value_counts())/df_male.shape[0]).ix[order].plot(kind="bar",title="Male Passengers")

plt2.set_xticklabels(['Survived','Not Survived'])
fig1,ax1 = plt.subplots(1,2) 

df.hist("Age",by="Sex",ax=ax1,color='orange',bins=(0,10,20,30,40,50,60,70,80,90))

df_survived.hist("Age",by="Sex",ax=ax1,color='green',bins=(0,10,20,30,40,50,60,70,80,90))

#df.plot("Age",kind='hist',ax=ax1)

#df_survived.hist("Age",by="Sex",ax=ax1)

plt.text(100,100,"All Passengers",color='Black',fontsize=10,backgroundcolor="Orange")

plt.text(100,90,"Survivors         ",color='Black',fontsize=10,backgroundcolor="green")

#Since we will be plotting a number of stacked bar charts from now, let us define a function for the same.

def plot_stacked_bar_chart(data1,data2,order,chart1_color,chart2_color):

    data1.ix[order].plot(kind="bar",color = chart1_color)

    data2.ix[order].plot(kind="bar",color = chart2_color,stacked="True")
order = [1,2,3]

fig = plt.figure()



plt1 = plt.subplot(121)

plot_stacked_bar_chart(df_male["Pclass"].value_counts(),df_survived_male["Pclass"].value_counts(),order,"Orange","green")

plt.title("Male Passengers by Class")



plt2 = plt.subplot(122)

plot_stacked_bar_chart(df_female["Pclass"].value_counts(),df_survived_female["Pclass"].value_counts(),order,"Orange","green")

plt.title("Female Passengers by Class")





plt.text(2.75,150,"All Passengers",color='Black',fontsize=10,backgroundcolor="Orange")

plt.text(2.75,140,"Survivors         ",color='Black',fontsize=10,backgroundcolor="green")
fig = plt.figure()

fig.set_figheight(10)

fig.set_figwidth(8)



title_dict = {}

fields = ["tot_rel","SibSp","Parch"]

title_base_string_1 = "Survival (Males) according \nto number of \n"

title_base_string_2 = "Survival (Females) according \nto number of \n"

title_dict = {"tot_rel":"Relatives","SibSp":"Siblings","Parch":"Parents & Children"} # To set plot titles

plot_num = 0

order1 = [0,1,2,3,4,5,6] #passed into function plotting bar chart, to set the order of x-values of the chart

for field in fields:

    plot_num = plot_num+1

    plt.subplot(3,2,plot_num)

    plot_stacked_bar_chart(df_male[field].value_counts(),df_survived_male[field].value_counts(),order1,"Orange","green")

    title_string = title_base_string_1+title_dict[field]

    plt.title(title_string)

    

    plot_num = plot_num+1

    plt.subplot(3,2,plot_num)

    plot_stacked_bar_chart(df_female[field].value_counts(),df_survived_female[field].value_counts(),order1,"Orange","green")

    title_string = title_base_string_2+title_dict[field]

    plt.title(title_string)

    

plt.tight_layout()

#plt.text(10,0.8,"All Passengers",color='Black',fontsize=14,backgroundcolor="blue")

#plt.text(10,0.7,"Survivors    ",color='Black',fontsize=14,backgroundcolor="red")



fig = plt.figure()

fields = ["tot_rel","SibSp","Parch"]

title_base_string = "Survival Rate according \nto number of \n"

title_dict = {"tot_rel":"Relatives","SibSp":"Siblings","Parch":"Parents & Children"} # To set plot titles

plots = [1,2,3]

order1 = [0,1,2,3,4,5,6] #passed into function plotting bar chart, to set the order of x-values of the chart



for (i,field) in zip(plots,fields):

    plt.subplot(1,3,i)

    data1 = df_survived_female[field].value_counts()/df_female[field].value_counts()

    data2 = df_survived_male[field].value_counts()/df_male[field].value_counts()

    plot_stacked_bar_chart(data1,data2,order1,"blue","red")

    title_string = title_base_string+title_dict[field]

    plt.title(title_string)

plt.tight_layout()



plt.text(10,0.8,"Female",color='Black',fontsize=14,backgroundcolor="blue")

plt.text(10,0.7,"Male    ",color='Black',fontsize=14,backgroundcolor="red")
fig = plt.figure()

data_1 = (df_survived_female["Embarked"].value_counts())/(df_female["Embarked"].value_counts())

data_2 = (df_survived_male["Embarked"].value_counts())/(df_male["Embarked"].value_counts())

order = ['S','C','Q']

plot_stacked_bar_chart(data_1,data_2,order,'Blue','Red')

plt.title("Survival Rate by Port of Embarkation")



plt.text(3,0.8,"Female",color='Black',fontsize=14,backgroundcolor="blue")

plt.text(3,0.7,"Male    ",color='Black',fontsize=14,backgroundcolor="red")



display('Survival rate of female passengers according to port of embarkation:(S=Southampton, C=Cherbourg, Q=Queenstown) : ')

display(df_survived_female["Embarked"].value_counts()/df_female["Embarked"].value_counts()) #survival rate among females according to Port of Embarkation

display("Survival rate of female passengers in the population: ")

display(df_survived_female.shape[0]/df_female.shape[0])
display("Female Survivors")

display(df_survived_female["Embarked"].value_counts()) 

display("Female Passengers")

display(df_female["Embarked"].value_counts())
p  = 0.7420382165605095

q  = 1 - p

n  = 73

x = 64



z = (x - n*p)/pow((n*p*q),0.5)

display(z)
display('Survival rate of Male passengers according to port of embarkation:(S=Southampton, C=Cherbourg, Q=Queenstown) : ')

display(df_survived_male["Embarked"].value_counts()/df_male["Embarked"].value_counts()) #survival rate among females according to Port of Embarkation

display("Survival rate of Male passengers in the population: ")

display(df_survived_male.shape[0]/df_male.shape[0])



display("Male Survivors")

display(df_survived_male["Embarked"].value_counts()) 

display("Male Passengers")

display(df_male["Embarked"].value_counts())
p  = 0.18890814558058924

q  = 1 - p

n  = 95

x = 29



z = (x - n*p)/pow((n*p*q),0.5)

display(z)
fig = plt.figure()

fig.set_figheight(14)

fig.set_figwidth(12)

order = [1,2,3]

ports = ["S","C","Q"]

port_names = {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}

title_base = "Passenger class for Port of Embarkation: "

plot_num = 0

for port in ports:

    plot_num = plot_num+1

    title_string = title_base+port_names[port]+"(Female)"

    plt.subplot(3,2,plot_num)

    data_1 = df_female.groupby("Embarked").get_group(port)["Pclass"].value_counts()

    data_1.ix[order].plot(kind="Bar")

    plt.title(title_string)

    

    plot_num = plot_num+1

    title_string = title_base+port_names[port]+"(Male)"

    plt.subplot(3,2,plot_num)

    data_2 = df_male.groupby("Embarked").get_group(port)["Pclass"].value_counts()

    data_2.ix[order].plot(kind="Bar")

    plt.title(title_string)

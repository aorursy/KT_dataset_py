# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib as mpl



import seaborn as sns

import plotly.graph_objects as go



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/homicide-reports/database.csv")

df = data.copy()

df.head()
df.tail()
df.info()
df.describe().T
# df[df["Incident"]>900].sort_values(by="Incident")

# incident sütununda 900lü sayılar neden var bunu anlamaya çalışıyorum.
df[(df["Agency Name"]=="Los Angeles") & (df["Agency Type"]=="Sheriff") & (df["Year"]==1980) & (df["Month"]=="August")].sort_values(by="Incident")
df[(df["Agency Name"]=="Miami") & (df["Agency Type"]=="Municipal Police") & (df["Year"]==2006) & (df["Month"]=="December")].sort_values(by="Incident")
#df[(df["Agency Name"]=="Orange County") & (df["Year"]==2013) & (df["Month"]=="April")].sort_values(by="Incident")
df["Incident"] = 1
df["Victim Sex"].value_counts()

#df["Victim Sex"].value_counts().plot.bar();
df["Perpetrator Sex"].value_counts()

#df["Perpetrator Sex"].value_counts().plot.bar();
def bar_plot(variable):

    

    var = df[variable]

    

    varValue = var.value_counts()

        

    plt.figure(figsize = (9,3))

    plt.bar(varValue.index, varValue)

    plt.xticks(varValue.index, varValue.index.values)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show

    #print("{}: \n {}".format(variable, varValue))

    

category1=["Victim Sex","Perpetrator Sex"]

list(map(lambda x:bar_plot(x), category1))

plt.show()



# Kategorik değişkenler için
df_known = df[(df['Victim Sex'] != 'Unknown') & (df['Perpetrator Sex'] != 'Unknown')]

df_gender = df_known.groupby(['Perpetrator Sex','Victim Sex']).size()

df_gender_df = df_gender.to_frame(name='Count')

df_gender_df.reset_index(inplace=True)

df_gender_df
plt.figure(figsize = (9,3))

clrs = ['#EE99AC',"#6890F0"]

sns.barplot(x='Perpetrator Sex', y="Count", hue='Victim Sex', palette=clrs, data=df_gender_df);

plt.title("A Comparison of Men and Women Who are Perpetrator and Victim");





# plt.figure(figsize = (9,3))

# sns.countplot(x='Perpetrator Sex', data=df, hue='Victim Sex');

# plt.title("A Comparison of Men and Women Who are Perpetrator and Victim");
def hist_plot(variable):

    

    plt.figure(figsize=(9,3))

    plt.hist(df[variable],bins = 50)

    

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} Distribution with histogram". format(variable))

    plt.show

numericalVariables=["Victim Age", "Victim Count"]

list(map(lambda x: hist_plot(x), numericalVariables))

plt.show()



#Sayısal değişkenler için.
df["Incident"] = 1

age = pd.cut(df["Victim Age"],[0,18,36,54,90])



df_age = df.pivot_table(["Incident"], index =["Victim Sex",age], aggfunc=sum)

df_age
df_age_f = pd.DataFrame({"0-18":df_age.iloc[0],

                         "18-36":df_age.iloc[1],

                        "36-54":df_age.iloc[2],

                        "54-90":df_age.iloc[3]})



                                    

df_age_f = df_age_f.T

df_age_f.columns = ["Incident"]



plt.figure(figsize = (9,6))

plt.subplot(2,2,1)

sns.barplot(x=df_age_f.index, y="Incident",data=df_age_f);

plt.title("Ages of Female Victims");

plt.xlabel("Age Groups");



##

df_age_m = pd.DataFrame({"0-18":df_age.iloc[4],

                         "18-36":df_age.iloc[5],

                        "36-54":df_age.iloc[6],

                        "54-90":df_age.iloc[7]})

df_age_m = df_age_m.T

df_age_m.columns = ["Incident"]



plt.figure(figsize = (9,6))

plt.subplot(2,2,2)

sns.barplot(x=df_age_m.index, y="Incident",data=df_age_m);

plt.title("Ages of Male Victims");

plt.xlabel("Age Groups");
df_age_b = pd.concat([df_age_m,df_age_f],axis=1)

df_age_b.columns = ["Male", "Female"]

df_age_b.plot.bar();

plt.title("Ages of Victims");
df["Weapon"].value_counts()
df_we = df.pivot_table(index="Year", columns="Weapon", aggfunc=sum)["Incident"]

df_we
plt.figure(figsize=(9,3))

sns.barplot(data=df_we);

plt.xticks(rotation=90);



# df["Weapon"].value_counts().plot.pie();
labels = df.groupby('Weapon').sum().index



explode = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

sizes = df.groupby('Weapon').sum()['Incident']



plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, labeldistance=1.5, autopct='%1.1f%%')

plt.title('Weapons',fontsize = 15);
w = df.groupby('Weapon').sum()['Incident']

w = pd.DataFrame(w)

#w

weapons = list(df["Weapon"].unique())



others = []

for i in weapons:

    if w.loc[i].values < 7000:

        others.append(w.loc[i].values)    

    

w.loc["Others"] = sum(others)

w_elim = w[w.Incident > 7000] 

#w_elim #eliminated



labels = w_elim.index

explode = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

sizes = w_elim['Incident']



plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%')

plt.title('Weapons',fontsize = 15);
plt.figure(figsize = (18,8))

# df["State"].value_counts().plot.bar();



states_vis = sns.countplot(x='State', order=df['State'].value_counts().index, data=df);



for item in states_vis.get_xticklabels():

    item.set_rotation(90)
city_usa = pd.read_csv("/kaggle/input/usa-cities/usa.txt")

#city_usa.head()



new = []

for i in list(city_usa.name):

    new.append(i.strip())

    

city_usa["name"] = new

city_usa.head()
cities = df.groupby("City").size()

cities_df = cities.to_frame(name='count')

cities_df.reset_index(inplace=True)

cities_df.columns=(["name","count"])

cities_df.head()



# cities = df.groupby("City")["Incident"].sum()

# cities_df = pd.DataFrame(cities)

# cities_df
df_map = pd.merge(cities_df, city_usa, on = "name", how='inner')

df_map.head()
df_map['text'] = df_map['name'] + ' - Number of Incident: ' + (df_map['count'].astype(str)) 

limits = [0,100,500,1000,50000]

colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]

cities = []



fig = go.Figure()

a= -4

for i in range(len(limits)):



    df_sub = df_map[(df_map["count"]>limits[i]) & (df_map["count"]<limits[a])]

     

    fig.add_trace(go.Scattergeo(

        locationmode = 'USA-states',

        lon = df_sub['lon'],

        lat = df_sub['lat'],

        text = df_sub['text'],

        marker = dict(

            size = df_map["count"],

            color = colors[i],

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        ),

        name = '{0} - {1}'.format(limits[i],limits[a])))

    a +=1

fig.update_layout(

        title_text = 'Homicide Numbers of USA Cities',

        showlegend = True,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(217, 217, 217)',

        )

    )



fig.show()
df["Relationship"].value_counts()
a = df["Relationship"].replace('Unknown', np.nan)

a.value_counts().plot.barh();



# sns.countplot(x='Relationship', data=df.replace('Unknown', np.nan), order=df["Relationship"].value_counts().index)



#df["Relationship"].value_counts().plot.barh();

# bu şekilde yapınca unknown lar çok olduğu için anlamsız görünüyor.
df["Perpetrator Race"].value_counts().plot.bar();
df_race_pv = df.pivot_table(index=["Perpetrator Race","Victim Race"], aggfunc=sum)["Incident"]

df_race_pv = pd.DataFrame(df_race_pv)

df_race_pv.head()
df_race = pd.DataFrame({"White-White":df_race_pv.loc["White","White"].values,

                 "White-Black":df_race_pv.loc["White","Black"].values,

                 "Black-White":df_race_pv.loc["Black","White"].values,

                 "Black-Black":df_race_pv.loc["Black","Black"].values})



df_race = df_race.T

df_race.columns = ["Incident"]



sns.barplot(x=df_race.index, y="Incident", data=df_race);

plt.title("Relationship Between Black and White Races");

plt.xlabel("Perpetrator Race - Victim Race");
df.groupby("Year")["Month"].value_counts()

# her ay kaç cinayet işlenmiş onu buluyoruz.



# df.groupby("Year")["Month"].value_counts().sum() 

# sağlamasını yapmış oldu. toplam satır sayısına eşit oluyor.
df_pv = df.pivot_table(index=["Year"], columns=["Month"], aggfunc=(lambda x : x.count()), margins = True)["Incident"]

# sütunları alfabetik sıraya göre sıralıyordu onu değiştirdik.

df_pv = df_pv.reindex(columns=["January","February","March","April","May","June","July","August","September","October","November","December","All"])

df_pv
df_pv = df.pivot_table(index=["Year"], columns=["Month"],aggfunc=(lambda x : x.count()))["Incident"]

df_pv = df_pv.reindex(columns=["January","February","March","April","May","June","July","August","September","October","November","December"])



df_pv.plot();

#zaman serisi grafiği çizme
sns.heatmap(df_pv);
df_year = df.groupby("Year")["Incident"].sum()

df_year = pd.DataFrame(df_year)



plt.title("Number of Homicide by Years");

sns.lineplot(x=df_year.index, y= "Incident", data=df_year);
df_s_y = df.pivot_table(index=["Victim Sex","Year"], aggfunc=sum)["Incident"]

df_s_y = pd.DataFrame(df_s_y)

df_s_y.head()

# df_s_y.loc["Female",1980].values

# df_s_y.loc["Female",1980].index
def gender_plot(sex):

    data = df_s_y.loc[sex]



    plt.plot(data.index, data.values)

    

# gender_plot("Female")

genders = ["Male","Female"]



plt.figure(figsize = (18, 8))

for i in genders:

    gender_plot(i)



plt.title("Homicide Victims by Years and Genders");

plt.legend(genders);
df_s_r_y = df.pivot_table(index=["Victim Sex","Relationship","Year"], aggfunc=sum)["Incident"]

df_s_r_y = pd.DataFrame(df_s_r_y)

df_s_r_y.head()
plt.figure(figsize = (18, 8))



def sex_rel(sex, relation):

    data = df_s_r_y.loc[sex, relation]



    plt.plot(data.index, data.values)

    

relations = ["Wife","Ex-Wife","Girlfriend","Mother","Sister","Daughter"]

             

for relation in relations:

    sex_rel("Female", relation)



plt.title("Number of Female Homicide Victims by Years");

plt.xlabel("Year");

plt.ylabel("Numbers of Incident");

plt.legend(relations);
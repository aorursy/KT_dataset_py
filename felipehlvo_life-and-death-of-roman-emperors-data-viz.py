# data scraping
import requests
import urllib.request
from bs4 import BeautifulSoup

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# operations
import numpy as np
# List of Roman Emperors from Wikipedia
website_url = requests.get(
    'https://en.wikipedia.org/wiki/List_of_Roman_emperors').text
soup = BeautifulSoup(website_url, "html")
# finding the table I want
my_tables = soup.findAll('table', {"class": 'wikitable'})
# Scraping Wikipedia Page and saving each column in a different variable

A = []
B = []
C = []
D = []
E = []
F = []
G = []
H = []
for table in my_tables:
    for row in table.findAll('tr'):
        cells = row.findAll('td')
        if len(cells) == 7:
            death = cells[6].findAll(text=True)
            death_text = ""
            for i in death:
                death_text = death_text + " " + i
            A.append(cells[0].find(text=True))
            B.append(cells[1].find(text=True))
            C.append(cells[2].find(text=True))
            D.append(cells[3].find(text=True))
            F.append(cells[5].find(text=True))
            reign = cells[4].findAll(text=True)
            reign_text = ""
            for i in reign:
                reign_text = reign_text + " " + i
            E.append(reign_text)

            G.append(death_text)
#Creating DataFrame

df = pd.DataFrame(B, columns=['Name'])
df['Birth'] = C
df['Succession'] = D
df['Reign'] = E
df['Time'] = F
df["Deaths"] = G
df.head()
df.to_csv("romans.csv", index=False)
#make all strings lowercase except the names for easier data extraction
for i in df.columns.values:
    if i != "Name":
        df[i] = df[i].str.lower()
        
#drop Succession column
df = df.drop(df.columns[[2]], axis=1)
df.to_csv(index=False)
df.head()
## Parsing through the Death column to find out cause of death
# Disclaimer: some rows contain more than one cause of death,
# which means that some information might be lost or misinterpreted


deaths = []
for i in df.Deaths:
    if "assassin" in i or "murdered" in i or ("killed" in i and not "battle" in i):
        deaths.append("Assassinated")
    elif "natural" in i:
        deaths.append("Natural Causes")
    elif "suicide" in i:
        deaths.append("Suicide")
    elif "executed" in i or "beheaded" in i:
        deaths.append("Executed")
    elif "battle" in i:
        deaths.append("Killed in Battle")
    elif "poison" in i:
        deaths.append("Poisoned")

    elif "unknown" in i or i[-2] == ")" or "constantinople" in i:
        deaths.append("Other/Unknown")
    elif "illness" in i:
        deaths.append("Illness")
    elif any(disease in i for disease in ["tuberculosis", "edema", "gout", 'carbuncle', 'dysentery', "epilepsy" ]):
        deaths.append("Illness")
    else:
        deaths.append("Other/Unknown")

df["Cause"] = deaths
# Parsing through the Age column and getting only the age at death

ages = []
for i in df.Deaths:
    if "age" in i:
        a = i.split("age", 1)
        b = a[-1].split(")")
        c = b[0].split(" ")
        d = c[-1].split("-")
        e = d[0].split("~")
        f = e[-1].split("/")
        try:
            age = int(f[-1])
            ages.append(age)
        except:
            ages.append("Unknown")
            pass
    else:
        ages.append("Unknown")

df["Age"] = ages
# Parsing through the Reign column and getting only the year when reign ended

reign_end = []
for i in df.Reign:
    end = "Unknown"
    if "ad" not in i:
        a = i.split()
        b = a[-1].split("?")
        c = b[-1].split("–")
        try:
            end = int(c[-1])
        except:
            try:
                a = i.split("(")
                b = a[-2].split()
                end = int(b[-1])
            except:
                print(a,b)
                print("")
                pass
            pass
    else:
        a = i.split()
        end = int(a[-2])
    reign_end.append(end)   

df["End of Reign"] = reign_end
# Parsing through the Reign column and getting only the year of birth
births = []
for i in df.Birth:
    if "?" in i:
        year = "Unknown"
    elif "ad," in i:
        a = i.split(" ad", 1)
        b = a[0].split()
        year = int(b[-1])
    elif "bc," in i:
        a = i.split(" bc", 1)
        b = a[0].split()
        year = -int(b[-1])
    elif "," in i:
        a = i.split(",")
        b = a[-2].split()
        c = b[-1].split(".")
        d = c[-1].split("/")
        year = int(d[-1]) 
    elif "c." in i:
        a = i.split(".")
        b = a[-1].split()
        try:
            year = int(b[0])
        except:
            year = "Unknown"
            pass
    else:
        a = i.split()
        try:
            year = int(a[-1])
        except:
            year = "Unknown"
            pass
    births.append(year)

#fixing wrong entry    
births[71] = 384    
df["Births"] = births
# Parsing through the Time in Office column to get only numeric value
lengths = []

for i in df.Time:
    year = 0
    month = 0
    if "year" in i:
        a = i.split()
        b = a[0].split("/")
        year = int(b[0])
        if "month" in i:
            d = i.split("month")
            e = d[0].split()
            month = int(e[-1])
        length = year
        lengths.append(length)
    elif "month" in i:
        a = i.split()
        b = a[0].split("–")
        month = int(b[0])
        length = year
        lengths.append(length)
    elif "day" in i:
        length = year
        lengths.append(length)
    else:
        lengths.append("Unknown")
df["Length"] = lengths
df.head()
# palette
green = '#35d0ba'
blue = "#00b8a9"
red = '#d92027'
orange = '#ff9234'
yellow = '#ffcd3c'
yellow2 = "#ffde7d"
beige = "#f8f3d4"
red2 = "#f6416c"
black = "#222831"
#plotting histogram of age at time of death
sns.set_style("white")
num_ages = []
for i in ages:
    if type(i) == int:
        num_ages.append(i)

fig, ax  = plt.subplots(figsize = (10,10))
plt.hist(num_ages, bins = 20, color = black)
ax.set(title = "Age at Death", ylabel = "Count", xlabel = "Time (years)")
plt.show()
#plotting time in ofice histogram

num_lenghts = []
for i in lengths:
    if type(i) == int:
        num_lenghts.append(i)

fig, ax = plt.subplots(figsize=(8, 8))
plt.hist(num_lenghts, bins=20, color = black)
ax.set(title = "Time in power", ylabel = "Count", xlabel = "Time (years)")

plt.show()
#plotting histogram of causes of death
sns.set(palette = "YlOrRd")
sns.countplot(y = 'Cause',
              data = df,
              order = df['Cause'].value_counts().index, palette = "YlOrRd_r")
#getting only rows where we have all of the numerical information
df0 = df.drop(df.columns[[1,2,3,4,5]], axis=1)
df1 = df0.iloc[:,1:].apply(pd.to_numeric, errors='coerce')
df1 = df1.dropna()

df1 = pd.concat([df.Name,df1], join = "inner", axis = 1)
df1 = pd.concat([df.Cause,df1], join = "inner", axis = 1)
df1.head()
#Fixing wrong entry
df1["End of Reign"][df1["Name"] == "Alexios I Komnenos"] = 1118
before = df1["End of Reign"] - df1.Births - df1.Length #time lived before becoming an emperor
middle = before + df1.Length + 0.5 #length of the reign
after = df1.Age + 0.5 #time lived after the reign ended (if any) 

df1["before"] = before
df1["middle"] = middle
df1["after"] = after

#percentage of emperors that lived after their reign
print(len(df1["after"][(df1["after"] - df1["middle"]) > 0]) / 131 * 100, "%")
sns.set_style("whitegrid")
sns.set()
fig, ax = plt.subplots(figsize=(6, 40))

sns.barplot(x="after", y="Name", data=df1, color=green, ci=None, label="After mandate")
sns.barplot(x="middle", y='Name', data=df1, color=red, ci=None, label = "During mandate")
sns.barplot(x="before", y='Name', data=df1, color=yellow, ci=None, label="Before mandate")

ax.set(xlabel="Age (years)")
ax.legend(ncol=3, bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
ax.set_title("Which Emperors lived the most after their mandate?", pad=40, fontsize = "x-large")
plt.show()
%matplotlib inline

# sorting dataframe by age at the time of death (looks pretty)
df3 = df1.sort_values(['Age'], ascending=False).reset_index(drop=True)


for i in set(deaths):
    fig, ax = plt.subplots(figsize=(5,
                                    max(len(df3[df3["Cause"] == i]) // 4, 2)))
    sns.barplot(x="after",
                y="Name",
                data=df3[df3["Cause"] == i],
                color=green,
                ci=None,
                label="After mandate")
    sns.barplot(x="middle",
                y='Name',
                data=df3[df3["Cause"] == i],
                color=red,
                ci=None,
                label="During mandate")
    sns.barplot(x="before",
                y='Name',
                data=df3[df3["Cause"] == i],
                color=yellow,
                ci=None,
                label="Before mandate")
    ax.set_title(i, pad=30)
    ax.set(xlabel="Time")
    ax.legend(ncol=3, bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    plt.show()
#making dataframe with only numeric values in Age column

hey = df["Age"]
age = hey.apply(pd.to_numeric, errors='coerce')
age = age.dropna()
age = age.to_frame()
# Appending column with age range
r_age = []
for i in age["Age"]:
    if i < 20:
        r_age.append("0-20")
    elif i < 30:
        r_age.append("20-30")
    elif i < 40:
        r_age.append("30-40")
    elif i < 50:
        r_age.append("40-50")
    elif i < 60:
        r_age.append("50-60")
    elif i < 70:
        r_age.append("60-70")
    else:
        r_age.append("70+")
        
age["Range"] = r_age
age = pd.concat([df.Cause,age], join = "inner", axis = 1)
age.head(10)
#How many people dies from each cause for each age range
young = []
for i in set(deaths):
    first = age[age.Range == "0-20"]
    first.head()
    young.append(len(first["Cause"][first["Cause"] == i]))
    
twenties = []
for i in set(deaths):
    first = age[age.Range == "20-30"]
    twenties.append(len(first["Cause"][first["Cause"] == i]))
    
tirties = []
for i in set(deaths):
    first = age[age.Range == "30-40"]
    tirties.append(len(first["Cause"][first["Cause"] == i]))
    
forties = []
for i in set(deaths):
    first = age[age.Range == "40-50"]
    forties.append(len(first["Cause"][first["Cause"] == i]))
    
fifties = []
for i in set(deaths):
    first = age[age.Range == "50-60"]
    fifties.append(len(first["Cause"][first["Cause"] == i]))
    
sixties = []
for i in set(deaths):
    first = age[age.Range == "60-70"]
    sixties.append(len(first["Cause"][first["Cause"] == i]))
    
old = []
for i in set(deaths):
    first = age[age.Range == "70+"]
    old.append(len(first["Cause"][first["Cause"] == i]))
#Creating dataframe out of the lists

lst = [young, twenties, tirties, forties, fifties, sixties, old]
percentages = pd.DataFrame(lst, columns=[i for i in set(deaths)], dtype=float)
percentages.head(10)
# Lists of normalized values for each type of death

totals = [
    i + j + k + l + m + n for i, j, k, l, m, n in zip(
        percentages["Assassinated"] +
        percentages["Poisoned"], percentages['Natural Causes'],
        percentages["Executed"], percentages["Killed in Battle"],
        percentages["Illness"], percentages["Suicide"])
]
assassinated = [
    i / j * 100
    for i, j in zip(percentages['Assassinated'] +
                    percentages["Poisoned"], totals)
]
natural = [i / j * 100 for i, j in zip(percentages['Natural Causes'], totals)]
executed = [i / j * 100 for i, j in zip(percentages['Executed'], totals)]
battle = [i / j * 100 for i, j in zip(percentages['Killed in Battle'], totals)]
illness = [i / j * 100 for i, j in zip(percentages['Illness'], totals)]
suicide = [i / j * 100 for i, j in zip(percentages['Suicide'], totals)]
# plot most common causes of death for each age range

fig, ax = plt.subplots(figsize=(9, 9))
barWidth = 0.85
names = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
r = [1, 2, 3, 4, 5, 6, 7]

plt.bar(r,
        assassinated,
        color=red,
        edgecolor='white',
        width=barWidth,
        label="Assassinated")
plt.bar(r,
        executed,
        bottom=assassinated,
        color="#fe346e",
        edgecolor='white',
        width=barWidth,
        label="Executed")
plt.bar(r,
        battle,
        bottom=[i + j for i, j in zip(assassinated, executed)],
        color=orange,
        edgecolor='white',
        width=barWidth,
        label="Killed in Battle")
plt.bar(r,
        suicide,
        bottom=[i + j + k for i, j, k in zip(assassinated, executed, battle)],
        color=yellow,
        edgecolor='white',
        width=barWidth,
        label="Suicide")
plt.bar(r,
        illness,
        bottom=[i + j + k + l for i, j, k, l in zip(assassinated, battle, suicide, executed)],
        color= green,
        edgecolor='white',
        width=barWidth,
        label="Illness")
plt.bar(r,
        natural,
        bottom=[i + j + k + l + m for i, j, k, l , m in zip(assassinated, suicide, battle, executed, illness)],
        color="#16817a",
        edgecolor='white',
        width=barWidth,
        label="Natural Causes")

ax.set_title("Most common cause of death by age", pad=40, fontsize = "x-large")
plt.xticks(r, names)
plt.xlabel("Age", fontsize = "x-large")
plt.ylabel("Cause of death (%)", fontsize = "x-large")
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), ncol=1, bbox_to_anchor=(1, 0.6), loc='lower left', fontsize='x-large')
palette = [red, "gray","blue","#fe346e", orange, yellow, green, "#16817a" ]
palette = ["gray", yellow, red,"#16817a", green, orange, "#fe346e", "#562349"]
palette = ["#16817a", red, yellow, "#fe346e", orange, "gray", green, "purple"]
sns.set(font_scale=1.3)
g = sns.relplot(x="End of Reign",
            y="Length",
            data=df1,
            s=300,
            height=10,
            hue="Cause",
            palette=palette,
            alpha=0.8)
g.set(ylabel = "Length of mandate", xlabel = "End of Reign (years ac.)", title = "Roman Emperors Scatterplot")
df1["Size"] = [i * 0.4 for i in df1["Age"]]
df1 = pd.concat([df.Reign,df1], join = "inner", axis = 1)

df1.head()
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.models import LabelSet
from bokeh.io import output_notebook
from bokeh.embed import file_html
from bokeh.resources import CDN

TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select"

categories = [
    'Assassinated', 'Other/Unknown', 'Poisoned', 'Killed in Battle',
    'Natural Causes', 'Executed', 'Illness', 'Suicide'
]

TOOLTIPS = [("Emperor", "@Name"), ("Reign", "@Reign"),("Age at death", "@Age"), ("Cause of death", "@Cause")]

source = ColumnDataSource(data=df1)
p = figure(title = "Roman Emperors - Life and Death", tools=TOOLS, tooltips=TOOLTIPS)

p.scatter(x="End of Reign",
          y="Length",
          source=source,
          line_color="black",
          color=factor_cmap('Cause', palette='Set1_8', factors=categories),
          fill_alpha=0.8,
          size="Size",
          legend_group="Cause")

output_file("roman_emperors.html", title="Roman Emperors")
output_notebook()

p.xaxis[0].axis_label = 'End of Reign (years ac.)'
p.yaxis[0].axis_label = 'Time in Power'

show(p)  # open a browser

#getting only rows where we have all of the information

trend_df = df[["Cause","End of Reign"]]
trend_df = trend_df.iloc[:,1:].apply(pd.to_numeric, errors='coerce')
trend_df = trend_df.dropna()



trend_df = pd.concat([df.Cause,trend_df], join = "inner", axis = 1)
trend_df = trend_df[trend_df["Cause"] != "Other/Unknown"]
trend_df = trend_df[trend_df["End of Reign"] < 500]
trend_df[80:150]
counts = [[], [], [], [], [], [], []]
deaths_lst = [
    "Assassinated", "Suicide", "Killed in Battle", "Executed", "Illness",
    "Poisoned", "Natural Causes"
]
for i in range(len(deaths_lst)):
    for b in range(0, 500, 50):
        counts[i].append(
            len(trend_df["Cause"][(trend_df["Cause"] == deaths_lst[i])
                                  & (b < trend_df["End of Reign"]) &
                                  (trend_df["End of Reign"] < b + 100)]))
assassinated, suicide, battle, executed, illness, poisoned, natural = counts
sns.set(palette = "Paired")
sns.set_style("white")

fig, ax = plt.subplots(figsize = (10,10))

for i in range(len(counts)):
    plt.plot(range(0, 500, 50), counts[i], label = deaths_lst[i], linewidth = 3)
    
ax.legend()
plt.xticks(range(0, 500, 50), [str(i) + " - " + str(i + 150) for i in range(0, 1500, 150)], fontsize = 12, rotation = 90)
plt.yticks(fontsize = 12)
ax.set_title("Cause of death trends until the end of the Western Empire", pad=40, fontsize = "large")
plt.xlabel("Time (years AD)", fontsize = "large")
plt.ylabel("Count", fontsize = "large")
plt.show()
# summing up violent deaths
violent = [a+b+c+d for a,b,c,d in zip(assassinated,battle,executed,poisoned)]
v_df = []
for i in range(10):
    v_df.append(violent)
# Creating DataFrame
heat = pd.DataFrame(v_df)
# Plotting heatmap
fig, ax  = plt.subplots(figsize = (15,1))
sns.heatmap(heat, robust=True, cmap='OrRd', yticklabels=False, cbar=False, xticklabels = False)
plt.xticks(range(11), range(0, 501, 50),fontsize = 13)
plt.xlabel("Time (years AC)",fontsize = 13)
plt.show()
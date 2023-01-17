import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
results = pd.read_csv("../input/stack-overflow-2018-developer-survey/survey_results_public.csv")
results.iloc[1, :]
from collections import Counter
student = Counter(np.array(results["Student"].dropna()))
labels = list(student)
count = [student[i] for i in labels]
# Data to plot
sizes = [i/sum(count)*360 for i in count]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Student?")
plt.axis('equal')
plt.show()
employment = Counter(np.array(results["Employment"].dropna()))
labels = list(employment)
count = [employment[i] for i in labels]
# Data to plot
sizes = [i/sum(count)*360 for i in count]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0.1, 0, 0, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Employment Type?")
plt.axis('equal')
plt.show()
country = Counter(np.array(results["Country"].dropna()))
labels = [i.upper() for i in list(country)]
count = [country[i] for i in labels]
countries = pd.read_csv("../input/counties-geographic-coordinates/countries.csv")
cont = countries.iloc[:, [1, 2]]
cont = np.array(cont)
country = np.array(countries.iloc[:, 3])
country = [i.upper() for i in country]

lats = []
longs = []
errs = []
for i in labels:
    try:
        b = country.index(i)
        lats.append(list(cont[b])[0])
        longs.append(list(cont[b])[1])
    except:
        errs.append("err")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.figure(figsize=(14, 8))
earth = Basemap()
earth.bluemarble(alpha=0.02)
earth.drawcoastlines()
earth.drawstates()
earth.drawcountries()
earth.drawcoastlines(color='#555566', linewidth=1)
plt.scatter(longs, lats, c='red',alpha=1, zorder=10)
plt.xlabel("Usage from countries")
plt.savefig('usage.png', dpi=350)
companySize = Counter(np.array(results["CompanySize"].dropna()))
labels = list(companySize)
count = [companySize[i] for i in labels]
# Data to plot
sizes = [i/sum(count)*360 for i in count]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
#explode = (0.1, 0, 0, 0, 0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Company Sizes?")
plt.axis('equal')
plt.show()
exercise = Counter(np.array(results["Exercise"].dropna()))
labels = list(exercise)
count = [exercise[i] for i in labels]
sizes = [i/sum(count)*360 for i in count]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0, 0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, labels=labels, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Are developers fit? Here's the exercise pattern")
plt.axis('equal')
plt.show()
#country vs education
cou_edu = results[["Country", "FormalEducation"]].dropna().groupby(["Country"])
list_cou_edu = list(cou_edu)
edu_countries = []
for i in list_cou_edu:
    a = { i[0]: list(i[1]["FormalEducation"]) }
    edu_countries.append(a)
import matplotlib.pyplot as plt
max_con = []
ctr = 0
min_con = []
ctr_min = 100
for i in edu_countries:
    for di in i:
        edu = i[str(di)]
        qual = Counter(np.array(edu))
        labels = list(qual)
        count = [qual[i] for i in labels]
        if(len(labels)>=ctr):
            ctr = len(labels)
            g = { "name": di, "quals": labels, "count": count }
            max_con.append(g)
        if(len(labels)<=ctr_min):
            ctr_min = len(labels)
            g = { "name": di, "quals": labels, "count": count }
            min_con.append(g)
        #explode = (0, 0, 0.1, 0)  # explode 1st slice
 
        # Plot
        if(sum(count) > 2000):
            name = "Qualifications in "+str(di)
            plt.rcdefaults()
            fig, ax = plt.subplots()
            ax.barh(np.arange(len(labels)), count, align='center',
                    color='orange', ecolor='black')
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_title(name)

            plt.show()
countries = []
qual_count = []
for i in min_con:
    countries.append(i["name"])
    qual_count.append(len(i["quals"]))

#countries = countries.reverse()
#qual_count = qual_count.reverse()
df = pd.DataFrame({"CountryName" : countries, "No. of qualifications": qual_count})
df.iloc[::-1].iloc[1:18, :]
countries = []
qual_count = []
for i in max_con:
    countries.append(i["name"])
    qual_count.append(len(i["quals"]))

#countries = countries.reverse()
#qual_count = qual_count.reverse()
df = pd.DataFrame({"CountryName" : countries, "No. of qualifications": qual_count})
df.iloc[::-1].iloc[:-2, :]
country_details = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")
country_details.iloc[1, :]
coun_names = [i.upper() for i in np.array(country_details["Country"])]
population = np.array(country_details["Population"])
pop_x = []
pop_y = []
errs = []
for i in edu_countries:
    for b in i:
        try:
            #print(b.upper())
            ind = coun_names.index(b.upper()+" ")
            pop_x.append(population[ind])
            pop_y.append(len(Counter(i[b])))
        except:
            errs.append("err")
plt.title("No. of types of qualification vs Population")
plt.xlabel("No. of types of qualifications")
plt.ylabel("Population")
plt.bar(pop_y, pop_x)
plt.show()
gdp = np.array(country_details["GDP ($ per capita)"])
pop_x = []
pop_y = []
errs = []
for i in edu_countries:
    for b in i:
        try:
            #print(b.upper())
            ind = coun_names.index(b.upper()+" ")
            pop_x.append(gdp[ind])
            pop_y.append(len(Counter(i[b])))
        except:
            errs.append("err")
plt.title("No. of types of qualification vs GDP ($ per capita)")
plt.xlabel("No. of types of qualifications")
plt.ylabel("GDP ($ per capita)")
plt.bar(pop_y, pop_x)
plt.show()
orientation = Counter(np.array(results["SexualOrientation"].fillna("Did not respond")))
labels = list(orientation)
labels = labels[0:5]
count = [orientation[i] for i in labels]
import matplotlib.pyplot as plt
plt.rcdefaults()
fig, ax = plt.subplots()

ax.barh(np.arange(len(labels)), count, align='center',
        color='green', ecolor='black')
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Sexual Orientations')

plt.show()
language_worked_with = np.array(results[["Exercise", "LanguageWorkedWith"]].dropna().iloc[:, :])
programs = {}
for i in language_worked_with:
    try:
        if(i[1].index(";")):
            ments = i[1].split(";")
            for j in ments:
                try:
                    elem = programs[j]
                    elem.append(i[0])
                    programs[j] = elem
                except:
                    programs[j] = [i[0]]
    except:
        try:
            elem = programs[i[1]]
            elem.append(i[0])
            programs[i[1]] = elem
        except:
            programs[i[1]] = [i[0]]
langs = []
users = []
for i in programs:
    langs.append(i)
    users.append(len(programs[i]))

df = pd.DataFrame({"Language" : langs, "users": users})
df = df.sort_values(["users"], ascending = False)
langs = list(df.iloc[:, 0])
users = list(df.iloc[:, 1])

plt.rcdefaults()
fig, ax = plt.subplots()

ax.barh(np.arange(len(langs)), users, align='center',
        color='green', ecolor='black')
ax.set_yticks(np.arange(len(langs)))
ax.set_yticklabels(langs)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Programming languages used')

plt.show()
l = []
c = []
import matplotlib.pyplot as plt
for i in programs:
    temp = Counter(np.array(programs[i]))
    labels = list(temp)
    count = [temp[i] for i in labels]
    c.append(count[labels.index("I don't typically exercise")])
    l.append(i)
    if(sum(count)>20000):
        plt.rcdefaults()
        fig, ax = plt.subplots()

        ax.barh(np.arange(len(labels)), count, align='center',
                color='green', ecolor='black')
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        name = 'Exercise pattern of ' + str(i) + " developers"
        ax.set_title(name)
        plt.show()
df = pd.DataFrame({"Language" : l, "laziness": c})
df = df.sort_values(["laziness"], ascending = False).iloc[0:20, :]
labels = list(df.iloc[:, 0])
count = list(df.iloc[:, 1])
plt.rcdefaults()
fig, ax = plt.subplots()

ax.barh(np.arange(len(labels)), count, align='center',
        color='green', ecolor='black')
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
name = "Laziness of developers based on language used"
ax.set_title(name)
plt.show()
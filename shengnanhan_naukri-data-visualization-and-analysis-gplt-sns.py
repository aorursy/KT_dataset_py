import csv

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

%matplotlib inline

import geopandas as gpd

import geoplot as gplt



# data = []

# with open('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv', newline='', encoding="utf8") as csvfile:

#     spamreader = csv.reader(csvfile)

#     for row in spamreader:

#         sub = np.array(row)

#         if sub[2] == "":

#             continue

#         data.append(sub)

pd.set_option('display.max_columns',15)

pd.set_option('display.max_rows',None)

data_df=pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')

# data = np.array(data)

data_df.head()
data_df.dtypes
data_df.isna().any()
sns.heatmap(data_df.isnull(),cbar=True,cmap='gnuplot')
selected_col = ['Job Title', 'Job Salary', 'Job Experience Required', 'Key Skills', 'Role Category', 'Location', 'Functional Area', 'Industry', 'Role']

missing = []

for col in selected_col:

    print('Number of missing data in', col,':', data_df[col].isna().value_counts()[1])

    missing.append((col, data_df[col].isna().value_counts()[1]))

print('Total number of missing data :', len(data_df))
missing.sort(key=lambda x: x[1], reverse = True)

x_pos = [i for i in range(len(missing))]

nums = [words[1] for words in missing]

x = [words[0] for words in missing]

sns.set(rc={'figure.figsize':(19,9)})

sns.barplot(x=x, y=nums)
data_df.dropna(axis=0,inplace=True)

data_df.isna().any()
fp = '../input/data-ana-for-jobs-on-naukri-supplements/india-polygon.shp'

map_df = gpd.read_file(fp)

map_df.head()
map_df["st_nm"]
# ind will pass the industry into the function, if ind == '', this means that we are looking at all industries

def get_ind_city_cnt(ind):

    cityCnt = {}

    sub_data = data_df[data_df['Industry'].str.contains(ind)]

    for i in range(len(sub_data)):

        cities = sub_data.iloc[i, 7]

        if "(" in cities:

            index = cities.find("(")

            cities = cities[:index]

        cities = cities.split(",")

        uniqueCities = set()

        for city in cities:

            tmp = city

            # cleaning data

            if ")" in tmp:

                print(city)

            if "/" in tmp:

                sameName = city.split("/")

                sameName.sort()

                tmp = sameName[0]

            tmp = tmp.strip()

            uniqueCities.add(tmp.lower())

        for city in uniqueCities:

            cityCnt[city] = cityCnt.get(city, 0) + 1



    # some location information need to be deleted

    cleanSet = ["electronics city", "1700000", "300000", "400000", "500000", "700000"]

    for i in cleanSet:

        if i in cityCnt:

            del cityCnt[i]

    return cityCnt

cityCnt = get_ind_city_cnt("")

print("Number of unique locations in dataset is", len(cityCnt))
cityData = pd.ExcelFile("../input/data-ana-for-jobs-on-naukri-supplements/Town_Codes_2001.xls")

city_df = cityData.parse("Sheet1")

city_df.head()
notFound = set([])

for name in city_df["State/Union territory"]:

    if map_df[map_df["st_nm"].isin([name])].empty:

        notFound.add(name)

notFound
stateTran = {'Andaman & Nicobar Islands *':'Andaman and Nicobar Islands' ,

             "Chandigarh *":'Chandigarh' ,

             "Dadra & Nagar Haveli *":'Dadra and Nagar Haveli',

             "Daman & Diu *":'Daman and Diu',

             "Lakshadweep *":'Lakshadweep',

             "Delhi *":'Delhi',

             "Pondicherry *":'Puducherry',

             "Uttaranchal": 'Uttarakhand',

             "Orissa":'Odisha',

             "Jammu & Kashmir": 'Jammu and Kashmir'

            }
notFoundCities = []

cityToState = {}

for name in cityCnt.keys():

    if name == "":

        continue

    tmp = name[0].upper()+name[1:]

    if city_df[city_df["City/Town"].isin([tmp])].empty:

        if city_df[city_df["State/Union territory"].isin([tmp])].empty:

            sub = tmp.split()

            add = False

            for i in sub:

                i = i[0].upper()+i[1:]

                if city_df[city_df["City/Town"].isin([i])].empty and city_df[city_df["State/Union territory"].isin([i])].empty:

                    add = True

                else:

                    add = False

            if add:

                notFoundCities.append(name)

print("Number of location that is no matched in either cities or states",len(notFoundCities))
# # Save

# np.save('unmatchedLocations.npy', cityTran) 



# Load

cityTran = np.load('../input/data-ana-for-jobs-on-naukri-supplements/unmatchedLocations.npy',allow_pickle='TRUE').item()
# retrieve state according to its location

def getState(city):

    if city == "":

        return ""

    tmp = city[0].upper()+city[1:]

    if city_df[city_df["City/Town"].isin([tmp])].empty:

        if city_df[city_df["State/Union territory"].isin([tmp])].empty:

            sub = tmp.split()

            add = False

            for i in sub:

                i = i[0].upper()+i[1:]

                if city_df[city_df["City/Town"].isin([i])].empty:

                    if city_df[city_df["State/Union territory"].isin([i])].empty:

                        add = True

                    else:

                        return city_df.loc[city_df["State/Union territory"]==i]["State/Union territory"].values[0]

                else:

                    return city_df.loc[city_df["City/Town"]==i]["State/Union territory"].values[0]

                    

            if add:

                return ""

    else:

        return city_df.loc[city_df["City/Town"]==tmp]["State/Union territory"].values[0]



# translate the state

def validState(state):

    if state in stateTran:

        return stateTran[state]

    else:

        return state



# get the number of jobs with respect to its state

def get_state_cnt(cityCnt):

    stateCnt = {}

    missing = []

    for city, num in cityCnt.items():

        state = getState(city)

        if state == "" or state is None:

            if city in cityTran:

                state = validState(cityTran[city])

                stateCnt[state] = stateCnt.get(state, 0) + num

            else:

                missing.append(city)



        else:

            state = validState(state)

            stateCnt[state] = stateCnt.get(state, 0) + num

    return stateCnt, missing



stateCnt, missing = get_state_cnt(cityCnt)

print("The number of locations that is still not mapped to states", len(missing))
missing
def plot_geo(stateCnt, ind):

    stateData = []

    for key, val in stateCnt.items():

        stateData.append((key, val))

    for state in map_df["st_nm"]:

        if state not in stateCnt:

            stateData.append((state, 0))

    

    jobNum = [0 for _ in range(len(DATA_df))]

    new_col = ind+" Number of Jobs"

    DATA_df[new_col] = jobNum

    

    for state, num in stateData:

        DATA_df.loc[DATA_df["st_nm"] == state, new_col] = num

    merged= map_df.merge(DATA_df, on = "st_nm", how = "left")

    

    fig, ax = plt.subplots(1, figsize=(10, 10))

    ax.axis("off")

    ax.set_title(ind+" Job data", fontdict={"fontsize": "25", "fontweight" : "10"})

    merged.plot(column=new_col,cmap="YlGnBu", linewidth=0.8, ax=ax, edgecolor="0", legend=True,markersize=[39.739192*2, -104.990337*2])

    plt.show()

    merged = merged.sort_values(new_col, ascending=False)

    sns.barplot(x=merged['st_nm'][:10], y=merged[new_col][:10])

    

def ind_geo_dist(ind):

    city_cnt = get_ind_city_cnt(ind)

    state_cnt, missing = get_state_cnt(city_cnt)

    plot_geo(state_cnt, ind)



DATA_df = pd.read_excel("../input/data-ana-for-jobs-on-naukri-supplements/data_ecxel.xlsx")

DATA_df.rename(columns={"Name of State / UT": "st_nm"},inplace=True)

ind_geo_dist("")
data_df['Industry'].value_counts()[:10]
data_df.loc[data_df['Industry'].str.contains('IT-Software / Software Services',case=False)]='IT-Software, Software Services'

data_df.loc[data_df['Industry'].str.contains('Recruitment / Staffing',case=False)]='Recruitment , Staffing'

data_df['Industry'].value_counts()[:10]
print("We will focus on the top 10 industries because they compose", '{:.1%}'.format(sum(data_df['Industry'].value_counts()[:10])/len(data_df)), 'of all data')
top_ten_state = data_df['Industry'].value_counts()[:10].index

for ind in top_ten_state:

    ind_geo_dist(ind)
data_df['Job Salary'].value_counts()[:10]
data_df['Job Experience Required'].value_counts()[:10]
exp_cat = ['' for _ in range(len(data_df))]

data_df['Experience Categories'] = exp_cat

data_df.head()
CAT = {(0, 1):"Newbie", (1, 5):"Semiprofessional", (5, 10):"Professional", (10, 100):"Expert"}

def get_cat(yrs):

    if len(yrs) != 2:

        return ""

    start, end = str(yrs[0]), str(yrs[1])

    start, end = start.strip(), end.strip()

    if not start.isnumeric():

        return ""

    start, end = int(start), int(end)

    res = ''

    for key, val in CAT.items():

        if start <= key[1] and end > key[0]:

            res += ' ' + val

    if start >= 10:

        res += ' Expert'

    res = res.strip()

    return res



for i in range(len(data_df)):

    job_req = data_df.iloc[i, 4].lower()

    #print(job_req)

    index = job_req.find('y')

    job_req = job_req[:index]

    yrs = job_req.split('-')

    cat = get_cat(yrs)

    data_df.iloc[i, -1] = cat



data_df.head()
value_cnt = {"Newbie":0, "Semiprofessional":0, "Professional":0, "Expert":0}

for i in range(len(data_df)):

    sub_cat = data_df.iloc[i, -1]

    sub_cat = sub_cat.split()

    for key in value_cnt.keys():

        if key in sub_cat:

            value_cnt[key] += 1

for key, val in value_cnt.items():

    print(key, ":", val)

plt.pie(x=[val for val in value_cnt.values()], labels=[key for key in value_cnt.keys()], autopct='%1.1f%%')
new_df = data_df[data_df['Experience Categories'].str.contains('Newbie', case=True)]

new_cnt = new_df['Industry'].value_counts()

plot = new_cnt[:10].plot.pie(figsize=(7, 7), autopct='%.1f')
semipro_df = data_df[data_df['Experience Categories'].str.contains('Semiprofessional', case=True)]

semipro_cnt = semipro_df['Industry'].value_counts()

plot = semipro_cnt[:10].plot.pie(figsize=(7, 7), autopct='%.1f')
pro_df = data_df[data_df['Experience Categories'].str.contains('Professional', case=True)]

pro_cnt = pro_df['Industry'].value_counts()[:10]

plot = pro_cnt[:10].plot.pie(figsize=(7, 7), autopct='%.1f')
exp_df = data_df[data_df['Experience Categories'].str.contains('Expert', case=True)]

exp_cnt = exp_df['Industry'].value_counts()[:10]

plot = exp_cnt[:10].plot.pie(figsize=(7, 7), autopct='%.1f')
pro_cnt = pd.DataFrame(pro_cnt)

pro_cnt.reset_index(inplace=True)

pro_cnt.rename(columns={'Industry':'Professional'},inplace=True)

semipro_cnt = pd.DataFrame(semipro_cnt)

semipro_cnt.reset_index(inplace=True)

semipro_cnt.rename(columns={'Industry':'Semiprofessional'},inplace=True)

new_cnt = pd.DataFrame(new_cnt)

new_cnt.reset_index(inplace=True)

new_cnt.rename(columns={'Industry':'Newbie'},inplace=True)

exp_cnt = pd.DataFrame(exp_cnt)

exp_cnt.reset_index(inplace=True)

exp_cnt.rename(columns={'Industry':'Expert'},inplace=True)

merged = new_cnt.merge(semipro_cnt, on = "index", how = "left")

merged = merged.merge(pro_cnt, on = "index", how = "left")

merged = merged.merge(exp_cnt, on = "index", how = "left")

merged.head()
merged = merged.fillna(0)

merged.isnull().values.any()
total, per_semi, per_pro, per_new, per_exp= [], [], [], [], []



for i in range(len(merged)):

    sub_t = merged["Semiprofessional"][i] + merged["Professional"][i] + merged["Newbie"][i]+merged['Expert'][i]

    total.append(sub_t)

    per_semi.append(merged['Semiprofessional'][i] / sub_t)

    per_pro.append(merged['Professional'][i] / sub_t)

    per_new.append(merged['Newbie'][i] / sub_t)

    per_exp.append(merged['Expert'][i] / sub_t)

merged["Total Number of Jobs"] = total

merged["Percent of Semipro"] = per_semi

merged["Percent of Pro"] = per_pro

merged["Percent of New"] = per_new

merged["Percent of Exp"] = per_exp

merged.head()
fig, axs = plt.subplots(5, 2, figsize=(15, 20))

for i in range(5):

    for j in range(2):

        val = [merged.iloc[i*2+j, k] for k in range(6, 10)]

        index = ['Percent of Semipro', 'Percent of Pro', 'Percent of New', 'Percent of Exp']

        tmp = pd.DataFrame({"Val":val}, index = index)

        axs[i, j].pie(tmp, autopct='%.0f%%', labels=tmp.index)

        axs[i, j].set_title(merged.iloc[i*2+j,0] + ' Recruitment Pattern')
merged = merged.sort_values("Percent of New", ascending=False)

print('<', merged.iloc[0][0], '> is The industry that recruits most Newbies','\nPercent of recruitment is', '{:.1%}'.format(merged.iloc[0][8]))

merged = merged.sort_values("Percent of Semipro",ascending=False)

print('<', merged.iloc[0][0], '> is The industry that recruits most Semiprofessionals','\nPercent of recruitment is', '{:.1%}'.format(merged.iloc[0][6]))

merged = merged.sort_values("Percent of Pro",ascending=False)

print('<', merged.iloc[0][0], '> is The industry that recruits most Professionals','\nPercent of recruitment is', '{:.1%}'.format(merged.iloc[0][7]))

merged = merged.sort_values("Percent of Exp",ascending=False)

print('<', merged.iloc[0][0], '> is The industry that recruits most Experts','\nPercent of recruitment is', '{:.1%}'.format(merged.iloc[0][9]))
print('<', merged.iloc[0][0], '> is The industry that recruits least Newbies','\nPercent of recruitment is', '{:.1%}'.format(merged.iloc[-1][8]))

print('<', merged.iloc[0][0], '> is The industry that recruits least Semiprofessionals','\nPercent of recruitment is', '{:.1%}'.format(merged.iloc[-1][6]))

print('<', merged.iloc[0][0], '> is The industry that recruits least Professionals','\nPercent of recruitment is', '{:.1%}'.format(merged.iloc[-1][7]))

print('<', merged.iloc[0][0], '> is The industry that recruits least Experts','\nPercent of recruitment is', '{:.1%}'.format(merged.iloc[-1][9]))
from wordcloud import WordCloud, STOPWORDS

def topSkills(Industry):

    ind = data_df[data_df['Industry'] == Industry]

    keySkill = ind['Key Skills'].value_counts()

    keySkill.head()

    skillCnt = {}

    wcloud = []

    for row, cnt in keySkill.iteritems():

        skill_arr = row.split('|')

        for skill in skill_arr:

            skill = skill.strip().lower()

            skillCnt[skill] = skillCnt.get(skill, 0) + cnt

            for i in range(cnt):

                wcloud.append(skill)

    skillCnt = pd.Series(skillCnt).to_frame('Count')

    skillCnt.reset_index(inplace=True)

    skillCnt.rename(columns={'index':'Skill'},inplace=True)

    skillCnt.head()

    #print(skillCnt)

    skillCnt = skillCnt.sort_values("Count",ascending=False)

    skillRange = 10 if len(skillCnt) >= 10 else len(skillCnt)

    ax = skillCnt.head(skillRange).plot(kind='bar', figsize=(9, 5))

    plt.xticks(range(skillRange), skillCnt.head(skillRange)['Skill'])

    ax.set_ylabel("Total Number of Skill")

    ax.set_xlabel("Skill")

    t = Industry+" Industry Top Skills"

    plt.title(t)

    wordcloud = WordCloud(width = 500, height = 500, 

                    background_color ='white', 

                    min_font_size = 10).generate(str(wcloud))

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 

    plt.show()
topSkills('IT-Software, Software Services')
topSkills('Recruitment, Staffing')
topSkills('BPO, Call Centre, ITeS')
topSkills('Banking, Financial Services, Broking')
topSkills('Education, Teaching, Training')
topSkills('Medical, Healthcare, Hospitals')
topSkills("Strategy, Management Consulting Firms")
topSkills("Internet, Ecommerce")
topSkills("Media, Entertainment, Internet")
topSkills("Travel , Hotels , Restaurants , Airlines , Railways")
jobTitle = data_df['Job Title']

dic = {}

for job in jobTitle:

    dic[job] = dic.get(job, 0) + 1
def findTop(arr, key, val, num):

    arr.append((val, key))

    arr.sort(reverse=True)

    if len(arr) > num:

        arr.pop()

    

uniequeJobTitle = []

topTenJobTitle = []

for key, val in dic.items():

    findTop(topTenJobTitle, key, val, 10)

    if val == 1:

        uniequeJobTitle.append(key)

print(len(uniequeJobTitle), "number of unique Job Title")
x_pos = [i for i in range(len(topTenJobTitle))]

nums = [words[0] for words in topTenJobTitle]

x = [words[1] for words in topTenJobTitle]

ax = sns.barplot(x=x, y=nums)

ax.set(xlabel="Top 10 Job Title", ylabel = "Occurence of the Top 10 Job Title")
uniequeJobTitle = []

topTenJobTitle = []

for key, val in dic.items():

    findTop(topTenJobTitle, key, val, 11)

    if val == 1:

        uniequeJobTitle.append(key)

x_pos = [i for i in range(10)]

nums = [words[0] for words in topTenJobTitle[1:11]]

x = [words[1] for words in topTenJobTitle[1:11]]

ax = sns.barplot(x=x, y=nums)

ax.set(xlabel="Top 10 Job Title", ylabel = "Occurence of the Top 10 Job Title")
keywords = {}

wordFilter = ["for", "in", "opening", ""]

for key, val in dic.items():

    for sub in key.split():

        if sub in "~!@#$%^&*()-=+~\|]}[{';: /?.>,<." or sub.lower() in wordFilter:

            continue

        keywords[sub] = keywords.get(sub, 0) + val

print("Number of keywords", len(keywords))
topTwentyKeywords = []

for key, val in keywords.items():

    findTop(topTwentyKeywords, key, val, 20)

x_pos = [i for i in range(len(topTwentyKeywords))]

nums = [words[0] for words in topTwentyKeywords]

x = [words[1] for words in topTwentyKeywords]

ax = sns.barplot(x=x, y=nums)

ax.set(xlabel="Top 20 Keywords", ylabel = "Occurence of the top 20 keywords in Job Title")
jobTypeInUrgent = {}

counter = 0

for index, job in data_df.iterrows():

    if job['Job Title'].lower().find("urgent") >= 0 and job['Role Category']:

        jobTypeInUrgent[job['Role Category']] = jobTypeInUrgent.get(job['Role Category'], 0) + 1

        

topTwentyJobInUrgent = []

for key, val in jobTypeInUrgent.items():

    findTop(topTwentyJobInUrgent, key, val, 10)



x_pos = [i for i in range(len(topTwentyJobInUrgent))]

nums = [words[0] for words in topTwentyJobInUrgent]

x = [words[1] for words in topTwentyJobInUrgent]

ax = sns.barplot(x=x, y=nums)

ax.set(xlabel="Top 20 Urgent Roles", ylabel = "Occurence of the top 20 Urgent Role")
x
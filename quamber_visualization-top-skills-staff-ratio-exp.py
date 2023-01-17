import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from subprocess import check_output

df_=pd.read_csv("../input/Pakistan Intellectual Capital - Computer Science - Ver 1.csv", encoding = "ISO-8859-1")

df_["Country"].fillna("Unknown", inplace=True)

df_["Year"].fillna("Unknown", inplace=True)

df_["Graduated from"].fillna("Unknown", inplace=True)

df_["Designation"].fillna("Unknown", inplace=True)

df_.head()
df_.loc[df_['Designation'].str.contains('Assistant Professor', case=False), 'Designation'] = 'Assistant P'

df_.loc[df_['Designation'].str.contains('Associate Professor', case=False), 'Designation'] = 'Associate P'

df_.loc[df_['Designation'].str.contains('Professor', case=False), 'Designation'] = 'Professor'

df_.loc[df_['Designation'].str.contains('Assistant P', case=False), 'Designation'] = 'Assistant Professor'

df_.loc[df_['Designation'].str.contains('Associate P', case=False), 'Designation'] = 'Associate Professor'

df_.loc[df_['Designation'].str.contains('Lecturer', case=False), 'Designation'] = 'Lecturer'

df_.loc[df_['Designation'].str.contains('Instructor', case=False), 'Designation'] = 'Instructor'

plt.rcParams['figure.figsize']=(12,18)

fig, axes = plt.subplots(nrows=3, ncols=2)

staff = ["Assistant Professor","Associate Professor","Professor","Lecturer","Instructor"]

df_["Designation"][(df_["Designation"].isin(staff))&(df_["Province University Located"]=='Punjab')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[0,0],title="Punjab",label="")

df_["Designation"][(df_["Designation"].isin(staff))&(df_["Province University Located"]=='Capital')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[0,1],title="Federal Capital Territory",label="")

df_["Designation"][(df_["Designation"].isin(staff))&(df_["Province University Located"]=='Sindh')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[1,0],title="Sindh",label="")

df_["Designation"][(df_["Designation"].isin(staff))&(df_["Province University Located"]=='KPK')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[1,1],title="KPK",label="")

df_["Designation"][(df_["Designation"].isin(staff))&(df_["Province University Located"]=='Balochistan')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[2,0],title="Balochistan",label="")

df_["Designation"][(df_["Designation"].isin(staff))].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[2,1],title="Pakistan",label="")

plt.show()

from collections import defaultdict

from collections import Counter

plt.rcParams['figure.figsize']=(12,10)

default_dict2 = defaultdict(lambda:0)

for line in df_["Area of Specialization/Research Interests"]:

    if str(line)!= 'nan':

        for obj in str(line).split(","):

            default_dict2[obj.upper().strip()]+=1

d = Counter(default_dict2)

dataFrame= pd.DataFrame(d.most_common(40))

dataFrame=dataFrame.sort_values(by=1)

dataFrame.plot.barh(x=dataFrame[0],legend=False,title="Top 40 Skills / Research Intrests")

plt.ylabel("Skills")

plt.xlabel("# of People with Skill")

plt.show()

(2018-df_["Year"][df_["Year"]!="Unknown"]).astype(int).value_counts().sort_values().plot.barh(title="Year Count from Terminal Degree",label="Years")

plt.ylabel("Years")

plt.xlabel("# of People")

plt.show()
plt.close()

plt.rcParams['figure.figsize']=(12,18)

map_TDC = {'Pakistan':'Pakistan','Unknown':"Not-Known"}

df_['ForeignTD'] = df_['Country'].copy()

df_['ForeignTD'] = np.where(df_['ForeignTD'].isin(map_TDC.keys()), df_['ForeignTD'].replace(map_TDC), "Foreign")

fig, axes = plt.subplots(nrows=3, ncols=2)

staff = ["Pakistan","Foreign","Not-Known"]

df_["ForeignTD"][(df_["ForeignTD"].isin(staff))&(df_["Province University Located"]=='Punjab')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[0,0],title="Punjab",label="")

df_["ForeignTD"][(df_["ForeignTD"].isin(staff))&(df_["Province University Located"]=='Capital')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[0,1],title="Fedral Capital",label="")

df_["ForeignTD"][(df_["ForeignTD"].isin(staff))&(df_["Province University Located"]=='Sindh')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[1,0],title="Sindh",label="")

df_["ForeignTD"][(df_["ForeignTD"].isin(staff))&(df_["Province University Located"]=='KPK')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[1,1],title="KPK",label="")

df_["ForeignTD"][(df_["ForeignTD"].isin(staff))&(df_["Province University Located"]=='Balochistan')].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[2,0],title="Balochistan",label="")

df_["ForeignTD"][(df_["ForeignTD"].isin(staff))].value_counts().plot.pie(autopct='%1.1f%%',ax = axes[2,1],title="Pakistan",label="")

plt.show()
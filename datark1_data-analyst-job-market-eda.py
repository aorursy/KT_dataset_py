# importing basic libraries 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
print("In our dataset there are: {} rows and {} columns.".format(data.shape[0],data.shape[1]))
data.head()
data.dtypes
data["Founded"] = data["Founded"].astype("Int64")

data["Easy Apply"] = data["Easy Apply"].astype("bool")
data.drop(columns=["Unnamed: 0"], inplace=True)

data.replace([-1,-1.0,"-1"],np.nan, inplace=True)
data["Python"] = data["Job Description"].apply(lambda x: 1 if "Python" in x or "python" in x else 0)

data["R"] = data["Job Description"].apply(lambda x: 1 if " R " in x or " R/" in x or "R," in x else 0)



toolset = ["Python", "R","SQL", "Excel", "SAS","AWS", "Stata", "Power BI", "Microstrategy", "Tableau", "VBA"]



for tool in toolset[2:]:

    data[tool] = data["Job Description"].apply(lambda x: 1 if tool in x else 0)
tools_sum = data[toolset].sum().sort_values(ascending=False).div(len(data)).mul(100)

plt.style.use('ggplot')

ax, fig = plt.subplots(figsize=(12,6))

sns.barplot(tools_sum.index,

            tools_sum)

plt.title("DA tools in job offers")

plt.ylabel("Percentage")

plt.show()
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles



py = data["Python"].sum()

r = data["R"].sum()

sql = data["SQL"].sum()

excel = data["Excel"].sum()



py_r = data[(data["Python"]==1) & (data["R"]==1)]["Python"].sum()

py_sql = data[(data["Python"]==1) & (data["SQL"]==1)]["Python"].sum()

r_sql = data[(data["R"]==1) & (data["SQL"]==1)]["Python"].sum()

py_r_sql = data[(data["Python"]==1) & (data["R"]==1) & (data["SQL"]==1)]["Python"].sum()

py_excel = data[(data["Python"]==1) & (data["Excel"]==1) & (data["SQL"]==1)]["Python"].sum()



fig, axes = plt.subplots(2,2,figsize=(10,8))



venn2(subsets = (py, r, py_r), set_labels = ("Python", "R"), ax=axes[0][0], set_colors=('red', 'green'))

venn2_circles(subsets = (py, r, py_r), ax=axes[0][0])



venn2(subsets = (py, sql,py_sql), set_labels = ("Python", "SQL",), ax=axes[0][1], set_colors=('red', 'blue'))

venn2_circles(subsets = (py, sql,py_sql), ax=axes[0][1])



venn2(subsets = (r, sql, r_sql), set_labels = ("R", "SQL",), ax=axes[1][0], set_colors=('green', 'blue'))

venn2_circles(subsets = (r, sql, r_sql), ax=axes[1][0])



venn2(subsets = (py, excel, py_excel), set_labels = ("Python", "Excel",), ax=axes[1][1], set_colors=('green', 'yellow'))

venn2_circles(subsets = (py, excel, py_excel), ax=axes[1][1])



fig.suptitle("Venn diagrams - DA tools in job offers", size=15)



plt.show()
fig, ax = plt.subplots(1,1,figsize=(8,8))



venn3(subsets = {

    "100":py, "010":r, "001":sql,

    "110":py_r, "101":py_sql, "011":r_sql,

    "111":py_r_sql},

    set_labels = ("Python", "R", "SQL"),

    ax=ax)



venn3_circles(subsets = {

    "100":py, "010":r, "001":sql,

    "110":py_r, "101":py_sql, "011":r_sql,

    "111":py_r_sql},

    ax=ax)

plt.show()
def job_name_cleaner(cell,pos):

    try:

        value = str(cell).split(",")[pos]

        return value

    except:

        return np.nan

    

data["Job Title 1"] = data["Job Title"].apply(lambda x: job_name_cleaner(x,0))

data["Job Title 2"] = data["Job Title"].apply(lambda x: job_name_cleaner(x,1))
jobT_1 = data["Job Title 1"].value_counts(normalize=True).mul(100)

print("There are {} various job titles.".format(len(jobT_1)))
jobT_1 = data["Job Title 1"].value_counts(normalize=True).mul(100)



ax, fig = plt.subplots(figsize=(14,6))

sns.barplot(x=jobT_1.index[:20], 

            y=jobT_1.values[:20])

plt.ylabel("Percentage")

plt.xticks(rotation=70, ha="right")

plt.show()
def experience(job):

    for w in ["Junior Data Analyst", "Jr.", "Data Analyst I", "Jr", "1"]:

        if w in job:

            return "Junior"    

    for w in ["Senior", "III", "Lead", "Sr", "Sr.", "3", "Principal", "Master"]:

        if w in job:

            return "Senior"

    else:

        return "Regular/Other"



data["Exp. Level"] = data["Job Title 1"].apply(experience)
jobT_2 = data["Exp. Level"].value_counts(normalize=True).mul(100)



ax, fig = plt.subplots(figsize=(12,6))

plt.pie(jobT_2.values, labels=jobT_2.index, autopct='%1.1f%%', shadow = True, startangle=90, colors=["#fccb05","#059efc","#36fa28"], textprops={"size":14})

plt.title("Job offers exerience levels")

plt.show()
def salary_cleaner(cell,pos):

    if cell == -1:

        return np.nan

    else:

        try:

            value = str(cell).split("K")[pos].replace("$","").replace("-","")

            return int(value)

        except:

            return np.nan

    

data["lower_salary"] = data["Salary Estimate"].apply(lambda x: salary_cleaner(x,0))

data["upper_salary"] = data["Salary Estimate"].apply(lambda x: salary_cleaner(x,1))

data["average_salary"] = (data["lower_salary"]+data["upper_salary"])/2
locations_salaries = data.groupby(["Location"])[["lower_salary","upper_salary","average_salary","Job Title"]].mean().round(1)

locations_salaries["offers"] = data.groupby(["Location"])[["Job Title"]].count()

locations_salaries = locations_salaries.sort_values(by=["average_salary"], ascending=False)

locations_salaries.head(10)
locations = data["Location"].str.split(r",",expand=True)

locations.columns = ["City","State","temp"]

locations.drop(["temp"],axis=1,inplace=True)



#dealing with "... ,Arapahoe, CO" syntax

locations[locations["State"]==" Arapahoe"] = " CO"

locations["State"] = locations["State"].str.strip()



# concatenating

data = pd.concat([data,locations],axis=1)
states_salaries = data.groupby(["State"])[["lower_salary","upper_salary","average_salary"]].mean().round(1)

states_salaries["offers"] = data.groupby(["State"])[["average_salary"]].count()

states_salaries = states_salaries.sort_values(by=["average_salary"], ascending=False)

states_salaries
ax, fig = plt.subplots(figsize=(14,6))

sns.barplot(x=states_salaries.index, 

            y=states_salaries["average_salary"])

plt.ylabel("Salary [k$]")

plt.xticks(rotation=70, ha="right")

plt.title("Average Salaries in various US states")

plt.show()
import plotly.graph_objects as go



fig = go.Figure(data=go.Choropleth(

    locations=states_salaries.index, # Spatial coordinates

    z = states_salaries['average_salary'], # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'Reds',

    colorbar_title = "Average salary [k$]",

))



fig.update_layout(

    title_text = 'Average salary',

    geo_scope='usa', # limite map scope to USA

)



fig.show()
industries = data["Industry"].value_counts(normalize=True).mul(100)



plt.style.use('ggplot')

ax, fig = plt.subplots(figsize=(14,6))

sns.barplot(x=industries.index[:20], 

            y=industries.values[:20])

plt.ylabel("Percentage")

plt.xticks(rotation=70, ha="right")

plt.text(15,16, "No. of industries: {}".format(len(industries)), size=15)

plt.show()
sizes = data["Size"].value_counts(normalize=True).mul(100)



plt.style.use('ggplot')



ax, fig = plt.subplots(figsize=(12,6))

sns.barplot(x=sizes.index, 

            y=sizes.values,

            order = ['1 to 50 employees', '51 to 200 employees',

                  '201 to 500 employees', '501 to 1000 employees',

                  '1001 to 5000 employees', '5001 to 10000 employees',

                  '10000+ employees', 'Unknown'])

plt.ylabel("Percentage")

plt.xticks(rotation=70, ha="right")

plt.show()
data["Company Name"] = data["Company Name"].apply(lambda x: str(x).split("\n")[0])
data["Company Name"].value_counts().head()
data[data["Industry"]=="Health Care Services & Hospitals"][["Company Name","Headquarters"]].value_counts().head()
data[data["Industry"]=="Computer Hardware & Software"][["Company Name","Headquarters"]].value_counts().head()
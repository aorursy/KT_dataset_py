import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # Visualization
import matplotlib.pyplot as plt # Visualization
import folium # Choropleth Visualization

pd.set_option('max_columns', None)
survey = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")
survey = survey.drop([0])

#Subset of people who strictly identify as students
students = survey.loc[(survey.Q7 == "I am a student") & (survey.Q6 == "Student")]
professionals = survey.loc[(survey.Q7 != "I am a student") & (survey.Q6 != "Student") & (survey.Q6 != "Not employed")]

professionals.head(20)
world_geo = '../input/countries-of-the-world/countries.json'

#Copy and reorganize data strictly for country of origin/counts
student_origins = students[["Q3"]].copy(deep=True)
student_origins = students.Q3.value_counts().reset_index().rename(columns ={"index": "Country", 0: "count"})

#Remove "Other" and "Do not wish to disclose"
student_origins = student_origins.drop(student_origins.index[3])
student_origins = student_origins.drop(student_origins.index[9])

m = folium.Map(location = [0, 0], zoom_start = 2)

folium.Choropleth(
    geo_data = world_geo,
    name = 'choropleth',
    data = student_origins,
    columns = ["Country", "Q3"],
    key_on = "feature.properties.name",
    bins = 6,
    fill_color = 'YlOrRd',
    fill_opacity = 0.7,
    line_opacity = 0.2,
    legend_name = "Number of Students"
).add_to(m)

folium.LayerControl().add_to(m)

m
fig, student_degrees = plt.subplots(2, 1)
fig.set_size_inches(8, 10)
majors = sns.countplot(y = "Q5",
             data = students,
             order = students["Q5"].value_counts().index,
             orient = "v",
             ax = student_degrees[0])
majors.axes.set_title("What Students are Studying", fontsize = 15)
majors.set_ylabel("Majors", fontsize = 15)

degrees = sns.countplot(y = "Q4",
             data = students,
             order = students["Q4"].value_counts().index,
             orient = "v",
             ax = student_degrees[1])
degrees.axes.set_title("The Degrees Students are Getting", fontsize = 15)
degrees.set_ylabel("Education Level", fontsize = 15)
degrees.set_xlabel("Number of Students", fontsize = 15)

student_degrees[0].set_xlabel(' ')
plt.show()
students_heatmap = students[["Q4", "Q5"]].copy()
students_heatmap.dropna()

students_heatmap = students_heatmap.groupby(["Q4", "Q5"]).size().reset_index(name="Number")
df = students_heatmap.pivot_table(index="Q5", columns="Q4", values="Number", fill_value = 0)

#Rearrange columns of table in sequential degree levels
cols = df.columns.tolist()
df = df[[cols[5], cols[0], cols[3], cols[1], cols[4], cols[2]]]

#Shorten name of column for "some college"
df = df.rename(columns= {cols[5]: "Some College/University"})
df

plt.figure(figsize=(5.5, 4))
s_hmap = sns.heatmap(data = df,
                    center = 500,
                    cmap = "RdBu_r")

s_hmap.axes.set_title("Education Level of Students", fontsize = 20)
s_hmap.set_ylabel(" ")
s_hmap.set_xlabel(" ")
plt.show()
plt.figure(figsize=(5, 4))
s_ages = sns.countplot(x = "Q2",
             data = students,
             order = students["Q2"].value_counts().index,
             orient = "h")
s_ages.axes.set_title("Age of Students", fontsize = 15)
s_ages.set_xlabel("Age Range", fontsize = 10)
plt.show()
world_geo = '../input/countries-of-the-world/countries.json'

#Copy and reorganize data strictly for country of origin/counts
prof_origins = professionals[["Q3"]].copy(deep=True)
prof_origins = professionals.Q3.value_counts().reset_index().rename(columns ={"index": "Country", 0: "count"})

#Remove "Other" and "Do not wish to disclose"
prof_origins = prof_origins.drop(prof_origins.index[3])
prof_origins = prof_origins.drop(prof_origins.index[9])

prof_map = folium.Map(location = [0, 0], zoom_start = 2)

folium.Choropleth(
    geo_data = world_geo,
    name = 'choropleth',
    data = prof_origins,
    columns = ["Country", "Q3"],
    key_on = "feature.properties.name",
    bins = 6,
    fill_color = 'YlOrRd',
    fill_opacity = 0.7,
    line_opacity = 0.2,
    legend_name = "Number of Professionals"
).add_to(prof_map)

folium.LayerControl().add_to(prof_map)

prof_map
fig, prof_degrees = plt.subplots(2, 1)
fig.set_size_inches(8, 10)
majors = sns.countplot(y = "Q5",
             data = professionals,
             order = professionals["Q5"].value_counts().index,
             orient = "v",
             ax = prof_degrees[0])
majors.axes.set_title("What Professionals Studied", fontsize = 15)
majors.set_ylabel("Majors", fontsize = 15)

degrees = sns.countplot(y = "Q4",
             data = professionals,
             order = professionals["Q4"].value_counts().index,
             orient = "v",
             ax = prof_degrees[1])
degrees.axes.set_title("What Degrees Professionals Earned", fontsize = 15)
degrees.set_ylabel("Education Level", fontsize = 15)
degrees.set_xlabel("Number of Students", fontsize = 15)

prof_degrees[0].set_xlabel(' ')
plt.show()
prof_heatmap = professionals[["Q4", "Q5"]].copy()
prof_heatmap.dropna()

prof_heatmap = prof_heatmap.groupby(["Q4", "Q5"]).size().reset_index(name="Number")
prof_df = prof_heatmap.pivot_table(index="Q5", columns="Q4", values="Number", fill_value = 0)

#Rearrange columns of table in sequential degree levels
cols = prof_df.columns.tolist()
prof_df = prof_df[[cols[5], cols[0], cols[3], cols[1], cols[4], cols[2]]]

#Shorten name of column for "some college"
prof_df = prof_df.rename(columns= {cols[5]: "Some College/University"})
prof_df

plt.figure(figsize=(5.5, 4))
s_hmap = sns.heatmap(data = prof_df,
                    cmap = "RdBu_r")

s_hmap.axes.set_title("Education Level of Professionals", fontsize = 20)
s_hmap.set_ylabel(" ")
s_hmap.set_xlabel(" ")
plt.show()
plt.figure(figsize=(7, 4))
p_ages = sns.countplot(x = "Q2",
             data = professionals,
             order = professionals["Q2"].value_counts().index,
             orient = "h")
p_ages.axes.set_title("Age of Professionals", fontsize = 15)
p_ages.set_xlabel("Age Range", fontsize = 10)
plt.show()
young_prof = professionals.loc[(professionals.Q2 == "25-29") & (professionals.Q4 == "Bachelor’s degree")]

fig, tools = plt.subplots(2, 1)
fig.set_size_inches(8, 10)
student_tools = sns.countplot(y = "Q12_MULTIPLE_CHOICE",
             data = students,
             order = students["Q12_MULTIPLE_CHOICE"].value_counts().index,
             orient = "v",
             ax = tools[0])
student_tools.axes.set_title("What Tools Students Use", fontsize = 15)
student_tools.set_ylabel("Tools", fontsize = 15)
student_tools.set_xlabel(" ", fontsize = 15)

yprof_tools = sns.countplot(y = "Q12_MULTIPLE_CHOICE",
             data = young_prof,
             order = young_prof["Q12_MULTIPLE_CHOICE"].value_counts().index,
             orient = "v",
             ax = tools[1])
yprof_tools.axes.set_title("What Young Professionals Use", fontsize = 15)
yprof_tools.set_ylabel("Tools", fontsize = 15)
yprof_tools.set_xlabel("Count", fontsize = 15)

plt.show()
fig, tools = plt.subplots(2, 1)
fig.set_size_inches(8, 10)
student_lang = sns.countplot(y = "Q17",
             data = students,
             order = students["Q17"].value_counts().index,
             orient = "v",
             ax = tools[0])
student_lang.axes.set_title("What Languages Students Use", fontsize = 15)
student_lang.set_ylabel("Languages", fontsize = 15)
student_lang.set_xlabel(" ", fontsize = 15)

yprof_lang = sns.countplot(y = "Q17",
             data = young_prof,
             order = young_prof["Q17"].value_counts().index,
             orient = "v",
             ax = tools[1])
yprof_lang.axes.set_title("What Languages Young Professionals Use", fontsize = 15)
yprof_lang.set_ylabel("Languages", fontsize = 15)
yprof_lang.set_xlabel("Count", fontsize = 15)

plt.show()
fig, tools = plt.subplots(2, 1)
fig.set_size_inches(8, 10)
student_lang = sns.countplot(y = "Q25",
             data = students,
             order = students["Q25"].value_counts().index,
             orient = "v",
             ax = tools[0])
student_lang.axes.set_title("Student # of Years using ML", fontsize = 15)
student_lang.set_ylabel("Languages", fontsize = 15)
student_lang.set_xlabel(" ", fontsize = 15)

yprof_lang = sns.countplot(y = "Q25",
             data = young_prof,
             order = young_prof["Q25"].value_counts().index,
             orient = "v",
             ax = tools[1])
yprof_lang.axes.set_title("Professional # of Years using ML", fontsize = 15)
yprof_lang.set_ylabel("Languages", fontsize = 15)
yprof_lang.set_xlabel("Count", fontsize = 15)

plt.show()
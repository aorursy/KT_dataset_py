import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
import plotly.offline as po
po.init_notebook_mode(connected=True)
df  = pd.read_csv("../input/job_skills.csv")
df.head()
df = df.drop("Company",axis=1)
df.shape
freq_category = [list(df["Category"]).count(i) for i in list(set(df["Category"]))]
barplot_category = [go.Bar(
    x = list(set(df["Category"])),
    y = freq_category   
)]
layout = go.Layout(
    margin = dict(
        b=140
        )
)
fig_category = go.Figure(data=barplot_category,layout=layout  ) 

po.iplot(fig_category, config={'showLink': False})
df["Minimum Qualifications"] = df["Minimum Qualifications"].fillna("Not available")
# df["Qualfications"] = df["Minimum Qualifications"].apply(lambda x: x.split("\n").strip() if x is not "Not available" else "Not available")
# df["Degree"] = df["Minimum Qualifications"].apply(lambda x: x.split("degree")[0].strip() if x is not "Not available" else "")
# df["Degree"][3]
# freq_qualification = [list(df["Degree"]).count(i) for i in list(set(df["Degree"]))]
# barplot_qualification = [go.Bar(
#     x = list(df["Degree"]),
#     y = freq_qualification   
# )]
# df["Minimum Qualifications"][0]
# po.iplot(barplot_qualification, config={'showLink': False})
df = df.sort_values("Location")
df["Cities"] = df["Location"].apply(lambda x: x.split(",")[0].strip() if len(x.split(","))>1 else "N.A." )
print(list(set((df["Cities"]))))
freq_cities = [list(df["Cities"]).count(i) for i in sorted(list(set(df["Cities"])))]
barplot_cities = [go.Scatter(
    x = sorted(list(set(df["Cities"]))),
    y = freq_cities,
    mode = "lines+markers" 
)]
po.iplot(barplot_cities, config={'showLink': False})
df["Countries"] = df["Location"].apply(lambda x: x.split(",")[len(x.split(","))-1].strip() if len(x.split(","))>1 else x.split(",")[0].strip())
# list((df["Countries"])).remove("United States")
list_of_countries = list(set(df["Countries"]))
list_of_countries.remove("USA")
list_of_countries.remove("United States")
print(list_of_countries)
freq_countries = [list(df["Countries"]).count(i) for i in sorted(list_of_countries)]
barplot_countries = [go.Scatter(
    x = sorted(list_of_countries),
    y = freq_countries,
    mode = "lines+markers" ,
    hoverinfo = 'y',
)]
layout = go.Layout(
    margin = dict(
    b=130
    ))
fig_countries = go.Figure(data=barplot_countries, layout=layout)
po.iplot(fig_countries, config={'showLink': False})
it_jobs = ['Data Center & Network',
          'Hardware Engineering',
          'IT & Data Management',
          'Network Engineering',
          'Software Engineering',
          'Technical Infrastructure',
          'User Experience & Design'
         ] 
df = df.drop("Location",axis = 1)
df.head()
countries_with_it_jobs=[]
for i in range(len(df["Category"])):
    if df["Category"][i] in it_jobs:
        countries_with_it_jobs.append(df["Countries"][i])
freq_countries_it_jobs = [countries_with_it_jobs.count(i) for i in sorted(list(set(countries_with_it_jobs)))]
freq_countries_it_jobs
barplot_it_jobs = [go.Scatter(
    x = sorted(list(set(countries_with_it_jobs))),
    y = freq_countries_it_jobs,
    mode = "lines+markers" ,
    hoverinfo = 'y',
)]
layout = go.Layout(
    margin = dict(
    b=130
    ))
fig_countries_it_jobs = go.Figure(data=barplot_it_jobs, layout=layout)
po.iplot(fig_countries_it_jobs, config={'showLink': False})
countries_with_it_jobs1 = sorted(list(set(countries_with_it_jobs)))
countries_with_it_jobs1.remove("United States")
countries_with_it_jobs1
freq_countries_it_jobs = [countries_with_it_jobs.count(i) for i in sorted(list(set(countries_with_it_jobs1)))]
freq_countries_it_jobs
barplot_it_jobs = [go.Scatter(
    x = sorted(list(set(countries_with_it_jobs1))),
    y = freq_countries_it_jobs,
    mode = "lines+markers" ,
    hoverinfo = 'y',
)]
layout = go.Layout(
    margin = dict(
    b=130
    ))
fig_countries_it_jobs = go.Figure(data=barplot_it_jobs, layout=layout)
po.iplot(fig_countries_it_jobs, config={'showLink': False})
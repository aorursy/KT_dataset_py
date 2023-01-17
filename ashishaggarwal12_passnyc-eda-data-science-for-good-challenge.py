import pandas as pd
import numpy as np
import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
import os
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import plotly.graph_objs as go
from plotly import tools
print(os.listdir("../input"))
school=pd.read_csv('../input/2016 School Explorer_new.csv')
school=pd.DataFrame(school)
print(school.head(6,))
school.shape
school.dtypes
#Removing 'Adjusted Grade' , 'New?' and 'Other Location Code in LCGMS' columns.
school=school.drop(['Adjusted Grade','New?','Other Location Code in LCGMS'],axis=1)
print(school.shape)
print(school.head(6,))
#Removing $ and , from 'School Income Estimate' and converting it to float
school["School Income Estimate"]=school["School Income Estimate"].replace('[\$,]','',regex=True)
school["School Income Estimate"]=school["School Income Estimate"].astype(float)
print(type(school["School Income Estimate"]))
school["School Income Estimate"]
def percent_remove(x):
    school[x]=school[x].replace('[%$]','',regex=True).astype(float)
percent_remove("Percent ELL")
percent_remove("Percent Asian")
percent_remove("Percent Black")
percent_remove("Percent Hispanic")
percent_remove("Percent Black / Hispanic")
percent_remove("Percent White")
percent_remove("Student Attendance Rate")
percent_remove("Percent of Students Chronically Absent")
percent_remove("Rigorous Instruction %")
percent_remove("Collaborative Teachers %")
percent_remove("Supportive Environment %")
percent_remove("Effective School Leadership %")
percent_remove("Strong Family-Community Ties %")
percent_remove("Trust %")
school.dtypes
NA_count = school.isnull().sum().reset_index()
missing_values = NA_count[NA_count[0] != 0]
missing_values["%"]=(missing_values[0]/school.shape[0])*100
missing_values = missing_values.sort_values(by = "%",ascending =False)
missing_values
sns.set_style("whitegrid")
plt.figure(figsize=(8,8))
ax = sns.barplot("%","index",data=missing_values,
                 linewidth=1 ,palette="vlag",edgecolor="k"*len(missing_values))
plt.ylabel("columns")
for i,j in enumerate(np.around(missing_values["%"],1).astype(str) + " %"):
    ax.text(.7,i,j ,weight = "bold")
plt.title("Percentage of missing values in Schools data")
plt.grid(True)
plt.show()
#Imputing missing values in numeric variables by mean 
school["School Income Estimate"] = school["School Income Estimate"].fillna(school["School Income Estimate"].mean())
school["Economic Need Index"] = school["Economic Need Index"].fillna(school["Economic Need Index"].mean())
school["Student Attendance Rate"] = school["Student Attendance Rate"].fillna(school["Student Attendance Rate"].mean())
school["Percent of Students Chronically Absent"] = school["Percent of Students Chronically Absent"].fillna(school["Percent of Students Chronically Absent"].mean())
school["Rigorous Instruction %"] = school["Rigorous Instruction %"].fillna(school["Rigorous Instruction %"].mean())
school["Collaborative Teachers %"] = school["Collaborative Teachers %"].fillna(school["Collaborative Teachers %"].mean())
school["Average ELA Proficiency"] = school["Average ELA Proficiency"].fillna(school["Average ELA Proficiency"].mean())
school["Average Math Proficiency"] = school["Average Math Proficiency"].fillna(school["Average Math Proficiency"].mean())
school["Percent Asian"] = school["Percent Asian"].fillna(school["Percent Asian"].mean())
school["Percent Black"] = school["Percent Black"].fillna(school["Percent Black"].mean())
school["Percent Hispanic"] = school["Percent Hispanic"].fillna(school["Percent Hispanic"].mean())
school["Percent White"] = school["Percent White"].fillna(school["Percent White"].mean())
school["Rigorous Instruction %"] = school["Rigorous Instruction %"].fillna(school["Rigorous Instruction %"].mean())
school["Collaborative Teachers %"] = school["Collaborative Teachers %"].fillna(school["Collaborative Teachers %"].mean())
school["Supportive Environment %"] = school["Supportive Environment %"].fillna(school["Supportive Environment %"].mean())
school["Effective School Leadership %"] = school["Effective School Leadership %"].fillna(school["Effective School Leadership %"].mean())
school["Strong Family-Community Ties %"] = school["Strong Family-Community Ties %"].fillna(school["Strong Family-Community Ties %"].mean())
school["Trust %"] = school["Trust %"].fillna(school["Trust %"].mean())

#Imputing missing values in categorical variables by 'Unknown' 
school["Rigorous Instruction Rating"] = school["Rigorous Instruction Rating"].fillna("Unknown")
school["Collaborative Teachers Rating"] = school["Collaborative Teachers Rating"].fillna("Unknown")
school["Supportive Environment Rating"] = school["Supportive Environment Rating"].fillna("Unknown")
school["Effective School Leadership Rating"] = school["Effective School Leadership Rating"].fillna("Unknown")
school["Strong Family-Community Ties Rating"] = school["Strong Family-Community Ties Rating"].fillna("Unknown")
school["Trust Rating"] = school["Trust Rating"].fillna("Unknown")
school["Student Achievement Rating"] = school["Student Achievement Rating"].fillna("Unknown")
#Checking all columns for missing values after performing missing value treatment
NA_count1 = school.isnull().sum().reset_index()
missing_values1 = NA_count1[NA_count1[0] != 0]
missing_values1
#Taking Latitude and Longitude values of all unique cities for locating in a map
unique_city=school['City'].unique()
locations_index=[]
locationlist1=[]
locationlist2=[]
#locationlist3=pd.DataFrame()
locations = school[['Latitude', 'Longitude']]
ab=school['City'].tolist()
for i in unique_city:    
    locations_index.append(ab.index(i))
for j in locations_index:
    locationlist1.append(locations.iloc[j,0])
    locationlist2.append(locations.iloc[j,1])
#locationlist = locationlist1.values.tolist()
#len(locationlist)
dict1={"l1":locationlist1,"l2":locationlist2}
locationlist3=pd.DataFrame(dict1)
locationlist3
locations = locationlist3[['l1', 'l2']]
locationlist = locations.values.tolist()
# locationlist contains Latitude and Longitude values of all unique cities
locationlist
for point in range(0, len(unique_city)):
    map = folium.Map(location=[40.714301, -73.982966], zoom_start=12)
    folium.Marker(locationlist[point], popup=school['School Name'][point]).add_to(map)
map
ab=school.describe()
print(ab)
tab = pd.crosstab(index = school["City"],  columns=school["Community School?"], colnames = ['']) 
print(tab)
NYC_Asians=school[(school.City=="NEW YORK")& (school["Community School?"]=="Yes")]
print(NYC_Asians["Percent Asian"])
plt.hist(NYC_Asians["Percent Asian"],bins=18)
plt.title("Histogram of of Asians in Community schools in New York")
plt.xlabel("Asians")
plt.ylabel("Percent")
plt.show()
school["Others"]=1-(school["Percent Asian"]+school["Percent Black"]+school["Percent Hispanic"]+school["Percent White"])

#Assigning 0 of -ve % value
school.iloc[3,158]=0
school.iloc[3,158]
#initializing plotly offline for ipython notebooks
init_notebook_mode(connected=True)
#Grades High and Low across region
%matplotlib inline
sns.countplot(school["Grade Low"],palette="vlag")
sns.countplot(school["Grade High"],palette="vlag")
sns.countplot(school["Community School?"])
data = []
city_list = list(school["City"].value_counts().index)
for i in city_list:
    data.append(
        go.Bar(
          y = [school["Percent Asian"][school["City"] == i].mean(), school["Percent Black"][school["City"] == i].mean(), school["Percent Hispanic"][school["City"] == i].mean(), school["Percent White"][school["City"] == i].mean(), school["Others"][school["City"] == i].mean()],
          x = ['Asian','Black','Hispanic', 'White', 'Others'],
          name = i,
          opacity = 0.6
        )
    )
k=0
fig = tools.make_subplots(rows=15, cols=3, subplot_titles=city_list, print_grid=False)
for i in range(1,16):
    for j in range(1,4):
        fig.append_trace(data[k], i, j)
        k = k + 1
fig['layout'].update(height=2000, title='Average racial distribution in different cities',showlegend=False)
iplot(fig)
y=[]
city_list = list(school["City"].value_counts().index)
#print(city_list)
for i in city_list:
    y.append(school["Average ELA Proficiency"][school["City"] == i].mean())
dframe=[('City',city_list),('Average ELA Score',y)]
new_df1 = pd.DataFrame.from_items(dframe)
new_df1
data = [go.Bar(
            x=new_df1["City"],
            y=new_df1["Average ELA Score"],
            text=new_df1["Average ELA Score"],
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )]
iplot(data)
y=[]
city_list = list(school["City"].value_counts().index)
#print(city_list)
for i in city_list:
    y.append(school["Average Math Proficiency"][school["City"] == i].mean())
dframe=[('City',city_list),('Average Math Proficiency',y)]
new_df = pd.DataFrame.from_items(dframe)
new_df
data = [go.Bar(
            x=new_df["City"],
            y=new_df["Average Math Proficiency"],
            text=new_df["Average Math Proficiency"],
            textposition = 'auto',
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )]
iplot(data)
trace1 = go.Bar(
    y=new_df1['City'],
    x=new_df1['Average ELA Score'],
    name='Average ELA Score',
    orientation = 'h'
)
trace2 = go.Bar(
    y=new_df['City'],
    x=new_df['Average Math Proficiency'],
    name='Average Math Proficiency',
    orientation = 'h'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack',
    showlegend = True,
    margin=go.Margin(
        l=350,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    height = 800,
    
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


#### School Performance Features
features = ['Rigorous Instruction %',
'Collaborative Teachers %',
'Supportive Environment %',
'Effective School Leadership %',
'Strong Family-Community Ties %',
'Trust %']

school[['School Name'] + features ].head(10,)
corr = school[features].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr, cmap='PuBuGn')
temp = plt.xticks(rotation=75, fontsize=11) 
temp = plt.yticks(fontsize=11)
plt.figure(figsize=(12,7))
temp = sns.distplot(school[['Economic Need Index']].values, kde=True,color = 'y')
temp= plt.title("ENI distribution", fontsize=15)
temp = plt.xlabel("ENI", fontsize=15)
temp = plt.ylabel("School count", fontsize=15)
sns.lmplot(x='School Income Estimate', y='Economic Need Index', data=school,
           fit_reg=False)
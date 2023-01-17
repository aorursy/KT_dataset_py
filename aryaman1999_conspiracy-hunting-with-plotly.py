# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pb_poverty = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv",encoding = "ISO-8859-1")

p_killing_US = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding = "ISO-8859-1")

s_race_city = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv",encoding = "ISO-8859-1")

P_HighSchool = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv",encoding = "ISO-8859-1")

M_2015 = pd.read_csv("/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv",encoding = "ISO-8859-1")
pb_poverty.head()
pb_poverty.info()
import plotly.graph_objects as go

def plot_doughnut_chart_by_race(df,geo_feature_name,area_name,col_name,num_cities=10):

    '''

    Plots doughnut chart by area

    

    Args: df: The dataframe from which data is to be plotted

          geo_feature_name: Geographical feature column name

          area_name: The area whose doughnut chart is to be plotted

          col_name: The feature which is to be plotted

          num_cities: top number of cities to be plotted

    

    Returns: None

    

    Output: Doughnut chart representing share of that particular race in that particular area. 

    '''

    area = df[df[geo_feature_name]==area_name]

    area = area.sort_values(by=["City"],ascending=False)

    fig = go.Figure(data=[go.Pie(labels=area["City"][:num_cities], values=area[col_name][:num_cities], hole=.3)])

    fig.update_layout(title="Top "+str(num_cities)+" cities in terms of "+col_name+" in "+area_name,template="plotly_dark")

    fig.show()

plot_doughnut_chart_by_race(pb_poverty,"Geographic Area","AK","poverty_rate")
pb_poverty = pb_poverty[pb_poverty["poverty_rate"]!="-"]

pb_poverty["poverty_rate"] = pb_poverty["poverty_rate"].astype("float")
poverty_state = pb_poverty.groupby(["Geographic Area"])

poverty_state = poverty_state.mean()
poverty_state.head()
import plotly.express as px

fig = px.bar(x=poverty_state.index,y=poverty_state.poverty_rate)

fig.update_layout(title="Poverty Rate Visualization",xaxis_title="State",yaxis_title="Poverty rate",template="plotly_dark")

fig.update_traces(marker_color="mediumseagreen")
fig = px.choropleth(locations=poverty_state.index, locationmode="USA-states", color=poverty_state.poverty_rate, scope="usa",template="plotly_dark")

fig.show()
s_race_city.head()
s_race_city = s_race_city[s_race_city!="(X)"]

s_race_city.dropna(inplace=True)

s_race_city.info()
for col in s_race_city.columns:

    if col not in ["Geographic area","City"]:

        print(col)

    

        s_race_city[col] = pd.to_numeric(s_race_city[col])

        

s_race_state = s_race_city.groupby(["Geographic area"])

s_race_state = s_race_state.mean()
s_race_state.head()
from plotly.subplots import make_subplots

main_plot = make_subplots(rows=1, cols=3)





fig1 = px.choropleth(locations=s_race_state.index, locationmode="USA-states", color=s_race_state.share_white, scope="usa",template="plotly_dark")

fig1.update_layout(title="USA White")

fig2 = px.choropleth(locations=s_race_state.index, locationmode="USA-states", color=s_race_state.share_black, scope="usa",template="plotly_dark")

fig2.update_layout(title="USA Black")

fig3 = px.choropleth(locations=s_race_state.index, locationmode="USA-states", color=s_race_state.share_hispanic, scope="usa",template="plotly_dark")

fig3.update_layout(title="USA Hispanic")



fig1.show()

fig2.show()

fig3.show()
import plotly.graph_objects as go

def plot_doughnut_chart_by_area(df,city_name):

    '''

    Plots doughnut chart by area

    

    Args: df: The dataframe from which data is to be plotted

          city_name: name of city

         

    

    Returns: None

    

    Output: Doughnut chart representing share of that particular race in that particular city. 

    '''

    area = df[df["City"]==city_name]

    

    fig = go.Figure(data=[go.Pie(labels=area.columns[2:], values=area.iloc[:,2:].values[0], hole=.3)])

    fig.update_layout(title="Race distribution in "+city_name,template="plotly_dark")

    fig.show()

plot_doughnut_chart_by_area(s_race_city,"Woodson CDP")
p_killing_US.rename(columns={"state":"Geographic Area"},inplace=True)

p_killing_US.head()
groups  = p_killing_US.groupby("Geographic Area")

counts = groups.count()["id"]

fig3 = px.choropleth(locations=counts.index, locationmode="USA-states", color=counts.values, scope="usa",template="plotly_dark")

fig3.update_layout(title="Number of Killings in the USA by state")

def plot_racial_shootings(df,race_name):

    '''Plots chlorpeth map showing number of people killed by race

       args: df: Name of dataframe

             race_name: Name of race

       output: Map colour coded according to number of people of the given race

       returns None

    '''

    race_dict = {"B":"Black","W":"White","A":"Asian","O":"Others","H":"Hispanic","N":"Native American"}

    groups2 = df.groupby(["Geographic Area","race"])

    states = groups2.count().id.xs(race_name, level=1, drop_level=False).index.get_level_values(0)

    share_black_shot = groups2.count().id.xs(race_name, level=1, drop_level=False).values

    fig3 = px.choropleth(locations=states, locationmode="USA-states", color=share_black_shot, scope="usa",template="plotly_dark")

    

    fig3.update_layout(title="Number of "+race_dict[race_name]+" Killings in the USA by state")

    fig3.show()

plot_racial_shootings(p_killing_US,"B")
plot_racial_shootings(p_killing_US,"W")

plot_racial_shootings(p_killing_US,"H")

plot_racial_shootings(p_killing_US,"N")

plot_racial_shootings(p_killing_US,"A")

from plotly.subplots import make_subplots



def show_count_plot(df,rows,cols,start_index=0):

    '''

    This function plots the counts of the desired features in the data

    

    args: df: Data to plot

          start_index: Index to start plotting from

          rows: Number of rows in subplot

          cols: Number of columns in subplot

    

    returns: None

    

    output: Countplots of required features

    

    

    '''



    s_titles = [col for col in df.columns[start_index:]]

    fig = make_subplots(rows=rows,cols=cols,subplot_titles=(s_titles))

    k = start_index

    for i in range(1,rows+1):

        for j in range(1,cols+1):

            plot_data = p_killing_US.iloc[:,k].value_counts()

            col_name = p_killing_US.columns[k]

            fig.add_trace(

            go.Bar(x=plot_data.index,y=plot_data.values,name=col_name),

            row=i,col=j,

            )

            k+=1

            if(k>13):

                break

    fig.update_layout(width=1000,height=1000,template="plotly_dark")

    fig.show()

    

show_count_plot(p_killing_US,4,3,start_index=3)
body_cam = p_killing_US[p_killing_US["body_camera"]==True]

body_cam.head()
body_cam = p_killing_US[p_killing_US["body_camera"]==False]

body_cam.head()
ser = body_cam.race.value_counts()

ser_total = p_killing_US.race.value_counts()

ser_div = ser.divide(ser_total)

fig = go.Figure(data=[go.Pie(labels=ser_div.index, values=ser_div.values, hole=.3)])

fig.update_layout(title="Percentage racial distribution of shootings where the body camera was turned off",template="plotly_dark")

fig.show()
def plot_doughnut_composition(df,feature_name,category_name,og_df):

    '''

    Plots doughnut chart of racial composition fitting a certain criteria

    

    args: df: input dataframe

          feature_name: feature being plotted

          category_name: category to be avoided eg. don't visualise people who were not fleeing

          og_df: original dataframe from which df is selected

    output: Doughnut plot

    returns: None

    '''



    fleeing = df

    for c in category_name:

        fleeing = fleeing[fleeing[feature_name]!=c]



    fleeing.dropna(inplace=True)



    fleeing_race = fleeing.race.value_counts()/og_df.race.value_counts()

    

    fig = go.Figure(data=[go.Pie(labels=fleeing_race.index, values=fleeing_race.values, hole=.3)])

    fig.update_layout(title="Percentage racial distribution of shootings where the body camera was turned off and were termed to be fleeing",template="plotly_dark")

    fig.show()



plot_doughnut_composition(body_cam,"flee",["Not fleeing"],p_killing_US)

    
fleeing = body_cam[body_cam["flee"]!="Not fleeing"]

fleeing.dropna(inplace=True)

fleeing_race = fleeing.race.value_counts()/p_killing_US.race.value_counts()

print(fleeing_race)

fig2 = px.bar(x=fleeing_race.index,y=fleeing_race.values,template="plotly_dark")

fig2.update_traces(marker_color="mediumseagreen")

fig2.update_layout(title="Fleeing americans without bodycam",xaxis_title="race",yaxis_title="count")

fig2.show()


plot_doughnut_composition(body_cam,"armed",["undetermined","unarmed"],p_killing_US)

guns = body_cam[body_cam["armed"]=="gun"]

guns_race = guns.race.value_counts()/p_killing_US.race.value_counts()

fig = go.Figure(data=[go.Pie(labels=guns_race.index, values=guns_race.values, hole=.3)])

fig.update_layout(title="Distribution of gun ownership of the victims",template="plotly_dark")

fig.show()
fig = px.bar(x=guns_race.index,y=guns_race.values)

fig.update_layout(title="Distribution of guns across races",xaxis_title="Race",yaxis_title="Counts",template="plotly_dark")

fig.update_traces(marker_color="mediumseagreen")
p_killing_US.head()
fleeing = p_killing_US[p_killing_US["flee"]!="Not fleeing"]

groups = fleeing.groupby(["body_camera","race"])

grouped_counts = groups.count().dropna()["id"]

arr1 = grouped_counts.values[:6]

arr2 = grouped_counts.values[6:]

arr2 = np.insert(arr2,4,0)



body_camera = ["A","B","H","N","O","W"]



fig = go.Figure()



fig.add_trace(go.Bar(

        x = body_camera,

        y = arr1,

        name="No body camera"

))



fig.add_trace(go.Bar(

       x = body_camera,

       y = arr2,

       name= "With body camera"

))



fig.update_layout(barmode="group",title="Racial distribution of people shot without body cam reported to be fleeing",template="plotly_dark")

fig.show()

armed = p_killing_US[p_killing_US["armed"]!="unarmed"]

armed = armed[armed["armed"]!="undetermined"]

groups = fleeing.groupby(["body_camera","race"])

grouped_counts = groups.count().dropna()["id"]

race_counts = p_killing_US.race.value_counts().values



arr1 = grouped_counts.values[:6]



arr2 = grouped_counts.values[6:]

arr2 = np.insert(arr2,4,0)

body_camera = ["A","B","H","N","O","W"]



fig = go.Figure()



fig.add_trace(go.Bar(

        x = body_camera,

        y = arr1,

        name="No body camera"

))



fig.add_trace(go.Bar(

       x = body_camera,

       y = arr2,

       name= "With body camera"

))



fig.update_layout(barmode="group",title="Racial distribution of people without body cams reported to be armed",template="plotly_dark")

fig.show()

p_killing_US.threat_level.value_counts()
threat = p_killing_US[p_killing_US.threat_level == "attack"]

threat_percentage = threat.race.value_counts()/p_killing_US.race.value_counts()

fig = go.Figure(data=[go.Pie(labels=threat_percentage.index, values=threat_percentage.values, hole=.3)])

fig.update_layout(template="plotly_dark")

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=threat_percentage.index,y=threat_percentage.values))

fig.update_layout(title="Attack level distribution",xaxis_title="Race",yaxis_title="Count",template="plotly_dark")

fig.show()

groups = p_killing_US.groupby(["threat_level","race"])

grouped_counts = groups.count().dropna()["id"]

arr1 = grouped_counts.values[:6]



arr2 = grouped_counts.values[6:12]

arr3 = grouped_counts.values[12:]

arr3 = np.insert(arr3,4,0)

race = ["A","B","H","N","O","W"]



fig = go.Figure()



fig.add_trace(go.Bar(

        x = race,

        y = arr1,

        name="attack"

))



fig.add_trace(go.Bar(

       x = race,

       y = arr2,

       name= "other"

))



fig.add_trace(go.Bar(

       x = race,

       y=arr3,

       name="undetermined"

))

fig.update_layout(barmode="group",title="Racial distribution of people without body cams reported to be armed",xaxis_title="race",yaxis_title="count",template="plotly_dark")

fig.show()

groups = p_killing_US.groupby(["threat_level","race"])

grouped_counts = groups.count().dropna()["id"]

fig = px.bar(x=p_killing_US["Geographic Area"].value_counts().index,y=p_killing_US["Geographic Area"].value_counts().values)



fig.update_layout(title="Distribution of attackers according to state",xaxis_title="State",

                  yaxis_title="Number of killings",template="plotly_dark")
def plot_statewise_bar(df,state_name):

    '''Plots statewise racial distribution bar plot

    

    args: df: Dataframe from which data is to be plotted

             state_name: Name of state which is to be plotted

    

    output: barplot of racial distribution

    

    returns None

    '''

    CA = df[df["Geographic Area"]==state_name]

    counts = CA.race.value_counts()

    fig = px.bar(x=counts.index,y=counts.values,template="plotly_dark")

    fig.update_layout(title=state_name+" killings racial distribution",xaxis_title="Race",yaxis_title="Count")

    fig.show()

plot_statewise_bar(p_killing_US,"CA")
plot_statewise_bar(p_killing_US,"TX")
plot_statewise_bar(p_killing_US,"FL")
P_HighSchool.info()
P_HighSchool = P_HighSchool[P_HighSchool.percent_completed_hs!="-"]
P_HighSchool.percent_completed_hs = pd.to_numeric(P_HighSchool.percent_completed_hs)
### Again lets go back to chloropeth maps

grouped_state = P_HighSchool.groupby("Geographic Area")

hs_state = grouped_state.mean()
fig1 = px.choropleth(locations=hs_state.index, locationmode="USA-states", color=hs_state.percent_completed_hs, scope="usa",template="plotly_dark")

fig1.show()
##### Composition #####
M_2015.head()
M_2015 = M_2015[M_2015["Median Income"]!="(X)"]

M_2015.dropna(inplace=True)

temp_list = []

for rec in M_2015["Median Income"]:

    

    if ("-" in rec):

        rec = rec.rstrip("-")

    if("+" in rec):

        rec = rec.rstrip("+")

    if("," in rec):

        ls = rec.split(",")

        rec = "".join(ls)

    temp_list.append(rec)

M_2015["Median Income"] = temp_list
M_2015["Median Income"] = pd.to_numeric(M_2015["Median Income"])

grouped_count = M_2015.groupby("Geographic Area")

M_2015_state = grouped_count.mean()
M_2015_state.head()
fig1 = px.choropleth(locations=M_2015_state.index, locationmode="USA-states", color=M_2015_state["Median Income"], scope="usa",template="plotly_dark")

fig1.show()
#pb_poverty 

#p_killing_US 

#s_race_city 

#P_HighSchool 

#M_2015 

s_race_city = s_race_city.rename(columns={"Geographic area":"Geographic Area"})

temp1 = pb_poverty.merge(s_race_city,on=["City","Geographic Area"])

#temp2 = temp1.merge(p_killing_US,on="City")

temp2 = temp1.merge(P_HighSchool,on=["City","Geographic Area"])

temp3 = temp2.merge(M_2015,on=["City","Geographic Area"])

temp3 =temp3.rename(columns={"City":"city"})

p_killing_US = p_killing_US.rename(columns={"state":"Geographic Area"})

temp3.head()
### Plot poverty_rate against white####



fig1 = px.histogram(temp3,x="poverty_rate",y="share_white",height=600,width=900,template="plotly_dark")

fig2 = px.scatter(temp3,x="poverty_rate",y="share_white",height=600,width=900,template="plotly_dark")

fig1.update_layout(title="Share of white people vs the poverty rate")

fig1.update_traces(marker_color="mediumseagreen")

fig1.show()

fig2.show()
### Plot poverty_rate against share black###

fig1 = px.histogram(temp3,x="poverty_rate",y="share_black",height=600,width=900,template="plotly_dark")

fig1.update_layout(title="Share of black people vs the poverty rate")

fig1.update_traces(marker_color="mediumseagreen")

fig2 = px.scatter(temp3,x="poverty_rate",y="share_black",height=600,width=900,template="plotly_dark")

fig1.show()

fig2.show()
###Plot poverty_rate against share_hispanic###

fig1 = px.histogram(temp3,x="poverty_rate",y="share_hispanic",height=600,width=900,template="plotly_dark")

fig1.update_layout(title="Share of hispanic people vs the poverty rate")

fig1.update_traces(marker_color="mediumseagreen")

fig2 = px.scatter(temp3,x="poverty_rate",y="share_hispanic",height=600,width=900,template="plotly_dark")

fig1.show()

fig2.show()
###Plot poverty_rate against share_native###



fig1 = px.histogram(temp3,x="poverty_rate",y="share_native_american",height=600,width=900,template="plotly_dark")

fig1.update_layout(title="Share of Native American people vs the poverty rate")

fig1.update_traces(marker_color="mediumseagreen")

fig2 = px.scatter(temp3,x="poverty_rate",y="share_native_american",height=600,width=900,template="plotly_dark")

fig2.update_layout(title="Share of Native American people vs the poverty rate")

fig1.show()

fig2.show()
###Plot poverty_rate against share_asian###



fig1 = px.histogram(temp3,x="poverty_rate",y="share_asian",height=600,width=900,template="plotly_dark")

fig1.update_layout(title="Share of Asian people vs the poverty rate")

fig1.update_traces(marker_color="mediumseagreen")

fig2 = px.scatter(temp3,x="poverty_rate",y="share_asian",height=600,width=900,template="plotly_dark")

fig1.show()

fig2.show()
fig = px.scatter(temp3,x="poverty_rate",y="percent_completed_hs",template="plotly_dark")

fig.update_traces(marker_color="mediumseagreen")

fig.show()
fig = px.scatter(temp3,x="Median Income",y="percent_completed_hs",template="plotly_dark")

fig.update_traces(marker_color="mediumseagreen")

fig.show()
fig = px.scatter(temp3,x="poverty_rate",y="Median Income",template="plotly_dark")

fig.update_traces(marker_color="mediumseagreen")

fig.show()
p_killing_US.head()
def plot_feature_vs_num_killed(df,feature):

    '''Plots a feature mean per state vs number of people killed per state

        args: df: Name of dataframe

             feature: Name of feature

        output: scatterplot of feature vs num killed

        returns: None

    '''



    group = df.groupby("Geographic Area")

    group_counts = group.count().id

    group_t3 = temp3.groupby("Geographic Area")

    mean_poverty_rate = group_t3.mean()[feature]

    df_mean = pd.concat([mean_poverty_rate,group_counts],axis=1)

    df_mean.rename(columns={"id":"Number of people killed"},inplace=True)

    fig = px.scatter(x=df_mean[feature],y=df_mean["Number of people killed"],color=df_mean.index,template="plotly_dark")

    fig.update_layout(title="Mean poverty rate per state by number of people killed per state",xaxis_title=feature,yaxis_title="Num Killed")

    #fig.update_traces(marker_color="mediumseagreen")

    fig.show()

plot_feature_vs_num_killed(p_killing_US,"poverty_rate")
plot_feature_vs_num_killed(p_killing_US,"percent_completed_hs")
plot_feature_vs_num_killed(p_killing_US,"Median Income")
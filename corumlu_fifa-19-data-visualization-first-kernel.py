# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go 

from plotly.offline import init_notebook_mode, iplot, plot

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import seaborn as sns 



init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
fdata = pd.read_csv("../input/fifa19/data.csv")



fdata = fdata.iloc[:,[

 0,  # Index

 2,  # Name

 3,  # Age

 5,  # Nationality

 7,  # Overall

 8,  # Potential

 11, # Value

 12, # Wage

 14, # Preferred Foot

 19, # Body Type 

 21, # Position

 22, # Jersey Number

 26, # Height

 27, # Weight

 69, # Shot Power

 71, # Stamina

 74, # Aggression

 75, # Interceptions

 76, # Positioning

 77, # Vision

 79, ]] # Composure
fdata.corr()
fdata.rename(columns = {"Preferred Foot" : "preferred_foot",

                        "International Reputation" : "International_Reputation",

                        "Skill Moves" : "Skill_Moves",

                        "Work Rate" : "Work_Rate",

                        "Body Type" : "Body_Type",     

                        "Jersey Number" : "Jersey_Number"

                         },inplace = True)

# 48 player has Nan values in so much feature so I deleted these 48 players. 

fdata.drop(fdata[fdata.Body_Type.isnull()].index, inplace = True)

fdata.isnull().sum()


fdata["Position"].fillna("UN",inplace = True )

fdata["Jersey_Number"].fillna("99",inplace = True)

fdata["Jersey_Number"] = fdata["Jersey_Number"].astype("float")
nationalities = list(fdata.Nationality.unique())

mean_list = []

nationalities_mean_weight = {}

def weight_lbs(value) :

    output = value.replace("lbs","")

    return output



for j in nationalities :

    weight = fdata[fdata["Nationality"] == j ].Weight.apply(lambda x  : weight_lbs(x)).astype("float").mean()

    mean_list.append(weight)

    

data_dict = {"nationalities" : nationalities,"weight_mean" : mean_list}



last_data = pd.DataFrame.from_dict(data_dict).sort_values(by = "weight_mean",ascending = False)



trace = go.Scatter(x = last_data.nationalities , 

                  y = last_data.weight_mean, 

                  mode = "lines")





layout = {"title" : "Average weight for Nationalities ","xaxis": {"title"  : "Nationalities"},"yaxis" : {"title": "Weight(Lbs)"}}

iplot({"data" : [trace],"layout" : layout})



plt.show()
Forward =  ["RF" , "ST" , "LW" , "LF" , "RS" , "LS" , "RW" , "CF"] 

Middle = ["RCM" , "LCM" , "LDM" , "CAM" , "CDM" , "RM" , "LAM" , "LM" ,"RDM" , "CM" , "RAM" ]

Stopper = ["RCB" , "CB" , "LCB" , "LB" , "RB" , "RWB" , "LWB" ]   

Goal_Keeper =  ["GK"] 

    

forward_data = pd.DataFrame()

middle_data = pd.DataFrame()

stopper_data = pd.DataFrame()

gk_data = fdata[fdata.Position == "GK"]



for i in Forward : 

    forward_data = forward_data.append(fdata[fdata.Position == i])

for i in Middle : 

    middle_data = middle_data.append(fdata[fdata.Position == i])

for i in Stopper :

    stopper_data = stopper_data.append(fdata[fdata.Position == i])



# 

fig = make_subplots(rows=2, cols=2)

fig.add_trace(go.Histogram(x = forward_data.Jersey_Number ,name= "Forward" , histnorm='probability'),row = 1 , col = 1)

fig.add_trace(go.Histogram(x = middle_data.Jersey_Number , name = "Middle" , histnorm='probability'), row = 1 , col =2 )

fig.add_trace(go.Histogram(x = stopper_data.Jersey_Number , name = "Stopper" , histnorm='probability') , row = 2 ,col=1)

fig.add_trace(go.Histogram(x = gk_data.Jersey_Number , name = "Goal Keeper" , histnorm='probability'),row=2,col=2)

fig.update_layout(height=750, width=1300,title = "Distribution of Jersey Number by Positions")

fig.show()
preferred_data = pd.DataFrame({"Nationality" : [],"Right": [],"Left" : []} )



index = 0   

for i in nationalities : 

    preferred_data.loc[index] = fdata[fdata.Nationality == i].preferred_foot.value_counts() 

    preferred_data.loc[index,"Nationality"] = i

    index+=1

    



preferred_data.sort_values(by = ["Right","Left"],ascending= False,inplace = True)

fig = go.Figure([go.Bar(x = preferred_data.Right,y = preferred_data.Nationality.head(50),orientation = "h",name ="Right Foot"),go.Bar(x = preferred_data.Left,y = preferred_data.Nationality.head(50),orientation = "h",name = "Left Foot")])      

fig.update_layout(barmode='overlay')

fig.show()
bmi_data = fdata.loc[:,["Name","Height","Weight"]]



def to_bmi(data) :

    return  data[1]/ (data[0] **2 ) * 10000



def to_kg(lbs) :

    kg = float(lbs[0].replace("lbs","")) / 2.2046

    return kg



def to_m(height) : 

    meters = float(height[0].replace("'",".")[0]) * 30.48 + float(height[0].replace("'",".")[2]) * 2.54

    return meters



bmi_data["height_meters"] = bmi_data.loc[:,["Height"]].apply(lambda x :to_m(x) , axis =  1)

bmi_data["weight_kg"] = bmi_data.loc[:,["Weight"]].apply(lambda x :to_kg(x) , axis =  1)

bmi_data["bmi"] = bmi_data.loc[:,["height_meters","weight_kg"]].apply(lambda x :to_bmi(x) , axis =  1)



under_data = bmi_data[bmi_data.bmi < 18.5 ]

normal_data = bmi_data[bmi_data.bmi > 18.5  ]

over_data = bmi_data[bmi_data.bmi >  24.9]

very_data = bmi_data[bmi_data.bmi > 29.9 ]



bmi_data.bmi = bmi_data.bmi.astype("str")

trace1 = go.Scatter(x =under_data.weight_kg, y = under_data.height_meters, mode = "markers" , marker = dict(color = "rgba(0,0,240,0.8)" ), text = bmi_data.bmi + bmi_data.Name ,name = "Underweight")

trace2 = go.Scatter(x =normal_data.weight_kg, y = normal_data.height_meters, mode = "markers" , marker = dict( color = "rgba(0,255,0,0.8)"), text = bmi_data.bmi + bmi_data.Name, name = "Normal Weight")

trace3 = go.Scatter(x =over_data.weight_kg, y = over_data.height_meters, mode = "markers" , marker = dict( color = "rgba(255,128,0,0.8)"), text = bmi_data.bmi + bmi_data.Name, name = "Overweight")

trace4 = go.Scatter(x =very_data.weight_kg, y = very_data.height_meters, mode = "markers" , marker = dict( color = "rgba(255,0,0,0.8)"), text = bmi_data.bmi + bmi_data.Name , name = "Very Overweight")



layout = dict(title = 'Body Massive Index ',xaxis= dict(title= 'Weight(kg)',ticklen= 5,zeroline= False),yaxis= dict(title= 'Height(cm)',ticklen= 5,zeroline= False))

data = [trace1,trace2,trace3,trace4]



fig = dict(data = data,layout = layout)

iplot(fig)
potential_list = list()

for j in nationalities : 

    potential_list.append(fdata.Potential[fdata.Nationality == j ].mean())

    

potential_dict = dict(Nationality = nationalities , Potential_mean = potential_list)



potential_data = pd.DataFrame.from_dict(potential_dict)



trace = [go.Choropleth(

            colorscale = 'YlOrRd',

            locationmode = 'country names',

            locations = potential_data['Nationality'],

            text = potential_data['Nationality'],

            z = potential_data['Potential_mean'],

)]



layout = go.Layout(title = 'Country Potential Mean ')



fig = go.Figure(data = trace, layout = layout)

iplot(fig)
fdata.Wage = fdata.Wage.str.replace("€", "")

fdata.Wage = fdata.Wage.str.replace("K", "000")

fdata.Wage = fdata.Wage.astype("float")



fdata["Age_Group"] = ["Under 25" if each < 25 else "25-30" if each < 30 else "30-35" if each < 35 else "Upper 35 " for each in fdata.Age]



fig, ax = plt.subplots()

fig.set_size_inches(15, 15)



wage_plot = sns.violinplot(data = fdata , x = "Age_Group" , y = "Wage" ) 

fig.show()
fdata.Value = fdata.Value.str.replace("€","")

fdata.Value = fdata.Value.str.replace("M","000000")

fdata.Value = fdata.Value.str.replace("K","000")

fdata.Value = fdata.Value.str.replace(".50","5")

fdata.Value = fdata.Value.str.replace(".20","2")

fdata.Value = fdata.Value.astype("float")



fig, ax = plt.subplots()

fig.set_size_inches(15, 15)

value_plot = sns.violinplot(data = fdata , x = "Age_Group" , y = "Value" ) 
import seaborn as sns

sns.set()



fig, ax = plt.subplots()

fig.set_size_inches(12, 12)



cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

ax = sns.scatterplot(x="Stamina", y="ShotPower",

                     size="Age",

                     palette="gnuplot2", sizes=(10,200),

                     data=fdata)
fig, ax = plt.subplots()

fig.set_size_inches(10, 10)

sns.regplot(x = "Composure", y ="Overall" , data = fdata)



fig.show()
forward_new = forward_data.copy()

middle_new = middle_data.copy()

stopper_new = stopper_data.copy()

gk_new = gk_data.copy()

forward_new["Position_Group"] = "Forward"

middle_new["Position_Group"] = "Middle"

stopper_new["Position_Group"] = "Stopper"

gk_new["Position_Group"] = "Goal Keeper"

data_ps = pd.concat([forward_new,middle_new,stopper_new,gk_new])





fig, ax = plt.subplots()

fig.set_size_inches(40, 10)

sns.set(style="whitegrid", palette="muted")

sns.swarmplot(x = "Position_Group", y = "Aggression" ,hue = "preferred_foot" , data = data_ps ) 

fig.show()
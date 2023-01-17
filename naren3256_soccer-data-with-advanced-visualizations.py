import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib.patches import Arc, Arrow , Circle , ConnectionPatch , Rectangle

import matplotlib.pyplot as plt

import json

from pandas.io.json import json_normalize



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Create a normal dataframe from the input json data

with open('../input/france-vs-belgium-soccer-play/France_Belgium.json') as data_file:    

    data = json.load(data_file)

df = json_normalize(data, sep = "_")
# Different events captured from a gameplay

df["type_name"].value_counts()
# eden hazard pass data

EdenHazard_pass = df[(df['type_name'] == "Pass") & (df['player_name']=='Eden Hazard')]

pass_column = [i for i in df.columns if i.startswith("pass")]

EdenHazard_pass = EdenHazard_pass[["id", "period", "timestamp", "location", "pass_end_location", "pass_recipient_name"]]

EdenHazard_pass.head()



#eden hazard shot data

EdenHazard_shot = df[(df['type_name'] == "Shot") & (df['player_name']=='Eden Hazard')]

EdenHazard_shot  = EdenHazard_shot[["id","location","shot_end_location","type_name","shot_technique_name","shot_outcome_name"]]

EdenHazard_shot.head()
#pass end-location and start-location

pe = list(EdenHazard_pass["pass_end_location"])

ps = list(EdenHazard_pass["location"])



#shot end-location and start-location

se = list(EdenHazard_shot["shot_end_location"])

ss = list(EdenHazard_shot["location"])
# method for creating a football court

def CreatePitch(ax):

    

    # boarder line

    OuterLine = Rectangle([0,0],width = 120, height = 80, fill = True, color = 'green')

    LeftPenalty = Rectangle([0,19.85],width =16.5 , height = 40.3, fill = False)

    RightPenalty = Rectangle([103.5,19.85],width =16.5 , height =40.3  ,fill = False)

    MidLine = ConnectionPatch([60,0],[60,80],"data","data")

   

    # Mid Circle

    MidCircle = Circle([60,40],9.15,fill = False)

   

    # 6 Yards area

    LeftSix = Rectangle([0,30.85],width = 5.5,height = 18.3,fill = False)

    RightSix = Rectangle([114.5,30.85],width = 5.5,height = 18.3,fill = False)

   

    # Dot near penalty area

    LDot = Circle([11,40],0.8,fill = True,color = 'white')

    RDot = Circle([109,40],0.8,fill = True,color = 'white')

   

    # Goal post

    Lgoal = Rectangle([0,36.35],width = 2.3 , height = 7.3 ,

                      angle = 360 , fill = True, color = 'white')

    Rgoal = Rectangle([117.5,36.35],width = 2.3 , height = 7.3 ,

                      angle = 360 ,fill = True, color = 'white')

    

    # Arcs

    leftArc = Arc((11,40),height=18,width=20,angle=0,theta1=310,theta2=50,color="black")

    rightArc = Arc((109,40),height=16,width=20,angle=0,theta1=130,theta2=230,color="black")

   

    # mid point

    MidPoint = Circle([60,40], 0.8,fill = True,color = 'white')

    objects = [OuterLine,LeftPenalty,RightPenalty,MidLine,MidCircle,LeftSix,

               RightSix, MidPoint,LDot,RDot,Lgoal,Rgoal,leftArc,rightArc]

    

    for i in objects:

        ax.add_patch(i)

    for i,j in zip(pe,ps):

        ax.annotate("", xy = (i[0],i[1]),xycoords = "data",

                    xytext = (j[0],j[1]),textcoords = 'data',

                   arrowprops=dict(arrowstyle="->",connectionstyle="arc3", color = "black"))

    for i,j in zip(se,ss):

        ax.annotate("",xy = (i[0],i[1]),xytext = (j[0],j[1]),textcoords = 'data',

                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3", color = "red"))

       
fig=plt.figure(figsize = (10,6)) 

ax=fig.add_subplot(1,1,1)

CreatePitch(ax) #overlay our different objects on the pitch

plt.ylim(-10, 90)

plt.xlim(-10, 132)

plt.xlabel("Width of the pitch <---120 meters--->", color = 'black')

plt.ylabel("height of the pitch <---80 meters--->", color = 'black')

plt.title("Eden Hazard's pass(Black arrow) and shots(Red arrow)")

plt.show()
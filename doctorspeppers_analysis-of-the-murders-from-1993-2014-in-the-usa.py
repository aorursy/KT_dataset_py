# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





def add_value_labels(ax, spacing=5, fontsize=20):

    for rect in ax.patches:

        y_value = rect.get_height()

        x_value = rect.get_x() + rect.get_width() / 2

        space = spacing

        va = 'bottom'

        if y_value < 0:

            space *= -1

            va = 'top'

        label = "{:.1f}".format(y_value)

        ax.annotate(

            label,                      

            (x_value, y_value),         

            xytext=(0, space),          

            textcoords="offset points", 

            ha='center',                

            va=va, fontsize=fontsize)                          

    return ax



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

db = pd.read_csv("../input/database.csv")

# Any results you write to the current directory are saved as output.

db.rename(columns={'Record ID':u'Record_ID','Agency Code':u'Agency_Code','Agency Name':u'Agency_Name','Agency Type':u'Agency_Type','Crime Type':u'Crime_Type',

                   'Crime Solved':u'Crime_Solved', 'Victim Sex':u'Victim_sex', 'Victim Age':u'Victim_Age','Victim Ethnicity':u'Victim_Ethnicity','Victim Race':u'Victim_Race',

                   'Perpetrator Age':u'Perpetrator_Age','Perpetrator Sex':'Perpetrator_Sex','Perpetrator Race':u'Perpetrator_Race','Perpetrator Ethnicity':u'Perpetrator_Ethnicity','Victim Count':u'Victim_Count',

                  'Perpetrator Count':u'Perpetrator_Count', 'Record Source':u'Record_Source'},inplace=True)

db = db.drop(["Record_ID","Agency_Code","Agency_Type","Record_Source"], 1)
dic = {"Partner":["Girlfriend", "Wife", "Boyfriend", "Boyfriend/Girlfriend","Common-Law Partner", "Husband","Partner/Partner","Common-Law Husband"],

        "Parents/In-law/StepParents":["Mother", "Father","Stepfather","Stepmother","In-Law"],"Ex-Partner":["Ex-Partner","Ex-Husband ","Ex-Partner"],

        "brothers/Stepbrothers":["Stepson","Sister","Brother"],"children":["Stepdaughter","Daughter","Son"],"Acquaintance":["Acquaintance","Friend","Neighbor"],

      "Coworkers":["Employer","Employee"]}

for a,b in dic.items():

    for z in b:

        db.Relationship = db.Relationship.apply(lambda x: x.replace(z,a))



dic = {"Gun":["Handgun","Firearm","Shotgun","Rifle"]}

for a,b in dic.items():

    for z in b:

        db.Weapon = db.Weapon.apply(lambda x: x.replace(z,a))



victimns_db = db.drop(["Perpetrator_Ethnicity","Perpetrator_Race","Perpetrator_Ethnicity", "Victim_Age", "Perpetrator_Age","Perpetrator_Count"],1)

perpetrator_db = db.drop(["Victim_Ethnicity","Perpetrator_Age","Perpetrator_Ethnicity","Victim_Age"],1)
victimns_db
fig, (ax1, ax2) = plt.subplots(2, 1)



victimns_db.Victim_sex.value_counts().plot(title="murder by sex",y=victimns_db.Victim_sex.value_counts(),kind="pie",figsize=(20,20),autopct='%.2f%%',colormap="tab20c", ax=ax1)

victimns_db.Relationship.value_counts().plot(title="Killer victim relationship (General)",kind="pie",figsize=(20,20),autopct='%.2f%%',colormap="tab20c", ax=ax2)



months = db.Month.value_counts().plot(title="Assassinations nested in months",kind="bar",x=db.State.unique(), y=db.Month.value_counts().mean(), figsize=(20,20))

add_value_labels(months, spacing=5, fontsize=10)
graph = perpetrator_db["Year"].value_counts().plot(title="Number of nested killings per year (1993-2014)",kind="bar",x=perpetrator_db.Year, y=perpetrator_db["Year"].value_counts(), figsize=(20,20))

add_value_labels(graph, spacing=5, fontsize=10)
perpetrator_db.Weapon.value_counts().plot(title="Major weapons used to commit murder",kind="pie", figsize=(15,15))
perpetrator_db[perpetrator_db["Perpetrator_Sex"]=="Male"][perpetrator_db["Victim_sex"]=="Female"].Relationship.value_counts()
fig, (ax1, ax2) = plt.subplots(2, 1)

graph1 = perpetrator_db[perpetrator_db["Perpetrator_Sex"]=="Male"].Victim_sex.value_counts().plot(title="Assassinations in numbers by sex",kind="bar",x=["Male","Female"],

                                                                                                  y=perpetrator_db[perpetrator_db["Perpetrator_Sex"]=="Male"].Victim_sex.value_counts(),

                                                                                                  figsize=(20,20), ax=ax1)

add_value_labels(graph1, spacing=5, fontsize=10)

graph2 = perpetrator_db[perpetrator_db["Perpetrator_Sex"]=="Male"][perpetrator_db["Victim_sex"]=="Female"].Relationship.value_counts().plot(title="Relations that the victim had with the murderer",kind="bar",x=["Male","Female"],

                                                                                                  y=perpetrator_db[perpetrator_db["Perpetrator_Sex"]=="Male"][perpetrator_db["Victim_sex"]=="Female"].Relationship.value_counts(),

                                                                                                  figsize=(20,20), ax=ax2)

add_value_labels(graph2, spacing=5, fontsize=10)





fig, (ax3, ax4) = plt.subplots(2, 1)

victimns_db[victimns_db["Victim_sex"]=="Female"].Relationship.value_counts().plot(title="Relationship between murderer and victim (woman)",kind="pie",figsize=(20,20),autopct='%.2f%%',colormap="tab20c", ax=ax3)

victimns_db[victimns_db["Victim_sex"]=="Male"].Relationship.value_counts().plot(title="Relation between murderer and victim (man)",kind="pie",figsize=(20,20),autopct='%.2f%%',colormap="tab20c", ax=ax4)
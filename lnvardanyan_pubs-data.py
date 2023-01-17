import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from IPython.display import Image as PImage

from subprocess import check_call

from PIL import Image, ImageDraw, ImageFont
df = pd.read_csv('../input/armenian_pubs.csv')
df.shape
df.columns
df.columns = ['Timestamp', 'Age', 'Gender', 'Income', 'Occupation', 'Fav_Pub',

       'WTS', 'Freq', 'Prim_Imp', 'Sec_Imp', 'Stratum', 'Lifestyle', 'Occasions']
max(set(list(df['Fav_Pub'])), key=list(df['Fav_Pub']).count) 
set(df['Occupation'])
df = df[df['Occupation']!='army']

df['Occupation'] = df['Occupation'].replace(['CEO', 'Entrepreneur / Software Engineer', 'Working '], 'Working')

df['Occupation'] = df['Occupation'].astype('category').cat.reorder_categories(['Student', 'Student + working', 'Working']).cat.codes
df.head()
set(df["Freq"])
df["Freq"] = df["Freq"].astype('category').cat.reorder_categories(['rarely (once two week/or a month)', 'Several times in a month', 'Several times a week']).cat.codes
set(df["Gender"])
df['Gender'] = df['Gender'].astype('category').cat.reorder_categories(['Male', 'Female']).cat.codes
max(df['Income'])
df['Prim_Imp'] = df['Prim_Imp'].astype('category').cat.reorder_categories(['Environment', 'Menu', 'Music', 'Pricing']).cat.codes

df['Sec_Imp'] = df['Sec_Imp'].astype('category').cat.reorder_categories(['Environment', 'Menu', 'Music', 'Pricing']).cat.codes

df['Stratum'] = df['Stratum'].astype('category').cat.reorder_categories(['Capital', 'Rural', 'Urban']).cat.codes

df.drop(['Timestamp'], axis=1, inplace=True)

df.head()
df = df.reset_index(drop=True)

df = df[[True if type(df['Occasions'][i])!=float else False for i in range(len(df))]]

df = df.reset_index(drop=True)

df = df[[True if type(df['Fav_Pub'][i])!=float else False for i in range(len(df))]]

df = df.reset_index(drop=True)

df = df[[True if type(df['Lifestyle'][i])!=float else False for i in range(len(df))]]

df = df.reset_index(drop=True)

df = df[[True if df['WTS'][i] >= 0 else False for i in range(len(df))]]

df = df.reset_index(drop=True)

df.head()
set(df['Occasions'])
df['Occasions'] = df['Occasions'].replace(['Nowere','chem aycelum'], 'Never')

df['Occasions'] = df['Occasions'].astype('category').cat.reorder_categories(['Never', 'Birthdays', 'Special events/parties', 'For listening  good music ', 'Hang outs with friends']).cat.codes
sorted(list(set(df['Fav_Pub'])))
df['Fav_Pub'] = df['Fav_Pub'].replace('Bulldog', 'BullDog')

df['Fav_Pub'] = df['Fav_Pub'].replace('DAS ', 'DAS')

df['Fav_Pub'] = df['Fav_Pub'].replace(['I have none', 'I don\'t like pubs'], 'Do not have one')

df['Fav_Pub'] = df['Fav_Pub'].replace('Liberty ', 'Liberty')

df['Fav_Pub'] = df['Fav_Pub'].replace(['Tom Collins ','Tom collins'], 'Tom Collins')

df['Fav_Pub'] = df['Fav_Pub'].replace('Venue ','Venue')

df['Fav_Pub'] = df['Fav_Pub'].replace('37 pub', 'Pub 37')

df['Fav_Pub'] = df['Fav_Pub'].replace('VOID', 'Void')
df.head()
# The 'Lifestile' column is still a little confusing for me, so I'll leave it for now :) 

X = df.loc[:, list(set(df.columns) - set(['Fav_Pub', 'Lifestyle']))] 

y =  df['Fav_Pub']

X = X.fillna(0)
DT = DecisionTreeClassifier(random_state=42)

DT.fit(X, y)
# Export our trained model as a .dot file

with open("tree.dot", 'w') as f:

     f = export_graphviz(DT, out_file=f,

                         feature_names = list(X),

                         impurity = True, rounded = True, filled= True )

        

#Convert .dot to .png to allow display in web notebook

check_call(['dot','-Tpng','tree.dot','-o','tree.png'])



img = Image.open("tree.png")

draw = ImageDraw.Draw(img)

font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)



img.save('sample-out.png')

PImage("sample-out.png")
df.to_excel('Pub_Data_Clean.xlsx')
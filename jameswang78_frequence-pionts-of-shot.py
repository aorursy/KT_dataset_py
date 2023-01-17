import xml.etree.ElementTree as ET
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
%matplotlib inline

def xml2df(xml_data):
    root = ET.XML(xml_data) # element tree
    all_records = []
    for child in root:
        record = {}
        for subchild in child:
            if subchild.text == None:
                subrecord = []
                for grandchild in subchild:
                    subrecord.append(grandchild.text)
                record[subchild.tag] = subrecord
            else:
                record[subchild.tag] = subchild.text
        all_records.append(record)
    df = pd.DataFrame(all_records)
    return df
engine = create_engine('sqlite:///../input/database.sqlite')
con = engine.connect()
rs = con.execute('SELECT shoton,shotoff,goal FROM Match')
df = pd.DataFrame(rs.fetchall())
df.columns = rs.keys()
df.dropna(inplace = True)
df_shoton_coo= pd.DataFrame()
df_shotoff_coo=pd.DataFrame()
df_goal_coo=pd.DataFrame()

for i in df.index:
    if df.shoton[i].find('coordinates')!= -1:
        df_shoton_temp = xml2df(df.shoton[i])
        df_coordinates_temp = pd.DataFrame(df_shoton_temp.coordinates)
        df_shoton_coo = df_shoton_coo.append(df_coordinates_temp,ignore_index=True)
    if df.shotoff[i].find('coordinates')!=-1:
        df_shotoff_temp = xml2df(df.shotoff[i])
        df_coordinates_temp = pd.DataFrame(df_shotoff_temp.coordinates)
        df_shotoff_coo = df_shotoff_coo.append(df_coordinates_temp,ignore_index=True)
    if df.goal[i].find('coordinates')!=-1:
        df_goal_temp = xml2df(df.goal[i])
        df_coordinates_temp = pd.DataFrame(df_goal_temp.coordinates)
        df_goal_coo = df_goal_coo.append(df_coordinates_temp,ignore_index=True)
    
df_shoton_coo.dropna(inplace=True)
df_shotoff_coo.dropna(inplace=True)
df_goal_coo.dropna(inplace=True)
df_shoton_coo['x']=df_shoton_coo.apply(lambda row: int(row.coordinates[0]),axis=1)
df_shoton_coo['y']=df_shoton_coo.apply(lambda row: int(row.coordinates[1]),axis=1)

df_shotoff_coo['x']=df_shotoff_coo.apply(lambda row: int(row.coordinates[0]),axis=1)
df_shotoff_coo['y']=df_shotoff_coo.apply(lambda row: int(row.coordinates[1]),axis=1)

df_goal_coo['x']=df_goal_coo.apply(lambda row: int(row.coordinates[0]),axis=1)
df_goal_coo['y']=df_goal_coo.apply(lambda row: int(row.coordinates[1]),axis=1)

df_shoton_coo['str']=df_shoton_coo.apply(lambda row: ' '.join(map(str, row.coordinates)),axis=1)
df_shotoff_coo['str']=df_shotoff_coo.apply(lambda row: ' '.join(map(str, row.coordinates)),axis=1)
df_goal_coo['str']=df_goal_coo.apply(lambda row: ' '.join(map(str, row.coordinates)),axis=1)
df_shoton_coo['size'] = df_shoton_coo.groupby('str')['str'].transform('count')
df_shotoff_coo['size'] = df_shotoff_coo.groupby('str')['str'].transform('count')
df_goal_coo['size'] = df_goal_coo.groupby('str')['str'].transform('count')
sns.set(rc={'figure.figsize':(10,15)})

ax = sns.scatterplot(x = 'x',y = 'y',data = df_goal_coo,hue ='size',size = 'size')
sns.set(rc={'figure.figsize':(10,15)})

ax = sns.scatterplot(x = 'x',y = 'y',data = df_shoton_coo,hue ='size',size = 'size')
sns.set(rc={'figure.figsize':(10,15)})

ax = sns.scatterplot(x = 'x',y = 'y',data = df_shotoff_coo,hue ='size',size = 'size')

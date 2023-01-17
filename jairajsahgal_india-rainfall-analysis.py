# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input"))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
['indiarain', 'keraladistricts', 'rainfall-in-india']
import matplotlib.pyplot as plt

from PIL import Image

import seaborn as sns

import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from IPython.display import HTML

import matplotlib

matplotlib.rcParams['animation.embed_limit'] = 2**256

import datetime

import random
# img=np.array(Image.open('../input/annualmeanrainfallmapofindia/Annual-mean-rainfall-map-of-India.png'))

# fig=plt.figure(figsize=(10,10))

# plt.imshow(img,interpolation='bilinear')

# plt.axis('off')

# plt.ioff()

# plt.show()
India = pd.read_csv("../input/rainfall-in-india/rainfall in india 1901-2015.csv",sep=",")

India.head()
print('Rows     :',India.shape[0])

print('Columns  :',India.shape[1])

print('\nFeatures :\n     :',India.columns.tolist())

print('\nMissing values    :',India.isnull().values.sum())

print('\nUnique values :  \n',India.nunique())
total = India.isnull().sum().sort_values(ascending=False)

percent = (India.isnull().sum()/India.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()

India.info()
India['JAN'].fillna((India['JAN'].mean()), inplace=True)

India['FEB'].fillna((India['FEB'].mean()), inplace=True)

India['MAR'].fillna((India['MAR'].mean()), inplace=True)

India['APR'].fillna((India['APR'].mean()), inplace=True)

India['MAY'].fillna((India['MAY'].mean()), inplace=True)

India['JUN'].fillna((India['JUN'].mean()), inplace=True)

India['JUL'].fillna((India['JUL'].mean()), inplace=True)

India['AUG'].fillna((India['AUG'].mean()), inplace=True)

India['SEP'].fillna((India['SEP'].mean()), inplace=True)

India['OCT'].fillna((India['OCT'].mean()), inplace=True)

India['NOV'].fillna((India['NOV'].mean()), inplace=True)

India['DEC'].fillna((India['DEC'].mean()), inplace=True)

India['ANNUAL'].fillna((India['ANNUAL'].mean()), inplace=True)

India['Jan-Feb'].fillna((India['Jan-Feb'].mean()), inplace=True)

India['Mar-May'].fillna((India['Mar-May'].mean()), inplace=True)

India['Jun-Sep'].fillna((India['Jun-Sep'].mean()), inplace=True)

India['Oct-Dec'].fillna((India['Oct-Dec'].mean()), inplace=True)

India.describe().T
data=India.copy()

data=data.iloc[:,[1,14]]

Writer = animation.writers['ffmpeg']

writer = Writer(fps=10, metadata=dict(artist='Jairaj'), bitrate=1800)

fig = plt.figure(figsize=(10,6))

def animate(i):

    p = sns.lineplot(x=data.iloc[:int(i+1),0], y=data.iloc[:int(i+1),1], data=data, color="r")

    p.tick_params(labelsize=17)

    plt.setp(p.lines,linewidth=1)

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(data["YEAR"].unique()), repeat=True)

# ani.save('HeroinOverdosesJumpy.mp4', writer=writer)

HTML(ani.to_jshtml())
df=India.copy()

df.columns
df=df.iloc[:,[0,1,14]]



df["SUBDIVISION"].value_counts()

df=df[~df['SUBDIVISION'].isin(["ARUNACHAL PRADESH","LAKSHADWEEP","ANDAMAN & NICOBAR ISLANDS"])]

df["SUBDIVISION"].value_counts()

df.columns
def makecode(i):

    s=""

    for j in i.split(" "):

        s=s+j[0]

    return s



        

l=[]

for i in df["SUBDIVISION"]:

    l.append(makecode(i))

print(len(np.unique(l)))

print(len(df["SUBDIVISION"].unique()))

df["CODE"]=l

# print(type(df["CODE"]))


# df.sort_values(by="Date",ascending=False).head(10)

dff=(df[df['YEAR'].eq(df["YEAR"].max())].sort_values(by="ANNUAL",ascending=False).head(10))

dff=dff.reindex(index=dff.index[::-1])

fig,ax=plt.subplots(figsize=(15,8))

ax.barh(dff['SUBDIVISION'],dff['ANNUAL'])





list1=df["CODE"].unique().tolist()

s=['#ffffcc', '#fff2c8', '#ffdfc8', '#fdefff', '#ffff99', '#ffeda2', '#ffdaad', '#ffeaf9', '#ffff66', '#ffea86', '#ffd79d', '#ffe6f5', '#ffff33', '#ffe975', '#ffd594', '#ffe5f3', '#ffff00', '#ffe871', '#ffd592', '#ffe4f2', '#ffccff', '#cfd7ff', '#e1d8fd', '#fbd1e1', '#ffcccc', '#ded8d2', '#f1d2cb', '#ffcad8', '#ffcc99', '#e5d69d', '#fccd99', '#ffc4d1', '#ffcc66', '#e9d469', '#ffca6f', '#ffc0cd', '#ffcc33', '#ecd435', '#ffc857', '#ffbfca', '#ffcc00', '#ecd30f', '#ffc750', '#ffbeca', '#ff99ff', '#aabdff', '#b0bcf9', '#f6a9b5', '#ff99cc', '#b1b8e0', '#c5b5c7', '#fc9faa', '#ff9999', '#bdb6a8', '#d2b095', '#ff99a2', '#ff9966', '#c4b470', '#daac62', '#ff959e', '#ff9933', '#c7b43a', '#deab2a', '#ff949d', '#ff9900', '#c8b317', '#dfaa00', '#ff949c', '#ff66ff', '#98b2ff', '#85a7f5', '#f28791', '#ff66cc', '#82a0f6', '#a09fc3', '#f87981', '#ff6699', '#999db9', '#b19992', '#fd6e74', '#ff6666', '#a59b7c', '#bb955e', '#ff666c', '#ff6633', '#aa9a42', '#bf9322', '#ff656a', '#ff6600', '#ac9a1e', '#c09300', '#ff6569', '#ff33ff', '#96b1ff', '#679bf2', '#f07178', '#ff33cc', '#779dff', '#8a92c1', '#f75e63', '#ff3399', '#7a8ece', '#9d8b8f', '#fb4c4f', '#ff3366', '#8f8c8b', '#a7875c', '#fe3d3e', '#ff3333', '#988b4a', '#ad841c', '#ff3332', '#ff3300', '#9a8b23', '#ae8600', '#ff3331', '#ff00ff', '#96b1ff', '#5e98f1', '#f06a71', '#ff00cc', '#7ba0ff', '#838fc0', '#f6555a', '#ff0099', '#6e89d7', '#97888e', '#fa4042', '#ff0066', '#888892', '#a2835b', '#fd2b28', '#ff0033', '#93874e', '#a8801a', '#fe1a00', '#ff0000', '#968726', '#a98200', '#fe1c00', '#ccffff', '#f8f4f8', '#ffeafd', '#cbefff', '#ccffcc', '#fff1c5', '#ffdecc', '#d3efff', '#ccff99', '#ffeb97', '#ffd8ab', '#d9eeff', '#ccff66', '#ffe873', '#ffd497', '#ddeeff', '#ccff33', '#ffe75c', '#ffd38c', '#e0eeff', '#ccff00', '#ffe655', '#ffd389', '#e0eeff', '#ccccff', '#c4ceff', '#cbccff', '#c6d1e1', '#cccccc', '#cfcbcb', '#dec6cd', '#cecad9', '#cccc99', '#d7c997', '#eac19b', '#d3c4d3', '#cccc66', '#dbc764', '#f1be6a', '#d7c1cf', '#cccc33', '#ddc631', '#f5bc3b', '#d8bfcd', '#cccc00', '#ddc600', '#f6c600', '#d9becc', '#cc99ff', '#98b1ff', '#8faefb', '#bfa9b6', '#cc99cc', '#9ea8d7', '#aaa7c9', '#c79fab', '#cc9999', '#aaa5a0', '#baa198', '#cd98a2', '#cc9966', '#b1a46a', '#c49d65', '#d0929d', '#cc9933', '#b5a336', '#c99b32', '#d2909a', '#cc9900', '#b5a20e', '#ca9a00', '#d38f99', '#cc66ff', '#87a7ff', '#4c97f6', '#bb8791', '#cc66cc', '#648ceb', '#7b8dc5', '#c37982', '#cc6699', '#7f88af', '#918694', '#c96e75', '#cc6666', '#8c8675', '#9e8161', '#cc656c', '#cc6633', '#92853c', '#a47e2b', '#ce6067', '#cc6600', '#948415', '#a67f00', '#cf5f65', '#cc33ff', '#8aaaff', '#008def', '#b87178', '#cc33cc', '#4387ff', '#577ec2', '#c15e64', '#cc3399', '#5275c8', '#767591', '#c64c50', '#cc3366', '#707387', '#856f5f', '#ca3d3f', '#cc3333', '#7b7146', '#8c6c27', '#cc3334', '#cc3300', '#7e711b', '#8f6d00', '#cc3030', '#cc00ff', '#8cabff', '#008cec', '#b76a71', '#cc00cc', '#5d91ff', '#4b7ac0', '#c0555a', '#cc0099', '#356fd5', '#6e7190', '#c64043', '#cc0066', '#646d90', '#7e6a5e', '#ca2b2b', '#cc0033', '#746c4c', '#856726', '#cb170b', '#cc0000', '#786c1e', '#886900', '#cc1600', '#99ffff', '#efecf4', '#f4e4ff', '#a6efff', '#99ffcc', '#f7e9c1', '#ffddd0', '#acefff', '#99ff99', '#fbe790', '#ffd6a9', '#b0eeff', '#99ff66', '#fee65e', '#ffd291', '#b4eeff', '#99ff33', '#ffe532', '#ffd184', '#b5edff', '#99ff00', '#ffe41c', '#ffd080', '#b6edff', '#99ccff', '#b9c5fa', '#b9c4ff', '#91d1e1', '#99cccc', '#c4c1c6', '#cebdcf', '#9ccad9', '#99cc99', '#cbbf93', '#dbb89d', '#a3c4d3', '#99cc66', '#cfbd61', '#e4b56c', '#a8c1cf', '#99cc33', '#d1bc2f', '#e8b33f', '#aabfcd', '#99cc00', '#d2bc00', '#e9b22a', '#abbecd', '#9999ff', '#83a4ff', '#6fa4fd', '#86a9b6', '#9999cc', '#8f9cce', '#929bcc', '#929fab', '#999999', '#9b9899', '#a6949a', '#9a97a3', '#999966', '#a29665', '#b29068', '#9f929d', '#999933', '#a59532', '#b78e37', '#a2909a', '#999900', '#a69500', '#a69500', '#a28f99', '#9966ff', '#709bff', '#0090f3', '#7e8791', '#9966cc', '#497bdf', '#507fc7', '#8c7982', '#996699', '#6a77a5', '#737696', '#946d75', '#996666', '#78746d', '#847064', '#9a656c', '#996633', '#7e7237', '#8c6d31', '#9c6067', '#996600', '#7f720d', '#8e6c00', '#9d5f66', '#9933ff', '#7da2ff', '#008de8', '#7a7078', '#9933cc', '#007af6', '#0073c2', '#885d64', '#993399', '#005ebf', '#496192', '#914c51', '#993366', '#505b80', '#625961', '#963d40', '#993333', '#5f5941', '#6c552d', '#993335', '#993300', '#635913', '#705600', '#9a3032', '#9900ff', '#83a6ff', '#008ce5', '#796a71', '#9900cc', '#2782ff', '#0073c0', '#87555b', '#990099', '#0067d0', '#385b90', '#904044', '#990066', '#39538f', '#57535f', '#962b2c', '#990033', '#55514a', '#634e2c', '#981612', '#990000', '#5a5117', '#674f00', '#991100', '#66ffff', '#eae7f0', '#ebdfff', '#88efff', '#66ffcc', '#f1e4bf', '#ffddd3', '#8befff', '#66ff99', '#f6e28d', '#ffd5a7', '#8eeeff', '#66ff66', '#f8e15d', '#ffd08b', '#90eeff', '#66ff33', '#fae02a', '#ffcf7c', '#92edff', '#66ff00', '#fae000', '#ffce79', '#92edff', '#66ccff', '#b2bef5', '#adbeff', '#57d1e1', '#66cccc', '#bdbbc1', '#c3b7d1', '#6bcad9', '#66cc99', '#c4b88f', '#d2b29f', '#77c4d3', '#66cc66', '#c8b65e', '#daae6e', '#7ec1cf', '#66cc33', '#cab52d', '#dfac42', '#81bfcd', '#66cc00', '#cab500', '#e0ac2e', '#82becd', '#6699ff', '#6e98fe', '#519cfe', '#3ea9b6', '#6699cc', '#8593c7', '#8093cd', '#5a9fab', '#669999', '#918f93', '#978c9c', '#6997a3', '#669966', '#988c61', '#a4876a', '#71929d', '#669933', '#9b8b2f', '#aa853b', '#74909a', '#669900', '#9b8b00', '#ac8421', '#758f9a', '#6666ff', '#518dff', '#0090ee', '#228791', '#6666cc', '#326ed5', '#0076c9', '#4d7982', '#666699', '#59699c', '#586a98', '#5e6d75', '#666666', '#686666', '#6f6367', '#67656c', '#666633', '#6d6432', '#795f35', '#6b6067', '#666600', '#6f6300', '#7b5e11', '#6c5f66', '#6633ff', '#6e9aff', '#008ce4', '#007179', '#6633cc', '#0074ea', '#0076c2', '#445d64', '#663399', '#0059b4', '#005995', '#584c51', '#663366', '#324676', '#3d4763', '#623d41', '#663333', '#46433a', '#4f4031', '#663336', '#663300', '#4a420b', '#534000', '#673033', '#6600ff', '#7aa1ff', '#008be1', '#006b73', '#6600cc', '#007cf8', '#0076bf', '#42545b', '#660099', '#0065ca', '#005a94', '#563f44', '#660066', '#00478e', '#253d60', '#602b2d', '#660033', '#323748', '#3f352f', '#651615', '#660000', '#3c360f', '#453500', '#660b00', '#33ffff', '#e7e5ef', '#e7ddff', '#74efff', '#33ffcc', '#eee2bd', '#fcdbd4', '#76efff', '#33ff99', '#f3df8c', '#ffd4a6', '#78eeff', '#33ff66', '#f5de5c', '#ffcf88', '#79eeff', '#33ff33', '#f7dd29', '#ffcd78', '#7aedff', '#33ff00', '#f7dd00', '#ffcd74', '#7aedff', '#33ccff', '#aebbf2', '#a7bbff', '#00d1e0', '#33cccc', '#b9b7bf', '#bdb4d1', '#3ecad9', '#33cc99', '#c0b48e', '#ccaea0', '#54c4d3', '#33cc66', '#c4b25d', '#d6ab6f', '#5ec1cf', '#33cc33', '#c6b22c', '#daa943', '#63bfcd', '#33cc00', '#c6b100', '#dba830', '#64bfd7', '#3399ff', '#6993fa', '#3999ff', '#00abb7', '#3399cc', '#7f8ec3', '#758fce', '#00a0ac', '#339999', '#8c8a90', '#8f879d', '#3997a3', '#339966', '#92875e', '#9d826b', '#49929d', '#339933', '#95862e', '#a4803c', '#4f909a', '#339900', '#958600', '#a58600', '#518f9a', '#3366ff', '#2581ff', '#0090ec', '#009099', '#3366cc', '#2067cd', '#0078c9', '#007f87', '#336699', '#506195', '#43639a', '#146d76', '#336666', '#5e5d61', '#615b68', '#35656d', '#336633', '#645b2f', '#6d5737', '#3f6068', '#336600', '#655b00', '#705617', '#415f66', '#3333ff', '#6094ff', '#008ce2', '#008187', '#3333cc', '#0070e1', '#0078c2', '#006a70', '#333399', '#0054aa', '#005d99', '#005056', '#333366', '#19376a', '#003b65', '#263c41', '#333333', '#343333', '#373133', '#333236', '#333300', '#373200', '#3d2f09', '#363033', '#3300ff', '#739dff', '#008bdf', '#007c82', '#3300cc', '#0079f2', '#0076bf', '#006469', '#330099', '#0062c4', '#005d96', '#00474b', '#330066', '#00468b', '#003f67', '#212a2d', '#330033', '#002448', '#131e30', '#301517', '#330000', '#1e1b08', '#221a00', '#330600', '#00ffff', '#e6e4ee', '#e6dcff', '#6eefff', '#00ffcc', '#ede1bd', '#fbdad4', '#70efff', '#00ff99', '#f2df8c', '#ffd3a6', '#72eeff', '#00ff66', '#f5dd5c', '#ffcf87', '#73eeff', '#00ff33', '#f6dd29', '#ffcd77', '#73edff', '#00ff00', '#f6dc00', '#ffcd72', '#73edff', '#00ccff', '#adbaf2', '#a5bbff', '#00d0df', '#00cccc', '#b8b6bf', '#bbb3d1', '#29cad9', '#00cc99', '#bfb38d', '#cbaea0', '#47c4d3', '#00cc66', '#c3b25d', '#d4aa6f', '#54c1cf', '#00cc33', '#c5b12b', '#d9a844', '#59bfcd', '#00cc00', '#c5b000', '#daa831', '#5abfcd', '#0099ff', '#6792f8', '#3398ff', '#00acb7', '#0099cc', '#7e8dc2', '#728ecf', '#00a1ac', '#009999', '#8a898f', '#8d869d', '#1f97a3', '#009966', '#90865e', '#9b816c', '#39929d', '#009933', '#93852d', '#a27e3d', '#42909a', '#009900', '#948500', '#a37e25', '#448f9a', '#0066ff', '#007efe', '#0090ec', '#00929a', '#0066cc', '#1a66cc', '#0078c9', '#00828a', '#006699', '#4e5f93', '#3d629a', '#00727a', '#006666', '#5c5b5f', '#5e5a69', '#15656d', '#006633', '#61592e', '#6a5538', '#2a6068', '#006600', '#635800', '#6d5418', '#2d5f66', '#0033ff', '#5b91ff', '#008ce2', '#008389', '#0033cc', '#006fde', '#0078c2', '#006f74', '#003399', '#0053a6', '#005e9a', '#00595e', '#003366', '#0d3366', '#003e68', '#004347', '#003333', '#2e2e30', '#2f2d34', '#0a3236', '#003300', '#312c00', '#362a0c', '#173033', '#0000ff', '#719cff', '#008bdf', '#007f85', '#0000cc', '#0078f0', '#0076be', '#006a6e', '#000099', '#0060c1', '#005e97', '#005155', '#000066', '#004487', '#004168', '#003739', '#000033', '#002346', '#002135', '#001c1d']

random.shuffle(s)

colour_blind=s[:len(list1)]

# colour_blind=['#cd98a2', '#7aa1ff','#E5BE01', '#ffddd3', '#33ff66', '#8f8c8b', '#796a71', '#6600cc', '#847064', '#663399', '#a7875c', '#00cc66', '#003399', '#cfcbcb', '#b87178', '#cc3300', '#ffc857', '#aaa5a0', '#009099', '#686666', '#615b68', '#ffcd78', '#ffe75c', '#326ed5', '#666699', '#005e9a', '#0079f2', '#666633', '#719cff', '#ff9900', '#82becd', '#90eeff', '#ecd435', '#356fd5', '#645b2f', '#99ff99', '#d3c4d3', '#86a9b6', '#006f74', '#a27e3d', '#ff66cc', '#005ebf', '#fa4042', '#767591', '#c99b32', '#99ffcc', '#5e98f1', '#82a0f6', '#ecd30f', '#669999', '#57d1e1', '#cc6666', '#a59b7c', '#ede1bd', '#a98200', '#336600', '#ff3332', '#c8b65e', '#cccc99', '#330000', '#838fc0', '#7e8791', '#a6efff', '#b76a71', '#cc00ff', '#ffdecc', '#cccccc', '#3333ff', '#ad841c', '#ffff99', '#74909a', '#76efff', '#b19992', '#ff666c', '#bfa9b6', '#8093cd', '#ff3366', '#009933', '#ffffcc', '#8c8a90', '#a5bbff', '#cc0099', '#ffea86', '#ff00cc', '#91d1e1', '#b9b7bf', '#cab500', '#99cc99', '#33cc99', '#669933', '#0066cc', '#373133', '#59699c', '#856726', '#fc9faa', '#563f44', '#ffbeca', '#9d5f66', '#005d96', '#c4ceff', '#ccff99', '#dec6cd', '#008de8', '#6f6300', '#0070e1', '#bb955e', '#0066ff', '#1e1b08', '#aebbf2', '#55514a', '#666666', '#4a420b', '#ff6699', '#00cc99', '#635800', '#8f6d00', '#660033', '#003f67', '#6666cc', '#996666', '#330099', '#8a92c1', '#d7c1cf', '#3366ff', '#00ffcc', '#9d8b8f', '#e0ac2e', '#674f00', '#0074ea', '#993366', '#00595e', '#c4c1c6', '#fee65e', '#fccd99', '#8befff', '#42545b', '#005995', '#cc0033', '#99ccff', '#856f5f', '#a8801a', '#586a98', '#9a656c', '#33ffff', '#993300', '#660099', '#705600', '#918f93', '#f7e9c1', '#ccffff', '#baa198', '#963d40', '#ff3300', '#ff949d', '#ffd080', '#cccc33', '#ffff33', '#47c4d3', '#7e6a5e', '#bdbbc1', '#b9c4ff', '#cc66cc', '#0090ee', '#9a8b23', '#d2b29f', '#66cc66', '#7f8ec3', '#b4eeff', '#cc00cc', '#cf5f65', '#00d0df', '#9d826b', '#74efff', '#663336', '#b1b8e0', '#999900', '#c5b000', '#004347', '#ccff66', '#363033', '#6666ff', '#66cc00', '#19376a', '#339966', '#83a4ff', '#ffca6f', '#385b90', '#ccff33', '#cc33ff', '#cc3399', '#cecad9', '#46433a', '#8cabff', '#54c1cf', '#73eeff', '#c49d65', '#663366', '#cc9999', '#0059b4', '#415f66', '#fbe790', '#1f97a3', '#78746d', '#9b8b2f', '#6c552d', '#a3c4d3', '#bbb3d1', '#a4876a', '#ff0066', '#cc3366']

print(len(list1))

print(len(colour_blind))

df["CODE"]=l

colors=dict(zip(list1,colour_blind))

group_lk=df.set_index('SUBDIVISION')['CODE'].to_dict()

# print(len(group_lk)?)



fig,ax=plt.subplots(figsize=(15,8))

dff=dff[::-1] #flip values from to bottom

#pass colors values to color=

ax.barh(dff['SUBDIVISION'],dff['ANNUAL'],color=[colors[group_lk[x]] for x in dff['SUBDIVISION']])

for i,(value,name) in enumerate(zip(dff['ANNUAL'],dff['SUBDIVISION'])):

#   ax.text(value,i,name,ha='right') #tokyo: name

  ax.text(value,i,group_lk[name],ha='right') #asia : group name

  ax.text(value,i,value,ha='left') #28194.2

  #Add year right middle portion of canvas

ax.text(1,0.4,df["YEAR"].max(),transform=ax.transAxes,size=46,ha='right')
fig,ax=plt.subplots(figsize=(15,8))

def draw_barchart(year):

  dff=df[df['YEAR'].eq(year)].sort_values(by='ANNUAL',ascending=True).tail(10)

  ax.clear()

  ax.barh(dff['SUBDIVISION'],dff['ANNUAL'],color=[colors[group_lk[x]] for x in dff['SUBDIVISION']])

  dx=dff['ANNUAL'].max()/200

  for i,(value,name) in enumerate(zip(dff['ANNUAL'],dff['SUBDIVISION'])):

    ax.text(value-dx,i,name,size=14,weight=600,ha='right',va='bottom')

    ax.text(value-dx,i-.25,group_lk[name],size=10,color='#444444',ha='right',va='baseline')

    ax.text(value+dx,i,f'{value:,.0f}',size=14,ha='left',va='center')

    #...polished styles

    ax.text(1,0.4,year,transform=ax.transAxes,color='#777777',size=46,ha='right',weight=800)

    ax.text(0,1.06,'ANNUAL',transform=ax.transAxes,size=12,color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x',colors='#777777',labelsize=12)

    ax.set_yticks([])

    ax.margins(0,0.01)

    ax.grid(which='major',axis='x',linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0,1.12,'Annual rainfall in India',transform=ax.transAxes,size=24,weight=600,ha='left')

    ax.text(1,0,'made by Jairaj Sahgal',transform=ax.transAxes,ha='right',color='#777777',bbox=dict(facecolor='white',alpha=0.8,edgecolor='white'))

    plt.box(False)

draw_barchart(df["YEAR"].max())
fig, ax=plt.subplots(figsize=(15,8))

animator=animation.FuncAnimation(fig,draw_barchart,frames=np.sort(df["YEAR"].unique()))

HTML(animator.to_jshtml())
ax=India.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(600,2200),color='b',marker='o',linestyle='-',linewidth=2,figsize=(12,8));

India['MA10'] = India.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()

India.MA10.plot(color='r',linewidth=4)

plt.xlabel('Year',fontsize=20)

plt.ylabel('Annual Rainfall (in mm)',fontsize=20)

plt.title('Annual Rainfall in India from Year 1901 to 2015',fontsize=25)

ax.tick_params(labelsize=15)

plt.grid()

plt.ioff()
India[['YEAR','Jan-Feb', 'Mar-May',

       'Jun-Sep', 'Oct-Dec']].groupby("YEAR").mean().plot(figsize=(13,8));

plt.xlabel('Year',fontsize=20)

plt.ylabel('Seasonal Rainfall (in mm)',fontsize=20)

plt.title('Seasonal Rainfall from Year 1901 to 2015',fontsize=25)

ax.tick_params(labelsize=15)

plt.grid()

plt.ioff()
India[['SUBDIVISION', 'Jan-Feb', 'Mar-May',

       'Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").mean().sort_values('Jun-Sep').plot.bar(width=0.5,edgecolor='k',align='center',stacked=True,figsize=(16,8));

plt.xlabel('Subdivision',fontsize=30)

plt.ylabel('Rainfall (in mm)',fontsize=20)

plt.title('Rainfall in Subdivisions of India',fontsize=25)

ax.tick_params(labelsize=15)

plt.grid()

plt.ioff()
drop_col = ['ANNUAL','Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']



fig, ax = plt.subplots()



(India.groupby(by='YEAR')

 .mean()

 .drop(drop_col, axis=1)

 .T

 .plot(alpha=0.1, figsize=(12, 6), legend=False, fontsize=12, ax=ax)

)

ax.set_xlabel('Months', fontsize=12)

ax.set_ylabel('Rainfall (in mm)', fontsize=12)

plt.grid()

plt.ioff()
plt.rcParams['figure.figsize']=(23,10)

ax = sns.boxplot(x="SUBDIVISION", y="ANNUAL", data=India,width=0.8,linewidth=3)

ax.set_xlabel('Subdivision',fontsize=30)

ax.set_ylabel('Annual Rainfall (in mm)',fontsize=30)

plt.title('Annual Rainfall in Subdivisions of India',fontsize=40)

ax.tick_params(axis='x',labelsize=20,rotation=90)

ax.tick_params(axis='y',labelsize=20,rotation=0)

plt.grid()

plt.ioff()
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

x=India.groupby('SUBDIVISION').mean()

print(x.mean())

# x.mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'].plot('bar', color='b',width=0.65,linewidth=3,edgecolor='k',align='center',title='Subdivision wise Average Annual Rainfall', fontsize=20)

# plt.xticks(rotation = 90)

# plt.ylabel('Average Annual Rainfall (in mm)')

# ax.title.set_fontsize(30)

# ax.xaxis.label.set_fontsize(20)

# ax.yaxis.label.set_fontsize(20)

#print(India.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[0,1,2]])

#print(India.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'][[33,34,35]])
ax=India[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2,figsize=(16,8))

plt.xlabel('Month',fontsize=30)

plt.ylabel('Monthly Rainfall (in mm)',fontsize=20)

plt.title('Monthly Rainfall in India',fontsize=25)

ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()
India[['AUG']].mean()
#India1=India['JAN','FEB','ANNUAL']

fig=plt.gcf()

fig.set_size_inches(15,15)

fig=sns.heatmap(India.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
# img=np.array(Image.open('../input/kerela-district-map/Kerala-district-Map.png'))

# fig=plt.figure(figsize=(10,10))

# plt.imshow(img,interpolation='bilinear')

# plt.axis('off')

# plt.ioff()

# plt.show()
Kerala =India[India.SUBDIVISION == 'KERALA']

#Kerala
ax=Kerala.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(1000,5000),color='b',marker='o',linestyle='-',linewidth=2,figsize=(12,8));

#Kerala['MA10'] = Kerala.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()

#Kerala.MA10.plot(color='r',linewidth=4)

plt.xlabel('Year',fontsize=20)

plt.ylabel('Kerala Annual Rainfall (in mm)',fontsize=20)

plt.title('Kerala Annual Rainfall from Year 1901 to 2015',fontsize=25)

ax.tick_params(labelsize=15)

plt.grid()

plt.ioff()
print('Average annual rainfall received by Kerala=',int(Kerala['ANNUAL'].mean()),'mm')
print('Kerala received 4257.8 mm of rain in the year 1961')

a=Kerala[Kerala['YEAR']==1961]

a
print('Kerala received 4226.4 mm of rain in the year 1924')

b=Kerala[Kerala['YEAR']==1924]

b
Dist = pd.read_csv("../input/rainfall-in-india/district wise rainfall normal.csv",sep=",")
Dist.head()
KDist=Dist[Dist.STATE_UT_NAME == 'KERALA']

k=KDist.sort_values(by=['ANNUAL'])

ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

plt.xlabel('District',fontsize=30)

plt.ylabel('Annual Rainfall (in mm)',fontsize=20)

plt.title('Rainfall in Districts of Kerala',fontsize=25)

ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()
Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().head(5)
ax=Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

#ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

plt.xlabel('District',fontsize=30)

plt.ylabel('Annual Rainfall (in mm)',fontsize=20)

plt.title('Districts with Minumum Rainfall in India',fontsize=25)

ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()
Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().tail(5)
ax=Dist.groupby(['DISTRICT'])['ANNUAL'].max().sort_values().tail(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

#ax=k.plot.bar(x='DISTRICT',y='ANNUAL',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

plt.xlabel('District',fontsize=30)

plt.ylabel('Annual Rainfall (in mm)',fontsize=20)

plt.title('Districts with Maximum Rainfall in India',fontsize=25)

ax.tick_params(labelsize=20)

plt.grid()

plt.ioff()
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

m=Basemap(projection='mill',llcrnrlat=0,urcrnrlat=40,llcrnrlon=50,urcrnrlon=100,resolution='c')

m.drawcoastlines()

m.drawcountries()

m.drawstates()

#m.fillcontinents()

m.fillcontinents(color='coral',lake_color='aqua')

#m.drawmapboundary()

m.drawmapboundary(fill_color='aqua')

#m.bluemarble()

#x, y = m(25.989836, 79.450035)

#plt.plot(x, y, 'go', markersize=5)

#plt.text(x, y, ' Trivandrum', fontsize=12);

lat,lon=13.340881,74.742142

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, ' Udupi (4306mm)', fontsize=12);

lat,lon=28.879720,94.796970

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, ' Upper Siang(4402mm)', fontsize=12);

"""lat,lon=25.578773,91.893257

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, 'East Kashi Hills (6166mm)', fontsize=12);

lat,lon=25.389820,92.394913

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, 'Jaintia Hills (6379mm)', fontsize=10);"""

lat,lon=24.987934,93.495293

x,y=m(lon,lat)

m.plot(x,y,'go')

plt.text(x, y, 'Tamenglong (7229mm)', fontsize=12);

lat,lon=34.136389,77.604139

x,y=m(lon,lat)

m.plot(x,y,'ro')

plt.text(x, y, ' Ladakh(94mm)', fontsize=12);

"""lat,lon=25.759859,71.382439

x,y=m(lon,lat)

m.plot(x,y,'ro')

plt.text(x, y, ' Barmer(268mm)', fontsize=12);"""

lat,lon=26.915749,70.908340

x,y=m(lon,lat)

m.plot(x,y,'ro')

plt.text(x, y, ' Jaisalmer(181mm)', fontsize=12);

plt.title('Places with Heavy and Scanty Rainfall in India',fontsize=20)

plt.ioff()

plt.show()
India.groupby("YEAR").mean()['ANNUAL'].sort_values(ascending=False).head(10)
India.groupby("YEAR").mean()['ANNUAL'].sort_values(ascending=False).tail(10)
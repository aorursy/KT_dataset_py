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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
!pip install chart_studio
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 
color = sns.color_palette()
import chart_studio.plotly as py
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyo
pyo.init_notebook_mode()
from plotly.subplots import make_subplots
df = pd.read_csv('/kaggle/input/packaging-trends-survey-dataset/Packaging Survey .csv')
df.head()
df.shape
df.describe()
df.drop(columns=['Timestamp'], inplace=True)
df.head()
df['Occupation'].value_counts()
df.rename(columns={'Choose the most relevant or crucial factor that influences your preference for any packaging material?':'Crucial factor'},inplace=True)
df.rename(columns={'Which is the most preferred packaging for your product? [Fresh fruits and vegetables]':'Packaging for fruits and vegetables','Which is the most preferred packaging for your product? [Preserved food]':'Packaging for preserved food', 'Which is the most preferred packaging for your product? [Cosmetics]':'Packaging for cosmetics','Which is the most preferred packaging for your product? [Pharmaceuticals]':'Packaging for Pharmaceuticals','Which is the most preferred packaging for your product? [Clothes]':'Packaging for clothes','Which is the most preferred packaging for your product? [Accessories ]':'Packaging for accessories','Which is the most preferred packaging for your product? [E-Goods]':'Packaging for E-Goods','Which is the most preferred packaging for your product? [Others]':'Packaging for other goods'},inplace=True)
df.head()
df['Crucial factor'].value_counts()
print('The most crucial factor that influences the customers preference for packaging material:',df['Crucial factor'].value_counts().index[0])
print('The factor that people least care about in considering packaging material:', df['Crucial factor'].value_counts().index[-1])
df['Crucial factor'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('Crucial factor',data=df,palette='inferno',order=df['Crucial factor'].value_counts().index)
plt.xticks(rotation=90)
plt.xlabel('Factors')
plt.title('Factors influencing choice of packaging material')
plt.show()
df.rename(columns={'Does difficulty in opening a package negatively impact your chances of shopping again?':'Effect of difficulty in opening packaging'}, inplace=True)
df['Effect of difficulty in opening packaging'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()


df['Effect of difficulty in opening packaging'].value_counts()
plt.subplots(figsize=(15,4))
sns.countplot('Packaging for preserved food',data=df,palette='inferno',order=df['Packaging for preserved food'].value_counts().index)
plt.xticks(rotation=0)
plt.xlabel('Packaging options')
plt.title('Preferred Packaging for preserved food')
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('Packaging for fruits and vegetables',data=df,palette='inferno',order=df['Packaging for fruits and vegetables'].value_counts().index)
plt.xticks(rotation=0)
plt.xlabel('Packaging options')
plt.title('Preferred Packaging for fruits and vegetables')
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('Packaging for cosmetics',data=df,palette='inferno',order=df['Packaging for cosmetics'].value_counts().index)
plt.xticks(rotation=0)
plt.xlabel('Packaging options')
plt.title('Preferred Packaging for cosmetics')
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('Packaging for Pharmaceuticals',data=df,palette='inferno',order=df['Packaging for Pharmaceuticals'].value_counts().index)
plt.xticks(rotation=0)
plt.xlabel('Packaging options')
plt.title('Preferred Packaging for Pharmaceuticals')
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('Packaging for accessories',data=df,palette='inferno',order=df['Packaging for accessories'].value_counts().index)
plt.xticks(rotation=0)
plt.xlabel('Packaging options')
plt.title('Preferred Packaging for accessories')
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('Packaging for clothes',data=df,palette='inferno',order=df['Packaging for clothes'].value_counts().index)
plt.xticks(rotation=0)
plt.xlabel('Packaging options')
plt.title('Preferred Packaging for clothes')
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('Packaging for E-Goods',data=df,palette='inferno',order=df['Packaging for E-Goods'].value_counts().index)
plt.xticks(rotation=0)
plt.xlabel('Packaging options')
plt.title('Preferred Packaging for E-Goods')
plt.show()
plt.subplots(figsize=(15,4))
sns.countplot('Packaging for other goods',data=df,palette='inferno',order=df['Packaging for other goods'].value_counts().index)
plt.xticks(rotation=0)
plt.xlabel('Packaging options')
plt.title('Preferred Packaging for other goods')
plt.show()
df.rename(columns={'Which traditional packaging will you prefer the most?':'Preferred traditional packaging','Is moving towards smaller package sizes more preferable?':'Smaller packaging','Do you prefer fruits and vegetables to be prepacked in net bags of certain weights?':'Prepacked packed fruits and vegetables','Do you prefer aseptic packaging technology for juices, milk, and milk products that provide six layers of total protection? (Tetra packing)':'Aseptic/Tetra packing','Do you support and consume products with modified atmosphere packaging? (Nitrogen flushing - to prevent rancidity) Ex: Kurkure ':'Support modified atmosphere packaging','Do you prefer tamper evident packaging for items like shampoo, talcum powder, other cosmetics and packed food items?':'Prefer tamper evident packaging','Are you satisfied with the size and location of the vegetarian and non vegetarian mark on products?':'Satisfied with the veg/non-veg mark','Do you buy your toothpaste/cosmetic creams based on the colour code stripe at the bottom? ':'Colour code stripe on toothpastes/creams','Do you prefer changing the product if another company provides a more attractive/appealing and substantial packaging?':'Change company for more appealing packaging?','Do you expect the product image to be exactly the same on the product and package?':'Mirroring in product and product image','Do you want packaging to have clear and easily visible information on usage, consumption, side effects, potential dangers, etc? ':'Is information on packaging enough and visible?','Which colors are you more attracted to in packaging? ':'Preferred colour in packaging','Would you rather buy?':'Quality-Packaging preference',' What will you prefer to buy? Taking X product of a high quality, popular brand as an example. Supposedly, it costs Rs.90/each. On an offer, it costs Rs.150/for two. In an offer with any product Y of a lesser known/unpopular brand, it will cost Rs.120/-.':'Buying trends','Have you ever noticed the QR code scanner on the packaging? If yes, have you used it?':'QR code scanner usage','RANK IN ORDER OF PREFERENCE: For each attribute, select one designated rank, that will not be repeated for any other. (1 : highest, 8 : lowest) [Brand name]':'Rank preference for brand name','RANK IN ORDER OF PREFERENCE: For each attribute, select one designated rank, that will not be repeated for any other. (1 : highest, 8 : lowest) [Nutrition facts]':'Rank preference for nutrition facts','RANK IN ORDER OF PREFERENCE: For each attribute, select one designated rank, that will not be repeated for any other. (1 : highest, 8 : lowest) [Ingredients]':'Rank preference for ingredients','RANK IN ORDER OF PREFERENCE: For each attribute, select one designated rank, that will not be repeated for any other. (1 : highest, 8 : lowest) [Packaging design/graphics]':'Rank preference for packaging design','RANK IN ORDER OF PREFERENCE: For each attribute, select one designated rank, that will not be repeated for any other. (1 : highest, 8 : lowest) [Packaging material ]':'Rank preference for packaging material','RANK IN ORDER OF PREFERENCE: For each attribute, select one designated rank, that will not be repeated for any other. (1 : highest, 8 : lowest) [Size, weight and shape of item]':'Rank preference for dims of item','RANK IN ORDER OF PREFERENCE: For each attribute, select one designated rank, that will not be repeated for any other. (1 : highest, 8 : lowest) [Expiration date]':'Rank preference for exp date','RANK IN ORDER OF PREFERENCE: For each attribute, select one designated rank, that will not be repeated for any other. (1 : highest, 8 : lowest) [Cost]':'Rank preference for Cost'}, inplace = True)
df.columns
df.head()
df['Age '].value_counts(ascending=True)

plt.subplots(figsize=(15,5))
sns.countplot('Age ',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.xlabel('Age of customer')
plt.title('Number of customers of varied age groups')
plt.show()
df['Preferred traditional packaging'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()

plt.subplots(figsize=(15,5))
sns.countplot('Preferred traditional packaging',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=0)
plt.xlabel('Traditional packaging material')
plt.title('Preference count')
plt.show()
df['Smaller packaging'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()

df['Prepacked packed fruits and vegetables'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()

df['Aseptic/Tetra packing'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['Support modified atmosphere packaging'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['Prefer tamper evident packaging'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['Satisfied with the veg/non-veg mark'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['Colour code stripe on toothpastes/creams'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['Change company for more appealing packaging?'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['Mirroring in product and product image'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['Is information on packaging enough and visible?'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['Preferred colour in packaging'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
df['QR code scanner usage'].value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.figure(1, figsize=(40,40))
plt.show()
plt.subplots(figsize=(15,5))
sns.countplot('Quality-Packaging preference',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=0)
plt.xlabel('Quality-Packaging options')
plt.title('Preference count')
plt.show()
plt.subplots(figsize=(15,5))
sns.countplot('Buying trends',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.xlabel('Buying trends options')
plt.title('Preference count')
plt.show()
plt.subplots(figsize=(15,5))
sns.countplot('Is information on packaging enough and visible?',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=0)
#plt.xlabel('Buying trends options')
plt.title('Is information on packaging enough and visible?')
plt.show()
plt.subplots(figsize=(15,5))
sns.countplot('Change company for more appealing packaging?',data=df,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=0)
#plt.xlabel('Buying trends options')
plt.title('Change company for more appealing packaging?')
plt.show()
df.columns
df.groupby(['Effect of difficulty in opening packaging','Smaller packaging']).size().unstack().plot(kind='bar',stacked=True)
plt.show()
df.groupby(['Crucial factor','Prepacked packed fruits and vegetables']).size().unstack().plot(kind='bar',stacked=True)
plt.show()
df.groupby(['Crucial factor','Aseptic/Tetra packing']).size().unstack().plot(kind='bar',stacked=True)
plt.show()
df.rename(columns={'Overall monthly household income of the family (in number)': 'Income'}, inplace=True)
df['Income'].isnull().value_counts()
n= df['Income'].isnull().sum()
m=len(df.index)
print ("The % of people who didn't reveal their annual income is: ",end="") 
print ("{0:.2f}".format(((n)/m)*100)) 

label1=df['Crucial factor'].unique()
l=label1.tolist()
val1=df['Crucial factor'].value_counts()
v=val1.tolist()
new = pd.DataFrame(list(zip(l, v)), 
               columns =['Name', 'val']) 
new 

fig = px.bar(new, x="val", y="Name", color='Name', orientation='h',height=300,
             title='Factors that inluence Packaging Material ',
             color_discrete_sequence=px.colors.sequential.Viridis)
fig.show()
label2=df['Effect of difficulty in opening packaging'].unique()
label2.tolist()
val2=df['Effect of difficulty in opening packaging'].value_counts()
val2
night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)',
                'rgb(36, 55, 57)', 'rgb(6, 4, 4)']
fig = go.Figure(data=[go.Pie(labels=label2, values=val2,marker_colors=night_colors)])
fig.show()
top_labels = ['Glass', 'Metal','Plastic','Traditional<br>Packaging', 'Paper/Cloth']

x_data = [[37,14,68,43,41],
          [17,7,34,36,109],
          [56,47,46,30,24],
          [57,33,61,29,23],
         [16,27,50,42,68],
         [6,25,72,49,51],
         [3,3,46,37,114],
         [9,13,37,89,55]]

y_data = ['pharma pref',
          'fruits packaging ',
          'Preserved food', 'Cosmetics ' ,
          'Accessories',
          'Egoods','Clothes','Others']

colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
          'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
          'rgba(190, 192, 213, 1)']

fig = go.Figure()

for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        fig.add_trace(go.Bar(
            x=[xd[i]], y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(color='rgb(248, 248, 249)', width=1)
            )
        ))

fig.update_layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
    ),
    barmode='stack',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    margin=dict(l=120, r=10, t=140, b=80),
    showlegend=False,
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0] / 2, y=yd,
                            text=str(xd[0]) + '%',
                            font=dict(family='Arial', size=14,
                                      color='rgb(248, 248, 255)'),
                            showarrow=False))
    # labeling the first option (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd,
                                    text=str(xd[i]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling e
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

fig.update_layout(annotations=annotations)

fig.show()
l14=df['Support modified atmosphere packaging'].unique()
l14.tolist()
v14=df['Support modified atmosphere packaging'].value_counts()

sun_colors = ['rgb(245, 213, 34)', 'rgb(245, 161, 34)', 'rgb(245, 59, 34)',
                 'rgb(150, 245, 34)', 'rgb(177, 189, 11)']
fig = go.Figure(data=[go.Pie(labels=l14, values=v14,title='consumption products with modified atmosphere packaging',pull=[0.2, 0, 0],marker_colors=sun_colors)])
fig.show()


d1=pd.DataFrame(df['Rank preference for brand name'].value_counts())
d1.index.name = 'Rank'
d1.rename(columns={'Rank preference for brand name':'Brand name'}, inplace=True)
#df.set_index("Rank",inplace=True)
#
#d1
d1.reset_index()
d1.sort_values(by=['Rank'],ascending=True)

d1['Nutrition facts']=df['Rank preference for nutrition facts'].value_counts()
d1['Ingredients']=df['Rank preference for ingredients'].value_counts()
d1['Design']=df['Rank preference for packaging design'].value_counts()
d1['Material']=df['Rank preference for packaging material'].value_counts()
d1['Dims']=df['Rank preference for dims of item'].value_counts()
d1['Exp']=df['Rank preference for exp date'].value_counts()
d1['Cost']=df['Rank preference for Cost'].value_counts()
d1.index.name = 'Rank'
d1=d1.reset_index()

#d1.set_index()
d1.sort_values(by=['Rank'],ascending=True,inplace=True)

d1
ax = plt.gca()


d1.plot(kind='line',x='Rank',y='Brand name',ax=ax)
d1.plot(kind='line',x='Rank',y='Nutrition facts', color='red', ax=ax)
d1.plot(kind='line',x='Rank',y='Ingredients', color='green', ax=ax)
d1.plot(kind='line',x='Rank',y='Design', color='black', ax=ax)
d1.plot(kind='line',x='Rank',y='Material', color='yellow', ax=ax)
d1.plot(kind='line',x='Rank',y='Dims', color='darkorange', ax=ax)
d1.plot(kind='line',x='Rank',y='Exp', color='cyan', ax=ax)
d1.plot(kind='line',x='Rank',y='Cost', color='magenta', ax=ax)
plt.show()

d1.plot(kind='bar',x='Rank',y='Brand name')
d1.plot(kind='bar',x='Rank',y='Ingredients')

df.groupby(['Change company for more appealing packaging?','Preferred colour in packaging']).size().unstack().plot(kind='bar',stacked=True)
#plt.xticks(rotation=0)
plt.show()
df[['Age ']].plot(kind='hist',bins=[10,20,30,40,50,60,70],rwidth=0.8)
plt.show()

label3=df['QR code'].unique()
l11=label3.tolist()
val3=df['QR code'].value_counts()
v11=val3.tolist()
new1 = pd.DataFrame(list(zip(l11, v11)), 
               columns =['QR Code', 'val']) 
new1

ax = plt.gca()

df.plot(kind='line',x='',y='num_children',ax=ax)
df.plot(kind='line',x='name',y='num_pets', color='red', ax=ax)

plt.show()

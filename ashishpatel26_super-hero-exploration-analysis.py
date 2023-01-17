import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import cufflinks as cf
color = sns.color_palette()
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import random
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
import random

values = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
value_dict = [['A','10'],['B','11'],['C','12'],['D','13'],['E','14'],['F','15']]
start = "#"

def contrastcolour(colour):
    if colour[0] == '#':
        colour = colour[1:]
    rgb = (colour[0:2], colour[2:4], colour[4:6])
    comp = ['%02X' % (255 - int(a, 16)) for a in rgb]
    return ''.join(comp)

def startcolour():
    colour = "#"
    for i in range(6):
        x = values[random.randint(0,len(values)-1)]

        for thing in value_dict:
            if x == thing[1]:
                x = thing[0]
        colour = colour + x
    return colour

base = startcolour()
contrast = start + contrastcolour(base)

print("The colours: {0}".format([base, contrast]))
type_colors =[]
color_theme = dict(color = type_colors)
for i in range(1000):
    base = startcolour()
    contrast = start + contrastcolour(base)
    type_colors.append(contrast)        
# visualization
def pie_chart(data, title = "", filename = ""):
    '''
    data = pass the data or column,title = give the title of the graph, filename = give the file name of file
    '''
    temp_series = data.value_counts()
    labels = (np.array(temp_series.index))
    sizes = (np.array((temp_series / temp_series.sum())*100))
    # Draw plot
    trace = go.Pie(labels=labels, values=sizes)
    layout = go.Layout(
        title=title,
        width=900,
        height=900,
    )
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig, filename=filename)

def bar_graph(x,y, header = "", xaxis = "", yaxis=""):
    """
    x = put the x dimension of dataframe
    y = put the y dimemsion of dataframe
    header = give the title of the graph
    xaxis = x label value
    yaxis = y label value
    """
    data = [go.Bar(x=x,y=y,marker = color_theme)]
    layout = dict(title = header, xaxis = dict(title=xaxis),yaxis = dict(title=yaxis))
    fig = dict(data = data, layout = layout)
    return py.iplot(fig, filename='basic_bar')

def box_plot(y,x, header = "", xaxis = "", yaxis=""):
    """
    x = put the x dimension of dataframe
    y = put the y dimemsion of dataframe
    header = give the title of the graph
    xaxis = x label value
    yaxis = y label value
    """
    trace0 = go.Box(y=y,x=x, marker=dict(color='#FF851B'))
    data = [trace0]
    layout = go.Layout(yaxis=dict(title=yaxis,zeroline=False),xaxis=dict(title=xaxis,zeroline=False), title = header)
    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig)

def get_insights(path):
    """
    path :  pass your csv path here.
    """
    df = pd.read_csv(path)
    print("=====Data insight=====\n",df.head(10))
    print("=====Data Types====")
    print("Continues Data Columns Counts:\n",df.select_dtypes(include = ['float64', 'int64']).count())
    print("Continues Data Column List:",df.select_dtypes(include = ['float64', 'int64']).columns.tolist())
    conti = df.select_dtypes(include = ['float64', 'int64'])
    ctg = df.select_dtypes(include = ['object'])
    print("Categorical Data Columns Counts:\n",df.select_dtypes(include = ['object']).count())
    print("Categorical Data Columns:\n",df.select_dtypes(include = ['object']).columns.tolist())
    print("=====Missing Value Count========")
    print("Continues Data Column Missing Values:\n",df.select_dtypes(include = ['float64', 'int64']).isna().sum())
    mis_cont= df.select_dtypes(include = ['float64', 'int64']).isna().sum()
    print("Categorical Data Column Missing Values:\n",df.select_dtypes(include = ['object']).isna().sum())
    mis_ctg = df.select_dtypes(include = ['object']).isna().sum()
    print("=====Null Value Count========")
    print("Continues Data Column Null Values:\n",df.select_dtypes(include = ['float64', 'int64']).isnull().sum())
    null_cont= df.select_dtypes(include = ['float64', 'int64']).isnull().sum()
    print("Categorical Data Column Null Values:\n",df.select_dtypes(include = ['object']).isnull().sum())
    null_ctg = df.select_dtypes(include = ['object']).isnull().sum()
    return df,conti,ctg, mis_cont, mis_ctg, null_cont, null_ctg

def impute_missing(data):
    """
    data = pass the dataframe here
    """
    data.fillna('Other',inplace=True)
    return data
    
df,conti,ctg, mis_cont, mis_ctg, null_cont, null_ctg = get_insights("../input/superhero-set/heroes_information.csv")
impute_missing(df)
df1,conti1,ctg1, mis_cont1, mis_ctg1,null_cont1, null_ctg1 = get_insights("../input/superhero-set/super_hero_powers.csv")
impute_missing(df1)
df1 = df1*1
df1.head()
df1.loc[:, 'no_of_powers']  = df1.iloc[:, 1:].sum(axis=1)
df1.head(1)
df_all_power_hero=df1[['hero_names','no_of_powers']]
df_all_power_hero.head(1)
df_all_power_hero = df_all_power_hero.sort_values('no_of_powers', ascending=False)
xdf = df_all_power_hero["hero_names"].head(30)
ydf = df_all_power_hero["no_of_powers"].head(30)
bar_graph(xdf,ydf,header="Power of different Superhero", xaxis="Super Hero", yaxis="Power")
pie_chart(df["Publisher"],title="Comic Wise Super Heroes Distribution", filename = "file1")
pie_chart(df["Gender"],title="Gender Wise Superheroes distribution",filename = "file2")
pie_chart(df["Eye color"],title="Eyecolor wise Superheroes Distibution")
# temp_series = df1.ix[df1['Hair color']!='No Hair']['Hair color'].value_counts()
pie_chart(df.ix[df['Skin color']!='-']['Skin color'], title="Skin Color wise Superheroes Distibution")
box_plot(y=df["Height"],x = df["Gender"],header="Height Distributed By Gender", xaxis="Gender", yaxis="Height")
box_plot(y=df["Weight"],x = df["Gender"],header="Weight Distributed By Gender", xaxis="Gender", yaxis="Weight")
df_align =df.ix[df['Gender']=='Male']
count_align = df_align['Alignment'].value_counts()
bar_graph(x= count_align.index,y=count_align.values,header="Alignment of Male Superheroes", xaxis="Alignment", yaxis="Count")
df_align =df.ix[df['Gender']=='Female']
count_align = df_align['Alignment'].value_counts()
bar_graph(x= count_align.index,y=count_align.values,header="Alignment of Female Superheroes", xaxis="Alignment", yaxis="Count")
df_race = df['Race'].value_counts()
bar_graph(x= df_race.index,y=df_race.values,header="Racetypes of Superheroes", xaxis="Race", yaxis="Count")
df['hair'] = np.where(df['Hair color']=="No Hair", 'Bald', 'Non-Balded')
df_hair = df['hair'].value_counts()
# y=df_hair.index[::-1],x=df_hair.values[::-1]
bar_graph(x= df_hair.index[::-1],y=df_hair.values[::-1],header="How many superheroes are bald?", xaxis="Count", yaxis="Hair Color")
df2=pd.read_csv('../input/superhero-set/super_hero_powers.csv')
df2.head()
#Store Superhero in superhero variable to extract super hero name
df_superhero=df2['hero_names']

df2.drop('hero_names',axis=1,inplace=True)
df3 = pd.DataFrame()

for i in df2.columns:
    df3[i]=df2[i].value_counts()
df3.head()
df3.drop(df3.index == "False", inplace=True)
df3
df3.shape
df3 = df3.T.reset_index()
df3.columns = ["index","No_of_Superheroes"]
df3 = df3.sort_values('No_of_Superheroes', ascending=False)
df3=df3.ix[df3['No_of_Superheroes']>50]
len(df3)
bar_graph(x = df3["index"],y=df3["No_of_Superheroes"], xaxis= "powers", yaxis="Super Heroes")
from PIL import Image
from wordcloud import WordCloud
from nltk.corpus import stopwords
import numpy as np
mask = np.array(Image.open('../input/spider/spider.jpg'))
stopwords = stopwords.words('english')
wc = WordCloud(background_color="white", max_words=2000, mask=mask,stopwords=stopwords)
cloud = wc.generate(" ".join(df['Publisher']))
plt.figure(figsize=(20, 12))
plt.imshow(cloud)
plt.axis('off')

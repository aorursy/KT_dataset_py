# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pyspark import SparkContext

from pyspark.sql import SQLContext, SparkSession, Row
sc = SparkContext("local",'quiz-data-viz')
spark = SparkSession(sc)

data_quiz = sc.textFile('/kaggle/input/output1.csv').map(lambda line:line.split(','))
df_quiz = spark.read.csv('/kaggle/input/output1.csv',header=True)

df_quiz.head(3)
df_quiz.printSchema()
import ast

sum_male = df_quiz.rdd.filter(lambda x:x[6]=='Homme').count()

sum_female = df_quiz.rdd.filter(lambda x:x[6] == 'Femme').count()
def get_statistic_by(key_num):

    x_radio = df_quiz.rdd.map(lambda x:(x[key_num], ast.literal_eval(x[11]))).filter(lambda x: len(x[1]) > 0)

    x_music = df_quiz.rdd.map(lambda x:(x[key_num], ast.literal_eval(x[12]))).filter(lambda x: len(x[1]) > 0)

    x_ambiance = df_quiz.rdd.map(lambda x:(x[key_num], ast.literal_eval(x[13]))).filter(lambda x: len(x[1]) > 0)

    x_emission = df_quiz.rdd.map(lambda x:(x[key_num], ast.literal_eval(x[14]))).filter(lambda x: len(x[1]) > 0)

    # get flat

    x_radio_flat = x_radio.map(lambda x: [((x[0],y),1) for y in x[1]]).flatMap(lambda x: (t for t in x))

    x_music_flat = x_music.map(lambda x: [((x[0],y),1) for y in x[1]]).flatMap(lambda x: (t for t in x))

    x_ambiance_flat = x_ambiance.map(lambda x: [((x[0],y),1) for y in x[1].items()]).flatMap(lambda x: (t for t in x))

    x_emission_flat = x_emission.map(lambda x: [((x[0],y),1) for y in x[1]]).flatMap(lambda x: (t for t in x))

    # get statistics

    data_x_radio = x_radio_flat.reduceByKey(lambda a,b : (a+b)).collect()

    data_x_music = x_music_flat.reduceByKey(lambda a,b : (a+b)).collect()

    data_x_ambiance = x_ambiance_flat.reduceByKey(lambda a,b : (a+b)).collect()

    data_x_emission = x_emission_flat.reduceByKey(lambda a,b : (a+b)).collect()

    return data_x_radio, data_x_music, data_x_ambiance, data_x_emission
import matplotlib.pyplot as plt 

import numpy as np



def plot_data_gender(dataset, vr, name):

    data_male = list(filter(lambda x: x[0][0]=='Homme', dataset))

    data_female = list(filter(lambda x: x[0][0]=='Femme', dataset))

    dm = sorted(data_male ,key=lambda x:x[1], reverse=True)

    df = sorted(data_female ,key=lambda x:x[1], reverse=True)

    fig = plt.figure(figsize=(15,10), dpi=80, facecolor='w')

    ax1 = fig.add_subplot(311)

    ax1.bar(np.arange(len(dm)),[x[1] for x in dm], align='center')

    ax1.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(dm))))

    ax1.xaxis.set_major_formatter(plt.FixedFormatter([x[0][1] for x in dm]))

    plt.xticks(rotation=vr)

    plt.title("The "+name+" distribution for male")

    ax2 = fig.add_subplot(313)

    ax2.bar(np.arange(len(df)), [x[1] for x in df], align='center', color='r')

    ax2.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(df))))

    ax2.xaxis.set_major_formatter(plt.FixedFormatter([x[0][1] for x in df]))

    plt.xticks(rotation=vr)

    plt.title("The "+name+" distribution for female")

    plt.show()

    
import plotly.graph_objects as go

import plotly.express as px

data_gender_radio, data_gender_music, data_gender_ambiance, data_gender_emission = get_statistic_by(6)
def plot_data_by_gender(dataset, gtitle, xtitle):

    data_male = list(filter(lambda x: x[0][0]=='Homme', dataset))

    data_female = list(filter(lambda x: x[0][0]=='Femme', dataset))

    dm = sorted(data_male ,key=lambda x:x[1], reverse=True)

    df = sorted(data_female ,key=lambda x:x[1], reverse=True)

    fig = go.Figure(

        data=[

            go.Bar(name='Male', marker_color='#EB89B5', x=list(map(lambda x:x[0][1], dm[:12])), y=list(map(lambda x:x[1]/sum_male, dm[:12]))),

            go.Bar(name="Female", marker_color='#330C73', x=list(map(lambda x:x[0][1], df[:12])), y=list(map(lambda x:x[1]/sum_female, df[:12])))

        ], 

        layout=go.Layout(

            title=gtitle,

            xaxis=dict(

                title=xtitle,

                titlefont=dict(

                    family = 'Courier New, monospace',

                    size = 18,

                    color = '#7f7f7f'

                )

            )

        )

    )



    fig.update_layout(barmode='group')

    fig.show()

    return None





plot_data_by_gender(data_gender_radio, 'Comparison of the radio preference by gender', 'Radio')
plot_data_by_gender(data_gender_music, 'Comparison of the music preference by gender', 'music')
dga = list(map(lambda x: ( ( (x[0][0], x[0][1][0]+':'+str(x[0][1][1]) ),x[1]) ), data_gender_ambiance))

plot_data_by_gender(dga, 'Comparison of the ambiance preference by gender', 'ambiance')
plot_data_by_gender(data_gender_emission, 'Comparison of the emission preference by gender', 'emission')
sum_19to26 =  df_quiz.rdd.filter(lambda x:x[7]=='19-26').count()

sum_27to35 = df_quiz.rdd.filter(lambda x:x[7] == '27-35').count()

sum_36to50 = df_quiz.rdd.filter(lambda x:x[7] == '36-50').count()

sum_51to65 = df_quiz.rdd.filter(lambda x:x[7] == '51-65').count()

print(sum_19to26, sum_27to35, sum_36to50, sum_51to65)
data_age_radio, data_age_music, data_age_ambiance, data_age_emission = get_statistic_by(7)
def plot_data_by_age(dataset, gtitle, xtitle):

    data_19to26 = list(filter(lambda x: x[0][0]=='19-26', dataset))

    data_27to35 = list(filter(lambda x: x[0][0]=='27-35', dataset))

    data_36to50 = list(filter(lambda x: x[0][0]=='36-50', dataset))

    data_51to65 = list(filter(lambda x: x[0][0]=='51-65', dataset))

    

    d19 = sorted(data_19to26 ,key=lambda x:x[1], reverse=True)[:12]

    d27 = sorted(data_27to35 ,key=lambda x:x[1], reverse=True)[:12]

    d36 = sorted(data_36to50 ,key=lambda x:x[1], reverse=True)[:12]

    d51 = sorted(data_51to65 ,key=lambda x:x[1], reverse=True)[:12]

    

    def get_dict(l, somme):

        res = dict()

        for x in l:

            res.update({x[0][1]:x[1]/somme})

        return res

    

    dd19 = get_dict(d19, sum_19to26)

    dd27 = get_dict(d27, sum_27to35)

    dd36 = get_dict(d36, sum_36to50)

    dd51 = get_dict(d51, sum_51to65)

    

    keys = list(set(list(dd19.keys()) + list(dd27.keys()) + list(dd36.keys()) + list(dd51.keys())))

    

    def get_y(keys, dd):

        res = list()

        for k in keys:

            v = dd.get(k)

            if v:

                res.append(v)

            else:

                res.append(0)

        return res

    

    v19 = get_y(keys, dd19)

    v27 = get_y(keys, dd27)

    v36 = get_y(keys, dd36)

    v51 = get_y(keys, dd51)

    

    fig = go.Figure(

        data=[

            go.Bar(name='19-26', x=keys, y=v19),

            go.Bar(name="27-35", x=keys, y=v27),

            go.Bar(name="36-50", x=keys, y=v36),

            go.Bar(name="51-65", x=keys, y=v51),

            go.Scatter(name='19-26', x=keys, y=v19),

            go.Scatter(name="27-35", x=keys, y=v27),

            go.Scatter(name="36-50", x=keys, y=v36),

            go.Scatter(name="51-65", x=keys, y=v51),

        ], 

        layout=go.Layout(

            title=gtitle,

            xaxis=dict(

                title=xtitle,

                titlefont=dict(

                    family = 'Courier New, monospace',

                    size = 18,

                    color = '#7f7f7f'

                )

            )

        )

    )



    fig.update_layout(barmode='group')

    fig.show()

    return None



plot_data_by_age(data_age_radio, 'Comparison of the radio preference by age', 'Radio')
plot_data_by_age(data_age_music, 'Comparison of the music preference by age', 'Music')
daa = list(map(lambda x: ( ( (x[0][0], x[0][1][0]+':'+str(x[0][1][1]) ),x[1]) ), data_age_ambiance))

plot_data_by_age(daa, 'Comparison of the ambiance preference by age', 'Ambiance')
plot_data_by_age(data_age_emission, 'Comparison of the emission preference by age', 'Emission')
data_leisure_radio, data_leisure_music, data_leisure_ambiance, data_leisure_emission = get_statistic_by(8)
sum_s=  df_quiz.rdd.filter(lambda x:x[8]=='Sportif').count()

sum_t = df_quiz.rdd.filter(lambda x:x[8] == 'Touristique').count()

sum_c = df_quiz.rdd.filter(lambda x:x[8] == 'Culturel').count()

sum_i = df_quiz.rdd.filter(lambda x:x[8] == 'Informatique').count()

print(sum_s, sum_t, sum_c, sum_i)
def plot_data_by_leisure(dataset, gtitle, xtitle):

    data_s = list(filter(lambda x: x[0][0]=='Sportif', dataset))

    data_t = list(filter(lambda x: x[0][0]=='Touristique', dataset))

    data_c = list(filter(lambda x: x[0][0]=='Culturel', dataset))

    data_i = list(filter(lambda x: x[0][0]=='Informatique', dataset))

    

    ds = sorted(data_s ,key=lambda x:x[1], reverse=True)[:12]

    dt = sorted(data_t ,key=lambda x:x[1], reverse=True)[:12]

    dc = sorted(data_c ,key=lambda x:x[1], reverse=True)[:12]

    di = sorted(data_i ,key=lambda x:x[1], reverse=True)[:12]

    

    def get_dict(l, somme):

        res = dict()

        for x in l:

            res.update({x[0][1]:x[1]/somme})

        return res

    

    dds = get_dict(ds, sum_s)

    ddt = get_dict(dc, sum_c)

    ddc = get_dict(dt, sum_t)

    ddi = get_dict(di, sum_i)

    

    keys = list(set(list(dds.keys()) + list(ddt.keys()) + list(ddc.keys()) + list(ddi.keys())))

    

    def get_y(keys, dd):

        res = list()

        for k in keys:

            v = dd.get(k)

            if v:

                res.append(v)

            else:

                res.append(0)

        return res

    

    vs = get_y(keys, dds)

    vt = get_y(keys, ddt)

    vc = get_y(keys, ddc)

    vi = get_y(keys, ddi)

    

    fig = go.Figure(

        data=[

            go.Bar(name='Sportif', x=keys, y=vs),

            go.Bar(name="Touristique", x=keys, y=vt),

            go.Bar(name="Culturel", x=keys, y=vc),

            go.Bar(name="Informatique", x=keys, y=vi),

            go.Scatter(name='Sportif', x=keys, y=vs),

            go.Scatter(name="Touristique", x=keys, y=vt),

            go.Scatter(name="Culturel", x=keys, y=vc),

            go.Scatter(name="Informatique", x=keys, y=vi),

        ], 

        layout=go.Layout(

            title=gtitle,

            xaxis=dict(

                title=xtitle,

                titlefont=dict(

                    family = 'Courier New, monospace',

                    size = 18,

                    color = '#7f7f7f'

                )

            )

        )

    )



    fig.update_layout(barmode='group')

    fig.show()

    return None



plot_data_by_leisure(data_leisure_radio, 'Comparison of the radio preference by hobby', 'Radio')
plot_data_by_leisure(data_leisure_music, 'Comparison of the music preference by hobby', 'Music')
dla = list(map(lambda x: ( ( (x[0][0], x[0][1][0]+':'+str(x[0][1][1]) ),x[1]) ), data_leisure_ambiance))

plot_data_by_leisure(dla, 'Comparison of the ambiance preference by hobby', 'Ambiance')
plot_data_by_leisure(data_leisure_emission, 'Comparison of the emission preference by hobby', 'Emission')
color_radio = df_quiz.rdd.map(lambda line: (ast.literal_eval(line[10]), ast.literal_eval(line[11]))).filter(lambda x: len(x[0]) > 0)

color_music = df_quiz.rdd.map(lambda line: (ast.literal_eval(line[10]), ast.literal_eval(line[12]))).filter(lambda x: len(x[0]) > 0)

color_ambiance = df_quiz.rdd.map(lambda line: (ast.literal_eval(line[10]), ast.literal_eval(line[13]))).filter(lambda x: len(x[0]) > 0)

color_emission = df_quiz.rdd.map(lambda line: (ast.literal_eval(line[10]), ast.literal_eval(line[14]))).filter(lambda x: len(x[0]) > 0)
lColor = ['red', 'blue', 'green', 'yellow', 'brown', 'black', 'white']

def color2vec(dColor):

    res = [0 for i in range(7)]

    for k, v in dColor.items():

        res[lColor.index(k)] = (8-int(v))*30/(int(v)+1)

    return np.array(res)



lRadio = list(set(color_radio.flatMap(lambda x: x[1]).collect()))

lMusic = list(set(color_music.flatMap(lambda x: x[1]).collect()))



data_color_radio = color_radio.map(lambda line: [(color2vec(line[0]), lRadio.index(e) ) for e in line[1] ]).flatMap(lambda l: (t for t in l))

data_color_music = color_music.map(lambda line: [(color2vec(line[0]), lMusic.index(e) ) for e in line[1] ]).flatMap(lambda l: (t for t in l))
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

xy_vector_color_radio = pca.fit_transform(data_color_radio.map(lambda x:x[0]).collect())

xy_vector_color_music = pca.fit_transform(data_color_music.map(lambda x:x[0]).collect())



jColor = ['red', 'blue', 'green', 'yellow', 'brown', 'black', '#00ffae'] # replace white by #00ffae because withe is not visible

vector_color_radio_label = data_color_radio.map(lambda x: jColor[np.argmax(x[0])]).collect()

z_color_radio = data_color_radio.map(lambda x: x[1]).collect()

x_vector_color_radio = list(map(lambda x: x[0], xy_vector_color_radio))

y_vector_color_radio = list(map(lambda x: x[1], xy_vector_color_radio))



x_vector_color_music = list(map(lambda x: x[0], xy_vector_color_music))

y_vector_color_music = list(map(lambda x: x[1], xy_vector_color_music))

z_color_music = data_color_music.map(lambda x:x[1]).collect()

vector_color_music_label = data_color_music.map(lambda x: jColor[np.argmax(x[0])]).collect()
from plotly.subplots import make_subplots

fig = make_subplots(rows=1,

                    cols=2, 

                    specs=[[{'type': 'surface'}, {'type': 'surface'}]],

                    subplot_titles=("Radio", "Music")

                   )

fig.add_trace(

    go.Scatter3d(

        x=x_vector_color_radio,

        y=y_vector_color_radio,

        z=z_color_radio,

        marker_color = vector_color_radio_label,

        opacity=0.8,

        mode='markers'

    ),

    row=1,

    col=1

)

fig.add_trace(

     go.Scatter3d(

        x=x_vector_color_music,

        y=y_vector_color_music,

        z=z_color_music,

        marker_color = vector_color_music_label,

         opacity=0.8,

         mode='markers'

    ),

    row=1, col=2

)

fig.update_layout(height=600, width=800, title_text="Comparison of different color preference")

fig.show()
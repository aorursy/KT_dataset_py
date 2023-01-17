# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Se importan las librerías necesarias

import matplotlib.pyplot as plt

import seaborn as sns

import missingno

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Cargamos el dataset y procedemos a mostrar los primeros 5 registros

training_set = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

training_set.head()
# Para conocer las dimensiones del dataset procedems a correr el comando

training_set.shape
# Se pueden listar las columas de la siguiente manera

training_set.columns
# Es posible obtener algunas estadisticas relevantes de las caracteristicas numéricas facilmente

training_set.describe()
# Se puede visualizar el tipo de datos presentes en el dataset

training_set.info()
# Para obtener un conteo dedatos nulos, se puede hacer

training_set.isnull().sum()
# O de manera agrupada

training_set.isnull().sum().values.sum()
#Una forma interesante de ver datos nulos, desde la generalidad, es usando missingno

missingno.matrix(training_set, figsize = (30,10))
# La diversidad de valores en el dataset se puede observar así

print ("\nValores Unicos :  \n",training_set.nunique())
# Reemplazamos vacios por NaN en Total Charges

training_set['TotalCharges'] = training_set["TotalCharges"].replace(" ",np.nan)

training_set.isnull().sum()
# En la columna TotalCharges hay muy pocos valores nulos, por lo tanto se puden eliminar dichos registros

training_set = training_set[training_set["TotalCharges"].notnull()]

training_set = training_set.reset_index()[training_set.columns]



# También se puede convertir a tipo flotante.

training_set["TotalCharges"] = training_set["TotalCharges"].astype(float)



# Describimos nuevamente las variables numéricas

training_set.describe()
# En el caso de 'tenure' se puede crer un atributo categórico por rangos de permanencia en años (12 meses)

def tenureRange(dataset):

    if dataset["tenure"] <= 12 :

        return "0-12"

    elif (dataset["tenure"] > 12) & (dataset["tenure"] <= 24 ):

        return "12-24"

    elif (dataset["tenure"] > 24) & (dataset["tenure"] <= 36) :

        return "24-36"

    elif (dataset["tenure"] > 36) & (dataset["tenure"] <= 48) :

        return "36-48"

    elif (dataset["tenure"] > 48) & (dataset["tenure"] <= 60) :

        return "48-60"

    elif dataset["tenure"] > 60 :

        return "mayor_60"



# Se aplica la transformación, generandose una nueva característica

training_set["TenureRange"] = training_set.apply(lambda training_set:tenureRange(training_set),

                                      axis = 1)

# Reemplazamos SeniorCitizen por Categórico para fines de representación

training_set["SeniorCitizen"] = training_set["SeniorCitizen"].replace({1:"Yes",0:"No"})



# Visualizamos el resultado

training_set.head()
# Consultamos las dimensiones del dataset

training_set.shape
training_set.select_dtypes(include=['object']).columns
# Definimos como variables a excluir la varaible ID y la variable objetivo

Id_att     = ['customerID']

target_att = ["Churn"]



# Seleccionamos los labels de las distintas categorías.

cat_att   = training_set.nunique()[training_set.nunique() <= 6].keys().tolist()

cat_att   = [x for x in cat_att if x not in target_att]

num_att   = [x for x in training_set.columns if x not in cat_att + target_att + Id_att]
print ("Categóricos: ", cat_att)

print ("Numéricos  : ", num_att)
churn     = training_set[training_set["Churn"] == "Yes"]

not_churn = training_set[training_set["Churn"] == "No"]

numerical_attributes = training_set.select_dtypes(include=["int", "float"])

numerical_attributes.hist(figsize=(10,12))
categorical_attributes = training_set.select_dtypes(include=["object"])

for i in cat_att:

    plt.figure(figsize=(10,3))

    sns.countplot(data=categorical_attributes, x=i)


#labels

labels = training_set["Churn"].value_counts().keys().tolist()

#values

values = training_set["Churn"].value_counts().values.tolist()



trace = go.Pie(labels = labels ,

               values = values ,

               marker = dict(colors =  [ 'royalblue' ,'lime'],

                             line = dict(color = "white",

                                         width =  1.3)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Retiro de clientes",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )



data = [trace]

fig = go.Figure(data = data,layout = layout)

py.iplot(fig)
#función para graficar el histograma de las variables numéricas con respecto a la variable dependiente

def histogram(column) :

    trace1 = go.Histogram(x  = churn[column],

                          histnorm= "percent",

                          name = "Clientes retirados",

                          marker = dict(line = dict(width = .5,

                                                    color = "black"

                                                    )

                                        ),

                         opacity = .9 

                         ) 

    

    trace2 = go.Histogram(x  = not_churn[column],

                          histnorm = "percent",

                          name = "Clientes no retirados",

                          marker = dict(line = dict(width = .5,

                                              color = "black"

                                             )

                                 ),

                          opacity = .9

                         )

    

    data = [trace1,trace2]

    layout = go.Layout(dict(title =column + " - distribución en permanencia de clientes",

                            plot_bgcolor  = "rgb(243,243,243)",

                            paper_bgcolor = "rgb(243,243,243)",

                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                             title = column,

                                             zerolinewidth=1,

                                             ticklen=5,

                                             gridwidth=2

                                            ),

                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                             title = "Porcentaje",

                                             zerolinewidth=1,

                                             ticklen=5,

                                             gridwidth=2

                                            ),

                           )

                      )

    fig  = go.Figure(data=data,layout=layout)

    

    py.iplot(fig)



# Para todos los atributos numéricos

for i in num_att :

    histogram(i)
# Kernel Density Estimation

def densidadProbabilidadplot(feature):

    plt.figure(figsize=(9, 4))

    plt.title("Densidad de probabilidad para '{}'".format(feature))

    ax0 = sns.kdeplot(not_churn[feature].dropna(), color= 'orange', label= 'Clientes no retirados')

    ax1 = sns.kdeplot(churn[feature].dropna(), color= 'navy', label= 'Clientes retirados')



for i in num_att :

    densidadProbabilidadplot(i)
def barplot_percentages(feature, orient='v', axis_name="% clientes"):

    ratios = pd.DataFrame()

    g = training_set.groupby(feature)["Churn"].value_counts().to_frame()

    g = g.rename({"Churn": axis_name}, axis=1).reset_index()

    g[axis_name] = g[axis_name]/len(training_set)

    if orient == 'v':

        ax = sns.barplot(x=feature, y= axis_name, hue='Churn', data=g, orient=orient)

        ax.set_yticklabels(['{:,.0%}'.format(y) for y in ax.get_yticks()])

    else:

        ax = sns.barplot(x= axis_name, y=feature, hue='Churn', data=g, orient=orient)

        ax.set_xticklabels(['{:,.0%}'.format(x) for x in ax.get_xticks()])

    ax.plot()

barplot_percentages("SeniorCitizen")
training_set['churn_rate'] = training_set['Churn'].replace("No", 0).replace("Yes", 1)

g = sns.FacetGrid(training_set, col="SeniorCitizen", height=4, aspect=.9)

ax = g.map(sns.barplot, "gender", "churn_rate", palette = "Blues_d", order= ['Female', 'Male'])
fig, axis = plt.subplots(1, 2, figsize=(12,4))

axis[0].set_title("Tiene Pareja")

axis[1].set_title("Tiene personas a cargo")

axis_y = "% clientes"

# Graficando columna 'Partner'

gp_partner = training_set.groupby('Partner')["Churn"].value_counts()/len(training_set)

gp_partner = gp_partner.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()

ax = sns.barplot(x='Partner', y= axis_y, hue='Churn', data=gp_partner, ax=axis[0])

# Graficando columna 'Dependents'

gp_dep = training_set.groupby('Dependents')["Churn"].value_counts()/len(training_set)

gp_dep = gp_dep.to_frame().rename({"Churn": axis_y}, axis=1).reset_index()

ax = sns.barplot(x='Dependents', y= axis_y, hue='Churn', data=gp_dep, ax=axis[1])
#función para graficar el diagrama Pie de las variables Categóricas con respecto a la variable dependiente

def plot_pie(column) :    

    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),

                    labels  = churn[column].value_counts().keys().tolist(),

                    hoverinfo = "label+percent+name",

                    domain  = dict(x = [0,.48]),

                    name    = "Clientes retirados",

                    marker  = dict(line = dict(width = 2,

                                               color = "rgb(243,243,243)")

                                  ),

                    hole    = .6

                   )

    trace2 = go.Pie(values  = not_churn[column].value_counts().values.tolist(),

                    labels  = not_churn[column].value_counts().keys().tolist(),

                    hoverinfo = "label+percent+name",

                    marker  = dict(line = dict(width = 2,

                                               color = "rgb(243,243,243)")

                                  ),

                    domain  = dict(x = [.52,1]),

                    hole    = .6,

                    name    = "Clientes no retirados" 

                   )





    layout = go.Layout(dict(title = column + " - distribución en permanencia de clientes ",

                            plot_bgcolor  = "rgb(243,243,243)",

                            paper_bgcolor = "rgb(243,243,243)",

                            annotations = [dict(text = "Clientes retirados",

                                                font = dict(size = 13),

                                                showarrow = False,

                                                x = .15, y = .5),

                                           dict(text = "Clientes no retirados",

                                                font = dict(size = 13),

                                                showarrow = False,

                                                x = .88,y = .5

                                               )

                                          ]

                           )

                      )

    data = [trace1,trace2]

    fig  = go.Figure(data = data,layout = layout)

    py.iplot(fig)

    

# Para todos los atributos categóricos

for i in cat_att :

    plot_pie(i)
#function  for scatter plot matrix  for numerical columns in data

def scatter_matrix(df)  :

    

    df  = df.sort_values(by = "Churn" ,ascending = True)

    classes = df["Churn"].unique().tolist()

    classes

    

    class_code  = {classes[k] : k for k in range(2)}

    class_code



    color_vals = [class_code[cl] for cl in df["Churn"]]

    color_vals



    pl_colorscale = "Portland"



    pl_colorscale



    text = [df.loc[k,"Churn"] for k in range(len(df))]

    text



    trace = go.Splom(dimensions = [dict(label  = "tenure",

                                       values = df["tenure"]),

                                  dict(label  = 'MonthlyCharges',

                                       values = df['MonthlyCharges']),

                                  dict(label  = 'TotalCharges',

                                       values = df['TotalCharges'])],

                     text = text,

                     marker = dict(color = color_vals,

                                   colorscale = pl_colorscale,

                                   size = 3,

                                   showscale = False,

                                   line = dict(width = .1,

                                               color='rgb(230,230,230)'

                                              )

                                  )

                    )

    axis = dict(showline  = True,

                zeroline  = False,

                gridcolor = "#fff",

                ticklen   = 4

               )

    

    layout = go.Layout(dict(title  = 

                            "Scatter plot matrix para atributos numéricos en permanencia de clientes",

                            autosize = False,

                            height = 800,

                            width  = 800,

                            dragmode = "select",

                            hovermode = "closest",

                            plot_bgcolor  = 'rgba(240,240,240, 0.95)',

                            xaxis1 = dict(axis),

                            yaxis1 = dict(axis),

                            xaxis2 = dict(axis),

                            yaxis2 = dict(axis),

                            xaxis3 = dict(axis),

                            yaxis3 = dict(axis),

                           )

                      )

    data   = [trace]

    fig = go.Figure(data = data,layout = layout )

    py.iplot(fig)



#scatter plot matrix

scatter_matrix(training_set)
# Podemos buscar correlaciones entre las varaibles usando un mapa de calor

sns.heatmap(training_set.corr(), annot=True)
sns.pairplot(training_set)
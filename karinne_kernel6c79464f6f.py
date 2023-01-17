from math import sqrt



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.decomposition import PCA

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import plotly

import plotly.graph_objects as go

import plotly.offline as py

from plotly.offline import plot, iplot

plotly.offline.init_notebook_mode(connected=True)

from yellowbrick.features import FeatureImportances



import warnings

warnings.filterwarnings("ignore")
fifa = pd.read_csv("../input/fifa.csv")
columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",

                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",

                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",

                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",

                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",

                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",

                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",

                   "CB", "RCB", "RB", "Release Clause"

]



try:

    fifa.drop(columns_to_drop, axis=1, inplace=True)

except KeyError:

    logger.warning(f"Columns already dropped")
# Sua análise começa aqui.

fifa.head()
# Identificando o tipo de cada variável:

fifa.dtypes
# Verificando se existem valores nulos no Dataset.

if fifa.isnull().sum().sort_values(ascending=False).any() != 0:

    print(f'Existe valores missing no dataset? {True}')

else:

    print(f'Existe valores missing no dataset? {False}')
# Removendo os valores missing:

fifa.dropna(inplace=True)
# Separando os dados em componentes de input e output:

X = fifa.drop(['Overall'], axis=1)

y = fifa['Overall']
# Aplicando o PCA no nosso conjunto de dados:

pca = PCA(n_components=3)

pcamodel = pca.fit_transform(fifa)
# Gráfico do PCA:

fig = go.Figure()

fig = go.Figure(data=[go.Scatter3d(x=pcamodel[:, 0],

                    y=pcamodel[:, 1],

                    z=pcamodel[:, 2],

                    marker=dict(opacity=1,

                    reversescale=True,

                    colorscale='Blues',

                    color='#228B22',

                    size=2.5),

                    line=dict (width=0.09),

                    mode='markers')])



# Layout:

fig.update_layout(scene=dict(xaxis=dict( title="PCA1"),

                                yaxis=dict( title="PCA2"),

                                zaxis=dict(title="PCA3")),

                                template='plotly_dark',

                                title="PCA",

                                font=dict(family="Courier New, monospace",

                                          size=12, 

                                          color="#ffffff"),

                                          autosize=False,

                                          width=700,

                                          height=400)





# Plot:

py.iplot(fig)
def q1():

    # PCA:

    pca = PCA().fit(fifa)



    # Fração da variância:

    return float(round(pca.explained_variance_ratio_[0],3))

q1()
# PCA:

pca = PCA().fit(fifa)



# Gráfico:

fig = go.Figure()

fig.add_trace(go.Scatter(y=(np.cumsum(pca.explained_variance_ratio_)),

                         mode='lines', line=dict(color="#8B0000",width = 4)))

    

# Layout:  

fig.update_layout(showlegend=False,

                  title="Variância explicada pelo primeiro componente principal",

                  xaxis_title="Número de componentes",

                  yaxis_title="Variância explicada (%)",

                  template='plotly_dark',

                  font=dict(family="Courier New, monospace",

                            size=12, 

                            color="#ffffff"),



                  annotations=[dict(x=0,

                                    y=0.565,

                                    xref="x",

                                    yref="y",

                                    text="fraction of variance",

                                    showarrow=True,

                                    arrowhead=2,

                                    arrowsize=2,

                                    arrowcolor="#FF8C00",

                                    ax=100,

                                    ay=-1,

                                    font=dict(

                                    family="Courier New, monospace",

                                    size=12,

                                    color="#ffffff"))])



# Plot:

py.iplot(fig)
def q2():

    # PCA:

    pca = PCA(.95).fit_transform(fifa)



    # Número de componentes:

    return pca.shape[1]

q2()
# Gráfico:



fig = go.Figure()

fig.add_trace(go.Scatter(y=(np.cumsum(pca.explained_variance_ratio_)),

                         mode='lines', line=dict(color="#8B0000",width = 4)))

    



# Layout:  

fig.update_layout(showlegend=False,

                  title="Número de componentes principais que explicam 95% da variância",

                  xaxis_title="Número de componentes",

                  yaxis_title="Variância explicada (%)",

                  template='plotly_dark',

                  font=dict(family="monospace",

                            size=10, color="#ffffff"),



                  annotations=[dict(x=15,

                                    y=0.95,

                                    xref="x",

                                    yref="y",

                                    text="95% of the total variance",

                                    showarrow=True,

                                    arrowhead=6,

                                    arrowsize=2,

                                    arrowcolor="#FF8C00",

                                    ax=120,

                                    ay=-1,

                                    font=dict(

                                    family="Courier New, monospace",

                                    size=12,

                                    color="#ffffff"))])



# Plot:                                    

py.iplot(fig)
x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,

     -35.55091139, -37.29814417, -28.68671182, -30.90902583,

     -42.37100061, -32.17082438, -28.86315326, -22.71193348,

     -38.36945867, -20.61407566, -22.72696734, -25.50360703,

     2.16339005, -27.96657305, -33.46004736,  -5.08943224,

     -30.21994603,   3.68803348, -36.10997302, -30.86899058,

     -22.69827634, -37.95847789, -22.40090313, -30.54859849,

     -26.64827358, -19.28162344, -34.69783578, -34.6614351,

     48.38377664,  47.60840355,  45.76793876,  44.61110193,

     49.28911284

]
def q3():

    # PCA:

    pca = PCA(n_components=2).fit(fifa)



    # Coordenadas (primeiro e segundo componentes principais):

    return tuple([round(x,3) for x in pca.components_.dot(x)])

q3()
# Gráfico :

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot()



# Layout:

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.spines['bottom'].set_visible(False)



# Plot:

viz = FeatureImportances(LinearRegression(), ax=ax)

viz.fit(X, y)

viz.poof();
def q4():

    # Criação do modelo:

    modelo = LinearRegression()

    

    # RFE:

    rfe = RFE(modelo,5).fit(X,y)

    return list(X.loc[:, rfe.support_].columns)

q4()
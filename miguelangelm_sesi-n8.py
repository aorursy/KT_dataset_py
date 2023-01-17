import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
 
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('seaborn-whitegrid')
#Normal
normal = np.random.normal(10,5, 10000)
#Uniforme
uniforme = np.random.uniform(0,10,10000)
#Binomial
binomial = np.random.binomial(10,0.4, 10000)
# Chi cuadrado
chi_cuadrado = np.random.chisquare(10,10000)
fig = make_subplots(rows=2,cols=2, subplot_titles = ('Normal','Uniforme','Bionomial','Chi_Cuadrado') )

fig.add_trace(go.Histogram(x = normal), row =1, col=1)
fig.add_trace(go.Histogram(x = uniforme), row = 1, col = 2)
fig.add_trace(go.Histogram(x = binomial), row = 2, col = 1)
fig.add_trace(go.Histogram(x = chi_cuadrado), row = 2, col = 2)

fig.show()
#Histograma
X = np.random.normal(10,5, 1000)
Y = np.random.normal(20,5,1000) + X
fig2 = go.Figure()

fig2.add_scatter(x=X,y=Y, mode = 'markers')

fig2.update_layout(
        width = 700,
        height = 500
)

fig2.show()

# Correlación

a = pd.Series(np.random.normal(0,5, 1000))
b = pd.Series(3 + a*2 )
c = pd.Series(np.random.normal(0,5,1000)* a **2)
d = pd.Series(np.random.uniform(0,10,1000))
e = pd.Series(np.random.chisquare(10,1000))

df = pd.concat([a,b,c,d,e], axis = 1, keys=['a','b','c','d','e'])
df.corr()
sns.heatmap(df.corr(), cmap = 'summer', annot = True)
df = pd.read_csv('../input/cnpv-2018/CNPV2018_5PER_A2_05.CSV',
                  usecols=['U_DPTO', 'U_MPIO', 'P_NROHOG', 'P_NRO_PER', 'P_SEXO', 'P_EDADR',
                          'P_PARENTESCOR', 'PA1_GRP_ETNIC', 'PA11_COD_ETNIA', 'PA12_CLAN',
                          'PA21_COD_VITSA', 'PA22_COD_KUMPA', 'PA_HABLA_LENG', 'PA1_ENTIENDE',
                          'P_NIVEL_ANOSR'],
                  nrows = 100000)

serie = pd.Series(df['P_NRO_PER'])
indices_drop = serie[serie > 25].index.to_list()
df2 = df.copy()
df2.drop(indices_drop, axis=0, inplace = True)
df2.drop(df2[df['P_NIVEL_ANOSR'].isnull() == True].index.tolist(),axis = 0, inplace = True)
caracteristicas = df2[['P_NRO_PER', 'P_SEXO','PA1_GRP_ETNIC']]
objetivo = df2['P_NIVEL_ANOSR']
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('seaborn-whitegrid')

caracteristicas.hist()
plt.show()
X, y = np.array(caracteristicas), np.array(objetivo)
import plotly.express as px

fig = plt.figure()
ax = Axes3D(fig)
colores=px.colors.qualitative.Alphabet[:len(objetivo.unique())]
color  =[]
tupla = []
for i,j in zip(objetivo.unique(),np.arange(0,len(colores))):
    tupla.append((i,j))
for i in y:
    for j in tupla:
        if i == j[0]:
            color.append(colores[j[1]])

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color,s=60)
plt.legend([str(i) for i in objetivo.unique()])

plt.show()
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 432)
kmeans = KMeans(n_clusters=10, random_state = 14)
kmeans.fit(X_train,y_train)

predicciones = kmeans.predict(X_test)
from sklearn import metrics

metrics.adjusted_rand_score(y_test, predicciones)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(max_depth=3, random_state=0)

RF.fit(X_train,y_train)
RF_predict = RF.predict(X_test) 

metrics.adjusted_rand_score(y_test,RF_predict)
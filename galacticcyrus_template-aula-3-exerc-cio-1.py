from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

dfEleitorado = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
dfEleitorado.dataframeName = 'BR_eleitorado_2016_municipio.csv'
dfEleitorado
mask = (dfEleitorado.uf == 'BA')
x = dfEleitorado[mask]
x = x.sort_values(by='total_eleitores', ascending=False)
x.head(15)

plt.violinplot(x.head(15).gen_feminino,showmeans=False, showmedians=True) #default
plt.xticks([0,1,2], ('', 'Feminino',''))
plt.show()
plt.violinplot(x.head(15).gen_masculino,showmeans=False, showmedians=True) #default
plt.xticks([0,1,2], ('', 'Masculino',''))
plt.show()

mask2 = (dfEleitorado.uf == 'MG')
y = dfEleitorado[mask2]
y = y.sort_values(by='total_eleitores', ascending=False)
y.head(15)
plt.violinplot(y.head(15).f_18_20,showmeans=False, showmedians=True) #default
plt.xticks([0,1,2], ('', '18 a 20 anos',''))
plt.show()
plt.violinplot(y.head(15).f_21_24,showmeans=False, showmedians=True) #default
plt.xticks([0,1,2], ('', '21 a 24 anos',''))
plt.show()
plt.violinplot(y.head(15).f_70_79,showmeans=False, showmedians=True) #default
plt.xticks([0,1,2], ('', '70 a 79 anos',''))
plt.show()
plt.violinplot(y.head(15).f_sup_79,showmeans=False, showmedians=True) #default
plt.xticks([0,1,2], ('', 'Sup a 79 anos',''))
plt.show()
fig, ax = plt.subplots()
ax.set_title('Masculino e Feminino')
ax.boxplot([x.head(15).gen_feminino, x.head(15).gen_masculino])

fig2, ax2 = plt.subplots()
ax2.set_title('Idades')
ax2.boxplot([y.head(15).f_18_20, y.head(15).f_21_24, y.head(15).f_70_79, y.head(15).f_sup_79])
#testes removendo os 3 primeiros
x[3:].head(12).gen_feminino

fig = {
    "data": [
        {
            "type": 'violin',
            "y": x[3:].head(12).gen_masculino,
            "legendgroup": 'M',
            "scalegroup": 'M',
            "name": 'M',
            "box": {
                "visible": False
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        },
        {
            "type": 'violin',
            "y": x[3:].head(12).gen_feminino,
            "name": 'F',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'pink'
            }
        }
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}
py.iplot(fig, filename = 'violin/grouped', validate = False)

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
df = pd.read_csv('/kaggle/input/the-human-freedom-index/hfi_cc_2019.csv')
df.head()
df.info()
df.dtypes
columns = []
for i in df:
    columns.append(i)

print(columns)
df = df.replace('-', np.nan)
df = df.replace(' ', np.nan)
df.head()
import scipy
from scipy import stats

for i in df.columns:
    df[i] = df[i].replace(np.nan, scipy.stats.mode(df[i])[0][0])
    
df.head()
lista_cuantitativos = []
for i in df.iloc[0].values:
    try:
        if type(float(i)) == float:
            lista_cuantitativos.append('cuantitativo')
            #df[i] = df[i].astype('float')
    except:
            lista_cuantitativos.append('cualitativo')
            #df[i] = df[i].astype('category')

df.dtypes

print(lista_cuantitativos)
for i,j in zip(lista_cuantitativos,df.columns):
    if i == "cuantitativo":
        df[j] = df[j].astype("float")
    else:
        df[j]= df[j].astype("category")
        
df.dtypes
#np.max(np.array([9,7,5,4,2]))- np.min(np.array([9,7,5,4,2]))

diccionario_rangos = {}

for i in df.columns:
    if df[i].dtype == 'float64':
        diccionario_rangos[i] = np.max(df[i]) - np.min(df[i])
        
diccionario_rangos
diccionario_medias = {}

for i in df.columns:
    if df[i].dtype == 'float64':
        diccionario_medias[i] =np.mean(df[i])
        
diccionario_medias
diccionario_sd = {}

for i in df.columns:
    if df[i].dtype == 'float64':
        diccionario_sd[i] =np.std(df[i])
        
diccionario_sd
diccionario_var = {}

for i in df.columns:
    if df[i].dtype == 'float64':
        diccionario_var[i] =np.var(df[i])
        
diccionario_var
# skewness
from scipy.stats import skew 

diccionario_sk = {}
for i in df.columns:
    if df[i].dtype == 'float64':
        diccionario_sk[i] =skew(df[i])
        
diccionario_sk


# kurtosis
from scipy.stats import kurtosis 

diccionario_ku = {}
for i in df.columns:
    if df[i].dtype == 'float64':
        diccionario_ku[i] = kurtosis(df[i])
        
diccionario_ku
# porcentaje de datos entre sd y 2sd
diccionario_per_1sd = {}
for i in df.columns:
    if df[i].dtype == 'float64':
        diccionario_per_1sd[i] = sum(df[i].between(\
                                                (diccionario_medias[i] - diccionario_sd[i]),\
                                                (diccionario_medias[i] + diccionario_sd[i]))) / len(df[i])
        
diccionario_per_1sd
# porcentaje de datos entre sd y 2sd
diccionario_per_2sd = {}
for i in df.columns:
    if df[i].dtype == 'float64':
        diccionario_per_2sd[i] = sum(df[i].between(\
                                                (diccionario_medias[i] - 2 * diccionario_sd[i]),\
                                                (diccionario_medias[i] + 2 * diccionario_sd[i]))) / len(df[i])
        
diccionario_per_2sd
import seaborn as sns
import matplotlib.pyplot as plt
#graficar las distribuciones de 5 atributos 'hf_score', 'pf_ss_women_inheritance_daughters', 'pf_ss_homicide', 'pf_rol_procedural', 'pf_rank'

# hf_score

sns.distplot(df["hf_score"],hist=False, color="red")
#desviación positiva 0.311753 con cola izda, particurtica -0.5735679 más plana

sd_hfscore = diccionario_sd['hf_score']
sd2_hfscore = sd_hfscore * 2
avg_hfscore = diccionario_medias['hf_score']


plt.plot([avg_hfscore, avg_hfscore], [0, 1], color = 'blue' )

plt.plot([avg_hfscore-sd2_hfscore, avg_hfscore-sd2_hfscore], [0, 1], color = 'orange' )
plt.plot([avg_hfscore+sd2_hfscore, avg_hfscore+sd2_hfscore], [0, 1], color = 'orange' )
plt.plot([avg_hfscore-sd_hfscore, avg_hfscore-sd_hfscore], [0, 1], color = 'orange' )
plt.plot([avg_hfscore+sd_hfscore, avg_hfscore+sd_hfscore], [0, 1], color = 'orange' )


title_string = 'hf_score, ' + \
        ('\n skewness positive, ' if diccionario_sk['hf_score'] > 0 else 'skewness negative, ') + \
        ('\n kurtosis positive' if diccionario_ku['hf_score'] > 0 else 'kurtosis negative')
        
plt.title(title_string)

print(diccionario_sk['hf_score'])
print(diccionario_ku['hf_score'])

# pf_ss_women_inheritance_daughters

sns.distplot(df["pf_ss_women_inheritance_daughters"],hist=False, color="red")
#desviación positiva 0.311753 con cola izda, particurtica -0.5735679 más plana

sd_wom = diccionario_sd['pf_ss_women_inheritance_daughters']
sd2_wom = sd_wom * 2
avg_wom = diccionario_medias['pf_ss_women_inheritance_daughters']


plt.plot([avg_wom, avg_wom], [0, 1], color = 'blue' )

plt.plot([avg_wom-sd2_wom, avg_wom-sd2_wom], [0, 1], color = 'orange' )
plt.plot([avg_wom+sd2_wom, avg_wom+sd2_wom], [0, 1], color = 'orange' )
plt.plot([avg_wom-sd_wom, avg_wom-sd_wom], [0, 1], color = 'orange' )
plt.plot([avg_wom+sd_wom, avg_wom+sd_wom], [0, 1], color = 'orange' )

title_string = 'pf_ss_women_inheritance_daughters, ' + \
        ('\n skewness positive, ' if diccionario_sk['pf_ss_women_inheritance_daughters'] > 0 else 'skewness negative, ') + \
        ('\n kurtosis positive' if diccionario_ku['pf_ss_women_inheritance_daughters'] > 0 else 'kurtosis negative')
        
plt.title(title_string)

print(diccionario_sk['pf_ss_women_inheritance_daughters'])
print(diccionario_ku['pf_ss_women_inheritance_daughters'])

# pf_ss_homicide

sns.distplot(df["pf_ss_homicide"],hist=False, color="red")
#desviación positiva 0.311753 con cola izda, particurtica -0.5735679 más plana

sd_hom = diccionario_sd['pf_ss_homicide']
sd2_hom = sd_hom * 2
avg_hom = diccionario_medias['pf_ss_homicide']


plt.plot([avg_hom, avg_hom], [0, 1], color = 'blue' )

plt.plot([avg_hom-sd2_hom, avg_hom-sd2_hom], [0, 1], color = 'orange' )
plt.plot([avg_hom+sd2_hom, avg_hom+sd2_hom], [0, 1], color = 'orange' )
plt.plot([avg_hom-sd_hom, avg_hom-sd_hom], [0, 1], color = 'orange' )
plt.plot([avg_hom+sd_hom, avg_hom+sd_hom], [0, 1], color = 'orange' )

title_string = 'pf_ss_homicide, ' + \
        ('\n skewness positive, ' if diccionario_sk['pf_ss_homicide'] > 0 else 'skewness negative, ') + \
        ('\n kurtosis positive' if diccionario_ku['pf_ss_homicide'] > 0 else 'kurtosis negative')
        
plt.title(title_string)

print(diccionario_sk['pf_ss_homicide'])
print(diccionario_ku['pf_ss_homicide'])

# pf_rol_procedural

sns.distplot(df["pf_rol_procedural"],hist=False, color="red")
#desviación positiva 0.311753 con cola izda, particurtica -0.5735679 más plana

sd_proc = diccionario_sd['pf_rol_procedural']
sd2_proc = sd_proc * 2
avg_proc = diccionario_medias['pf_rol_procedural']


plt.plot([avg_proc, avg_proc], [0, 1], color = 'blue' )

plt.plot([avg_proc-sd2_proc, avg_proc-sd2_proc], [0, 1], color = 'orange' )
plt.plot([avg_proc+sd2_proc, avg_proc+sd2_proc], [0, 1], color = 'orange' )
plt.plot([avg_proc-sd_proc, avg_proc-sd_proc], [0, 1], color = 'orange' )
plt.plot([avg_proc+sd_proc, avg_proc+sd_proc], [0, 1], color = 'orange' )

title_string = 'pf_rol_procedural, ' + \
        ('\n skewness positive, ' if diccionario_sk['pf_rol_procedural'] > 0 else 'skewness negative, ') + \
        ('\n kurtosis positive' if diccionario_ku['pf_rol_procedural'] > 0 else 'kurtosis negative')
        
plt.title(title_string)

print(diccionario_medias['pf_rol_procedural'])
print(diccionario_sd['pf_rol_procedural'])
print(diccionario_sk['pf_rol_procedural'])
print(diccionario_ku['pf_rol_procedural'])
# pf_rank

sns.distplot(df["pf_rank"],hist=False, color="red")
#desviación positiva 0.311753 con cola izda, particurtica -0.5735679 más plana

sd_proc = diccionario_sd['pf_rank']
sd2_proc = sd_proc * 2
avg_proc = diccionario_medias['pf_rank']


plt.plot([avg_proc, avg_proc], [0, 0.1], color = 'blue' )

plt.plot([avg_proc-sd2_proc, avg_proc-sd2_proc], [0, 0.1], color = 'orange' )
plt.plot([avg_proc+sd2_proc, avg_proc+sd2_proc], [0, 0.1], color = 'orange' )
plt.plot([avg_proc-sd_proc, avg_proc-sd_proc], [0, 0.1], color = 'orange' )
plt.plot([avg_proc+sd_proc, avg_proc+sd_proc], [0, 0.1], color = 'orange' )

title_string = 'pf_rank, ' + \
        ('\n skewness positive, ' if diccionario_sk['pf_rank'] > 0 else 'skewness negative, ') + \
        ('\n kurtosis positive' if diccionario_ku['pf_rank'] > 0 else 'kurtosis negative')
        
plt.title(title_string)

print(diccionario_medias['pf_rank'])
print(diccionario_sd['pf_rank'])
print(diccionario_sk['pf_rank'])
print(diccionario_ku['pf_rank'])
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pv = pd.read_csv('/kaggle/input/curar-datos/PreguntasVsVariables.csv')

pv.head(40)
cv = pd.read_csv('/kaggle/input/curar-datos/ClasificaVariables.csv')

cv.fillna(' ')
dv = pd.read_csv('/kaggle/input/curar-datos/DescripcionVariables.csv')

dv.fillna(' ')
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import statsmodels.api as sm



#importar directorio común para el notebook



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Análisis de una variable 

import pandas as pd

dataset = pd.read_csv('/kaggle/input/mortalidad13/mortalidad_depurada_12.csv')

#dataset = pd.read_csv('/kaggle/input/mortalidad-perinatal-2/Mortalidad_Perinatal_2010_A_2016.csv')

dataset.describe()
dataset.info()
dataset.head()
100 * dataset['A_DEFUN'].value_counts() / len(dataset['A_DEFUN'])
plot = (100 * dataset['A_DEFUN'].value_counts() / len(dataset['A_DEFUN'])).plot(

kind='bar', title='Jurisdicción Territorial - Defunción')
100 * dataset['SIT_DEFUN'].value_counts() / len(dataset['SIT_DEFUN'])
plot = (100 * dataset['SIT_DEFUN'].value_counts() / len(dataset['SIT_DEFUN'])).plot(

kind='bar', title='Sitio de defunción')
100 * dataset['TIPO_DEFUN'].value_counts() / len(dataset['TIPO_DEFUN'])
plot = (100 * dataset['TIPO_DEFUN'].value_counts() / len(dataset['TIPO_DEFUN'])).plot(

kind='bar', title='Tipo de defunción')
100 * dataset['ANO'].value_counts() / len(dataset['ANO'])
plot = (100 * dataset['ANO'].value_counts() / len(dataset['ANO'])).plot(

kind='bar', title='Año de defunción')
100 * dataset['MES'].value_counts() / len(dataset['MES'])
plot = (100 * dataset['MES'].value_counts() / len(dataset['MES'])).plot(

kind='bar', title='Mes de defunción')
100 * dataset['SEXO'].value_counts() / len(dataset['SEXO'])
plot = (100 * dataset['SEXO'].value_counts() / len(dataset['SEXO'])).plot(

kind='bar', title='Genero del Fallecido')
#Análisis de una variable 



dataset2 = pd.read_csv('/kaggle/input/mortalidad-perinatal1/osb_mortalidad_perinatal_edit1.csv')

dataset2.describe()
dataset2.info()
100 * dataset2['Localidad'].value_counts() / len(dataset2['Localidad'])
plot = (100 * dataset2['Localidad'].value_counts() / len(dataset2['Localidad'])).plot(

kind='bar', title='Localidad')


100 * dataset2['Ano'].value_counts() / len(dataset2['Ano'])
plot = (100 * dataset2['Ano'].value_counts() / len(dataset2['Ano'])).plot(

kind='bar', title='Año')
import pandas as pd

import matplotlib.pyplot as plt

import  numpy as np 

import scipy.stats as sp

phh = pd.read_csv('/kaggle/input/mortalidad13/mortalidad_depurada_12.csv')

pg= phh[['A_DEFUN','SIT_DEFUN','TIPO_DEFUN','SEXO','EDAD','AREA_RES','SEG_SOCIAL','PMAN_MUER','MU_PARTO','T_PARTO','TIPO_EMB','T_GES','PESO_NAC','EDAD_MADRE','N_HIJOSV','N_HIJOSM','EST_CIVM','NIV_EDUM']]

pg.corr(method="pearson")
plt.matshow(pg.corr())
phh.columns
feature_cols = [ u'A_DEFUN',u'SIT_DEFUN',u'TIPO_DEFUN',u'SEXO',u'EDAD',u'AREA_RES',u'SEG_SOCIAL',u'PMAN_MUER',u'MU_PARTO',u'T_PARTO',u'TIPO_EMB',u'T_GES',u'PESO_NAC',u'EDAD_MADRE',u'N_HIJOSV',u'N_HIJOSM',u'EST_CIVM',u'NIV_EDUM']

x = phh[feature_cols]

y = phh['N_HIJOSM']
import seaborn as sns

%matplotlib inline



sns.pairplot(phh,x_vars=feature_cols,y_vars="N_HIJOSM",size=7,aspect=0.7,kind = 'reg')
def compute_freq_chi2(x,y):



    freqtab = pd.crosstab(x,y)

    print("Frequency table")

    print("============================")

    print(freqtab)

    print("============================")

    chi2,pval,dof,expected = sp.chi2_contingency(freqtab)

    print("ChiSquare test statistic: ",chi2)

    print("p-value: ",pval)

    return



compute_freq_chi2(pg.EDAD_MADRE, pg.N_HIJOSM)

import pandas as pd

import matplotlib.pyplot as plt

import  numpy as np 

import scipy.stats as sp

phh = pd.read_csv('/kaggle/input/mortalidad/mortalidad_depurada_1.csv')

pg= phh[['ANO', 'NIV_EDUM', 'AREA_RES', 'T_PARTO', 'T_GES', 'PESO_NAC', 'EDAD_MADRE', 'N_HIJOSV', 'N_HIJOSM']]



from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

pg = pg.copy()

pg['NIV_EDUM'] = lb.fit_transform(pg['NIV_EDUM'].astype(str))

pg['AREA_RES'] = lb.fit_transform(pg['AREA_RES'].astype(str)) 

pg['T_PARTO'] = lb.fit_transform(pg['T_PARTO'].astype(str))

from sklearn.model_selection import train_test_split

train,test = train_test_split(pg, random_state=3)

x_train = train[['T_GES', 'PESO_NAC', 'T_PARTO', 'AREA_RES', 'EDAD_MADRE', 'NIV_EDUM']]

y_train = train[['N_HIJOSM']]

x_test = test[['T_GES', 'PESO_NAC', 'T_PARTO', 'AREA_RES', 'EDAD_MADRE', 'NIV_EDUM']]

y_test = test[['N_HIJOSM']]



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

model = DecisionTreeClassifier(criterion='gini')

model.fit(x_train, y_train)

model.score(x_test, y_test)
predicted= model.predict(x_test)

predicted
from sklearn.model_selection import train_test_split

train,test = train_test_split(pg, random_state=3)

x_train = train[['T_GES', 'PESO_NAC', 'T_PARTO', 'AREA_RES', 'EDAD_MADRE', 'NIV_EDUM']]

y_train = train[['N_HIJOSV']]

x_test = test[['T_GES', 'PESO_NAC', 'T_PARTO', 'AREA_RES', 'EDAD_MADRE', 'NIV_EDUM']]

y_test = test[['N_HIJOSV']]
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

model = DecisionTreeClassifier(criterion='gini')

model.fit(x_train, y_train)

model.score(x_test, y_test)
predicted= model.predict(x_test)

predicted
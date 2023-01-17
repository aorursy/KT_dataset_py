# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/ufcdata/data.csv")

preprocessed_data = pd.read_csv("../input/ufcdata/preprocessed_data.csv")

raw_fighter_details = pd.read_csv("../input/ufcdata/raw_fighter_details.csv")

raw_total_fight_data = pd.read_csv("../input/ufcdata/raw_total_fight_data.csv")
df.describe()
df.isnull().sum() #Chequeando por missing values
df2 = df[df.columns[df.columns.isin(['R_fighter','B_fighter','Referee','Winner','weight_class','no_of_rounds','title_bout','R_weight_lbs','R_Reach_cms','R_Height_cms','R_Stance','B_age','R_age'])]]

#Filtrando la data
df2.head() #Viendo la nueva data
df2.dropna(axis = 'rows', inplace = True ) # Dropping missing values, para un mejor analisis. Aun queda gran parte de la muestra por lo que se puede asumir que sigue siendo representativa.
df2['B_age'].describe() #Estadisticas descriptivas de la edad del luchador en la esquina azul

df2['B_age'].describe() #Estadisticas descriptivas de la edad del luchador en la esquina roja
Azul = df.groupby(['B_age']).count()['Winner']

BAge = Azul.sort_values(axis=0, ascending = False)

BAge.head(3)

#Para ver la edad mas ganadora de la esquina azul
Rojo = df.groupby(['R_age']).count()['Winner']

RAge = Rojo.sort_values(axis=0, ascending = False)

RAge.head(3)

#Edad mas ganadora de la esquina roja
sns.distplot(df2['B_age'])

sns.distplot(df2['R_age'])

#Distribucion de las edades
sns.heatmap(df2.corr()) #grafico de correlacion

df2.corr(method ='pearson') #tabla de correlacion
((df[df["title_bout"]==1]["Referee"].value_counts())) #Cuenta los referees que fueron a las peleas por el titulo
((df[df["title_bout"]==1]["Winner"].value_counts())) #numero de ganadores por color
sns.countplot(x='Winner',data=df)

plt.title('Quien gana más?',fontsize=15) #grafica de los ganadores por color de esquina
((df[df["title_bout"]==1]["weight_class"].value_counts())) #Contar las peleas de titulo por peso
sns.countplot(x='title_bout',data=df)

plt.title('Clasificacion de peleas',fontsize=15)

#Que tantas peleas fueron por el titulo
sns.countplot(x = 'Winner', data = df2, hue = 'R_Stance')
x= df2[['R_age','B_age','title_bout','R_Reach_cms','R_Height_cms']]

y = df2['no_of_rounds']

#Creando las variables
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4,

                                                   random_state = 101)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4,

                                                   random_state = 101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
#Evaluacion del modelo 

print(lm.intercept_)
coef = pd.DataFrame(lm.coef_, x.columns, columns=['Coeficientes'])

coef #Coeficientes del modelo 
predict = lm.predict(x_test)
sns.distplot((y_test - predict),bins = 30)
from sklearn import metrics 
print ('MAE: ', metrics.mean_absolute_error(y_test,predict))

print ('MSE: ', metrics.mean_squared_error(y_test,predict))

print ('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,predict)))
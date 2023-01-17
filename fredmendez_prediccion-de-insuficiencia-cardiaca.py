#cargar datos desde drive acceso libre

FILEID = "1Fx8V-8KUcatpoR7VQBlr-rHJNlXnfKIT"

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILEID -O codigos.zip && rm -rf /tmp/cookies.txt

!unzip codigos.zip

!dir
# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)











# To plot pretty figures

# magic function to render figure

%matplotlib inline 

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



from sklearn.impute import SimpleImputer 

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, cross_val_predict

from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline



#crear carpeta con resultados

import os

try:

  os.mkdir('results')

except:

  print("Carpeta results ya existe")



import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

import plotly.figure_factory as ff

import plotly.graph_objs as go

import plotly.express as px

import pandas as pd

%matplotlib inline

csv_analisispredictorio = 'heart_failure_clinical_records_dataset.csv'

pincardi= pd.read_csv(csv_analisispredictorio)   

pincardi.head(20) 
pincardi.info()



# paso 2: partición entrenamiento y validacion.

# tamañ0 Xtrain 70%, Tamaño Xtest 30%.

#en mi caso el analisis de datos esta adecuada para hace 

# un buen analisis,no es conveniente codificar,pero si se debe hacer particion de entrenamiento y validación.



import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import train_test_split

Xtrain, Xtest = train_test_split(pincardi,test_size=0.3)

col_smoke = "smoking"

ytrain = Xtrain[col_smoke]

ytest = Xtest[col_smoke]

Xtrain.drop(columns=col_smoke,inplace=True)

Xtest.drop(columns=col_smoke,inplace= True)









from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")

ytrain = imputer.fit_transform(pd.DataFrame(ytrain))

ytrain = ytrain.reshape(-1)



ytest = imputer.transform(pd.DataFrame(ytest))

ytest = ytest.reshape(-1)



print(ytrain.shape,ytest.shape)

#codificar la salida

#en este caso no hay necesidad de codificar,pues los datos estan

#organizados y nos ahorramos esa parte...

# 1-->fuma y 0-->no fuma

ytrain[:4]
ytest[:4]
import matplotlib.pyplot as plt

plt.hist(ytrain,bins=200)

plt.savefig('results/salida_entrenamiento.pdf', format='pdf', dpi=300)

plt.show()



plt.boxplot(ytrain)

plt.savefig('results/salida_entrenamiento2.pdf', format='pdf', dpi=300)

plt.show()



#podemos deducur que hay mas gente que no fuma,pero este dato no es SUFICIENTE PARA GENERAR UNA PRECDICCION MAS CERTERA.

#POR ESO ANALICE ESTE CASO, HACIENDO GRAFICAS DE PROBABILIDAD DESCRIPTIVA PARA ENTENDER MEJOR MIS VARIABLES DE INTERES.
'''vamos a interesarnos por la segunda opcion,pues una de la mayoria de las fallas que 

ocurre en el corazon, son a raiz del consumo del tabaco, este es uno de los

grandes problemas que afectan a las personas y puede ser de los que consumen

el tabaco tengan mas decesos de muerte que los otros,por eso lo analizaresmos.'''





fig =px.violin(pincardi, y="age", x="smoking",color="DEATH_EVENT",box= True, points="all",hover_data= pincardi.columns )



fig.update_layout(title_text = "Analisis edad y fumadores- mueren o sobreviven")

plt.savefig('results/analisis_edad_y_fumadores(mueren_o_sobreviven).pdf', format='pdf', dpi=300)

fig.show()
'''analizaremos cuantas personas fuman y no fuman.'''



fuma = pincardi[pincardi['smoking']==1]

no_fuma = pincardi[pincardi['smoking']==0]



labels = ['No fuma','fuma']

values = [len(no_fuma), len(fuma)]



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(title_text="no fuma - fuma")

plt.savefig('results/no_fuma_Vs_fuma.pdf', format='pdf', dpi=300)

fig.show()



#generraremos una Razón de eventos de muerte por tabaquismo 

fig = px.pie(pincardi, values='smoking',names='DEATH_EVENT', title='Eventos de muerte por fumar')

plt.savefig('Eventos_de_Muerte_por_Fumar.pdf', format='pdf', dpi=300)

fig.show()
#por ultimo un analisis de quienes viven y mueren por fumar o no.



smoking_yes_survi = fuma[pincardi["DEATH_EVENT"]==0]

smoking_yes_not_survi = fuma[pincardi["DEATH_EVENT"]==1]

smoking_no_survi = no_fuma[pincardi["DEATH_EVENT"]==0]

smoking_no_not_survi = no_fuma[pincardi["DEATH_EVENT"]==1]



labels = ['Fumador - Vive','Fumador - No Vive', 'No Fuma - Vive', 'No Fuma- No Vive']

values = [len(fuma[pincardi["DEATH_EVENT"]==0]),len(fuma[pincardi["DEATH_EVENT"]==1]),

         len(no_fuma[pincardi["DEATH_EVENT"]==0]),len(no_fuma[pincardi["DEATH_EVENT"]==1])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(title_text="Analisi de sobrevivencia-fumadores")

plt.savefig('results/Analisis_de_sobreviviencia-Fumadores.pdf', format='pdf', dpi=300)

fig.show()
'''Podemos concluir que en esta base de datos, hay 203 personas no fumadoras de la cuales 137 sobrevivieron y las otras 66 desafortunadamente fallecieron.



 Luego de las 96 personas que fuman, 66 sobrevivieron y 30 falleciron.'''
# definir columnas tipo string para año y deceso"no necesita decodificar", estdistica

# de pincardi y categorias.



#col_añde = ['age','DEATH_EVENT','sex','smoking']; #columna de año y deceso de una persona



#col_pincardi =['anaemia','creatinine_phosphokinase','diabetes',

#               'ejection_fraction','high_blood_pressure','platelets',

#               'serum_creatinine','serum_sodium']; # estadistica de para

                                   #prediccion de insuficiencia cardiaca.



#cat =  ['sex','smoking']  #categorias     



#items=[]



#for i in cat:

#   items += [list(pincardi[i].value_counts().index)]

#cat_usr = dict(zip(cat,items))   



'''NO HAY NECESIDAD DE CODIFICAR LAS COLUMNAS,PUES EL DATA FRAME YA ESTA 

ADECUADO PARA TRABAJARLO COMO ESTA, NO HAY NECESIDAD DE CODIFICAR.'''



#crear clase propia de proceso para prediccion de insuficiencia cardiaca



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer 

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder





class dummy_picard(BaseEstimator,TransformerMixin):

    #inicializacion de clase y varaibles

    def __init__(self, col_añde,col_pincardi,): #constructor clase

        self.col_añde = col_añde #lista atributos tipo moneda

        self.col_pincardi = col_pincardi #lista atributos tipo estadistica

        #self.cat_usr = cat_usr #lista de atributos categoricos



    def fit(self,X, *_):

        Xi = X.copy() #copiar dataset para no reemplazar original

        self.imputer_num = SimpleImputer(strategy="most_frequent") #crear imputador tipo modo

        self.a = Xi.columns[np.sum(Xi.isna())> 0] #encontrar columnas con datos faltantes

        #print(a)

        self.imputer_num.fit(Xi[self.a]) # ajustar imputador  



        return self

        

    def fit_transform(self,X,*_):

         self.fit(X)

         return self.transform(X)
#dummy = dummy_picard(col_añde=col_añde,col_pincardi=col_pincardi)



#xtrain_pre =dummy.fit_transform(Xtrain)

#Xtrain_pre = dummy.fit_transform(Xtrain)



'''no es necesario crear clase propia de procesos,pues esta base de datos 

ya esta bien organizada y no seria optimo volver a procesar los mismos datos para obtener

los mismos datos,asi que podemos hacer la predicción de la o las variables de 

interes apartir de estos datos.'''
Xtrain.info()
from sklearn.decomposition import PCA 

from sklearn.preprocessing import StandardScaler

sca = StandardScaler()

Xtrain_igual_z = sca.fit_transform(Xtrain)
Xtrain_igual_z.var(axis=0)
Xtrain.var(axis=0)
red = PCA()

zz = red.fit_transform(Xtrain_igual_z)
plt.scatter(zz[:,0],zz[:,1],c=ytrain,s = 100*Xtrain['DEATH_EVENT']/(Xtrain['DEATH_EVENT'].max()))

plt.colorbar()

plt.savefig('results/prediccion_De_variable(DEATH_EVENT)_respecto_a_fumadores.pdf', format='pdf', dpi=300)

plt.show()
# PROCESO CON test



Xtest['DEATH_EVENT']
Xtest.info()
zztest = red.transform(sca.transform((Xtest))) # una sola linea
plt.scatter(zz[:,0],zz[:,1],c=ytrain,s = 100*Xtrain['DEATH_EVENT']/(Xtrain['DEATH_EVENT'].max()),label='train')

plt.colorbar()

plt.scatter(zztest[:,0],zztest[:,1],c=ytest,s=100*Xtest['DEATH_EVENT']/Xtest['DEATH_EVENT'].max(),marker='d',label='test')

plt.legend()

plt.savefig('results/prediccion_De_variable(DEATH_EVENT)Xtrain_Vs_ytest_respecto_a_fumadores.pdf', format='pdf', dpi=300)

plt.show()
# definir modelos de predicción

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge , Lasso

from sklearn.kernel_ridge import KernelRidge

steps = [

         [('scaler', StandardScaler()), #estandarizar cada atributo columna de xtrain centrada en 0 y var = 1

          ('reg', LinearRegression())],

         

         [('scaler',StandardScaler()),

          ('reg',ElasticNet())],

         

         [('scaler', StandardScaler()), #estandarizar cada atriuto columna de xtrain centrada en 0 y var = 1

         ('reg',  KernelRidge(alpha=1,gamma=None))], #clasificador 

         

         [ ('scaler', StandardScaler()),

          ('reg', Ridge())],



         [ ('scaler', StandardScaler()),

          ('reg', Lasso())], 

         ]        



#parametros a buscar por busqueda por grilla

parameters =[ 

             {'reg__fit_intercept':[True, False]             

             },

             {

              'reg__alpha': [0,1e-3,1e-2,1e-1,1,10], #parametros n_neighbors debe ser siempre un int

              'reg__l1_ratio':[0,0.25,0.5,0.75,1]

             },

             {

              'reg__alpha':[0,1e-3,1e-2,1e-1,1,10],              

              'reg__gamma':[0,0.25,0.5,0.75,1] 

             },

             {

              'reg__alpha':[0,1e-3,1e-2,1e-1,1,10],

              #'reg__fit_intercept':[True, False]

              'reg__fit_intercept':[0,0.25,0.5,0.75,1] 

             },

             {

              'reg__alpha':[0,1e-3,1e-2,1e-1,1,10],

              #'reg__fit_intercept':[True, False]

              'reg__fit_intercept':[0,0.25,0.5,0.75,1]    

             }

              ]



label_model = ['Nor+RegLin','Nor+ElasticNet','Nor+KernelRidge','Nor+Ridge','Nor+Lasso']                    
parameters
import os 

pathpre = 'datospre'



try:

 os.mkdir(pathpre)

except:

  print("carpetas resultados ya existe") 
from joblib import dump, load

from sklearn.metrics import mean_absolute_error as msa 

Niter = 20 #numero particiones outter loop nested cross-validation

msev =np.zeros((Niter,len(steps)))#arreglo para guardar acierto/error

Nmod = len(steps) #numero de modelos a probar

best_estimators = Niter*[None]#mejor metodo por iteracion

###clave del funcionamiento

best_hyperpar = Niter*[None]#mejor metodo por iteracion 





#############################

for j in range(Niter): #outter loop # SI TIENE MENOS DE 1000 DATOS BORRAR ESTE CICLO SOLO CV EN LINEA 16

      #print('it %d/%d'%(j+1,Niter))

      #particiono datos outter loop

      X_trainj, X_testj, y_trainj, y_testj = train_test_split(Xtrain,ytrain,test_size=0.3) # xtrain 60, xtest 26

      list_est = [] #lista lazo interno para guardar mejor estimador por modelo para iteracion j

      list_hyper = [] #lista lazo interno para guardar mejores hyperparametros por modelo para iteracion j

      for r in range(Nmod): #recorro todos los posibles modelos a probar en iteracion j del outter loop

          grid_search = GridSearchCV(Pipeline(steps[r],memory=pathpre), parameters[r],cv=5,verbose=5,scoring='neg_mean_absolute_error',n_jobs=-1) #cv inner loop

          #xtrain gridsearchcv xtrain split en 12 / cv, 60/5 = 12, xtrain 48 datos validar 12

          # cv = N -> leave one out N <30

          #generar mejor modelo

          #grid_search.fit(X_trainj,y_trainj)

          grid_search.fit(X_trainj,y_trainj)

          #estimar salida conjunto de test

          y_pred = grid_search.best_estimator_.predict(X_testj)

          #guardar mejor modelo

          list_est.append(grid_search.best_estimator_)

          list_hyper.append(grid_search.best_params_)

          #guardar acierto

          msev[j,r] = msa(y_testj,y_pred)

          print('it %d/%d-Modelo %d/%d'%(j+1,Niter,r+1,len(steps)))

          print('best hyper', grid_search.best_params_)

          print('msa:',msev[j,r])

          

      best_estimators[j] = list_est #guardar mejores modelos 

      best_hyperpar[j] = list_hyper #mejores hyperparametros

          

          

      savedata = {

          'acc':msev,

          'best_models':best_estimators,

          'best_parameters':best_hyperpar,

            } 

      dump(savedata,'smokingpincardi.joblib')
from scipy.stats import mode

from datetime import date



r = 1 #hyperparameters model 0,model 1,model 2,model 3,model 4



nh = len(best_hyperpar[0][r])

hyperpar_r = np.zeros((Niter,nh)) 

for i in range(Niter):

  for j in range(nh):

    hyperpar_r[i,j] = best_hyperpar[i][r].get(list(best_hyperpar[i][r].keys())[j])

    

    

#revisar numero entero para realizar casting

aa = list(best_hyperpar[0][r].keys())

c = -1

for i in range(len(aa)):

    if aa[i].find('n_neighbors') > -1:

      c = i



plt.boxplot(hyperpar_r)

plt.xticks(ticks=np.arange(nh)+1,labels=list(best_hyperpar[0][r].keys()))

plt.title('Best_hyperparameters '+label_model[r])

plt.grid()

plt.savefig('results/SELECCION_MEJOR_MODELO.pdf', format='pdf', dpi=300)

plt.show()



mode_hyper = mode(hyperpar_r,axis=0)[0][0]

print("Modes= ", mode_hyper)
r = 1 #camino elastic net

steps_final = [('scaler',StandardScaler()),

               ('reg',ElasticNet(alpha=0.01,l1_ratio=0))]  

         

modelo_final = Pipeline(steps_final)

modelo_final.fit(Xtrain,ytrain)
ytest_e = modelo_final.predict(Xtest) #simular casos nuevos
print('MAE_test=', msa(ytest,ytest_e))
pathpre = 'resultados'



try:

  os.mkdir(pathpre)

except:

  print("Carpeta results ya existe")
import shutil

from joblib import dump, load

from datetime import date, datetime

from google.colab import files



modelo_final = {'modelo':modelo_final,

          'pasos':label_model[1],

          'mae_test':msa(ytest,ytest_e),

            } 



dump(modelo_final,'resultados/modelo_final.joblib')

namefile = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%d"))+'__modelo'

shutil.make_archive(namefile, 'zip', 'resultados')

files.download(namefile+'.zip')
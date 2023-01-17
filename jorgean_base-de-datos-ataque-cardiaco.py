#cargar datos desde drive acceso libre
FILEID = "1K73mni1gZXVctsNPGEGI00TAYdYZ5sb9"
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILEID -O codigos.zip && rm -rf /tmp/cookies.txt
!unzip codigos.zip
!dir
#Paso 1: Lectura
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go

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



import pandas as pd
%matplotlib inline
csv_path = 'heart_failure_clinical_records_dataset.csv'

Xdata = pd.read_csv(csv_path)

col_drop = ['time' ]
Xdata.drop(columns = col_drop, inplace = True)
Xdata.head(15)
Xdata.info()
Xdata["DEATH_EVENT"].describe() # estadística básica de las variable de interés
# Tamaño Xtrain 70%, Tamaño Xtest 30%
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
Xtrain, Xtest = train_test_split(Xdata,test_size=0.3)
col_DEATH = "DEATH_EVENT"
ytrain = Xtrain[col_DEATH]
ytest = Xtest[col_DEATH]
Xtrain.drop(columns=col_DEATH,inplace=True)
Xtest.drop(columns=col_DEATH,inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="most_frequent")
ytrain = imputer.fit_transform(pd.DataFrame(ytrain))
ytrain = ytrain.reshape(-1)

ytest = imputer.transform(pd.DataFrame(ytest))
ytest = ytest.reshape(-1)

print(ytrain.shape, ytest.shape)
# Entrenamiento
# DEATH EVENT -> 0 = No, 1 = Yes
ytrain[:20]
import matplotlib.pyplot as plt
plt.hist(ytrain,bins = 100)
plt.show()

plt.boxplot(ytrain)
plt.savefig('results/out_train.pdf', format='pdf', dpi=300)
plt.show()
male = Xdata[Xdata["sex"]==1]
female = Xdata[Xdata["sex"]==0]

male_survi = male[Xdata["DEATH_EVENT"]==0]
male_not = male[Xdata["DEATH_EVENT"]==1]
female_survi = female[Xdata["DEATH_EVENT"]==0]
female_not = female[Xdata["DEATH_EVENT"]==1]

labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]
values = [len(male[Xdata["DEATH_EVENT"]==0]),len(male[Xdata["DEATH_EVENT"]==1]),
         len(female[Xdata["DEATH_EVENT"]==0]),len(female[Xdata["DEATH_EVENT"]==1])]
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
fig.update_layout(
    title_text="Analysis on Survival - Gender")
plt.savefig('results/Analysis_on_Survival.pdf', format='pdf', dpi=300)
fig.show()
surv = Xdata[Xdata["DEATH_EVENT"]==0]["age"]
not_surv = Xdata[Xdata["DEATH_EVENT"]==1]["age"]
hist_data = [surv,not_surv]
group_labels = ['Survived', 'Not Survived']
fig = ff.create_distplot(hist_data, group_labels, bin_size=0.9)
fig.update_layout(
    title_text="Análisis en edad sobre estado de supervivencia")
plt.savefig('results/Analysis_on_Survival.pdf', format='pdf', dpi=300)
fig.show()
fig = px.violin(Xdata, y="age", x="DEATH_EVENT", color="DEATH_EVENT", box=True, points="all", hover_data=Xdata.columns)
fig.update_layout(title_text="Análisis en edad y género sobre el estado de supervivencia")
plt.savefig('results/Analysis_of_sex_and_DEATH_EVENT.pdf', format='pdf', dpi=300)
fig.show()
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
Xtrain.info()
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
sca = StandardScaler()
Xtrain_pre_z = sca.fit_transform(Xtrain)
Xtrain_pre_z.var(axis=0)
Xtrain.var(axis=0)
red = PCA()
zz = red.fit_transform(Xtrain_pre_z)
plt.scatter(zz[:,0],zz[:,1],c=ytrain,s = 100*Xtrain['age']/(Xtrain['age'].max()))
plt.colorbar()
plt.savefig('results/Xtrain.pdf', format='pdf', dpi=300)
plt.show()
# proceso con test
Xtest['age']
Xtest.info()
zztest = red.transform(sca.transform(Xtest)) # una sola linea
plt.scatter(zz[:,0],zz[:,1],c=ytrain,s = 100*Xtrain['age']/(Xtrain['age'].max()),label='train')
plt.colorbar()
plt.scatter(zztest[:,0],zztest[:,1],c=ytest,s=100*Xtest['age']/Xtest['age'].max(),marker='d',label='test')
plt.legend()
plt.savefig('results/Xtest.pdf', format='pdf', dpi=300)
plt.show()
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge

steps = [
         [('scaler', StandardScaler()), #estandarizar cada atributo columna de xtrain centrada en 0 y var = 1
          ('reg', LinearRegression())],
         
         [('scaler',StandardScaler()),
          ('reg',ElasticNet())],
         
         [('scaler', StandardScaler()), #estandarizar cada atriuto columna de xtrain centrada en 0 y var = 1
          ('reg',  KernelRidge(alpha=1, gamma=None ))], #clasificador kernel = 'rbf'
         
         [('scaler', StandardScaler()), 
          ('reg',  Ridge())],  

         [('scaler', StandardScaler()), 
          ('reg',  Lasso())],
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
              'reg__gamma': [0,0.25,0.5,0.75,1]
             },
             {
              'reg__alpha':[0,1e-3,1e-2,1e-1,1,10],
              'reg__fit_intercept': [0,0.25,0.5,0.75,1]
             },
             {
              'reg__alpha':[0,1e-3,1e-2,1e-1,1,10],
              'reg__normalize': [0,0.25,0.5,0.75,1]
             }
              ]

label_model = ['Nor+RegLin','Nor+ElasticNet','Nor+KernelRidge','Nor+Ridge','Nor+Lasso']
parameters
import os
pathpre = 'datospre' #datos preprocesados

try:
  os.mkdir(pathpre)
except:
  print("Carpeta results ya existe")
from joblib import dump, load
from sklearn.metrics import mean_absolute_error as msa 
Niter = 10 #numero particiones outter loop nested cross-validation
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
      dump(savedata,'death.joblib')
from scipy.stats import mode
from datetime import date

r = 1 #hyperparameters model 1
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
plt.show()

mode_hyper = mode(hyperpar_r,axis=0)[0][0]
print("Modes= ", mode_hyper)
plt.savefig('results/hyperpar.pdf', format='pdf', dpi=300)

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
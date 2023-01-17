import pandas as pd

noshow = pd.read_csv('../input/medicalappointmentnoshown/KaggleV2-May-2016.csv',sep = ',')

noshow.head(3)
noshow.isna().sum()
print('--> No-show vs Alcoholism')

print(noshow.groupby(['No-show','Alcoholism'])['PatientId'].count())

print('-------------------------------------------------------------------')



print('--> No-show vs Diabetes')

print(noshow.groupby(['No-show','Diabetes'])['PatientId'].count())

print('-------------------------------------------------------------------')



print('--> No-show vs SMS_received')

print(noshow.groupby(['No-show','SMS_received'])['PatientId'].count())

print('-------------------------------------------------------------------')





print('--> No-show vs Hipertension')

print(noshow.groupby(['No-show','Hipertension'])['PatientId'].count())

print('-------------------------------------------------------------------')



print('--> No-show vs Scholarship')

print(noshow.groupby(['No-show','Scholarship'])['PatientId'].count())

print('-------------------------------------------------------------------')



print('--> No-show vs Handcap')

print(noshow.groupby(['No-show','Handcap'])['PatientId'].count())

print('-------------------------------------------------------------------')



print('--> No-show vs Gender')

print(noshow.groupby(['No-show','Gender'])['PatientId'].count())

print('-------------------------------------------------------------------')

import seaborn as sns

from matplotlib import pyplot as plt

box1 = plt.subplots()

box1 = sns.boxplot(x='No-show', y='Age', data=noshow)

box1.set_title('Boxplot do ano pela presença ou não de no show')

box1.set_xlabel('Paciente teve no show?')

box1.set_ylabel('Idade')

plt.show()
grafico = sns.FacetGrid(noshow, col='No-show')

grafico.map(sns.distplot, 'Age', rug=True)

plt.show()
#Filtrando pessoas com idade negativa

noshow[noshow['Age']<0]
#capturando as mediana das pessoas com desfecho dnegativo

median_noshow_no = noshow[noshow['No-show']=='No']['Age'].median()

median_noshow_no
import numpy as np

noshow['Age'] = np.where(noshow['Age']<1,median_noshow_no,noshow['Age'])

print('No-show = Não com ajuste')

print('----------------------------')

print(noshow.loc[noshow['No-show'] == 'No','Age'].describe()) #nova descrição de idade
conditions  = [ noshow['Age'] < 10

               , (noshow['Age'] < 20) & (noshow['Age']>= 10)

               , (noshow['Age'] < 30) & (noshow['Age']>= 20)

               , (noshow['Age'] < 40) & (noshow['Age']>= 30)

               , (noshow['Age'] < 50) & (noshow['Age']>= 40)

               , (noshow['Age'] < 60) & (noshow['Age']>= 50)

               , (noshow['Age'] < 70) & (noshow['Age']>= 60)

               ,  noshow['Age'] >= 70 ]



choices     = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','>70']



noshow['fx_etaria'] = np.select(conditions, choices, default=np.nan)
noshow['ScheduledDay'] = pd.to_datetime(noshow['ScheduledDay'])

noshow['AppointmentDay'] = pd.to_datetime(noshow['AppointmentDay'])



#variaveis relacionada a scheduled

noshow['Scheduled_Month'] = noshow['ScheduledDay'].apply(lambda x: x.month)

noshow['Scheduled_Year'] = noshow['ScheduledDay'].apply(lambda x: x.year)

noshow['Scheduled_WeekDay'] = noshow['ScheduledDay'].apply(lambda x: x.strftime("%A"))



#variaveis reacionadas ao appointment

noshow['Appointment_Month'] = noshow['AppointmentDay'].apply(lambda x: x.month)

noshow['Appointment_Year'] = noshow['AppointmentDay'].apply(lambda x: x.year)

noshow['Appointment_WeekDay'] = noshow['AppointmentDay'].apply(lambda x: x.strftime("%A"))



#Diferenca entre datas

noshow['DeltaScheduleAppointment_Days'] = noshow['ScheduledDay']-noshow['AppointmentDay']

noshow['DeltaScheduleAppointment_Days'] = noshow['DeltaScheduleAppointment_Days']/np.timedelta64(1,'D')



#tratando dados diferenca negativa

noshow['DeltaScheduleAppointment_Days'] = np.where(noshow['DeltaScheduleAppointment_Days'] < 0 

                                                   ,0,noshow['DeltaScheduleAppointment_Days'] )

from category_encoders.one_hot import OneHotEncoder

noshow_bin = noshow



binarizar = OneHotEncoder(cols= ['Gender','Handcap','Appointment_WeekDay','Scheduled_WeekDay','fx_etaria'],use_cat_names=True)

binarizar.fit(noshow_bin)

noshow_bin = binarizar.transform(noshow_bin)



noshow_bin.head()
noshow_bin.reset_index()



from sklearn import preprocessing

padronizar = preprocessing.StandardScaler().fit(noshow_bin[['Age','Scheduled_Month','Scheduled_Year'

                                                            ,'Appointment_Month','Appointment_Year'

                                                            ,'DeltaScheduleAppointment_Days']])



noshow_bin[['Age','Scheduled_Month','Scheduled_Year','Appointment_Month','Appointment_Year'

,'DeltaScheduleAppointment_Days']] = padronizar.transform(noshow_bin[['Age'

                                                                     ,'Scheduled_Month'

                                                                     ,'Scheduled_Year'

                                                                     ,'Appointment_Month'

                                                                     ,'Appointment_Year'

                                                                     ,'DeltaScheduleAppointment_Days']])
noshow_bin.columns
x = noshow_bin.loc[:,['Age','Appointment_Month', 'Appointment_Year','Scheduled_Year','DeltaScheduleAppointment_Days'

                      ,'Scheduled_Month'#numericos

                      ,'Gender_F', 'Gender_M','Scholarship','Hipertension','Diabetes', 'Alcoholism'#categoricos

                      , 'Handcap_0.0', 'Handcap_1.0', 'Handcap_2.0','Handcap_3.0', 'Handcap_4.0'

                      , 'SMS_received','Scheduled_WeekDay_Friday'

                      ,'Scheduled_WeekDay_Wednesday', 'Scheduled_WeekDay_Tuesday'

                      ,'Scheduled_WeekDay_Thursday', 'Scheduled_WeekDay_Monday'

                      ,'Scheduled_WeekDay_Saturday','Appointment_WeekDay_Friday'

                      , 'Appointment_WeekDay_Tuesday','Appointment_WeekDay_Monday'

                      , 'Appointment_WeekDay_Wednesday','Appointment_WeekDay_Thursday'

                      , 'Appointment_WeekDay_Saturday','fx_etaria_60-70', 'fx_etaria_50-60'

                      , 'fx_etaria_0-10', 'fx_etaria_>70','fx_etaria_20-30', 'fx_etaria_30-40'

                      , 'fx_etaria_10-20','fx_etaria_40-50']]

y = noshow_bin.loc[:,'No-show']
from sklearn.preprocessing import LabelEncoder



x[['Gender_F', 'Gender_M','Scholarship','Hipertension','Diabetes', 'Alcoholism'

          , 'Handcap_0.0', 'Handcap_1.0', 'Handcap_2.0','Handcap_3.0', 'Handcap_4.0'

          , 'SMS_received','Scheduled_WeekDay_Friday'

          ,'Scheduled_WeekDay_Wednesday', 'Scheduled_WeekDay_Tuesday'

          ,'Scheduled_WeekDay_Thursday', 'Scheduled_WeekDay_Monday'

          ,'Scheduled_WeekDay_Saturday','Appointment_WeekDay_Friday'

          , 'Appointment_WeekDay_Tuesday','Appointment_WeekDay_Monday'

          , 'Appointment_WeekDay_Wednesday','Appointment_WeekDay_Thursday'

          , 'Appointment_WeekDay_Saturday','fx_etaria_60-70', 'fx_etaria_50-60'

          , 'fx_etaria_0-10', 'fx_etaria_>70','fx_etaria_20-30', 'fx_etaria_30-40'

          , 'fx_etaria_10-20','fx_etaria_40-50']].apply(LabelEncoder().fit_transform)



y = LabelEncoder().fit_transform(y)
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.3,random_state=1)
import xgboost as xgb

import matplotlib.pyplot as plt 

from sklearn.metrics import recall_score,accuracy_score,classification_report,confusion_matrix



xgboost_ = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.8, learning_rate = 0.2,

                max_depth = 7, n_estimators = 100,random_state=0)

xgboost_.fit(x_treino,y_treino)



#realização do predict

previsoes = xgboost_.predict(x_teste)



print('recall:' , recall_score(previsoes,y_teste))

print('accuracy:' , accuracy_score(previsoes,y_teste))

print('---------------------------------------------')

print(confusion_matrix(previsoes,y_teste))

print('---------------------------------------------')

print(classification_report(previsoes,y_teste))



xgb.plot_importance(xgboost_)

plt.show() 

#Visulaizando feature_importance

feature = []

for feature in zip(x_treino, xgboost_.feature_importances_):

    print(feature)
from sklearn.model_selection import GridSearchCV



parametros = [{'learning_rate':[0.01,0.1,0.2],

                'max_depth':[5,7],

                'colsample_bytree':[0.7,0.8,0.9]}]



xgboost = xgb.XGBClassifier(objective ='reg:logistic', n_estimators = 100,random_state=0)



grid_search =  GridSearchCV(xgboost,parametros,scoring='recall',cv=4,verbose=1)



grid_search.fit(x_treino,y_treino)
#Melhores hiperparametros

grid_search.best_params_
from skopt import dummy_minimize

import lightgbm as lgb   

gradient = lgb.LGBMClassifier(learning_rate=0.09955911573844406 #resultado do randomsearch

                             ,colsample_bytree=0.7472177953903952  #resultado do randomsearch

                             ,max_depth=6  #resultado do randomsearch

                             ,n_estimators=176  #resultado do randomsearch

                              ,random_state=0

                                )

gradient.fit(x_treino,y_treino)

previsoes = gradient.predict(x_teste)

print('recall:' , recall_score(previsoes,y_teste))

print('accuracy:' , accuracy_score(previsoes,y_teste))

print('---------------------------------------------')

print(confusion_matrix(previsoes,y_teste))

print('---------------------------------------------')

print(classification_report(previsoes,y_teste))
from skopt import dummy_minimize

from lightgbm import LGBMClassifier

def treinar_modelo(params):

    learning_rate = params[0]

    colsample_bytree = params[1]

    max_depth = params[2]

    n_estimators= params[3]

    

    print(params, '\n')

    

    modelo = LGBMClassifier(learning_rate=learning_rate

                         ,colsample_bytree=colsample_bytree

                         ,max_depth=max_depth

                         ,n_estimators=n_estimators,random_state = 0)

    modelo.fit(x_treino, y_treino)

    

    previsoes = modelo.predict(x_treino)

    

    return -recall_score(y_treino, previsoes,average="binary")



space = [(1e-3, 1e-1, 'log-uniform'), #learning rate

         (0.7,0.9),#colsample_bytree

         (5,9), #max_depth

         (100, 200)] #n_estimators



resultado = dummy_minimize(treinar_modelo, space, random_state=1, verbose=1, n_calls=30)
resultado.x
#Visulaizando feature_importance

feature_lgbm = []

for feature_lgbm in zip(x_treino, gradient.feature_importances_):

    print(feature_lgbm)
from sklearn.feature_selection import SelectFromModel



#Treinando o modelo usando as features mais importantes

thresholds = sorted(gradient.feature_importances_,reverse=True) #ordenando as features com mais poder 



for thresh in thresholds:

    selection = SelectFromModel(gradient, threshold=thresh, prefit=True)

    select_x_treino = selection.transform(x_treino)



    #treinando o modelo

    selection_model = lgb.LGBMClassifier(learning_rate=0.09955911573844406 #resultado do randomsearch

                             ,colsample_bytree=0.7472177953903952  #resultado do randomsearch

                             ,max_depth=6  #resultado do randomsearch

                             ,n_estimators=176  #resultado do randomsearch

                              ,random_state=0

                                )

    selection_model.fit(select_x_treino, y_treino)



    #avaliando os modelos

    select_x_teste = selection.transform(x_teste)

    y_pred = selection_model.predict(select_x_teste)

    previsoes= [round(value) for value in y_pred]

    accuracy = accuracy_score(previsoes,y_teste)

    recall = recall_score(previsoes,y_teste)

    print("Thresh=%.3f, n=%d, Recall:%.2f%%, Accuracy: %.2f%%" % (thresh, select_x_treino.shape[1]

                                                                 ,recall*100.0, accuracy*100.0))
from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import recall_score,accuracy_score,classification_report,confusion_matrix



modelo = BaggingClassifier(bootstrap=True,n_jobs = -1,n_estimators=100)

modelo.fit(x_treino,y_treino)

previsoes = modelo.predict(x_teste)

print('recall:' , recall_score(previsoes,y_teste))

print('accuracy:' , accuracy_score(previsoes,y_teste))

print('---------------------------------------------')

print(confusion_matrix(previsoes,y_teste))

print('---------------------------------------------')

print(classification_report(previsoes,y_teste))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import recall_score,accuracy_score,classification_report,confusion_matrix



modelo = RandomForestClassifier(max_depth=5, random_state=0,n_estimators=100)

modelo.fit(x_treino,y_treino)

previsoes = modelo.predict(x_teste)

print('recall:' , recall_score(previsoes,y_teste))

print('accuracy:' , accuracy_score(previsoes,y_teste))

print('---------------------------------------------')

print(confusion_matrix(previsoes,y_teste))

print('---------------------------------------------')

print(classification_report(previsoes,y_teste))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score,accuracy_score,classification_report,confusion_matrix



modelo = LogisticRegression(random_state=0,solver='liblinear',penalty='l1')

modelo.fit(x_treino,y_treino)

previsoes = modelo.predict(x_teste)

print('recall:' , recall_score(previsoes,y_teste))

print('accuracy:' , accuracy_score(previsoes,y_teste))

print('---------------------------------------------')

print(confusion_matrix(previsoes,y_teste))

print('---------------------------------------------')

print(classification_report(previsoes,y_teste))
#transformando o feature inportance do modelo LGBM em um dataframe

results=pd.DataFrame()

results['columns']=x_treino.columns

results['importances'] = gradient.feature_importances_

results.sort_values(by='importances',ascending=False,inplace=True)

results.reset_index().head(3)
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.3,random_state=1)

#Cortando as features de acordo com os modelos rodados na etapa feautre selection com lgbm

features_final = results.iloc[:31,0] #Resultado com feature selection 



#redefinindo x_treino e x_teste

x_treino = x_treino.loc[:,features_final]

x_teste = x_teste.loc[:,features_final]
#Reaplicado o algoritimo

gradient = lgb.LGBMClassifier(learning_rate=0.09955911573844406 #resultado do randomsearch

                             ,colsample_bytree=0.7472177953903952  #resultado do randomsearch

                             ,max_depth=6  #resultado do randomsearch

                             ,n_estimators=176  #resultado do randomsearch

                              ,random_state=0

                                )

gradient.fit(x_treino,y_treino)

previsoes = gradient.predict(x_teste)

print('Resultado final:')

print('----------------------------------------')

print('Dos exemplos que são noshow positivo qual a porcetangem de acerto do modelo?')

print('recall final:' , recall_score(previsoes,y_teste)*100)

print('----------------------------------------')

print('Porcentagem de acerto da minhas observações?')

print('accuracy final:', accuracy_score(previsoes,y_teste)*100)

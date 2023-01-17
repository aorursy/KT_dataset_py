import pandas as pd
import numpy as np
import random

exemplo=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
data=pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')
dataTest=pd.read_csv('/kaggle/input/titanic/test.csv',index_col='PassengerId')
data
data.describe()
#Proporción de supervivintes en train
rowsData,_=data.shape
prop=data.Survived.sum()/rowsData
#Proporción de supervivintes en test
rows,_=dataTest.shape
superV=int(prop*rows)
superV,rows
#Seleccion superV posicións dentro de rows
lista=[]
while lista.__len__()<superV:
    x=random.randrange(rows)
    if not x in lista:
        lista.append(x)
lista.sort()
lista.__len__()

# Poñemos un 1 en cada posición de lista e 0 na que non
# Así obtemos a mesma proporción de supervivintes
def getListaSup(lista,rows):
    listaSuperv=[]
    for x in range(rows):
        if x in lista:
            listaSuperv.append(1)
        else:
            listaSuperv.append(0)
    return listaSuperv
listaSuperv=getListaSup(lista,rows)
#submit.rename(columns={index:"PassengerId"},inplace=True)
submit=pd.DataFrame({'Survived':listaSuperv},index=dataTest.index)
submit
submit.Survived.sum()
#submit.to_csv('/kaggle/input/titanic/randomSubmit.csv')
dataSex=data.groupby(['Sex','Survived'])
dataSex['Survived'].count()

total_female=dataSex['Survived'].count()['female'].sum()
total_male=dataSex['Survived'].count()['male'].sum()
surv_female=dataSex['Survived'].count()['female'][1]
surv_male=dataSex['Survived'].count()['male'][1]
prob_surv_cond_female=surv_female/total_female
prob_surv_cond_male=surv_male/total_male

prob_surv_cond_female, prob_surv_cond_male


dataSex['Survived'].count()
total_survived_female=dataSex['Survived'].count()[1]
total_survived_male=dataSex['Survived'].count()[3]
prob_female_cond_survived=total_survived_female/data.Survived.sum()
prob_male_cond_survived=total_survived_male/data.Survived.sum()
prob_female_cond_survived,prob_male_cond_survived
x=random.randrange(rows)
dataTest.iloc[3].Sex[0]
lista=['m','m','m','f','f','m','f','m','f','f','m']
lista.count('f')
#Seleccion superV posicións dentro de rows
#Selecciono homes e mulleres na proporción  prob_female_cond_survived
lista=[]
listaSex=[]
i=0
while ((lista.__len__()<superV)):
    i+=1
    x=random.randrange(rows)
    sex=dataTest.iloc[x].Sex[0]
    #Pásolle a condición cando a asignación sexa maior que 10
    if lista.__len__()>5:
        #Esta condición garante que a proporción de mulleres estea 
         #dentro da do test cun erro de +/- 10%
        if not (((propFem< (prob_female_cond_survived-0.02)) & (sex=='m')) |
                ((propFem> (prob_female_cond_survived+0.02)) & (sex=='f'))):
            if not x in lista:
                lista.append(x)
                listaSex.append(sex)
                propFem=listaSex.count('f')/listaSex.__len__()
    else:
        if not x in lista:
                lista.append(x)
                listaSex.append(sex)
                propFem=listaSex.count('f')/listaSex.__len__()
                
lista.__len__()
lista.sort()
listaSuperv=getListaSup(lista,rows)
propFem
submit=pd.DataFrame({'Survived':listaSuperv},index=dataTest.index)
submit.head()
#submit.to_csv('./kaggle/input/titanic/randomSubmitSex.csv')
data.groupby('Age').count()
data['Age2']=data['Age']

def getAgeGroup(d):
    if d<18:
        return 'nenos'
    elif d<55:
        return 'adultos'
    else:
        return 'vellos'
getAgeGroup(54)
data['Age2']=data.Age.apply(getAgeGroup)
data.groupby('Age2')['Survived'].sum(), data.groupby('Age2')['Survived'].sum()/data.Survived.sum()
propVellos=data.groupby('Age2')['Survived'].sum()[2]/data.Survived.sum()
propVellos

data.groupby('Age2')['Survived'].count(), data.groupby('Age2')['Survived'].count()/rowsData
dataTest['Age2']=dataTest.Age.apply(getAgeGroup)
#Seleccion superV posicións dentro de rows
#Selecciono homes e mulleres na proporción  prob_female_cond_survived
#Selecciono supervivintes na proporción de nenos propNenos
lista=[]
listaSex=[]
listaAge=[]
i=0
while ((lista.__len__()<superV)):
    i+=1
    x=random.randrange(rows)
    sex=dataTest.iloc[x].Sex[0]
    age=dataTest.iloc[x].Age2[0]
    #Pásolle a condición cando a asignación sexa maior que 10
    if lista.__len__()>5:
        #Esta condición garante que a proporción de mulleres estea 
         #dentro da do test cun erro de +/- 10%
        if not (((propFem< (prob_female_cond_survived-0.02)) & (sex=='m')) |
                ((propFem> (prob_female_cond_survived+0.05)) & (sex=='f'))):
            
            if not (((propVello< (propVellos-0.02)) & (age!='v')) |
                ((propVello> (propVellos+0.05)) & (age=='v'))):
            
                if not x in lista:
                    lista.append(x)
                    listaSex.append(sex)
                    listaAge.append(age)
                    propFem=listaSex.count('f')/listaSex.__len__()
                    propVello=listaAge.count('v')/listaAge.__len__()

    else:
        if not x in lista:
                lista.append(x)
                listaSex.append(sex)
                listaAge.append(age)
                propFem=listaSex.count('f')/listaSex.__len__()
                propVello=listaAge.count('v')/listaAge.__len__()
                
lista.__len__()
lista.sort()
listaSuperv=getListaSup(lista,rows)
propVello,propFem
submit=pd.DataFrame({'Survived':listaSuperv},index=dataTest.index)
submit.head()
#submit.to_csv('/home/manu/Programacion/kaggle/titanic/randomSubmitSexAge.csv')
data.groupby('Pclass')['Survived'].count(),data.groupby('Pclass')['Survived'].count()/rowsData
data.groupby('Pclass')['Survived'].sum(),data.groupby('Pclass')['Survived'].sum()/data.Survived.sum()
prop1Class=data.groupby('Pclass')['Survived'].sum()[1]/data.Survived.sum()
prop1Class
#Seleccion superV posicións dentro de rows
#Selecciono homes e mulleres na proporción  prob_female_cond_survived
#Selecciono supervivintes na proporción de clase prop3Class
lista=[]
listaSex=[]
listaClass=[]
i=0
while ((lista.__len__()<superV)):
    i+=1
    x=random.randrange(rows)
    sex=dataTest.iloc[x].Sex[0]
    clase=dataTest.iloc[x].Pclass
    #Pásolle a condición cando a asignación sexa maior que 10
    if lista.__len__()>5:
        #Esta condición garante que a proporción de mulleres estea 
         #dentro da do test cun erro de +/- 10%
        if not (((propFem< (prob_female_cond_survived-0.01)) & (sex=='m')) |
                ((propFem> (prob_female_cond_survived+0.03)) & (sex=='f'))):
            
            if not (((prop< (prop1Class-0.01)) & (clase!=1)) |
                ((prop> (prop1Class+0.03)) & (clase==1))):
                
                
            
                if not x in lista:
                    lista.append(x)
                    listaSex.append(sex)
                    listaClass.append(clase)
                    propFem=listaSex.count('f')/listaSex.__len__()
                    prop=listaClass.count(1)/listaClass.__len__()

    else:
        if not x in lista:
                lista.append(x)
                listaSex.append(sex)
                listaClass.append(clase)
                propFem=listaSex.count('f')/listaSex.__len__()
                prop=listaClass.count(1)/listaClass.__len__()
                
lista.__len__()
lista.sort()
listaSuperv=getListaSup(lista,rows)
prop,propFem
submit=pd.DataFrame({'Survived':listaSuperv},index=dataTest.index)
submit.head()
#submit.to_csv('/home/manu/Programacion/kaggle/titanic/randomSubmitSexClass.csv')
#Seleccion superV posicións dentro de rows
#Selecciono homes e mulleres na proporción  prob_female_cond_survived
#Selecciono supervivintes na proporción de clase prop3Class
lista=[]
listaSex=[]
listaClass=[]
listaAge=[]
i=0
while ((lista.__len__()<superV)):
    i+=1
    x=random.randrange(rows)
    sex=dataTest.iloc[x].Sex[0]
    clase=dataTest.iloc[x].Pclass
    age=dataTest.iloc[x].Age2[0]
    #Pásolle a condición cando a asignación sexa maior que 10
    if lista.__len__()>5:
        #Esta condición garante que a proporción de mulleres estea 
         #dentro da do test cun erro de +/- 10%
        if not (((propFem< (prob_female_cond_survived-0.02)) & (sex=='m')) |
                ((propFem> (prob_female_cond_survived+0.03)) & (sex=='f'))):
            
            if not (((prop< (prop1Class-0.02)) & (clase!=1)) |
                ((prop> (prop1Class+0.03)) & (clase==1))):
                
                if not (((propVello< (propVellos-0.02)) & (age!='v')) |
                ((propVello> (propVellos+0.03)) & (age=='v'))):
                
            
                    if not x in lista:
                        lista.append(x)
                        listaSex.append(sex)
                        listaClass.append(clase)
                        listaAge.append(age)
                        propFem=listaSex.count('f')/listaSex.__len__()
                        prop=listaClass.count(1)/listaClass.__len__()
                        propVello=listaAge.count('v')/listaAge.__len__()
                    

    else:
        if not x in lista:
                lista.append(x)
                listaSex.append(sex)
                listaClass.append(clase)
                listaAge.append(age)
                propFem=listaSex.count('f')/listaSex.__len__()
                prop=listaClass.count(1)/listaClass.__len__()
                propVello=listaAge.count('v')/listaAge.__len__()
                
lista.__len__()
propFem, prop, propVello
lista.sort()
listaSuperv=getListaSup(lista,rows)
submit=pd.DataFrame({'Survived':listaSuperv},index=dataTest.index)
#submit.to_csv('/home/manu/Programacion/kaggle/titanic/randomSubmitSexClassAge.csv')

## importation des librairies python


import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

## création d'une fonction de conversion des dates de la base en datetime

def date_convert(date_to_convert):

    return datetime.strptime(date_to_convert, '%Y/%m/%d %H:%M:%S.%f')   
## Chargement  du fichier csv et lecture en latin1

df=pd.read_csv("../input/Squence_d_27appel_JMC.csv", sep = ',',encoding="latin1")
df2=df.iloc[382:, :]
df2['Date']=df2['Date'].apply(date_convert)


### Selection de toutes les lignes journalières
datday = pd.Timestamp('today')
daytoday = datday.day
#df2['day'] = df2['Date'].dt.day
#df2 = df2[df2['day']== daytoday]
#df2


## Sélection des lignes KPI et C3 pour processe 3 RAD/LAD

dfa_KPI = df2[df2['Message;'].str.contains('KPI', regex=False, case=False, na=False)]
dfa_KPI_C3 = df2[df2['Message;'].str.contains('C3 ', regex=False, case=False, na=False)]


##Conversion de la colonne 'Date' en liste


test=dfa_KPI_C3['Date'].tolist()


##Conversion de la colonne 'Message' en liste 1 et conversion de la liste 1 en liste 2 donc le contenu tokenise chaque mot


list1=dfa_KPI_C3['Message;'].tolist()
list2=[ sent.split() for sent in list1]


## fonction de catégorisation de chaque ligne si 'status = 1' est dans le contenu de la colonne 'message' 
## alors on la considère comme une de Process en cours, à l'inverse c'est une ligne de Process terminé


idlist=[]
Ouvert_Ferme=[]
for i in range(len(list2)):
    if "status = 1" in list1[i]:
        Ouvert_Ferme.append("Process en cours")  
    else:
        Ouvert_Ferme.append("Process termine")
## Recherche et stockage de l'ID du dossier correspondant
    for j in range(len(list2[i])):
        if "DOS" in list2[i][j]:
            idlist.append(list2[i][j])
            
            
## Appel au fonction précédente sur la base de données


dfa_KPI_C3["ID"]=idlist
dfa_KPI_C3["Etat_process"]=Ouvert_Ferme 


 ### Catégorisation de la base de données en fonction du type de document, JUDI ou JUDO

typeinjection=[]

for k in range(len(list2)):
        if "JUDI" in list1[k]:
            typeinjection.append("JUDI")
        else:
            typeinjection.append("JUDO")
            
dfa_KPI_C3["TYPE_INJECTION"]=typeinjection


### Catégorisation de la base données en fonction de la catégorie du document


typedoc=[]
for l in range(len(list2)):
        if "ID" in list1[l]:
            typedoc.append("carte identite")
        elif "PASS" in list1[l]:
            typedoc.append("passport")
        elif "TS" in list1[l]:
            typedoc.append("titre de sejour")
        elif "TF" in list1[l]:
            typedoc.append("taxe fonciere")
        elif "ISR" in list1[l]:
            typedoc.append("impot sur revenu")
        elif "TH" in list1[l]:
            typedoc.append("taxe d'habitation")
        elif "FA" in list1[l] :
            typedoc.append("facture")
        else :
            typedoc.append("autre")
       

dfa_KPI_C3["TYPE_DOCUMENT"]=typedoc



### Déterminsation de l'ID du document

          
iddoc=[]

for l in range(len(list2)):
    test=""
    for m in range(len(list2[l])):
        if "ID" in list2[l][m]:
            test=list2[l][m]
        elif "PASS" in list2[l][m]:
            test=list2[l][m]
        elif "TS" in list2[l][m]:
            test=list2[l][m]
        elif "TF" in list2[l][m]:
            test=list2[l][m]
        elif "ISR" in list2[l][m]:
            test=list2[l][m]
        elif "TH" in list2[l][m]:
            test=list2[l][m]
        elif "FA" in list2[l][m] :
            test=list2[l][m]
    if test=="":
        iddoc.append("autre")
    else :
        iddoc.append(test)
dfa_KPI_C3["ID_DOCUMENT"]=iddoc

### Création d'une colonne sur le status du RAD/LAD: OK si LAD = OK et RAD = OK, sinon on considère le RAD/LAD comme KO

radladlist=[]

for n in range(len(list2)):
        if "RAD = OK" and "LAD = OK" in list1[n]:
            radladlist.append("OK RAD/LAD")
        else :
             radladlist.append("KO")
                
dfa_KPI_C3["Process_RAD_LAD"]=radladlist


## Conversion de la colonne 'Message' en une liste

dfa_KPI_C3["Message;"].tolist()
## Triage de la base de données en fonction de l'ID du dossier, de l'ID du document, et de la date

dfa2_KPI_C3=dfa_KPI_C3.sort_values(by=['ID', 'ID_DOCUMENT',  'Date' ],ascending=False)



## Une fois la base de données triée, selection de la première ligne de chaque ID qui correspond au dernier message de chaque ID
## , plus particulièrement de son message de retour 

dfa2_KPI_C3_first=dfa2_KPI_C3.drop_duplicates(subset=['ID_DOCUMENT'],keep='first')

dfa2_KPI_C3_last=dfa2_KPI_C3.drop_duplicates(subset=['ID_DOCUMENT'],keep='last')


dfa4 =[dfa2_KPI_C3_first, dfa2_KPI_C3_last]
dfa4 = pd.concat(dfa4)
dfa4=dfa4.sort_values(by=['ID', 'ID_DOCUMENT',  'Date' ],ascending=False)


list3=dfa4['Process_RAD_LAD'].tolist()
list4=dfa4['Date'].tolist()
list5=dfa4["ID_DOCUMENT"].tolist()
TimeResponse=[]
TimeResponse2=[]
for i in range(len(list4)-1):
    if list3[i]=="OK RAD/LAD" and  list3[i+1]=="KO" and list5[i]==list5[i+1]:
        

        TimeResponse.append((list4[i]-list4[i+1]).total_seconds()*1000)
        TimeResponse2.append((list4[i]-list4[i+1]).total_seconds()*1000)
    else:
        TimeResponse2.append("Nan")
TimeResponse2.append("Nan") 


##Nouvelle colonne avec le temps de réponse entre le premier reçu de l'état du processus RAD/LAD et le dernier processus correspondant au processus terminé

dfa4['TimeResponse']=TimeResponse2


dfa4_finish = dfa4[dfa4['Etat_process'].str.contains('Process termine', regex=False, case=False, na=False)]
dfa4_finish = dfa4_finish[dfa4_finish['TimeResponse'] != 'Nan']


### Vérification du temps de réponse des processus JUDI s'il est inférieur à 5 secondes

dfa4_finish_JUDI = dfa4_finish[dfa4_finish['TYPE_INJECTION'].str.contains('JUDI', regex=False, case=False, na=False)]


### Vérification du temps de réponse des processus JUDO si il est inférieur à 30 secondes

dfa4_finish_JUDO = dfa4_finish[dfa4_finish['TYPE_INJECTION'].str.contains('JUDO', regex=False, case=False, na=False)]

#######Création d'une colonne de vérification du temps de réponse

list7=dfa4_finish['TimeResponse'].tolist()
list8= dfa4_finish['TYPE_INJECTION'].tolist()
timeresplist=[]
for n in range(len(list7)):
        if list7[n]<= 5000 and list8[n]=="JUDI" :
            timeresplist.append("OK")
        elif list7[n] > 5000 and list8[n]=="JUDI" :
            timeresplist.append("timeout")
        elif list7[n] > 30000 and list8[n]=="JUDO" :
            timeresplist.append("timeout")
        elif list7[n] <= 30000 and list8[n]=="JUDO" :
            timeresplist.append("OK")
        else :
             timeresplist.append("out")
                
dfa4_finish["Seuil de temps"]=timeresplist

### Création d'une dataframe avec des retour de timeout
searchfor = ['timeout', 'out']
dfa4_finish_timeout = dfa4_finish[dfa4_finish['Seuil de temps'].str.contains('|'.join(searchfor))]




## Création de Visualisation

import random
     
def colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r/256,g/256,b/256)) 
    return ret


dfa2_KPI_C3_last

dfa2_KPI_C3_first


### Filtre des lignes dont le temps de réponse est différent du nul

dfa4_finish
dfa4_finish_timeout
## Comptage du nombre de processus RAD/LAD = OK et ceux égaux à KO

list3=dfa2_KPI_C3_first['Process_RAD_LAD'].tolist()
IDok=0
for i in range(len(list3)):
    if list3[i]=="KO":
        IDok+=1
print('le nombre de ko pour le process RAD/LAD est de '+str(IDok))
print('\n')
print('le nombre de ok pour le process RAD/LAD est de '+str(len(list3)-IDok)) 
print('\n')
print('Le Temps de reponse moyen est de :'+str(np.mean(TimeResponse))+' ms')
print('\n')
#print("La variance est de :"+str(np.var(TimeResponse)))
#print('\n')

list4=dfa4_finish['Seuil de temps'].tolist()
list5=dfa2_KPI_C3_last['Process_RAD_LAD'].tolist()
list6=dfa4_finish_timeout['Process_RAD_LAD'].tolist()

print('le taux journalier de validation du RAD/LAD est de '+ str(len(list4)/len(list5))+'%')
print('\n')
print('le taux journalier d\'acceptation du RAD/LAD est de '+ str((len(list4)-(len(list6)))/len(list5))+'%')


list10=dfa4_finish_JUDI['TimeResponse'].tolist()
IDok=0
for i in range(len(list10)):
        if list10[i]<= 5000:
            IDok+=1
print('le nombre de processus JUDI dans les temps est de '+str(IDok))
print('\n')
print('le nombre de processus JUDI pas dans les temps est de '+str(len(list10)-IDok)) 
print('\n')
list11=dfa4_finish_JUDO['TimeResponse'].tolist()
IDok=0
for i in range(len(list11)):
        if list11[i]<= 30000:
            IDok+=1
print('le nombre de processus JUDO dans les temps est de '+str(IDok))
print('\n')
print('le nombre de processus JUDO pas dans les temps est de '+str(len(list11)-IDok)) 

df_0= dfa4_finish.groupby(["TYPE_INJECTION"]).size()
x = list(df_0.index)
y=list(df_0)
couleur=colors(len(x))

plt.bar(x, y,color=couleur)
plt.title('nombre de pièce JUDI/JUDO')
plt.show()



df_0= dfa2_KPI_C3_first.groupby(["Process_RAD_LAD"]).size()
x = list(df_0.index)
y=list(df_0)
couleur=colors(len(x))

plt.bar(x, y,color=couleur)
plt.title('Etat process terminé')
plt.show()

##histogramme retour KO/ retour OK

dfa4_finish_JUDI.set_index('Date', drop=False, inplace=True)
dfa4_finish_JUDO.set_index('Date', drop=False, inplace=True)



ax = dfa4_finish_JUDI.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', label='JUDI', figsize=(20,15))
dfa4_finish_JUDO.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', label='JUDO')
plt.legend(loc='upper left')
dfa4_finish.set_index('Date', drop=False, inplace=True)
dfa4_finish_timeout.set_index('Date', drop=False, inplace=True)



ax = dfa4_finish.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', label='All', figsize=(20,15))
dfa4_finish_timeout.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', label='Timeout')
plt.legend(loc='upper left')


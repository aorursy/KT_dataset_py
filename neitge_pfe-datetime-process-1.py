## importation des librairies python

import csv
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
     
## création d'une fonction de conversion des dates de la base en datetime

def date_convert(date_to_convert):

    return datetime.strptime(date_to_convert, '%Y/%m/%d %H:%M:%S.%f')   
## Chargement  du fichier csv et lecture en latin1
df=pd.read_csv("../input/testproc1/Squence_d_27appel_JMC10.csv",encoding="latin1")[371:]

## Appel de la fonction de conversion des dates sur la colonne 'Date' du fichier csv

df['Date']=df['Date'].apply(date_convert)

## Sélection des ligne 'Création de dossier'
dfa_KPI_C1 = df[df['Message;'].str.contains('C1', regex=False, case=False, na=False)]




### Selection de toutes les lignes journalières
#datday = pd.Timestamp('today')
#daytoday = datday.day
#df['day'] = df['Date'].dt.day
#df = df[df['day']== daytoday]
#df



##Conversion de la colonne 'Date' en liste
test=df['Date'].tolist()

##Conversion de la colonne 'Message' en liste 1 et conversion de la liste 1 en liste 2 donc le contenu tokenise chaque mot

list1=df['Message;'].tolist()
list2=[ sent.split() for sent in list1]

## fonction de catégorisation de chaque ligne si 'http=200' est dans le contenu de la colonne 'message' 
## alors on la considère comme une ligne de reçu, à l'inverse c'est une ligne d'envoi

idlist=[]
Ouvert_Ferme=[]
for i in range(len(list2)):
    if "http=200" in list1[i]:
        Ouvert_Ferme.append("Recu")
    else:
        Ouvert_Ferme.append("Envoi")
## si le contenu du message a un mot commençant par 'DOS' alors ce mot est l'ID du dossier        
    for j in range(len(list2[i])):
        if "DOS" in list2[i][j]:
            idlist.append(list2[i][j])
            
            
## Appel au fonction précédente sur la base de données

df["ID"]=idlist    
df["Ouvert/Ferme"]=Ouvert_Ferme  

## Triage de la base de données en fonction de l'ID et de la Date

df2=df.sort_values(by=['ID','Date'],ascending=False)




## Une fois la base de données triée, selection de la première ligne de chaque ID qui correspond au dernier message de chaque ID
## , plus particulièrement de son message de retour 

df3=df2.drop_duplicates(subset=['ID'],keep='first')
nb_demande=len(df2.drop_duplicates(subset=['ID'],keep='last'))
demande = df2.drop_duplicates(subset=['ID'],keep='last')
nb_retour_demande = len(df3)



demanddoc=[]


for z in range(len(demande)):
    demanddoc.append("demande")

demande['Etat']=demanddoc


returndoc=[]

for y in range(len(df3)):
        returndoc.append("return")
df3['Etat']=returndoc


frames = [df3, demande]
result1 = pd.concat(frames)
result1.to_csv('mycsvfileresult1.csv',index=False)
result1=result1.sort_values(by=['ID','Date'],ascending=False)



## Fonction pour connaître la durée du processus de création de dossier
list3=result1['Etat'].tolist()
list4=result1['Date'].tolist()
list5=result1["ID"].tolist()
TimeResponse=[]
TimeResponse2=[]
for i in range(len(list4)-1):
    if list3[i]=="return" and  list3[i+1]=="demande" and list5[i]==list5[i+1]:
        

        TimeResponse.append((list4[i]-list4[i+1]).total_seconds()*1000)
        TimeResponse2.append((list4[i]-list4[i+1]).total_seconds()*1000)
    else:
        TimeResponse2.append("nul")
TimeResponse2.append("nul") 

## Affichage du temps de réponse pour chaque message final

result1['TimeResponse']=TimeResponse2

returndoc=[]
list6=result1['TimeResponse'].tolist()
list7=result1['Etat'].tolist()


for i in range(len(list6)):
    
    if list6[i] != 'nul' and list6[i] != 0 and list7[i] == "return":
        returndoc.append("OK")


    elif list7[i] == "demande":
            returndoc.append("demande")   
    else:
        returndoc.append("KO")
    

result1['Return_validation'] = returndoc

df21= result1[result1['Etat'].str.contains('return')].groupby(["Return_validation"]).size()
df21.index

result1["Date"] = result1["Date"].values.astype('datetime64[ms]')
    
result1

## Compter le nombre de message de retour comme étant le status du processus de création de dossier
## , si c'est 'Recu' et donc OK ou pas et donc KO 

list13=df3['Ouvert/Ferme'].tolist()
IDok=0
for i in range(len(list13)):
    if list13[i]=="Recu":
        IDok+=1
print('le nombre de ok est de '+str(IDok))
print('le nombre de ko est de '+str(len(list13)-IDok))


## Affichage des résultats du temps moyen de réponse, et de la variance
print("Le Temps de reponse moyen est de :"+str(np.mean(TimeResponse)))
print("La variance est de :"+str(np.var(TimeResponse)))

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
df21= result1.groupby(["Etat"] ).size()
df21.index


x = list(df21.index)
y=list(df21)
couleur=colors(len(x))

plt.bar(x, y,color=couleur)
plt.show()



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


df22= result1.groupby(["Return_validation"] ).size()
df22.index

u = list(df22.index)
w=list(df22)
couleur=colors(len(u))

plt.figure(figsize=(10,10))
plt.bar(u, w,color=couleur)
plt.title('retour positif')
plt.show()
result_OK = result1[result1['Return_validation'].str.contains('OK', regex=False, case=False, na=False)]
result_KO = result1[result1['Return_validation'].str.contains('KO', regex=False, case=False, na=False)]
result_demande = result1[result1['Return_validation'].str.contains('demande', regex=False, case=False, na=False)]

#dfa_KPI_C1 = df[df['Message;'].str.contains('C1', regex=False, case=False, na=False)]
result_OK.set_index('Date', drop=False, inplace=True)
#df.groupby(["symbol",pd.TimeGrouper("30T", key="datetime")]).count()
result_OK.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', figsize=(20,15))

#result_KO.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', figsize=(20,15))
result_KO.set_index('Date', drop=False, inplace=True)
result_KO.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', figsize=(20,15))

result_demande.set_index('Date', drop=False, inplace=True)
result_demande.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', figsize=(20,15))
result1.groupby([ result1["Etat"] == "return", result1["Return_validation"] == "OK"]).size().plot(kind="bar")
##histogramme retour KO/ retour OK

ax = result_KO.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', label='retour KO', figsize=(20,15))
result_OK.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='area', label='retour OK')
plt.legend(loc='upper left')






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



df=pd.read_csv("../input/final-donnee/Squence_d_27appel_JMC10(1).csv",encoding="latin1")[373:]

## Appel de la fonction de conversion des dates sur la colonne 'Date' du fichier csv

df['Date']=df['Date'].apply(date_convert)


### Selection de toutes les lignes journalières
datday = pd.Timestamp('today')
daytoday = datday.day
#df['day'] = df['Date'].dt.day
#df = df[df['day']== daytoday]
#df

## Sélection des lignes KPI

#dfa_KPI = df[df['Message'].str.contains('KPI', regex=False, case=False, na=False)]

## Sélection des ligne 'InjectionDocument'

dfa_KPI_C2 = df[df['Message;'].str.contains('InjectionDocument ', regex=False, case=False, na=False)]



##Conversion de la colonne 'Date' en liste

test=dfa_KPI_C2['Date'].tolist()

##Conversion de la colonne 'Message' en liste 1 et conversion de la liste 1 en liste 2 donc le contenu tokenise chaque mot


list1=dfa_KPI_C2['Message;'].tolist()
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


dfa_KPI_C2["ID"]=idlist    
dfa_KPI_C2["Ouvert/Ferme"]=Ouvert_Ferme    




idlist=[]
Ouvert_Ferme=[]
typeinjection=[]
typedoc=[]
iddoc=[]


### Catégorisation de la base de données en fonction des messages reçus ou envoyés

for i in range(len(list2)):
    if "http=200" in list1[i]:
        Ouvert_Ferme.append("Recu")
    else:
        Ouvert_Ferme.append("Envoi")
    
    for j in range(len(list2[i])):
        if "DOS" in list2[i][j]:
            idlist.append(list2[i][j])
            
 ### Catégorisation de la base de données en fonction du type de document, JUDI ou JUDO

for k in range(len(list2)):
        if "JUDI" in list1[k]:
            typeinjection.append("JUDI")
        else:
            typeinjection.append("JUDO")
            
### Catégorisation de la base données en fonction de la catégorie du document
            
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

            
### Déterminsation de l'ID du document

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
dfa_KPI_C2["ID_DOSS"]=idlist 
dfa_KPI_C2["Ouvert/Ferme"]=Ouvert_Ferme
dfa_KPI_C2["TYPE_INJECTION"]=typeinjection
dfa_KPI_C2["TYPE_DOCUMENT"]=typedoc
dfa_KPI_C2["ID_DOCUMENT"]=iddoc



## Conversion de la colonne 'Message' en une liste

dfa_KPI_C2["Message;"].tolist()


## Triage de la base de données en fonction de l'ID du dossier, ID du document, de la date de la ligne, et le type d'injection

dfa2_KPI_C2=dfa_KPI_C2.sort_values(by=['ID_DOSS', 'ID_DOCUMENT', 'Date', 'TYPE_INJECTION'],ascending=False)

## Une fois la base de données triée, selection de la première ligne de chaque ID qui correspond au dernier message de chaque ID
## , plus particulièrement de son message de retour 

dfa3_KPI_C2=dfa2_KPI_C2.drop_duplicates(subset=['ID_DOCUMENT'],keep='first')

## Une fois la base de données triée, selection de la première ligne de chaque ID qui correspond au premier message de chaque ID
## , plus particulièrement de son message de retour 

dfa4_KPI_C2=dfa2_KPI_C2.drop_duplicates(subset=['ID_DOCUMENT'],keep='last')

## Fonction pour connaître la durée du processus d'injection de document


list3=dfa2_KPI_C2['Ouvert/Ferme'].tolist()
list4=dfa2_KPI_C2['Date'].tolist()
list5=dfa2_KPI_C2["ID_DOCUMENT"].tolist()
TimeResponse=[]
TimeResponse2=[]
for i in range(len(list4)-1):
    if list3[i]=="Recu" and  list3[i+1]=="Envoi" and list5[i]==list5[i+1]:
        

        TimeResponse.append((list4[i]-list4[i+1]).total_seconds()*1000)
        TimeResponse2.append((list4[i]-list4[i+1]).total_seconds()*1000)
    else:
        TimeResponse2.append("Nan")
TimeResponse2.append("Nan") 



dfa2_KPI_C2['TimeResponse']=TimeResponse2

df22=dfa2_KPI_C2[dfa2_KPI_C2['Ouvert/Ferme'].str.contains('Envoi', regex=False, case=False, na=False)]
df2= df22.groupby(["TYPE_DOCUMENT"]).size()

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


df23=dfa2_KPI_C2[dfa2_KPI_C2['Ouvert/Ferme'].str.contains('Recu', regex=False, case=False, na=False)]
df3= df23.groupby(["TYPE_DOCUMENT"]).size()


## Compter le nombre de message de retour comme étant le status du processus d'injection de document
## , si c'est 'Recu' et donc OK ou pas et donc KO 


list3=dfa3_KPI_C2['Ouvert/Ferme'].tolist()
IDok=0
for i in range(len(list3)):
    if list3[i]=="Recu":
        IDok+=1
print('le nombre de ok est de '+str(IDok))
print('le nombre de ko est de '+str(len(list3)-IDok))


## Affichage des résultats du temps moyen de réponse, et de la variance


print("Le Temps de reponse moyen est de :"+str(np.mean(TimeResponse)))
print("La variance est de :"+str(np.var(TimeResponse)))
x = list(df2.index)
y=list(df2)
couleur=colors(len(x))

plt.bar(x, y,color=couleur)
plt.title('demande de pièce de document')
plt.show()

x = list(df3.index)
y=list(df3)
z = list(df2.index)
w = list(df2)
couleur=colors(len(x))

plt.bar(z, w,color=couleur)
plt.bar(x, y,color=couleur)
plt.title('validation de pièce de document')
plt.show()


df4= df22.groupby(["TYPE_INJECTION"]).size()
x = list(df4.index)
y=list(df4)
couleur=colors(len(x))

plt.bar(x, y,color=couleur)
plt.title('nombre de pièce JUDI/JUDO')
plt.show()
import seaborn as sns
g =sns.factorplot(x='Date', col='Ouvert/Ferme', kind='count', data=dfa2_KPI_C2,
                  col_wrap=3, size=3, aspect=1.3,  palette='muted')
g.set_xticklabels(rotation=90)
h =sns.factorplot(x='Date', col='TYPE_INJECTION', kind='count', data=dfa2_KPI_C2,
                  col_wrap=3, size=3, aspect=1.3,  palette='muted')
h.set_xticklabels(rotation=90)
i =sns.factorplot(x='Date', col='TYPE_DOCUMENT', kind='count', data=dfa2_KPI_C2,
                  col_wrap=3, size=3, aspect=1.3,  palette='muted')
i.set_xticklabels(rotation=90)
dfa3_KPI_C2
#Histogramme des demandes de retour
dfa3_KPI_C2.set_index('Date', drop=False, inplace=True)
dfa3_KPI_C2.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='bar')
##Histogramme des demandes

dfa4_KPI_C2.set_index('Date', drop=False, inplace=True)
dfa4_KPI_C2.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='bar')


ax = dfa4_KPI_C2.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='bar', label='envoi', figsize=(20,15))
dfa3_KPI_C2.groupby(pd.TimeGrouper(freq='5Min')).size().plot(kind='bar', label='retour', figsize=(20,15))
plt.legend(loc='upper left')



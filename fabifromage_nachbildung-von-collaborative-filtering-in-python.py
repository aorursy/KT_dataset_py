#Import von Paketen und Datenquellen (Movielens Dataset)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
film = pd.read_csv('../input/movie.csv') #enthält Informationen über Filme
ratings = pd.read_csv('../input/rating.csv') #enthält User-Ids sowie Bewertungen für Filme
#Tabelle in Form einer Recommender Matrix erstellen
s = []
for i in range(1,31):
    s.append(i)
    
table = pd.DataFrame(columns =film.movieId, index = s)
table.index.name = 'userid'
table
#Tabelle mit Bewertungen aus 'ratings' füllen
for i in range(2000):
    table.ix[ratings.ix[i,'userId'], ratings.ix[i, 'movieId']] = ratings.ix[i, 'rating']
table 
#Empty-Values mit den Mittelwerten der Bewertungen eines User füllen, um eine ausreichende Datenmenge zu erhalten
newtable = newtableone.astype('float64')

for j in range (16):
    summe = 0
    count = 0
    schnitt = 0
    for k in range (9999):
        if newtable.iloc[j,k] != 0:
            summe = summe + newtable.iloc[j,k]
            count = count + 1
            
    if count == 0:
        schnitt = 0
    else:
        schnitt = summe / count
    for m in range (13126):
        if newtable.iloc[j,m] == 0:
            newtable.iloc[j,m] = schnitt
            
table1 = newtable
            

    
        
        
#Reduzieren der Datenmenge (aus Performancegründen. Es soll ausschließlich die Funktionsweise dargestellt werden.)
newtable =table1.iloc[0:16,0:1000]
newtable.index.name = 'userid'
newtable
#Definition einer Recommenderfunktion die als Inputs userId und filmId für die 
#gewünschte Vorhersage entgegennimmt und nächste Nachbarn sowie die Vorhersage ausgibt.
# Die Vorhersage wird durch den Mittelwert des nächsten und zweitnächsten Nachbarn gebildet.
def vorschlag(user, film):
    user = user-1
    film = film-1
    threshold=-1
    for i in range(0, len(newtable)):
        if i != user:
            korrelation = np.corrcoef(newtable.iloc[user,:], newtable.iloc[i,:])
            korrelation = korrelation.min()
            if korrelation > threshold:
                threshold = korrelation
                maxkorrelation = i
                
    for i in range(0, len(newtable)):
        threshold=-1
        if i != user and i != maxkorrelation:
            korrelation = np.corrcoef(newtable.iloc[user,:], newtable.iloc[i,:])
            korrelation = korrelation.min()
            if korrelation > threshold:
                threshold = korrelation
                zweithoechstekorrelation = i
    prediction = (newtable.iloc[maxkorrelation, film] + newtable.iloc[zweithoechstekorrelation, film])/2
    naechsterNachbar = maxkorrelation + 1
    zweitnaechsterNachbar = zweithoechstekorrelation + 1
    print("Nächster Nachbar: ", naechsterNachbar )
    print("2.Nächster Nachbar: ",  zweitnaechsterNachbar )
    return prediction
#Anwendung der Funktion
vorschlag(3,1)
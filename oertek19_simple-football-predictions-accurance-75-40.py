

import numpy as np 

import pandas as pd 







import sqlite3

import matplotlib.pyplot as plt





path = "../input/"  #Insert path here

database = path + 'database.sqlite'

conn = sqlite3.connect(database)

data = pd.read_sql("""SELECT Match.id, 

                                        Country.name AS country_name, 

                                        League.name AS league_name, 

                                        season, 

                                        stage, 

                                        date,

                                        HT.team_long_name AS  home_team,

                                        AT.team_long_name AS away_team,

                                        home_team_goal, 

                                        away_team_goal                                        

                                FROM Match

                                JOIN Country on Country.id = Match.country_id

                                JOIN League on League.id = Match.league_id

                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id

                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id

                                WHERE country_name in ('Spain', 'Germany', 'France', 'Italy', 'England')

                                

                                ORDER by date

                                LIMIT 100000;""", conn)

data.head()
data1=data[["home_team","away_team","season","home_team_goal","away_team_goal"]]

data=pd.DataFrame({"HomeTeam":data1.home_team+data1.season,"AwayTeam":data1.away_team+data1.season,"FTHG":data1.home_team_goal,"FTAG":data1.away_team_goal})

## data1.home_team+data1.season >>>> Real Madrid at 2010-2011 season is differant at 2011-2012 season

## i think teams are different every season
# Team goal means



data['par1'] = data['FTHG'].groupby(data['HomeTeam']).transform('mean')

data['par2'] = data['FTHG'].groupby(data['AwayTeam']).transform('mean')

data['par3'] = data['FTAG'].groupby(data['HomeTeam']).transform('mean')

data['par4'] = data['FTAG'].groupby(data['AwayTeam']).transform('mean')



data.head()
## if Home team is win, win=1

## if Away team is win or draw, win=0

win=[]

for l in range(0,len(data)):

    if data.FTHG[l]>data.FTAG[l]:

        k1=1

        win.append(k1)

    else:

        k1=0

        win.append(k1)

df2=pd.DataFrame({"mx1":data.par1,"mx2":data.par2,"mx3":data.par3,"mx4":data.par4,"a":win})

from sklearn.model_selection import train_test_split

df2train, df2test= train_test_split(

 df2, test_size=0.35, random_state=42)



df2=np.array(df2)

df2test=np.array(df2test)

Y=df2[:,0]

X=df2[:,1:5]

Y1=df2test[:,0]

X1=df2test[:,1:5]
## Model XBClassifier



from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X,Y)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X1)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(Y1, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
## KNeighbors Model

from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier

seed = 42

kfold = model_selection.KFold(n_splits=2, random_state=seed)

model1 = KNeighborsClassifier()

model1.fit(X,Y)

y_pred = model1.predict(X1)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(Y1, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
## Gaussian Model



from sklearn import model_selection

from sklearn.naive_bayes import GaussianNB



seed = 42

kfold = model_selection.KFold(n_splits=2, random_state=seed)

model2 = GaussianNB()

model2.fit(X,Y)

y_pred = model2.predict(X1)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(Y1, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
## Discriminant Analysis



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

seed = 42

kfold = model_selection.KFold(n_splits=2, random_state=seed)

model3 = LinearDiscriminantAnalysis()

model3.fit(X,Y)

y_pred = model3.predict(X1)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(Y1, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
## Logistic Regression Model



from sklearn.linear_model import LogisticRegression

seed = 42

kfold = model_selection.KFold(n_splits=2, random_state=seed)

model4 = LogisticRegression()

model4.fit(X,Y)

y_pred = model4.predict(X1)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(Y1, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
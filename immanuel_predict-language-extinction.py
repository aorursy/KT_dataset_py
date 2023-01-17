import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import svm
lang = pd.read_csv('../input/data.csv')
#Drop unwanted columns

lang = lang.drop(['Name in English','Name in French','Name in Spanish','ID','ISO639-3 codes','Name in the language','Alternate names','Description of the location'], axis=1)
lang.info()
#Number of speakers fill NA

lang['Number of speakers'] = lang['Number of speakers'].fillna(lang["Number of speakers"].median()).astype(int)



#Countries has one NA so fill with most occuring country ie "USA"

lang = lang.fillna({"Countries": "United States of America"})



#Countries codes alpha 3

lang = lang.fillna({"Country codes alpha 3": "USA"})



#Language sources has NAs fill with most occuring values

lang = lang.fillna({"Sources": "Wurm, Stephen A. 2007. 'Australia and the Pacific', pp. 424-557 in Christopher Moseley (ed.) Encyclopedia of the World's Endangered Languages. London/New York: Routledge."})



#Lat Long fill NA

lang['Latitude'] = lang['Latitude'].fillna(lang["Latitude"].median()).astype(int)

lang['Longitude'] = lang['Longitude'].fillna(lang["Longitude"].median()).astype(int)
X = lang[['Countries','Country codes alpha 3','Number of speakers','Latitude','Longitude']]

y = lang['Degree of endangerment']
#countries >>

dummy_countries = pd.get_dummies(X['Countries'])

X = X.drop('Countries', axis=1).join(dummy_countries)
#Country codes alpha 3 >>

dummy_col = pd.get_dummies(X['Country codes alpha 3'])

X = X.drop('Country codes alpha 3', axis=1).join(dummy_col)
#Split Dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
clf = svm.LinearSVC()

clf.fit(X_train,y_train) 

y_pred = clf.predict(X_test)

clf.score(X_train, y_train)
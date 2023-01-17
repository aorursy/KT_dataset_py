import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

beer = pd.read_csv("../input/beers.csv")

brew = pd.read_csv("../input/breweries.csv")
####Transfrom the beer Data###



#IBU will just be converted to the mean if the feature column, not ideal, but whatevs im lazy



beer = beer.dropna(subset = ["style"])

class_map = {label:idx for idx,label in enumerate((beer["style"]))}

name_map = {label:idx for idx,label in enumerate((beer["name"]))}

beer["style"] = beer["style"].map(class_map)

beer["name"] = beer["name"].map(name_map)
from sklearn.preprocessing import Imputer

imp = Imputer()

imp.fit(beer)

imputed_data = imp.transform(beer)

beer = pd.DataFrame(imputed_data, columns = beer.columns)

beer.head()
###Transform the brewery Data###

#Given that the Name of the brewery,city and state are not ranked either do a simmilr conversion

#the inv_ _map is just a switcharoo: 

#{v:k for k,v in _map.items()}

cities = np.unique(brew.city)



brewery_map = {label:idx for idx,label in enumerate((brew["name"]))}

city_map = {label:idx for idx,label in enumerate(cities)}

state_map = {label:idx for idx, label in enumerate((brew["state"]))}



brew["name"] = brew["name"].map(brewery_map)

brew["city"] = brew["city"].map(city_map)

brew["state"] = brew["state"].map(state_map)

brew.head()
#Im just going to drop the fucking NAs at this point, there's only 16 left



df = pd.merge(beer,brew)

df = df.dropna()

df.head()
##Feature Examination, another reason why im using a Random Forest...



from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



forest = RandomForestClassifier()

X,y = df[['abv',"ibu","id","name","brewery_id","city","state"]].values, df["style"].values



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



feat_labels = df[['abv',"ibu","id","name","brewery_id","city","state"]]

forest = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)

forest.fit(X_train,y_train)

importance = forest.feature_importances_

weights = ['abv',"ibu","id","name","brewery_id","city","state"]

indicies = np.argsort(importance)[::-1]

y_pred = forest.predict(X_test)



print("Misclassified Samples: %d" %(y_test != y_pred).sum())

print("Accuracy: %.2f" % accuracy_score(y_test,y_pred))



#Plot the rankings

plt.title("Feature Importance")

plt.bar(range(X_train.shape[1]),importance[indicies],color = "lightblue",align = 'center')

plt.xticks(range(X_train.shape[1]),feat_labels[indicies],rotation = 90)

plt.xlim([-1,X_train.shape[1]])

plt.tight_layout()

plt.show()
#Try With less features and only the beer data

#Not many NAs left at this point

beer = beer.dropna()



X,y = beer[['abv',"ibu", "ounces"]].values, beer["style"].values



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)



forest = RandomForestClassifier(criterion="entropy",n_estimators=10, random_state=1, n_jobs=2)

forest.fit(X_train,y_train)

y_pred = forest.predict(X_test)



feat_labels = beer[['abv',"ibu", "ounces"]]

forest = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)

forest.fit(X_train,y_train)

importance = forest.feature_importances_

weights = ['abv',"ibu", "ounces"]

indicies = np.argsort(importance)[::-1]



print("Misclassified Samples: %d" %(y_test != y_pred).sum())

print("Accuracy: %.2f" % accuracy_score(y_test,y_pred))



#Plot the rankings

plt.title("Feature Importance")

plt.bar(range(X_train.shape[1]),importance[indicies],color = "lightblue",align = 'center')

plt.xticks(range(X_train.shape[1]),feat_labels[indicies],rotation = 90)

plt.xlim([-1,X_train.shape[1]])

plt.tight_layout()

plt.show()
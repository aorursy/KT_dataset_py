import numpy as np

import seaborn as sns

import pandas as pd

from sklearn import svm

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

import matplotlib.cm as cm

le = LabelEncoder()
stars = pd.read_csv("../input/star-dataset/6 class csv.csv")
stars.head()
stars["Star color"].unique()
stars = stars.assign(Star_color_label = le.fit_transform(stars['Star color']), Spectral_class_label = le.fit_transform(stars['Spectral Class']))
stars.head()
# Normalizing the Luminosity and Radius Feature (l2)

stars['Luminosity_norm'] = preprocessing.normalize(np.array(stars['Luminosity(L/Lo)']).reshape(-1, 1), axis = 0)

stars['Radius_norm'] = preprocessing.normalize(np.array(stars['Radius(R/Ro)']).reshape(-1, 1), axis = 0)
stars.head()
plt.figure(figsize = (10,6))

plt.title('Average Luminosity(L/Lo) for Star Type(3, 4, 5)')

sns.barplot(x = stars['Star type'], y = stars['Luminosity(L/Lo)'])

plt.ylabel('Luminosity_norm(L/Lo)')

plt.xlim((2.5, 5.5))
plt.figure(figsize = (10,6))

plt.title('Average Luminosity(L/Lo) for Star Type(0, 1, 2)')

sns.barplot(x = stars['Star type'], y = stars['Luminosity(L/Lo)'])

plt.ylabel('Luminosity_norm(L/Lo)')

plt.xlabel('Star Type(0, 1, 2)')

plt.ylim((0,0.009))

plt.xlim((-0.5, 2.5))
plt.figure(figsize= (10,6))

sns.barplot(x = stars['Star type'], y = stars['Temperature (K)'])

plt.title('Average Temperature (k) for each Star Type')
plt.figure(figsize = (10,6))

sns.barplot(x = stars['Star type'], y = stars['Radius(R/Ro)'])

plt.xlabel('Star Type')

plt.xlim((2.5, 5.5))

plt.xlabel('Star Type (3, 4, 5)')

plt.title('Average Radius for Star Type(3, 4, 5)')
plt.figure(figsize= (10,6))

sns.barplot(x = stars['Star type'], y = stars['Radius_norm'])

plt.ylim((0,0.00005))

plt.xlim((-0.5, 2.5))

plt.xlabel('Star Type (0, 1, 2)')

plt.title('Average Normalized Radius for Star Type(0, 1, 2)')
stars_0 = stars[stars["Star type"] == 0]

stars_1 = stars[stars["Star type"] == 1]

stars_2 = stars[stars["Star type"] == 2]

stars_3 = stars[stars["Star type"] == 3]

stars_4 = stars[stars["Star type"] == 4]

stars_5 = stars[stars["Star type"] == 5]
plt.figure(figsize = (10,6))

plt.title("Radius vs Luminosity Scatterplot for Star Type(3, 4, 5)")

plt.scatter(x = stars_3["Radius(R/Ro)"], y = stars_3["Luminosity(L/Lo)"], color = 'orange', label = '3')

plt.scatter(x = stars_4["Radius(R/Ro)"], y = stars_4["Luminosity(L/Lo)"], color = 'brown', label = '4')

plt.scatter(x = stars_5["Radius(R/Ro)"], y = stars_5["Luminosity(L/Lo)"], color = 'red', label = '5')

plt.legend()

plt.xlabel("Radius(R/Ro)")

plt.ylabel("Luminosity(L/Lo)")
plt.figure(figsize = (10,6))

plt.title("Radius vs Luminosity Scatterplot for Star Type(0, 1, 2)")

plt.scatter(x = stars_0["Radius(R/Ro)"], y = stars_0["Luminosity(L/Lo)"], color = 'green', label = '0')

plt.scatter(x = stars_1["Radius(R/Ro)"], y = stars_1["Luminosity(L/Lo)"], color = 'purple', label = '1')

plt.scatter(x = stars_2["Radius(R/Ro)"], y = stars_2["Luminosity(L/Lo)"], color = 'grey', label = '2')

plt.legend()

plt.xlabel("Radius(R/Ro)")

plt.ylabel("Luminosity(L/Lo)")
plt.figure(figsize = (14,5))

plt.title('Average Radius for each Star Color')

sns.barplot(x = stars['Star color'], y = stars['Radius(R/Ro)'], color = (0.2,0.5,0.8))

plt.xticks(rotation=90)
plt.figure(figsize = (14,5))

plt.title('Average Luminosity for each Star Color')

sns.barplot(x = stars['Star color'], y = stars['Luminosity(L/Lo)'], color = (0.4, 0.8, 0.5))

plt.xticks(rotation=90)
plt.figure(figsize = (14,5))

plt.title('Average Temperature for each Star Color')

sns.barplot(x = stars['Star color'], y = stars['Temperature (K)'], color = 'orange')

plt.xticks(rotation=90)
plt.figure(figsize = (12,6))

plt.title('Average Radius for each Spectral Class')

sns.barplot(x = stars['Spectral Class'], y = stars['Radius(R/Ro)'], color = (0.2,0.5,0.8))
plt.figure(figsize = (12,6))

plt.title('Average Luminosity for each Spectral Class')

sns.barplot(x = stars['Spectral Class'], y = stars['Luminosity(L/Lo)'], color = (0.4,0.8,0.5))
plt.figure(figsize = (12,6))

plt.title('Average Temperature for each Spectral Class')

sns.barplot(x = stars['Spectral Class'], y = stars['Temperature (K)'], color = 'orange')
stars.head()
plt.figure(figsize= (10,6))

plt.title('Average Absolute magnitude for each Star Type')

sns.barplot(x = stars['Star type'], y = stars['Absolute magnitude(Mv)'], color = 'cyan')

plt.axhline(0, ls='-', color = 'black')

plt.xlabel('Star Type')
colours = cm.rainbow(np.linspace(0, 1, 6))

plt.figure(figsize = (10,6))

plt.title("Absolute magnitude vs Temperature Scatterplot for each Star Type")

plt.scatter(x = stars_0["Absolute magnitude(Mv)"], y = stars_0["Temperature (K)"], color = colours[0], label = '0')

plt.scatter(x = stars_1["Absolute magnitude(Mv)"], y = stars_1["Temperature (K)"], color = colours[1], label = '1')

plt.scatter(x = stars_2["Absolute magnitude(Mv)"], y = stars_2["Temperature (K)"], color = colours[2], label = '2')

plt.scatter(x = stars_3["Absolute magnitude(Mv)"], y = stars_3["Temperature (K)"], color = colours[3], label = '3')

plt.scatter(x = stars_4["Absolute magnitude(Mv)"], y = stars_4["Temperature (K)"], color = colours[4], label = '4')

plt.scatter(x = stars_5["Absolute magnitude(Mv)"], y = stars_5["Temperature (K)"], color = colours[5], label = '5')

plt.legend()

plt.xlabel("Absolute magnitude(Mv)")

plt.ylabel("Temperature (k)")
colours = cm.rainbow(np.linspace(0, 1, 6))

plt.figure(figsize = (10,6))

plt.title("Absolute magnitude vs Radius Scatterplot for each Star Type")

plt.scatter(x = stars_0["Absolute magnitude(Mv)"], y = stars_0["Radius(R/Ro)"], color = colours[0], label = '0')

plt.scatter(x = stars_1["Absolute magnitude(Mv)"], y = stars_1["Radius(R/Ro)"], color = colours[1], label = '1')

plt.scatter(x = stars_2["Absolute magnitude(Mv)"], y = stars_2["Radius(R/Ro)"], color = colours[2], label = '2')

plt.scatter(x = stars_3["Absolute magnitude(Mv)"], y = stars_3["Radius(R/Ro)"], color = colours[3], label = '3')

plt.scatter(x = stars_4["Absolute magnitude(Mv)"], y = stars_4["Radius(R/Ro)"], color = colours[4], label = '4')

plt.scatter(x = stars_5["Absolute magnitude(Mv)"], y = stars_5["Radius(R/Ro)"], color = colours[5], label = '5')

plt.legend()

plt.xlabel("Absolute magnitude(Mv)")

plt.ylabel("Radius(R/Ro)")
colours = cm.rainbow(np.linspace(0, 1, 6))

plt.figure(figsize = (10,6))

plt.title("Absolute magnitude vs Luminosity Scatterplot for each Star Type")

plt.scatter(x = stars_0["Absolute magnitude(Mv)"], y = stars_0["Luminosity(L/Lo)"], color = colours[0], label = '0')

plt.scatter(x = stars_1["Absolute magnitude(Mv)"], y = stars_1["Luminosity(L/Lo)"], color = colours[1], label = '1')

plt.scatter(x = stars_2["Absolute magnitude(Mv)"], y = stars_2["Luminosity(L/Lo)"], color = colours[2], label = '2')

plt.scatter(x = stars_3["Absolute magnitude(Mv)"], y = stars_3["Luminosity(L/Lo)"], color = colours[3], label = '3')

plt.scatter(x = stars_4["Absolute magnitude(Mv)"], y = stars_4["Luminosity(L/Lo)"], color = colours[4], label = '4')

plt.scatter(x = stars_5["Absolute magnitude(Mv)"], y = stars_5["Luminosity(L/Lo)"], color = colours[5], label = '5')

plt.legend()

plt.xlabel("Absolute magnitude(Mv)")

plt.ylabel("Luminosity (L/Lo)")
stars.head()
# Time for some prediction



#For shuffling the data by rows

stars = stars.sample(frac = 1)
features = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']

X = stars[features]

X = np.array(X)
Y = stars["Star type"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
# Now lets try to predict Star Type using KneighboourClassifier

KNC = KNeighborsClassifier(n_neighbors=3)

KNC.fit(x_train, y_train)
predictions = KNC.predict(x_test)

print('Accuracy for KNeighbourClassifier: ', accuracy_score(predictions, y_test))
# Now lets try to predict Star Type using SVM

clf = svm.SVC(kernel="linear", C=2)

clf.fit(x_train, y_train)
predictions2 = clf.predict(x_test)

print('Accuracy for SVM: ', accuracy_score(y_test, predictions2))
# Now lets try to predict Star Type using DecisionTreeClassifier

DTC = DecisionTreeClassifier(min_samples_leaf = 10)

DTC.fit(x_train, y_train)
predictions3 = DTC.predict(x_test)

print('Accuracy for DecisionTreeClassifier: ', accuracy_score(y_test, predictions2))
import numpy as np

import seaborn as sns

import pandas as pd

from sklearn import svm

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
bmi_health = pd.read_csv("../input/500-person-gender-height-weight-bodymassindex/500_Person_Gender_Height_Weight_Index.csv")
bmi_health.head()
plt.figure(figsize = (10,6))

plt.title("Average Weight for each Index value")

sns.barplot(x = bmi_health["Index"], y = bmi_health["Weight"], hue = bmi_health["Gender"])
plt.figure(figsize = (10,6))

plt.title("Average Height for each Index value")

sns.barplot(x = bmi_health["Index"], y = bmi_health["Height"], hue = bmi_health["Gender"])
#Dividing into two different Dataframes according to Gender

bmi_health_m = bmi_health[bmi_health["Gender"] == 'Male']

bmi_health_f = bmi_health[bmi_health["Gender"] == 'Female']
plt.figure(figsize = (10,6))

plt.hist(bmi_health_m["Index"], bins = [0. , 0.25, 1. , 1.25, 2. , 2.25, 3. , 3.25, 4. , 4.25, 5. , 5.25], label = "Male")

plt.hist(bmi_health_f["Index"]+0.25, bins = [0.25 , 0.5, 1.25 , 1.5, 2.25 , 2.5, 3.25 , 3.5, 4.25 , 4.5, 5.25 , 5.5], label = "Female")

plt.legend()

plt.xlabel("Index value")

plt.ylabel("Frequency")

plt.title("Distribution of Indexes over Gender")
#Making a BMI Column using the formula for BMI

bmi_health["BMI"] = round(bmi_health["Weight"]/((bmi_health["Height"])/100)**2, 2)   #Weight(Kg)/Height(m)^2
bmi_health.head()
plt.figure(figsize = (10,6))

plt.title("Average BMI for each Index value")

sns.barplot(x = bmi_health["Index"], y = bmi_health["BMI"], hue = bmi_health["Gender"])
plt.figure(figsize = (10,6))

plt.title("Index vs BMI Lineplot")

sns.lineplot(x = bmi_health["Index"], y = bmi_health["BMI"], color = "green", marker = 'o')

plt.axhline(18, ls='--')

plt.axhline(25, ls='--')

plt.text(4,26, "BMI = 25")

plt.text(4,19, "BMI = 18")
bmi_health_0 = bmi_health[bmi_health["Index"] == 0]

bmi_health_1 = bmi_health[bmi_health["Index"] == 1]

bmi_health_2 = bmi_health[bmi_health["Index"] == 2]

bmi_health_3 = bmi_health[bmi_health["Index"] == 3]

bmi_health_4 = bmi_health[bmi_health["Index"] == 4]

bmi_health_5 = bmi_health[bmi_health["Index"] == 5]
plt.title("Weight vs Height Scatterplot with Index values")

plt.scatter(x = bmi_health_0["Height"], y = bmi_health_0["Weight"], color = 'yellow', label = '0')

plt.scatter(x = bmi_health_1["Height"], y = bmi_health_1["Weight"], color = 'pink', label = '1')

plt.scatter(x = bmi_health_2["Height"], y = bmi_health_2["Weight"], color = 'green', label = '2')

plt.scatter(x = bmi_health_3["Height"], y = bmi_health_3["Weight"], color = 'orange', label = '3')

plt.scatter(x = bmi_health_4["Height"], y = bmi_health_4["Weight"], color = 'brown', label = '4')

plt.scatter(x = bmi_health_5["Height"], y = bmi_health_5["Weight"], color = 'red', label = '5')

plt.legend(loc = "center left", bbox_to_anchor = (1, 0.5))

plt.xlabel("Height")

plt.ylabel("Weight")
# Now lets try to predict Index value from Height and Weight using KneighboourClassifier

X = bmi_health[["Height", "Weight"]]

X = np.array(X)

Y = bmi_health["Index"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)
KNC = KNeighborsClassifier(n_neighbors=3)

KNC.fit(x_train, y_train)
predictions = KNC.predict(x_test)

accuracy_score(predictions, y_test)
# Now lets try to predict Index value from Height and Weight using SVM

clf = svm.SVC(kernel="linear", C=2)

clf.fit(x_train, y_train)
predictions2 = clf.predict(x_test)

accuracy_score(y_test, predictions2)
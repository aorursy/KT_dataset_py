import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.metrics import confusion_matrix, accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
# below table shows the first couple of features for different samples



df.head()
# following graph shows the range of quality and count of the target variable

sns.countplot(x = "quality", data = df, palette = "Set3")

plt.title("Quality Count")

plt.show()
sns.catplot(x = "quality", y = "fixed acidity", data = df, kind = "boxen")

plt.title("Fixed Acidity vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "volatile acidity", data = df, kind = "boxen")

plt.title("Volatile Acidity vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "citric acid", data = df, kind = "boxen")

plt.title("Citric Acidity vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "residual sugar", data = df, kind = "boxen")

plt.title("Residual Sugar vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "chlorides", data = df, kind = "boxen")

plt.title("Chlorides vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "free sulfur dioxide", data = df, kind = "boxen")

plt.title("Free Sulfur Dioxide vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "total sulfur dioxide", data = df, kind = "boxen")

plt.title("Total Sulfur Dioxide vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "density", data = df, kind = "boxen")

plt.title("Density vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "pH", data = df, kind = "boxen")

plt.title("pH vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "sulphates", data = df, kind = "boxen")

plt.title("Sulphates vs. Quality")

plt.show()
sns.catplot(x = "quality", y = "alcohol", data = df, kind = "boxen")

plt.title("Alcohol vs. Quality")

plt.show()
# "rating" is the new column which is correlated with wine quality 

rating = []

for i in df['quality']:

    if i >= 3 and i < 5:

        rating.append('1')

    elif i >= 5 and i < 8:

        rating.append('2')

    elif i == 8:

        rating.append('3')

df['rating'] = rating





# we can see the total count of elements in the new grading system below

Counter(df['rating'])
# defined target and feature arrays as 'rating' being the target and other 11 colums as features



x = df.iloc[:,:11]

y = df['rating']



print(x.head(5))

print("-----------------------------------------")

print(y.head(5))
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_std = sc.fit_transform(x)
from sklearn.decomposition import PCA



pca = PCA()

x_pca = pca.fit_transform(x_std)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size = 0.3,random_state = 16)

print("Training Feature Set:", x_train.shape,"\nTraining Output Set:", y_train.shape, "\n\nTest Feature Set:",x_test.shape, "\nTest Output Set", y_test.shape)
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(random_state = 16)

model.fit(x_train,y_train)

prediction = model.predict(x_test).reshape(-1,1)
accuracy = accuracy_score(y_test, prediction)

print("The Decision Tree Classifier test is %",accuracy*100, "accurate.")
from sklearn.ensemble import RandomForestClassifier



model_2 = RandomForestClassifier(n_estimators = 100, random_state = 16)

model_2.fit(x_train, y_train)

prediction_2 = model_2.predict(x_test).reshape(-1,1)



accuracy_2 = accuracy_score(y_test, prediction_2)

print("The Random Forest Classifier test is %",accuracy_2*100, "accurate.")
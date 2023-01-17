import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
def overview():

    data = pd.read_csv("../input/80-cereals/cereal.csv")

    print("The first 5 rows of data are:\n")

    print(data.head)

    print("\n\n\nDataset has {} rows and {} columns".format(data.shape[0], data.shape[1]))

    print("\n\n\nDatatype: \n")

    print(data.dtypes)

    print("\n\n\nThe number of null values for each column are: \n")

    print(data.isnull().sum())

    print("\n\n\nData summary: \n")

    print(data.describe())

    return data

    

data = overview()
# Count the number of -1 in carbo, sugars and potass column

data[data == -1].count(axis=0)
# Remove affected rows

data = data[(data.carbo >= 0) & (data.sugars >= 0) & (data.potass >= 0)]

data[data == -1].count(axis=0)
# Counting number of manufacturers 

data['mfr'].value_counts()
plt.figure(figsize = (10, 8))

sns.boxplot(data = data, x = "mfr", y = "rating")
# Finding out what are the 4 Nabisco cereals that have high ratings

data.loc[data['mfr'] == "N"]
# Lowest rating cereal

data.loc[data['rating'] == min(data.rating)]
# Best rating cereal

data.loc[data['rating'] == max(data.rating)]
print(data['type'].value_counts())

data.loc[data['type'] == "H"]
fig = plt.figure()



plt.rcParams['figure.figsize'] = [10,10]

plt.rcParams["font.weight"] = "bold"



fontdict={'fontsize': 25,

          'weight' : 'bold'}



fontdicty={'fontsize': 18,

          'weight' : 'bold',

          'verticalalignment': 'baseline',

          'horizontalalignment': 'center'}



fontdictx={'fontsize': 18,

          'weight' : 'bold',

          'horizontalalignment': 'center'}



plt.subplots_adjust(wspace=0.2, hspace=0.2)



fig.suptitle('Can calories and macronutrients affect cereal rating?', fontsize=25,fontweight="bold", color="black", 

             position=(0.5,1.01))



ax1 = fig.add_subplot(221)

ax1.scatter('calories', 'rating', data= data, c="green")

ax1.set_title('Calories', fontdict=fontdict, color="green")





ax2 = fig.add_subplot(222)

ax2.scatter('fat', 'rating', data=data, c="orange")

ax2.set_title('Fat', fontdict=fontdict, color="orange")





ax3 = fig.add_subplot(223)

ax3.scatter('protein', 'rating', data=data, c="brown")

ax3.set_title('Protein', fontdict=fontdict, color="brown")



ax4 = fig.add_subplot(224)

ax4.scatter('carbo', 'rating', data=data, c="blue")

ax4.set_title("Carbs", fontdict=fontdict, color="blue")
fig = plt.figure()



plt.rcParams['figure.figsize'] = [10,10]

plt.rcParams["font.weight"] = "bold"



fontdict={'fontsize': 25,

          'weight' : 'bold'}



fontdicty={'fontsize': 18,

          'weight' : 'bold',

          'verticalalignment': 'baseline',

          'horizontalalignment': 'center'}



fontdictx={'fontsize': 18,

          'weight' : 'bold',

          'horizontalalignment': 'center'}



plt.subplots_adjust(wspace=0.2, hspace=0.2)



fig.suptitle('Can micronutrients affect cereal rating?', fontsize=25,fontweight="bold", color="black", 

             position=(0.5,1.01))



ax1 = fig.add_subplot(221)

ax1.scatter('sodium', 'rating', data= data, c="green")

ax1.set_title('Sodium', fontdict=fontdict, color="green")





ax2 = fig.add_subplot(222)

ax2.scatter('fat', 'rating', data=data, c="orange")

ax2.set_title('Potassium', fontdict=fontdict, color="orange")





ax3 = fig.add_subplot(223)

ax3.scatter('fiber', 'rating', data=data, c="brown")

ax3.set_title('Fiber', fontdict=fontdict, color="brown")



ax4 = fig.add_subplot(224)

ax4.scatter('vitamins', 'rating', data=data, c="blue")

ax4.set_title("Vitamins", fontdict=fontdict, color="blue")
fig = plt.figure()



plt.rcParams['figure.figsize'] = [10,10]

plt.rcParams["font.weight"] = "bold"



fontdict={'fontsize': 25,

          'weight' : 'bold'}



fontdicty={'fontsize': 18,

          'weight' : 'bold',

          'verticalalignment': 'baseline',

          'horizontalalignment': 'center'}



fontdictx={'fontsize': 18,

          'weight' : 'bold',

          'horizontalalignment': 'center'}



plt.subplots_adjust(wspace=0.2, hspace=0.2)



fig.suptitle('Can sugar, display shelf level, weight and cups affect rating?', fontsize=25,fontweight="bold", color="black", 

             position=(0.5,1.01))



ax1 = fig.add_subplot(221)

ax1.scatter('sugars', 'rating', data= data, c="green")

ax1.set_title('Sugar', fontdict=fontdict, color="green")





ax2 = fig.add_subplot(222)

ax2.scatter('shelf', 'rating', data=data, c="orange")

ax2.set_title('Display shelf level', fontdict=fontdict, color="orange")





ax3 = fig.add_subplot(223)

ax3.scatter('weight', 'rating', data=data, c="brown")

ax3.set_title('Ounces per serving', fontdict=fontdict, color="brown")



ax4 = fig.add_subplot(224)

ax4.scatter('cups', 'rating', data=data, c="blue")

ax4.set_title("Cups per serving", fontdict=fontdict, color="blue")
# Use label encoder on categorical data

le = LabelEncoder()

 

# Implementing LE on mfr

le.fit(data.mfr.drop_duplicates()) 

data.mfr = le.transform(data.mfr)



# Implementing LE on type

le.fit(data.type.drop_duplicates()) 

data.type = le.transform(data.type)
plt.figure(figsize=(15,10))

 

corrMatrix = data.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
# Creating training and testing dataset

y = data['rating']

X = data.drop(['rating', 'name'], axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)



lr = LinearRegression().fit(X_train,y_train)

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



print(lr.score(X_test,y_test))
# Packages for analysis





import pandas as pd



import numpy as np

from sklearn import svm



# Packages for visuals

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(font_scale=1.2)



# Allows charts to appear in the notebook

%matplotlib inline



# Pickle package

import pickle
# Read in muffin and cupcake ingredient data

recipes = pd.read_csv('../input/iris/Iris.csv')

recipes.drop(recipes[recipes["Species"] == "Iris-virginica"].index, inplace=True)





recipes=recipes.sample(n = 19)

recipes

# Plot two ingredients

sns.lmplot('SepalLengthCm', 'SepalWidthCm', data=recipes, hue='Species',

           palette='Set1', fit_reg=False, scatter_kws={"s": 70});
# Specify inputs for the model

# ingredients = recipes[['Flour', 'Milk', 'Sugar', 'Butter', 'Egg', 'Baking Powder', 'Vanilla', 'Salt']].as_matrix()

ingredients = recipes[['SepalLengthCm','SepalWidthCm']].to_numpy()

type_label = np.where(recipes['Species']=="Iris-versicolor", 0, 1)

print(np.unique(type_label))



# Feature names

recipe_features = recipes.columns.values[1:].tolist()

recipe_features
model = svm.SVC(kernel='linear')

model.fit(ingredients, type_label)



print("SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)")
model = svm.SVC(kernel='linear')

#model = svm.SVC(kernel='rbf', C=1, gamma=2**-5)

model.fit(ingredients, type_label)


model = svm.SVC(kernel='linear')

model.fit(ingredients, type_label)
# Get the separating hyperplane

w = model.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(4, 7)

yy = a * xx - (model.intercept_[0]) / w[1]



# Plot the parallels to the separating hyperplane that pass through the support vectors

b = model.support_vectors_[0]

yy_down = a * xx + (b[1] - a * b[0])

b = model.support_vectors_[-1]

yy_up = a * xx + (b[1] - a * b[0])
# Plot the hyperplane

sns.lmplot('SepalLengthCm', 'SepalWidthCm', data=recipes, hue='Species', palette='Set1', fit_reg=False, scatter_kws={"s": 70})

plt.plot(xx, yy, linewidth=2, color='black');
# Look at the margins and support vectors

sns.lmplot('SepalLengthCm', 'SepalWidthCm', data=recipes, hue='Species', palette='Set1', fit_reg=False, scatter_kws={"s": 70})

plt.plot(xx, yy, linewidth=2, color='black')

plt.plot(xx, yy_down, 'k--')

plt.plot(xx, yy_up, 'k--')

plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],

            s=80, facecolors='none');
# Create a function to guess when a recipe is a muffin or a cupcake

def muffin_or_cupcake( SepalLengthCm, SepalWidthCm):

    if(model.predict([[SepalLengthCm, SepalWidthCm]]))==0:

        print('You\'re looking at a Iris-versicolor')

    else:

        print('You\'re looking at a Iris-setosa')
# Predict if 50 parts flour and 20 parts sugar

muffin_or_cupcake(4.9, 3.1)
# Plot the point to visually see where the point lies

sns.lmplot('SepalLengthCm', 'SepalWidthCm', data=recipes, hue='Species', palette='Set1', fit_reg=False, scatter_kws={"s": 70})

plt.plot(xx, yy, linewidth=2, color='black')

plt.plot(3, 7, 'yo', markersize='9');
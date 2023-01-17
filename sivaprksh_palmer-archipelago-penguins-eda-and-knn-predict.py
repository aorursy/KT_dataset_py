# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns
#reading in the data



penguins = pd.read_csv('/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')

penguins.head(5)
penguins.shape
#Checking for missing values

penguins.info()

penguins.isna().sum()

#missing values present in most columns, 'Sex' Column has the highest (10)
#exploring the data

sns.countplot(x = "species", data = penguins)
#Adelie are the highest in number followed by Gentoo and Chintrap
sns.countplot(x = "island", data = penguins)
#most of the Penguins belong to Biscoe island and least are from Torgersen
#culmen length and depth wrt species 



fig,axs = plt.subplots(ncols = 2)

fig.tight_layout()



sns.boxplot(y= 'culmen_length_mm', x = 'species', data = penguins, ax= axs[0])

sns.boxplot(y= 'culmen_depth_mm', x = 'species', data = penguins, ax= axs[1])
#flipper length by species



sns.boxplot(x = 'species', y = 'flipper_length_mm', data = penguins)
#Gentoo species has the longest flippers while Adelie has the shortest
#body mass of different Species



sns.boxplot(x = 'species', y = 'body_mass_g', data = penguins)
#Gentoo are the heaviest of the three species whereas Adelie and Chinstrap weigh around the same
#Species count by sex



sns.countplot(x='sex', data = penguins, hue = 'species')
#all species are evenly distributed across the genders
#Let us try to plot culmen length & depth. Let us see if we can get a clear grouping of species 



sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm', hue = 'species', data = penguins)
#we can observe that Adelie and Gentoo form two distinctive grouping with Chinstrap being a little more spread out
#Handling missing values using mean for numerical variables



penguins["culmen_length_mm"] = penguins["culmen_length_mm"].fillna(value = penguins["culmen_length_mm"].mean())

penguins["culmen_depth_mm"] = penguins["culmen_depth_mm"].fillna(value = penguins["culmen_depth_mm"].mean())

penguins["flipper_length_mm"] = penguins["flipper_length_mm"].fillna(value = penguins["flipper_length_mm"].mean())

penguins["body_mass_g"] = penguins["body_mass_g"].fillna(value = penguins["body_mass_g"].mean())



penguins.isna().sum()
#filling in missing values of categorical variable 'Sex'



penguins["sex"] = penguins["sex"].fillna("FEMALE")



penguins.isna().sum()
#Let us try to predict the Penguin Species using KNN now

#importing necessary modules



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV 

#encoding categorical variables

y = penguins["species"]

penguins_main = penguins.iloc[:,1:]

X = pd.get_dummies(penguins_main)

X.head()
#splitting the data into train and test



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=21, stratify=y)



X_test.shape
#creating a GridSearch for finding out best value of n_neighbors for KNN



from sklearn.model_selection import GridSearchCV



param_grid = {'n_neighbors': np.arange(1,10)}

knn = KNeighborsClassifier()



knn_cv = GridSearchCV(knn, param_grid, cv = 5)

#fitting on train data to find out best parameters



knn_cv.fit(X_train, y_train)
#finding out the best parameter



knn_cv.best_params_
#finding out best train accuracy



knn_cv.best_score_
#predicting test data with best parameters



knn_best = knn_cv.best_estimator_



y_pred = knn_best.predict(X_test)
#Checking accuracy on test dataset



knn_best.score(X_test, y_test)
#We have predicted the Species of Penguins in the test data with a 74% accuracy



#Kindly upvote if you find this notebook useful. Cheers !!
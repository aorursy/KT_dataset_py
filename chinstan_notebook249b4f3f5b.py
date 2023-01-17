#imports 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
food_df = pd.read_csv("../input/en.openfoodfacts.org.products.tsv", delimiter='\t');

#food_df = pd.read_csv("FoodFacts.csv")



food_df.head()
food_df.info()
#split train/validate/test set



train, validate, test = np.split(food_df.sample(frac=1), [int(.6*len(food_df)), int(.8*len(food_df))])

print("Train: %s\nTest: %s\nValidate: %s" % (train.shape, test.shape, validate.shape))
#still working on this so I wanted to practice with a small dataset

df = validate

a, b = np.split(df.sample(frac=1), [int(.1*len(df))])

a.shape
a.head()
#clean data

#remove from entry data that has empty values in the energy

a = a[~pd.isnull(a['energy_100g'])]

a.drop(["code", "url", "creator", "created_t", "created_datetime", "last_modified_t", "last_modified_datetime"], axis=1)

a.describe()
## analyze data 

# which features are available in the dataset? 

print(a.columns.values)
a.describe(exclude=['O'])
#analyze data, get foods with only grades a, b, c, d, e

#get only their carb/protein/fat as preliminary analysis



only_a = a.loc[a['nutrition_grade_fr'] == 'a']

only_b = a.loc[a['nutrition_grade_fr'] == 'b']

only_c = a.loc[a['nutrition_grade_fr'] == 'c']

only_d = a.loc[a['nutrition_grade_fr'] == 'd']

only_e = a.loc[a['nutrition_grade_fr'] == 'e']



mac_only_a = only_a[['carbohydrates_100g', 'proteins_100g', 'fat_100g']]

mac_only_b = only_b[['carbohydrates_100g', 'proteins_100g', 'fat_100g']]

mac_only_c = only_c[['carbohydrates_100g', 'proteins_100g', 'fat_100g']]

mac_only_d = only_d[['carbohydrates_100g', 'proteins_100g', 'fat_100g']]

mac_only_e = only_e[['carbohydrates_100g', 'proteins_100g', 'fat_100g']]
mac_only_a.describe()
order = sorted(a.nutrition_grade_fr.unique()[1:])

ax = sns.boxplot(x="nutrition_grade_fr", y="fat_100g", data=a, order=order)
#order = sorted(a.nutrition_grade_fr.unique()[1:])

order = ["a", "b", "c", "d", "e"]

ax = sns.boxplot(x="nutrition_grade_fr", y="sugars_100g", data=a, order=order)

sns.plt.title('Sugar vs Grade')
ax = sns.boxplot(x="nutrition_grade_fr", y="sugars_100g", data=a, order=order)

sns.plt.title('Fat vs Grade')

ax = sns.boxplot(x="nutrition_grade_fr", y="proteins_100g", data=a, order=order)

sns.plt.title('Protein vs Grade')

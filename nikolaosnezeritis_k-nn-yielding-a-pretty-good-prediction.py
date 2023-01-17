import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import numpy as np





# Reading the data. Using 'id' as the column index

breast_cancer_df = pd.read_csv('../input/data.csv', index_col='id', header=0)



# Preliminary EDA 

print(breast_cancer_df.info())



# Dropping the last column because it is full of NaNs

breast_cancer_df.drop('Unnamed: 32', axis=1, inplace=True)



# Replacing with numeric values in order to make plots

breast_cancer_df['diagnosis'] = breast_cancer_df['diagnosis'].replace({'M':1, 'B':0})



breast_cancer_df.info()



# Converting the diagnosis column to categorical to save space

breast_cancer_df['diagnosis'] = breast_cancer_df['diagnosis'].astype('category')

heatmap_data = breast_cancer_df.corr()

fig, ax = plt.subplots(figsize = (15,15))

sns.heatmap(heatmap_data, cbar = True, annot=True, ax=ax)

plt.show()

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split



breast_cancer_df_scaled = scale(np.array(breast_cancer_df))

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_df_scaled,

                                                    breast_cancer_df.iloc[:, 0],

                                                    test_size=0.3, stratify=breast_cancer_df.iloc[:, 0])



test_accuracy = np.empty(25)

train_accuracy = np.empty(25)

for n in range(1,25):

    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    test_accuracy[n] = knn.score(X_test, y_test) 

    train_accuracy[n] = knn.score(X_train, y_train)



plt.style.use('ggplot')

plt.plot(np.linspace(1,25,25),test_accuracy, color ='red', label='test accuracy')

plt.plot(np.linspace(1,25,25),train_accuracy, color = 'blue', label='train accuracy')

plt.xlim((2, 25))

plt.xlabel("Number of neighbours")

plt.ylabel("Prediction Accuracy")

plt.title("Prediction Accuracy Vs. Number of Neighbours")

plt.ylim((0.85, 1.05))

plt.legend(loc='upper right') 

plt.show()

# By inspection, k =4 seems like a good choice

print (test_accuracy[4])
# Trying to see if it is possible to decompose into fewer components

from sklearn import decomposition

pca = decomposition.PCA()

pca.fit(breast_cancer_df)

df_reduced = pca.fit_transform(breast_cancer_df)

df_reduced.shape
# Checking to see if our inspection for 4 neighbors is correct

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report





steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier())]

pipeline = Pipeline(steps)

parameters = {'knn__n_neighbors':np.arange(1,50)}

cv = GridSearchCV(pipeline,parameters)

cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)

print(cv.best_params_)

print(cv.score(X_test, y_test))

print(classification_report(y_test, y_pred))
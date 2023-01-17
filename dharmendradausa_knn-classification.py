import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
fruits = pd.read_table("../input/fruits-with-colors-dataset/fruit_data_with_colors.txt")
fruits.head(0)
lookup_name = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
lookup_name
x = fruits[['mass','width','height','color_score']]
y = fruits['fruit_label']
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)     #traning the classifier
knn.score(x_test,y_test) #testing the classifier and checking accuracy
prediction = knn.predict([[20,4,6,7]])
lookup_name[prediction[0]]
prediction = knn.predict([[100,4,6,7]])
lookup_name[prediction[0]]

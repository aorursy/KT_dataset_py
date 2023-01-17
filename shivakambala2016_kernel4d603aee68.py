import numpy as np
import pandas as pd
Iris = pd.read_csv("../input/Iris_data.csv",na_values = "n/a")
Iris.head()
Iris.Species.unique()
k = Iris[Iris.Species == 'Iris-setosa']
k
Versicolor_petal_length = Iris[Iris.Species == 'Iris-versicolor']
Versicolor_petal_length = Versicolor_petal_length.iloc[:,2]
Versicolor_petal_length = np.array(Versicolor_petal_length)
Versicolor_petal_length
differences = Versicolor_petal_length - np.mean(Versicolor_petal_length) 
differences
differences_Squares = differences * differences
differences_Squares

avg_differences_Squares = np.average(differences_Squares)
avg_differences_Squares
np.var(Versicolor_petal_length)
std = np.sqrt(avg_differences_Squares)
std
np.std(Versicolor_petal_length)

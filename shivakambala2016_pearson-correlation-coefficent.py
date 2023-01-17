
# import libraries
import numpy as np
import pandas as pd

Iris = pd.read_csv("../input/Iris_data.csv",na_values="n/a")
Iris.head()
Iris_vc = Iris[Iris.Species == 'Iris-versicolor']
Iris_vc
Iris_versicolor_petal_length = Iris_vc.iloc[:,2]
Iris_versicolor_petal_width = Iris_vc.iloc[:,3]
Iris_versicolor_petal_length = np.array(Iris_versicolor_petal_length)
Iris_versicolor_petal_width = np.array(Iris_versicolor_petal_width)
np.corrcoef(Iris_versicolor_petal_length,Iris_versicolor_petal_width)
def pearson_r(x,y):
    
    corr_mat = np.corrcoef(x,y)
    return corr_mat[0,1]
    
r = pearson_r(Iris_versicolor_petal_length,Iris_versicolor_petal_width)
print(r)

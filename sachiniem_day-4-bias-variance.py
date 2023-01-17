import pandas as pd

curve = pd.read_csv("../input/curve-dataset/curve.csv")

student = pd.read_csv("../input/students/student.csv")



import numpy as np

import seaborn as sn

import matplotlib.pyplot as plt

curve.head()

def fit_poly(degree):

    p=np.polyfit(curve.x,curve.y,deg=degree)

    curve['fit']=np.polyval(p,curve.x)

    sn.regplot(curve.x,curve.y,fit_reg=False)

    return plt.plot(curve.x,curve.fit,label='fit')

fit_poly(3)

plt.xlabel("x values")

plt.ylabel("y values")
from sklearn.model_selection import train_test_split

from sklearn import metrics



train_X, test_X, train_y, test_y = train_test_split( curve.x, curve.y, test_size = 0.40, random_state = 100 )

rmse_df = pd.DataFrame( columns = ["degree", "rmse_train", "rmse_test"] )

def get_rmse( y, y_fit ):

    return np.sqrt( metrics.mean_squared_error( y, y_fit ) )

for i in range( 1, 15 ):

    # fitting model

    p = np.polyfit( train_X, train_y, deg = i )

    rmse_df.loc[i-1] = [ i,

                            get_rmse( train_y, np.polyval( p, train_X ) ),

                            get_rmse( test_y, np.polyval( p, test_X ) ) ]

rmse_df
plt.plot(rmse_df.degree,rmse_df.rmse_train, label="RMSE_TRAIN",c="red")

plt.plot(rmse_df.degree,rmse_df.rmse_test,label="RMSE_TEST",c="green")

plt.xlabel("Degree")

plt.ylabel("RMSE")

plt.legend()
student.head()

subset = student.drop(['Name'], axis=1)

subset.head(14)



testdata = [1, 1]



lst_dist = []



for ind in subset.index:

    dist_row = np.sqrt(np.square(testdata[0] - subset['Aptitude'][ind]) + np.square(testdata[1] - subset['Communication'][ind]))

    lst_dist.append([dist_row, subset['Class'][ind]]) 

    

df = pd.DataFrame(lst_dist)

df.columns = ('Distance', 'class')

df_sorted = df.sort_values('Distance')

df

k = 4

df_sorted_kval = df_sorted.head(k)

print(df_sorted_kval)
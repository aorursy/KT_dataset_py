# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sn

import matplotlib.pyplot as plt

curve = pd.read_csv("../input/curve.csv")
def fit_poly(degree):

    p=np.polyfit(curve.x,curve.y,deg=degree)

    curve['fit']=np.polyval(p,curve.x)

    sn.regplot(curve.x,curve.y,fit_reg=False)

    return plt.plot(curve.x,curve.fit,label='fit')

fit_poly(2)

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


plt.plot( rmse_df.degree,

            rmse_df.rmse_train,

            label='RMSE on Training Set',

            color = 'r' )



plt.plot( rmse_df.degree,

            rmse_df.rmse_test,

            label='RMSE on Test Set',

            color = 'g' )



plt.legend(bbox_to_anchor=(1.05, 1),

            loc=2,

            borderaxespad=0.);

plt.xlabel("Model Degrees")

plt.ylabel("RMSE");




fit_poly( 10 );

plt.xlabel("x values")

plt.ylabel("y values");



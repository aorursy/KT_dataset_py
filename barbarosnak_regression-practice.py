# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings        

# ignore filters

warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.



plt.style.use('ggplot') # style of plots. ggplot is one of the most used style, I also like it.

# Any results you write to the current directory are saved as output.
bfop_df1=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")



bfop_df2=pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
bfop_df1.head(10)
bfop_df1.info()
bfop_df2.head(10)
bfop_df2.info()
from sklearn.linear_model import LinearRegression



linear_reg=LinearRegression()



x=bfop_df1.pelvic_incidence.values.reshape(-1,1)

y=bfop_df1.pelvic_radius.values.reshape(-1,1)



linear_reg.fit(x,y)



minimum=int(min(x))

maximum=int(max(x))



#array=linear_reg.predict(x)



plt.scatter(x,y)



y_head=linear_reg.predict(x)



plt.plot(x,y_head,color="blue")

plt.show()



print("r_score:",r2_score(y,y_head))
from sklearn.linear_model import LinearRegression



multiple_linear_regression=LinearRegression()



x=bfop_df2.iloc[:,[0,1]].values # pelvic incidence and pelvic tilt numeric

y=bfop_df2.pelvic_radius.values.reshape(-1,1)







multiple_linear_regression.fit(x,y)



array=[]



for i in np.array(range(35,88)):

    

    print(multiple_linear_regression.predict([[i,i]]),"\n")



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



polynomial_regression=PolynomialFeatures(degree=4)



x=bfop_df1.lumbar_lordosis_angle.values.reshape(-1,1)

y=bfop_df1.sacral_slope.values.reshape(-1,1)



x_polynomial=polynomial_regression.fit_transform(x)



linear_regression2=LinearRegression()

linear_regression2.fit(x_polynomial,y)



y_head=linear_regression2.predict(x_polynomial)



plt.scatter(x,y)

plt.xlabel("Lumbar Lordosis Angle")

plt.ylabel("Sacral Slope")

plt.plot(x,y_head,color="blue")

plt.show()



print("r_score:",r2_score(y,y_head))
from sklearn.tree import DecisionTreeRegressor



x=bfop_df2.pelvic_incidence.values.reshape(-1,1)

y=bfop_df2.pelvic_radius.values.reshape(-1,1)



tree_reg=DecisionTreeRegressor()

tree_reg.fit(x,y)



x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=tree_reg.predict(x_)

y_head2=tree_reg.predict(x)



plt.scatter(x,y)

plt.plot(x_,y_head,color="blue")

plt.show()



print("r_score:",r2_score(y,y_head2))
from sklearn.ensemble import RandomForestRegressor



rf=RandomForestRegressor(n_estimators=100, random_state=42)



x=bfop_df2.lumbar_lordosis_angle.values.reshape(-1,1)

y=bfop_df2.sacral_slope.values.reshape(-1,1)



rf.fit(x,y)



y_head=rf.predict(x)



plt.scatter(x,y)

plt.plot(x,y_head,color="blue")

plt.show()



print("r_score:",r2_score(y,y_head))
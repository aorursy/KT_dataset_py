!pip install plotnine
import warnings  

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

from plotnine import *





x = np.linspace(-10, 10,100)

y = (10*x + 20) + np.random.normal(0,5 , 100)



df = pd.DataFrame({"x" : x , "y" : y})





(ggplot(df , aes(x="x" , y="y"))

  + geom_point()

)





def l1(m,x,y):

    return np.sum(np.abs(m[0]*x + m[1]-y))



def l2(m,x,y):

    return np.sum(np.power(m[0]*x + m[1]-y , 2))
from scipy.optimize import minimize



def Compare(df , doPrint=True):

    x = df["x"]

    y = df["y"]



    l1reg = minimize(l1 ,[0,0] , args=(x,y))

    l1regdf = pd.DataFrame({"x" : x , "y" : l1reg.x[0]*x + l1reg.x[1] })



    l2reg = minimize(l2 , [0,0] , args=(x,y))

    l2regdf = pd.DataFrame({"x" : x , "y" :  l2reg.x[0]*x + l2reg.x[1] })





    p = (ggplot(df )

      + geom_point(aes(x="x" , y="y") , color="blue")

      + geom_line(l1regdf , aes(x="x" , y="y") , color="red")

      + geom_line(l2regdf , aes(x="x" , y="y") , color="green")

    )

    if doPrint:

        print(p)

        print( f"l1 : {l1reg.x} ,  l2: {l2reg.x}")

    return l1reg.x , l2reg.x





coeffs = Compare(df)

# Add an outlier

odf = df.append(pd.DataFrame({"x" : [10] , "y" : [-100]}))



Compare(odf)



sdf = df.copy()



params = []

for y in range(10):

    sdf = sdf.sample(frac=0.8)

    params.append(Compare(sdf , False))





l1var = np.std([ l1[0] for l1,l2 in params])

l2var = np.std([ l2[0] for l1,l2 in params])

print(f"l1 - coefficient std : {l1var} , l2 - coefficient std : {l2var}")





# % Markdown

## L2 is more stable than L1
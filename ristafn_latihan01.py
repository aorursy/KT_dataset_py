import pandas as pd

import numpy as np

from pandas import DataFrame
pd.Series(np.random.rand(5), index=['Rista','Fajar','Dika','Edka','Faris'])
a = pd.Series([1, 2, 3]).array

b = pd.Series([2, 3, 4]).array

print (a,b)
cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],

        'Price': [22000,25000,27000,35000]

        }



df = pd.DataFrame(cars, columns = ['Brand', 'Price'])



max1 = df['Price'].max()

print (max1,df)
numbers = {'set_of_numbers': [1,2,3,4,5,6,7,8,9,10,np.nan,np.nan]}

df = DataFrame(numbers,columns=['set_of_numbers'])

print (df)
df.loc[df['set_of_numbers'].isnull(), 'set_of_numbers'] = 0

print (df)
def f(x):

    return (16*(x**3))-(22*x)+9

a = np.zeros(40)

b = np.zeros(40)

c = np.zeros(40)



a[0] = 0.7

b[0] = 1



c[0] = (a[0]+b[0])/2
for n in range(39):

    if f(a[n])*f(c[n])<0:

        a[n+1]=a[n]

        b[n+1]=c[n]

    else:

        a[n+1]=c[n]

        b[n+1]=b[n]

    c[n+1]=(a[n+1]+b[n+1])/2

    print(c[n+1],f(c[n+1]))

        
c[-1]
f(c[-1])
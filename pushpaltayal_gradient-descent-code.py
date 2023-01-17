import pandas as pd

import numpy as np

#Creating a dataframe to be used for calculating Gradient Descent

data = {'Name': ["Ram", "Laxman", "Bharat", "Shatrugan", "Krish"],

       'Maths': [67,70,79,80,90],

       'English': [34,35,36,89,90]}

df= pd.DataFrame(data, columns= ['Name', 'Maths', 'English'])

df
def GradientDescent(x,y):

    m_curr = b_curr = 0

    rate= 0.1

    k= 10

    n= len(x)

    for i in range(k):

        y_pred= m_curr * x + b_curr

        md= -(2/n)*(sum*(x(y-y_pred)))

        bd= -(2/n)*(sum*(y-y_pred))

        m_curr = m_curr - rate* md

        b_curr = b_curr - rate* bd

        cost= (1/n)* sum* (val ** 2 for val in (y-y_pred))

        print("m_curr= ", "b_curr=", "cost= ", format(m_curr,b_curr, cost))

    
x= np.array([1,2,3,4,5])

y= np.array([2,5,7,9,8])

GradientDescent(x,y)
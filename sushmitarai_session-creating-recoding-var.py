# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# importing pandas as pd 

import pandas as pd 

  



dict = {'Name':["aparna", "pankaj", "sudhir", "Geeku"], 

        'degree': ["MBA", "BCA", "M.Tech", "MBA"], 

        'Math1_score':[90, 40, 80, 98],

        'Math2_score':[76,34,67,78],

        'Sci1_score':[99,78,90,89],

        'Sci2_score':[70,69,86,78],

        'produc_bought':["pensil","Books","Pens","colorBox"],

        'Quantity':[30,4,10,12],

         'Rate':[5,400,10,30]} 

  

df = pd.DataFrame(dict) 
#creating new variables



# (1) different method of adding the observations of two variables

df['Maths_total'] = df.Math1_score+ df.Math2_score 

df['Total_MathsScore'] = df.apply(lambda row: row.Math1_score + row.Math2_score, axis = 1) 

# (2) subtracting the observations of two variables

df['subt_math12']= df['Math1_score']- df['Math2_score']

df['subM12'] = df.apply(lambda row: row.Math1_score - row.Math2_score, axis = 1) 
# (3) Multipling the observations of two variables 

df['Total_cost'] = df.apply(lambda row: row.Quantity * row.Rate, axis = 1) 

df['Toral_cost']= df['Quantity'] * df['Rate']

# (4) Dividing the values of two variables and storing it in new variable

df['Quant_1_cal'] = df.apply(lambda row: row.Total_cost/row.Rate, axis = 1) 

df['Quant_1']= df['Total_cost']/df['Rate']
#expessions

#An expression is a combination of values, variables, operators, and calls to functions.

dict2 ={'Length': [23,45,67,89],

        'Breadth':[16,23,45,36]}



df2=pd.DataFrame(dict2)



df2['Area']=df2['Length']* df2['Breadth']
#Recoding variable

def grades(Sci1_score):

    if Sci1_score < 80:

        return "C"

    elif 80 <= Sci1_score < 90:

        return "B"

    elif 90 <= Sci1_score:

        return "A"



df['SCI_grade'] = df['Sci1_score'].apply(grades)

#renaming variable (First method)

new_data = df.rename(columns = {"produc_bought": "Item_purchased", 

                                  "Sci2_score":"Science2_score"}) 



#renaming variable (second method)

df.rename(index=str, columns={"produc_bought": "Item_purchased", "degree": "Degree"})





# logical operation

import numpy as np

 

data_fr = {

    'State':['Arizona AZ','Georgia GG','Newyork NY','Indiana IN','Florida FL'],

   'Score1':[4,47,55,74,31],

   'Score2':[5,67,54,56,12]}

 

data_fr = pd.DataFrame(data_fr,columns=['State','Score1','Score2'])

print(data_fr)



data_fr['score_eval'] = np.logical_and(data_fr['Score1'] >35,data_fr['Score2'] > 35)

print(data_fr)
a = True

b = False

print(('a and b is',a and b))

print(('a or b is',a or b))

print(('not a is',not a))
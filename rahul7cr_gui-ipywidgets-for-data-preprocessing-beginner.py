import pandas as pd 

import pandas as pd

import numpy as np

import ipywidgets as wg

#Create a DataFrame

d = {

    'Name':['Alisa','Bobby','jodja','jack','raghu','Cathrine',

            'M-han','Shu-Bom','Jex','Sandy','Alex','Bube'],

    'Age':[26,24,23,25,23,25,26,25,22,26,26,26],

    'Attendance':[80, 65,40, 70, 29, 80, 90, 70, 30, 70, 83, 75],

      

       'Score':[85.7, 63.3, 55.8, 74.6, 31.57, 77.3, 85.86, 63.45, 42.2, 62.21, 89.3, 77.76],

    'Class':[1,3,0,2,0,2,1,3,0,3,1,2]



}

 

df = pd.DataFrame(d,columns=['Name','Age','Attendance','Score','Class'])

df
str1=df.columns[0]

col1=wg.Checkbox(value=True,description=df.columns[0]+"   "+str(df.dtypes[0]))

col2=wg.Checkbox(value=True,description=df.columns[1]+"   "+str(df.dtypes[1]))

col3=wg.Checkbox(value=True,description=df.columns[2]+"   "+str(df.dtypes[2]))

col4=wg.Checkbox(value=True,description=df.columns[3]+"   "+str(df.dtypes[3]))

col5=wg.Checkbox(value=True,description=df.columns[4]+"   "+str(df.dtypes[4]))

display(col1,col2,col3,col4,col5)
X=[]

col=[col1,col2,col3,col4,col5]

j=0

for i in col:

    if i.value==True:

        X.append(df[df.columns[j]])

    j=j+1

X=np.asarray(X)

X=X.transpose()

print(X)
values=wg.FloatSlider(

    value=0.2,

    min=0,

    max=0.5,

    step=0.1,

    description='Test:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='.2f',

)

display(values)

print(values.value)
import matplotlib.pyplot as plt 

import numpy as np 

from sklearn import datasets, linear_model, metrics 

from sklearn.linear_model import LogisticRegression

print(X)

#X=[]

#for i in range(len(df['Age'])):

#    X.append([df['Age'][i],df['Attendance'][i]])

y=np.asarray(df['Class'])



print(values.value)  

# splitting X and y into training and testing sets 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=values.value, 

                                                    random_state=1) 

  

# create linear regression object 

Regr = LogisticRegression()

  

# train the model using the training sets 

Regr.fit(X_train, y_train) 

  

# regression coefficients 

print('Coefficients: \n', Regr.coef_) 

  

print(Regr.predict(np.array([[25, 70]])))

  

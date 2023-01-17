import ipywidgets as widgets

from ipywidgets import *



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn import neighbors

from sklearn import tree

from sklearn import linear_model



df = pd.read_csv('../input/dataset2.csv',index_col='country')



df = df.drop('Norway',axis=0)

X = df.drop('max_ev_p',axis=1)

y = df['max_ev_p']



train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=0)
columns = ['ssm','ppp','nat_p']

model = linear_model.LinearRegression()

model.fit(train_X[columns],train_y)



print(model.coef_)



def predict_ev(Name,ssm, ppp, nat_p):

    x = pd.DataFrame({'ssm':ssm,'ppp':ppp,'nat_p':nat_p},index=[0],columns=columns)

    #print(x)

    pred = max(0,model.predict(x))

    #print('The country "%s" is predicted to have a EV market share of %s' % (Name,pred))

    print('Predicted EV market share in %s: %s' % (Name,pred))

    

style = {'description_width': 'initial'}

wide_layout = Layout(width='350px')

inter=interact(predict_ev,Name='country',

               ssm=Checkbox(description='Same-sex marriage'), 

               ppp=IntSlider(min=0,max=100000,description='GDP (PPP) per capita',style=style,layout=wide_layout),

               nat_p=IntSlider(min=0,max=100,description='Nationalist parties',style=style,layout=wide_layout) )

display(inter)

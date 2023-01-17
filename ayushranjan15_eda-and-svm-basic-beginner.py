import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import svm
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


pd.set_option('display.max_column', None)
pd.set_option('display.max_row', None)
food=pd.read_csv("../input/FAO.csv" , encoding="latin1")
food=food.rename(columns={'Area Code':'area_code','Item Code':'item_code','Element Code':'element_code','Area Abbreviation':'area_abbreviation'})
food.shape
food['Unit']=food['Unit'].apply(lambda x:int(x.strip('tonnes')))
food.head()
area=food['area_abbreviation'].value_counts()
labels=(np.array(area.index))
value=(np.array((area/area.sum())*100))
trace=go.Pie(labels=labels,values=value)
layout=go.Layout(title='area')
data=[trace]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig,filename='area')
plt.figure(figsize=(24,12))
item=food['Item'].value_counts()[:50]
sns.barplot(item.values,item.index)
sns.despine(left=True,right=True)
plt.show()
ele=food['Element'].value_counts()
labels=(np.array(ele.index))
values=(np.array((ele/ele.sum())*100))
trace=go.Pie(labels=labels,values=values)
layout=go.Layout(title="element")
data=[trace]
fig=go.Figure(data=data,layout=layout)
py.iplot(fig,filename=ele)
plt.savefig('joint.png')
wor_df=pd.DataFrame(food['Area'].value_counts()).reset_index()
wor_df.columns=['cont','Production']
wor_df=wor_df.reset_index().drop('index',axis=1)

data = [ dict(
        type = 'choropleth',
        locations = wor_df['cont'],
        locationmode = 'country names',
        z = wor_df['Production'],
        text = wor_df['cont'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = ' ',
            title = 'food production'),
      ) ]

layout = dict(
    title = 'production of food around globe',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='world-map' )
food['Area'].value_counts()[:5]
food['area_abbreviation'].value_counts()[:5]
j=food['area_abbreviation'].unique()
len(j)
i=food['Area'].unique()
len(i)
food=food.rename(columns={'Area Code':'area_code','Item Code':'item_code','Element Code':'element_code'})
food.shape
food.drop(food.columns[[0,2,4,6]],axis=1,inplace=True)
food.describe()
food['Y1961'].isnull().sum()
for i in range(6,len(food.columns)):
    val=food.columns[i]
    food=food[np.isfinite(food[val])]
food['element_code'].unique()
target=food['element_code']
food=food.drop('element_code',axis=1)
x_train,x_test,y_train,y_test=train_test_split(food,target,test_size=0.2)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
parameters = {'C': [1,10],'gamma':[0.001,0.01,1]}
clf=svm.SVC() 
clf = GridSearchCV(clf, parameters)
clf.fit(x_train,y_train)
classifier = clf.cv_results_
print(classifier)
clf.best_estimator_
clf.best_params_
clf=svm.SVC(C=1,gamma=0.001)
grid_clf=clf.fit(x_train,y_train)
pred=clf.predict(x_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
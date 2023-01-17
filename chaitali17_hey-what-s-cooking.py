import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import json

import os



# To create plots

from matplotlib.colors import rgb2hex

from matplotlib.cm import get_cmap

import matplotlib.pyplot as plt



# To create nicer plots

import seaborn as sns



from ipywidgets import interact,interactive, fixed, interact_manual

from ipywidgets.widgets import Select





# To create interactive plots

import plotly

import plotly.offline as pyo

import plotly.graph_objs as go



# Set notebook mode to work in offline

pyo.init_notebook_mode(connected=True)

import plotly.graph_objs as go





import os

print(os.listdir("../input"))





import re, string, unicodedata

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC



%matplotlib inline
train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')
train.head()
train.shape
test.shape
Cuisine_count=train.groupby(['cuisine'])['id'].count().sort_values(ascending=False)
def barplot(table_name,title_name,x_name,y_name):

            n = table_name.shape[0]

            colormap = get_cmap('viridis')

            colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]

            data = go.Bar(x = table_name.index,

              y = table_name,

              marker = dict(color = colors)

             )

            layout = go.Layout(title = title_name,

                   xaxis = dict(title = x_name),

                   yaxis = dict(title = y_name))

            fig = go.Figure(data=[data], layout=layout)

            pyo.iplot(fig)
barplot(Cuisine_count,'Number of Cuisine Recipe','Cuisine','Number of Recipe')
def text_clean(text):

    text=re.sub(r'([^a-zA-Z\s]+?)','',text)

    text=re.sub(' ','',text)

    text=re.sub('P{P}+','',text)

    return text
train['Data'] = 'Train'

test['Data'] = 'Test'

both_df = pd.concat([train, test], axis=0,sort=False).reset_index(drop=True)

both_df['Ing']=" "
both_df.tail()
ingredi=[]

for i,item in both_df.iterrows():

    ingredient=[]

    for ingre in item.ingredients:

        ingred=text_clean(ingre)

        if ingred not in ingredient:

            ingredient.append(ingred)

    ingredi.append(ingredient)

both_df['Ing']=ingredi
cuisine=[]

ingred=[]

id_=[]



for i,row in train.iterrows():

    cusine=row.cuisine

    id=row.id

    for ingredient in row.ingredients:

        cuisine.append(cusine)

        ingred.append(ingredient)

        id_.append(id)
data=pd.DataFrame({'id':id_,'target':cuisine,'ingredient':ingred})
data.groupby(['id'])['ingredient'].count().hist(bins=50)
data.groupby(['ingredient'])['target'].count().sort_values(ascending=False)[:15]
data[data['ingredient']=='hot pepperoni']                          
data[data['ingredient'].str.contains('pepperoni')].groupby(['ingredient','target']).count()
unique=[]



def unique_words(text):

    v=text.split(' ',)

    for i in v:

        if i not in unique:

            unique.append(i)

        

for i,item in data.iterrows():

    unique_words(item.ingredient)
data.head()
@interact(Cuisine=Cuisine_count.index)



def plot(Cuisine):

    res=data[data['target']==Cuisine].groupby('ingredient')['id'].count().sort_values(ascending=False)[:10]

    tile_name='Top Ingredients for'+' '+str.upper(Cuisine) +' '+ 'Cuisine'

    barplot(res,tile_name,'Ingredient','Count') 
ingred_c=data.groupby('ingredient')['target'].unique()
Ingred_Cu=pd.DataFrame({'ingred':ingred_c.index,'Cuisine':ingred_c.values})
@interact(Ingredient=Ingred_Cu['ingred'])



def Ing(Ingredient):

    print('The ' + str.upper(Ingredient)+ ' is added in the following Ingredients:')

    for item in Ingred_Cu[Ingred_Cu['ingred']==Ingredient].Cuisine:

        for p in item:

            print(p)
both_df['Ing'] = both_df['Ing'].map(";".join)
both_df.head()
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

X = cv.fit_transform(both_df['Ing'])
X.shape
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

y = enc.fit_transform(both_df[both_df['Data']=='Train']['cuisine'])
enc.classes_
X1=X[0:39774,]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2)
print(list(cv.vocabulary_.keys())[:100])
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()

logistic.fit(X_train,y_train)
logistic.score(X_test, y_test)
from sklearn.metrics import confusion_matrix



plt.figure(figsize=(10, 10))



cm = confusion_matrix(y_test, logistic.predict(X_test))

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



plt.imshow(cm_normalized, interpolation='nearest')

plt.title("confusion matrix")

plt.colorbar(shrink=0.3)

cuisines = both_df[both_df['Data']=='Train']['cuisine'].value_counts().index

tick_marks = np.arange(len(cuisines))

plt.xticks(tick_marks, cuisines, rotation=90)

plt.yticks(tick_marks, cuisines)

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')
name={'0':'brazilian', '1':'british', '2':'cajun_creole', '3':'chinese', '4':'filipino',

       '5':'french', '6':'greek', '7':'indian', '8':'irish', '9':'italian', '10':'jamaican',

       '11':'japanese', '12':'korean', '13':'mexican', '14':'moroccan', '15':'russian',

       '16':'southern_us', '17':'spanish', '18':'thai', '19':'vietnamese'}
from sklearn.metrics import classification_report

y_pred = logistic.predict(X_test)

print(classification_report(y_test, y_pred, target_names=cuisines))
X2=X[39774:,]
pred=logistic.predict(X2)
test_id=both_df[both_df['Data']=='Test']['id']
sub=pd.DataFrame({'id':test_id,'cuisine':pred})
sub['cuisine']=sub['cuisine'].astype(str).replace(name)
sub.to_csv('sample_submission.csv',index=False)
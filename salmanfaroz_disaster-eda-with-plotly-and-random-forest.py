import matplotlib.pyplot as plt

import warnings

import seaborn as sns

import pandas as pd

import numpy as np

from colorama import Fore, Back, Style 

from sklearn.model_selection import train_test_split

from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import xgboost

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly.express as px

from statsmodels.formula.api import ols

import plotly.graph_objs as gobj



init_notebook_mode(connected=True)

warnings.filterwarnings("ignore")

import plotly.figure_factory as ff



%matplotlib inline



ds=pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")



trn={"Sex":{"M":0,"F":1}}

trn1={"Category":{"P":0,"C":1}}







ds.replace(trn,inplace=True)

ds.replace(trn1,inplace=True)

ds.head(5)
hist_data =[ds["Age"].values]

group_labels = ['Age'] 



fig = ff.create_distplot(hist_data, group_labels)

fig.update_layout(title_text='Age Distribution plot')



fig.show()
fig = px.box(ds, x="Sex", y="Age", points="all",)

fig.update_layout(

    title_text="Gender wise Age Spread - Male = 0 Female =1")

fig.show()
male=ds[ds["Sex"]==0]

female=ds[ds["Sex"]==1]



male_survi=male[ds["Survived"]==1]

male_not=male[ds["Survived"]==0]

female_survi=female[ds["Survived"]==1]

female_not=female[ds["Survived"]==0]



labels = ['Male - Survived','Male - Not Survived', "Female -  Survived", "Female - Not Survived"]

values = [len(male[ds["Survived"]==1]),len(male[ds["Survived"]==0]),

         len(female[ds["Survived"]==1]),len(female[ds["Survived"]==0])]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

fig.update_layout(

    title_text="Analysis on Survival - Gender")

fig.show()

surv=ds[ds["Survived"]==1]["Age"]

not_surv=ds[ds["Survived"]==0]["Age"]

hist_data = [surv,not_surv]



group_labels = ['Survived', 'Not Survived']



fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.update_layout(

    title_text="Analysis in Age on Survival Status")

fig.show()
fig = px.violin(ds, y="Age", x="Sex", color="Survived", box=True, points="all",

          hover_data=ds.columns)

fig.update_layout(

    title_text="Analysis in Age and Gender on Survival Status")

fig.show()
labels = ['Survived','Not Survived']

a,b=len(ds[ds["Survived"]==1][ds["Category"]==0]),len(ds[ds["Survived"]==0][ds["Category"]==0])



c,d=len(ds[ds["Survived"]==1][ds["Category"]==1]),len(ds[ds["Survived"]==0][ds["Category"]==1])







fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels, values=[a,b], name="Passenger"),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=[c,d], name="Crew"),

              1, 2)



fig.update_traces(hole=.4, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Analysis in Category based on Survival Status",

    annotations=[dict(text='Passenger', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Crew', x=0.82, y=0.5, font_size=20, showarrow=False)])

fig.show()
from sklearn.ensemble import RandomForestClassifier



from sklearn.datasets import make_classification



x=ds.loc[:,["Sex","Age","Category"]]

y=ds.loc[:,["Survived"]]



x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=400)





clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(x_train, y_train)





pred=clf.predict(x_test)
print(Fore.GREEN + "Accuracy is : ",clf.score(x_test,y_test))
def plot_confusion_matrix(cm, names, title="Random Forest Model- Confusion matrix", cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(names))

    plt.xticks(tick_marks, names, rotation=45)

    plt.yticks(tick_marks, names)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

cm = confusion_matrix(y_test, pred)

np.set_printoptions(precision=2)

print('Confusion matrix, without normalization')

print(cm)

plt.figure()

plot_confusion_matrix(cm, ["Not Survived","Survived"])
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.tools import FigureFactory as ff
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from PIL import Image
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import seaborn as sns
import missingno as msno # check missing value


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sales = pd.read_csv('../input/videogamesales/vgsales.csv')
sales.head()
sales.info()
sales.describe()
msno.matrix(sales)
sales.isnull().sum()
#Just Cleaning the missing ones
sales.dropna(how="any",inplace = True)
sales.info()
#Sales - float to int
sales.Year = sales.Year.astype(int)
sales.head()
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot
from bokeh.models.widgets import Tabs,Panel
output_notebook()
#Color Mapping
source = ColumnDataSource(sales)
factors = list(sales.Genre.unique()) # what we want to color map. I choose genre of games
colors = ["green","red","black","blue","orange","grey","brown","purple","yellow","white","pink","peru"]
mapper = CategoricalColorMapper(factors = factors,palette = colors)
plot =figure()
plot.circle(x= "Year",y = "Global_Sales",source=source,color = {"field":"Genre","transform":mapper})
show(plot)
df = sales.head(50)
NA = go.Scatter(
                    x = df.Rank,
                    y = df.NA_Sales,
                    mode = "markers",
                    name = "North America",
                    marker = dict(color = 'rgba(249, 94, 28, 0.8)',size=8),
                    text= df.Name)

EU = go.Scatter(
                    x = df.Rank,
                    y = df.EU_Sales,
                    mode = "markers",
                    name = "Europe",
                    marker = dict(color = 'rgba(28, 140, 230, 0.8)',size=8),
                    text= df.Name)
JP = go.Scatter(
                    x = df.Rank,
                    y = df.JP_Sales,
                    mode = "markers",
                    name = "Japan",
                    marker = dict(color = 'rgba(180, 24, 95, 0.8)',size=8),
                    text= df.Name)
OTHER = go.Scatter(
                    x = df.Rank,
                    y = df.Other_Sales,
                    mode = "markers",
                    name = "Other",
                    marker = dict(color = 'lime',size=8),
                    text= df.Name)
                    

data = [NA,EU,JP,OTHER]
layout = dict(title = 'North America, Europe, Japan and Other Sales Top50',
              xaxis= dict(title= 'Rank',ticklen= 2,zeroline= False,zerolinewidth=1,gridcolor="black"),
              yaxis= dict(title= 'Sales(Millions)',ticklen= 2,zeroline= False,zerolinewidth=1,gridcolor="red",),
              paper_bgcolor='rgb(243, 243, 243)',
              plot_bgcolor='rgb(243, 243, 243)' )
fig = dict(data = data, layout = layout)
iplot(fig)
sales.NA_Sales = sales.NA_Sales.astype(int)
sales.EU_Sales = sales.EU_Sales.astype(int)
sales.JP_Sales = sales.JP_Sales.astype(int)
sales.Other_Sales = sales.Other_Sales.astype(int)
sales.Global_Sales = sales.Global_Sales.astype(int)
sales.Global_Sales = sales.Global_Sales.astype(int)
sales.head()
plt.figure(figsize = (12,12))
sns.barplot(x = sales['NA_Sales'].value_counts().index,
           y=sales['NA_Sales'].value_counts().values)
plt.xlabel('NA_Sales')
plt.ylabel('Frequency')
plt.title('NA_Sales Bar Plot')
plt.show()
plt.figure(figsize = (50,12))
sns.barplot(x = sales['EU_Sales'].value_counts().index,
           y=sales['EU_Sales'].value_counts().values)
plt.xlabel('EU_Sales')
plt.ylabel('Frequency')
plt.title('EU_Sales Bar Plot')
plt.show()
plt.figure(figsize = (50,12))
sns.barplot(x = sales['JP_Sales'].value_counts().index,
           y=sales['JP_Sales'].value_counts().values)
plt.xlabel('JP_Sales')
plt.ylabel('Frequency')
plt.title('JP_Sales Bar Plot')
plt.show()
plt.figure(figsize = (50,12))
sns.barplot(x = sales['Other_Sales'].value_counts().index,
           y=sales['Other_Sales'].value_counts().values)
plt.xlabel('Other_Sales')
plt.ylabel('Frequency')
plt.title('Other_Sales Bar Plot')
plt.show()
plt.figure(figsize = (50,12))
sns.barplot(x = sales['Global_Sales'].value_counts().index,
           y=sales['Global_Sales'].value_counts().values)
plt.xlabel('Global_Sales')
plt.ylabel('Frequency')
plt.title('Global_Sales Bar Plot')
plt.show()
sales.corr()
sns.heatmap(sales.corr(),annot=True)
plt.show()
plt.figure(figsize=(12,6))
sns.barplot(df['Rank'][:20],sales.Name[:20])
plt.title('Top 20 Game')
plt.show()
trace = go.Histogram(x=df.Publisher,marker=dict(color="black",line=dict(color='white', width=2)),opacity=0.75)
layout = go.Layout(
    title='Top 50 Game Publishers',
    xaxis=dict(
        title='Publishers'
    ),
    yaxis=dict(
        title='Count'
    ),
    bargap=0.1,
    bargroupgap=0.1, paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor="rgb(243, 243, 243)")
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)

stopwords = set(STOPWORDS)
plt.subplots(figsize=(12,12))
wordcloud = WordCloud(background_color="white",colormap="hsv" ,contour_width=2, contour_color="black",
                      width=1000,stopwords=stopwords,
                          height=1000
                         ).generate(" ".join(sales.Name))

plt.imshow(wordcloud ,interpolation='bilinear')
plt.axis('off')

plt.show()
#Float to int
sales.NA_Sales = sales.NA_Sales.astype(int)
sales.EU_Sales = sales.EU_Sales.astype(int)
sales.JP_Sales = sales.JP_Sales.astype(int)
sales.Other_Sales = sales.Other_Sales.astype(int)
sales.Global_Sales = sales.Global_Sales.astype(int)
sales.Global_Sales = sales.Global_Sales.astype(int)
sales.head()
sales.info()
#useless columns are removed
x = sales.drop(['Rank','Name','Platform','Global_Sales'], axis = 1)
y = sales['Global_Sales']
#Rename for LGB
x.columns = ['year','genre','publisher','nasales','eusales','jpsales','othersales']
y.columns = ['globalsales']
#Encoding For object Columns
x = pd.get_dummies(x, columns=['genre','publisher'],prefix = ['genre','publisher'])
x.head()
#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from sklearn.model_selection import cross_val_score

knn_model=KNeighborsClassifier().fit(X_train,y_train)
lr_model=LogisticRegression().fit(X_train,y_train)
rf_model=RandomForestClassifier().fit(X_train,y_train)
xgb_model=XGBClassifier().fit(X_train,y_train)
gbm_model=GradientBoostingClassifier().fit(X_train,y_train)

models=[lr_model,rf_model,gbm_model,xgb_model,knn_model]

sc_fold = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

for model in models:
    names = model.__class__.__name__
    accuracy = cross_val_score(model,X_train,y_train,cv=sc_fold)
    print("{}s score:{}".format(names,accuracy.mean()))

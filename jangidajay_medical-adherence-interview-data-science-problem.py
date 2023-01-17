# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.metrics import confusion_matrix,classification_report
import plotly.figure_factory as ff

from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/dataset/Training Data.csv')
print(train.shape)
train.head()
test = pd.read_csv('/kaggle/input/dataset/Test Data.csv')
print(test.shape)
test.head()
train.info()
print ("Rows     : " ,train.shape[0])
print ("Columns  : " ,train.shape[1])
print ("\nFeatures : \n" ,train.columns.tolist())
print ("\nMissing values :  ", train.isnull().sum().values.sum())
print ("\nUnique values :  \n",train.nunique())
train_0 = train[train['Age']==0]
print(train_0.shape)
train_0.head()
train.describe()
temp = train.groupby('Adherence').count().reset_index().sort_values(by='patient_id',ascending=False)
temp.style.background_gradient(cmap='Purples')
## Visualize the data
#labels
lab = train["Adherence"].value_counts().keys().tolist()
#values
val = train["Adherence"].value_counts().values.tolist()

trace = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ 'royalblue' ,'lime'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Target data distribution",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data,layout = layout)
py.iplot(fig)
fig = go.Figure(go.Funnelarea(
    text=temp.Adherence,
    values=temp.patient_id,
    title={'position':'top center','text':'Funnel-chart of adherence dsitribution'}))
fig.show()
train.columns
#Separating churn and non churn customers
adhere     = train[train["Adherence"] == "Yes"]
not_adhere = train[train["Adherence"] == "No"]

cat_feature = ['Gender','Diabetes','Alcoholism','HyperTension','Smokes','Tuberculosis','Sms_Reminder']


#Separating catagorical and numerical columns
Id_col     = ['patient_id']
target_col = ["Adherence"]
cat_cols   = cat_feature
num_cols   = [x for x in train.columns if x not in cat_cols + target_col + Id_col]

for features in cat_feature:

    #labels
    lab = train[features].value_counts().keys().tolist()
    #values
    val = train[features].value_counts().values.tolist()

    trace = go.Pie(labels = lab ,
                   values = val ,
                   marker = dict(colors =  [ 'royalblue' ,'lime'],
                                 line = dict(color = "white",
                                             width =  1.3)
                                ),
                   rotation = 90,
                   hoverinfo = "label+value+text",
                   hole = .5
                  )
    layout = go.Layout(dict(title = "Categorical feature distribution: {}".format(features),
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                           )
                      )

    data = [trace]
    fig = go.Figure(data = data,layout = layout)
    py.iplot(fig)
def plot_pie(column) :
    
    trace1 = go.Pie(values  = adhere[column].value_counts().values.tolist(),
                    labels  = adhere[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "Adherence Patients",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6
                   )
    trace2 = go.Pie(values  = not_adhere[column].value_counts().values.tolist(),
                    labels  = not_adhere[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "Not adherence patients" 
                   )


    layout = go.Layout(dict(title = column + " distribution in patient's feature ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "Adherence Patients",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .15, y = .5),
                                           dict(text = "Non adherence patients",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .88,y = .5
                                               )
                                          ]
                           )
                      )
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    py.iplot(fig)


#function  for histogram for customer attrition types
def histogram(column) :
    trace1 = go.Histogram(x  = adhere[column],
                          histnorm= "percent",
                          name = "Adherence Patient's",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    
    trace2 = go.Histogram(x  = not_adhere[column],
                          histnorm = "percent",
                          name = "Non adherence patients",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in patients features ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)
    
# #function  for scatter plot matrix  for numerical columns in data
def scatter_matrix(df)  :
    
    df  = df.sort_values(by = "Adherence" ,ascending = True)
    classes = df["Adherence"].unique().tolist()
    classes
    
    class_code  = {classes[k] : k for k in range(2)}
    class_code

    color_vals = [class_code[cl] for cl in df["Adherence"]]
    color_vals

    pl_colorscale = "Portland"

    pl_colorscale

    text = [df.loc[k,"Adherence"] for k in range(len(df))]
    text

    trace = go.Splom(dimensions = [dict(label  = "Age",
                                       values = df["Age"]),
                                  dict(label  = 'Prescription_period',
                                       values = df['Prescription_period'])],
                     text = text,
                     marker = dict(color = color_vals,
                                   colorscale = pl_colorscale,
                                   size = 3,
                                   showscale = False,
                                   line = dict(width = .1,
                                               color='rgb(230,230,230)'
                                              )
                                  )
                    )
    axis = dict(showline  = True,
                zeroline  = False,
                gridcolor = "#fff",
                ticklen   = 4
               )
    
    layout = go.Layout(dict(title  = 
                            "Scatter plot matrix for Numerical columns for customer attrition",
                            autosize = False,
                            height = 800,
                            width  = 800,
                            dragmode = "select",
                            hovermode = "closest",
                            plot_bgcolor  = 'rgba(240,240,240, 0.95)',
                            xaxis1 = dict(axis),
                            yaxis1 = dict(axis),
                            xaxis2 = dict(axis),
                            yaxis2 = dict(axis),
                            xaxis3 = dict(axis),
                            yaxis3 = dict(axis),
                           )
                      )
    data   = [trace]
    fig = go.Figure(data = data,layout = layout )
    py.iplot(fig)

#for all categorical columns plot pie
for i in cat_feature :
    plot_pie(i)

#for all categorical columns plot histogram    
for i in num_cols :
    histogram(i)
    
scatter_matrix(train)
sns.distplot(train['Age'], color='g', bins=50, hist_kws={'alpha': 0.4});
sns.distplot(train['Prescription_period'], color='b', bins=50, hist_kws={'alpha': 0.4});
## getting features from prescription period i.e Week and Year

def _get_weeks_(number_of_days):
    week = int((number_of_days % 365) / 7)
    
    return week

def _get_years(number_of_days):
    year = int(number_of_days / 365)
    
    return year
train['Prescription_period_week'] = train['Prescription_period'].apply(lambda x: _get_weeks_(x))
train['Prescription_period_year'] = train['Prescription_period'].apply(lambda x: _get_years(x))
train['Gender']=train['Gender'].map({'F':0,'M':1})
train['Adherence']=train['Adherence'].map({'No':1,'Yes':0})
X = train.drop(['patient_id','Adherence'],axis=1)
corr = X.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Target class
Y = train['Adherence']
# Splitting data into train and test with test size of 0.2

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=10)
# Helper function

def _get_confusion_matrix(df):
    cf_matrix = pd.DataFrame(df,["Positive","Negative"],
                        ["Positive","Negative"],
                         dtype=int)
    sns.heatmap(cf_matrix,annot=True,annot_kws={"size": 16}, fmt='g')
    plt.title('Confusion Matrix')
    
    print('Classification Report\n',classification_report(prediction,ytest))
    
    
    print("Precision for Yes :{}%".format((df[0][0])*100/(df[0][0]+df[0][1])))
    print("Recall for Yes : {}%".format((df[0][0])*100/(df[0][0]+df[1][0])))
    print("Precision for No : {}%".format((df[1][1])*100/(df[1][1]+df[1][0])))
    print("Recall for No : {}%".format((df[1][1])*100/(df[1][1]+df[0][1])))
model = LGBMClassifier(learning_rate=0.02,
                    boosting_type='gbdt',
                    max_depth=4,
                    random_state=100,
                    n_estimators=800,
                    reg_alpha=0,
                    reg_lambda=1,
                    n_jobs=-1)
model.fit(X,Y)
model.fit(xtrain,ytrain)
prediction = model.predict(xtest)
print("Accuraccy score",accuracy_score(prediction,ytest))
co_matrix = confusion_matrix(prediction,ytest)
# Let's see feature importances
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,5))
_get_confusion_matrix(co_matrix)
from sklearn.model_selection import GridSearchCV

n_estimators = [50,100]
max_depth = [3,5, 8, 15]
min_samples_split = [2, 5, 10, 15]
min_samples_leaf = [1, 2, 5, 10] 

forest = RandomForestClassifier()

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 
                      n_jobs = 6)
bestF = gridF.fit(xtrain, ytrain)
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report

model = RandomForestClassifier(max_depth=8,min_samples_leaf=1,min_samples_split=10,n_estimators=50)
model.fit(xtrain,ytrain)
prediction = model.predict(xtest)
print("Accuraccy score",accuracy_score(prediction,ytest))
co_matrix = confusion_matrix(prediction,ytest)
_get_confusion_matrix(co_matrix)
# Let's the feature importances
feat_imp = pd.Series(model.feature_importances_, index=xtrain.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,5))
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

n_estimators = [50,100,300]
max_depth = [3,5, 8, 15]
learning_rate = [0.001,0.01,0.1,1]

xgb_model = xgb.XGBClassifier(
                      scale_pos_weight=1,
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      reg_alpha = 0.3, 
                      gamma=10)

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,learning_rate=learning_rate)

gridF = GridSearchCV(xgb_model, hyperF, cv = 3, verbose = 1, 
                      n_jobs = 6)
bestF = gridF.fit(xtrain, ytrain)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

model = xgb.XGBClassifier( 
                      scale_pos_weight=1,
                      learning_rate=1,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=300, 
                      reg_alpha = 0.3,
                      max_depth=5, 
                      gamma=10)

model.fit(xtrain,ytrain)
prediction = model.predict(xtest)
print("Accuraccy score",accuracy_score(prediction,ytest))
co_matrix = confusion_matrix(prediction,ytest)
_get_confusion_matrix(co_matrix)
# Let's the feature importances
feat_imp = pd.Series(model.feature_importances_, index=xtrain.columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,5))
from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["Models","Accuracy","Recall","precision"]

x.add_row(["lgm",89.64,90,90])
x.add_row(["Randomfores",89.64,90,90])
x.add_row(["Xgboost",89.67,90,90])

print(x)

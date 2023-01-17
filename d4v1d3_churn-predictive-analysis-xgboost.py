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
df = pd.read_csv("/kaggle/input/bank-churn-modelling/Churn_Modelling.csv")

df.head()
print(f"Datasets length: {len(df)}")

for col in df.columns:

    print(f"Unique values for {col}: {len(set(df[col]))}")
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title="Customer churn")

profile.to_widgets()
import pandas as pd

import seaborn as sns#visualization

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization

import matplotlib.pyplot as plt



lab = ["No", "Yes"]

#values

val = df["Exited"].value_counts().values.tolist()



trace = go.Pie(labels = lab ,

               values = val ,

               name="Churn",

               marker = dict(colors =  [ 'royalblue' ,'lime'],

                             line = dict(color = "white",

                                         width =  1.3)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Customer attrition in data",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )



data = [trace]

fig = go.Figure(data = data,layout = layout)

py.iplot(fig)

import matplotlib as mpl

mpl.style.use('ggplot')

churn     = df[df["Exited"] == 1]

not_churn = df[df["Exited"] == 0]



def plot_pie(column) :

    

    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),

                    labels  = churn[column].value_counts().keys().tolist(),

                    hoverinfo = "label+percent+name",

                    domain  = dict(x = [0,.48]),

                    name    = "Churn",

                    marker  = dict(line = dict(width = 2,

                                               color = "rgb(243,243,243)")

                                  ),

                    hole    = .6

                   )

    trace2 = go.Pie(values  = not_churn[column].value_counts().values.tolist(),

                    labels  = not_churn[column].value_counts().keys().tolist(),

                    hoverinfo = "label+percent+name",

                    marker  = dict(line = dict(width = 2,

                                               color = "rgb(243,243,243)")

                                  ),

                    domain  = dict(x = [.52,1]),

                    hole    = .6,

                    name    = "Non churn" 

                   )





    layout = go.Layout(dict(title = column + " distribution in customer attrition ",

                            plot_bgcolor  = "rgb(243,243,243)",

                            paper_bgcolor = "rgb(243,243,243)",

                            annotations = [dict(text = "Churn",

                                                font = dict(size = 13),

                                                showarrow = False,

                                                x = .15, y = .5),

                                           dict(text = "Non churn",

                                                font = dict(size = 13),

                                                showarrow = False,

                                                x = .88,y = .5

                                               )

                                          ]

                           )

                      )

    data = [trace2,trace1]

    fig  = go.Figure(data = data,layout = layout)

    py.iplot(fig)





#function  for histogram for customer attrition types

def histogram(column) :

    trace1 = go.Histogram(x  = churn[column],

                          histnorm= "percent",

                          name = "Churn",

                          marker = dict(line = dict(width = .5,

                                                    color = "black"

                                                    )

                                        ),

                         opacity = .9 

                         ) 

    

    trace2 = go.Histogram(x  = not_churn[column],

                          histnorm = "percent",

                          name = "Non churn",

                          marker = dict(line = dict(width = .5,

                                              color = "black"

                                             )

                                 ),

                          opacity = .9

                         )

    

    data = [trace2,trace1]

    layout = go.Layout(dict(title =column + " distribution in customer attrition ",

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

    

#function  for scatter plot matrix  for numerical columns in data

def scatter_matrix(df)  :

    

    df  = df.sort_values(by = "Exited" ,ascending = False)

    classes = df["Exited"].unique().tolist()

    classes

    

    class_code  = {classes[k] : k for k in range(2)}

    class_code



    color_vals = [class_code[cl] for cl in df["Exited"]]

    color_vals



    pl_colorscale = "Portland"



    pl_colorscale



    text = [df.loc[k,"Exited"] for k in range(len(df))]

    text



    trace = go.Splom(dimensions = [dict(label  = "Tenure",

                                       values = df["Tenure"]),

                                  dict(label  = 'Balance',

                                       values = df['Balance']),

                                  dict(label  = 'EstimatedSalary',

                                       values = df['EstimatedSalary'])],

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



    

cat_cols = ["Geography", "Gender", "NumOfProducts","HasCrCard", "IsActiveMember"]

num_cols = ["Age", "Balance", "EstimatedSalary","CreditScore","Tenure"]

#for all categorical columns plot pie

for i in cat_cols :

    plot_pie(i)



#for all categorical columns plot histogram    

for i in num_cols :

    histogram(i)



#scatter plot matrix

scatter_matrix(df)
correlation = df.corr()

#tick labels

matrix_cols = correlation.columns.tolist()

#convert to array

corr_array  = np.array(correlation)



#Plotting

trace = go.Heatmap(z = corr_array,

                   x = matrix_cols,

                   y = matrix_cols,

                   colorscale = "Viridis",

                   colorbar   = dict(title = "Pearson Correlation coefficient",

                                     titleside = "right"

                                    ) ,

                  )



layout = go.Layout(dict(title = "Correlation Matrix for variables",

                        autosize = False,

                        height  = 720,

                        width   = 800,

                        margin  = dict(r = 0 ,l = 210,

                                       t = 25,b = 210,

                                      ),

                        yaxis   = dict(tickfont = dict(size = 9)),

                        xaxis   = dict(tickfont = dict(size = 9))

                       )

                  )



data = [trace]

fig = go.Figure(data=data,layout=layout)

py.iplot(fig)
from sklearn import preprocessing

encs = {}

cols = ["Geography", "Gender"]

for c in cols:

    encs[c] = preprocessing.LabelEncoder()

    encs[c].fit(df[c])

    df[c] = encs[c].transform(df[c])

    

df.head()
from sklearn.decomposition import PCA

from sklearn.preprocessing import scale, normalize



pca = PCA(n_components = 2)

Id_col = ['RowNumber', 'CustomerId', 'Surname']

target_col = ["Exited"]

X = df[[i for i in df.columns if i not in Id_col + target_col]]

Xscal = scale(X)

Xnorm = normalize(X)

Y = df[target_col + Id_col]





def plot(X,Y, title):



    principal_components = pca.fit_transform(X)

    pca_data = pd.DataFrame(principal_components,columns = ["PC1","PC2"])

    pca_data = pca_data.merge(Y,left_index=True,right_index=True,how="left")

    pca_data["Churn"] = pca_data["Exited"].replace({1:"Churn",0:"Not Churn"})



    

    def pca_scatter(target,color) :

        tracer = go.Scatter(x = pca_data[pca_data["Churn"] == target]["PC1"] ,

                            y = pca_data[pca_data["Churn"] == target]["PC2"],

                            name = target,mode = "markers",

                            marker = dict(color = color,

                                          line = dict(width = .5),

                                          symbol =  "diamond-open"),

                            text = ("Customer Id : " + 

                                    pca_data[pca_data["Churn"] == target]['Surname'])

                           )

        return tracer



    layout = go.Layout(dict(title = title,

                            plot_bgcolor  = "rgb(243,243,243)",

                            paper_bgcolor = "rgb(243,243,243)",

                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                         title = "principal component 1",

                                         zerolinewidth=1,ticklen=5,gridwidth=2),

                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                         title = "principal component 2",

                                         zerolinewidth=1,ticklen=5,gridwidth=2),

                            height = 600

                           )

                      )

    trace1 = pca_scatter("Churn",'red')

    trace2 = pca_scatter("Not Churn",'royalblue')

    data = [trace2,trace1]

    fig = go.Figure(data=data,layout=layout)

    py.iplot(fig)



plot(X,Y, "Visualizing data with Principal Component Analysis on raw data")

plot(Xnorm,Y, "Visualizing data with Principal Component Analysis on normalized data")

plot(Xscal,Y, "Visualizing data with Principal Component Analysis on scaled data")
X = df[['CreditScore', 'Geography','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard','IsActiveMember']]

y = df["Exited"]
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)
import xgboost as xgb



D_train = xgb.DMatrix(X_train, label=Y_train)

D_test = xgb.DMatrix(X_test, label=Y_test)



param = {

    'eta': 0.3, 

    'max_depth': 5,  

    'objective': 'multi:softprob',  

    'num_class': 2

} 



steps = 60  # The number of training iterations



model = xgb.train(param, D_train, steps)



import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score



preds = model.predict(D_test)

best_preds = np.asarray([np.argmax(line) for line in preds])



print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))

print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))

print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))
# Feature importance

from xgboost import plot_importance

plot_importance(model)
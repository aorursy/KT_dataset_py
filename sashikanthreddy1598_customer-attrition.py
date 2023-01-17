# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing libraries

import numpy as np #linear Algebra

import pandas as pd #data preprocessing



import matplotlib.pyplot as plt #visualization

from PIL import Image

%matplotlib inline

import seaborn as sns #visualization

import itertools

import warnings

warnings.filterwarnings("ignore")

import io

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization
telcom = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

telcom.head()

#first few rows
print ("Rows     : " ,telcom.shape[0])

print ("Columns  : " ,telcom.shape[1])

print ("\nFeatures : \n" ,telcom.columns.tolist())

print ("\nMissing values :  ", telcom.isnull().sum().values.sum())

print ("\nUnique values :  \n",telcom.nunique())
telcom.dtypes
#Data Manipulation



#Replacing spaces with null values in total charges column

telcom["TotalCharges"] = telcom["TotalCharges"].replace(" ",np.nan)



#Droping null values from total charges column which contaon .15% missing values

telcom = telcom[telcom["TotalCharges"].notnull()]

telcom = telcom.reset_index()[telcom.columns]



#convert to float type

telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)



#replace 'No Internet service' to No for the following columns



replace_cols =[ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',

              'StreamingMovies']

for i in replace_cols:

    telcom[i] = telcom[i].replace({'No internet service':'No'})

    

#replace values

telcom["SeniorCtizen"] = telcom["SeniorCitizen"].replace({1:'Yes',0:"No"})



#Tenure to categorical column

def tenure_lab(telcom):

    

    if telcom["tenure"] <= 12:

        return "Tenure_0-12"

    elif (telcom['tenure']> 12) & (telcom["tenure"] <= 24 ):

        return "Tenure_12-24"

    elif (telcom['tenure']>24) & (telcom["tenure"] <= 48):

        return 'Tenure_24-48'

    elif (tecom['tenure'] >48) & (telcom["tenure"]<=60):

        return "Tenure_48-60"

    elif telcom["tenure"]> 60:

        return "Tenure_gt_60"

    telcom["tenure_group"] = telcom.apply(lambda telcom:tenure_lab(telcom), axis = 1)

    

    

#Separating churn and non churn customers



churn   = telcom[telcom["Churn"] == 'Yes']

not_churn = telcom[telcom["Churn"] == "No"]
#Separating catagorical and numerical columns

ID_col = ["customerID"]

target_col = ['Churn']

cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()

cat_cols = [x for x in cat_cols if x not in target_col]

num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + ID_col]
#labels



lab = telcom["Churn"].value_counts().keys().tolist()



#values 

val = telcom["Churn"].value_counts().values.tolist()



trace = go.Pie(labels = lab,

               values = val,

               marker = dict(colors = [ 'royalblue', 'lime'],

                            line = dict(color = "white",

                                       width = 1.3)

                            ),

              rotation = 90,

              hoverinfo = 'label +value+text',

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
#function  for pie plot for customer attrition types

def plot_pie(column) :

    

    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),

                    labels  = churn[column].value_counts().keys().tolist(),

                    hoverinfo = "label+percent+name",

                    domain  = dict(x = [0,.48]),

                    name    = "Churn Customers",

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

                    name    = "Non churn customers" 

                   )





    layout = go.Layout(dict(title = column + " distribution in customer attrition ",

                            plot_bgcolor  = "rgb(243,243,243)",

                            paper_bgcolor = "rgb(243,243,243)",

                            annotations = [dict(text = "churn customers",

                                                font = dict(size = 13),

                                                showarrow = False,

                                                x = .15, y = .5),

                                           dict(text = "Non churn customers",

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

    trace1 = go.Histogram(x  = churn[column],

                          histnorm= "percent",

                          name = "Churn Customers",

                          marker = dict(line = dict(width = .5,

                                                    color = "black"

                                                    )

                                        ),

                         opacity = .9 

                         ) 

    

    trace2 = go.Histogram(x  = not_churn[column],

                          histnorm = "percent",

                          name = "Non churn customers",

                          marker = dict(line = dict(width = .5,

                                              color = "black"

                                             )

                                 ),

                          opacity = .9

                         )

    

    data = [trace1,trace2]

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

    

    df  = df.sort_values(by = "Churn" ,ascending = True)

    classes = df["Churn"].unique().tolist()

    classes

    

    class_code  = {classes[k] : k for k in range(2)}

    class_code



    color_vals = [class_code[cl] for cl in df["Churn"]]

    color_vals



    pl_colorscale = "Portland"



    pl_colorscale



    text = [df.loc[k,"Churn"] for k in range(len(df))]

    text



    trace = go.Splom(dimensions = [dict(label  = "tenure",

                                       values = df["tenure"]),

                                  dict(label  = 'MonthlyCharges',

                                       values = df['MonthlyCharges']),

                                  dict(label  = 'TotalCharges',

                                       values = df['TotalCharges'])],

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

for i in cat_cols :

    plot_pie(i)



#for all categorical columns plot histogram    

for i in num_cols :

    histogram(i)



#scatter plot matrix

scatter_matrix(telcom)
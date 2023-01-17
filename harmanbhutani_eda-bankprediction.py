import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import keras as keras

import os



import plotly.offline as py 

py.init_notebook_mode(connected=True) 

import plotly.graph_objs as go 

import plotly.tools as tls 

import plotly.figure_factory as ff 

import warnings

warnings.filterwarnings('ignore')
print(os.listdir("../input"))
dataset = pd.read_csv('../input/churn-modellingcsv/Churn_Modelling.csv')
dataset.head()
dataset.columns
dataset.shape
dataset.isna().sum()
dataset.describe().T
dataset.nunique()
dataset = dataset.drop(["RowNumber", "CustomerId","Surname"], axis = 1)
churn     = dataset[dataset["Exited"] == 1]

not_churn = dataset[dataset["Exited"] == 0]
target_column = ["Exited"]

categorical_column   = dataset.nunique()[dataset.nunique() < 6].keys().tolist()

categorical_column   = [x for x in categorical_column if x not in target_column]

numerical_column   = [x for x in dataset.columns if x not in categorical_column + target_column]
print("\tCategorical Featuers")

print(categorical_column)

print("\n**********************************************")

print("\tNumerical Featuers")

print(numerical_column)
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
plot_pie(categorical_column[0])
plot_pie(categorical_column[1])
plot_pie(categorical_column[2])
plot_pie(categorical_column[3])
plot_pie(categorical_column[4])
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
histogram(numerical_column[0])
histogram(numerical_column[1])
histogram(numerical_column[2])
histogram(numerical_column[3])
histogram(numerical_column[4])
sns.pairplot(dataset, kind ='scatter', hue= "Exited")
"""

Gender_division = dataset["Gender"].value_counts()

print("Gender Division\n")

print(Gender_division)

print("*****************************************************")

Has_card = dataset["HasCrCard"].value_counts()

print("Customer having the cards\n")

print(Has_card)

print("1 = Yes, 0 = No")

print("*****************************************************")

num_prod = dataset["NumOfProducts"].value_counts()

print(num_prod)

print("*****************************************************")

country = dataset["Geography"].value_counts()

print(country)

print("*****************************************************")

tenure = dataset["Tenure"].value_counts()

print(tenure)

print("*****************************************************")

active_member = dataset["IsActiveMember"].value_counts()

print(active_member)

print("1 = Active, 0 = Not Active")

print("*****************************************************")

cust_status = dataset["Exited"].value_counts()

print(cust_status)

print("1 = Customer Left, 0 = Not Left")

"""
"""

#Gender_division = Gender_division.to_frame()

#Gender_division.insert(0, column="Gender_Type" , value = ("Male","Female"))

#Gender_division



Has_card = Has_card.to_frame()

Has_card.insert(0, column ="Taken_Card", value = ("Yes","No"))



country = country.to_frame()

country.insert(0, column="Country_Name", value= ("France", "Germany", "Spain"))

"""
"""

labels = ["France", "Germany","Spain"]

size = country.iloc[:,1]

explode = (0.1,0.1,0.1)

colors = ["c","r","g"]



fig1, ax1 = plt.subplots()

ax1.pie(size, explode= explode, labels= labels, colors=colors,

       autopct = "%1.1f%%", shadow = True, startangle = 90)

ax1.axis = ("equal")

plt.tight_layout()

plt.show()

"""
"""

labels = ["Yes", "No"]

size = Has_card.iloc[:,1]

explode = (0.1,0)

colors = ["#ff9999","#99ff99"]



fig1, ax1 = plt.subplots()

ax1.pie(size, explode= explode, labels= labels, colors=colors,

       autopct = "%1.1f%%", shadow = True, startangle = 90)

ax1.axis = ("equal")

plt.tight_layout()

plt.show()

"""
"""

labels = ["Male", "Female"]

size = Gender_division.iloc[:,1]

explode = (0.1,0)

colors = ["#ff9999","#99ff99"]



fig1, ax1 = plt.subplots()

ax1.pie(size, explode= explode, labels= labels, colors=colors,

       autopct = "%1.1f%%", shadow = True, startangle = 90)

ax1.axis = ("equal")

plt.tight_layout()

plt.show()

"""
"""

dataset["Age"].value_counts()

dataset["EstimatedSalary"].value_counts()

"""
"""

tenure = tenure.to_frame()

tenure.insert(0, column="Tenure_Year", value=(2,1,7,8,5,3,4,9,6,10,0))

"""
"""

tenure_plot = sns.barplot(tenure.iloc[:,0],tenure.iloc[:,1], data = tenure, palette="Paired")

tenure_plot.set(xlabel = "Tenure Year of Customers" , ylabel = "Count"  )

plt.show()

"""
dataset[dataset.columns].corr()
sns.set()

sns.set(font_scale = 1.25)

sns.heatmap(dataset.corr(), annot = True,fmt = ".1f")

plt.show()
trace = []

def gen_boxplot(df):

    for feature in df:

        trace.append(

            go.Box(

                name = feature,

                y = df[feature]

            )

        )



new_df = dataset[numerical_column[:1]]

gen_boxplot(new_df)

data = trace

py.iplot(data)
trace = []

def gen_boxplot(df):

    for feature in df:

        trace.append(

            go.Box(

                name = feature,

                y = df[feature]

            )

        )

new_df = dataset[numerical_column[1:3]]

gen_boxplot(new_df)

data = trace

py.iplot(data)
trace = []

def gen_boxplot(df):

    for feature in df:

        trace.append(

            go.Box(

                name = feature,

                y = df[feature]

            )

        )

new_df = dataset[numerical_column[3:]]

gen_boxplot(new_df)

data = trace

py.iplot(data)
ageNew = []

for val in dataset.Age:

    if val <= 85:

        ageNew.append(val)

    else:

        ageNew.append(dataset.Age.median())

        

dataset.Age = ageNew
dataset1 = dataset
list_cat = ['Geography', 'Gender']

dataset1 = pd.get_dummies(dataset1, columns = list_cat, prefix = list_cat)
from sklearn.ensemble import RandomForestClassifier

import numpy as np
X1 = dataset1.drop('Exited', axis=1)

y1 = dataset1.Exited



features_label = X1.columns



forest = RandomForestClassifier (n_estimators = 10000, random_state = 0, n_jobs = -1)

forest.fit(X1, y1)



importances = forest.feature_importances_



indices = np.argsort(importances)[::-1]



for i in range(X1.shape[1]):

    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i], importances[indices[i]]))
plt.title('Feature Importances')

plt.bar(range(X1.shape[1]), importances[indices], color = "red", align = "center")

plt.xticks(range(X1.shape[1]), features_label, rotation = 90)

plt.show()
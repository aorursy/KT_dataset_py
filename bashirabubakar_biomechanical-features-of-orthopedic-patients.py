print('Hello My name is Bashir Abubakar and welcome to this exploration!')
# import necessary libraries



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

!pip install chart_studio

!pip install cufflinks

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff 

from chart_studio.plotly import plot, iplot

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')

import plotly.graph_objs as go

import chart_studio.plotly as py

import plotly

import chart_studio

chart_studio.tools.set_credentials_file(username='bashman18', api_key='••••••••••')

init_notebook_mode(connected=True)

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.tools as tls

import itertools

import time



# close warning

import warnings

warnings.filterwarnings("ignore")



print('All modules imported')
# import class with two column labels "Normal" and "Abnormal"

df = pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
# read in the data and check the first 5 rows

df.head()
df['class']=df['class'].map({'Normal':0,'Abnormal':1})
# assign our categorical variables to a dataframe

A = df[(df['class'] != 0)]

N = df[(df['class'] == 0)]
# general summary of the dataframe

df.info()
# display the number of rows and columns in a tuple

df.shape
df.head()
# check number of missing values

null_feat = pd.DataFrame(len(df['pelvic_incidence']) - df.isnull().sum(), columns = ['Count'])

null_feat
df.describe()
# Display positive and negative correlation between columns

df.corr()
#correlation

correlation = df.corr()

#tick labels

matrix_cols = correlation.columns.tolist()

#convert to array

corr_array  = np.array(correlation)
tst = df.corr()['class'].copy()

tst = tst.drop('class')

tst.sort_values(inplace=True)

tst.iplot(kind='bar',title='Feature Importances',xaxis_title="Features",

    yaxis_title="Correlation")
def plot_ft1_ft2(ft1, ft2) :  

    trace0 = go.Scatter(

        x = A[ft1],

        y = A[ft2],

        name = 'malignant',

        mode = 'markers', 

        marker = dict(color = '#FFD700',

            line = dict(

                width = 1)))



    trace1 = go.Scatter(

        x = N[ft1],

        y = N[ft2],

        name = 'benign',

        mode = 'markers',

        marker = dict(color = '#7EC0EE',

            line = dict(

                width = 1)))



    layout = dict(title = ft1 +" "+"vs"+" "+ ft2,

                  yaxis = dict(title = ft2,zeroline = False),

                  xaxis = dict(title = ft1, zeroline = False)

                 )



    plots = [trace0, trace1]



    fig = dict(data = plots, layout=layout)

    py.iplot(fig)
plot_ft1_ft2('pelvic_incidence','pelvic_tilt numeric')

plot_ft1_ft2('pelvic_incidence','lumbar_lordosis_angle')

plot_ft1_ft2('pelvic_incidence','sacral_slope')

plot_ft1_ft2('pelvic_incidence','pelvic_radius')

plot_ft1_ft2('pelvic_incidence','degree_spondylolisthesis')



trace = go.Heatmap(z = corr_array,

                   x = matrix_cols,

                   y = matrix_cols,

                   xgap = 2,

                   ygap = 2,

                   colorscale='Viridis',

                   colorbar   = dict() ,

                  )

layout = go.Layout(dict(title = 'Correlation Matrix for variables',

                        autosize = False,

                        height  = 720,

                        width   = 800,

                        margin  = dict(r = 0 ,l = 210,

                                       t = 25,b = 210,

                                     ),

                        yaxis   = dict(tickfont = dict(size = 9)),

                        xaxis   = dict(tickfont = dict(size = 9)),

                       )

                  )

fig = go.Figure(data = [trace],layout = layout)

py.iplot(fig)
#sorts all correlations with ascending sort.

df.corr().unstack().sort_values().drop_duplicates()
df.columns
# import figure factory

import plotly.figure_factory as ff



data_matrix = df.loc[:,['pelvic_incidence', 'pelvic_tilt numeric', 'lumbar_lordosis_angle',

       'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis']]

data_matrix["index"] = np.arange(1,len(data_matrix)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data_matrix, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=1000, width=1000)

fig.update_layout(font=dict(family="Tahoma",size=6),titlefont=dict(size=10))

iplot(fig)
df_abnormal = df[df["class"]==1]

pd.plotting.scatter_matrix(df_abnormal.loc[:, df_abnormal.columns != "class"],

                                       c="red",

                                       figsize= [15,15],

                                       diagonal="hist",

                                       alpha=0.5,

                                       s = 200,

                                       

                          )

plt.show()
df_normal = df[df['class']==0]

pd.plotting.scatter_matrix(df_normal.loc[:, df_normal.columns != "class"],

                                       c="blue",

                                       figsize= [15,15],

                                       diagonal="hist",

                                       alpha=0.5,

                                       s = 200,

                                       

                                       edgecolor= "black")

plt.show()
# prepare data

data1 = len(df["class"][df["class"] == 1])

data2 = len(df["class"][df["class"] == 0])
trace = go.Bar(x=df['class'].value_counts(), y = ['Abnormal', 'Normal'], orientation = 'h', opacity = 0.8, marker=dict(

        color=[ 'black', 'gold'],

        line=dict(color='#000000',width=1.0)))



layout = dict(title =  'Count of diagnosis variable')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)



trace = go.Pie(labels = ['Abnormal','Normal'], values = df['class'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['black', 'gold'], 

                           line=dict(color='#000000', width=1.5)))



layout = dict(title =  'Distribution of diagnosis variable')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
from plotly.subplots import make_subplots



fig = make_subplots(rows=6,cols=1,subplot_titles = ("Pelvic Incidence","Lumbar Lordosis Angle","Pelvic Tilt Numeric","Sacral Slope","Degree Spondylolisthesis","Pelvic Raidus"))



fig.append_trace(go.Scatter(

x = df.index,

y = df.pelvic_incidence,

mode = "lines",

name = "Pelvic Incidence",

marker = dict(color = 'rgba(16, 112, 2, 0.8)')),row = 1, col = 1)



fig.append_trace(go.Scatter(

x = df.index,

y = df["pelvic_tilt numeric"],

mode = "lines",

name = "Pelvic Tilt Numeric",

marker = dict(color = 'rgba(80, 26, 80, 0.8)')),row = 2, col = 1)



fig.append_trace(go.Scatter(

x = df.index,

y = df.lumbar_lordosis_angle,

mode = "lines",

name = "Lumbar Lordosis Angle",

marker = dict(color = 'rgba(160, 112, 20, 0.8)')),row = 3, col = 1)



fig.append_trace(go.Scatter(

x = df.index,

y = df.sacral_slope,

mode = "lines",

name = "Sacral Slope",

marker = dict(color = 'rgba(12, 12, 140, 0.8)')),row = 4, col = 1)



fig.append_trace(go.Scatter(

x = df.index,

y = df.pelvic_radius,

mode = "lines",

name = "Pelvic Radius",

marker = dict(color = 'rgba(245, 128, 2, 0.8)')),row = 5, col = 1)



fig.append_trace(go.Scatter(

x = df.index,

y = df.degree_spondylolisthesis,

mode = "lines",

name = "Degree Spondylolisthesis",

marker = dict(color = 'rgba(235, 144, 235, 0.8)')),row = 6, col = 1) #174



fig.update_xaxes(title_text="Patient Number", row=1, col=1)

fig.update_xaxes(title_text="Patient Number", row=2, col=1)

fig.update_xaxes(title_text="Patient Number", row=3, col=1)

fig.update_xaxes(title_text="Patient Number", row=4, col=1)

fig.update_xaxes(title_text="Patient Number", row=5, col=1)

fig.update_xaxes(title_text="Patient Number", row=6, col=1)



fig.update_yaxes(title_text="Pelvic Incidence", row=1, col=1)

fig.update_yaxes(title_text="Lumbar Lordosis Angle", row=2, col=1)

fig.update_yaxes(title_text="Pelvic Tilt Numeric", row=3, col=1)

fig.update_yaxes(title_text="Sacral Slope", row=4, col=1)

fig.update_yaxes(title_text="Degree Spondylolisthesis", row=5, col=1)

fig.update_yaxes(title_text="Pelvic Radius", row=6, col=1)



fig.update_layout(height = 1800, width = 1000, title = "Biomechanical Features of Patients")



iplot(fig)
y = df["class"].values

x_data = df.drop(["class"], axis=1)  
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
x.head()
# check number of missing values

x_null = pd.DataFrame(x.isnull().sum(), columns = ['Count'])

x_null
print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1)



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
from sklearn.linear_model import LogisticRegression



lr_model = LogisticRegression()

lr_model.fit(x_train,y_train)



#Print Train Accuracy

lr_train_accuracy = lr_model.score(x_train,y_train)

print("lr_train_accuracy = ",lr_model.score(x_train,y_train))

#Print Test Accuracy

lr_test_accuracy = lr_model.score(x_test,y_test)

print("lr_test_accuracy = ",lr_model.score(x_test,y_test))
data = [go.Bar(

            x=["lr_train_accuracy","lr_test_accuracy"],

            y=[lr_train_accuracy,lr_test_accuracy],

            orientation = 'v', opacity = 0.8, marker=dict(

        color=[ 'black', 'gold'],

        line=dict(color='#000000',width=0.2),

        )

 

    )]

fig.update_layout(title='<i><b>Logistic Regression Classification Accuracy</b></i>')

iplot(data, filename='text-hover-bar')
from sklearn.metrics import confusion_matrix

y_pred = lr_model.predict(x_test)

y_true = y_test



cm_lr = confusion_matrix(y_true,y_pred)

import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(cm_lr, colorscale='Viridis')

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>')

fig.show()
tp ,fp ,fn ,tn= cm_lr.ravel()

print("lr_RECALL = ",tp/(tp+fn))

print("lr_PRECISION = ",(tp/(tp+fp))) 
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(x_train,y_train)



#Print Train Accuracy

knn_train_accuracy = knn_model.score(x_train,y_train)

print("knn_train_accuracy = ",knn_model.score(x_train,y_train))

#Print Test Accuracy

knn_test_accuracy = knn_model.score(x_test,y_test)

print("knn_test_accuracy = ",knn_model.score(x_test,y_test))
# Model complexity

neighboors = np.arange(1,30)

train_accuracy = []

test_accuracy = []



for i, k in enumerate(neighboors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    train_accuracy.append(knn.score(x_train, y_train))           

    test_accuracy.append(knn.score(x_test, y_test))              





import plotly.graph_objs as go





trace1 = go.Scatter(

                    x = neighboors,

                    y = train_accuracy,

                    mode = "lines",

                    name = "train_accuracy",

                    marker = dict(color = 'rgba(160, 112, 2, 0.8)'),

                    text= "train_accuracy")



trace2 = go.Scatter(

                    x = neighboors,

                    y = test_accuracy,

                    mode = "lines+markers",

                    name = "test_accuracy",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= "test_accuracy")

data = [trace1, trace2]

layout = dict(title = 'K Value vs Accuracy',

              xaxis= dict(title= 'Number of Neighboors',ticklen= 10,zeroline= True)

             )

fig = dict(data = data, layout = layout)

iplot(fig)



knn_train_accuracy_two = np.max(train_accuracy)

knn_test_accuracy_two = np.max(test_accuracy)

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))
data = [go.Bar(

            x=["knn_train_accuracy","knn_test_accuracy"],

            y=[knn_train_accuracy,knn_test_accuracy],

     orientation = 'v', opacity = 0.8, marker=dict(

        color=[ 'black', 'gold'],

        line=dict(color='#000000',width=0.2),

        )

 

    )]



iplot(data, filename='text-hover-bar')
y_pred = knn_model.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm_knn = confusion_matrix(y_true,y_pred)



import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(cm_knn, colorscale='Viridis')

fig.update_layout(title_text='<i><b>KNN Classification Confusion matrix</b></i>')

fig.show()
tp ,fp ,fn ,tn= cm_knn.ravel()

print("knn_RECALL = ",tp/(tp+fn))

print("knn_PRECISION = ",(tp/(tp+fp)))
from sklearn.svm import SVC



svm_model = SVC(random_state=1)

svm_model.fit(x_train,y_train)



#Print Train Accuracy

svm_train_accuracy = svm_model.score(x_train,y_train)

print("svm_train_accuracy = ",svm_model.score(x_train,y_train))

#Print Test Accuracy

svm_test_accuracy = svm_model.score(x_test,y_test)

print("svmr_test_accuracy = ",svm_model.score(x_test,y_test))
data = [go.Bar(

            x=["svm_train_accuracy","svm_test_accuracy"],

            y=[svm_train_accuracy,svm_test_accuracy],

     orientation = 'v', opacity = 0.8, marker=dict(

        color=[ 'black', 'gold'],

        line=dict(color='#000000',width=0.2),

        )

 

    )]



iplot(data, filename='text-hover-bar')
y_pred = svm_model.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm_svm = confusion_matrix(y_true,y_pred)



import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(cm_svm, colorscale='Viridis')

fig.update_layout(title_text='<i><b>Support Vector Confusion matrix</b></i>')

fig.show()
tp ,fp ,fn ,tn= cm_svm.ravel()

print("svm_RECALL = ",tp/(tp+fn))

print("svm_PRECISION = ",(tp/(tp+fp)))
from sklearn.naive_bayes import GaussianNB



nb_model = GaussianNB()

nb_model.fit(x_train,y_train)



#Print Train Accuracy

nb_train_accuracy = nb_model.score(x_train,y_train)

print("nb_train_accuracy = ",nb_model.score(x_train,y_train))

#Print Test Accuracy

nb_test_accuracy = nb_model.score(x_test,y_test)

print("nb_test_accuracy = ",nb_model.score(x_test,y_test))
data = [go.Bar(

            x=["nb_train_accuracy","nb_test_accuracy"],

            y=[nb_train_accuracy,nb_test_accuracy],

     orientation = 'v', opacity = 0.8, marker=dict(

        color=[ 'black', 'gold'],

        line=dict(color='#000000',width=0.2),

        )

 

    )]



iplot(data, filename='text-hover-bar')
y_pred = nb_model.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm_nb = confusion_matrix(y_true,y_pred)

import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(cm_nb, colorscale='Viridis')

fig.update_layout(title_text='<i><b>Naive Bayes Classification Confusion matrix</b></i>')

fig.show()
tp ,fp ,fn ,tn= cm_nb.ravel()

print("nb_RECALL = ",tp/(tp+fn))

print("nb_PRECISION = ",(tp/(tp+fp)))
from sklearn.tree import DecisionTreeClassifier

#if you remove random_state=1, you can see how accuracy is changing

#Accuracy changing depends on splits

dt_model = DecisionTreeClassifier(random_state=1)

dt_model.fit(x_train,y_train)



#Print Train Accuracy

dt_train_accuracy = dt_model.score(x_train,y_train)

print("dt_train_accuracy = ",dt_model.score(x_train,y_train))

#Print Test Accuracy

dt_test_accuracy = dt_model.score(x_test,y_test)

print("dt_test_accuracy = ",dt_model.score(x_test,y_test))
data = [go.Bar(

            x=["dt_train_accuracy","dt_test_accuracy"],

            y=[dt_train_accuracy,dt_test_accuracy],

     orientation = 'v', opacity = 0.8, marker=dict(

        color=[ 'black', 'gold'],

        line=dict(color='#000000',width=0.2),

        )

 

    )]



iplot(data, filename='text-hover-bar')
y_pred = dt_model.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm_dt = confusion_matrix(y_true,y_pred)



import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(cm_dt, colorscale='Viridis')

fig.update_layout(title_text='<i><b>Decision Tree Classification Confusion matrix</b></i>')

fig.show()
tp ,fp ,fn ,tn= cm_dt.ravel()

print("dt_RECALL = ",tp/(tp+fn))

print("dt_PRECISION = ",(tp/(tp+fp)))
from sklearn.ensemble import RandomForestClassifier



#n_estimators = 100 => Indicates how many trees we have

rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

rf_model.fit(x_train,y_train)



#Print Train Accuracy

rf_train_accuracy = rf_model.score(x_train,y_train)

print("rf_train_accuracy = ",rf_model.score(x_train,y_train))

#Print Test Accuracy

rf_test_accuracy = rf_model.score(x_test,y_test)

print("rf_test_accuracy = ",rf_model.score(x_test,y_test))
data = [go.Bar(

            x=["rf_train_accuracy","rf_test_accuracy"],

            y=[rf_train_accuracy,rf_test_accuracy],

     orientation = 'v', opacity = 0.8, marker=dict(

        color=[ 'black', 'gold'],

        line=dict(color='#000000',width=0.2),

        )

 

    )]



iplot(data, filename='text-hover-bar')
y_pred
y_pred = ["Normal" if x < 0.5 else "Abnormal" for x in y_pred]
y_pred[1:6]
y_pred = rf_model.predict(x_test)

y_true = y_test



from sklearn.metrics import confusion_matrix



cm_rf = confusion_matrix(y_true,y_pred)



import plotly.figure_factory as ff

fig = ff.create_annotated_heatmap(cm_rf, colorscale='Viridis')

fig.update_layout(title_text='<i><b>Random Forest Classification Confusion matrix</b></i>')

fig.show()
tp ,fp ,fn ,tn= cm_rf.ravel()

print("rf_RECALL = ",tp/(tp+fn))

print("rf_PRECISION = ",(tp/(tp+fp)))
from plotly.subplots import make_subplots



fig = make_subplots(rows=6,cols=1,subplot_titles = ("Logistic Regression Classification","Decision Tree Classification","K Nearest Neighbors(KNN) Classification","Naive Bayes Classification","Random Forest Classification","Support Vector Machine(SVM) Classification"))

fig.append_trace(go.Heatmap(z=cm_lr),row = 1, col = 1)

fig.append_trace(go.Heatmap(z=cm_dt,text=cm_lr,),row = 2, col = 1)

fig.append_trace(go.Heatmap(z=cm_knn,text=cm_lr,),row =3, col = 1)

fig.append_trace(go.Heatmap(z=cm_nb,text=cm_lr,),row = 4, col = 1)

fig.append_trace(go.Heatmap(z=cm_rf,text=cm_lr,),row = 5, col = 1)

fig.append_trace(go.Heatmap(z=cm_svm,text=cm_lr,),row = 6, col = 1)

fig.update_layout(height=1500, width=800, title_text="Patients' Classes According to Biomechanical Features")

fig.update_traces(showscale=False)

iplot(fig)
svm_test_accuracy
knn_test_accuracy_two
lr_test_accuracy
nb_test_accuracy
dt_test_accuracy
rf_test_accuracy
lr_s = lr_test_accuracy.round(3)

knn_s = knn_test_accuracy_two.round(4)

svm_s = svm_test_accuracy.round(3)

nb_s = nb_test_accuracy.round(2)

dt_s = dt_test_accuracy.round(3)

rf_s = rf_test_accuracy.round(3)



list_scores = [lr_s,knn_s,svm_s,nb_s,dt_s,rf_s]

list_scores.sort()

list_names = []



for i in list_scores:

    if i == lr_s:

        list_names.append("Logistic Regression")

    elif i == knn_s:

        list_names.append("KNN")

    elif i == svm_s:

        list_names.append("SVM")

    elif i == nb_s:

        list_names.append("NB")

    elif i == dt_s:

        list_names.append("Decision Tree")

    elif i == rf_s:

        list_names.append("Random Forest")



trace1 = go.Bar(

    x = list_names,

    y = list_scores,

    text = list_scores,

    textposition = "inside",

    marker=dict(color = list_scores,colorbar=dict(

            title="Colorbar"

        ),colorscale="Viridis",))



data = [trace1]

layout = go.Layout(title = "Comparison of Models")



fig = go.Figure(data = data, layout = layout)

fig.update_xaxes(title_text = "Names")

fig.update_yaxes(title_text = "Scores")

fig.show()
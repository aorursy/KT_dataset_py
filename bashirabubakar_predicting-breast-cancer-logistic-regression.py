print('Hello My name is Bashir Abubakar and welcome to this exploration!')
# import necessary libraries
# data cleaning and manipulation 
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip install chart_studio
!pip install cufflinks
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

# machine learning
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.feature_selection import RFE
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score

# initialize some package settings
sns.set(style="whitegrid", color_codes=True, font_scale=1.3)

%matplotlib inline

print('All modules imported')
# read in the data and check the first 10 rows
df = pd.read_csv('../input/data.csv')
df.head(10)
# general summary of the dataframe
df.info()
# check number of missing values
null_feat = pd.DataFrame(len(df['id']) - df.isnull().sum(), columns = ['Count'])
null_feat
# remove the 'Unnamed: 32' column
df = df.drop('Unnamed: 32', axis=1)
# Reassign target
df.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)
# check the data type of each column
df.dtypes
# drop the id column as well and check the dataframe
df=df.drop("id",axis=1)
df.head()
# assign our categorical variables to a dataframe
M = df[(df['diagnosis'] != 0)]
B = df[(df['diagnosis'] == 0)]
# check what the dataframe looks like
df.head()
trace = go.Bar(x = (len(M), len(B)), y = ['malignant', 'benign'], orientation = 'h', opacity = 0.8, marker=dict(
        color=[ 'gold', 'black'],
        line=dict(color='#000000',width=1.0)))

layout = dict(title =  'Count of diagnosis variable')
                    
fig = dict(data = [trace], layout=layout)
py.iplot(fig)

trace = go.Pie(labels = ['benign','malignant'], values = df['diagnosis'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['black', 'gold'], 
                           line=dict(color='#000000', width=1.5)))

layout = dict(title =  'Distribution of diagnosis variable')
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)
benign, malignant = df['diagnosis'].value_counts()
print('Number of cells labeled Benign: ', benign)
print('Number of cells labeled Malignant : ', malignant)
print('')
print('% of cells labeled Benign', round(benign / len(df) * 100, 2), '%')
print('% of cells labeled Malignant', round(malignant / len(df) * 100, 2), '%')
def plot_distribution(df_f, size_bin) :  
    tmp1 = M[df_f]
    tmp2 = B[df_f]
    hist_data = [tmp1, tmp2]
    
    group_labels = ['malignant', 'benign']
    colors = ['#FFD700', '#7EC0EE']

    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, bin_size = size_bin, curve_type='kde')
    
    fig['layout'].update(title = df_f)

    py.iplot(fig, filename = 'Density plot')
plot_distribution('radius_mean', .5)
plot_distribution('texture_mean', .5)
plot_distribution('perimeter_mean', 5)
plot_distribution('area_mean', 10)
#plot_distribution('smoothness_mean', .5)
#plot_distribution('compactness_mean' .5)
#plot_distribution('concavity_mean' .5)
#plot_distribution('concave points_mean' .5)
#plot_distribution('symmetry_mean' .5)
#plot_distribution('fractal_dimension_mean' .5)
#correlation
correlation = df.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)
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
def plot_ft1_ft2(ft1, ft2) :  
    trace0 = go.Scatter(
        x = M[ft1],
        y = M[ft2],
        name = 'malignant',
        mode = 'markers', 
        marker = dict(color = '#FFD700',
            line = dict(
                width = 1)))

    trace1 = go.Scatter(
        x = B[ft1],
        y = B[ft2],
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
plot_ft1_ft2('perimeter_mean','radius_worst')
plot_ft1_ft2('area_mean','radius_worst')
plot_ft1_ft2('texture_mean','texture_worst')
plot_ft1_ft2('area_worst','radius_worst')
plot_ft1_ft2('smoothness_mean','texture_mean')
plot_ft1_ft2('radius_mean','fractal_dimension_worst')
plot_ft1_ft2('texture_mean','symmetry_mean')
plot_ft1_ft2('texture_mean','symmetry_se')
plot_ft1_ft2('area_mean','fractal_dimension_mean')
plot_ft1_ft2('radius_mean','fractal_dimension_mean')
plot_ft1_ft2('area_mean','smoothness_se')
plot_ft1_ft2('smoothness_se','perimeter_mean')
df.head()
# define X, y functions for our model
X=df.drop('diagnosis',axis=1)
X.head()
y=df['diagnosis']
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
logReg = LogisticRegression(solver = 'lbfgs', max_iter=9000,multi_class = 'multinomial', random_state = 42)

start_time=time.time()

logReg.fit(X_train, y_train)

end_time=time.time()

print("---%s seconds ---" % (end_time - start_time))
y_pred = logReg.predict(X_test)
print(y_pred.shape)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print('Accuracy of logistic regression classifier on train set: {:.2f}'.format(logReg.score(X_train, y_train)))
X_train
X_test
y_pred[1:6]
y_pred = ["M" if x < 0.5 else "B" for x in y_pred]
y_pred[1:6]
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score
# Validating the train on the model
y_train_pred =logReg.predict(X_train)
y_train_prob =logReg.predict_proba(X_train)[:,1]

print("Accuracy Score of train", accuracy_score(y_train,y_train_pred))
print("AUC of the train ", roc_auc_score(y_train,y_train_prob))
print(" confusion matrix \n" , confusion_matrix(y_train,y_train_pred))
# Validating the test on the model
y_test_pred=logReg.predict(X_test)
y_test_prob=logReg.predict_proba(X_test)[:,1]

print("Accuracy Score of test", accuracy_score(y_test,y_test_pred))
print("AUC od the test ", roc_auc_score(y_test,y_test_prob))
print(" confusion matrix \n" , confusion_matrix(y_test,y_test_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test,y_test_pred))
# read in the data and check the first 10 rows
dataset = pd.read_csv('../input/data.csv')
dataset.head(10)
cols = ['id', 
        'Unnamed: 32']
dataset = dataset.drop(cols, axis=1)
dataset.head()
# drop unncessary columns
dataset=dataset.drop(["perimeter_mean","radius_mean","compactness_mean","concave points_mean","radius_se","perimeter_se",
                     "radius_worst","perimeter_worst","compactness_worst","concave points_worst","compactness_se",
                     "concave points_se","texture_worst","area_worst"],axis=1)
X1=dataset.drop("diagnosis",axis=1)
X1.head()
y1=dataset['diagnosis']
y1.head()
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state = 42)
logReg1 = LogisticRegression(solver = 'lbfgs', max_iter=9000,multi_class = 'multinomial', random_state = 42)

start_time=time.time()

logReg1.fit(X1_train, y1_train)

end_time=time.time()

print("---%s seconds ---" % (end_time - start_time))
y1_pred = logReg1.predict(X1_test)
print(y1_pred.shape)
print(accuracy_score(y1_test, y1_pred))
print(confusion_matrix(y1_test, y1_pred))
# Validating the train on the model
y1_train_pred =logReg1.predict(X1_train)
y1_train_prob =logReg1.predict_proba(X1_train)[:,1]

print("Accuracy Score of train", accuracy_score(y1_train,y1_train_pred))
print("AUC of the train ", roc_auc_score(y1_train,y1_train_prob))
print(" confusion matrix \n" , confusion_matrix(y1_train,y1_train_pred))
# Validating the test on the model
y1_test_pred=logReg1.predict(X1_test)
y1_test_prob=logReg1.predict_proba(X1_test)[:,1]

print("Accuracy Score of test", accuracy_score(y1_test,y1_test_pred))
print("AUC of the test ", roc_auc_score(y1_test,y1_test_prob))
print(" confusion matrix \n" , confusion_matrix(y1_test,y1_test_pred))
from sklearn.metrics import classification_report

print(classification_report(y1_test,y1_test_pred))
new_df = pd.read_csv('../input/data - Copy.csv')
# drop unncessary columns
new_df=new_df.drop(["perimeter_mean","radius_mean","compactness_mean","concave points_mean","radius_se","perimeter_se",
                     "radius_worst","perimeter_worst","compactness_worst","concave points_worst","compactness_se",
                     "concave points_se","texture_worst","area_worst","fractal_dimension_worst","id"],axis=1)
new_df.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)
new_df
prediction = logReg1.predict(new_df)
prediction
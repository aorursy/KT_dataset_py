# Importing numpy, pandas and Series + DataFrame:

import numpy as np

import pandas as pd

from pandas import Series, DataFrame



# Imports for plotly:

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



# Imports for plotting:

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
# Importing and loading the iris dataset

iris = pd.read_csv('../input/Iris.csv')
# Show first 5 rows of dataset:

header = ff.create_table(iris.head())



header.show()
# Check unique values for Species:

print(iris['Species'].unique())
# Function to describe variables

def desc(df):

    d = pd.DataFrame(df.dtypes,columns=['Data_Types'])

    d = d.reset_index()

    d['Columns'] = d['index']

    d = d[['Columns','Data_Types']]

    d['Missing'] = df.isnull().sum().values    

    d['Uniques'] = df.nunique().values

    return d





descr = ff.create_table(desc(iris))



descr.show()
# Distritution of Species:



s_df = pd.DataFrame(iris.groupby(['Species'])['Species'].count())



data=go.Bar(x = s_df.index

           , y = s_df.Species

           ,  marker=dict( color=['#0e9aa7', '#f6cd61', '#fe8a71'])

           )







layout = go.Layout(title = 'Distribution of Iris Species'

                   , xaxis = dict(title = 'Species')

                   , yaxis = dict(title = 'Volume')

                  )



fig = go.Figure(data,layout)

fig.show()
# Create a dataset for each of the species:

setosa = iris[iris.Species == 'Iris-setosa']

versicolor = iris[iris.Species == 'Iris-versicolor']

virginica = iris[iris.Species == 'Iris-virginica']



# Histogram data for Sepal Length

hist_data  = [setosa.SepalLengthCm, versicolor.SepalLengthCm, virginica.SepalLengthCm]



group_labels = ['setosa', 'versicolor', 'virginica']

colors = ['#0e9aa7', '#f6cd61', '#fe8a71']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels, colors=colors,

                         bin_size=.1, show_rug=False)

# Add title

fig.update_layout(title_text='Histogram for Sepal Length'

                  , xaxis = dict(title = 'lenght (cm)')

                  , yaxis = dict(title = 'count')

                 )





fig.show()
# Histogram data for Sepal Width



hist_data  = [setosa.SepalWidthCm, versicolor.SepalWidthCm, virginica.SepalWidthCm]



group_labels = ['setosa', 'versicolor', 'virginica']

colors = ['#0e9aa7', '#f6cd61', '#fe8a71']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels, colors=colors,

                         bin_size=.1, show_rug=False)

# Add title

fig.update_layout(title_text='Histogram for Sepal Width'

                  , xaxis = dict(title = 'width (cm)')

                  , yaxis = dict(title = 'count')

                 )



fig.show()

# Histogram data for Petal Length



hist_data  = [setosa.PetalLengthCm, versicolor.PetalLengthCm, virginica.PetalLengthCm]



group_labels = ['setosa', 'versicolor', 'virginica']

colors = ['#0e9aa7', '#f6cd61', '#fe8a71']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels, colors=colors,

                         bin_size=.1, show_rug=False)

# Add title

fig.update_layout(title_text='Histogram for Petal Length'

                  , xaxis = dict(title = 'lenght (cm)')

                  , yaxis = dict(title = 'count')

                 )



fig.show()
# Histogram data for Petal Width



hist_data  = [setosa.PetalWidthCm, versicolor.PetalWidthCm, virginica.PetalWidthCm]



group_labels = ['setosa', 'versicolor', 'virginica']

colors = ['#0e9aa7', '#f6cd61', '#fe8a71']



# Create distplot with curve_type set to 'normal'

fig = ff.create_distplot(hist_data, group_labels, colors=colors,

                         bin_size=.1, show_rug=False)

# Add title

fig.update_layout(title_text='Histogram for Petal Width'

                  , xaxis = dict(title = 'width (cm)')

                  , yaxis = dict(title = 'count')

                 )



fig.show()

# Scattergraph for Iris Sepal (length vs width):



fig = go.Figure()



fig.add_trace(go.Scatter(

      x=setosa.SepalLengthCm

    , y=setosa.SepalWidthCm

    , name='setosa'

    , mode='markers'

    , marker_color='#0e9aa7'

))



fig.add_trace(go.Scatter(

      x=versicolor.SepalLengthCm

    , y=versicolor.SepalWidthCm

    , name='versicolor'

    , mode='markers'

    , marker_color='#f6cd61'

))



fig.add_trace(go.Scatter(

      x=virginica.SepalLengthCm

    , y=virginica.SepalWidthCm

    , name='virginica'

    , mode='markers'

    , marker_color='#fe8a71'

))



# Set options common to all traces with fig.update_traces

fig.update_traces(mode='markers'

                 # , marker_line_width=2

                  , marker_size=10)



fig.update_layout(title='Iris Sepal (length vs width)'

                  , xaxis = dict(title = 'length (cm)')

                  , yaxis = dict(title = 'width (cm)')

                 )





fig.show()
# Scattergraph for Iris Petal (length vs width):



fig = go.Figure()



fig.add_trace(go.Scatter(

      x=setosa.PetalLengthCm

    , y=setosa.PetalWidthCm

    , name='setosa'

    , mode='markers'

    , marker_color='#0e9aa7'

))



fig.add_trace(go.Scatter(

      x=versicolor.PetalLengthCm

    , y=versicolor.PetalWidthCm

    , name='versicolor'

    , mode='markers'

    , marker_color='#f6cd61'

))



fig.add_trace(go.Scatter(

      x=virginica.PetalLengthCm

    , y=virginica.PetalWidthCm

    , name='virginica'

    , mode='markers'

    , marker_color='#fe8a71'

))



# Set options common to all traces with fig.update_traces

fig.update_traces(mode='markers'

                 # , marker_line_width=2

                  , marker_size=10)



fig.update_layout(title='Iris Petal (length vs width)'

                  , xaxis = dict(title = 'length (cm)')

                  , yaxis = dict(title = 'width (cm)')

                 )





fig.show()
# Plotting the features of our dataset (this gives us density graphs and scatter plots): 



columns = list(iris.columns)[1:] # remove id column



sns.set(style="ticks")

sns.pairplot(iris[columns]

             , hue='Species'

             , palette=['#0e9aa7', '#f6cd61', '#fe8a71']

             , diag_kind = 'kde'

             , height = 2.8)

plt.show()
# Box plot for Sepal Length:



fig = go.Figure()

fig.add_trace(go.Box(y=setosa.SepalLengthCm, name='setosa', marker_color='#0e9aa7'))

fig.add_trace(go.Box(y=virginica.SepalLengthCm, name='virginica', marker_color='#fe8a71'))

fig.add_trace(go.Box(y=versicolor.SepalLengthCm, name = 'versicolor',  marker_color='#f6cd61'))



fig.update_layout(title='Sepal Length'

                  , xaxis = dict(title = 'species')

                  , yaxis = dict(title = 'length (cm)')

                 )



fig.show()
# Box plot for Sepal Width:



fig = go.Figure()

fig.add_trace(go.Box(y=setosa.SepalWidthCm, name='setosa', marker_color='#0e9aa7'))

fig.add_trace(go.Box(y=virginica.SepalWidthCm, name='virginica', marker_color='#fe8a71'))

fig.add_trace(go.Box(y=versicolor.SepalWidthCm, name = 'versicolor',  marker_color='#f6cd61'))



fig.update_layout(title='Sepal Width'

                  , xaxis = dict(title = 'species')

                  , yaxis = dict(title = 'length (cm)')

                 )



fig.show()
# Box plot for Petal Length:



fig = go.Figure()

fig.add_trace(go.Box(y=setosa.PetalLengthCm, name='setosa', marker_color='#0e9aa7'))

fig.add_trace(go.Box(y=virginica.PetalLengthCm, name='virginica', marker_color='#fe8a71'))

fig.add_trace(go.Box(y=versicolor.PetalLengthCm, name = 'versicolor',  marker_color='#f6cd61'))



fig.update_layout(title='Petal Length'

                  , xaxis = dict(title = 'species')

                  , yaxis = dict(title = 'length (cm)')

                 )



fig.show()
# Box plot for Petal Width:



fig = go.Figure()

fig.add_trace(go.Box(y=setosa.PetalWidthCm, name='setosa', marker_color='#0e9aa7'))

fig.add_trace(go.Box(y=virginica.PetalWidthCm, name='virginica', marker_color='#fe8a71'))

fig.add_trace(go.Box(y=versicolor.PetalWidthCm, name = 'versicolor',  marker_color='#f6cd61'))



fig.update_layout(title='Sepal Width'

                  , xaxis = dict(title = 'species')

                  , yaxis = dict(title = 'length (cm)')

                 )



fig.show()
# Create Parallel Coordinates:



def spc_id(i):    

    if i == 'Iris-setosa':

        return 1

    elif i == 'Iris-versicolor':

        return 2

    else:

        return 3



iris['species_id'] = iris['Species'].apply(spc_id)

iris = iris.drop('Id', axis = 1)





fig = px.parallel_coordinates(iris

                              , color='species_id'

                              , labels={'species_id':'Species'

                                        ,'SepalWidthCm':'Sepal Width'

                                        ,'SepalLengthCm':'Sepal Length'

                                        ,'PetalWidthCm':'Petal Width'

                                        ,'PetalLengthCm':'Petal Length'}

                              , color_continuous_scale = ['#0e9aa7', '#f6cd61', '#fe8a71']

                              , color_continuous_midpoint=2                           

                             )

fig.show()
from sklearn.model_selection import train_test_split
# Defining target set y, and a training set X:

y = iris.Species

X = iris.drop(['Species','species_id'], axis = 1)



# Split data into train and test part:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)
# The most important parameter of k-Nearest Neighbors classifier is the number of neighbors, which we will set to 17:

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 17)
# Fitting the data with knn model:

knn.fit(X_train, y_train)
# Using the predict method on KNN to predict values for X_test:

y_pred = knn.predict(X_test)
print('Test set score {:.2f}'.format(knn.score(X_test,y_test)))
# Importing classification_method and confusion_matrix:

from sklearn.metrics import classification_report, confusion_matrix
# Printing out classification report:

print(classification_report(y_test,y_pred))
z = confusion_matrix(y_test, y_pred)



x = ['setosa', 'versicolor', 'virginica']

y = ['setosa', 'versicolor', 'virginica']



# change each element of z to type string for annotations

z_text = [[str(y) for y in x] for x in z]



# set up figure 

fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Portland')



# add title

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',

                  #xaxis = dict(title='x'),

                  #yaxis = dict(title='x')

                 )



# add custom xaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=0.5,

                        y=-0.15,

                        showarrow=False,

                        text="Predicted value",

                        xref="paper",

                        yref="paper"))



# add custom yaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=-0.35,

                        y=0.5,

                        showarrow=False,

                        text="Real value",

                        textangle=-90,

                        xref="paper",

                        yref="paper"))



# adjust margins to make room for yaxis title

fig.update_layout(margin=dict(t=50, l=200))



# add colorbar

fig['data'][0]['showscale'] = True

fig.show()
# Creating a for loop that trains various KNN models with different K values:

# Keeping a track of the error_rate for each of these models with a list

error_rate = []



for i in range(1,50,2):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
# Line graph for k vs. error rate:



x = list(range(1,50,2))



fig = go.Figure()

fig.add_trace(go.Scatter(x=x

                         , y=error_rate

                         , mode='lines'

                         , name='Error Rate line'

                        )

             )



fig.add_trace(go.Scatter(x=x

                         , y=error_rate

                         , mode='markers'

                         , name='Error Rate point'

                        )

             )



fig.update_layout(title='Line graph for K value vs. Error Rate'

                  , xaxis_title='K'

                  , yaxis_title='Error Rate'

                 )



fig.show()
# Using brute force to find best value for K:



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



cv_scores = []

neighbors = list(np.arange(1,50,2))

for n in neighbors:

    knn = KNeighborsClassifier(n_neighbors = n,algorithm = 'brute')    

    cross_val = cross_val_score(knn,X_train,y_train,cv = 5 , scoring = 'accuracy')

    cv_scores.append(cross_val.mean())

    

error = [1-x for x in cv_scores]

optimal_n = neighbors[ error.index(min(error)) ]

knn_optimal = KNeighborsClassifier(n_neighbors = optimal_n,algorithm = 'brute')

knn_optimal.fit(X_train,y_train)

pred = knn_optimal.predict(X_test)

acc = accuracy_score(y_test,pred)*100



print("The accuracy for optimal K = {0} using brute is {1}".format(optimal_n,acc))
# NOW WITH K=5

knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=5')

print('\n')



print(classification_report(y_test,pred))
# Graph for Decision Boundaries



from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets



n_neighbors = 5



# prepare data

X=iris.values[:, :2]

y=iris.species_id

h = .02



# Create color maps

cmap_light = ListedColormap(['#0e9aa7', '#f6cd61', '#fe8a71'])

cmap_bold = ListedColormap(['#154360', '#7D6608', '#900C3F'])



# we create an instance of Neighbours Classifier and fit the data.

clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

clf.fit(X, y)



# calculate min, max and limits

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

np.arange(y_min, y_max, h))



# predict class using data and kNN classifier

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(figsize=(12,7))

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



# Plot also the training points

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())



plt.title("3-Class classification (k = %i)" % (n_neighbors))

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')



plt.show()

z = confusion_matrix(y_test, pred)



x = ['setosa', 'versicolor', 'virginica']

y = ['setosa', 'versicolor', 'virginica']



# change each element of z to type string for annotations

z_text = [[str(y) for y in x] for x in z]



# set up figure 

fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Portland')



# add title

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',

                  #xaxis = dict(title='x'),

                  #yaxis = dict(title='x')

                 )



# add custom xaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=0.5,

                        y=-0.15,

                        showarrow=False,

                        text="Predicted value",

                        xref="paper",

                        yref="paper"))



# add custom yaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=-0.35,

                        y=0.5,

                        showarrow=False,

                        text="Real value",

                        textangle=-90,

                        xref="paper",

                        yref="paper"))



# adjust margins to make room for yaxis title

fig.update_layout(margin=dict(t=50, l=200))



# add colorbar

fig['data'][0]['showscale'] = True

fig.show()
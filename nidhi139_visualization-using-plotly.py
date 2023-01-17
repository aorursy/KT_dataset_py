# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree

import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import accuracy_score





import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from sklearn.datasets import load_wine



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_wine = load_wine()

print(data_wine.DESCR)
for num,value in data_wine.items():

    print(num,'\n',value,'\n')
#Check the size of features in data_wine

print('data.shape\t',data_wine['data'].shape,'\ntarget.shape \t',data_wine['target'].shape)
wine = pd.DataFrame(data_wine['data'],columns= data_wine['feature_names'])

wine['target'] = data_wine['target']

wine.head()
# Check for the null values in dataset

wine.isnull().sum()
wine.describe()
#Lets Analyse some of the basic contents in wine

#univariate Analysis using plotly

fig, arr = plt.subplots(3,3,figsize = (30,15))



wine['alcohol'].plot.hist(ax=arr[0][0],fontsize = 10, color = 'violet')

arr[0][0].set_title('Content of Alcohol',fontsize=12)



wine['malic_acid'].plot.hist(ax=arr[0][1],fontsize = 10, color = 'crimson')

arr[0][2].set_title('Content of ash',fontsize=12)



wine['ash'].plot.hist(ax=arr[0][2],fontsize = 10, color = 'y')

arr[0][2].set_title('Content of ash',fontsize=12)



wine['alcalinity_of_ash'].plot.hist(ax=arr[1][0],fontsize = 10, color = 'aquamarine')

arr[1][0].set_title('Content of alcalinity_of_ash',fontsize=12)



wine['magnesium'].value_counts().plot.hist(ax=arr[1][1],fontsize = 10, color = 'red')

arr[1][1].set_title('Content of magnesium',fontsize=12)



wine['total_phenols'].value_counts().plot.hist(ax=arr[1][2],fontsize = 10, color = 'brown')

arr[1][2].set_title('Content of total_phenols',fontsize=12)



wine['flavanoids'].value_counts().plot.hist(ax=arr[2][0],fontsize = 10, color = 'pink')

arr[2][0].set_title('Content of flavanoids',fontsize=12)



wine['nonflavanoid_phenols'].value_counts().plot.hist(ax=arr[2][1],fontsize = 10, color = 'indigo')

arr[2][1].set_title('Content of nonflavanoid_phenols',fontsize=12)



wine['proanthocyanins'].value_counts().plot.hist(ax=arr[2][2],fontsize = 10, color = 'tomato')

arr[2][2].set_title('Content of proanthocyanins',fontsize=12)
sns.distplot(wine['color_intensity'],color= 'indigo')

plt.show()
fig = go.Figure(data=go.Histogram(x=wine['od280/od315_of_diluted_wines'],marker_color = 'mediumorchid',xbins= dict(start =0.4,end=4,size=0.2)))



fig.update_layout(title_text = 'od280/od315_of_diluted_wines',

                 xaxis_title_text = 'od280/od315_of_diluted_wines',

                 yaxis_title_text ='COUNT',

                 bargap = 0.05)

fig.show()
fig = go.Figure(data=go.Histogram(x=wine['proline'],marker_color = 'crimson'))



fig.update_layout(title_text = 'proline',

                 xaxis_title_text = 'proline',

                 yaxis_title_text ='COUNT',

                 bargap = 0.05)

fig.show()
fig = px.pie(wine,values = 'alcohol',names = 'target',title = 'Analyse Alcohol with Category 0,1 and 2')

fig.show()
wine.head()
fig = px.scatter_matrix(wine, dimensions = ['ash','malic_acid','hue','od280/od315_of_diluted_wines','proline','alcalinity_of_ash'],

                       color='target')



fig.show()
fig = px.scatter(wine, x="magnesium", y="total_phenols", color="target",

                 size='od280/od315_of_diluted_wines', hover_data=['color_intensity'])

fig.show()
fig = px.pie(wine, values='color_intensity', names='target', color='target')

             

fig.show()
fig = px.box(wine, x="target", y="nonflavanoid_phenols",points = 'all')

fig.show()
wine.columns
#Linear Regression



# Split data into training and test splits

train_idx, test_idx = train_test_split(wine.index, test_size=.25, random_state=0)

wine['split'] = 'train'

wine.loc[test_idx, 'split'] = 'test'



X = wine[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',

       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',

       'proanthocyanins', 'color_intensity', 'hue',

       'od280/od315_of_diluted_wines', 'proline']]

y = wine['target']



X_train = wine.loc[train_idx, ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',

       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',

       'proanthocyanins', 'color_intensity', 'hue',

       'od280/od315_of_diluted_wines', 'proline']]

y_train = wine.loc[train_idx, 'target']



# Condition the model on the features.

model = LinearRegression()

model.fit(X_train, y_train)

wine['prediction'] = model.predict(X)

wine['residual'] = wine['prediction'] - wine['target']



fig = px.scatter(

    wine, x='prediction', y='residual',

    marginal_y='violin',

    color='split', trendline='ols'

)

fig.show()
N_FOLD =5



# Define and fit the grid

model = DecisionTreeRegressor()

param_grid = {

    'criterion': ['mse', 'friedman_mse', 'mae'],

    'max_depth': range(2, 5)

}

grid = GridSearchCV(model, param_grid, cv=N_FOLD)

grid.fit(X, y)

grid_df = pd.DataFrame(grid.cv_results_)



# Convert the wide format of the grid into the long format

# accepted by plotly.express

melted = (

    grid_df

    .rename(columns=lambda col: col.replace('param_', ''))

    .melt(

        value_vars=[f'split{i}_test_score' for i in range(N_FOLD)],

        id_vars=['mean_test_score', 'mean_fit_time', 'criterion', 'max_depth'],

        var_name="cv_split",

        value_name="r_squared"

    )

)



# Format the variable names for simplicity

melted['cv_split'] = (

    melted['cv_split']

    .str.replace('_test_score', '')

    .str.replace('split', '')

)



# Single function call to plot each figure

fig_hmap = px.density_heatmap(

    melted, x="max_depth", y='criterion',

    histfunc="sum", z="r_squared",

    title='Grid search results on individual fold',

    hover_data=['mean_fit_time'],

    facet_col="cv_split", facet_col_wrap=3,

    labels={'mean_test_score': "mean_r_squared"}

)



fig_box = px.box(

    melted, x='max_depth', y='r_squared',

    title='Grid search results ',

    hover_data=['mean_fit_time'],

    points='all',

    color="criterion",

    hover_name='cv_split',

    labels={'mean_test_score': "mean_r_squared"}

)



# Display

fig_hmap.show()

fig_box.show()
# Condition the model on features, predict the target

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)



fig = px.scatter(x=y, y=y_pred, labels={'x': 'ground truth', 'y': 'prediction'})

fig.add_shape(

    type="line", line=dict(dash='dash'),

    x0=y.min(), y0=y.min(),

    x1=y.max(), y1=y.max()

)

fig.show()
X_train,X_test,y_train,y_test = train_test_split(data_wine['data'],data_wine['target'],test_size=0.3)
deep_m = Sequential()

deep_m.add(Dense(13, input_dim = 13,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 10,activation ='relu'))

deep_m.add(Dense(units = 3,activation ='softmax'))

deep_m.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
deep_hist =  deep_m.fit(X_train,y_train,epochs = 150, validation_data = (X_test,y_test))


plt.plot(deep_hist.history['accuracy'])

plt.plot(deep_hist.history['val_accuracy'])

plt.title(['Accuracy'])

plt.legend(['train','test'])

plt.show()
plt.plot(deep_hist.history['loss'])

plt.plot(deep_hist.history['val_loss'])

plt.title(['Loss'])

plt.legend(['train','test'])

plt.show()
dt2 =  tree.DecisionTreeClassifier(random_state=1, max_depth = 2)

dt2.fit(X_train,y_train)

dt2_score_train = dt2.score(X_train,y_train)

print('Training score:',dt2_score_train)

dt2_score_test = dt2.score(X_test,y_test)

print('Test score:',dt2_score_test)
#Decision tree with depth = 3

dt3 =  tree.DecisionTreeClassifier(random_state=1, max_depth = 3)

dt3.fit(X_train,y_train)

dt3_score_train = dt3.score(X_train,y_train)

print('Training score:',dt3_score_train)

dt3_score_test = dt3.score(X_test,y_test)

print('Test score:',dt3_score_test)
#Decision tree with depth = 4

dt4 =  tree.DecisionTreeClassifier(random_state=1, max_depth = 4)

dt4.fit(X_train,y_train)

dt4_score_train = dt4.score(X_train,y_train)

print('Training score:',dt4_score_train)

dt4_score_test = dt4.score(X_test,y_test)

print('Test score:',dt4_score_test)
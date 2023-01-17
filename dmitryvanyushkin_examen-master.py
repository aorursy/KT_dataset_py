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
import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn import preprocessing, model_selection, metrics

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as pltMath

from sklearn.manifold import TSNE

import seaborn as sns

import time

color = sns.color_palette()

%matplotlib inline

import matplotlib.patheffects as PathEffects

from sklearn.cluster import KMeans



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls





pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 9999
def reclassifyDataSet(df):

    for i,row in df.iterrows():

            if row['price'] < 6000000:

                df.at[i, 'price'] = 0

            elif row['price'] < 12000000:

                df.at[i, 'price'] = 1

            elif row['price'] < 18000000:

                df.at[i, 'price'] = 2

            elif row['price'] < 24000000:

                df.at[i, 'price'] = 3

            else:

                df.at[i, 'price'] = 4

            i += 1

    print(df)

    return df

def MyScatter(x, colors):

    # choose a color palette with seaborn.

    num_classes = len(np.unique(colors))

    palette = np.array(sns.color_palette("hls", num_classes))



    # create a scatter plot.

    f = pltMath.figure(figsize=(8, 8))

    ax = pltMath.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])

    pltMath.xlim(-25, 25)

    pltMath.ylim(-25, 25)

    ax.axis('off')

    ax.axis('tight')



    # add the labels for each digit corresponding to the label

    txts = []



    for i in range(num_classes):



        # Position of each label at median of data points.



        xtext, ytext = np.median(x[colors == i, :], axis=0)

        txt = ax.text(xtext, ytext, str(i), fontsize=24)

        txt.set_path_effects([

            PathEffects.Stroke(linewidth=5, foreground="w"),

            PathEffects.Normal()])

        txts.append(txt)



    return f, ax, sc, txts





def buildTSNE(dataSet):



#     (nsamples, nx, ny, nz) = dataSet.shape

#     D2 = dataSet.reshape((nsamples, nx * ny * nz))



    # T-SNE Implementation



    t0 = time.time()

    X_reduced_tsne = TSNE(n_components=2,

                          random_state=42).fit_transform(dataSet)

    t1 = time.time()

    print("T-SNE took {:.2} s".format(t1 - t0))

    

    MyScatter(X_reduced_tsne, dataSet['price'])



    return X_reduced_tsne

    

def plotTSNE(X_tsne):

    kmeans_tsne = KMeans(n_clusters=5).fit(X_tsne)

    

    pltMath.figure(figsize=(12, 5))

    cmap = pltMath.get_cmap('nipy_spectral')



    pltMath.subplot(1,2,1)

    pltMath.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cmap(kmeans_tsne.labels_ / 5))

    pltMath.title('TSNE')

train = pd.read_csv("/kaggle/input/realty-prices-minor2020/reality_data_train.csv")

test = pd.read_csv("/kaggle/input/realty-prices-minor2020/reality_data_test.csv")



print("Train rows and columns : ", train.shape)

print("Test rows and columns : ", test.shape)
train.head()
plt.figure(figsize=(8,6))

plt.scatter(range(train.shape[0]), train['price'].values)

plt.xlabel('index', fontsize=12)

plt.ylabel('Price', fontsize=12)

plt.title("Price Distribution", fontsize=14)

plt.show()
dtype_df = train.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
for column in train.columns:

    if train[column].dtype=='object':

        label = LabelEncoder()

        train[column] = label.fit_transform(train[column])

        print(dict(enumerate(label.classes_)))

        

train.info()
Y = train.price.values

X = train.drop(["id", "price"], axis=1)



train_x, val_x, train_y, val_y = train_test_split(X,Y,random_state=42)



for column in test.columns:

    if test[column].dtype=='object':

        label = LabelEncoder()

        test[column] = label.fit_transform(test[column])

        print(dict(enumerate(label.classes_)))

        

test.info()
def plotBarChart(values,description):

    pltMath.plot(values)

    pltMath.title(description)

    pltMath.ylabel('accuracy')

    pltMath.xlabel('estimator')

    pltMath.legend(['mae'], loc='upper left')

    pltMath.show()
xgb_model = XGBRegressor(

    objective = 'reg:squarederror',

    learning_rate = 0.01,

    max_depth = 55,

    n_estimators = 1000)



print('start training')

xgb_model.fit(train_x, train_y, eval_set=[(train_x,train_y),(val_x,val_y)],eval_metric='mae', verbose=False)

print('finish training')







mae = xgb_model.evals_result()["validation_0"]["mae"]

plotBarChart(mae,"mae values")

mae2 = xgb_model.evals_result()["validation_1"]["mae"]

plotBarChart(mae2,"mae values")



xgb_prediction = xgb_model.predict(val_x).round(0)

mae_xgb = mean_absolute_error(val_y, xgb_prediction)

print(mae_xgb)

print('start training')

xgb_model.fit(X, Y)

print('finish training')



print(xgb_model.evals_result().keys())

mae = xgb_model.evals_result()["validation_0"]["mae"]

plotBarChart(mae,"mae values")

mae2 = xgb_model.evals_result()["validation_1"]["mae"]

plotBarChart(mae2,"mae values")

forest_model = RandomForestRegressor(n_estimators=30, 

                                 criterion='mae',

                                 max_features=3,

                                 random_state=1,

                                 max_depth=55,

                                 min_samples_split=5

                                 )



print('start training')

forest_model.fit(train_x, train_y)

print('finish training')
forest_prediction = forest_model.predict(val_x).round(0)

mae_forest = mean_absolute_error(val_y, forest_prediction)

print(mae_forest)

print('start training')

forest_model.fit(X, Y)

print('finish training')
def plotBarChart(true,predict):

    pltMath.figure(figsize=(20,5))

    pltMath.bar([1,2],[true,predict])

    pltMath.ylabel("mae", fontsize=16)

    pltMath.title("Histogram mae")

    pltMath.show()
plotBarChart(mae_xgb, mae_forest)
x_tsne = buildTSNE(reclassifyDataSet(train))
plotTSNE(x_tsne)
test_x = test.drop(["id"], axis=1)

xgb_prediction_test = xgb_model.predict(test_x).round(0)

forest_prediction_test = forest_model.predict(test_x).round(0)

rounded_prediction_forest = np.round(forest_prediction_test,0)

rounded_prediction_xgb = np.round(xgb_prediction_test,0)



submission_xgb = pd.DataFrame()

submission_xgb['Id'] = test.id.values

submission_xgb['Price'] = rounded_prediction_xgb

submission_xgb.to_csv('submission_xgb.csv',index=False)



print("XGB submission is ready")



submission_forest = pd.DataFrame()

submission_forest['Id'] = test.id.values

submission_forest['Price'] = rounded_prediction_forest

submission_forest.to_csv('submission_forest.csv',index=False)



print("Forest submission is ready")
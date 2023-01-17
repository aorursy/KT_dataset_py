# In a regression problem, we aim to predict the output of a continuous value, like a price or a probability. 
# Contrast this with a classification problem, where we aim to predict a discrete label (for example, where a 
# picture contains an apple or an orange).
# 
# This notebook uses the classic Auto MPG Dataset and builds a model to predict the fuel efficiency of late-1970s 
# and early 1980s automobiles. To do this, we'll provide the model with a description of many models from that time period. 
# 
# This description includes attributes like: cylinders, displacement, horsepower, and weight.
# 
# This example uses the tf.keras API, and mentioned in TensorFlow tutorial page. Here I tried to use the same
# dataset and enhanced whenever I could.
from __future__ import absolute_import, division, print_function

import pathlib
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ML Libraries: 
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.utils import shuffle

import plotly.figure_factory as ff
import plotly.offline as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools 
py.init_notebook_mode (connected = True)

import cufflinks as cf
cf.go_offline()
import platform

print(tf.__version__)
print(platform.python_version())
# The Auto MPG dataset
# --------------------
# The dataset is available from the UCI Machine Learning Repository.
# 
# Get the data
# First download the dataset.

#dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#dataset_path
# Import it using pandas

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
# raw_dataset = pd.read_csv("../input/auto-mpg.csv", names=column_names,
#                       na_values = "?", comment='\t',
#                       sep=" ", skipinitialspace=True)

raw_dataset = pd.read_csv("../input/auto-mpg.csv")

dataset = raw_dataset.copy()
dataset.tail()
# Clean the data
# --------------
# The dataset contains a few unknown values.

dataset.isna().sum()
# To keep this initial tutorial simple drop those rows.

dataset = dataset.dropna()

# The "Origin" column is really categorical, not numeric. So convert that to a one-hot:

origin = dataset.pop('origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()
# Split the data into train and test
# ----------------------------------
# 
# Now split the data into a train and a test set.
# We will use the test set in the final evaluation of out model.

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Let's copy training dataset into working dataset for data visualization
working_dataset = train_dataset.copy();

train_dataset.pop('car name')
test_dataset.pop('car name')
# Breif information about the Data Sets
train_dataset.info()
print("________________________________")
test_dataset.info()

working_dataset['Continent'] = np.where(working_dataset['USA'] == 1.0, 'USA',
                                np.where(working_dataset['Japan'] == 1.0, 'Japan', 'Europe'))
working_dataset.tail()
usa = working_dataset[working_dataset["USA"]==1.0]
europe = working_dataset[working_dataset["Europe"]==1.0]
japan = working_dataset[working_dataset["Japan"]==1.0]

usa_car_count = working_dataset["USA"].value_counts()
europe_car_count = working_dataset["Europe"].value_counts()
japan_car_count = working_dataset["Japan"].value_counts()

working_dataset = shuffle(working_dataset)
train_continet_car_count = working_dataset["Continent"].value_counts()

test_dataset = shuffle(test_dataset)
test_continet_car_count = test_dataset["mpg"].count()

print(usa_car_count)
print(europe_car_count)
print(japan_car_count)

print("Training Set Car Count ", train_continet_car_count)
print("Test Set Car Count", test_continet_car_count)
#1. Pie Chart for car count
colors = ['aqua', 'pink', 'teal']

trace_train= go.Pie(labels = train_continet_car_count.index,
              values = train_continet_car_count.values, marker=dict(colors=colors))

layout = go.Layout(title = "Training Data :: Car Distribution")
data = [trace_train]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# 1 Boxplot 

usa = working_dataset[working_dataset["Continent"]=="USA"]
europe = working_dataset[working_dataset["Continent"]=="Europe"]
japan = working_dataset[working_dataset["Continent"]=="Japan"]

trace = go.Box(y = usa["mpg"],fillcolor="aqua", name= "USA" )
trace1 = go.Box(y = europe["mpg"], fillcolor="pink", name= "Europe" )
trace2 = go.Box(y = japan["mpg"], fillcolor="teal", name= "Japan" )

layout = go.Layout(title="Fuel Efficiency Distribution w.r.t Continent :: [Box Plot]", 
                   yaxis=dict(title="Fuel Efficiency (mile per gallon)"), 
                   xaxis= dict(title="Continent (USA / Europe  /Japan)"))

data=[trace, trace1, trace2]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)

# 2 Violin Plot 
trace1 = go.Violin( y = usa["mpg"], fillcolor="aqua", name="USA")
trace2 = go.Violin( y = europe["mpg"],fillcolor="pink", name="Europe")
trace3 = go.Violin( y = japan["mpg"],fillcolor="teal", name="Japan")

layout = go.Layout(title="Fuel Efficiency Distribution w.r.t Continent :: [Violin Plot]", 
                   yaxis=dict(title="Fuel Efficiency (mile per gallon)"), 
                   xaxis= dict(title="Continent (USA / Europe  /Japan)"))

data=[trace1, trace2, trace3]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)
# Car vs Mileage 
#age_count = df["Age"].dropna().value_counts()
top_fifteen_train_dataset = working_dataset.nlargest(15, 'mpg')
train_car_names = top_fifteen_train_dataset["car name"].dropna()
train_car_mileage = top_fifteen_train_dataset["mpg"].dropna()
print(train_car_mileage)
trace = go.Bar(x = train_car_names,
              y = train_car_mileage, 
              marker = dict(color = train_dataset["mpg"],
                           colorscale = "Jet", 
                           showscale = True))
layout = go.Layout(title = "Car Mileage Distribution :: Most Fuel Efficient Cars (Top 15)", 
                  yaxis = dict(title = "Fuel Efficiency"))
data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
# Displacement vs Mileage [Scatter Plot]
train_displacements = working_dataset["displacement"].dropna()
trace = go.Scatter(x = train_displacements, y = working_dataset["mpg"].dropna(), 
              mode = 'markers',                 
              marker = dict(color = train_dataset["mpg"],
                           colorscale = "Jet", 
                           showscale = True))
layout = go.Layout(title = "Diasplacement vs Mileage Distribution :: [Scatter Plot]", 
                  xaxis = dict(title = 'Car Displacement'), 
                  yaxis = dict(title = 'Car Fuel Efficiency (mile per gallon)'))
data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
# Inspect the data
# ----------------

# Have a quick look at the joint distribution of a few pairs of columns from the training set.

sns.pairplot(working_dataset[["mpg", "cylinders", "displacement", "weight"]], diag_kind="kde")
# Also look at the overall statistics:

train_stats = train_dataset.describe()
train_stats.pop("mpg")
train_stats = train_stats.transpose()
train_stats
# Training Set Car Count
trace = go.Bar(x = ['USA', 'Europe', 'Japan'],
              y = [usa['mpg'].count(),europe['mpg'].count(), japan['mpg'].count()], 
              marker = dict(color = [111, 10, 225, 175],
                           colorscale = "Viridis", 
                           showscale = True))
layout = go.Layout(title = "Training Set Car Count ", 
                  yaxis = dict(title = "Car Count"))
data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
# Plotting HeatMap
# print(working_dataset["Continent"].unique())
trace = go.Heatmap(z=working_dataset[["displacement", "mpg", "cylinders", "horsepower", "model year"]].dropna().values,
                   x=['Displacement', 'MPG', 'Cylinders', 'Horsepower', 'Model Year'],
#                    y=['USA', 'Europe', 'Japan'],
                   colorscale = 'Viridis')
#                    y=["USA", "Europe", "Japan"])
layout = go.Layout(title = "Displacement, MPG, Cylinders, Horsepower, Model Year Distribution",
                  yaxis = dict(title = "Car Record Count"))
data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)
#Data preparation for plotting various charts
df = working_dataset.copy()
df['Origin'] = np.where(df['Continent'] == 'USA', 1,
                                np.where(df['Continent'] == 'Europe', 2, 3))
df.tail()    
#Ploting Stacked Bar Chart
df.iplot(kind='bar', barmode='stack', filename='cufflinks/grouped-bar-chart')

#Plotting Bubble Chart
df.iplot(kind='bubble', x='displacement', y='mpg', size='Origin', text='Continent',
             xTitle='Displacement', yTitle='Fuel Efficiency (mpg)', colors='rgb(255,0,0)', 
             filename='simple-bubble-chart')
# Separate the target value, or "label", from the features. This label is the value that you 
# will train the model to predict.
train_labels = train_dataset.pop('mpg')
test_labels = test_dataset.pop('mpg')
train_dataset.tail()
# Look again at the train_stats block above and note how different the ranges of each feature are.
# It is good practice to normalize features that use different scales and ranges. Although the model 
# might converge without feature normalization, it makes training more difficult, and it makes the 
# resulting model dependent on the choice of units used in the input.

# This normalized data is what we will use to train the model.

train_dataset.dropna()
test_dataset.dropna()

train_dataset.pop('horsepower')
test_dataset.pop('horsepower')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
# Let's build our model. Here, we'll use a TensorFlow Keras Sequential model with two densely 
# connected hidden layers, and an output layer that returns a single, continuous value. The model 
# building steps are wrapped in a function, build_model, since we'll create a second model, later on.

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
model = build_model()

# Use the .summary method to print a simple description of the model

model.summary()
# Now try out the model. Take a batch of 10 exampes from the training data and call 
# model.predict on it.

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

# Train the model
# ---------------

# The model is trained for 1000 epochs, and record the training and validation accuracy in the history object.

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
# Visualize the model's training progress using the stats stored in the history object.

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,20])

plot_history(history)
# This graph shows little improvement, or even degradation in the validation error after a few 
# hundred epochs. Let's update the model.fit method to automatically stop training when the 
# validation score doesn't improve. We'll use a callback that tests a training condition for 
# every epoch. If a set amount of epochs elapses without showing improvement, then automatically 
# stop the training.

# You can learn more about this callback here.

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
# The graph shows that on the validation set, the average error usually around +/- 2 MPG. Is this good? 
# We'll leave that decision up to you.

# Let's see how did the model performs on the test set, which we did not use when training the model:

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
# Make predictions
# ----------------

# Finally, predict MPG values using data in the testing set:

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
# This notebook introduced a few techniques to handle a regression problem.

#     1. Mean Squared Error (MSE) is a common loss function used for regression problems 
#        (different than classification problems).
#     2. Similarly, evaluation metrics used for regression differ from classification. A common 
#        regression metric is Mean Absolute Error (MAE).
#     3. When input data features have values with different ranges, each feature should be scaled independently.
#     4. If there is not much training data, prefer a small network with few hidden layers to avoid overfitting.
#     5. Early stopping is a useful technique to prevent overfitting.
# Features Scaling
columns = normed_train_data.columns
column_test = normed_test_data.columns
y_train = train_labels.copy()

lab_enc = preprocessing.LabelEncoder()
y_train_encoded = lab_enc.fit_transform(y_train)

scaler = preprocessing.Normalizer()
X_train = scaler.fit_transform(normed_train_data)
X_train = pd.DataFrame(X_train, columns=columns)

X_test = scaler.transform(normed_test_data)
X_test = pd.DataFrame(X_test, columns=column_test)
# Scikit learn Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train_encoded)

pred_log = log.predict(X_test)

log.score(X_train, y_train_encoded)
logistic_score = round(log.score(X_train, y_train_encoded)*100,2)
logistic_score
# Scikit learn RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train_encoded)
pred_random = rf.predict(X_test)

rf.score(X_train, y_train_encoded)

random_score = round(rf.score(X_train, y_train_encoded)*100,2)
random_score
# Scikit learn DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train_encoded)

pred_tree = tree.predict(X_test)

tree.score(X_train, y_train_encoded)
tree_score = round(tree.score(X_train, y_train_encoded)*100,2)
tree_score
# Scikit learn KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train_encoded)
pred_knn = knn.predict(X_test)

knn.score(X_train, y_train_encoded)
knn_score = round(knn.score(X_train, y_train_encoded)*100,2)
knn_score
# Scikit learn GaussianNB (Gaussian Naive Bayes)

gaus = GaussianNB()
gaus.fit(X_train, y_train_encoded)
pred_gaus = gaus.predict(X_test)

gaus.score(X_train, y_train_encoded)
gaus_score = round(gaus.score(X_train, y_train_encoded)*100,2)
gaus_score
# Scikit learn Perceptron

per = Perceptron(max_iter=5)
per.fit(X_train, y_train_encoded)

perd_per = per.predict(X_test)

per.score(X_train, y_train_encoded)
perceptron_score = round(per.score(X_train, y_train_encoded) * 100, 2)
perceptron_score
# Scikit learn LinearSVC (Linear Support Vector Machine)

svc = LinearSVC()
svc.fit(X_train, y_train_encoded)

pred_svc = svc.predict(X_test)

svc.score(X_train, y_train_encoded)
svc_score = round(svc.score(X_train, y_train_encoded) * 100, 2)
svc_score
df_score = pd.DataFrame({"Models": ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Decision Tree'], 
                       "Score": [svc_score, knn_score, logistic_score, random_score, gaus_score, 
                                 perceptron_score, tree_score]})
df_score.sort_values(by= "Score", ascending=False)
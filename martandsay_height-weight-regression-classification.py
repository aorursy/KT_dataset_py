# UPVOTE IF YOU LIKE THE TUTORIAL



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# By the end of this, you will be able to draw basic histograms, KDE and scatter plot and will get to knwo what is univariate and bivariate distributions.

# like us on fb: https://www.facebook.com/codemakerz



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
print(os.listdir("../input"))

df= pd.read_csv('../input/weight-height.csv')
# Lets see what we got in df. head prints by default 5 rows.

df.head()

df.isnull().sum() # As we can see, there is no null values so we dont have to take care of missing values, which is abig relief.
# Lets draw a basic histogram. 

df.Height.plot(kind="hist", title='Univariate: Height Histogram', color='c');
# univariate distribution

#Above histogram same we can do with matplotlib also. Matplotlib allows you to write more flexible code

# title again means you plot name.

# xlabel means label on x-axis and ylabel for y-axis

# bins will be explained in the next plot

plt.hist(x=df.Height, color='c')

plt.title("Univariate: Height Histogram")

plt.xlabel("Height")

plt.ylabel("Total Counts")

plt.plot();
# Lets draw a basic histogram. 

df.Weight.plot(kind="hist", title='Univariate: Weight Histogram', color='c');
# Weight histogram using matplotlib 

plt.hist(x=df.Weight, color='c')

plt.title("Univariate: Weight Histogram")

plt.xlabel("Weight")

plt.ylabel("Total Counts")

plt.plot();
# So with the above histograms you can estimate the number of people having a particular value of height and weight.

# e.g. number of people having height of 100 cms is around 250.

# Lets say you want increas the bars(thos cyan colored bars). We can increase them by using bins property.

plt.hist(x=df.Weight, color='c', bins=20)

plt.title("Univariate: Weight Histogram")

plt.xlabel("Height")

plt.ylabel("Total Counts")

plt.plot();
# So we are done with the histogram try to find out more about histogram.

#lets plot a kde. KDE is more like smoother curve. It is very simple just change the kind to kde
# KDE distribution for Height

df.Height.plot(kind="kde", title='Univariate: Height KDE', color='c');
# KDE distribution for Weight

df.Weight.plot(kind="kde", title='Univariate: Weight KDE', color='c');
# So you can imagine KDE as the outline of histogram.

# KDE is more smoother and it gives you a bell shape curve.
# Bivariate: Lets plot bivariate using scatter plot

# So here we are plotting a  bivariate plot type called scatter plot between height and weight.

# x = data to show in x-axis(numeric)

# y = data to show in y-axis(numeric)

# color: color of the dots

# title: name for your plot

df.plot.scatter(x="Height", y="Weight", color='c', title='Height Vs Weight');
# lets do same with matplotlib

plt.scatter(x=df["Height"], y=df["Weight"], color='c')

plt.title("Bivariate: Height Vs Weight Using Matplotplib")

plt.xlabel("Height")

plt.ylabel("Weight")

plt.plot(); # this means now plot the visualisation with all the settings.
# So I think it would be helpfull for you. If you like it, please follow us on facebook: https://www.facebook.com/codemakerz
df.head()
X = df.iloc[:, 1:2].values

y = df.iloc[:, 2:3].values
# First lets split our data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=31)
# Lets create a k-fold structure to train our model in better way.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model_fit = regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
y_predict
plt.scatter(X_test, y_test)

plt.plot( X_train, regressor.predict(X_train), color='r')

plt.xlabel("Height")

plt.ylabel("Weight")

plt.title("Heigth Vs Weight Prediction")

plt.show()
# So you can see our regression line is finely fitting our data points.

# blue points are actual values & points on red line are predicted values.
# To find accuracy of model we have many metries like 

# R-square, mean-squared-error(MSE), root-mean-squared-error(RMSE)
from sklearn.metrics import r2_score
print(f"Model Accuracy is: {regressor.score(X_test, y_test)}")
# Cool we got an accuracy of 85%, which is not bad.
r2_score(y_test, y_predict)
# R-squared Close to 1 is good.
# SO finally you just created your simple linear regression model.
# Previously we saw how to predict Weigth by Height, which is simple because:

# 1. Both were numeric values

# 2. There was only one independent variable or feature(Height).

# 3. Previously we were finding continues value like Weight but now we are finding discrete values like Male or Female. So

# here we will use classification.

# Now we wil talk about KNN classification. Here we will use two features(Heigth & Weigth) to predict Gender.

# Lets Split data
df.head()
X_ml = df.iloc[:, 1:3].values

y_ml = df.iloc[:, 0:1].values
X_ml.shape
# Can you see any problem here? We can see our target variable is in string form, ML model only understands numeric values.

# To make our model work, we need to encode it to numeric values. We will use LabelEncoder class for it.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_ml = encoder.fit_transform(y_ml)
y_ml # so we can see now our string values are encoded. Male as 1 & Female as 0
y_ml.shape
# So now we can perform KNN classification. Before that we need to split train test set.
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_ml, y_ml, test_size=0.3, random_state=31)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3) #n_neighbors is the number of neighbours
knn.fit(X_train_ml, y_train_ml)
y_predict_ml = knn.predict(X_test_ml)
y_predict_ml # So you can see now our model is classifying the gender
# Lets plot confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_ml, y_predict_ml)
# So our confusion matrix shows we predicted 1315+1390=2705 values correct and 138+157=295 incorrect. Not bad but can we do it better?
# Lets change the n_neighbors values in KNN classifier and see
knn = KNeighborsClassifier(n_neighbors=5) #n_neighbors is the number of neighbours

knn.fit(X_train_ml, y_train_ml)

y_predict_ml = knn.predict(X_test_ml)

cm = confusion_matrix(y_test_ml, y_predict_ml)

cm
print("accuracy:", knn.score(X_test_ml, y_test_ml)) # accuracy of 90% which is a very good number. But let see can we increase this
# correct = 2723 & incorrect = 277. So We can see it depend on the number of the knn neighbours. You have to check 

# multiple value to find the best result. In our case i think 5 seems the good values.
from sklearn.model_selection import GridSearchCV

# below is the params dictionary object. Here we can add the parameters which we can experiment with. so now our model will get evaluated with all the possible 

# combination of below params and let you the best score and param combination.

params = {

    "n_neighbors": [5, 10, 20],

    'leaf_size' : [30, 40, 50],

    'algorithm': ["ball_tree", "kd_tree", "brute"],

    'p': [1, 2]

}

gs = GridSearchCV(estimator=knn, cv=10, param_grid=params )
gsresult = gs.fit(X_train_ml, y_train_ml ) # it may take a while as grid search cv will run your mode with all the possible combinations of given params.
print(gsresult.best_score_) # So you can see we increased accuracy with more than 1% for training data.

print(gsresult.best_params_) # So this is the best possible combination for our model. Lets try with that.
knn_best_fit = KNeighborsClassifier(algorithm = "ball_tree", leaf_size= 30, n_neighbors = 20, p=1)

knn_best_fit.fit(X_train_ml, y_train_ml)
y_predict_best = knn_best_fit.predict(X_test_ml)
cm_best = confusion_matrix(y_test_ml, y_predict_best)

cm_best
print("accuracy:", knn.score(X_test_ml, y_predict_best)) # accuracy of our new model is 97% which means 7% more than our old model for unseen data or test data, which is amazing.

# So here we will stop with this configuration.
# As our data is very small but still GridSearchCV performed amazing. But remember if you have big data it can give you even better results so always try to configure your model with it.

# For larger dataset it can take a long time to train model using gridsearchcv.
# Though we can see our confusion matrix but it is always good to plot it in a user friendly way.

sns.heatmap(cm, annot=True, fmt='g')
sns.heatmap(cm_best, annot=True, fmt='g')
import keras

from keras.layers import Dense, Dropout

from keras.models import Sequential
# input_dim is no of inputs in ANN which is equal to no of columns in X_train 

# Dense() adding a layer

# Dropout() is to avoid overfit. It drops given percentage of neurons in the next layer.

clf = Sequential([

    Dense(input_dim=2, units=20, activation='relu'),

    #Dropout(0.2),

    Dense(units=20, activation='relu'),

    Dense(units=2, activation='sigmoid') # final layer is output unit. Here units should be equal to the number of class in you output.

])

clf.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
clf.summary() # you can check your model details
clf.fit(X_train_ml, y_train_ml, batch_size=10, epochs=50)
clf.evaluate(X_train_ml, y_train_ml) # So accuracy of 91% in training data
clf.evaluate(X_test_ml, y_test_ml) # So accuracy of 91% in test data
# So we can see neural network did not perform well on data. It may be because we have very less data. But you may try adding more Dense layers.

# our main aim was to show the implementation of neural network.
# You can predict like

y_predict_nn = clf.predict_classes(X_test_ml)
cm_nn = confusion_matrix(y_test_ml, y_predict_nn)

cm_nn
# So we have a very bad model. So finally after evaluation KNN and ANN we found that in our case KNN with optimized values is best model for us.

# AGAIN... IT ISNOT NECESSARY THAT NEURAL NETWORK WILL ALWAYS PERFORM GOOD FOR YOUR DATA.
# So now finally you just finished your simple classifier.

# I hope you understood the process.

# Like us on facebook fo more tutorials :  https://www.facebook.com/codemakerz
# Logistic Regression

# Logistic Regression is a sibling of Linear Regression, but unlike predicting/regressing real values, it predicts the probability of an even happening.



# Once we get the probability, we can decide a threshold, depending on the use case and then we can classify the point as one of the classes.



# For this example, our Logistic Regression will predict the probabilities (given the attributes of a new data), of the data point of being one of the

# three species. We can decide the threshold of that probability (normally 0.5, but it will totally depend on the use case) and use it give the final 

# class label.



# So how do we get tese probabilities?

# We use a squashing function, also called logistic function or sigmoid function to predict the probabilities.



# To use the sigmoid function, we describe the log of Odds as linear function of attribute values.

# Odds are described as ratio of probability if event happening and probability eof even not happening.

# Log (p/(1-p)) = ax + b, this is the proble formulation

# Logistic function actually limits the output b/w 0 to 1. 



# Lets jump into the code.
# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
# In this notebook, we will not explore the data as it is already been done here

# https://www.kaggle.com/gadaadhaarigeek/another-eda-on-iris-dataset

# Here, we will implement Logistic regression and other topics associated with it.
# Processing and preparing of data 

import pandas as pd

import numpy as np



# To show the floats rather than

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})



# Visualization of data 

import matplotlib.pyplot as plt

import seaborn as sns



# Supress warnings 

import warnings 

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/iris/Iris.csv")

data.head()
# We have 150 data points, 50 each for each classes.

# Ther are three classes or species of Iris flower - setosa, virgininca and versicolor

# we will drop the id column as it is just an identifier of the data points.



data.drop(labels=["Id"], axis=1, inplace=True)
data.head()
# Unlike many other ML algorithms, logistic regression doesn't require scaling or standardization of attributes.

# But we will encode the values for Species.
# Let's look at the values of species 

data["Species"].value_counts()
# As mentioned above, we have three different species, that too 50 each.

# Let'c encode the Species 
# LabelEncoder for encoding the Species column

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data["Species"] = le.fit_transform(data["Species"])
data["Species"].value_counts().sort_index()
# We can see the inverse mapping also

le.inverse_transform([0, 1, 2])
# So, 0-setosa, 1-versicolor and 2-verginica
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values

y = data["Species"]
# Let's split the data first 

# stratify attribute will make sure that equal classes are distributed in training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty="l2", C=1.0, multi_class="ovr")

lr.fit(X_train, y_train)
# Prediction of values

y_pred = lr.predict(X_test)
# Let's see the confusion matrix

# We will plot the confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_true=y_test, y_pred=y_pred)
# Here the diaogonal values show how many points are correctly classified and other values are the number of points incorrectly classified

# In a two class setting, these values will be TP, TN, FP, FN
# Accuracy Score

from sklearn.metrics import accuracy_score

round(accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True)*100, 2)
# Accuracy score is 95% which is good
# Let's have a classification report

from sklearn.metrics import classification_report

cr = classification_report(y_true=y_test, y_pred=y_pred, target_names = le.inverse_transform([0, 1, 2]))

print(cr)
# So we have different metrics in classification report. I willl give the definition of each below with respect to setosa class



# Precision - Out of points classified as setosa, what fraction is actually setosa. In our case its 1, so 100%.

# In a two class setting it will be TP/(TP+FP)



# Recall - Out of total versicolor points, what fraction is correctly classified as versicolor. In our case, its .93, so not all the setosa's are correctly 

# classified

# In a two classsetting it is TP/(TP + FN)



# NOte: Depending upon the use case, you would any of the above to be highest. ACCURACY IS NOT ALWAYS THE GOOD CHOICE OF PERFORMANCE.



# F1-Score - harmonic mean of precision and recall.

# The F-score has been widely used in the natural language processing literature, such as the evaluation of named entity recognition and word segmentation.
# We can actually get the log-loss, which is something logistic regression builts upon.

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import log_loss



# Predict the probabilities rather than classes for calculating the logg loss

y_pred_prob = lr.predict_proba(X_test)



# One hot enoding the class labels in y_test to calculate the log loss

ohe = OneHotEncoder(sparse=False)

y_test_ohe = ohe.fit_transform(y_test.values.reshape(-1, 1))





log_loss(y_true=y_test_ohe, y_pred=y_pred_prob)
# For a multiclass (more than two classes), this value of log loss is actually good
# Let's do soem plotting, but for that we need to make the 4 dimensions of our data to 2d for visual purpose

# We will use PCA for that

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train_transformed = pca.fit_transform(X_train)
# We can see how much variance is explained by two components 

pca.explained_variance_ratio_
# Its more than 95%, which is good
# Plot decision trees for training data

from mlxtend.plotting import plot_decision_regions

lr = LogisticRegression()

lr.fit(X_train_transformed, y_train)

plot_decision_regions(X_train_transformed, y_train.values, clf=lr, legend=3)
# For test values 

X_test_transformed = pca.transform(X_test)

y_pred = lr.predict(X_test_transformed)

plot_decision_regions(pca.transform(X_test), y_pred, clf=lr, legend=3)
# The regions which are plotted show decision boundary for the classifier whic his actually good.
# As of now, I will stop here.

# I will add grid search and hyperparameter tuning later on in this kernel.
# DO LET ME KNOW WHAT YOU THINK ABOUT THIS KERNEL

# THANKS FOR READING THIS FAR
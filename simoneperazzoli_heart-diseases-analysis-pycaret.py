!pip3 install bubbly

#!pip3 install pandas-profiling

#!pip3 install shap

!pip3 install pycaret
# for basic operations

import numpy as np

import pandas as pd

#import dtale

# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns

# for advanced visualizations 

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from bubbly.bubbly import bubbleplot

# for providing path

import os

# for model 

import pycaret



%matplotlib inline
# reading the data

dataset = pd.read_csv('../input/heart-disease-uci/heart.csv')

data = dataset.copy()

# getting the shape

data.shape
# getting info about data

data.info()
# reading the head of the data

data.head()
# describing the data

data.describe()
# making a heat map

plt.rcParams['figure.figsize'] = (20, 15)

plt.style.use('ggplot')

sns.heatmap(data.corr(), annot = True)

plt.title('Heatmap for the Dataset', fontsize = 20)

plt.show()
# checking the distribution of age among the patients

from scipy.stats import norm

sns.distplot(data['age'], fit=norm, kde=False)

plt.title('Distribution of Age', fontsize = 10)

plt.show()
# Checking Target

sns.countplot(data['target'])

plt.xlabel(" Target")

plt.ylabel("Count")

plt.show()
# plotting a donut chart for visualizing each of the recruitment channel's share

size = data['sex'].value_counts()

colors = ['lightblue', 'lightgreen']

labels = "Male", "Female"

explode = [0, 0.01]

my_circle = plt.Circle((0, 0), 0.7, color = 'white')

plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')

plt.title('Distribution of Gender', fontsize = 15)

p = plt.gcf()

p.gca().add_artist(my_circle)

plt.legend()

plt.show()
# Checking chest type

sns.countplot(data['cp'])

plt.xlabel(" Chest type")

plt.ylabel("Count")

plt.show()
# tresbps vs target

sns.boxplot(data['target'], data['trestbps'], palette = 'viridis')

plt.title('Relation of tresbps with target', fontsize = 20)

plt.show()
# cholestrol vs target

sns.violinplot(data['target'], data['chol'], palette = 'colorblind')

plt.title('Relation of Cholestrol with Target', fontsize = 20, fontweight = 30)

plt.show()
# Resting electrocardiographic measurement vs target

  

dat = pd.crosstab(data['target'], data['restecg']) 

dat.div(dat.sum(1).astype(float), axis = 0).plot(kind = 'bar', 

                                                 stacked = False, 

                                                 color = plt.cm.rainbow(np.linspace(0, 1, 4)))

plt.title('Relation of ECG measurement with Target', fontsize = 20, fontweight = 30)

plt.show()
# slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

# checking the relation between slope and target



plt.rcParams['figure.figsize'] = (15, 9)

sns.boxenplot(data['target'], data['slope'], palette = 'copper')

plt.title('Relation between Peak Exercise and Target', fontsize = 20, fontweight = 30)

plt.show()
#ca: The number of major vessels (0-3)



sns.boxenplot(data['target'], data['ca'], palette = 'Reds')

plt.title('Relation between no. of major vessels and target', fontsize = 20, fontweight = 30)

plt.show()
# relation between age and target



plt.rcParams['figure.figsize'] = (15, 9)

sns.swarmplot(data['target'], data['age'], palette = 'winter', size = 10)

plt.title('Relation of Age and target', fontsize = 20, fontweight = 30)

plt.show()
# relation between sex and target

sns.boxenplot(data['target'], data['sex'], palette = 'Set3')

plt.title('Relation of Sex and target', fontsize = 20, fontweight = 30)

plt.show()
# checking the relation between 

#thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)



sns.boxenplot(data['target'], data['thal'], palette = 'magma')

plt.title('Relation between Target and Blood disorder-Thalessemia', fontsize = 20, fontweight = 30)

plt.show()
# target vs chol and hue = thalach



plt.scatter(x = data['target'], y = data['chol'], s = data['thalach']*100, color = 'yellow')

plt.title('Relation of target with cholestrol and thalessemia', fontsize = 20, fontweight = 30)

plt.show()
# multi-variate analysis

sns.boxplot(x = data['target'], y = data['trestbps'], hue = data['sex'], palette = 'rainbow')

plt.title('Checking relation of tresbps with genders to target', fontsize = 20, fontweight = 30)

plt.show()
#check the shape of data

dataset.shape
dataset.head()
data = dataset.sample(frac=0.95, random_state=786)

data_unseen = dataset.drop(data.index).reset_index(drop=True)

data.reset_index(drop=True, inplace=True)



print('Data for Modeling: ' + str(data.shape))

print('Unseen Data For Predictions: ' + str(data_unseen.shape))

from pycaret.classification import *



exp_clf101 = setup(data = data, target = 'target', session_id=123,

                   normalize = True, 

                   transformation = True, 

                   ignore_low_variance = True,

                   remove_multicollinearity = True, 

                   multicollinearity_threshold = 0.95)
compare_models()
lr = create_model('lr', fold = 10, round = 3)
#trained model object is stored in the variable 'lr'. 

print(lr)
tuned_lr = tune_model('lr', fold = 10, round = 3, optimize = 'AUC')
# Checking hyperparameters

plot_model(tuned_lr, plot = 'parameter')
#AUC plot

plot_model(tuned_lr, plot = 'auc')
# Precision-Recall Curve

plot_model(tuned_lr, plot = 'pr')
# Feature Importance 

plot_model(tuned_lr, plot='feature')
# Confusion Matrix

plot_model(tuned_lr, plot = 'confusion_matrix')
evaluate_model(tuned_lr)
predict_model(tuned_lr);
final_lr = finalize_model(tuned_lr)

print(final_lr)
unseen_predictions = predict_model(final_lr, data=data_unseen)

unseen_predictions.head()
save_model(final_lr,'Final_LR_Model_Jun2020')
saved_final_lr = load_model('Final_LR_Model_Jun2020')
new_prediction = predict_model(saved_final_lr, data=data_unseen)

new_prediction.head()
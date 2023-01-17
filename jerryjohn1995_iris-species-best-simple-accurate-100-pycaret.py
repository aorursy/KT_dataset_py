#The AutoML we are using here is pycaret, this is the step to install pycaret.

!pip install pycaret

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # Data visualisation 
import matplotlib.pyplot as plt # Data visualisation 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Getting the dataset to "dataset" variable

dataset = pd.read_csv("../input/iris/Iris.csv") # the iris dataset is now a Pandas DataFrame
# Showing first 5 rows.

dataset.head()
sns.FacetGrid(dataset,hue='Species',size=5).map(plt.scatter,'PetalLengthCm','PetalWidthCm').add_legend()
sns.FacetGrid(dataset,hue='Species',size=5).map(plt.scatter,'SepalWidthCm','PetalWidthCm').add_legend()
sns.FacetGrid(dataset,hue='Species',size=5).map(plt.scatter,'SepalWidthCm','PetalLengthCm').add_legend()
sns.FacetGrid(dataset,hue='Species',size=5).map(plt.scatter,'SepalLengthCm','PetalWidthCm').add_legend()
sns.FacetGrid(dataset,hue='Species',size=5).map(plt.scatter,'SepalLengthCm','PetalLengthCm').add_legend()
sns.FacetGrid(dataset,hue='Species',size=5).map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()
dataset.info()
dataset['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.show()
data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))
data['Species'].value_counts()
# Imporing pycaret classification method

from pycaret.classification import *
# This is the first step of model selection
# Here the data is our dataset, target is the labeled column(dependent variable), section is is random number for future identification.
exp = setup(data = data, target = 'Species', session_id=77 )

# After this we will get a list of our columns and its type, just conferm they are the same. Then hit enter.
#This comand is used to compare different models with our dataset.
#The acuuracy,F1 etc of each model is listed in a table.
#Choose which model you want
compare_models()
# With this command we are creating a Naives Byes model
# The code for Naives Byes is " nb "
# fold is the number of fold you want

nb_model = create_model('nb', fold = 10)

nb_tuned = tune_model('nb')
plot_model(nb_tuned, plot = 'auc')
plot_model(nb_tuned, plot = 'confusion_matrix')
predict_model(nb_tuned);
new_prediction = predict_model(nb_tuned, data=data_unseen)
new_prediction
!pip install pycaret
#import the dataset from pycaret repository
from pycaret.datasets import get_data
data = get_data('anomaly')

#import anomaly detection module
from pycaret.anomaly import *

#intialize the setup
exp_ano = setup(data)
knn_model = create_model('knn') #k nearest neighbour

# assign a model 
knn_df = assign_model(knn_model) 
plot_model(knn_model)
# generate predictions using trained model
knn_predictions = predict_model(knn_model, data = data)
knn_predictions

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

from pycaret.classification import *
data.head()
data.info()
import matplotlib.pyplot as plt 
plt.figure(figsize=(16,6))

import seaborn as sns
sns.set_style('darkgrid')
sns.countplot(x='Class',data=data)
print(len(data[data['Class']==1]))
print(len(data[data['Class']==0]))
!pip install imbalanced-learn
x=data.drop('Class',axis=1)
y=data.Class
from imblearn.under_sampling import NearMiss
under_sampler = NearMiss()
x_res,y_res = under_sampler.fit_sample(x,y)

from collections import Counter
print("before oversampling:",Counter(y))
print("after oversampling:",Counter(y_res))
y_res
x_res=pd.concat([x_res,y_res],axis=1)
x_res.head()
import matplotlib.pyplot as plt 
plt.figure(figsize=(10,6))

import seaborn as sns
sns.set_style('darkgrid')
sns.countplot(x='Class',data=x_res)
classifier = setup(data=x_res,target='Class')
!pip install scikit-learn==0.23.1
compare_models()
#creating the model
lightgbm_model= create_model('lightgbm')
print(lightgbm_model)
plot_model(lightgbm_model,plot='learning')
plot_model(lightgbm_model,plot='feature')
plot_model(lightgbm_model,plot='confusion_matrix')
plot_model(lightgbm_model,plot='class_report')
plot_model(lightgbm_model,plot='error')
interpret_model(lightgbm_model, plot ='correlation', feature = None, observation = None)
interpret_model(lightgbm_model)

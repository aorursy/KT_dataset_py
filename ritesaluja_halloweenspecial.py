# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.
#Data Summary - investigation 
df = pd.read_csv("../input/train.csv") #bone length measurements, severity of rot, extent of soullessness
dftest = pd.read_csv("../input/test.csv")
df.head()
df.describe()
sns.countplot(df['type'],palette=['purple','#996515','grey'],edgecolor='black',hatch='.....')
sns.countplot(df['type'],facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3),hatch='.....')
plt.xlabel('Types')
plt.title('Scary types')
plt.show()
#Soul distribution - Where on  earth is the Soulness? 
df['color'].value_counts()
plt.figure(figsize=(18,9))
#sns.violinplot(x=df['type'], y=df['has_soul'], hue=df['color'], data=df)
sns.swarmplot(x=df['type'], y=df['has_soul'], hue=df['color'],palette={'clear':'#C0C0C0','green':'green','black':'black','white':'#ffffff','blue':'blue','blood':'red'},\
              edgecolor='black',linewidth=0.2, data=df)
plt.show()
#Rotting Flesh and has Soul has relation ?
sns.jointplot(x='rotting_flesh', y='has_soul', 
              data=df, color ='grey', kind ='reg', 
              size = 8.0)
plt.show()
#Bone and Hair Length has any relation ?
sns.jointplot(x='bone_length', y='hair_length', 
              data=df, color ='orange', kind ='reg', 
              size = 8.0)
plt.show()
f, ax = plt.subplots(2,2,figsize=(12, 8))
sns.distplot(df['bone_length'], ax=ax[0][0])
sns.distplot(df['rotting_flesh'], ax=ax[0][1])
sns.distplot(df['hair_length'], ax=ax[1][0])
sns.distplot(df['has_soul'], ax=ax[1][1])
plt.subplots_adjust(hspace=.3)
plt.show()
#Correlation Matrix
d = df.drop(columns=['id'])
corr = d.corr()
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
#lets prdict something?
#Data preprocessing
df.isnull().sum() #all clean
df = pd.get_dummies(df,columns=['color'])
df.head(1)
#we can drop one color as it 5 other colors will be enough for representation, lets go ahead and make a training set by dropping white color and target varaible
y = df['type']
X = df.drop(columns=['id','color_white','type'])
#lanbel encoding target variable
from sklearn.preprocessing import LabelEncoder
col = LabelEncoder()
y = col.fit_transform(y)
#Random Forest model and training 
#splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=40, shuffle=True)

#model defination
rfc = ExtraTreesClassifier(n_jobs=-1 ) 

# Use a grid over parameters of interest
param_grid = { 
           "n_estimators" : [18, 27, 36, 45, 54, 63],
           "max_depth" : [1, 5, 10, 15, 20, 25],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10]}
 
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 10)
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)

#fitting, predicting and validating 
CV_rfc = rfc
CV_rfc.fit(X_train, y_train)
y_pred = CV_rfc.predict(X_train)
# eval
print("\nTraining Accuracy ",accuracy_score(y_train, y_pred))
y_pred = CV_rfc.predict(X_val)
print("\nValidation Accuracy ",accuracy_score(y_val, y_pred))

model = CV_rfc
#developing test data
dftest = pd.get_dummies(dftest,columns=['color'])
dftest.head(1)
Xtest = dftest.drop(columns=['id','color_white'])
#Prediction on test data
y_pred = model.predict(Xtest)
targettest = col.inverse_transform(y_pred) #inverse transform to get back original labels 
#submitting file for submission - only required for competition 
submit = pd.DataFrame()
submit["id"] = dftest["id"]
submit["type"] = targettest
submit.to_csv("sample_submission.csv", index = False)

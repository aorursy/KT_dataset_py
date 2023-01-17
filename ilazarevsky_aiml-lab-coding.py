# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.DataFrame({"A": [1,2,3], "B":["ab","cd","dffds"]})
penguins_df = pd.read_csv('/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')
penguins_df.shape
# pd.Series
penguins_df['body_mass_g'].mean()
penguins_df.loc[221]
species_mapping = {sp:i for i, sp in enumerate(penguins_df['species'].unique())}
species_mapping
from sklearn.model_selection import train_test_split
penguins_train, penguins_test = train_test_split(penguins_df, test_size=0.2)
penguins_train
sns.scatterplot(data=penguins_train, x='culmen_length_mm', y='culmen_depth_mm', hue='species')
sns.scatterplot(data=penguins_train, x='culmen_length_mm', y='culmen_depth_mm', hue='sex')
sns.distplot(a=penguins_df['culmen_length_mm'])
sns.pairplot(data=penguins_train, hue='sex')
from collections import Counter
for sp in penguins_train['species'].unique():
    print(sp)
    subdf = penguins_train[penguins_train.species==sp]
    print(Counter(subdf.sex,))
#     print(np.unique(subdf.sex, return_counts=True))
penguins_train.sample(6)
penguins_train.sex.unique()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
penguins_train.columns

def prepare_data(penguins, pipeline_stages=None, use_dummy=False):
    penguins = penguins.copy()
    penguins = penguins[penguins.sex != '.']
    penguins.dropna(axis=0, inplace=True)
    
    categories_for_one_hot = ['species', 'island']
    if not pipeline_stages:
        if use_dummy:
            one_hot = OneHotEncoder(sparse=False,drop='first')
        else:
            one_hot = OneHotEncoder(sparse=False)
        one_hot.fit(penguins.loc[:,categories_for_one_hot])
    else:
        one_hot = pipeline_stages['one_hot']
        
    one_hot_result = one_hot.transform(penguins.loc[:,categories_for_one_hot])
    
    offset = 0
    print(one_hot.categories_)
    for category_values, category_name in zip(one_hot.categories_, categories_for_one_hot):
        if use_dummy:
            category_values = category_values[1:]
        for  value in category_values:
            column_name = f'{category_name}={value}'
            penguins[column_name] = one_hot_result[:,offset]
            offset += 1
            
    
    
    penguins.drop(['species', 'island'], axis=1, inplace=True)
    
    if not pipeline_stages:
        sex_encoder = LabelEncoder()
        sex_encoder.fit(penguins['sex'])
    else:
        sex_encoder = pipeline_stages['sex_encoder']
    
    sex_column = sex_encoder.transform(penguins['sex'])
    
    size_features = ['culmen_length_mm', 'culmen_depth_mm','flipper_length_mm', 'body_mass_g']
    
    if not pipeline_stages:
        scaler = StandardScaler().fit(penguins.loc[:,size_features].values)
    else:
        scaler = pipeline_stages['scaler']
    
    scaled_features = scaler.transform(penguins.loc[:,size_features].values)
    print(scaled_features)
    penguins.loc[:,size_features] = scaled_features
    
    penguins.drop(['sex'], axis=1, inplace=True)
    
    
    
    
    print(penguins.columns)
    
    return penguins.values, sex_column, {"one_hot": one_hot, "sex_encoder": sex_encoder, "feature_names": penguins.columns, "scaler": scaler}
    
X_train, y_train, pipeline = prepare_data(penguins_train, use_dummy=True)
X_train.shape
X_train[:4]
pipeline
pipeline['scaler'].mean_

X_train[:4]
X_train.std(axis=0)
y_train
X_test, y_test, _ = prepare_data(penguins_test, pipeline, use_dummy=True)
y_test
X_test.shape
X_test
X_test.mean(axis=0)
X_test.std(axis=0)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
y_test
y_pred = dt.predict(X_test)
from sklearn import metrics
pd.DataFrame(metrics.confusion_matrix(y_test, y_pred), index=pipeline['sex_encoder'].classes_, columns=pipeline['sex_encoder'].classes_)
metrics.accuracy_score(y_test,y_pred)
print(metrics.classification_report(y_test, y_pred, target_names=pipeline['sex_encoder'].classes_))
from sklearn.tree import export_graphviz
graph = export_graphviz(dt,feature_names=pipeline['feature_names'], class_names=pipeline['sex_encoder'].classes_)
!pwd
with open('graph.dot','w') as f:
    f.write(graph)
!dot -Tpng graph.dot -o tree.png 
from IPython.display import Image 

Image("./tree.png")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_train, y_train)
lpred = clf.predict(X_test)
pd.DataFrame(metrics.confusion_matrix(y_test, lpred), index=pipeline['sex_encoder'].classes_, columns=pipeline['sex_encoder'].classes_)
clf.coef_
pd.DataFrame({"weight": clf.coef_[0]}, index=pipeline["feature_names"])
print(clf.intercept_)
graph

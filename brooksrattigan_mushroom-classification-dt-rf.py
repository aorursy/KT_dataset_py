# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
df = data.copy()
display(df.head())
display(df.tail())
df.info()
df.isnull().sum()
df.columns
len(df.columns)
df.columns = ['Mushroom_class', 'Cap_shape', 'Cap_surface', 'Cap_color', 'Bruises', 'Odor', 'Gill_attachment', 'Gill_spacing', 'Gill_size', 'Gill_color',
            'Stalk_shape', 'Stalk_root', 'Stalk_surface_above_ring', 'Stalk_surface_below_ring', 'Stalk_color_above_ring', 'Stalk_color_below_ring', 
            'Veil_type', 'Veil_color', 'Ring_number', 'Ring_type', 'Spore_print_color', 'Population', 'Habitat']
df.head()
df.columns
cols = df.columns
df[cols] = df[cols].astype('category')
# Alternative way
# df.columns.apply(lambda x: x.astype('category'))
df.info()
for col in df.columns:
    print(df[col].unique())
df.describe()
for col in df.columns:
    print(df[col].value_counts())
for i, col in enumerate(df.columns):
    plt.figure(i)
    plt.title(col, color = 'blue',fontsize=15)
    sns.countplot(x=col, data=df ,order=df[col].value_counts().index)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
df1 = df.copy()
df1.drop('Mushroom_class',axis=1,inplace=True)
def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df
df1 = one_hot(df1,df1.columns)
df1.head()
df1.info()
from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()
df['Mushroom_class_bin'] = lbe.fit_transform(df['Mushroom_class'])
df.head()
y = df["Mushroom_class_bin"]
X = df1.select_dtypes(exclude='category')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)
cart_model
!pip install skompiler
y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)
cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }
cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
print("Best Parameters: " + str(cart_cv_model.best_params_))
cart = tree.DecisionTreeClassifier(max_depth = 8, min_samples_split = 2)
cart_tuned = cart.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
Importance = pd.DataFrame({"Importance": cart_tuned.feature_importances_*100}, index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[:10]
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[:10].plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Categories");
# df2 = df.copy()
# df2.drop('Mushroom_class',axis=1,inplace=True)
# df2 = df2[['Odor', 'Stalk_root', 'Stalk_surface_below_ring', 'Spore_print_color', 'Ring_type']]
# df2.head()
# def one_hot(df, cols):
#     for each in cols:
#         dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
#         df = pd.concat([df, dummies], axis=1)
#     return df
# df2 = one_hot(df2,df2.columns)
# df2 = df2.select_dtypes(exclude=['category'])
# df2.head()
y = df["Mushroom_class_bin"]
X = df1[['Odor_n', 'Stalk_root_c', 'Stalk_surface_below_ring_y', 'Spore_print_color_r', 'Odor_a']]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
cart1 = DecisionTreeClassifier()
cart_model1 = cart.fit(X_train, y_train)
cart_model1
y_pred = cart_model1.predict(X_test)
accuracy_score(y_test, y_pred)
cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }
cart1 = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart1, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
print("Best Parameters: " + str(cart_cv_model.best_params_))
cart1 = tree.DecisionTreeClassifier(max_depth = 4, min_samples_split = 2)
cart_tuned1 = cart1.fit(X_train, y_train)
y_pred = cart_tuned1.predict(X_test)
accuracy_score(y_test, y_pred)
Importance1 = pd.DataFrame({"Importance": cart_tuned1.feature_importances_*100}, index = X_train.columns)
Importance1.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[:10]
Importance1.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[:10].plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Categories");
y = df["Mushroom_class_bin"]
X = df1.select_dtypes(exclude='category')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
rf_model
y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)
rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2) 
rf_cv_model.fit(X_train, y_train)
print("Best Parameters: " + str(rf_cv_model.best_params_))
rf_tuned = RandomForestClassifier(max_depth = 8, 
                                  max_features = 5, 
                                  min_samples_split = 2,
                                  n_estimators = 500)

rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:10]
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:10].plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Variables")
y = df["Mushroom_class_bin"]
X = df1[['Odor_n', 'Odor_f', 'Gill_size_b', 'Gill_size_n', 'Gill_color_b']]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
rf_model1 = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model1.predict(X_test)
accuracy_score(y_test, y_pred)
rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}
rf_model1 = RandomForestClassifier()

rf_cv_model1 = GridSearchCV(rf_model1, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2) 
rf_cv_model1.fit(X_train, y_train)
print("Best Parameters: " + str(rf_cv_model1.best_params_))
rf_tuned1 = RandomForestClassifier(max_depth = 2, 
                                  max_features = 2, 
                                  min_samples_split = 2,
                                  n_estimators = 500)

rf_tuned1.fit(X_train, y_train)
y_pred = rf_tuned1.predict(X_test)
accuracy_score(y_test, y_pred)
Importance = pd.DataFrame({"Importance": rf_tuned1.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:10]
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:10].plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Variables")

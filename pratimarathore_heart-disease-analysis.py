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

# installing Bubbly 

!pip install bubbly
!pip install pandas-profiling
import warnings

warnings.filterwarnings('ignore')



import pandas_profiling



# for data visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for advanced visualizations 

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected = True)

from bubbly.bubbly import bubbleplot



# for model explanation

import shap
#Reading input file 

heart_data = pd.read_csv('/kaggle/input/heart.csv')

heart_data.head(5)
# describing the data

heart_data.describe()
# Lets see the names of the feature and how much of information they give about the values in column

heart_data.columns
# let's change the names of the  columns for better understanding



heart_data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']



heart_data.columns
profile = pandas_profiling.ProfileReport(heart_data)

profile
figure = bubbleplot(dataset = heart_data, x_column = 'resting_blood_pressure', y_column = 'cholesterol', 

    bubble_column = 'sex', time_column = 'age', size_column = 'st_depression', color_column = 'sex', 

    x_title = "Resting Blood Pressure", y_title = "Cholestrol", title = 'BP vs Chol. vs Age vs Sex vs Heart Rate',

    x_logscale = False, scale_bubble = 3, height = 650)



py.iplot(figure, config={'scrollzoom': True})


df_dummies = pd.get_dummies(heart_data, columns = ['sex', 'chest_pain_type', 'fasting_blood_sugar','rest_ecg','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope', 'thalassemia'], drop_first = True)
df_dummies.columns
#changing few names of column for better description 

df_dummies.rename(columns = {'sex_1':'sex_male','fasting_blood_sugar_1':'fasting_blood_sugar_high','exercise_induced_angina_1':'exercise_induced_angina_yes',

                            'st_slope_1':'flat', 'st_slope_2':'downsloping'}, inplace = True) 



df_dummies.columns
data_df=df_dummies
#Splitting data in X and y datframe



y = data_df['target']



X = data_df.drop('target', axis = 1)



# checking the shapes of x and y

print("Shape of X:", X.shape)

print("Shape of y:", y.shape)

#Lets see the distributuin in the response/target variable 



y.value_counts()


from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# getting the shapes

print("Shape of x_train :", X_train.shape)

print("Shape of x_test :", X_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



dt = RandomForestClassifier(n_estimators = 10, max_depth = 5)

dt.fit(X_train, y_train)



y_train_pred = dt.predict(X_train)

y_test_pred = dt.predict(X_test)



# evaluating the model

print("Training Accuracy :", dt.score(X_train, y_train))

print("Testing Accuracy :", dt.score(X_test, y_test))



# classification report

cr = classification_report(y_test, y_test_pred)

print(cr)
# cofusion matrix

cm = confusion_matrix(y_test, y_test_pred)

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')

dt = RandomForestClassifier(random_state=42)



from sklearn.model_selection import GridSearchCV



# Create the parameter grid based on the results of random search 

params = {

    'max_depth': [1, 2, 5, 10, 20],

    'min_samples_leaf': [5, 10, 20, 50, 100],

    'max_features': [2,3,4],

    'n_estimators': [10, 30, 50, 100, 200]

}

# Instantiate the grid search model

grid_search = GridSearchCV(estimator=dt, 

                           param_grid=params, 

                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")



grid_search.fit(X_train, y_train)
grid_search.best_estimator_
dt_best = grid_search.best_estimator_
# evaluating the model

print("Training Accuracy :", dt_best.score(X_train, y_train))

print("Testing Accuracy :", dt_best.score(X_test, y_test))

# cofusion matrix

cm = confusion_matrix(y_test, dt_best.predict(X_test))

plt.rcParams['figure.figsize'] = (5, 5)

sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')



# classification report

cr = classification_report(y_test, dt_best.predict(X_test))

print(cr)
from sklearn.tree import export_graphviz



estimator = dt_best.estimators_[1]

feature_names = [i for i in X_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no disease'

y_train_str[y_train_str == '1'] = 'disease'

y_train_str = y_train_str.values





export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)



from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=50'])



from IPython.display import Image

Image(filename = 'tree.png')
total=sum(sum(cm))



sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])

print('Sensitivity : ', sensitivity )



specificity = cm[1,1]/(cm[1,1]+cm[0,1])

print('Specificity : ', specificity)
from sklearn.metrics import plot_roc_curve

plot_roc_curve(dt_best, X_train, y_train)

plt.show()
#for SHAP values

import shap 

from pdpbox import pdp, info_plots #for partial plots



# let's see the shap values



explainer = shap.TreeExplainer(dt_best)

shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test)
# let's create a function to check the patient's conditions



def patient_analysis(model, patient):

  explainer = shap.TreeExplainer(model)

  shap_values = explainer.shap_values(patient)

  shap.initjs()

  return shap.force_plot(explainer.expected_value[1], shap_values[1], patient)



patients = X_test.iloc[1,:].astype(float)

patient_analysis(dt_best, patients)
patients = X_test.iloc[10,:].astype(float)

patient_analysis(dt_best, patients)
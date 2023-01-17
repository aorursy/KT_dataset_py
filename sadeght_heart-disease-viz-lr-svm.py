import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))
# Remove Warnings in JupyterNotebook (I hate those warnings!)

import warnings

warnings.simplefilter('ignore')
df = pd.read_csv('../input/heart.csv')

df.sample(3)
# Let's change the column names based on information on dataset description. It would be much more understandable.

df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target' ]
df.shape
# There's no Null value inside the data frame

df.isnull().sum().value_counts()
# Check the Correlation between variables

df_corr = df.corr()

plt.figure(figsize=(15,8))

sns.heatmap(df_corr, cmap='coolwarm', annot=True)
# Turn our data into matrix form

dfmat = df.pivot_table(columns='age', index='st_depression', values='target')



# Use heatmap to draw the chart. I also used invert_yaxis() to change the order of y_axis,

# you can delete this part to see the difference.

sns.heatmap(dfmat, cmap='coolwarm', cbar=False).invert_yaxis()



# Let's have more clever chart! illustrating labels and legend

plt.ylabel('ST Depression')

plt.xlabel('Age')



from matplotlib.patches import Patch

red_patch = Patch(color='#B40426', label='Target= 1')

blue_patch = Patch(color='blue', label='Target= 0')

plt.legend(handles=[red_patch, blue_patch])
sns.jointplot(x=df.age, y=df.st_depression, data=df, kind='kde')
sns.lmplot(data=df, x='age', y='max_heart_rate_achieved', hue='sex', markers=['o','v'])
FG = sns.FacetGrid(data=df, row='sex', col='chest_pain_type')

FG.map(sns.distplot, 'age')
i = df.groupby(['chest_pain_type'])['sex']

i

#sns.jointplot(x=df.age, y=df.chest_pain_type, data=df, kind='kde')
sns.violinplot(data=df, x='chest_pain_type', y='age', hue='sex', split=True)
df.columns
from sklearn.model_selection import train_test_split



X = df.drop('target', axis=1)

y = df['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
from sklearn.linear_model import LinearRegression



# create an instance from LinearRegression name "lm" and fit the model

lm = LinearRegression()

lm.fit(X_train, y_train)
coeff_df = pd.DataFrame((lm.coef_)*100, X.columns, columns=['Coefficient (percentage %)'])

coeff_df
prediction = lm.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print(classification_report(y_test, prediction.round()))
print('The Accuarcy Score of our Model would be: ', accuracy_score(y_test, prediction.round())*100, '%')
from sklearn.svm import SVC



# create an instance from SCV and fit the model

SVM_model = SVC()

SVM_model.fit(X_train, y_train)
prediction2= SVM_model.predict(X_test)
print(confusion_matrix(y_test, prediction2))

print(classification_report(y_test,prediction2))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 



from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))

print(classification_report(y_test,grid_predictions))
print('The Accuarcy Score of our Model would be: ', accuracy_score(y_test, grid_predictions.round())*100, '%')
y = (accuracy_score(y_test, grid_predictions.round())*100, accuracy_score(y_test, prediction.round())*100)

x = ['SVM', 'Linear Regression']

plt.bar(x, y)

plt.ylabel('Accuracy Score (%)')
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



X = df.drop('target', axis=1)

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)



data = df

my_model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
row_to_show = 5

data_for_prediction = X_test.iloc[row_to_show]

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

my_model.predict_proba(data_for_prediction_array)
import shap



# Create object that can calculate shap values

explainer = shap.TreeExplainer(my_model)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
explainer = shap.TreeExplainer(my_model)

shap_values = explainer.shap_values(X_test)



shap.summary_plot(shap_values[1], X_test, plot_type="bar")
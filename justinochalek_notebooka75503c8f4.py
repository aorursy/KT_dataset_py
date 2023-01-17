# Import useful libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from bokeh.plotting import figure, output_notebook, show

from sklearn.metrics import confusion_matrix

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from IPython.display import Image

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV



# Import files

# Load dataset and look at the first five samples

df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv', na_values=['?'])

df.head()
# Check for null elements

df.isnull().sum()
# Check for duplicates

df.duplicated().sum()
# Drop the duplicate

df.drop_duplicates()
# Descriptive statistics summary

df.describe()
#Investigate nominal features

print('Class labels:', np.unique(df['cp']))

print('Class labels:', np.unique(df['restecg']))

print('Class labels:', np.unique(df['thal']))
sum(df['thal']==0)
df_original = pd.read_csv('/kaggle/input/processedclevelanddata/processed.cleveland.data', na_values=['?'], header=None)

print('Class labels:', np.unique(df_original[12]))
#Let's compare the total number of elements holding each categorical value.

print('Original 3:', sum(df_original[12]==3))

print('Original 6:', sum(df_original[12]==6))

print('Original 7:', sum(df_original[12]==7))

print('Kaggle 1:', sum(df['thal']==1))

print('Kaggle 2:', sum(df['thal']==2))

print('Kaggle 3:', sum(df['thal']==3))

df = df.replace({'thal': 0}, np.nan)

df = df.dropna(axis=0)

df.head()
# Transform nominal feature columns into dummy features for each unique value

dummies = pd.get_dummies(df, columns=['cp','restecg', 'thal'], drop_first=True)

dummies.head()
# Give the features more descriptive names

dummies.rename(columns={'trestbps': 'blood_pressure', 'chol': 'cholesterol', 'fbs': 'blood_sugar',

                        'thalach': 'max_heart_rate', 'exang': 'exercise_induced_angina',

                        'oldpeak': 'exercise_induced_ST_depression', 'slope': 'ST_slope',

                        'ca': 'calcification', 'cp_1': 'atypical_angina', 'cp_2': 'non_anginal_pain',

                        'cp_3': 'asymptomatic', 'restecg_1': 'ST_T_wave_abnormality',

                        'restecg_2': 'Ventricalular_hypertrophy', 'thal_2.0': 'Scintigraphy_normal',

                        'thal_3.0': 'Scintigraphy_reversable'}, inplace=True)

dummies.head()
# I prefer 'target' to be the last column.

dft = dummies.pop('target')

df2 = pd.concat([dummies, dft], axis=1)

df2.head()
# Check for appropriate data types

df2.dtypes
# Assign feature set and target

X = df2.iloc[:, :17]

y = df2.target
# Investigate what clinical insight can be gained by visualizing the data.



# output to jupyter notebook

output_notebook()



# create a new plot with a title and axis labels

p = figure(plot_width=700, plot_height=650, title="Heart Disease by age and cholesterol", x_axis_label='age in years', y_axis_label='serum cholestoral in mg/dl')



# add a line renderer with legend and line thickness

p.inverted_triangle(X[:165].age, X[:165].cholesterol, size=10, color='crimson', alpha=0.5, legend_label='Heart Disease')

p.circle(X[165:303].age, X[165:303].cholesterol, size=10, color='darkcyan', alpha=0.5, legend_label='null')

        



# show the results

show(p)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
print('Labels counts in y:', np.bincount(y))

print('Labels counts in y_train:', np.bincount(y_train))

print('Labels counts in y_test:', np.bincount(y_test))
model1 = RandomForestClassifier(criterion='entropy',

                                n_estimators=25, 

                                random_state=1,

                                n_jobs=4, max_depth=5)

model1.fit(X_train, y_train)

y_pred = model1.predict(X_test)

model1.score(X_test, y_test)
importances = model1.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30, df2.columns[:][indices[f]], importances[indices[f]]))
perm = PermutationImportance(model1, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
confusion_matrix1 = confusion_matrix(y_test, y_pred)

confusion_matrix1
tn, fp, fn, tp = confusion_matrix1.ravel()

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)

print('Sensitivity : ', sensitivity)

print('Specificity : ', specificity)

#X.columns[10]

#X_clinical.head()

X_clinical_train, X_clinical_test = X_train.iloc[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]], X_test.iloc[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]]
model2 = RandomForestClassifier(criterion='entropy',

                                n_estimators=250, 

                                random_state=1,

                                n_jobs=4, max_features='log2')

model2.fit(X_clinical_train, y_train)

y_clinical_pred = model2.predict(X_clinical_test)

model2.score(X_clinical_test, y_test)
confusion_matrix_clinical = confusion_matrix(y_test, y_clinical_pred)

tn, fp, fn, tp = confusion_matrix_clinical.ravel()

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)

print('Sensitivity : ', sensitivity)

print('Specificity : ', specificity)
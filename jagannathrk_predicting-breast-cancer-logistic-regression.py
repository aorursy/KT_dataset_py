# import dependencies

# data cleaning and manipulation 

import pandas as pd

import numpy as np



# data visualization

import matplotlib.pyplot as plt

import seaborn as sns



# machine learning

from sklearn.preprocessing import StandardScaler



import sklearn.linear_model as skl_lm

from sklearn import preprocessing

from sklearn import neighbors

from sklearn.metrics import confusion_matrix, classification_report, precision_score

from sklearn.model_selection import train_test_split





import statsmodels.api as sm

import statsmodels.formula.api as smf





# initialize some package settings

sns.set(style="whitegrid", color_codes=True, font_scale=1.3)



%matplotlib inline
# read in the data and check the first 5 rows

df = pd.read_csv('../input/data.csv', index_col=0)

df.head()
# general summary of the dataframe

df.info()
# remove the 'Unnamed: 32' column

df = df.drop('Unnamed: 32', axis=1)
# check the data type of each column

df.dtypes
# visualize distribution of classes 

plt.figure(figsize=(8, 4))

sns.countplot(df['diagnosis'], palette='RdBu')



# count number of obvs in each class

benign, malignant = df['diagnosis'].value_counts()

print('Number of cells labeled Benign: ', benign)

print('Number of cells labeled Malignant : ', malignant)

print('')

print('% of cells labeled Benign', round(benign / len(df) * 100, 2), '%')

print('% of cells labeled Malignant', round(malignant / len(df) * 100, 2), '%')
# generate a scatter plot matrix with the "mean" columns

cols = ['diagnosis',

        'radius_mean', 

        'texture_mean', 

        'perimeter_mean', 

        'area_mean', 

        'smoothness_mean', 

        'compactness_mean', 

        'concavity_mean',

        'concave points_mean', 

        'symmetry_mean', 

        'fractal_dimension_mean']



sns.pairplot(data=df[cols], hue='diagnosis', palette='RdBu')
# Generate and visualize the correlation matrix

corr = df.corr().round(2)



# Mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set figure size

f, ax = plt.subplots(figsize=(20, 20))



# Define custom colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.tight_layout()
# first, drop all "worst" columns

cols = ['radius_worst', 

        'texture_worst', 

        'perimeter_worst', 

        'area_worst', 

        'smoothness_worst', 

        'compactness_worst', 

        'concavity_worst',

        'concave points_worst', 

        'symmetry_worst', 

        'fractal_dimension_worst']

df = df.drop(cols, axis=1)



# then, drop all columns related to the "perimeter" and "area" attributes

cols = ['perimeter_mean',

        'perimeter_se', 

        'area_mean', 

        'area_se']

df = df.drop(cols, axis=1)



# lastly, drop all columns related to the "concavity" and "concave points" attributes

cols = ['concavity_mean',

        'concavity_se', 

        'concave points_mean', 

        'concave points_se']

df = df.drop(cols, axis=1)



# verify remaining columns

df.columns
# Draw the heatmap again, with the new correlation matrix

corr = df.corr().round(2)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()
# Split the data into training and testing sets

X = df

y = df['diagnosis']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
# Create a string for the formula

cols = df.columns.drop('diagnosis')

formula = 'diagnosis ~ ' + ' + '.join(cols)

print(formula, '\n')
# Run the model and report the results

model = smf.glm(formula=formula, data=X_train, family=sm.families.Binomial())

logistic_fit = model.fit()



print(logistic_fit.summary())
# predict the test data and show the first 5 predictions

predictions = logistic_fit.predict(X_test)

predictions[1:6]
# Note how the values are numerical. 

# Convert these probabilities into nominal values and check the first 5 predictions again.

predictions_nominal = [ "M" if x < 0.5 else "B" for x in predictions]

predictions_nominal[1:6]
print(classification_report(y_test, predictions_nominal, digits=3))



cfm = confusion_matrix(y_test, predictions_nominal)



true_negative = cfm[0][0]

false_positive = cfm[0][1]

false_negative = cfm[1][0]

true_positive = cfm[1][1]



print('Confusion Matrix: \n', cfm, '\n')



print('True Negative:', true_negative)

print('False Positive:', false_positive)

print('False Negative:', false_negative)

print('True Positive:', true_positive)

print('Correct Predictions', 

      round((true_negative + true_positive) / len(predictions_nominal) * 100, 1), '%')
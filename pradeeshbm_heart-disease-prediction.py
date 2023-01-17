import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random as rnd

import os

import warnings



from operator import add



MEDIUM_SIZE = 10

SMALL_SIZE = 8

MEDIUM_SIZE = 10

BIGGER_SIZE = 12



%matplotlib inline

warnings.filterwarnings('ignore')



print(os.listdir("../input"))

os.chdir("../input")
df = pd.read_csv('heart.csv')

df.head()
print(f'Dataset contains {df.shape[0]} samples, {df.shape[1] - 1} independent features 1 target continuous variable.')
print(df.info())

missing_values = (df.isnull().sum() / len(df)) * 100

print("\nFeatures with missing values: \n", missing_values[missing_values > 0])
df.describe()
print(np.char.center(" Unique values of categorical variables ", 60, fillchar = "*"))

print("\nSex: ", df.sex.unique())

print("Cp: ", sorted(df.cp.unique()))

print("fbs: ", sorted(df.fbs.unique()))

print("restecg: ", sorted(df.restecg.unique()))

print("exang: ", sorted(df.exang.unique()))

print("slope: ", sorted(df.slope.unique()))

print("ca: ", sorted(df.ca.unique()))

print("thal: ", sorted(df.thal.unique()))

print("target: ", sorted(df.target.unique()))
def draw_semi_pie_chart(data, column, fig, renamed_index_dict, title):

    default_colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666']

    rnd.shuffle(default_colors)

    ax = df[column].value_counts().rename(index = renamed_index_dict).plot.pie(colors = default_colors, autopct='%1.1f%%', startangle=90, title = title)

    ax.set_ylabel('')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):

        item.set_fontsize(20)

        

    centre_circle = plt.Circle((0,0), 0.70, fc='white')

    fig.gca().add_artist(centre_circle)
fig = plt.gcf()

fig.set_size_inches(18, 17)

grid_rows = 3

grid_cols = 3



# Draw Sex Pie chart

plt.subplot(grid_rows, grid_cols, 1)

draw_semi_pie_chart(df, 'sex', fig, {0: 'Female', 1: 'Male'}, 'Sex')



# Draw Chest pain type chart

plt.subplot(grid_rows, grid_cols, 2)

draw_semi_pie_chart(df, 'cp', fig, {0:'Typical Angina', 1:'Atypical Angina', 2:'Non-anginal Pain',3:'Asymptomatic'}, 'Chest Pain Type')



# Draw Fasting blood sugar chart

plt.subplot(grid_rows, grid_cols, 3)

draw_semi_pie_chart(df, 'fbs', fig, {0:'True', 1:'False'}, 'Fasting Blood Sugar')



# Draw restecg - resting electrocardiographic results

plt.subplot(grid_rows, grid_cols, 4)

draw_semi_pie_chart(df, 'restecg', fig, {0:'Normal', 1:'Abnormality', 2:'Left Ventricular Hypertrophy'}, 'Resting Electrocardiographic Results')



# Draw exang - exercise induced angina

plt.subplot(grid_rows, grid_cols, 5)

draw_semi_pie_chart(df, 'exang', fig, {0:'Not Induced', 1:'Induced'}, 'Exercise Induced Angina')



# Draw exang - exercise induced angina

plt.subplot(grid_rows, grid_cols, 6)

draw_semi_pie_chart(df, 'slope', fig, {0:'Upsloping', 1:'Flat', 2:'Downsloping'}, 'Slope')



# Draw ca

plt.subplot(grid_rows, grid_cols, 7)

draw_semi_pie_chart(df, 'ca', fig, {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}, 'CA')



# Draw thal

plt.subplot(grid_rows, grid_cols, 8)

draw_semi_pie_chart(df, 'thal', fig, {0:'0', 1:'1', 2:'2', 3:'3'}, 'Thal')



fig.tight_layout()

plt.show()
def create_percent_stacked_barchart(data, title = None, ylabel = None, xlabel = None):

    default_colors = ['#019600', '#3C5F5A', '#219AD8']

    # From raw value to percentage

    totals = data.sum(axis=1)

    bars = ((data.T / totals) * 100).T

    r = list(range(data.index.size))



    # Plot

    barWidth = 0.95

    names = data.index.tolist()

    bottom = [0] * bars.shape[0]



    # Create bars

    color_index = 0

    plots = []

    for bar in bars.columns:

        plots.append(plt.bar(r, bars[bar], bottom=bottom, color=default_colors[color_index], edgecolor='white', width=barWidth))

        bottom = list(map(add, bottom, bars[bar]))

        color_index = 0 if color_index >= len(default_colors) else color_index + 1



    # Custom x axis

    plt.title(title)

    plt.xticks(r, names)

    plt.xlabel(data.index.name if xlabel is None else xlabel)

    plt.ylabel(data.columns.name if ylabel is None else ylabel)

    ax = plt.gca()

        

    y_labels = ax.get_yticks()

    ax.set_yticklabels([str(y) + '%' for y in y_labels])



    flat_list = [item for sublist in data.T.values for item in sublist]

    for i, d in zip(ax.patches, flat_list):

        data_label = str(d) + " (" + str(round(i.get_height(), 2)) + "%)"

        ax.text(i.get_x() + 0.45, i.get_y() + 5, data_label, horizontalalignment='center', verticalalignment='center', fontdict = dict(color = 'white', size = 20))



    for item in ([ax.title]):

        item.set_fontsize(27)

        

    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):

        item.set_fontsize(24)

    

    legend = ax.legend(plots, bars.columns.tolist(), fancybox=True)

    plt.setp(legend.get_texts(), fontsize='20')
fig = plt.gcf()

fig.set_size_inches(25, 35)

grid_rows = 4

grid_cols = 2



# Draw Disease Status vs Sex chart

plt.subplot(grid_rows, grid_cols, 1)

temp = df[['sex','target']].groupby(['sex','target']).size().unstack('target')

temp.rename(index={0:'Female', 1:'Male'}, columns={0:'No Disease', 1:'Has Disease'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Disease Status vs Sex', ylabel = 'Population')



# Draw Disease Status vs Chest pain type chart

plt.subplot(grid_rows, grid_cols, 2)

temp = df[['cp','target']].groupby(['cp','target']).size().unstack('target')

temp.rename(index={0:'Typical \nAngina', 1:'Atypical \nAngina', 2:'Non-\nanginal\nPain',3:'Asymptomatic'}, columns={0:'No Disease', 1:'Has Disease'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Disease Status vs Chest Pain Type (cp)', ylabel = 'Population', xlabel = 'Chest Pain Type')



# Draw fbs - fasting blood sugar chart

plt.subplot(grid_rows, grid_cols, 3)

temp = df[['fbs','target']].groupby(['fbs','target']).size().unstack('target')

temp.rename(index={0:'True', 1:'False'}, columns={0:'No Disease', 1:'Has Disease'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Disease Status vs Fasting Blood Sugar(fbs)', ylabel = 'Population', xlabel = 'Fasting Blood Sugar > 120 mg/dl')



# Draw restecg - resting electrocardiographic results chart

plt.subplot(grid_rows, grid_cols, 4)

temp = df[['restecg','target']].groupby(['restecg','target']).size().unstack('target')

temp.rename(index={0:'Normal', 1:'Abnormality', 2:'Left Ventricular \nHypertrophy'}, columns={0:'No Disease', 1:'Has Disease'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Disease Status vs Resting Electrocardiographic Results (restecg)', ylabel = 'Population', xlabel = 'Resting Electrocardiographic Results')



# Draw exang - exercise induced angina chart

plt.subplot(grid_rows, grid_cols, 5)

temp = df[['exang','target']].groupby(['exang','target']).size().unstack('target')

temp.rename(index={0:'Not Induced', 1:'Induced'}, columns={0:'No Disease', 1:'Has Disease'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Disease Status vs Exercise Induced Angina (exang)', ylabel = 'Population', xlabel = 'Exercise Induced Angina')



# Draw slope - the slope of the peak exercise ST segment chart

plt.subplot(grid_rows, grid_cols, 6)

temp = df[['slope','target']].groupby(['slope','target']).size().unstack('target')

temp.rename(index={0:'Upsloping', 1:'Flat', 2:'Downsloping'}, columns={0:'No Disease', 1:'Has Disease'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Disease Status vs Slope', ylabel = 'Population', xlabel = 'Slope')



# Draw ca - number of major vessels (0-3) colored by flourosopy chart

plt.subplot(grid_rows, grid_cols, 7)

temp = df[['ca','target']].groupby(['ca','target']).size().unstack('target')

temp.rename(columns={0:'No Disease', 1:'Has Disease'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Disease Status vs CA', ylabel = 'Population', xlabel = 'CA')



# Draw thal chart

plt.subplot(grid_rows, grid_cols, 8)

temp = df[['thal','target']].groupby(['thal','target']).size().unstack('target')

temp.rename(columns={0:'No Disease', 1:'Has Disease'}, inplace = True)

create_percent_stacked_barchart(temp, title = 'Disease Status vs Thal', ylabel = 'Population', xlabel = 'Thal')

fig.tight_layout()

plt.show()
fig = plt.gcf()

fig.set_size_inches(15, 8)

sns.heatmap(df.corr(), annot = True)

plt.show()
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']

number_of_columns = len(continuous_features)

number_of_rows = 5

plt.figure(figsize=(23, 18))



for i, f in enumerate(continuous_features):

    plt.subplot(number_of_rows + 1, number_of_columns, i + 1)

    sns.distplot(df[f], kde=True)
sns.pairplot(df, hue = 'target', markers=["o", "s"], vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], palette = sns.color_palette("bright", 10))
nominal_features = ['cp', 'slope', 'thal', 'restecg']

x = pd.get_dummies(df.drop(['target'], axis = 1), columns = nominal_features, drop_first=True).values

y = df.target.values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
print("Shape of X before Dimensionlity Reduction: ", x_train.shape)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()

x_train = lda.fit_transform(x_train, y_train)

x_test = lda.transform(x_test)



print("Shape of X after Dimensionlity Reduction: ", x_train.shape)
# SVM

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(x_train, y_train)

y_pred_svm = classifier.predict(x_test)



# KNN

from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier()

classifier_knn.fit(x_train, y_train)

y_pred_knn = classifier_knn.predict(x_test)
from sklearn.metrics import confusion_matrix



print("SVM Confusion Matrix")

cm = confusion_matrix(y_test, y_pred_svm)

print(cm)



print("KNN Confusion Matrix")

cm = confusion_matrix(y_test, y_pred_knn)

print(cm)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(classifier, x_train, y_train, cv = 10)

print("Scores: ", scores)

print("Accuracy: ", round(scores.mean(), 2) * 100, "%")

print("Standard Deviation: +/-", scores.std())
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(x_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print("Best Score: ", best_accuracy)

print("Best Params: ", best_parameters)
from sklearn.model_selection import GridSearchCV

parameters = {'n_neighbors': np.arange(1, 10)}

grid_search = GridSearchCV(estimator = classifier_knn, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(x_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print("Best Score: ", best_accuracy)

print("Best Params: ", best_parameters)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', C = 1, random_state = 0, probability = True)

classifier.fit(x_train, y_train)

probs = classifier.predict_proba(x_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(fpr, tpr, marker='.')

plt.show()
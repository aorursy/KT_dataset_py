# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%config IPCompleter.greedy=True

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

missing_values=["n/a", "?", "na"]

df=pd.read_csv("../input/heart.csv", na_values=missing_values)

df.head(10)
df.shape
df.columns
df.dtypes
df['target'].value_counts()
df.isnull().sum()
countFemale = len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
sns.countplot(x='sex', data=df, palette="mako_r")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
#df1 = len(df[(df.sex == 1) & (df.target == 1)])

#df2 = len(df[(df.sex == 0) & (df.target == 0)])

print("Percentage of Male having heart disease: {:.2f}%".format((len(df[(df.sex == 1) & (df.target == 1)]) / (len(df[(df.sex == 1)]))*100)))

print("Percentage of Male Not having heart disease: {:.2f}%".format((len(df[(df.sex == 1) & (df.target == 0)]) / (len(df[(df.sex == 1)]))*100)))

print("Percentage of Female having heart disease: {:.2f}%".format((len(df[(df.sex == 0) & (df.target == 1)]) / (len(df[(df.sex == 0)]))*100)))

print("Percentage of Female Not having heart disease: {:.2f}%".format((len(df[(df.sex == 0) & (df.target == 0)]) / (len(df[(df.sex == 0)]))*100)))
print("Percentage of Male having heart disease:" ,len(df[(df.sex == 1) & (df.target == 1)]))

print("Percentage of Male Not having heart disease: ",len(df[(df.sex == 1) & (df.target == 0)]))

print("Percentage of Female having heart disease: ",len(df[(df.sex == 0) & (df.target == 1)]))

print("Percentage of Female Not having heart disease: ",len(df[(df.sex == 0) & (df.target == 0)]))
sns.countplot(df.sex, hue=df.target)
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])

plt.title('Heart Disease Frequency for Slope')

plt.xlabel('The Slope of The Peak Exercise ST Segment ')

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])

plt.title('Heart Disease Frequency According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.show()
df['sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%')
from operator import add

def create_percent_stacked_barchart(data, title = None, ylabel = None, xlabel = None):

    default_colors = ['red', 'blue', 'black']

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

plt.savefig("all.png")

plt.show()

#The feature 'sex' is biased in the sample. Hence saying 'most of the Females are tend to have heart disease' is untrue.

#Pain with chest pain types Atypical Angina, Non-Anginal Pain, Asymptomatic more likely to have heart disease.
fig = plt.gcf()

fig.set_size_inches(15, 8)

sns.heatmap(df.corr(), annot = True)

plt.show()

#There is no features with more than 0.5 correlation. This is a sad thing. :(
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']

number_of_columns = len(continuous_features)

number_of_rows = 5

plt.figure(figsize=(30, 18))



for i, f in enumerate(continuous_features):

    plt.subplot(number_of_rows + 1, number_of_columns, i + 1)

    sns.distplot(df[f], kde=True)



    

#The features Age, trestbps, chol are normally distributed.

#The likelihood of getting heart disease of more for the people with age 50 - 60.

#The target variable is balanced.

#since 'cp', 'thal' and 'slope' are categorical variables we'll turn them into dummy variables

a = pd.get_dummies(df['cp'], prefix = "cp")

b = pd.get_dummies(df['thal'], prefix = "thal")

c = pd.get_dummies(df['slope'], prefix = "slope")
frames = [df, a, b, c]

df = pd.concat(frames, axis = 1)

df.head()
df = df.drop(columns = ['cp', 'thal', 'slope'])

df.head()
y = df.target.values

x_data = df.drop(['target'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#Feature Scalling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
#Linear Discriminant Analysis (LDA) will be used as dimensionality reduction technique for this dataset since it's a classification problem.

print("Shape of X before Dimensionlity Reduction: ", x_train.shape)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()

x_train = lda.fit_transform(x_train, y_train)

x_test = lda.transform(x_test)



print("Shape of X after Dimensionlity Reduction: ", x_train.shape)
# SVM

from sklearn.svm import SVC

svm = SVC(kernel = 'linear', random_state = 0)

svm.fit(x_train, y_train)

svm_prediction = svm.predict(x_test)

print("Test Accuracy of SVM {:.2f}%".format(svm.score(x_test,y_test)*100))

print(svm)

from sklearn.metrics import confusion_matrix



print("SVM Confusion Matrix")

cm = confusion_matrix(y_test, svm_prediction)

sns.heatmap(cm,annot=True)

plt.show()

TN=cm[0][0]

FN=cm[1][0]

TP=cm[1][1]

FP=cm[0][1]

recall=TP/(FN+TP)*100 #means 91% time model told that disease has present

print("recall :",recall)

precision=TP/(FP+TP)*100

print("precision :",precision)

F1_Score=2*(recall * precision) / (recall + precision)

print("F1_Score :",F1_Score)
#K-Fold Cross Validation(10 fold is used here)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(svm, x_train, y_train, cv = 10)

print("Scores: ", scores)

print("Accuracy: ", round(scores.mean(), 2) * 100, "%")

print("Standard Deviation: +/-", scores.std())
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = svm, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(x_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print("Best Score: ", best_accuracy)

print("Best Params: ", best_parameters)
#AUC



from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



from sklearn.svm import SVC

svm = SVC(kernel = 'linear', C = 1, random_state = 0, probability = True)

svm.fit(x_train, y_train)

probs = svm.predict_proba(x_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y_test, probs)

print('SVM AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(fpr, tpr, marker='.')

plt.show()



#Linear SVM with C = 1, will be chosen as the best model for this problem.

#The best accuracy has been obtained as 91.4%
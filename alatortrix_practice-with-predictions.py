# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for heatmap

import matplotlib.pyplot as plt # for common plotting

import graphviz as gv # for decision tree plotting

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import train_test_split

# GaussianNB, BernoulliNB, MultinomialNB

from sklearn.naive_bayes import GaussianNB as bayes_model

from sklearn.metrics import r2_score as model_score, classification_report, confusion_matrix, accuracy_score

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

animals = pd.read_csv("../input/train/train.csv")

test = pd.read_csv("../input/test/test.csv")

# Any results you write to the current directory are saved as output.

animals.head(3)
animals.describe()
animalsBySpeed = animals.groupby(['AdoptionSpeed']).size().reset_index(name='Count')

dataToPlot = pd.DataFrame({'Percentage': pd.Series(animalsBySpeed.apply(lambda row: row['Count']/len(animals),axis=1), index=animalsBySpeed.index)})

dataToPlot.set_index([['Same Day','1st week','1st Month','2nd & 3rd Month','Other']], inplace=True)

dataToPlot
dataToPlot.plot.pie(y='Percentage', figsize=(10,10))
animalsByType = animals.groupby(['Type']).size().reset_index(name='Count')

animalsByType.plot.barh(y='Count',x='Type')
animals['Age'].corr(animals['AdoptionSpeed'])
corr = animals.corr()

# plot the heatmap

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.5, square=True)
animalsBow = animals._get_numeric_data()

distortions = []

K = range(1,10)

for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(animalsBow)

    kmeanModel.fit(animalsBow)

    distortions.append(sum(np.min(cdist(animalsBow, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / animalsBow.shape[0])



# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
features = animals[['Age', 'MaturitySize', 'FurLength', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', 'AdoptionSpeed']]

kmeans = KMeans(n_clusters = 6).fit_predict(features)

rows, cols = features.shape

fig, axs = plt.subplots(cols, cols,figsize=(50, 50))

for i in range(0, cols):

    for j in range(0, cols):

        if j != i:

            axs[i, j].scatter(features.iloc[:,i], features.iloc[:,j], c=kmeans)

            axs[i,j].set(xlabel=features.columns[i],ylabel=features.columns[j])

plt.show()
#features = animals.drop(['Name', 'RescuerID', 'PetID', 'Description', 'Quantity', 'Vaccinated', 'AdoptionSpeed'], axis=1)

features = animals[['Age', 'MaturitySize', 'FurLength', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']]

speed = animals[['AdoptionSpeed']]

classifier = tree.DecisionTreeClassifier().fit(features, speed)

cleaned_test = test[['Age', 'MaturitySize', 'FurLength', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt']]

#cleaned_test = test.drop(['Name', 'RescuerID', 'PetID', 'Description', 'Quantity', 'Vaccinated'], axis=1)

prediction = classifier.predict(cleaned_test)

prediction
graph_data = tree.export_graphviz(classifier, out_file=None, max_depth=5)

graph = gv.Source(graph_data)

graph
result = test[['PetID']].assign(AdoptionSpeed=pd.Series(prediction))

result.to_csv('submission.csv', index=False)
myvar = animals[['Dewormed','Vaccinated','Sterilized','Fee']].copy()

myvar['Total'] = myvar.sum(axis=1)

corr = myvar.corr()

# plot the heatmap

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.5, square=True)
myvar['Total'].corr(myvar['Fee'])
fig, axs = plt.subplots(2, 1)



#Con datos del dataset train

#data_x = myvar.drop('Fee', axis=1)

data_x = myvar[['Total']].copy()

data_y = myvar[['Fee']].copy()

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, random_state=42)

lm = LinearRegression()

lm.fit(train_x, train_y)

pred = lm.predict(test_x)

axs[0].scatter(test_y, pred)

print('Puntuación usando train:', model_score(test_y, pred))

#Con datos del dataset test

lm2 = LinearRegression()

lm2.fit(data_x, data_y)

real_test = test[['Dewormed','Vaccinated','Sterilized']].copy()

real_test['Total'] = real_test.sum(axis=1)

pred2 = lm2.predict(real_test[['Total']].copy())

axs[1].scatter(test[['Fee']], pred2, color='red')

print('Puntuación usando test:', model_score(test[['Fee']], pred2))

animals2 = animals._get_numeric_data()

x_train, x_test = train_test_split(animals2, test_size=0.3, random_state=1)

gnb = bayes_model()

used_features = list(animals2)[:len(list(animals2))-2]

print(used_features)

# Train classifier

gnb.fit(

    x_train[used_features].values,

    x_train["AdoptionSpeed"]

)

pred = gnb.predict(x_test[used_features])

print('Puntuación usando test:', model_score(x_test['AdoptionSpeed'], pred))

print(classification_report(x_test['AdoptionSpeed'], pred))

print(confusion_matrix(x_test['AdoptionSpeed'], pred))
lambdafunc = lambda x: pd.Series([1 if x['AdoptionSpeed']==0 else 0,1 if x['AdoptionSpeed']==1 else 0, 1 if x['AdoptionSpeed']==2 else 0, 1 if x['AdoptionSpeed']==3 else 0, 1 if x['AdoptionSpeed']==4 else 0])

animals[['var0','var1', 'var2', 'var3','var4']] = animals.apply(lambdafunc, axis=1)

categoric_animals = animals.drop(['Name', 'Description', 'PetID', 'RescuerID', 'VideoAmt', 'State', 'Vaccinated', 'Quantity'], axis=1)

categoric_animals.head()
no_var_animals = categoric_animals.drop(['var0', 'var1', 'var2', 'var3', 'var4', 'AdoptionSpeed'], axis=1)

var_animals = categoric_animals[['var0', 'var1', 'var2', 'var3', 'var4', 'AdoptionSpeed']].copy() #AdoptionSpeed se omitirá más adelante



x_train, x_test, y_train, y_test = train_test_split(no_var_animals, var_animals, test_size=0.20, random_state=114)



var_col_names = var_animals.columns.values

var_col_names = var_col_names[0: len(var_col_names)-1] #Se borra AdoptionSpeed de los nombres de las columnas sobre las que se correra el algoritmo



all_predictions = pd.DataFrame()

for col in var_col_names:  

    lg = LogisticRegression(solver='liblinear')

    lg.fit(x_train, y_train[[col]].values.ravel())

    prediction = lg.predict_proba(x_test)

    all_predictions[col] = pd.Series(np.array(prediction)[:, 1])

    

def bigger_probability(row):

    selected = 0

    for i in range(1, len(row)):

        if row[i] > row[selected]:

            selected = i

    return selected



result = all_predictions.apply(lambda x: bigger_probability(x), axis=1).values



score = accuracy_score(y_test['AdoptionSpeed'], result)

cm = confusion_matrix(y_test['AdoptionSpeed'], result)



plt.figure(figsize=(5,5))

sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Oranges');

plt.ylabel('Categorías Reales');

plt.xlabel('Categorías Predichas');

title = 'Precisión: {}'.format(round(score, 3))

plt.title(title, size = 15);
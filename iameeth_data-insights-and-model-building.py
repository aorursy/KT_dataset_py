from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 6



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

%config InlineBackend.figure_format = 'retina'

sns.set_style("whitegrid")





from pandas.tools.plotting import scatter_matrix

from sklearn import cross_validation

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC



from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline
df = pd.read_csv('../input/maintenance_data.csv')

df.head()

df.shape
df.groupby('provider').broken.value_counts()

df.groupby('team').broken.value_counts()
ax=sns.violinplot(x="provider", y="lifetime", hue="broken",split=True, inner="stick",data=df,palette="Set1")
ax=sns.violinplot(x="team", y="lifetime", hue="broken", inner="stick",split=True,data=df,palette="Set1")
ax=sns.swarmplot(x="team", y="lifetime", hue="broken",data=df)
ax=sns.distplot(df['temperatureInd'])
ax=sns.distplot(df['pressureInd'])
ax=sns.distplot(df['moistureInd'])
g = sns.FacetGrid(df, col="provider", hue="broken")

g.map(plt.scatter, "temperatureInd", "pressureInd", alpha=.7)

g.add_legend()
g = sns.FacetGrid(df, col="team", hue="broken")

g.map(plt.scatter, "temperatureInd", "pressureInd", alpha=.7)

g.add_legend()
sns.jointplot(x= "temperatureInd", y= "pressureInd", kind="hex", color="k",data=df)
f, ax = plt.subplots(figsize=(10, 10))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.temperatureInd, df.pressureInd, cmap=cmap, n_levels=1000, shade=True)
f, ax = plt.subplots(figsize=(10, 10))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.temperatureInd, df.moistureInd, cmap=cmap, n_levels=1000, shade=True)
f, ax = plt.subplots(figsize=(10, 10))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.pressureInd, df.moistureInd, cmap=cmap, n_levels=1000, shade=True)
sns.regplot(x="temperatureInd", y="pressureInd", data=df)
sns.lmplot(x="temperatureInd", y="pressureInd", hue="broken", data=df,markers=["o", "x"], palette="Set1",size=10)
sns.pairplot(df, x_vars=["temperatureInd", "pressureInd","moistureInd"], y_vars=["lifetime"],

             hue="broken", size=7, aspect=1, kind="reg")
data=df[['moistureInd','temperatureInd','pressureInd','broken']]

data.shape

dataset=data.values

X = dataset[:,0:3].astype(float)

Y = dataset[:,3]

#Splitting the dataset into train and validation sets

validation_size = 0.30

seed = 21

X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)



num_folds = 10

num_instances = len(X_train)

seed = 3





models = []

models.append(('Logistic Regression', LogisticRegression()))

models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART Decision Tree', DecisionTreeClassifier()))

models.append(('Naive Bayes', GaussianNB()))

models.append(('SVM', SVC()))

# evaluate each model

results = []

names = []

for name, model in models:

    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "model = %s:\n mean = %f std = (%f)\n" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# create model

model = Sequential()

model.add(Dense(20, input_dim=3,activation='relu'))

model.add(Dense(18, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(14, activation='relu'))

model.add(Dense(12, activation='relu'))

model.add(Dense(1, activation='softmax'))



optimizers=['adam']



accu={}

for myoptimizer in optimizers:

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer=myoptimizer, metrics=['accuracy'])

    # Fit the model

    model.fit(X_train,Y_train, nb_epoch=1000,verbose=0, batch_size=10)

    # evaluate the model

    scores = model.evaluate(X_validation, Y_validation)

    accu[myoptimizer]=scores[1]*100

print ("\n",accu)
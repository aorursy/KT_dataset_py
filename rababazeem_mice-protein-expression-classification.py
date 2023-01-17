# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn as sk



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



mice = pd.read_csv('../input/Data_Cortex_Nuclear.csv')



# Any results you write to the current directory are saved as output.
mice.head(8)
mice.describe()
mice.isnull().sum()
# counting null values by row

mice.isnull().sum(axis=1)
#Dropping and filling null values

nmice = mice.dropna(how='any', thresh=75)



nmice = nmice.fillna(nmice.mean())
nmice.isnull().sum(axis=1)
# checking if dropping the rows worked

nmice.isnull().sum()
# looking for changes

# everything increased a bit

nmice.describe()
# Use a feature selection to find the important proteins for predicting the treatment, behavior and class



# Assgning the data and target

X = nmice.loc[:, 'DYRK1A_N':'CaNA_N']

y = nmice['class']
# Splitting the train and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
#Creating a random forest classifier

clf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=1)



# Train the classifier

clf.fit(X_train, y_train)



# assigning the all importance values to a series

importance = pd.Series(clf.feature_importances_)



# Print the name and gini importance of each feature

for feature in zip(nmice.loc[: ,'DYRK1A_N':'CaNA_N'], clf.feature_importances_):

    print(feature)
# Test the classifer to get the accuracy

y_pred = clf.predict(X_test)



accuracy_score(y_test, y_pred)

#Let's use a threshold to find the most important proteins



# Create a selector object that will use the random forest classifier to identify

# features that have an importance of more than 0.014

# accuracy of limited feature model gets lower if I do lower than 0.014 or higher

sfm = SelectFromModel(clf, threshold=0.014)



# Train the selector

sfm.fit(X_train, y_train)



# making arrays to keep track of the important proteins' importance values and index

imp = []

imp_index = []



# Print the names of the most important features

for feature_list_index in sfm.get_support(indices=True):

    imp.append(list(nmice.loc[:, 'DYRK1A_N':'CaNA_N'])[feature_list_index])

    print(list(nmice.loc[:, 'DYRK1A_N':'CaNA_N'])[feature_list_index])

    imp_index.append(feature_list_index)



# Making the list of important proteins so I can make a dataframe with it

s_imp = pd.Series(imp)



# Getting the importance values of only the important proteins

protein_importance = (importance)[imp_index]



# Making a dataframe of the proteins with thei importance values

p = {'Important Proteins': imp, 'Importance Value': protein_importance}

df_importance = pd.DataFrame(p)
# sorting the proteins by importance and printing it

df_importance.sort_values('Importance Value', ascending=False, inplace=True)

print(df_importance)
# ceating a new data set from the sfm model to test the accuracy of a limited feature model

X_imp_train = sfm.transform(X_train)

X_imp_test = sfm.transform(X_test)
# creating a new random forest model classifier for the most important features

clf_imp = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=1)



# training the random forest classfier on the new limited feature data sets

clf_imp.fit(X_imp_train, y_train)
# Test the classifer to get the accuracy

y_imp_pred = clf_imp.predict(X_imp_test)



# Getting the accuracy score of the limited classifier model

accuracy_score(y_test, y_imp_pred)



# There is almost no difference between the accuracies of the 2 models 
# creating a data frame of the values of only the important proteins for plotting

protein_ex = nmice.loc[:, imp]

protein_ex



# creating a dataframe with only the the types of mice

description = nmice.loc[:, ['Genotype', 'Treatment','Behavior', 'class']]

description



# joining the two dataframes together

protein_exd = protein_ex.join(description)

protein_exd
yg = protein_exd['Genotype']



# Splitting the train and testing data

yg_train, yg_test = train_test_split(yg, test_size=0.4, random_state=1)



#making a new random forest classifier with a max_dept so I make a smaller visual of a tree

clf_imptd = RandomForestClassifier(max_depth = 3, n_estimators=100, random_state=1, n_jobs=1)



# training the random forest classfier on the new limited feature data sets

clf_imptd.fit(X_imp_train, yg_train)
# Test the classifer to get the accuracy

yg_imp_pred = clf_imptd.predict(X_imp_test)



# Getting the accuracy score of the limited classifier model

accuracy_score(yg_test, yg_imp_pred)
#Visualizing the model

from sklearn.tree import export_graphviz

# Export as dot file

export_graphviz(clf_imptd.estimators_[5], out_file='tree.dot', 

                feature_names = imp,

                class_names = ['Control', 'Trisomal'],

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')
# Creating the test and train data for the genotype

protein_exd['Genotype'] = protein_exd['Genotype'].map({'Control': 0, 'Ts65Dn': 1})



yg = protein_exd['Genotype']



# Splitting the train and testing data

yg_train, yg_test = train_test_split(yg, test_size=0.4, random_state=1)



# making a new random forest classifer model that uses only the top 3 important proteins for the visualization

clf_impt = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=1)



# training the random forest classfier on the new limited feature data sets

clf_impt.fit(X_imp_train, yg_train)
# Test the classifer to get the accuracy

yg_imp_pred = clf_impt.predict(X_imp_test)



# Getting the accuracy score of the limited classifier model

accuracy_score(yg_test, yg_imp_pred)
#Making a ROC curve 

# Making a visual for the ROC curve for the logistic regression model

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc = roc_auc_score(yg_test, clf_impt.predict(X_imp_test))

fpr, tpr, thresholds = roc_curve(yg_test, clf_impt.predict_proba(X_imp_test)[:,1:26])

plt.figure(figsize=(12,8))

plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlabel('False Positive Rate', fontsize=14)

plt.ylabel('True Positive Rate', fontsize=14)

plt.title('Receiver Operating Characteristic', fontsize=16)

plt.legend(loc="lower right", fontsize=14)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.style.use('seaborn')

plt.show()
# Making a dataframe of the accuracies

a = {'Limited': [0.974477958236659], 'Full': [0.974477958236659]}

accuracies = pd.DataFrame(data=a)

accuracies



# making bar plot comparing the accuracies of the models

ax = accuracies.plot.bar(

    figsize= (13, 10),

    fontsize=15)

ax.set(ylabel = 'Accuracy')

ax.set(xlabel = 'Random Forest Models')

ax.set_xticklabels('', fontsize=14)

plt.legend(fontsize=14)

x_labels = ['A', 'B']

xticks = [-0.13, 0.13]

ax.set_xticks(xticks)

ax.set_xticklabels(x_labels, rotation=0, fontsize=14)

ax.set_facecolor('xkcd:white')

ax.set_facecolor(('#ffffff'))

ax.spines['left'].set_color('black')

ax.spines['bottom'].set_color('black')
# making violin plots of each protein for each group; treatment and control

fig, axs = plt.subplots(1, 5, figsize=(60, 30))



sns.set(font_scale=1.5)



sns.set_style('whitegrid')



sns.violinplot(

    y='SOD1_N',

    x='Genotype',

    data=protein_exd,

    palette='Set2',

    ax=axs[0]

    )



sns.violinplot(

    y='CaNA_N',

    x='Genotype',

    data=protein_exd,

    palette='Set2',

    ax=axs[1]

    )



sns.violinplot(

    y='Ubiquitin_N',

    x='Genotype',

    data=protein_exd,

    palette='Set2',

    ax=axs[2]

    )



sns.violinplot(

    y='APP_N',

    x='Genotype',

    data=protein_exd,

    palette='Set2',

    ax=axs[3]

    )



sns.violinplot(

    y='pERK_N',

    x='Genotype',

    data=protein_exd,

    palette='Set2',

    ax=axs[4]

    )
# Making an instance of the model

lr = LogisticRegression()



# fitting the model to the training data

lr.fit(X_imp_train, yg_train)



# use the model to predict on the testing data

lr.predict(X_imp_test)



# Printing the accuracy of the model

score = lr.score(X_imp_test, yg_test)

print(score)
# Making a visual for the ROC curve for the logistic regression model

logit_roc_auc = roc_auc_score(yg_test, lr.predict(X_imp_test))

fpr, tpr, thresholds = roc_curve(yg_test, lr.predict_proba(X_imp_test)[:,1:26])

plt.figure(figsize=(12, 8))

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlabel('False Positive Rate', fontsize=14)

plt.ylabel('True Positive Rate', fontsize=14)

plt.title('Receiver Operating Characteristic', fontsize=16)

plt.legend(loc="lower right", fontsize=14)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.style.use('seaborn')

plt.show()
# Making a violing plot of the top 5 important proteins by each class

sns.set(rc={'figure.figsize':(20,8)})

sns.set_style('ticks')

sns.set(font_scale = 1.75)

g = sns.violinplot(

    y='SOD1_N',

    x='class',

    data=protein_exd

    )

g.set_facecolor('xkcd:white')

g.spines['left'].set_color('black')

g.spines['bottom'].set_color('black')
sns.set(rc={'figure.figsize':(20,8)})

sns.set_style('ticks')

sns.set(font_scale = 1.75)

g = sns.violinplot(

    y='pPKCG_N',

    x='class',

    data=protein_exd

    )

g.set_facecolor('xkcd:white')

g.spines['left'].set_color('black')

g.spines['bottom'].set_color('black')
sns.set(rc={'figure.figsize':(20,8)})

sns.set_style('ticks')

sns.set(font_scale = 1.75)

g = sns.violinplot(

    y='CaNA_N',

    x='class',

    data=protein_exd

    )

g.set_facecolor('xkcd:white')

g.spines['left'].set_color('black')

g.spines['bottom'].set_color('black')
sns.set(rc={'figure.figsize':(20,8)})

sns.set_style('ticks')

sns.set(font_scale = 1.75)

g = sns.violinplot(

    y='Ubiquitin_N',

    x='class',

    data=protein_exd

    )

g.set_facecolor('xkcd:white')

g.spines['left'].set_color('black')

g.spines['bottom'].set_color('black')
sns.set(rc={'figure.figsize':(20,8)})

sns.set_style('ticks')

g = sns.violinplot(

    y='DYRK1A_N',

    x='class',

    data=protein_exd

    )
# Applying a logistic regression model to predict the classes



# Making an instance of the model

lr = LogisticRegression()



# fitting the model to the training data

lr.fit(X_imp_train, y_train)



# use the model to predict on the testing data

lr.predict(X_imp_test)



# Printing the accuracy of the model

score = lr.score(X_imp_test, y_test)

print(score)
# importing packages

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier



# creating an instance of the kNN model

# n_jobs=-1 makes it so that computations run in parallel

kmodel = KNeighborsClassifier(n_jobs=-1)



# setting the parameters I want to test

params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10],

          'leaf_size':[1,2,3,5],

          'weights':['uniform', 'distance'],

          'algorithm':['auto', 'ball_tree','kd_tree','brute'],

          'n_jobs':[-1]}



# creating a grid search with the parameters I chose

grid = GridSearchCV(kmodel, param_grid=params, n_jobs=1, scoring='accuracy')



# fitting the model

grid.fit(X_imp_train, y_train)



#print the best combination of parameters

print("Best Hyper Parameters:\n",grid.best_params_)
#getting the accuracy of the kNN model

pred = grid.predict(X_imp_test)



print('Accuracy:', accuracy_score(pred, y_test))
#Making a new model to predict genotypes to make an ROC curve

kgmodel = KNeighborsClassifier(n_jobs=-1)

kgmodel.fit(X_imp_train, yg_train)



#Getting the accuracy

pred_gen = kgmodel.predict(X_imp_test)



print('Accuracy:', accuracy_score(pred_gen, yg_test))
# Making a visual for the ROC curve for the KNN model

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc = roc_auc_score(yg_test, kgmodel.predict(X_imp_test))

fpr, tpr, thresholds = roc_curve(yg_test, kgmodel.predict_proba(X_imp_test)[:,1:26])

plt.figure(figsize=(12, 8))

plt.plot(fpr, tpr, label='K-Nearest Neighbours (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlabel('False Positive Rate', fontsize=14)

plt.ylabel('True Positive Rate', fontsize=14)

plt.title('Receiver Operating Characteristic', fontsize=16)

plt.legend(loc="lower right", fontsize=14)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.style.use('seaborn')

plt.show()
#do subplots with correlations between top 5 proteins and mice descriptions

from pylab import rcParams



sns.set(font_scale=1.3)

rcParams['figure.figsize'] = (20, 20)

cols = ['SOD1_N', 'Ubiquitin_N', 'CaNA_N']

x = protein_exd[['SOD1_N', 'Ubiquitin_N', 'CaNA_N']]

sns_plot = sns.pairplot(x[cols])
# making a heatmap of the correlation

sns.set(font_scale=3)

def plot_corr( df ):

    corr = df.corr()

    _, ax=plt.subplots( figsize=(50,25) )

    cmap = sns.diverging_palette( 240 , 10 , as_cmap = True)

    _ = sns.heatmap(corr,cmap=cmap,square=True, cbar_kws = {'shrink': .9}, ax=ax, annot=False)

    

plot_corr(protein_exd)
import matplotlib.font_manager

matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')



# Making a dataframe of the accuracies

a = {'Random Forest Classifier': [0.974477958236659], 'K-Nearest Neighbours': [0.974477958236659], 'Logistic Regression': [0.740139211136891]}

accuracies = pd.DataFrame(data=a)

#accuracies.rename(index={0:'Random Forest Classifier',1:'K-Nearest Neighbours', 2:'Logistic Regression'}, 

#                 inplace=True)



# making bar plot comparing the accuracies of the models

sns.set(font_scale=1)

ax = accuracies.plot.bar(

    figsize= (13, 10),

    fontsize=14)

plt.xticks(rotation=0, fontsize=14)

plt.xlabel('Models', fontsize=14)

plt.ylabel('Accuracy', fontsize=14)

x_labels = ['A', 'B', 'C']

xticks = [-0.17, 0,0.165]

ax.set_xticks(xticks)

ax.set_xticklabels(x_labels, rotation=0)

axbox = ax.get_position()

plt.legend(loc = (axbox.x0 + 0.65, axbox.y0 + 0.70), fontsize=14)

plt.title(' ')

ax.set_facecolor('xkcd:white')

ax.set_facecolor(('#ffffff'))

ax.spines['left'].set_color('black')

ax.spines['bottom'].set_color('black')
#Making a bar plot comparing all the accuracies of the models that predict genotype



# Making a dataframe of the accuracies

a = {'Random Forest Classifier': [0.9721577726218097], 'K-Nearest Neighbours': [0.9164733178654292], 'Logistic Regression': [0.8167053364269141]}

accuracies = pd.DataFrame(data=a)

#accuracies.rename(index={0:'Random Forest Classifier',1:'K-Nearest Neighbours', 2:'Logistic Regression'}, 

#                 inplace=True)



# making bar plot comparing the accuracies of the models

ax = accuracies.plot.bar(

    figsize= (13, 10),

    fontsize=14)

plt.xticks(rotation=0, fontsize=14)

plt.xlabel('Models', fontsize=14)

plt.ylabel('Accuracy', fontsize=14)

x_labels = ['A', 'B', 'C']

xticks = [-0.17, 0,0.165]

ax.set_xticks(xticks)

ax.set_xticklabels(x_labels, rotation=0)

axbox = ax.get_position()

plt.legend(loc = (axbox.x0 + 0.65, axbox.y0 + 0.70), fontsize=14)

# plt.title('The Accuracies of the Models that Predict Mice Genotype', fontsize=13.8, y=0.96, x=0.81)

plt.title(" ")

ax.set_facecolor('xkcd:white')

ax.set_facecolor(('#ffffff'))

ax.spines['left'].set_color('black')

ax.spines['bottom'].set_color('black')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will 

# list the files in the input directory

import time

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')

print(data.columns)
columns = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']

data = data[columns]

data['diagnosis'] = data['diagnosis'].map({'M':0,'B':1})

data.head()

data.describe()
# Draw a count of observations by diagnosis

sns.set(rc={'figure.figsize':(7,3)})

g = sns.countplot(x="diagnosis",data = data, palette="muted")

# Add labels  

for p in g.patches:

    g.annotate(round(p.get_height()), (p.get_x(), p.get_height()),ha='center',va='center',xytext=(85, 5),

               textcoords='offset points')

sns.set(font_scale=1.4)

plt.title('Figure 1: Distribution of Observations \n', fontweight = 'bold')

plt.xlabel('Diagnosis: 0= Malignant, 1= Benign')

plt.ylabel('Number of Observations')

plt.show()
# Boxplot for entire dataset

sns.set(rc={'figure.figsize':(20,4)})

sns.set(font_scale=2)



g = sns.boxplot(data = data, palette="muted")

plt.xticks(rotation=90)

plt.title('Figure 2: Distribution of all the attributes using Entire Dataset\n', fontweight = 'bold')

plt.ylabel('Distributions of Values')

plt.show()



# Boxplot for Malign Class 

g = sns.boxplot(data = data[data["diagnosis"]==0], palette="muted")

plt.xticks(rotation=90)

plt.title('Figure 3: Distribution of all the attributes for Malignant Class\n', fontweight = 'bold')

plt.ylabel('Distributions of Values')

plt.show()



#Boxplot for Benign Class

g = sns.boxplot(data = data[data["diagnosis"]==1], palette="muted")

plt.xticks(rotation=90)

plt.title('Figure 4: Distribution of all the attributes for benign Class\n', fontweight = 'bold')

plt.ylabel('Distributions of Values')

plt.show()
# Heatmap of Coefficient Correlation 

#g = sns.clustermap(data.corr(),linewidth=1,col_cluster=False,

 #                 row_cluster=False,linecolor="grey",annot=True,figsize=(35,35),square=False)

sns.set(font_scale=2)

#cmap = sns.diverging_palette(1, 300, as_cmap=True)

fig, ax = plt.subplots(figsize=(25,25))   

g = sns.heatmap(data.iloc[:,1:29].corr(),linewidth=1,annot=True,vmax=1,

                square=True,linecolor='black',cmap="Blues",annot_kws={"size": 12})

#plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.title('Figure 5: Pearson Correlation Plot\n', fontweight = 'bold')

plt.show()
features = ['radius_mean', 'texture_mean', 

       'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se',  'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

        'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']



train_data = data[features]

train_scaled = preprocessing.scale(train_data)



pca = PCA(n_components=24)

pca.fit(train_scaled)



#The amount of variance that each PC explains

var= pca.explained_variance_ratio_



#Cumulative Variance explains

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)



sns.set(font_scale=2)

print(var1)

plt.plot(var1,'--o')

plt.title('Figure 6: Principal Components\n', fontweight = 'bold')

plt.show()
# Looking at above plot I'm taking 4 variables

# fits PCA, transforms data and fits the RF classifier

# on the transformed data



X_train, X_test, y_train, y_test = train_test_split(train_data, data["diagnosis"], 

                                 test_size=0.5, stratify=data["diagnosis"], random_state=123456)

# Randon Forest

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)

rf.fit(X_train, y_train)



predicted = rf.predict(X_test)

accuracy = accuracy_score(y_test, predicted)

print(f'RF - Mean accuracy score: {accuracy:.3}')

print(f'RF - f1 score: {f1_score(y_test, predicted, average="macro"):.3}')

print(f'RF - precision: {+precision_score(y_test, predicted, average="macro"):.3}')

print(f'RF - recall: {recall_score(y_test, predicted, average="macro"):.3}')   



sns.set(font_scale=1)

fig, ax = plt.subplots(figsize=(2,2))   

cm = pd.DataFrame(confusion_matrix(y_test, predicted))

sns.heatmap(cm, annot=True)

plt.title('Figure 7: Confusion Matrix plot of RF Predictions\n', fontweight = 'bold')



  

# PCA and RandomForest

X_train_scaled = preprocessing.scale(X_train)

X_test_scaled =  preprocessing.scale(X_test)



pca = PCA(n_components=5)

pca_train = pca.fit_transform(X_train_scaled)

pca_test = pca.fit_transform(X_test_scaled)





rf = RandomForestClassifier(n_estimators=24, oob_score=True, random_state=123456)

rf.fit(pca_train, y_train)



predicted = rf.predict(pca_test)

accuracy = accuracy_score(y_test, predicted)

print(f'\nPCA & RF - Mean accuracy score: {accuracy:.3}')

print(f'PCA & RF - f1 score: {f1_score(y_test, predicted, average="macro"):.3}')

print(f'PCA & RF - precision: {+precision_score(y_test, predicted, average="macro"):.3}')

print(f'PCA & RF - recall: {recall_score(y_test, predicted, average="macro"):.3}')  



sns.set(font_scale=1)

fig, ax = plt.subplots(figsize=(2,2)) 

cm = pd.DataFrame(confusion_matrix(y_test, predicted))

sns.heatmap(cm, annot=True)

plt.title('Figure 8: Confusion Matrix plot of RF with PCA\n', fontweight = 'bold')



# Neural Network 



nn = MLPClassifier(

    activation = 'logistic',

    solver = 'lbfgs',

    alpha=1e-5,

    early_stopping=False,

    hidden_layer_sizes=(100),

    random_state=123456,

    batch_size='auto',

    max_iter=1000,

    learning_rate_init=0.05

)

nn.fit(X_train, y_train)



predicted = nn.predict(X_test)

accuracy = accuracy_score(y_test, predicted)

print(f'\nNN - Mean accuracy score: {accuracy:.3}')

print(f'NN - f1 score: {f1_score(y_test, predicted, average="macro"):.3}')

print(f'NN - precision: {+precision_score(y_test, predicted, average="macro"):.3}')

print(f'NN - recall: {recall_score(y_test, predicted, average="macro"):.3}')  



sns.set(font_scale=1)

fig, ax = plt.subplots(figsize=(2,2)) 

cm = pd.DataFrame(confusion_matrix(y_test, predicted))

sns.heatmap(cm, annot=True)

plt.title('Figure 9: Confusion Matrix plot of NN\n', fontweight = 'bold')



# PCA and NN

pca = PCA(n_components=5)

pca_train = pca.fit_transform(X_train_scaled)

pca_test = pca.fit_transform(X_test_scaled)



nn = MLPClassifier(

    activation = 'logistic',

    solver = 'lbfgs',

    alpha=1e-5,

    early_stopping=False,

    hidden_layer_sizes=(100),

    random_state=123456,

    batch_size='auto',

    max_iter=1000,

    learning_rate_init=0.05

)

nn.fit(pca_train, y_train)



predicted = nn.predict(pca_test)

accuracy = accuracy_score(y_test, predicted)

print(f'\nPCA & NN - Mean accuracy score: {accuracy:.3}')

print(f'PCA & NN - f1 score: {f1_score(y_test, predicted, average="macro"):.3}')

print(f'PCA & NN - precision: {+precision_score(y_test, predicted, average="macro"):.3}')

print(f'PCA & NN - recall: {recall_score(y_test, predicted, average="macro"):.3}')  



sns.set(font_scale=1)

fig, ax = plt.subplots(figsize=(2,2)) 

cm = pd.DataFrame(confusion_matrix(y_test, predicted))

sns.heatmap(cm, annot=True)



plt.title('Figure 10: Confusion Matrix plot of NN with PCA\n', fontweight = 'bold')





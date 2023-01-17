# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import seaborn as sns

from scipy import stats

# Any results you write to the current directory are saved as output.
#Nina Petreska 161092

import matplotlib.pyplot as plt #Libraries for visualization

from mpl_toolkits.mplot3d import Axes3D

from scipy import stats #Library for statistical analysis 

from sklearn import datasets #Libraries for machine learning

from sklearn.naive_bayes import GaussianNB 

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

%matplotlib notebook

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
pokemons = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')



X = pokemons.drop('Legendary', axis=1)

X = X.drop('Name', axis = 1)

X = X.drop('#', axis=1)



target_names = np.unique(pokemons['Legendary'])

feature_names = list(X.columns.values)



pokemons['Legendary'] = pokemons['Legendary'].map({False : 0, True : 1})

print(np.unique(pokemons['Type 1']))



#Tekstualnite podatoci se zamenivaat so integers

pokemons['Type 1'] = pokemons['Type 1'].map({'Bug' : 0, 'Dark' : 1,'Dragon' : 2,'Electric': 3, 'Fairy' : 4, 'Fighting':5, 'Fire':6, 'Flying':7,'Ghost':8 ,'Grass':9 ,'Ground':10, 'Ice':11, 'Normal':12, 'Poison':13, 'Psychic':14, 'Rock':15 ,'Steel':16,'Water':17,float('nan'):18 })

pokemons['Type 2'] = pokemons['Type 2'].map({'Bug' : 0, 'Dark' : 1,'Dragon' : 2,'Electric': 3, 'Fairy' : 4, 'Fighting':5, 'Fire':6, 'Flying':7,'Ghost':8 ,'Grass':9 ,'Ground':10, 'Ice':11, 'Normal':12, 'Poison':13, 'Psychic':14, 'Rock':15 ,'Steel':16,'Water':17,float('nan'):18 })

print(pokemons) 
p = pokemons.drop('Name', axis = 1)

p = p.drop('#',axis = 1)



#Ke probam bez ovie

pp = p.drop(['Type 1','Type 2','HP','Attack','Defense','Generation'],axis = 1)



#I ke probam bez types i bez generation, bidejki mislam deka tie se ramnomerno raspredelneni skoro

p_p = p.drop(['Type 1','Type 2','Generation'],axis=1)



#bez total isto kako tretoto

p4 = p.drop(['Type 1','Type 2','Generation','Total'],axis=1)



#izmenetiot fajl go zacuvuvam vo nov

p.to_csv("Pokemons_Nina.csv", sep=',', index=False)

pp.to_csv("P.csv", sep=',', index=False)

p_p.to_csv("poke.csv",sep= ',',index=False)

p4.to_csv('p4.csv',sep = ',', index = False)
column_names = np.loadtxt(open("Pokemons_Nina.csv", "rb"), delimiter=",", max_rows=1, dtype = str)

pokemon = np.loadtxt(open("Pokemons_Nina.csv", "rb"), delimiter=",", skiprows=1, dtype = str)

y = pokemon[:,-1].astype(np.int)

X = pokemon[:,:-1].astype(np.float)



number_of_features = X.shape[1]

number_of_classes = len(np.unique(y))



print("Number and names of classes:", len(np.unique(y)), target_names) 

print("Number and names of features:", len(np.unique(feature_names)), feature_names) 

print("Number of data points:", X.shape[0])
print(column_names)
colors = ['crimson', 'turquoise']

print(feature_names)

#X[feature_names[0]]

print(X)
fig = plt.figure(figsize=(10, 16))

fig.subplots(nrows=5, ncols=2)

for feat_i in range(number_of_features): #For each feature, we have a new subplot

    ax = plt.subplot(5,2, feat_i+1)

    f = feature_names[feat_i]

    plt.title(feature_names[feat_i])

    sns.distplot(X[:,feat_i])

    for class_i in range(number_of_classes): #After that we draw the within-class histograms of the same feature

        sns.distplot(X[y == class_i,feat_i], color=colors[class_i], label=target_names[class_i])

    plt.legend()

plt.show()



#Jas ke gi koristam Total, Sp.Atk, Sp.Def , Speed bidejkispored vzuelnite pretstavi izgledaat kako pobitni
fig = plt.figure(figsize=(15, 40))

plt.title("Scatterplots of the Pokemon dataset features")

fig.subplots(nrows=10, ncols=10)

for feat_i in range(number_of_features): #We go over all pairs of features (4x4 in this case)

    for feat_j in range(number_of_features):  

        ax = plt.subplot(10,10,10*feat_i + feat_j+1)

        # Plot the points

        for color, i, target_name in zip(colors, [0, 1], target_names):

            plt.scatter(X[y == i, feat_i], X[y == i, feat_j], alpha=.8, color=color, label=target_name) #We again extract the feature class specific data using the same method as before and then just use the scatter function

        plt.xticks(())

        plt.yticks(())

        plt.title("Feature "+str(feat_i)+" x Feature "+str(feat_j))

plt.show()
correlation_matrix = np.zeros((number_of_features,number_of_features))

for i in range(number_of_features): #We need a 10x10 matrix to represent the correlation matrix, where we set the value of Cij to be the correlation between the i'th and the j'th metric

    measure = X[:,i]

    for j in range(number_of_features):

        measure2 = X[:,j]

        corr, _ = stats.pearsonr(measure, measure2)

        correlation_matrix[i][j] = corr

plt.figure()

plt.imshow(correlation_matrix, cmap = "inferno") #We can draw the matrix using imshow

plt.colorbar()

plt.show()
pca = PCA(n_components=2) 

X_PCA = pca.fit(X).transform(X) 



plt.figure() 

for color, i, target_name in zip(colors, [0, 1, 2], target_names):

    plt.scatter(X_PCA[y == i, 0], X_PCA[y == i, 1], color=color, alpha=.8, lw=2,

                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('PCA of Pokemon dataset')

plt.show()
#Bidejki imam samo 2 klasi, nema moznost za 2D vizuelizacija

#Ovaa funkcija za binaren problem e 1D :(

lda = LDA(n_components=2)

lda = lda.fit(X, y) 

X_LDA = lda.transform(X)

#print(X_LDA[0:10])

plt.figure() 



for color, i, target_name in zip(colors, [0, 1], target_names):

    plt.scatter(X_LDA[y == i,0], X_LDA[y == i,0], alpha=.8, color=color, label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('LDA of Pokemon dataset')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42) #We split the original dataset (we use a fixed random state such as 42 so we always split the data the same way)

X_PCA_train, X_PCA_test, y_PCA_train, y_PCA_test = train_test_split(X_PCA, y, test_size=0.30, random_state=42) #We split the PCA dimensionaly reducted dataset

X_LDA_train, X_LDA_test, y_LDA_train, y_LDA_test = train_test_split(X_LDA, y, test_size=0.30, random_state=42) #We split the LDA dimensionaly reducted dataset
#Initialize the model, fit the training data and predict the test data

lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)

lda.fit(X_LDA_train, y_LDA_train)

y_pred = lda.predict(X_LDA_test)

print("LDA accuracy for the LDA dimensionaly reducted Pokemon dataset", np.round(np.sum(y_LDA_test == y_pred)/len(y_LDA_test),3))



#Visualize the test set and the errors which have occured

plt.figure()

for color, i, target_name in zip(colors, [0, 1, 2], target_names):

    plt.scatter(X_LDA_test[y_LDA_test == i, 0], X_LDA_test[y_LDA_test == i, 0], alpha=.8, color=color,

                label=target_name)

plt.title('LDA classification: LDA of Pokemon dataset')



incorrect = y_pred!=y_LDA_test

for i in range(len(incorrect)):

    if(incorrect[i]==True):

        plt.scatter(X_LDA_test[i][0], X_LDA_test[i][0], alpha=.5, color="black")

plt.legend(loc='best', shadow=False, scatterpoints=1)
gnb = GaussianNB() 

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy = np.round(np.sum(y_test == y_pred)/len(y_test),3)

print("Naive Bayes accuracy for the original Pokemon dataset", accuracy)

print("Precision_score",precision_score(y_test, y_pred, average='macro'))

print("Recall_score",recall_score(y_test, y_pred, average='macro'))

print("F1_score",f1_score(y_test, y_pred, average='macro'))
#probuvam poveke opcii koi koloni da gi koristam

column_names_2 = np.loadtxt(open("P.csv", "rb"), delimiter=",", max_rows=1, dtype = str)

pokemon_2 = np.loadtxt(open("P.csv", "rb"), delimiter=",", skiprows=1, dtype = str)

y2 = pokemon_2[:,-1].astype(np.int)

X2 = pokemon_2[:,:-1].astype(np.float)



number_of_features_2 = X2.shape[1]

number_of_classes_2 = len(np.unique(y2))
print(X2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.30, random_state=42)



gnb = GaussianNB() 

gnb.fit(X2_train, y2_train)

y2_pred = gnb.predict(X2_test)

accuracy = np.round(np.sum(y2_test == y2_pred)/len(y2_test),3)

print("Naive Bayes accuracy for the smaller Pokemon dataset", accuracy)

#Iako gi odbrav tie koi najmnogu bea razdaleceni, sepak se mnogu malku atributi i dobiv polosi rezultati osven vo recall

print("Precision_score",precision_score(y2_test, y2_pred, average='macro'))

print("Recall_score",recall_score(y2_test, y2_pred, average='macro'))

print("F1_score",f1_score(y2_test, y2_pred, average='macro'))
column_names_3 = np.loadtxt(open("poke.csv", "rb"), delimiter=",", max_rows=1, dtype = str)

pokemon_3 = np.loadtxt(open("poke.csv", "rb"), delimiter=",", skiprows=1, dtype = str)

y3 = pokemon_3[:,-1].astype(np.int)

X3 = pokemon_3[:,:-1].astype(np.float)



number_of_features_3 = X3.shape[1]

number_of_classes_3 = len(np.unique(y3))
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.30, random_state=42)



gnb = GaussianNB() 

gnb.fit(X3_train, y3_train)

y3_pred = gnb.predict(X3_test)

accuracy = np.round(np.sum(y3_test == y3_pred)/len(y3_test),3)

print("Naive Bayes accuracy for the smaller Pokemon dataset", accuracy)

#VO tretiot obid, vo koj gi trgnav tie koi mi izgledaa ramnomerno raspredeleni so Naive Bayes dobiv podobri rezultati

#Ponatamu ke prodolzam da rabotam so ovie podatoci
print(target_names)
#Bidejki imam samo 2 klasi, nema moznost za 2D vizuelizacija

#Ovaa funkcija za binaren problem e 1D :(

lda3 = LDA(n_components=2)

lda3 = lda3.fit(X3, y3) 

X3_LDA = lda3.transform(X3)

plt.figure() 



for color, i, target_name in zip(colors, [0, 1], target_names):

    plt.scatter(X3_LDA[y == i,0], X3_LDA[y == i,0], alpha=.8, color=color, label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('LDA of Pokemon3 dataset')
pca3 = PCA(n_components=2) 

X3_PCA = pca3.fit(X3).transform(X3)  



plt.figure()

for color, i, target_name in zip(colors, [0, 1], target_names):

    plt.scatter(X3_PCA[y == i, 0], X3_PCA[y == i, 1], color=color, alpha=.8, lw=2,

                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.title('PCA of Pokemon dataset')

plt.show()



X3_train, X3_test, y_train, y3_test = train_test_split(X3, y3, test_size=0.30, random_state=42) #We split the original dataset (we use a fixed random state such as 42 so we always split the data the same way)

X3_PCA_train, X3_PCA_test, y3_PCA_train, y3_PCA_test = train_test_split(X3_PCA, y3, test_size=0.30, random_state=42) #We split the PCA dimensionaly reducted dataset

X3_LDA_train, X3_LDA_test, y3_LDA_train, y3_LDA_test = train_test_split(X3_LDA, y3, test_size=0.30, random_state=42) #We split the LDA dimensionaly reducted dataset
#LDA model

lda = LinearDiscriminantAnalysis()

lda.fit(X3_train, y3_train) 

y3_pred = lda.predict(X3_test) 

accuracy = np.round(np.sum(y3_test == y3_pred)/len(y3_test),3)

print("LDA accuracy for the Pokemon3 dataset", accuracy) 

print("Precision_score",precision_score(y3_test, y3_pred, average='macro'))

print("Recall_score",recall_score(y3_test, y3_pred, average='macro'))

print("F1_score",f1_score(y3_test, y3_pred, average='macro'))
#QDA Model

clf = QDA()

clf.fit(X3_train, y3_train)

y3_pred = clf.predict(X3_test) 

accuracy = np.round(np.sum(y3_test == y3_pred)/len(y3_test),3)

print("QDA accuracy for the Pokemon3 dataset", accuracy) 

print("Precision_score",precision_score(y3_test, y3_pred, average='macro'))

print("Recall_score",recall_score(y3_test, y3_pred, average='macro'))

print("F1_score",f1_score(y3_test, y3_pred, average='macro'))
#Naive Bayes

gnb = GaussianNB() 

gnb.fit(X3_train, y3_train)

y3_pred = gnb.predict(X3_test)

accuracy = np.round(np.sum(y3_test == y3_pred)/len(y3_test),3)

print("Naive Bayes accuracy for the smaller Pokemon dataset", accuracy)

print("Precision_score",precision_score(y3_test, y3_pred, average='macro'))

print("Recall_score",recall_score(y3_test, y3_pred, average='macro'))

print("F1_score",f1_score(y3_test, y3_pred, average='macro'))
print(X3)
df = pd.read_csv("poke.csv")

print(df)

c = df.corr()
print(c)
print(c[c >= 0.7])
print(c[c >= 0.7].stack().reset_index(name='cor').query("abs(cor) < 1.0"))
#Bidejki total e koreliran so site, ke probam i bez nego



#Otkako gi trgnav kolinearnite koloni, dobiv najdobar rezultat so QDA modelot

#Bidejki speak dosta se razlikuvaat legendarnite pokemons od obicnite, ne e mnogu tesko da se razgranicat

#so Site modeli dobiv dosta dobri rezultati
column_names_4 = np.loadtxt(open("p4.csv", "rb"), delimiter=",", max_rows=1, dtype = str)

pokemon_4 = np.loadtxt(open("p4.csv", "rb"), delimiter=",", skiprows=1, dtype = str)

y4 = pokemon_4[:,-1].astype(np.int)

X4 = pokemon_4[:,:-1].astype(np.float)



number_of_features_4 = X3.shape[1]

number_of_classes_4 = len(np.unique(y4))



pca4 = PCA(n_components=2) 

X4_PCA = pca4.fit(X4).transform(X4) 



lda4 = LDA(n_components=2)

lda4 = lda4.fit(X4, y4) 

X4_LDA = lda4.transform(X4)



X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.30, random_state=42) #We split the original dataset (we use a fixed random state such as 42 so we always split the data the same way)

X4_PCA_train, X4_PCA_test, y4_PCA_train, y4_PCA_test = train_test_split(X4_PCA, y4, test_size=0.30, random_state=42) #We split the PCA dimensionaly reducted dataset

X4_LDA_train, X4_LDA_test, y4_LDA_train, y4_LDA_test = train_test_split(X4_LDA, y4, test_size=0.30, random_state=42) #We split the LDA dimensionaly reducted dataset
#LDA model

lda = LinearDiscriminantAnalysis()

lda.fit(X4_train, y4_train) 

y4_pred = lda.predict(X4_test) 

accuracy = np.round(np.sum(y4_test == y4_pred)/len(y4_test),3)

print("LDA accuracy for the Pokemon3 dataset", accuracy) 

print("Precision_score",precision_score(y4_test, y4_pred, average='macro'))

print("Recall_score",recall_score(y4_test, y4_pred, average='macro'))

print("F1_score",f1_score(y4_test, y4_pred, average='macro'))
#QDA Model

clf = QDA()

clf.fit(X4_train, y4_train)

y4_pred = clf.predict(X4_test) 

accuracy = np.round(np.sum(y4_test == y4_pred)/len(y4_test),3)

print("QDA accuracy for the Pokemon3 dataset", accuracy) 

print("Precision_score",precision_score(y4_test, y4_pred, average='macro'))

print("Recall_score",recall_score(y4_test, y4_pred, average='macro'))

print("F1_score",f1_score(y4_test, y4_pred, average='macro'))



#Naive Bayes

gnb = GaussianNB() 

gnb.fit(X4_train, y4_train)

y4_pred = gnb.predict(X4_test)

accuracy = np.round(np.sum(y4_test == y4_pred)/len(y4_test),3)

print("Naive Bayes accuracy for the smaller Pokemon dataset", accuracy)

print("Precision_score",precision_score(y4_test, y4_pred, average='macro'))

print("Recall_score",recall_score(y4_test, y4_pred, average='macro'))

print("F1_score",f1_score(y4_test, y4_pred, average='macro'))
#Otkako gi trgnav kolinearnite koloni, dobiv najdobar rezultat so QDA modelot

#Bidejki speak dosta se razlikuvaat legendarnite pokemons od obicnite, ne e mnogu tesko da se razgranicat

#so Site modeli dobiv dosta dobri rezultati
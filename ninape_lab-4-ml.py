# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Libraries for visualization

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from yellowbrick.classifier import ConfusionMatrix

from sklearn import model_selection

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import preprocessing

from sklearn.model_selection import train_test_split



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score
colnames=['Letter', 'X-box', 'Y-box', 'width','height','onpix total','x-bar mean','y-bar mean','x var','y var','xy correlation','mean x x y','mean x y y ','edge','correlation x-ege','edge bTt','correlation y-ege'] 

letters = pd.read_csv('/kaggle/input/letter-recognition.data',names = colnames, header=None)

letters
letters.isnull().sum()

#nema prazno :)
letters['LetterInt'] = letters['Letter'].apply(lambda x: ord(x.lower())-96)

letters = letters.drop('Letter',axis=1)
letters
columns = list(letters.columns)

data = letters.values



fig = plt.figure(figsize=(15, 50))

fig.subplots(17//2+1, ncols=2)

for feat_i in range(17): 

    ax = plt.subplot(17//2+1,2, feat_i+1)

    plt.title(columns[feat_i]) 

    sns.distplot(data[:,feat_i], color = "navy")

plt.show()
#Naive Bayes



training_points = np.array(letters[:15000].drop(['LetterInt'], 1))

training_labels = np.array(letters[:15000]['LetterInt'])



clf = GaussianNB()

clf.fit(training_points, training_labels)



test_points = np.array(letters[15000:].drop(['LetterInt'], 1))

test_labels = np.array(letters[15000:]['LetterInt'])



expected = test_labels

predicted = clf.predict(test_points)



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))
#KNN

training_points = np.array(letters[:15000].drop(['LetterInt'], 1))

training_labels = np.array(letters[:15000]['LetterInt'])



neigh = KNeighborsClassifier(n_neighbors=26)

neigh.fit(training_points, training_labels) 



test_points = np.array(letters[15000:].drop(['LetterInt'], 1))

test_labels = np.array(letters[15000:]['LetterInt'])



expected = test_labels

predicted = clf.predict(test_points)



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))
training_points = np.array(letters[:15000].drop(['LetterInt'], 1))

training_labels = np.array(letters[:15000]['LetterInt'])



clf = SVC()

clf.fit(training_points, training_labels) 



test_points = np.array(letters[15000:].drop(['LetterInt'], 1))

test_labels = np.array(letters[15000:]['LetterInt'])



expected = test_labels

predicted = clf.predict(test_points)



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))

X=letters.drop(['LetterInt'],axis=1)

y=letters['LetterInt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = letters.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_target = abs(cor["LetterInt"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.2]

relevant_features

#drop all other features apart from these
print(letters[["y-bar mean","mean x x y"]].corr())

print(letters[["mean x x y","correlation x-ege"]].corr())



#Moze y-bar mean da se dropne



#So Pearson "mean x x y","correlation x-ege"
import statsmodels.api as sm

X = letters.drop("LetterInt",1)   #Feature Matrix

y = letters["LetterInt"]          #Target Variable



#Adding constant column of ones, mandatory for Ordinary Least Squares model

X_1 = sm.add_constant(X)

#Fitting sm.OLS model

model = sm.OLS(y,X_1).fit()

model.pvalues





#which is greater than 0.05. Hence we will remove this feature and build the model once again. This is an iterative process and can be performed at once with the help of loop.
l1 = letters.drop(['height','mean x x y','correlation x-ege','x var','edge'],axis=1)
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
letters['LetterInt'].value_counts() #Class distribution
plt.figure()



from sklearn.decomposition import PCA

pca = PCA(n_components=6)

proj = pca.fit_transform(X)

plt.scatter(proj[:, 0], proj[:, 1], c=y, cmap="Paired")

plt.colorbar()
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(letters)
#2d vizuelizacija 

#letters['tsne-2d-one'] = tsne_results[:,0]

#letters['tsne-2d-two'] = tsne_results[:,1]

label = np.array( letters['LetterInt'])

plt.figure(figsize=(16,10))

sns.scatterplot(

    x=tsne_results[:,0], y=tsne_results[:,1],

    hue = label,

    palette=sns.color_palette("hls", 26),

    data=X,

    legend="full",

    alpha=0.3

)
letters.describe().T
l = letters.drop(['Y-box','correlation y-ege','x var','edge bTt','y-bar mean','y var','width'],axis=1)

l
training_points = np.array(l[:15000].drop(['LetterInt'], 1))

training_labels = np.array(l[:15000]['LetterInt'])



clf = SVC()

clf.fit(training_points, training_labels) 





test_points = np.array(l[15000:].drop(['LetterInt'], 1))

test_labels = np.array(l[15000:]['LetterInt'])



expected = test_labels

predicted = clf.predict(test_points)



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))



training_points = np.array(l1[:15000].drop(['LetterInt'], 1))

training_labels = np.array(l1[:15000]['LetterInt'])



clf = SVC()

clf.fit(training_points, training_labels) 





test_points = np.array(l1[15000:].drop(['LetterInt'], 1))

test_labels = np.array(l1[15000:]['LetterInt'])



expected = test_labels

predicted = clf.predict(test_points)



accuracy = clf.score(test_points, test_labels)



print(float(accuracy))



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))



clf = GaussianNB()

clf.fit(training_points, training_labels)

# predicts = clf.predict(test_points)



accuracy = clf.score(test_points, test_labels)



print(float(accuracy))



expected = test_labels

predicted = clf.predict(test_points)



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))
neigh = KNeighborsClassifier(n_neighbors=26)

neigh.fit(training_points, training_labels) 



accuracy = neigh.score(test_points, test_labels)

print(float(accuracy))



expected = test_labels

predicted = clf.predict(test_points)



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))
pca = PCA(n_components=6)

l_pca = pca.fit_transform(X)



pca.explained_variance_ratio_
pca_letters=pd.DataFrame(data=l_pca[0:,0:],

                       index=[i for i in range(l_pca.shape[0])],

                       columns=['f'+str(i) for i in range(l_pca.shape[1])])

pca_letters['LetterInt']=letters['LetterInt']

pca_letters
#Plotting the Cumulative Summation of the Explained Variance

pca = PCA(n_components=16)

pca = PCA().fit(X)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Pulsar Dataset Explained Variance')

plt.show()
training_points = np.array(pca_letters[:15000].drop(['LetterInt'], 1))

training_labels = np.array(pca_letters[:15000]['LetterInt'])



clf = SVC()

clf.fit(training_points, training_labels) 





test_points = np.array(pca_letters[15000:].drop(['LetterInt'], 1))

test_labels = np.array(pca_letters[15000:]['LetterInt'])



expected = test_labels

predicted = clf.predict(test_points)



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))



clf = GaussianNB()

clf.fit(training_points, training_labels)

# predicts = clf.predict(test_points)



accuracy = clf.score(test_points, test_labels)



print(float(accuracy))



expected = test_labels

predicted = clf.predict(test_points)



# summarize the fit of the model

target_names=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print(metrics.classification_report(expected, predicted,target_names=target_names))
pca_letters = letters

pca = PCA(n_components=6)

l_pca = pca.fit_transform(pca_letters)



pca.explained_variance_ratio_



pca_letters['pca-one'] = l_pca[:,0]

pca_letters['pca-two'] = l_pca[:,1]

pca_letters['pca-three'] = l_pca[:,2]

pca_letters['pca-four'] = l_pca[:,3]

pca_letters['pca-five'] = l_pca[:,4]

pca_letters['pca-six'] = l_pca[:,5]

pca_letters
training_points = np.array(pca_letters[:15000].drop(['LetterInt'], 1))

training_labels = np.array(pca_letters[:15000]['LetterInt'])



test_points = np.array(pca_letters[15000:].drop(['LetterInt'], 1))

test_labels = np.array(pca_letters[15000:]['LetterInt'])

clf = GaussianNB()

clf.fit(training_points, training_labels)

expected = test_labels

predicted = clf.predict(test_points)



# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(expected, predicted)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(expected, predicted,average="macro")

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(expected, predicted,average="macro")

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(expected, predicted,average="macro")

print('F1 score: %f' % f1)
clf = SVC()

clf.fit(training_points, training_labels) 



expected = test_labels

predicted = clf.predict(test_points)





accuracy = accuracy_score(expected, predicted)

print('Accuracy: %f' % accuracy)

precision = precision_score(expected, predicted,average="macro")

print('Precision: %f' % precision)

recall = recall_score(expected, predicted,average="macro")

print('Recall: %f' % recall)

f1 = f1_score(expected, predicted,average="macro")

print('F1 score: %f' % f1)
neigh = KNeighborsClassifier(n_neighbors=26)

neigh.fit(training_points, training_labels) 



expected = test_labels

predicted = neigh.predict(test_points)



accuracy = accuracy_score(expected, predicted)

print('Accuracy: %f' % accuracy)

precision = precision_score(expected, predicted,average="macro")

print('Precision: %f' % precision)

recall = recall_score(expected, predicted,average="macro")

print('Recall: %f' % recall)

f1 = f1_score(expected, predicted,average="macro")

print('F1 score: %f' % f1)
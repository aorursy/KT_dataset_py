# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn import model_selection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
#Importing Dataset

data=pd.read_csv('../input/top50.csv', encoding='ISO-8859-1')

data
print(data.shape)
data=data.dropna(how='all')
print(data.groupby('Genre').size())
data.isnull().sum()
data=data.sort_values(['Unnamed: 0'])

data=data.reindex(data['Unnamed: 0'])

data=data.drop("Unnamed: 0",axis=1)

data.head()
data.describe()
data.info()
data=data.loc[:49,:]
data=data.rename(columns={"Loudness..dB..": "Loudness", "Acousticness..": "Acousticness", "Speechiness.":"Speechiness","Valence.":"Valence","Length.":"Length"})

data.tail()
data.columns
data['GeneralGenre']=['hip hop' if each =='atl hip hop'

                      else 'hip hop' if each =='canadian hip hop'

                      else 'hip hop' if each == 'trap music'

                      else 'pop' if each == 'australian pop'

                      else 'pop' if each == 'boy band'

                      else 'pop' if each == 'canadian pop'

                      else 'pop' if each == 'dance pop'

                      else 'pop' if each == 'panamanian pop'

                      else 'pop' if each == 'pop'

                      else 'pop' if each == 'pop house'

                      else 'electronic' if each == 'big room'

                      else 'electronic' if each == 'brostep'

                      else 'electronic' if each == 'edm'

                      else 'electronic' if each == 'electropop'

                      else 'rap' if each == 'country rap'

                      else 'rap' if each == 'dfw rap'

                      else 'escape room' if each == 'hip hop'

                      else 'latin' if each == 'latin'

                      else 'r&b' if each == 'r&n en espanol'

                      else 'raggae' for each in data['Genre']]
data.head(10)
print(data.groupby('GeneralGenre').size())

sns.countplot(x="GeneralGenre", data=data)
color_list = ['red' if i=='electronic' 

              else 'green' if i=='escape room' 

              else 'blue' if i == 'hip hop' 

              else 'purple' if i == 'latin'

              else 'darksalmon' if i == 'pop'

              else 'darkcyan' if i == 'raggae'

              else 'greenyellow' for i in data.loc[:,'GeneralGenre']]

pd.plotting.scatter_matrix(data.loc[:,['Energy','Danceability','Length','Popularity']],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '+',

                                       edgecolor= "black")

plt.show()
# box graphic

data.plot(kind='box', subplots=True, sharex=False, sharey=False)

plt.gcf().set_size_inches(12, 8)

plt.show()





# histogram

data.hist()

plt.gcf().set_size_inches(12, 8)

plt.show()



# scatter plot matrix

scatter_matrix(data)

plt.gcf().set_size_inches(12, 8)

plt.show()
# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,['Beats.Per.Minute','Energy','Danceability','Loudness','Liveness','Valence','Length','Acousticness',

                    'Speechiness','Popularity']], data.loc[:,['GeneralGenre']]



knn.fit(x,y)

prediction = knn.predict(x)

print('Prediction: {}'.format(prediction))

print('Real: {}'.format(data['GeneralGenre']))
y
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)



#Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(x_train)

x_train=sc.transform(x_train)

x_test=sc.transform(x_test)



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity

neigh = np.arange(1, 20)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neigh):

    # k from 1 to 20

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[12,8])

plt.plot(neigh, test_accuracy, label = 'Testing Accuracy')

plt.plot(neigh, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('Value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neigh)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
# create data1 that includes Valence that is feature and Energy that is target variable

data1 = data[data['GeneralGenre'] =='pop']

x1 = np.array(data1.loc[:,'Valence']).reshape(-1,1)

y1 = np.array(data1.loc[:,'Energy']).reshape(-1,1)

# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x1,y=y1)

plt.xlabel('Valence')

plt.ylabel('Energy')

plt.show()
from sklearn.preprocessing import LabelEncoder

Encoder_y=LabelEncoder()

Y = Encoder_y.fit_transform(y)

Y=pd.DataFrame(Y)
#Scaling

from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()



x3=mms.fit_transform(x)

y3=mms.fit_transform(Y)



#Cross Validation



from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

k = 5

cv_result = cross_val_score(reg,x3,y3,cv=k) # uses R^2 as score 

print('CV Scores: ',cv_result)

print('CV scores average: ',np.sum(cv_result)/k)
# grid search cross validation with 1 hyperparameter

from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,20)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV

knn_cv.fit(x,y)# Fit



# Print hyperparameter

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 

print("Best score: {}".format(knn_cv.best_score_))
# Confusion matrix with random forest

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

x,y = data.loc[:,['Beats.Per.Minute','Energy','Danceability','Loudness','Liveness','Valence','Length','Acousticness',

                    'Speechiness','Popularity']], data.loc[:,['GeneralGenre']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)



#Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(x_train)

x_train=sc.transform(x_train)

x_test=sc.transform(x_test)



rf = RandomForestClassifier(random_state = 10)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

cm = confusion_matrix(y_test,y_pred)

print('Confusion matrix: \n',cm)

print('Classification report: \n',classification_report(y_test,y_pred))
# KMeans Clustering

data2 = data.loc[:,['Liveness','Popularity']]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)

kmeans.fit(data2)

labels = kmeans.predict(data2)

plt.scatter(data['Popularity'],data['Liveness'],c = labels)

plt.xlabel('Liveness')

plt.ylabel('Popularity')

plt.show()
#Standardization in cross tabulation table

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

scalar = StandardScaler()

kmeans = KMeans(n_clusters = 2)

pipe = make_pipeline(scalar,kmeans)

pipe.fit(x)

labels = pipe.predict(x)

df = pd.DataFrame({'labels':labels,"class":data['GeneralGenre']})

ct = pd.crosstab(df['labels'],df['class'])

print(ct)
from scipy.cluster.hierarchy import linkage,dendrogram



merg = linkage(x,method = 'single')

dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)

plt.show()
# PCA

from sklearn.decomposition import PCA

model = PCA()

model.fit(x)

transformed = model.transform(x)

print('Principle components: ',model.components_)
# PCA variance

scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler,pca)

pipeline.fit(x)



plt.bar(range(pca.n_components_), pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.show()
# apply PCA

pca = PCA(n_components = 3)

pca.fit(x)

transformed = pca.transform(x)

a = transformed[:,0]

b = transformed[:,1]

d = transformed[:,2]

plt.scatter(a,b,d,c = color_list)

plt.show()
# Creating dependent and independent variables

X = data.loc[:, ['Beats.Per.Minute','Energy','Danceability','Loudness','Liveness','Valence','Length','Acousticness',

                    'Speechiness','Popularity']]

Y = data.loc[:, ['Genre']]



# Split train and test data

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)
# Creating model list

models = [

    ('LR', LogisticRegression()),

    ('LDA', LinearDiscriminantAnalysis()),

    ('KNN', KNeighborsClassifier()),

    ('DT', DecisionTreeClassifier()),

    ('NB', GaussianNB()),

    ('SVM', SVC())

]



results = []

names = []



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(X_train)

X_train=sc.transform(X_train)

X_test=sc.transform(X_test)



for name, model in models:

    kfold = model_selection.KFold(n_splits=12, random_state=7)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")

    results.append(cv_results)

    names.append(name)

    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
svc = SVC()

svc.fit(X_train, Y_train)

predictions = svc.predict(X_test)



print('accuracy value :', accuracy_score(Y_test, predictions))

print(confusion_matrix(Y_test, predictions))

print(classification_report(Y_test, predictions))
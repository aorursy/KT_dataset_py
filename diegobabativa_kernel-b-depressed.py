import pandas as pd

data_pd = pd.read_csv("../input/b_depressed.csv")

data_pd_original = pd.read_csv("../input/b_depressed.csv")
#Describing the datatypes:

data_pd.dtypes
data_pd.columns
data_pd.groupby('depressed').size()

# People with depression: 1191

# People with NO depression: 238
# Or if you like the percentages as follows: 

fp = data_pd.groupby('depressed').size()/data_pd.shape[0]

print(fp)
data_pd.mean(axis=0)
import matplotlib.pyplot as plt



plt.matshow(data_pd.corr())

plt.show()



f = plt.figure(figsize=(19, 15))

plt.matshow(data_pd.corr(), fignum=f.number)

plt.xticks(range(data_pd.shape[1]), data_pd.columns, fontsize=14, rotation=45)

plt.yticks(range(data_pd.shape[1]), data_pd.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('C',fontsize=16);
hist = data_pd['Age'].hist()
#% matplotlib inline

import seaborn as sns

sns.set()

sns.countplot(x='sex', data = data_pd) 

# Where woman is 1 and man is 0
sns.factorplot(x='depressed', col='Married', kind='count', data= data_pd)
#If we explore the next feature 'no_lasting_investment' there are some rows without values:

data_pd.no_lasting_investmen.isnull().sum()
data_pd['no_lasting_investmen'].fillna(data_pd['no_lasting_investmen'].median(), inplace=True)
data_pd_original['no_lasting_investmen'].fillna(data_pd_original['no_lasting_investmen'].median(), inplace=True)
data_pd.no_lasting_investmen.isnull().sum()
data_pd_original.no_lasting_investmen.isnull().sum()
data_pd['no_lasting_investmen'] = data_pd['no_lasting_investmen'].astype(int)
data_pd_original['no_lasting_investmen'] = data_pd_original['no_lasting_investmen'].astype(int)
data_pd
columns_to_normalize = [ 'Age', 'Number_children','education_level', 'total_members', 'gained_asset', 'durable_asset',

       'save_asset', 'living_expenses', 'other_expenses',

       'incoming_agricultural', 'farm_expenses', 

       'lasting_investment', 'no_lasting_investmen']
# We use Scipy library zscore to the normalization:

from scipy import stats



for c in columns_to_normalize:

    data_pd[c] = stats.zscore(data_pd[c])
#Print the data normalizaed 

data_pd.head()
# Drop By Name:

data_pd = data_pd.drop(['Survey_id', 'Ville_id'], axis=1)

data_pd.head(5)
data_pd_original = data_pd_original.drop(['Survey_id', 'Ville_id'], axis=1)
for c in columns_to_normalize:

    data_pd[c] = pd.cut(data_pd[c], 5)



data_pd.head()
columns_two_bins = ['sex', 'Married',

       'incoming_own_farm', 'incoming_business','incoming_no_business', 

       'labor_primary', 'depressed']
for c in columns_two_bins:

  data_pd[c] = pd.cut(data_pd[c], 2)
data_pd.head()
total_data = pd.DataFrame()

for i in data_pd.columns:

  data_pd[i] = pd.Categorical(data_pd[i])

  data_frame_Dummies = pd.get_dummies(data_pd[i], prefix = i)

  total_data = pd.concat([total_data, data_frame_Dummies], axis=1)
total_data.head()
depression_normalized = total_data
depression_normalized
#Compute the time that the Algorithm takes in calculates the rules for this excercise of Association:

import time

from mlxtend.frequent_patterns import apriori



start_time = time.time()

frequent_apriori = apriori(total_data, min_support=0.65, use_colnames=True)

print("Execution time by Apriori Algorithm  %s" % (time.time() - start_time))
from mlxtend.frequent_patterns import association_rules



reglas_asociacion_dataset=association_rules(frequent_apriori, metric="confidence", min_threshold=0.8)

display(reglas_asociacion_dataset.sort_values(by = ['support','confidence'], ascending = [False,False]))

print("Total reglas encontradas %s" % (reglas_asociacion_dataset.shape[0]))
X = data_pd_original.copy()
X.head()


from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score



n_clusters = 1

km = KMeans( n_clusters=n_clusters)

km.fit(X)

y = km.predict(X)

X.shape
# It calculates the size of Cluster

pd.Series(y).value_counts()
# List the centroids of the clusters

km.cluster_centers_
data_pd_original.columns
# We decide to select some columns to separate the data and We can print the clusters very well created.



X_filtered = X.filter(['save_asset','durable_asset', 'save_asset',

                      'living_expenses','other_expenses','incoming_agricultural',

                       'farm_expenses','lasting_investment','no_lasting_investmen']).values+15
n_clusters = 3



km = KMeans( n_clusters=n_clusters)

km.fit(X_filtered)

y_filtered = km.predict(X_filtered)



print(X_filtered.shape)

print(y_filtered.shape)



#Print the lists of the centroids

km.cluster_centers_
# print the groups of the centroids



import numpy as np

cmap = plt.cm.plasma



cmap((y_filtered*255./(n_clusters-1)).astype(int))

for i in np.unique(y_filtered):

    cmap = plt.cm.bwr

    col = cmap((i*255./(n_clusters-1)).astype(int))

    Xr = X_filtered[y_filtered==i]

    plt.scatter(Xr[:,0], Xr[:,1], color=col, label="cluster %d"%i, alpha=.5)

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],marker="x", lw=5, s=200, color="black")

plt.legend()

Sum_of_squared_distances = []



#It iterates

K = range(2,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X_filtered)

    Sum_of_squared_distances.append(km.inertia_)

    

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Inertia')

plt.show()
X_filtered_2 = X.filter(['Age', 'Number_children', 'education_level','total_members']).values+15





def plot_kmeans(n_clusters):

    km = KMeans( n_clusters=n_clusters)

    km.fit(X_filtered_2)

    y_filtered_2 = km.predict(X_filtered_2)



    # Dibuja los grupos con sus centroides

    cmap = plt.cm.plasma



    cmap((y_filtered*255./(n_clusters-1)).astype(int))

    for i in np.unique(y_filtered_2):

        cmap = plt.cm.bwr

        col = cmap((i*255./(n_clusters-1)).astype(int))

        Xr = X_filtered_2[y_filtered_2==i]

        plt.scatter(Xr[:,0], Xr[:,1], color=col, label="cluster %d"%i, alpha=.5)

    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],marker="x", lw=5, s=200, color="black")

    plt.legend()



plot_kmeans(3)
plot_kmeans(8)
Sum_of_squared_distances = []



#Se itera 

K = range(2,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(X_filtered_2)

    Sum_of_squared_distances.append(km.inertia_)

    

plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Inertia')

plt.show()
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score

from sklearn.utils.multiclass import unique_labels



X_filtered_3 = X.copy()



y_true = X_filtered_3['depressed']

class_names = ['Depressed','No Depressed']



km = KMeans(n_clusters=2)

km = km.fit(X_filtered_2)

y_pred = km.labels_



cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix")

print(cm)
def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False, cmap=plt.cm.Blues):



    title = 'Confusion Matrix'

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True Label',

           xlabel='Predicted Label')



    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="red" if cm[i, j] > thresh else "red")

    fig.tight_layout()

    return ax



np.set_printoptions(precision=2)

plot_confusion_matrix(y_true, y_pred, classes=class_names)

plt.show()
def compute_purity(confusion_matrix):

    maximus = []

    purity_score = 0

    for i in range(0,confusion_matrix.shape[0]):

        maximus.append(np.max(cm[i]))

    purity_score = np.sum(maximus)/np.sum(confusion_matrix)

    return purity_score
purity_score= compute_purity(cm)

print("The purity calculated is: ", purity_score)
#Evaluar la precisión:

precision = precision_score(y_true, y_pred, average='macro') 

recall = recall_score(y_true, y_pred, average='macro')  

f1score = f1_score(y_true, y_pred, average='macro')  



print("The Precision calculated was ", precision)

print("The Recall calculated was: ", recall)

print("The F1-Score calculated was: ", f1score)
X_copy = X.copy()

y = X.depressed.values

X = X.values
from sklearn.model_selection import train_test_split



#Se parten los datos para usar 70 Entrenamiento y 30 test:

X_train_bayes, X_test_bayes, y_train_bayes, y_test_bayes = train_test_split(X, y,

                                                    test_size=.3)
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB



#Create a Gaussian Classifier

model = GaussianNB()

model.fit(X_train_bayes, y_train_bayes)
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, classification_report

y_pred_bayes = model.predict(X_test_bayes)
cm = confusion_matrix(y_test_bayes, y_pred_bayes)

print(cm)

# Print the precision and recall, among other metrics

print(classification_report(y_test_bayes, y_pred_bayes, digits=2))
#Import the libraries required

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(class_weight="balanced", random_state=1, max_iter=1000, solver="liblinear")



classifier.fit(X_train_bayes, y_train_bayes)

y_pred_logistic = classifier.predict(X_test_bayes)

score = classifier.score(X_test_bayes, y_test_bayes)
cm = confusion_matrix(y_test_bayes, y_pred_logistic)

print(cm)

# Print the precision and recall, among other metrics

print(classification_report(y_test_bayes, y_pred_logistic, digits=2))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)
report = classification_report(y_test, y_pred)

print(report)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential() # Creation of the model

model.add(Dense(units= 128, input_dim = 21, activation = 'relu')) # input_dim = Variables or attributes

model.add(Dense(units = 64, activation='relu')) # First activation Layer

model.add(Dense(units = 8, activation='relu')) # Second activation Layer

model.add(Dense(units = 1, activation='sigmoid')) # Sigmoid activation function for binary classification
# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
y_pred_MLP = model.fit(X_train_bayes, y_train_bayes, 

                       validation_data=([X_test_bayes],[y_test_bayes]) ,epochs=150, batch_size=10)
_, accuracy = model.evaluate(X_train_bayes, y_train_bayes)

accuracy
print(y_pred_MLP.history.keys())



# summarize history for accuracy

plt.plot(y_pred_MLP.history['accuracy'])

plt.plot(y_pred_MLP.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(y_pred_MLP.history['loss'])

plt.plot(y_pred_MLP.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred_NN = model.predict(X_test_bayes, batch_size=50, verbose=1)
y_pred_NN.shape
y_pred_NN_bool = np.argmax(y_pred_NN, axis=1)
cm = confusion_matrix(y_test_bayes, y_pred_NN_bool)

print(cm)

# Print the precision and recall, among other metrics

print(classification_report(y_test_bayes, y_pred_NN_bool, digits=2))
# Save depressed variable and after pup again,

y_target = X_copy['depressed'].values
print(y_target)
depression_normalized.values
# To Apply the Random Forest Model Again:

X = depression_normalized.values

y = y_target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.model_selection import KFold



indexes = []

scores = []

cv = KFold(n_splits=5, random_state=42, shuffle=False)

for train_index, test_index in cv.split(X):

    print("Train Index:", train_index)

    print("Test  Index:", test_index)

    X_train, X_test, y_train, y_test = X[train_index],X[test_index],y[train_index], y[test_index]

    rf = RandomForestClassifier()

    rf.fit(X_train,y_train)

    scores.append(rf.score(X_test, y_test))
folds = [0,1,2,3,4,5]
print(scores)
plt.plot(scores)

plt.title('K fold - Cross Validation')

plt.ylabel('Acuracy')

plt.xlabel('Fold')

plt.legend(['Training for each fold'], loc='upper left')

plt.show()
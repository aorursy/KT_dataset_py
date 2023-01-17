import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pkmn = pd.read_csv("../input/Pokemon.csv", index_col=0)
pkmn.shape
pkmn.columns
pkmn.isnull().sum()
sns.set_style('darkgrid')
plt.figure(figsize=(10,8))
pkmn['Type 1'].value_counts().plot.barh(width=.9).set_title('Pokémon Type', fontsize=14)
plt.xlabel('counts', fontsize=12)
plt.ylabel('type', fontsize=12)
pkmn['Legendary'].value_counts()
# 65 Legendary pokemons
pkmn[pkmn['Legendary']==True]['Type 1'].value_counts()
pkmn['Generation'].value_counts()
pkmn2= pkmn.drop(['Total', 'Legendary'], axis=1)
pkmn2.groupby('Generation').boxplot(figsize=(16,10))
pkmn_knn= pkmn.copy()
pkmn_knn.drop(['Name','Type 1', 'Type 2', 'Total'],axis=1, inplace=True)
pkmn_knn.head()
pkmn_knn.dtypes
pkmn_knn['Legendary'] = pkmn_knn['Legendary'].astype(int)
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(pkmn_knn.drop('Legendary', axis=1))
scaled_data= scaler.transform(pkmn_knn.drop('Legendary', axis=1))
scaled= pd.DataFrame(scaled_data, columns=pkmn_knn.columns[:-1])
X= scaled
y= pkmn_knn['Legendary']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=88)
from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predictions= knn.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, predictions):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, predictions, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, predictions, pos_label=None,
                              average='weighted')
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, predictions, pos_label=None, average='weighted')
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, predictions)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=30)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)

    return plt
cm= confusion_matrix(y_test, predictions)
fig = plt.figure(figsize=(8, 8))
plot = plot_confusion_matrix(cm, classes=['Non-Legendary','Legendary'], normalize=False, title='Confusion Matrix')
plt.show()
error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

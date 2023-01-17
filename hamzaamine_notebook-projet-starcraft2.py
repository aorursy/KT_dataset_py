%config IPCompleter.greedy=True
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os 

import seaborn as sn

import re

import time

from scipy.stats import randint as sp_randint

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,f1_score

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree,preprocessing

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import normalize

from imblearn.over_sampling import RandomOverSampler

from pandas.plotting import parallel_coordinates,scatter_matrix

from collections import Counter
def parse_file_train(filename):

    line = list()

    with open(filename) as f:

        lines = f.readline()

        counter = 1

        while lines:

            data = lines.strip('\n').split(',')

            actions = data[2:]

            player = data[0]

            race= data[1]

            line.append([player,race,actions])

            lines = f.readline()

            counter += 1

    return line



            

def parse_file_test(filename):

    line = list()

    with open(filename) as f:

        lines = f.readline()

        counter = 1

        while lines:

            data = lines.strip('\n').split(',')

            actions = data[1:]

            race = data[0]

            line.append([race,actions])

            lines = f.readline()

            counter += 1

    return line
train = parse_file_train("../input/starcraft-2-player-prediction-challenge-2019/TRAIN.CSV")

test=parse_file_test("../input/starcraft-2-player-prediction-challenge-2019/TEST.CSV")
print("La shape du jeu d'entrainement est " + str(len(train)))

print("La shape du jeu de test est " + str(len(test)))
def count_freq(line):

    player, race, actions = line

    counter = []  

    c=0

    c1=0

    c2=0

    j='t0'

    tmp=0

    t0=0

    t_max=t_min=0

    matching = [s for s in actions if s.startswith('t')]

    for j in matching:

        index = actions.index(j)

        hot= 's'

        c+=actions[t0:index].count(hot)

        hot='Base'

        c+=actions[t0:index].count(hot)

        hot='SingleMineral'

        c+=actions[t0:index].count(hot)

        for i in range(10):

            hot = 'hotkey'+str(i)+'0'

            c+=actions[t0:index].count(hot)

            hot = 'hotkey'+str(i)+'1'

            c+=actions[t0:index].count(hot)

            hot = 'hotkey'+str(i)+'2'

            c+=actions[t0:index].count(hot)

        if(tmp<c):

            tmp=c

            t_max=index

            t_min=t0

        t0=index

        c=0

    t=[int(s) for s in j.split('t') if s.isdigit()]

    if(t_max and t_min and t[0]):

        tmin=t[0]/60

        for itera in range(2): 

            hot='s'

            counter.append((actions.count(hot)/tmin))

            c+=actions.count(hot)

            hot='Base'

            counter.append((actions.count(hot)/tmin))

            c+=actions.count(hot)

            hot='SingleMineral'

            counter.append((actions.count(hot)/tmin))

            c+=actions.count(hot)

            for i in range(10):

                hot = 'hotkey'+str(i)+'0'

                counter.append((actions.count(hot)/tmin))

                c+=actions.count(hot)

                hot = 'hotkey'+str(i)+'1'

                counter.append((actions.count(hot)/tmin))

                c+=actions.count(hot)

                hot = 'hotkey'+str(i)+'2'

                counter.append((actions.count(hot)/tmin))

                c+=actions.count(hot)

            actions=actions[t_min:t_max]

            counter.append(c/tmin)

            c=0 

    return counter



def count_freq_test(line):

    race, actions = line

    counter = []  

    c=0

    c1=0

    c2=0

    j='t0'

    tmp=0

    t0=0

    t_max=t_min=0

    matching = [s for s in actions if s.startswith('t')]

    for j in matching:

        index = actions.index(j)

        hot= 's'

        c+=actions[t0:index].count(hot)

        hot='Base'

        c+=actions[t0:index].count(hot)

        hot='SingleMineral'

        c+=actions[t0:index].count(hot)

        for i in range(10):

            hot = 'hotkey'+str(i)+'0'

            c+=actions[t0:index].count(hot)

            hot = 'hotkey'+str(i)+'1'

            c+=actions[t0:index].count(hot)

            hot = 'hotkey'+str(i)+'2'

            c+=actions[t0:index].count(hot)

        if(tmp<c):

            tmp=c

            t_max=index

            t_min=t0

        t0=index

        c=0

    t=[int(s) for s in j.split('t') if s.isdigit()]

    if(t_max and t_min and t[0]):

        tmin=t[0]/60

        for itera in range(2): 

            counter.append(actions.count('s')/tmin)

            c+=actions.count('s')

            counter.append(actions.count('Base')/tmin)

            c+=actions.count('Base')

            counter.append(actions.count('SingleMineral')/tmin)

            c+=actions.count('SingleMineral')

            for i in range(10):

                hot = 'hotkey'+str(i)+'0'

                counter.append(actions.count(hot)/tmin)

                c+=actions.count(hot)

                hot = 'hotkey'+str(i)+'1'

                counter.append(actions.count(hot)/tmin)

                c+=actions.count(hot)

                hot = 'hotkey'+str(i)+'2'

                counter.append(actions.count(hot)/tmin)

                c+=actions.count(hot)

            actions=actions[t_min:t_max]

            counter.append(c/tmin)

            c=0 

    return counter
freq = list()

for line in train:

    freq.append([line[0]]+ [line[1]]+ count_freq(line))

train_dataset = pd.DataFrame(freq)
freq1 = list()

for line in test:

    freq1.append([line[0]]+ count_freq_test(line))

test_dataset = pd.DataFrame(freq1)
train_dataset.rename(columns={0:'Player',1:'Race',2:'s', 3:'Base',4:'SingleMineral',35:'TotAction',36:'mfr',37:'mfs',38:'mfb',39:'mfsm',},inplace=True)

k=5

for i in range(10):

    for j in range(3):

        train_dataset.rename(columns={k:'hotkey'+str(i)+str(j)},inplace=True)

        k+=1

k=40

for i in range(10):

    for j in range(3):

        train_dataset.rename(columns={k:'m_f_hotkey'+str(i)+str(j)},inplace=True)

        k+=1

train_dataset.head()
test_dataset.rename(columns={0:'Race',1:'s', 2:'Base',3:'SingleMineral',34:'TotAction',35:'mfr',36:'mfs',37:'mfb',38:'mfsm',},inplace=True)

k=4

for i in range(10):

    for j in range(3):

        test_dataset.rename(columns={k:'hotkey'+str(i)+str(j)},inplace=True)

        k+=1

k=39

for i in range(10):

    for j in range(3):

        test_dataset.rename(columns={k:'m_f_hotkey'+str(i)+str(j)},inplace=True)

        k+=1

test_dataset.head()
train_dataset=train_dataset.fillna(train_dataset.mean())

train_dataset.tail()
test_dataset=test_dataset.fillna(test_dataset.mean())

test_dataset.tail()
labels=train_dataset['Player']
data = train_dataset.iloc[:, 2:].values
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))

plt.title("Player actions")

dend = shc.dendrogram(shc.linkage(data, method='ward'))
#partie jouée par chaque jouer

players=train_dataset.groupby('Player').Player.count().reset_index(name='count').sort_values('count',ascending=False)

players.head()
max_games=players[players['count']==players['count'].min()]

l=max_games['Player'].values
player_with_max_games= train_dataset['Player'] ==l[0]

train_dataset[player_with_max_games]
train_dataset.hist(figsize = (16,20), stacked= False)
train_dataset['Player'] = train_dataset['Player'].apply( lambda x : int(str(x).split('/')[6]))
train_dataset.corr()['Player']
#on essaye de se débarasser des données qui sont pas corrélées

serie = pd.Series(train_dataset.corr()['Player'])

attributs = list(serie[abs(serie) > 0.01].index)

attributs.remove('Player')

attributs.append('Race')

attributs
corrMatt = train_dataset[attributs].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(40,20)

sn.heatmap(corrMatt, center = 0 ,cmap = "seismic",vmin=-1, vmax=1, square=True,annot=False)
# Encodage 

enc = preprocessing.LabelEncoder()

enc.fit(['Terran','Protoss','Zerg'])

test_dataset['Race']=enc.transform(test_dataset['Race']) 

train_dataset['Race']=enc.transform(train_dataset['Race']) 
features =  train_dataset.loc[:, train_dataset.columns != 'Player'] 

features=pd.DataFrame(features)

#split la data pour le training et le testing

features_train , features_test, labels_train , labels_test = train_test_split(features, labels, test_size=0.3,random_state=42)
accuracy = []

best_accuracy=0

for i in range(1,100):

   #Initialiser notre model

    dtree = tree.DecisionTreeClassifier(max_depth=i)

    #training

    dtree.fit(features_train,labels_train)

    #testing

    Y_pred = dtree.predict(features_test)

    #Cross validation

    accuracy.append(int(accuracy_score(labels_test, Y_pred)*100))

    if (best_accuracy<(dtree.score(features_test,labels_test)*100)):

        best_accuracy=dtree.score(features_test,labels_test)*100

        best_depth=i
#Initialiser notre model

dtree = tree.DecisionTreeClassifier(max_depth=best_depth) #best depth

#training

dtree.fit(features_train,labels_train)

#testing

Y_pred = dtree.predict(features_test)

#Cross validation 

print(f'accuracy of our decision tree model is around : {dtree.score(features_test,labels_test)*100} %' )
# plot accuracy

fig = plt.figure(figsize = (10, 5))

title = fig.suptitle("Decision Tree Accuracy", fontsize=14)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax = fig.add_subplot(1,1, 1)

ax.set_xlabel("Max depth")

ax.set_ylabel("Accuracy") 

ac = ([i for i in range(1,100)], accuracy)

ax.tick_params(axis='both', which='major', labelsize=8.5)

bar = ax.bar(ac[0], ac[1], color='steelblue', 

        edgecolor='black', linewidth=1)
#on essaye de tracer la matrice de confusion pour decision tree

mat=confusion_matrix(labels_test,Y_pred)

pd.DataFrame(normalize(mat,axis=1))
#on fait le hHyperparameter tuning : n_estimators 

for i in range(100,1001,50):

    classifier=ExtraTreesClassifier(n_estimators=i)

    classifier1=RandomForestClassifier(n_estimators=i)

    classifier.fit(features_train,labels_train)

    classifier1.fit(features_train,labels_train)

    print(f'accuracy of Extra trees classifier is around : {classifier.score(features_test,labels_test)*100:.2f} %' )

    print(f'accuracy of Random Forest is around : {classifier1.score(features_test,labels_test)*100:.2f} %' )

    print('*--------------------------------------------------------------------*')
classifier=ExtraTreesClassifier(n_estimators=500)#best accuracy

classifier.fit(features_train,labels_train)

print(f'accuracy of Extra trees classifier is around : {classifier.score(features_test,labels_test)*100:.3f} %' )

classifier1=ExtraTreesClassifier(n_estimators=500)#best accuracy

classifier1.fit(features_train,labels_train)

print(f'accuracy of Random forest classifier is around : {classifier1.score(features_test,labels_test)*100:.3f} %' )
Y_pred = classifier.predict(features_test)

Y_pred1 = classifier1.predict(features_test)

print('For Extra trees classifier')

print(classification_report(labels_test,Y_pred))

print('For Random Forest classifier')

print(classification_report(labels_test,Y_pred1))
def classification(Y, Y_pred):

    classif = pd.DataFrame({'Prediction' : Y_pred, 'Ground truth' : Y})

    return classif
#explorer plus 

class_df = classification(labels_test, Y_pred)

class_df.tail()
print(confusion_matrix(labels_test,Y_pred,labels=labels.unique()))
mat=confusion_matrix(labels_test,Y_pred)

pd.DataFrame(normalize(mat,axis=1))
Y_pred = classifier.predict(test_dataset)#for extra trees

Y_pred1 = classifier1.predict(test_dataset)#for random forest
#Save predictions

def pred(Y_pred):

    predictions_formatted = []

    for i, value in enumerate(Y_pred, 1):

        predictions_formatted.append([i, value])

    all_predict = pd.DataFrame(predictions_formatted)

    return all_predict

all_predict1=pred(Y_pred)

all_predict=pred(Y_pred1)
submission_path = "../working/submission_et.csv"# FOR EXTRA

submission_path1 = "../working/submission_rf.csv"# FOR RANDOM FOREST

if os.path.exists(submission_path):

    os.remove(submission_path)

all_predict.to_csv(submission_path, header=['RowId','prediction'], mode='w', index=False)

all_predict1.to_csv(submission_path1, header=['RowId','prediction'], mode='w', index=False)
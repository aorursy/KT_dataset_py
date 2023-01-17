# Imports and Helper Functions

# data Analysis

import pandas as pd

import numpy as np

import random as rng



# Visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#SciKit Learn Models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier





from sklearn.metrics import accuracy_score, classification_report,confusion_matrix



from subprocess import check_output

print(check_output(["ls", "../input/ufcdataset"]).decode("utf8"))

data = pd.read_csv("../input/ufcdataset/data.csv")

# Any results you write to the current directory are saved as output.
data.info()
data.describe()
data.describe(include=['O'])
data.head()
data.tail()
data.fillna(value=0,inplace=True)
data.tail()
dropdata = data.drop(['B_ID','B_Name','R_ID','R_Name','winby','Date'],axis=1)

dropdata.rename(columns={'BPrev':'B__Prev',

                         'RPrev':'R__Prev',

                         'B_Age':'B__Age',

                         'B_Height':'B__Height',

                         'B_Weight':'B__Weight',

                         'R_Age':'R__Age',

                         'R_Height':'R__Height',

                         'R_Weight':'R__Weight',

                         'BStreak':'B__Streak',

                         'RStreak': 'R__Streak'},inplace=True)

dropdata.describe()
dropdata.describe(include=['O'])
objecttypes = list(dropdata.select_dtypes(include=['O']).columns)

for col in objecttypes:

    dropdata[col] = dropdata[col].astype('category')
cat_columns = dropdata.select_dtypes(['category']).columns

dropdata[cat_columns] = dropdata[cat_columns].apply(lambda x: x.cat.codes)

dropdata.info()

dropdata.tail()
# Basic Correlation Matrix

# corrmat = data.corr()

# f, ax = plt.subplots(figsize=(12, 9))

# sns.heatmap(corrmat, vmax=.8, square=True);
# Subset Correlation Matrix

k = 10 #number of variables for heatmap

corrmat = dropdata.corr()

cols = corrmat.nlargest(k, 'winner')['winner'].index

cm = np.corrcoef(dropdata[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
 #We Store prediction of each model in our dict

# Helper Functions for our models. 



def percep(X_train,Y_train,X_test,Y_test,Models):

    perceptron = Perceptron(max_iter = 1000, tol = 0.001)

    perceptron.fit(X_train, Y_train)

    Y_pred = perceptron.predict(X_test)

    Models['Perceptron'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    return



def ranfor(X_train,Y_train,X_test,Y_test,Models):

    randomfor = RandomForestClassifier(max_features="sqrt",

                                       n_estimators = 700,

                                       max_depth = None,

                                       n_jobs=-1

                                      )

    randomfor.fit(X_train,Y_train)

    Y_pred = randomfor.predict(X_test)

    Models['Random Forests'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    return



def dec_tree(X_train,Y_train,X_test,Y_test,Models):

    decision_tree = DecisionTreeClassifier()

    decision_tree.fit(X_train, Y_train)

    Y_pred = decision_tree.predict(X_test)

    Models['Decision Tree'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    return



def SGDClass(X_train,Y_train,X_test,Y_test,Models):

    sgd = SGDClassifier(max_iter = 1000, tol = 0.001)

    sgd.fit(X_train, Y_train)

    Y_pred = sgd.predict(X_test)

    Models['SGD Classifier'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    return



def linSVC(X_train,Y_train,X_test,Y_test,Models):

    linear_svc = LinearSVC()

    linear_svc.fit(X_train, Y_train)

    Y_pred = linear_svc.predict(X_test)

    Models['SVM'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    return



def bayes(X_train,Y_train,X_test,Y_test,Models):

    gaussian = GaussianNB()

    gaussian.fit(X_train, Y_train)

    Y_pred = gaussian.predict(X_test)

    Models['Bayes'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]

    return



def Nearest(X_train,Y_train,X_test,Y_test,Models):

    knn = KNeighborsClassifier(n_neighbors = 3)

    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)

    Models['KNN'] = [accuracy_score(Y_test,Y_pred),confusion_matrix(Y_test,Y_pred)]



def run_all_and_Plot(df):

    Models = dict()

    from sklearn.model_selection import train_test_split

    X_all = df.drop(['winner'], axis=1)

    y_all = df['winner']

    X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

    percep(X_train,Y_train,X_test,Y_test,Models)

    ranfor(X_train,Y_train,X_test,Y_test,Models)

    dec_tree(X_train,Y_train,X_test,Y_test,Models)

    SGDClass(X_train,Y_train,X_test,Y_test,Models)

    linSVC(X_train,Y_train,X_test,Y_test,Models)

    bayes(X_train,Y_train,X_test,Y_test,Models)

    Nearest(X_train,Y_train,X_test,Y_test,Models)

    return Models





def plot_bar(dict):

    labels = tuple(dict.keys())

    y_pos = np.arange(len(labels))

    values = [dict[n][0] for n in dict]

    plt.bar(y_pos, values, align='center', alpha=0.5)

    plt.xticks(y_pos, labels,rotation='vertical')

    plt.ylabel('accuracy')

    plt.title('Accuracy of different models')

    plt.show()





def plot_cm(dict):

    count = 1

    fig = plt.figure(figsize=(10,10))

    for model in dict:

        cm = dict[model][1]

        labels = ['W','L','N','D']

        ax = fig.add_subplot(4,4,count)

        cax = ax.matshow(cm)

        plt.title(model,y=-0.8)

        fig.colorbar(cax)

        ax.set_xticklabels([''] + labels)

        ax.set_yticklabels([''] + labels)

        plt.xlabel('Predicted')

        plt.ylabel('True')

        # plt.subplot(2,2,count)

        count+=1

    plt.tight_layout()

    plt.show()
accuracies = run_all_and_Plot(dropdata)

CompareAll = dict()

CompareAll['Baseline'] = accuracies

for key,val in accuracies.items():

    print(str(key) +' '+ str(val[0]))

plot_bar(accuracies)

plot_cm(accuracies)
from sklearn.model_selection import train_test_split

from sklearn.grid_search import GridSearchCV

#X_all = dropdata.drop(['winner'], axis=1)

#y_all = dropdata['winner']

#X_train, X_test, Y_train, Y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=23)

#rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, max_depth=None) 

#param_grid = { 

#    'n_estimators': [200,700],

#    'max_features': ['auto', 'sqrt', 'log2']

#}



#CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

#CV_rfc.fit(X_train, Y_train)

#print(CV_rfc.best_params_)
dontchange = ['winner','Event_ID','Fight_ID','Max_round','Last_round','B_Age','R_Age']

numeric_cols = [col for col in dropdata if col not in dontchange]

dropdata[numeric_cols] += 1 
newDF = dropdata.copy()

blue_cols = [col for col in dropdata.columns if 'B__' in col]

red_cols = [col for col in dropdata.columns if 'R__' in col]

for (blue,red) in zip(blue_cols,red_cols):

    newkey = ''.join(str(blue).split('_')[2:])

    dropdata[newkey] = dropdata[str(blue)]/dropdata[str(red)]

    del dropdata[str(blue)]

    del dropdata[str(red)]
accuracies = run_all_and_Plot(dropdata)

for key,val in accuracies.items():

    print(str(key) +' '+ str(val[0]))

CompareAll['Blue/Red'] = accuracies

plot_bar(accuracies)

plot_cm(accuracies)

r4 = [col for col in dropdata.columns if "Round4" in col]

r5 = [col for col in dropdata.columns if "Round5" in col]

threerounds = dropdata.drop(r4+r5,axis = 1)

accuracies = run_all_and_Plot(threerounds)

for key,val in accuracies.items():

    print(str(key)+' '+str(val[0]))

CompareAll['DropR4&R5'] = accuracies

plot_bar(accuracies)

plot_cm(accuracies)
foobar = threerounds.loc[threerounds['Max_round'] == 3]

bewb = threerounds.drop(['Max_round','Last_round'],axis=1)

accuracies = run_all_and_Plot(bewb)

for key,val in accuracies.items():

    print(str(key)+' '+str(val[0]))

CompareAll['Drop5RoundFights'] = accuracies

plot_bar(accuracies)

plot_cm(accuracies)
blahblah = bewb[bewb.Prev != 1]

accuracies = run_all_and_Plot(blahblah)

for key,val in accuracies.items():

    print(str(key)+' '+str(val[0]))

CompareAll['DroppingDebut'] = accuracies

plot_bar(accuracies)

plot_cm(accuracies)
blue_cols

newDF.info()

b_feats = list(set([x[10:] for x in blue_cols if "Round" in x]))

r_feats = list(set([x[10:] for x in red_cols if "Round" in x]))

def sumshit(b_feats,cols):

    for x in b_feats:

        newDF.loc[:,x] = 0

        for y in cols:

            if x in y:

                newDF[x] += newDF[y]

                newDF.drop(y,axis=1,inplace=True)

sumshit(b_feats,blue_cols)

sumshit(r_feats,red_cols)

newDF.info()

newDF.describe()

accuracies = run_all_and_Plot(newDF)

for key,val in accuracies.items():

    print(str(key) +' '+ str(val[0]))

CompareAll['SumRounds'] = accuracies

plot_bar(accuracies)

plot_cm(accuracies)



blue_cols = [col for col in newDF.columns if 'B__' in col]

red_cols = [col for col in newDF.columns if 'R__' in col]

for (blue,red) in zip(blue_cols,red_cols):

    newkey = ''.join(str(blue).split('_')[2:])

    newDF[newkey] = newDF[str(blue)]/newDF[str(red)]

    del newDF[str(blue)]

    del newDF[str(red)]
accuracies = run_all_and_Plot(newDF)

for key,val in accuracies.items():

    print(str(key) +' '+ str(val[0]))

CompareAll['SumRounds'] = accuracies

plot_bar(accuracies)

plot_cm(accuracies)
reduced_features = newDF.drop(["Weight","B_HomeTown","B_Location", "Event_ID", "Fight_ID", "Max_round", "Last_round", "R_HomeTown", "R_Location"],axis = 1)

accuracies = run_all_and_Plot(reduced_features)

for key,val in accuracies.items():

    print(str(key) +' '+ str(val[0]))

CompareAll['Idunno'] = accuracies

plot_bar(accuracies)

plot_cm(accuracies)

reduced_features.info()
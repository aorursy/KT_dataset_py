import numpy as pd

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
data = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')
data.isnull().any()
data.describe()
data.iloc[:, :-1].boxplot(whis = "range", vert = False, figsize = (10,7))

plt.title("Range and distribution of the data in each column")
pos = data[data.iloc[:,8] == 1]

neg = data[data.iloc[:,8] == 0]
pos.describe()
neg.describe()
for i in range(8):

    fig = plt.figure(figsize = (6,8))

    fig = sns.violinplot(data = data, x = data.columns[-1], y = data.columns[i], scale = 'area', palette = {0: 'tab:orange', 1: 'tab:blue'})

    fig.set_title('Distribution of positive and negative cases')

    plt.show()

    
data1 = data.copy()

data1.loc[data1['target_class'] == 1, 'target_class'] = "Positive"

data1.loc[data1['target_class'] == 0, 'target_class'] = "Negative"



sns.pairplot(data1, hue = 'target_class', height = 4, markers = '.', vars = data1.columns[:-1], palette = 'bright', hue_order = ['Negative', 'Positive'])
fig, axes = plt.subplots(8, 8, figsize = (60,40))



labels = data.columns



for a in range(8):

    for b in range(8):

        

        if a == b:

            axes[a,b].hist(neg.iloc[:,a], bins = 40, color = 'r', alpha = 0.5, label = 'Negative', density = True)

            axes[a,b].hist(pos.iloc[:,a], bins = 40, color = 'b', alpha = 0.5, label = 'Positive', density = True)

            axes[a,b].set_xlabel(labels[a])

            axes[a,b].legend(markerscale = 20)

            

        else:

            axes[a,b].scatter(neg.iloc[:,a], neg.iloc[:,b], s = 0.1, c = 'red', alpha = 1, label = 'Negative')

            axes[a,b].scatter(pos.iloc[:,a], pos.iloc[:,b], s = 0.1, c = 'b', alpha = 1, label = 'Positive')

            axes[a,b].set_xlabel(labels[a])

            axes[a,b].set_ylabel(labels[b])

            axes[a,b].legend(markerscale = 20)

plt.figure(figsize = (8,6))

plt.scatter(neg.iloc[:,2], neg.iloc[:,0], s = 0.1, c = 'red', alpha = 1, label = 'Negative')

plt.scatter(pos.iloc[:,2], pos.iloc[:,0], s = 0.1, c = 'b', alpha = 1, label = 'Positive')

plt.xlabel(labels[2])

plt.ylabel(labels[0])

plt.legend(markerscale = 20)

plt.figure(figsize = (8,6))

plt.scatter(neg.iloc[:,5], neg.iloc[:,4], s = 0.1, c = 'red', alpha = 1, label = 'Negative')

plt.scatter(pos.iloc[:,5], pos.iloc[:,4], s = 0.1, c = 'b', alpha = 1, label = 'Positive')

plt.xlabel(labels[5])

plt.ylabel(labels[4])

plt.legend(markerscale = 20)


from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
xtrain, xtest, ytrain, ytest = train_test_split(data.iloc[:,0:8], data.iloc[:,8])
def metrics(test, pred):

    # tp: true positive, fp: false positive, tn: true negative, fnfalse negative

    tp, fp, tn, fn, score = 0, 0, 0, 0, 0

    for a, b in zip(test, pred):

        if b ==1:

            if a ==1: tp += 1

            else: fp +=1

        else:

            if a == 0: tn +=1

            else: fn +=1

    

    recall = tp / (tp + fn)

    accu = accuracy_score(test, pred)

    score = (recall + accu)/2

    

    return accu, tp,fp, tn, fn, recall, score





def print_metrics(test, pred):

    accu, tp,fp, tn, fn, recall, score = metrics(test,pred)

    print(f"Accuracy:       {accu: .3f} \nTrue positive:   {tp} \nFalse positive:  {fp} \nTrue negative:   {tn} \nFalse negative:  {fn} \nRecall:         {recall: .3f} \nScore:          {score: .3f}")
results = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])
LR = LogisticRegression()



LR.fit(xtrain, ytrain)

LR_prediction = LR.predict(xtest)

results['Logistic regression'] = metrics(ytest, LR_prediction)

print_metrics(ytest, LR_prediction)
results
GNB = GaussianNB()



GNB.fit(xtrain, ytrain)

GNB_prediction = GNB.predict(xtest)

results['Gaussian Naive Bayes'] = metrics(ytest, GNB_prediction)
KNC = KNeighborsClassifier(n_neighbors = 5)



KNC.fit(xtrain, ytrain)

KNC_prediction = KNC.predict(xtest)

results['K Neaghbors'] = metrics(ytest, KNC_prediction)
DTC = DecisionTreeClassifier(criterion = "entropy")



DTC.fit(xtrain, ytrain)

DTC_prediction = DTC.predict(xtest)

results['Decision tree'] = metrics(ytest, DTC_prediction)
RFC = RandomForestClassifier(n_estimators = 100)



RFC.fit(xtrain, ytrain)

RFC_prediction = RFC.predict(xtest)

results['Random forest'] = metrics(ytest, RFC_prediction)
MLPC = MLPClassifier(hidden_layer_sizes = (4), activation = "tanh", max_iter = 400)



MLPC.fit(xtrain, ytrain)

MLPC_prediction = MLPC.predict(xtest)

results['Neural Network'] = metrics(ytest, MLPC_prediction)
results
#plotting accuracy and recall in %





fig, ax = plt.subplots(figsize = (12,5))

w = 0.3

x = np.arange(len(results.columns))

labels = results.columns



rects1 = ax.bar(x - w/2, results.loc['Accuracy'] * 100, width = w, label = 'Accuracy')

rects2 = ax.bar(x + w/2, results.loc['Recall'] * 100, width = w, label = 'Recall')

ax.grid(axis = 'y')

ax.set_xticks(x)

ax.set_xticklabels(labels, rotation = 45)

ax.set_ylabel('Ratio (%)')

ax.legend(loc = 4)

ax.set_ylim(0,105)

ax.set_title('Model performance')



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{:4.1f}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),

                    textcoords="offset points",

                    ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)









plt.show()
names = ['DTC', 'RFC', 'MLPC']

d = {}                                   #the fitted models will be saved in this dictionary

results2 = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])

num = 200



for i in range(num):

    models = [DecisionTreeClassifier(), RandomForestClassifier(n_estimators = 100), MLPClassifier(hidden_layer_sizes = (4), activation = "tanh", max_iter = 400)] #definido dentro del bucle for para que en cada iteraci'on use un estado inicial pseudoaleatorio ditinto

    for name, model in zip(names, models):

        name += str(i)

        model.fit(xtrain, ytrain)

        prediction= model.predict(xtest)

        results2[name] = metrics(ytest, prediction)

        d[name] = model





for a in range(3):

    c = []

    for i in range(num):

        c.append(a + i*3)



    

    best = results2.iloc[-1,c].idxmax()

    results[best] = results2[best]
results
#plotting accuracy and recall in %





fig, ax = plt.subplots(figsize = (14,5))

w = 0.3

x = np.arange(len(results.columns))

labels = results.columns



rects1 = ax.bar(x - w/2, results.loc['Accuracy'] * 100, width = w, label = 'Accuracy')

rects2 = ax.bar(x + w/2, results.loc['Recall'] * 100, width = w, label = 'Recall')

ax.grid(axis = 'y')

ax.set_xticks(x)

ax.set_xticklabels(labels, rotation = 45)

ax.set_ylabel('Ratio (%)')

ax.legend(loc = 4)

ax.set_ylim(0,105)

ax.set_title('Model performance')



def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{:4.1f}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),

                    textcoords="offset points",

                    ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)









plt.show()
allDTC = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])

allRFC = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])

allMLPC = pd.DataFrame(index = ['Accuracy', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Recall', 'Score'])



for name in results2.columns:

    if name.startswith('DTC'):

        allDTC[name] = results2[name]

        

    elif name.startswith('RFC'):

        allRFC[name] = results2[name]

        

    else:

        allMLPC[name] = results2[name]



histfig, histaxes = plt.subplots(1,3, figsize = (16,4))

histaxes[0].hist(allDTC.loc['Score'], bins = 30)

histaxes[0].set_title('Histogram of Decision Tree \nClassifiers scores')

histaxes[1].hist(allRFC.loc['Score'], bins = 30)

histaxes[1].set_title('Histogram of Random Forest \nClassifiers scores')

histaxes[2].hist(allMLPC.loc['Score'], bins = 30)

histaxes[2].set_title('Histogram of Multi Layer Perceptron \nClassifiers scores')



plt.show()
worst_idx = allMLPC.loc['Score'].idxmin()



worst = d[worst_idx]

worst_prediction = worst.predict(xtest)

print_metrics(ytest, worst_prediction)
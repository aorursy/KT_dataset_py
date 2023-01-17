%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
testrun = False
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
def checkRelativeFrequencies(dataset):

    labeled_group = dataset.groupby('label')



    total_observation = len(dataset['label'])

    for label in range(0,10):

        print('RF {} = {:.3f} %, {}/{}'.format(label,labeled_group['label'].get_group(label).count()/total_observation*100, labeled_group['label'].get_group(label).count(),total_observation))

        

if testrun:

    checkRelativeFrequencies(train)
from sklearn.model_selection import StratifiedShuffleSplit



X = train.drop(['label'], axis = 1)

y = train['label']



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)



for train_index, test_index in sss.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]
if testrun:

    print(X_train.shape)

    print(X_test.shape)

    print(type(y_train))
if testrun:

    checkRelativeFrequencies(train)

    checkRelativeFrequencies(y_train.to_frame(name='label'))
%%time

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



testK = False

if testK:

    k_range = range(1,26)

    scores = []



    for k in k_range:

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train,y_train)

        y_pred = knn.predict(X_test)

        acc = accuracy_score(y_test,y_pred)

        scores.append(acc)

        print('k {} completed, accuracy: {}'.format(k, acc))



    print(scores)

    plt.plot(k_range, scores)

    plt.xlabel('Value of K')

    plt.ylabel('Testing accuracy')
if testrun:

    print(test.head())

    print(X.shape)

    print(y.shape)

    print(test.shape)
def predictTest(X, y, model, filepath, toFile=True):

    model.fit(X,y)

    y_pred = model.predict(test)



    if toFile:

        pd.DataFrame({"ImageId": list(range(1,len(test)+1)),"Label": y_pred}).to_csv(filepath, index=False,header=True)

        return

    else:

        return y_pred
%%time

if False: 

    predictTest(X, y, KNeighborsClassifier(n_neighbors=5), "Digit_recogniser_5.csv")
%%time

if False: 

    predictTest(X, y, KNeighborsClassifier(n_neighbors=3), "Digit_recogniser_3.csv")
if False: # deprecated

    sss33 = StratifiedShuffleSplit(n_splits=2, test_size=0.3333)



    for train_index, test_index in sss33.split(X, y):

        Xsplit, X2 = X.iloc[train_index], X.iloc[test_index]

        ysplit, y2 = y[train_index], y[test_index]



    sss50 = StratifiedShuffleSplit(n_splits=2, test_size=0.50)



    Xsplit, ysplit = Xsplit.reset_index(), ysplit.reset_index(drop=True)

    for train_index, test_index in sss50.split(Xsplit, ysplit):

        X1, X3 = Xsplit.iloc[train_index], Xsplit.iloc[test_index]

        y1, y3 = ysplit[train_index], ysplit[test_index]
%%time

from sklearn.utils import shuffle



Xshuffled, yshuffled = shuffle(X, y)#, random_state = 42)



test3Split, test3RF, testK3 = True, True, False



if test3Split:

    X1, X2, X3 = np.split(Xshuffled, [int(1 * len(Xshuffled)/3), int(2 * len(Xshuffled)/3)])

    y1, y2, y3 = np.split(yshuffled, [int(1 * len(yshuffled)/3), int(2 * len(yshuffled)/3)])
def checkRelativeFrequenciesOf3Split(ds, ds1, ds2, ds3): # Nasty function

    lg = ds.groupby('label')

    lg1 = ds1.groupby('label')

    lg2 = ds2.groupby('label')

    lg3 = ds3.groupby('label')



    total_observationsub = len(ds['label'])

    total_observation = len(ds1['label'])

    for label in range(0,10):

        lgRF = lg['label'].get_group(label).count()/total_observationsub*100

        lg1RF = lg1['label'].get_group(label).count()/total_observation*100

        lg2RF = lg2['label'].get_group(label).count()/total_observation*100

        lg3RF = lg3['label'].get_group(label).count()/total_observation*100

        print('RF {} = {:.3f}% // {:.3f}% {:.3f}% {:.3f}%'.format(label, lgRF, lg1RF, lg2RF, lg3RF))

if test3RF:

    checkRelativeFrequenciesOf3Split(train, y1.to_frame(name='label'), y2.to_frame(name='label'), y3.to_frame(name='label'))



    print(X1.shape, X2.shape, X3.shape)

    print(y1.shape, y2.shape, y3.shape)
%%time



if testK3:

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    k_range = range(1, 10)

    scores1, scores2, scores3 = [], [], []

    

    X1, y1 = X1.reset_index(), y1.reset_index(drop=True)

    for train_index, test_index in sss.split(X1, y1):

        X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]

        y1_train, y1_test = y1[train_index], y1[test_index]

    

    X2, y2 = X2.reset_index(), y2.reset_index(drop=True)

    for train_index, test_index in sss.split(X2, y2):

        X2_train, X2_test = X2.iloc[train_index], X2.iloc[test_index]

        y2_train, y2_test = y2[train_index], y2[test_index]

        

    X3, y3 = X3.reset_index(), y3.reset_index(drop=True)

    for train_index, test_index in sss.split(X3, y3):

        X3_train, X3_test = X3.iloc[train_index], X3.iloc[test_index]

        y3_train, y3_test = y3[train_index], y3[test_index]

    

    for k in k_range:

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X1_train, y1_train)

        y_pred = knn.predict(X1_test)

        acc1 = accuracy_score(y1_test, y_pred)

        scores1.append(acc1)

        

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X2_train, y2_train)

        y_pred = knn.predict(X2_test)

        acc2 = accuracy_score(y2_test, y_pred)

        scores2.append(acc2)

        

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X3_train, y3_train)

        y_pred = knn.predict(X3_test)

        acc3 = accuracy_score(y3_test, y_pred)

        scores3.append(acc3)

        print('k {}, acc: {:0.4f} {:0.4f} {:0.4f}'.format(k, acc1, acc2, acc3)) 



    print(scores1)

    print(scores2)

    print(scores3)

    plt.plot(k_range, scores1)

    plt.xlabel('Value of K')

    plt.ylabel('Testing accuracy')
%%time



import random



if test3Split:

    pred1 = predictTest(X1, y1, KNeighborsClassifier(n_neighbors=1), "x", toFile=False)

    pred2 = predictTest(X2, y2, KNeighborsClassifier(n_neighbors=1), "x", toFile=False)

    pred3 = predictTest(X3, y3, KNeighborsClassifier(n_neighbors=1), "x", toFile=False)



    predictionsSubteam = []

    random3Selection = 0

    for i in range(len(pred1)):

        results = [pred1[i], pred2[i], pred3[i]]



        if len(results) > len(set(results)):

            predictionsSubteam.append(max(set(results), key=results.count))

        else:

            predictionsSubteam.append(random.choice(results))

            random3Selection += 1



    pd.DataFrame({"ImageId": list(range(1,len(test)+1)),"Label": predictionsSubteam}).to_csv("Digit_recogniser_3Subsets.csv", index=False,header=True)

    print(len(test), len(predictionsSubteam))

    print(random3Selection)



    pd.DataFrame({"Label": pred1}).to_csv("pred1.csv", index=False,header=True)

    pd.DataFrame({"Label": pred2}).to_csv("pred2.csv", index=False,header=True)

    pd.DataFrame({"Label": pred3}).to_csv("pred3.csv", index=False,header=True)
%%time

test5Split, test5RF, testK5 = True, True, False



if test5Split:

    X1, X2, X3, X4, X5 = np.split(Xshuffled, [int(1 * len(Xshuffled)/5), int(2 * len(Xshuffled)/5), int(3 * len(Xshuffled)/5), int(4 * len(Xshuffled)/5)])

    y1, y2, y3, y4, y5 = np.split(yshuffled, [int(1 * len(yshuffled)/5), int(2 * len(yshuffled)/5), int(3 * len(yshuffled)/5), int(4 * len(yshuffled)/5)])
def checkRelativeFrequenciesOf5Split(ds, ds1, ds2, ds3, ds4, ds5): # Extension of nasty function

    lg = ds.groupby('label')

    lg1 = ds1.groupby('label')

    lg2 = ds2.groupby('label')

    lg3 = ds3.groupby('label')

    lg4 = ds4.groupby('label')

    lg5 = ds5.groupby('label')



    total_observationsub = len(ds['label'])

    total_observation = len(ds1['label'])

    for label in range(0,10):

        lgRF = lg['label'].get_group(label).count()/total_observationsub*100

        lg1RF = lg1['label'].get_group(label).count()/total_observation*100

        lg2RF = lg2['label'].get_group(label).count()/total_observation*100

        lg3RF = lg3['label'].get_group(label).count()/total_observation*100

        lg4RF = lg4['label'].get_group(label).count()/total_observation*100

        lg5RF = lg5['label'].get_group(label).count()/total_observation*100

        print('RF {} = {:.3f}% // {:.3f}% {:.3f}% {:.3f}% {:.3f}% {:.3f}%'.format(label, lgRF, lg1RF, lg2RF, lg3RF, lg4RF, lg5RF))
if test5RF:

    checkRelativeFrequenciesOf5Split(train, y1.to_frame(name='label'), y2.to_frame(name='label'), y3.to_frame(name='label'), y4.to_frame(name='label'), y5.to_frame(name='label'))



    print(X1.shape, X2.shape, X3.shape, X4.shape, X5.shape)

    print(y1.shape, y2.shape, y3.shape, y4.shape, y5.shape)
if False:

    X1, y1 = X1.reset_index(), y1.reset_index(drop=True)

    for train_index, test_index in sss.split(X1, y1):

        X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]

        y1_train, y1_test = y1[train_index], y1[test_index]



    def trainTestSplit(X, y):

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)



        X, y = X.reset_index(), y.reset_index(drop=True)

        for train_index, test_index in sss.split(X, y):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]

            y_train, y_test = y[train_index], y[test_index]



        return X_train, X_test, y_train, y_test



    x1, x2, x3, x4 = trainTestSplit(X1, y1)
%%time



if testK5:

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    k_range = range(1, 10)

    scores1, scores2, scores3, scores4, scores5 = [], [], [], [], []

    

    X1, y1 = X1.reset_index(), y1.reset_index(drop=True)

    for train_index, test_index in sss.split(X1, y1):

        X1_train, X1_test = X1.iloc[train_index], X1.iloc[test_index]

        y1_train, y1_test = y1[train_index], y1[test_index]

    

    X2, y2 = X2.reset_index(), y2.reset_index(drop=True)

    for train_index, test_index in sss.split(X2, y2):

        X2_train, X2_test = X2.iloc[train_index], X2.iloc[test_index]

        y2_train, y2_test = y2[train_index], y2[test_index]

        

    X3, y3 = X3.reset_index(), y3.reset_index(drop=True)

    for train_index, test_index in sss.split(X3, y3):

        X3_train, X3_test = X3.iloc[train_index], X3.iloc[test_index]

        y3_train, y3_test = y3[train_index], y3[test_index]

        

    X4, y4 = X4.reset_index(), y4.reset_index(drop=True)

    for train_index, test_index in sss.split(X4, y4):

        X4_train, X4_test = X4.iloc[train_index], X4.iloc[test_index]

        y4_train, y4_test = y4[train_index], y4[test_index]

        

    X5, y5 = X5.reset_index(), y5.reset_index(drop=True)

    for train_index, test_index in sss.split(X5, y5):

        X5_train, X5_test = X5.iloc[train_index], X5.iloc[test_index]

        y5_train, y5_test = y5[train_index], y5[test_index]

    

    for k in k_range:

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X1_train, y1_train)

        y_pred = knn.predict(X1_test)

        acc1 = accuracy_score(y1_test, y_pred)

        scores1.append(acc1)

        

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X2_train, y2_train)

        y_pred = knn.predict(X2_test)

        acc2 = accuracy_score(y2_test, y_pred)

        scores2.append(acc2)

        

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X3_train, y3_train)

        y_pred = knn.predict(X3_test)

        acc3 = accuracy_score(y3_test, y_pred)

        scores3.append(acc3)

        

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X4_train, y4_train)

        y_pred = knn.predict(X4_test)

        acc4 = accuracy_score(y4_test, y_pred)

        scores4.append(acc4)

        

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X5_train, y5_train)

        y_pred = knn.predict(X5_test)

        acc5 = accuracy_score(y5_test, y_pred)

        scores5.append(acc5)

        print('k {}, acc: {:0.4f} {:0.4f} {:0.4f} {:0.4f} {:0.4f}'.format(k, acc1, acc2, acc3, acc4, acc5)) 



    print(scores1)

    print(scores2)

    print(scores3)

    print(scores4)

    print(scores5)

    plt.plot(k_range, scores1)

    plt.xlabel('Value of K')

    plt.ylabel('Testing accuracy')
%%time

from collections import Counter



if test5Split:

    randomSelection = 0

    

    pred1 = predictTest(X1, y1, KNeighborsClassifier(n_neighbors=1), "x", toFile=False)

    pred2 = predictTest(X2, y2, KNeighborsClassifier(n_neighbors=1), "x", toFile=False)

    pred3 = predictTest(X3, y3, KNeighborsClassifier(n_neighbors=1), "x", toFile=False)

    pred4 = predictTest(X4, y4, KNeighborsClassifier(n_neighbors=1), "x", toFile=False)

    pred5 = predictTest(X5, y5, KNeighborsClassifier(n_neighbors=1), "x", toFile=False)

    print(len(test), len(pred1), len(pred2), len(pred3), len(pred4), len(pred5))



    predictionsSubteam = []

    for i in range(len(pred1)):

        results = [pred1[i], pred2[i], pred3[i], pred4[i], pred5[i]]



        if len(results) > len(set(results)):

            c = Counter(results).most_common()

            

            if len(c) > 1: # Check if there is a two way tie, e.g.: [1, 1, 2, 2, 3]

                if c[0][1] == c[1][1]:

                    predictionsSubteam.append(random.choice([c[0][0], c[1][0]]))

                    randomSelection += 1

                    continue

            

            predictionsSubteam.append(max(set(results), key=results.count))

        else:

            predictionsSubteam.append(random.choice(results))

            randomSelection += 1



    print(len(predictionsSubteam))

    print(randomSelection)

    pd.DataFrame({"ImageId": list(range(1,len(test)+1)),"Label": predictionsSubteam}).to_csv("Digit_recogniser_5Subsets.csv", index=False,header=True)



    pd.DataFrame({"Label": pred1}).to_csv("pred1.csv", index=False,header=True)

    pd.DataFrame({"Label": pred2}).to_csv("pred2.csv", index=False,header=True)

    pd.DataFrame({"Label": pred3}).to_csv("pred3.csv", index=False,header=True)

    pd.DataFrame({"Label": pred4}).to_csv("pred4.csv", index=False,header=True)

    pd.DataFrame({"Label": pred5}).to_csv("pred5.csv", index=False,header=True)
%%time

test10Split, test10RF, testK10 = False, False, False



lenXs = len(Xshuffled)/10

lenYs = len(yshuffled)/10



if test10Split:

    X1, X2, X3, X4, X5, X6, X7, X8, X9, X10 = np.split(Xshuffled, [int(lenXs), int(2 * lenXs), int(3 * lenXs), int(4 * lenXs), int(5 * lenXs), int(6 * lenXs), int(7 * lenXs), int(8 * lenXs), int(9 * lenXs)])

    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = np.split(yshuffled, [int(lenYs), int(2 * lenYs), int(3 * lenYs), int(4 * lenYs), int(5 * lenYs), int(6 * lenYs), int(7 * lenYs), int(8 * lenYs), int(9 * lenYs)])
def checkRelativeFrequenciesOf10Split(ds, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9, ds10): # The nastiest function

    lg = ds.groupby('label')

    lg1 = ds1.to_frame(name='label').groupby('label')

    lg2 = ds2.to_frame(name='label').groupby('label')

    lg3 = ds3.to_frame(name='label').groupby('label')

    lg4 = ds4.to_frame(name='label').groupby('label')

    lg5 = ds5.to_frame(name='label').groupby('label')

    lg6 = ds6.to_frame(name='label').groupby('label')

    lg7 = ds7.to_frame(name='label').groupby('label')

    lg8 = ds8.to_frame(name='label').groupby('label')

    lg9 = ds9.to_frame(name='label').groupby('label')

    lg10 = ds10.to_frame(name='label').groupby('label')



    total_observationsub = len(ds['label'])

    total_observation = len(ds1.to_frame(name='label')['label'])

    for label in range(0,10):

        lgRF = lg['label'].get_group(label).count()/total_observationsub*100

        lg1RF = lg1['label'].get_group(label).count()/total_observation*100

        lg2RF = lg2['label'].get_group(label).count()/total_observation*100

        lg3RF = lg3['label'].get_group(label).count()/total_observation*100

        lg4RF = lg4['label'].get_group(label).count()/total_observation*100

        lg5RF = lg5['label'].get_group(label).count()/total_observation*100

        lg6RF = lg6['label'].get_group(label).count()/total_observation*100

        lg7RF = lg7['label'].get_group(label).count()/total_observation*100

        lg8RF = lg8['label'].get_group(label).count()/total_observation*100

        lg9RF = lg9['label'].get_group(label).count()/total_observation*100

        lg10RF = lg10['label'].get_group(label).count()/total_observation*100

        print('RF {} = {:.3f}% // {:.3f}% {:.3f}% {:.3f}%, {:.3f}% {:.3f}% {:.3f}%, {:.3f}% {:.3f}% {:.3f}% {:.3f}%'.format(label, lgRF, lg1RF, lg2RF, lg3RF, lg4RF, lg5RF, lg6RF, lg7RF, lg8RF, lg9RF, lg10RF))
if test10RF:

    print(X1.shape, X2.shape, X3.shape, X4.shape, X5.shape, X6.shape, X7.shape, X8.shape, X9.shape, X10.shape)

    print(y1.shape, y2.shape, y3.shape, y4.shape, y5.shape, y6.shape, y7.shape, y8.shape, y9.shape, y10.shape)

    

    checkRelativeFrequenciesOf10Split(train, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10)
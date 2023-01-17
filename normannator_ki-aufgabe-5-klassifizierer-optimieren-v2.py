import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import
def read_data(filename):
    f = open('../input/'+filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

trainset1 = read_data("train.arff")
trainset2 = read_data("train-skewed.arff")
testset = pd.read_csv('../input/test-skewed.csv', index_col=0, header=0, names=['Id (String)', 'X', 'Y'])
testset = testset[['X','Y']].as_matrix()

#Train1
x_train1 = []
y_train1 = []
for i in trainset1:
    x_train1.append([i[0],i[1]])
    if (i[2] == -1):
        y_train1.append(-1)
    else:
        y_train1.append(1)

#Train1 with Train and Test split (2/3)
x_train1_train = []
y_train1_train = []
x_train1_test = []
y_train1_test = []
print(len(x_train1))
for i in range(0, len(x_train1)*2//3):
    x_train1_train.append(x_train1[i])
    y_train1_train.append(y_train1[i])
    
for i in range(len(x_train1)*2//3, len(x_train1)):
    x_train1_test.append(x_train1[i])
    y_train1_test.append(y_train1[i])
        
        ##############################################################################################################
        
#Train2
x_train2 = []
y_train2 = []
for i in trainset2:
    x_train2.append([i[0],i[1]])
    if (i[2] == -1):
        y_train2.append(-1)
    else:
        y_train2.append((1))

        #Train1 with Train and Test split (2/3)
x_train2_train = []
y_train2_train = []
x_train2_test = []
y_train2_test = []
print(len(x_train2))
for i in range(0, len(x_train2)*2//3):
    x_train2_train.append(x_train2[i])
    y_train2_train.append(y_train2[i])
    
for i in range(len(x_train2)*2//3, len(x_train2)):
    x_train2_test.append(x_train2[i])
    y_train2_test.append(y_train2[i])        
        
        ##############################################################################################################
        
#Test
x_testFinal = []
y_testFinal = []
for i in testset:
    x_testFinal.append([i[0],i[1]])
    
for i in range(0,200):
    y_testFinal.append(-1)

for i in range(200,400):
    y_testFinal.append(1)
    
def quote(istSet, sollSet):
    korrekt = 0
    falsch = 0
    zaehler = 0
    while (zaehler < len(istSet)):
        if (istSet[zaehler] == sollSet[zaehler]):
            korrekt += 1
        else:
            falsch += 1
    print("Korrekt: " + str(korrekt) + "  Falsch: " + str(falsch))
    print("Fehlerquote: " + str(100 * fehler / korrekt) + "%")

print("Train1: Länge=",len(x_train1),len(y_train1))
print("Train2: Länge=",len(x_train2),len(y_train2))
print("Test: Länge=",len(x_testFinal),len(y_testFinal))
from sklearn import model_selection
from sklearn import ensemble

rf_classifierka = ensemble.RandomForestClassifier()

rf_classifierka.fit(x_train1_train, y_train1_train)
predict = rf_classifierka.predict(x_train1_test)

print("Länge des Testsets:",len(predict))

def gibRichtigkeit(x1,x2):
    i = 0
    richtig = 0
    while (i < len(x1)):
        if (x1[i] == x2[i]):
            richtig += 1
        i += 1
    return richtig/len(x1)

print ("Genauigkeit (Train1): ", gibRichtigkeit(predict,y_train1_test))
maxRichtigkeit = gibRichtigkeit(predict,y_train1_test)
best_Depth = 2 #default
depth = 1
for i in range(1000):
    rf_classifierka = ensemble.RandomForestClassifier(max_depth=depth)
    rf_classifierka.fit(x_train1_train, y_train1_train)
    predict = rf_classifierka.predict(x_train1_test)
    aktRichtigkeit = gibRichtigkeit(predict,y_train1_test)
    if (aktRichtigkeit > maxRichtigkeit):
        best_Depth = depth
        maxRichtigkeit = aktRichtigkeit
    depth+=1
        
print ("best depth:",best_Depth,"  maximale Richtigkeit:",maxRichtigkeit)
best_estimators = 10 #default
estimators = 1
for i in range(100):
    rf_classifierka = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=estimators)
    rf_classifierka.fit(x_train1_train, y_train1_train)
    predict = rf_classifierka.predict(x_train1_test)
    aktRichtigkeit = gibRichtigkeit(predict,y_train1_test)
    if (aktRichtigkeit > maxRichtigkeit):
        best_estimators = estimators
        maxRichtigkeit = aktRichtigkeit
    estimators += 1
        
print ("best estimators:",best_estimators,"  maximale Richtigkeit:",maxRichtigkeit)
best_criterion = "gini"
rf_classifierka = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=best_estimators,criterion="entropy")
rf_classifierka.fit(x_train1_train, y_train1_train)
predict = rf_classifierka.predict(x_train1_test)
aktRichtigkeit = gibRichtigkeit(predict,y_train1_test)
if (aktRichtigkeit > maxRichtigkeit):
    best_criterion="entropy"
    maxRichtigkeit = aktRichtigkeit

print("best criterion:",best_criterion,"  maximale Richtigkeit:",maxRichtigkeit)
best_split = 2 #default
split = 2
for i in range(1000):
    rf_classifier = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=best_estimators,criterion=best_criterion,min_samples_split=split)
    rf_classifier.fit(x_train1_train, y_train1_train)
    predict = rf_classifier.predict(x_train1_test)
    aktRichtigkeit = gibRichtigkeit(predict,y_train1_test)
    if (aktRichtigkeit > maxRichtigkeit):
        best_split = split
        maxRichtigkeit = aktRichtigkeit
    split += 1
        
print ("best split:",best_split,"  maximale Richtigkeit:",maxRichtigkeit)
best_leaf = 1 #default
leaf = 1
for i in range(1000):
    rf_classifier = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=best_estimators,criterion=best_criterion,min_samples_split=best_split
                                                     ,min_samples_leaf=leaf)
    rf_classifier.fit(x_train1_train, y_train1_train)
    predict = rf_classifier.predict(x_train1_test)
    aktRichtigkeit = gibRichtigkeit(predict,y_train1_test)
    if (aktRichtigkeit > maxRichtigkeit):
        best_leaf = leaf
        maxRichtigkeit = aktRichtigkeit
    leaf += 1
        
print ("best leaf:",best_leaf,"  maximale Richtigkeit:",maxRichtigkeit)
rf_classifier = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=best_estimators,criterion=best_criterion,min_samples_split=best_split
                                                     ,min_samples_leaf=best_leaf)

rf_classifier.fit(x_train2_train, y_train2_train)
predict = rf_classifier.predict(x_train2_test)

print("Länge des Testsets:",len(predict))
print ("Genauigkeit (Train2): ", gibRichtigkeit(predict,y_train2_test))


#best_Depth
depth = 1
for i in range(1000):
    rf_classifier = ensemble.RandomForestClassifier(max_depth=depth,n_estimators=best_estimators,criterion=best_criterion,min_samples_split=best_split
                                                     ,min_samples_leaf=best_leaf)
    rf_classifier.fit(x_train2_train, y_train2_train)
    predict = rf_classifier.predict(x_train2_test)
    aktRichtigkeit = gibRichtigkeit(predict,y_train2_test)
    if (aktRichtigkeit > maxRichtigkeit):
        best_Depth = depth
        maxRichtigkeit = aktRichtigkeit
    depth+=1
print ("best depth:",best_Depth,"  maximale Richtigkeit:",maxRichtigkeit)

#estimators
estimators = 1
for i in range(100):
    rf_classifier = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=estimators,criterion=best_criterion,min_samples_split=best_split
                                                     ,min_samples_leaf=best_leaf)
    rf_classifier.fit(x_train2_train, y_train2_train)
    predict = rf_classifier.predict(x_train2_test)
    aktRichtigkeit = gibRichtigkeit(predict,y_train2_test)
    if (aktRichtigkeit > maxRichtigkeit):
        best_estimators = estimators
        maxRichtigkeit = aktRichtigkeit
    estimators += 1
print ("best estimators:",best_estimators,"  maximale Richtigkeit:",maxRichtigkeit)
    
#split
split = 2
for i in range(1000):
    rf_classifier = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=best_estimators,criterion=best_criterion,min_samples_split=split
                                                     ,min_samples_leaf=best_leaf)
    rf_classifier.fit(x_train2_train, y_train2_train)
    predict = rf_classifier.predict(x_train2_test)
    aktRichtigkeit = gibRichtigkeit(predict,y_train2_test)
    if (aktRichtigkeit > maxRichtigkeit):
        best_split = split
        maxRichtigkeit = aktRichtigkeit
    split += 1
print ("best split:",best_split,"  maximale Richtigkeit:",maxRichtigkeit)
    
#leaf
leaf = 1
for i in range(1000):
    rf_classifier = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=best_estimators,criterion=best_criterion,min_samples_split=best_split
                                                     ,min_samples_leaf=leaf)
    rf_classifier.fit(x_train2_train, y_train2_train)
    predict = rf_classifier.predict(x_train2_test)
    aktRichtigkeit = gibRichtigkeit(predict,y_train2_test)
    if (aktRichtigkeit > maxRichtigkeit):
        best_leaf = leaf
        maxRichtigkeit = aktRichtigkeit
    leaf += 1
print ("best leaf:",best_leaf,"  maximale Richtigkeit:",maxRichtigkeit)
rf_classifier = ensemble.RandomForestClassifier(max_depth=best_Depth,n_estimators=best_estimators,criterion=best_criterion,min_samples_split=best_split
                                                     ,min_samples_leaf=best_leaf)

rf_classifier.fit(x_train1, y_train1)
rf_classifier.fit(x_train2, y_train2)
predict = rf_classifier.predict(x_testFinal)

print("Länge des Testsets:",len(predict))
print(len(y_testFinal))
print ("Genauigkeit (Test): ", gibRichtigkeit(predict,y_testFinal))

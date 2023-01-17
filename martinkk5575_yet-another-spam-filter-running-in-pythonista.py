import csv

def readRawData(sourcePath, idxText, idxLabel=None):

    with open(sourcePath, 'r', encoding='latin-1') as csvfile:

        dataReader = csv.reader(csvfile, delimiter=',', quotechar='"')

        next(dataReader) #skip header

        if idxLabel==None:

            return [row[idxText] for row in dataReader]

        return [(row[idxText], row[idxLabel]) for row in dataReader]

    

raw = readRawData('/kaggle/input/sms-spam-collection-dataset/spam.csv', 1, 0)

raw[0:10]
# get X and y

X = [text for (text, cls) in raw]

y = [cls for (text, cls) in raw]



# split train and test

from math import floor

n = len(X)

percentage = 0.2

p = floor(n * percentage)



X_train = X[:-p]

y_train = y[:-p]

X_test = X[-p:]

y_test = y[-p:]



print('{} * {} = {}'.format(n, percentage, p))

print(len(X_train))

print(len(X_test))
from naivebayes import NaiveBayes



# without removing Stop Words

naiveBayes = NaiveBayes(removeStopWords=False)

naiveBayes.train(X_train, y_train)

y_predict = naiveBayes.predictAll(X_test)



# removing Stop Words

naiveBayes_NoStopWords = NaiveBayes(removeStopWords=True)

naiveBayes_NoStopWords.train(X_train, y_train)

y_predict_nsw = naiveBayes_NoStopWords.predictAll(X_test)
def getConfusionMatrix(forClass, gold, predictions):

    tp=0; tn=0; fp=0; fn=0

    n = len(gold)

    for i in range(n):

        c = gold[i]

        predict = predictions[i]

        isPositive = c==forClass

        isCorrect = c==predict

        if (isPositive):

            if (isCorrect):

                tp += 1

            else:

                fp += 1

        else:

            if (isCorrect):

                tn += 1

            else:

                fn += 1

    return (tp, tn, fp, fn)



def calcQualityMeasures(tp, tn, fp, fn):

    precision = tp / (tp + fn)

    recall = tp / (tp + fp)

    if (precision==0 and recall==0):

        f1 = float('nan')

    else:

        f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)





# Now lets calculate the measures

tp, tn, fp, fn = getConfusionMatrix('ham', y_test, y_predict)

precision, recall, f1 = calcQualityMeasures(tp, tn, fp, fn)



tp_nsw, tn_nsw, fp_nsw, fn_nsw = getConfusionMatrix('ham', y_test, y_predict_nsw)

precision_nsw, recall_nsw, f1_nsw = calcQualityMeasures(tp_nsw, tn_nsw, fp_nsw, fn_nsw)





# And finally show some output

def printRow(label, left, right, percentage=False):

    if percentage:

        print("{:>20} {:>20.2f} {:20.2f}".format(label, left * 100, right * 100))

    else:

        print("{:>20} {:>20} {:>20}".format(label, left, right))



    

printRow('Naive Bayes', 'with StopWords', 'without StopWords')

printRow('TP', tp, tp_nsw)

printRow('TN', tn, tn_nsw)

printRow('FP', fp, fp_nsw)

printRow('FN', fn, fn_nsw)

printRow('Precision', precision, precision_nsw, percentage=True)

printRow('Recall', recall, recall_nsw, percentage=True)

printRow('F1', f1, f1_nsw, percentage=True)
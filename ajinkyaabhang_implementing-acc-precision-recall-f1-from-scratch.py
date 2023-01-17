ground_truth = [1,0,1,1,1,0,1,0,1,1]

prediction   = [1,1,1,0,1,0,1,1,1,0]
#True Positive

def true_positive(ground_truth, prediction):

    tp = 0

    for gt, pred in zip(ground_truth, prediction):

        if gt == 1 and pred == 1:

            tp +=1

    return tp
#True Negative

def true_negative(ground_truth, prediction):

    tn = 0

    for gt, pred in zip(ground_truth, prediction):

        if gt == 0 and pred == 0:

            tn +=1

    return tn
#False Positive

def false_positive(ground_truth, prediction):

    fp = 0

    for gt, pred in zip(ground_truth, prediction):

        if gt == 0 and pred == 1:

            fp +=1

    return fp
#False Negative

def false_negative(ground_truth, prediction):

    fn = 0

    for gt, pred in zip(ground_truth, prediction):

        if gt == 1 and pred == 0:

            fn +=1

    return fn
true_positive(ground_truth, prediction)
true_negative(ground_truth, prediction)
false_positive(ground_truth, prediction)
false_negative(ground_truth, prediction)
def accuracy(ground_truth, prediction):

    tp = true_positive(ground_truth, prediction)  

    fp = false_positive(ground_truth, prediction)  

    fn = false_negative(ground_truth, prediction)  

    tn = true_negative(ground_truth, prediction)  

    acc_score = (tp + tn)/ (tp + tn + fp + fn)  

    return acc_score
accuracy(ground_truth, prediction)
### Lets comapre with Sklearn accuracy_score

from sklearn import metrics

metrics.accuracy_score(ground_truth, prediction)
def precision(ground_truth, prediction):

    tp = true_positive(ground_truth, prediction)  

    fp = false_positive(ground_truth, prediction)  

    prec = tp/ (tp + fp)  

    return prec
precision(ground_truth, prediction)
### Lets comapre with Sklearn precision

from sklearn import metrics

metrics.precision_score(ground_truth, prediction)
def recall(ground_truth, prediction):

    tp = true_positive(ground_truth, prediction)  

    fn = false_negative(ground_truth, prediction)  

    prec = tp/ (tp + fn)  

    return prec
recall(ground_truth, prediction)
### Lets comapre with Sklearn precision

from sklearn import metrics

metrics.recall_score(ground_truth, prediction)
def f1(ground_truth, prediction):

    p = precision(ground_truth, prediction)

    r = recall(ground_truth, prediction)

    f1_score = 2 * p * r/ (p + r) 

    return f1_score
f1(ground_truth, prediction)
### Lets comapre with Sklearn precision

from sklearn import metrics

metrics.f1_score(ground_truth, prediction)
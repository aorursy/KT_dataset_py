import pandas as pd

from pandas import Series,DataFrame

import numpy as np



from datetime import date



from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

from scipy import interp



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns
#data

df=pd.read_csv('../input/Combined_News_DJIA.csv')

df['Combined']=df.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
#train data

train=df.loc[(pd.to_datetime(df["Date"]) <= date(2014,12,31)),['Label','Combined']]

train.head()
#test data

test=df.loc[(pd.to_datetime(df["Date"]) > date(2014,12,31)),['Label','Combined']]

test.head()
#Text pre-processing



def text_process(text):

    """

    Takes in a string of text, then performs the following:

    1. Tokenizes and removes punctuation

    2. Removes  stopwords

    3. Stems

    4. Returns a list of the cleaned text

    """

     

    # tokenizing

    tokenizer = RegexpTokenizer(r'\w+')

    text_processed=tokenizer.tokenize(text)

    

    # removing any stopwords

    text_processed = [word.lower() for word in text_processed if word.lower() not in stopwords.words('english')]

    

    # steming

    porter_stemmer = PorterStemmer()

    

    text_processed = [porter_stemmer.stem(word) for word in text_processed]

    

    try:

        text_processed.remove('b')

    except: 

        pass



    return text_processed
def ROCCurves (Actual, Predicted):

    '''

    Plot ROC curves for the multiclass problem

    based on http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    '''

    # Compute ROC curve and ROC area for each class

    n_classes=2

    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    for i in range(n_classes):

        fpr[i], tpr[i], _ = roc_curve(Actual.values, Predicted)

        roc_auc[i] = auc(fpr[i], tpr[i])



    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = roc_curve(Actual.ravel(), Predicted.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################

    # Plot ROC curves for the multiclass problem



    # Compute macro-average ROC curve and ROC area



    # First aggregate all false positive rates



    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



    # Then interpolate all ROC curves at this points

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += interp(all_fpr, fpr[i], tpr[i])



    # Finally average it and compute AUC

    mean_tpr /= n_classes



    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves

    plt.figure()

    plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         linewidth=2)



    plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         linewidth=2)



    for i in range(n_classes):

        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'

                                   ''.format(i, roc_auc[i]))



    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Some extension of Receiver operating characteristic to multi-class')

    plt.legend(loc="lower right")
def heatmap(data):

  fig, ax = plt.subplots()

  heatmap = sns.heatmap(data, cmap=plt.cm.Blues,annot=True, annot_kws={"size": 8})

  # put the major ticks at the middle of each cell

  # want a more natural, table-like display

  ax.xaxis.tick_top()

  # rotate the

  plt.xticks(rotation=90)

  plt.yticks(rotation=0)

  plt.tight_layout()
def plot_classification_report(classification_report):

    lines = classification_report.split('\n')

    classes = []

    plotMat = []

    for line in lines[2 : (len(lines) - 3)]:

        t = line.split()

        classes.append(t[0])

        v = [float(x) for x in t[1: len(t) - 1]]

        plotMat.append(v)

    aveTotal = lines[len(lines) - 1].split()

    classes.append('avg/total')

    vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]

    plotMat.append(vAveTotal)

    df_classification_report = DataFrame(plotMat, index=classes,columns=['precision', 'recall', 'f1-score'])

    heatmap(df_classification_report)
def plot_confusion_matrix(confusion_matrix,classes=['0','1']):

    df_confusion_matrix = DataFrame(confusion_matrix, index=classes,columns=classes)

    df_confusion_matrix

    heatmap(df_confusion_matrix)
def Evaluation (Actual, Predicted):

    '''

        Prints and plots

        - classification report

        - confusion matrix

        - ROC-AUC

    '''

    print (classification_report(Actual,Predicted))

    plot_classification_report(classification_report(Actual,Predicted))

    print ('Confussion matrix:\n', confusion_matrix(Actual,Predicted))

    plot_confusion_matrix(confusion_matrix(Actual,Predicted))

    print ('ROC-AUC: ' + str(roc_auc_score(Actual,Predicted)))

    ROCCurves (Actual,Predicted)
#Creating a Data Pipeline for LogisticRegression classifier

lgr_pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ LogisticRegression classifier

])
lgr_pipeline.fit(train['Combined'],train['Label'])
predictions = lgr_pipeline.predict(test['Combined'])
Evaluation (test["Label"], predictions)
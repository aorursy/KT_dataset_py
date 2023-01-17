# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline  
human = pd.read_table('../input/humandnadata/human_data/human_data.txt')
chimp = pd.read_table('../input/chimpanzee-and-dog-dna/chimp_data.txt')
dog = pd.read_table('../input/chimpanzee-and-dog-dna/dog_data.txt')
human.head(3535)

human['sequence'][15]
def noise_check(arr):
    m=[]
    count = 0
    for i in range(0,len(arr)):
        for j in arr['sequence'][i]:
            if j!='A' and j!='T' and j!='C' and j!='G':
                count = count + 1
                m.append(j)
    if count>0:
        print(m)
        print('Noise Count',count)
    else:
        print('No noise')
noise_check(human)
noise_check(chimp)
noise_check(dog)
def remove_noise(arr):
    for i in range(0,len(arr)):
        arr['sequence'][i] = arr['sequence'][i].replace('N','')
            
remove_noise(human)
remove_noise(chimp)
remove_noise(dog)
noise_check(human)
noise_check(chimp)
noise_check(dog)
def missing_check(arr):
    classat=[]
    seqat=[]
    count = 0
    for i in range(0,len(arr)):
        if arr['sequence'][i]=='':
            seqat.append(i)
            count = count + 1
        if arr['class'][i]=='' or arr['class'][i]>6:
            classat.append(i)
            count = count + 1
    if count==0:
        print('No missing value')
    else:
        print('missing count = ',count)
    
missing_check(human)
missing_check(chimp)
missing_check(dog)
chimp.head()

dog.head()
# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
human['words'] = human.apply(lambda x: getKmers(x['sequence']), axis=1)
human = human.drop('sequence', axis=1)
chimp['words'] = chimp.apply(lambda x: getKmers(x['sequence']), axis=1)
chimp = chimp.drop('sequence', axis=1)
dog['words'] = dog.apply(lambda x: getKmers(x['sequence']), axis=1)
dog = dog.drop('sequence', axis=1)
print(human['words'][0])
human_texts = list(human['words'])
for item in range(len(human_texts)):
    human_texts[item] = ' '.join(human_texts[item])
y_h = human.iloc[:, 0].values 
y_h
len(y_h)
human_texts[0]


chimp_texts = list(chimp['words'])
for item in range(len(chimp_texts)):
    chimp_texts[item] = ' '.join(chimp_texts[item])
y_c = chimp.iloc[:, 0].values                       # y_c for chimp

dog_texts = list(dog['words'])
for item in range(len(dog_texts)):
    dog_texts[item] = ' '.join(dog_texts[item])
y_d = dog.iloc[:, 0].values   
# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
# The n-gram size of 4 was previously determined by testing
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(human_texts)
X_chimp = cv.transform(chimp_texts)
X_dog = cv.transform(dog_texts)
print(X.shape)
print(X_chimp.shape)
print(X_dog.shape)


human['class'].value_counts().sort_index().plot.bar()




chimp['class'].value_counts().sort_index().plot.bar()


dog['class'].value_counts().sort_index().plot.bar()
# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y_h, 
                                                    test_size = 0.20, 
                                                    random_state=42)


print(X_train.shape)
print(X_test.shape)


### Multinomial Naive Bayes Classifier ###
# The alpha parameter was determined by grid search previously
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
# Predicting the chimp, dog and worm sequences
y_pred_chimp = classifier.predict(X_chimp)
y_pred_dog = classifier.predict(X_dog)
# performance on chimp genes
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_c, name='Actual'), pd.Series(y_pred_chimp, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_c, y_pred_chimp)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
# performance on dog genes
print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_d, name='Actual'), pd.Series(y_pred_dog, name='Predicted')))
accuracy, precision, recall, f1 = get_metrics(y_d, y_pred_dog)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
from sklearn.metrics import roc_curve  
import seaborn as sns
def plot_roc_curve(fper, tper):  
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
len(y_test)
len(y_pred)

from sklearn.preprocessing import label_binarize
y_pred = label_binarize(y_pred, classes=[0,1,2,3,4,5,6])
y_test = label_binarize(y_test, classes=[0,1,2,3,4,5,6])
y_pred_chimp = label_binarize(y_pred_chimp, classes=[0,1,2,3,4,5,6])
y_pred_dog = label_binarize(y_pred_dog, classes=[0,1,2,3,4,5,6])
y_c = label_binarize(y_c, classes=[0,1,2,3,4,5,6])
y_d = label_binarize(y_d, classes=[0,1,2,3,4,5,6])
from sklearn.metrics import roc_curve, auc
from itertools import cycle
def get_roc(y_t,y_pd,val):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(7):
        fpr[i], tpr[i], _ = roc_curve(y_t[:, i], y_pd[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    lw=2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','blue','yellow'])
    for i, color in zip(range(7), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve {}'.format(val))
    plt.legend(loc="lower right")
    plt.show()


get_roc(y_test,y_pred,'Human')
get_roc(y_c,y_pred_chimp,'Chimpanzee')
get_roc(y_d,y_pred_dog,'Dog')
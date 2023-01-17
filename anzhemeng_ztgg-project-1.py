# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import missingno as msno 

import scipy

from scipy.sparse import hstack

from PIL import Image

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings



warnings.filterwarnings('ignore')



from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

%matplotlib inline

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
data.head()
plt.title('10 Most Active Users')

data['UserId'].value_counts(sort=True).nlargest(10).plot.bar()
print('There are', str(data.shape[0]), 'records in total.')
plt.title('10 Most Rated Products')

data['ProductId'].value_counts(sort=True).nlargest(10).plot.bar()
HelpfulnessNumerator0 = data[data['HelpfulnessNumerator'] == 0]['HelpfulnessNumerator'].value_counts()

HelpfulnessNumerator1 = data[data['HelpfulnessNumerator'] == 1]['HelpfulnessNumerator'].value_counts()

HelpfulnessNumerator2 = data[data['HelpfulnessNumerator'] == 2]['HelpfulnessNumerator'].value_counts()

HelpfulnessNumerator3 = data[data['HelpfulnessNumerator'] == 3]['HelpfulnessNumerator'].value_counts()

HelpfulnessNumeratorMoreThan3 = data[data['HelpfulnessNumerator'] > 3]['HelpfulnessNumerator'].value_counts()



labels = '0', '1', '2', '3', 'more than 3'

sizes = [HelpfulnessNumerator0.values.item(), HelpfulnessNumerator1.values.item(), HelpfulnessNumerator2.values.item(), 

         HelpfulnessNumerator3.values.item(), HelpfulnessNumeratorMoreThan3.values.sum()]

explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig, ax = plt.subplots()

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title(' Portions of Amount of Helpfulness Labels')

plt.show()
HelpfulnessDenominator0 = data[data['HelpfulnessDenominator'] == 0]['HelpfulnessDenominator'].value_counts()

HelpfulnessDenominator1 = data[data['HelpfulnessDenominator'] == 1]['HelpfulnessDenominator'].value_counts()

HelpfulnessDenominator2 = data[data['HelpfulnessDenominator'] == 2]['HelpfulnessDenominator'].value_counts()

HelpfulnessDenominator3 = data[data['HelpfulnessDenominator'] == 3]['HelpfulnessDenominator'].value_counts()

HelpfulnessDenominatorMoreThan3 = data[data['HelpfulnessDenominator'] > 3]['HelpfulnessDenominator'].value_counts()



labels = '0', '1', '2', '3', 'more than 3'

sizes = [HelpfulnessDenominator0.values.item(), HelpfulnessDenominator1.values.item(), HelpfulnessDenominator2.values.item(), 

         HelpfulnessDenominator3.values.item(), HelpfulnessDenominatorMoreThan3.values.sum()]

explode = (0, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig, ax = plt.subplots()

ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title(' Portions of Amount of Comments Watched')

plt.show()
plt.title('Scores')

data['Score'].value_counts().plot.bar()
fig = plt.figure(figsize=(14, 10))

ax = fig.add_subplot(121)

text = data.Summary.values

wordcloud = WordCloud(

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

plt.title('Summary Keywords')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)



ax = fig.add_subplot(122)

text = data.Text.values

wordcloud = WordCloud(

    background_color = 'white',

    stopwords = STOPWORDS).generate(str(text))

plt.title('Text Keywords')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
msno.matrix(data)
msno.heatmap(data)
# fives = data.loc[data['Score'] == 5]

# fives = fives.sample(frac=0.5)

# data = pd.concat([data.loc[data['Score'] != 5], fives])
data['Text'].loc[data['Text'].isna()] = ''

data['Summary'].loc[data['Summary'].isna()] = ''
X_train, X_valid, y_train, y_valid = train_test_split(data.drop('Score', axis=1), data['Score'], test_size=0.2, random_state=42)
X_train['Helpful'] = X_train['HelpfulnessNumerator']

X_train['Unhelpful'] = X_train['HelpfulnessDenominator'] -X_train['HelpfulnessNumerator']



X_valid['Helpful'] = X_valid['HelpfulnessNumerator']

X_valid['Unhelpful'] = X_valid['HelpfulnessDenominator'] - X_valid['HelpfulnessNumerator']



scaler = StandardScaler()



# only fit on the train set

scalerFitter = scaler.fit(X_train[['Helpful', 'Unhelpful', 'Time']])

X_train[['Helpful', 'Unhelpful', 'Time']] = scalerFitter.transform(X_train[['Helpful', 'Unhelpful', 'Time']])

X_valid[['Helpful', 'Unhelpful', 'Time']] = scalerFitter.transform(X_valid[['Helpful', 'Unhelpful', 'Time']])



X_train = X_train.drop(['HelpfulnessDenominator','HelpfulnessNumerator'], axis=1)

X_valid = X_valid.drop(['HelpfulnessDenominator','HelpfulnessNumerator'], axis=1)
text_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english', ngram_range=(1, 2))

summary_vectorizer = TfidfVectorizer(input='content', analyzer='word', stop_words='english', ngram_range=(1, 2))



text_fitter = text_vectorizer.fit(data['Text'])

text_matrix_train = text_fitter.transform(X_train['Text'])

text_matrix_valid = text_fitter.transform(X_valid['Text'])



summary_fitter = summary_vectorizer.fit(data['Summary'])

summary_matrix_train = summary_fitter.transform(X_train['Summary'])

summary_matrix_valid = summary_fitter.transform(X_valid['Summary'])
# the shape of text and summary matrices

text_matrix_train, summary_matrix_train, text_matrix_valid, summary_matrix_valid
# features in text

text_vectorizer.get_feature_names()
# features in summary

summary_vectorizer.get_feature_names()
OHE = OneHotEncoder(sparse=True)

ID_fitter = OHE.fit(data[['ProductId', 'UserId']])

IDs_train = ID_fitter.transform(X_train[['ProductId', 'UserId']])

IDs_valid = ID_fitter.transform(X_valid[['ProductId', 'UserId']])
numerical_train = scipy.sparse.csr_matrix(X_train[['Helpful', 'Unhelpful', 'Time']].values)

numerical_valid = scipy.sparse.csr_matrix(X_valid[['Helpful', 'Unhelpful', 'Time']].values)
X_train = hstack([text_matrix_train, summary_matrix_train, numerical_train, IDs_train])

X_valid = hstack([text_matrix_valid, summary_matrix_valid, numerical_valid, IDs_valid])
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler(random_state=42)

X_train, y_train = ros.fit_resample(X_train, y_train)
plt.title('Scores')

y_train.value_counts().plot.bar()
def CVKFold(k, X, y, model):

    np.random.seed(1)

    #reproducibility

    

    highest_accuracy = float('inf')

    best_model = None



    kf = KFold(n_splits = k,shuffle =True)

    #CV loop

    

    for train_index,test_index in kf.split(X):#generation of the sets

    #generate the sets    

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        #model fitting

        model.fit(X_train,y_train)

        y_test_pred = model.predict(X_test)

    

        test_accuracy = mean_squared_error(y_test_pred, y_test)

        print("The accuracy is " + str(test_accuracy))

        if test_accuracy < highest_accuracy:

          best_model = model

          highest_accuracy = test_accuracy



    print("The highest accuracy is " + str(highest_accuracy))

    return best_model, highest_accuracy
# Logistics Regression

model = LogisticRegression(random_state = 0)

model = model.fit(X_train, y_train)

clf_Log, accuracy_Log = CVKFold(3, X_train, y_train, model)

# Decision Tree

model = DecisionTreeClassifier(random_state = 0, max_depth=20)

clf_DTree, accuracy_DTree = CVKFold(3, X_train, y_train, model)

# # Random Forest

model = RandomForestClassifier(random_state = 0, max_depth=20)

clf_RF, accuracy_RF = CVKFold(3, X_train, y_train, model)
accuracies = {accuracy_Log: clf_Log, accuracy_DTree: clf_DTree, accuracy_RF: clf_RF}

clf = accuracies[min([accuracy_Log, accuracy_DTree, accuracy_RF])]
print('The most accurate classifier is:', clf)
from sklearn.preprocessing import label_binarize



y_valid = label_binarize(y_valid, classes=[1, 2, 3, 4, 5])

y_train = label_binarize(y_train, classes=[1, 2, 3, 4, 5])
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import roc_curve, auc



model = OneVsRestClassifier(clf)

y_score = model.fit(X_train, y_train).decision_function(X_valid)



# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(5):

    fpr[i], tpr[i], _ = roc_curve(y_valid[:, i], y_score[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = roc_curve(y_valid.ravel(), y_score.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
plt.figure()

lw = 2

plt.plot(fpr[2], tpr[2], color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
from scipy import interp

from itertools import cycle



n_classes = 5



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

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Some extension of Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import precision_score



print('Precision Score: ')

precision_score(y_valid, model.predict(X_valid), average='macro')
from sklearn.metrics import recall_score



print('Recall Score: ')

recall_score(y_valid, model.predict(X_valid), average='macro')
from sklearn.metrics import f1_score



print('F-1 Score: ')

f1_score(y_valid, model.predict(X_valid), average='macro')
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score



# For each class

precision = dict()

recall = dict()

average_precision = dict()

for i in range(n_classes):

    precision[i], recall[i], _ = precision_recall_curve(y_valid[:, i],

                                                        y_score[:, i])

    average_precision[i] = average_precision_score(y_valid[:, i], y_score[:, i])



# A "micro-average": quantifying score on all classes jointly

precision["micro"], recall["micro"], _ = precision_recall_curve(y_valid.ravel(), y_score.ravel())

average_precision["micro"] = average_precision_score(y_valid, y_score, average="micro")

print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
plt.figure()

plt.step(recall['micro'], precision['micro'], where='post')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title(

    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'

    .format(average_precision["micro"]))
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])



plt.figure(figsize=(7, 8))

f_scores = np.linspace(0.2, 0.8, num=4)

lines = []

labels = []

for f_score in f_scores:

    x = np.linspace(0.01, 1)

    y = f_score * x / (2 * x - f_score)

    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)

    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))



lines.append(l)

labels.append('iso-f1 curves')

l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)

lines.append(l)

labels.append('micro-average Precision-recall (area = {0:0.2f})'

              ''.format(average_precision["micro"]))



for i, color in zip(range(n_classes), colors):

    l, = plt.plot(recall[i], precision[i], color=color, lw=2)

    lines.append(l)

    labels.append('Precision-recall for class {0} (area = {1:0.2f})'

                  ''.format(i, average_precision[i]))



fig = plt.gcf()

fig.subplots_adjust(bottom=0.25)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Extension of Precision-Recall curve to multi-class')

plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))





plt.show()
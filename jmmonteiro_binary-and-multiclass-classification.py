import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')

plt.rcParams['figure.figsize'] = (14,10)
bible = pd.read_csv('../input/bible_data_set.csv')

bible.shape
# Books in the Old Testament (OT) and New Testament (NT)

OT_books = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 'Judges',

            'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles',

            'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs', 'Ecclesiastes',

            'Song of Solomon', 'Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel',

            'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk',

            'Zephaniah', 'Haggai',    'Zechariah',    'Malachi']



NT_books = ['Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans', '1 Corinthians',

            '2 Corinthians', 'Galatians', 'Ephesians', 'Philippians', 'Colossians',

            '1 Thessalonians', '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon',

            'Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude',

            'Revelation']



# New variables with the books and testaments as categories

books = pd.unique(bible.book)

book_codes = np.arange(1,len(books)+1)

book_codes = np.concatenate([books.reshape((-1,1)),book_codes.reshape((-1,1))], axis=1)



bible['book_code'] = np.nan

bible['testament'] = ''

bible['testament_code'] = np.nan

for i in range(0,bible.shape[0]):

    ind = bible.at[i,'book'] == book_codes[:,0]

    bible.at[i,'book_code'] = book_codes[ind,1]

    

    if bible.at[i,'book'] in OT_books:

        bible.at[i,'testament'] = 'Old Testament'

        bible.at[i,'testament_code'] = 0

    elif bible.at[i,'book'] in NT_books:

        bible.at[i,'testament'] = 'New Testament'

        bible.at[i,'testament_code'] = 1

    else:

        raise Exception('Book not found: ' + bible.at[i,'book'])



bible.book_code = bible.book_code.astype(int)

bible.book_code = pd.Categorical(bible.book_code)

bible.testament_code = bible.testament_code.astype(int)

bible.testament_code = pd.Categorical(bible.testament_code)

    

bible.head()
bible = bible.sample(frac=1, random_state=7).reset_index(drop=True)

bible.head()
print('# verses in the OT: ' + str(np.sum(bible.testament_code==0)))

print('# verses in the NT: ' + str(np.sum(bible.testament_code==1)))
n_cv_folds = 10



X = bible.text

y = np.asarray(bible.testament_code)

from sklearn.metrics import accuracy_score



from sklearn.svm import LinearSVC

machine = LinearSVC(class_weight='balanced')



clf = machine



from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()



# Use the stratified KFold strategy, in order to preserve the proportion of each class in all the folds

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=n_cv_folds)



acc = []

class_acc = []

for train_index, test_index in cv.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    X_train = vect.fit_transform(X_train)

    X_test = vect.transform(X_test)

    

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    

    # General accuracy

    acc.append(accuracy_score(y_test,y_pred))

    

    # Class accuracy

    ind = y_test == 0

    OT_acc = np.sum(y_test[ind] == y_pred[ind])

    OT_acc = OT_acc/np.sum(ind)

    

    ind = y_test == 1

    NT_acc = np.sum(y_test[ind] == y_pred[ind])

    NT_acc = NT_acc/np.sum(ind)

    

    class_acc.append([OT_acc, NT_acc])
class_acc = np.array(class_acc)

print('Accuracy: ' + str(np.mean(acc)))

print('Balanced Accuracy: ' + str(np.mean(class_acc[:])))
y = bible.book_code



from sklearn.metrics import confusion_matrix



from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(machine)



acc = []

class_acc = []

conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))

for train_index, test_index in cv.split(X, y):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    X_train = vect.fit_transform(X_train)

    X_test = vect.transform(X_test)

    

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    

    # Class accuracy

    fold_class_acc = []

    for c in np.unique(y):

        ind = y_test == c

        dummy = np.sum(y_test[ind] == y_pred[ind])

        fold_class_acc.append(dummy/np.sum(ind))

    class_acc.append(fold_class_acc)    

    

    # Accuracy

    acc.append(accuracy_score(y_test,y_pred))

    

    # Confusion matrix

    conf_mat = conf_mat + confusion_matrix(y_test,y_pred)
print('Accuracy: ' + str(np.mean(acc)))

print('Balanced Accuracy: ' + str(np.mean(class_acc[:])))
conf_mat_norm = np.transpose(np.transpose(conf_mat)/np.sum(conf_mat,axis=1))



ax = sns.heatmap(conf_mat_norm,

            xticklabels=books, yticklabels=books,

            linewidths=.5, cbar_kws={'label': 'Fraction of samples in row'})

ax.set(xlabel = 'Predicted Label', ylabel='True label')



plt.show()
class_acc_plot = np.array(class_acc)

m = np.mean(class_acc,axis=0)



i = np.argsort(m)

class_acc_plot = class_acc_plot[:,i]

books_plot = books[i]



class_acc_plot = pd.DataFrame(class_acc_plot, columns=books_plot)



ax = sns.boxplot(data=class_acc_plot)

ax.set(xlabel = 'Book', ylabel='Class accuracy')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)



plt.show()
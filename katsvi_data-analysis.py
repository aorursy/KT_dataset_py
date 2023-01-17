# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as graph

import matplotlib.pyplot as plt

import itertools

import re

import time

from sklearn.metrics import roc_curve, confusion_matrix, classification_report



from string import punctuation

from IPython.display import display



from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier



from sklearn.calibration import CalibratedClassifierCV

#from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

def GetWordFrequency(train_comments):

    return pd.Series(' '.join(train_comments).lower().split()).value_counts()
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train_number_of_rows = len(train.index)

train_number_of_cols = len(train.columns)

print("Тренировочная выборка - Количество строк: ", train_number_of_rows)

print("Тренировочная выборка - Количество столбцов: ", train_number_of_cols)



test_number_of_rows = len(test.index)

test_number_of_cols = len(test.columns)

print("\nТестовая выборка - Количество строк: ", test_number_of_rows)

print("Тестовая выборка - Количество столбцов: ", test_number_of_cols)

number_of_classes = train_number_of_cols - test_number_of_cols

print("Количество классов: ", number_of_classes)
output_classes = train.columns.values[test_number_of_cols:]

print("Классы: ", output_classes)

train_input = train[train.columns[0:test_number_of_cols]]

train_output = train[output_classes]



percentage_test = round(test_number_of_rows / (test_number_of_rows + train_number_of_rows) * 100, 2)

print("Тестовая выборка = ", percentage_test, "% от всего набора данных.")
words_frequency = GetWordFrequency(train.comment_text)

print("Количество уникальных слов: ", len(words_frequency))

lengths = train.comment_text.str.len()

print("Длина самого маленького комментария: ", lengths.min())

print("Длина самого длинного комментария: ", lengths.max())

print("Средняя длинна комментария: ", lengths.mean())

print("Среднеквадратическое отклонение длины комментария: ", lengths.std())
def comment_lengths(comments):

    for comment in comments:

        yield len(comment)

        

def comments_capitals_counter(comments):

    for comment in comments:

        yield sum(1 for char in comment if char.isupper())

        

def ratio_feature_vs_length(feature_rows, comment_lengths):

    for feature, comment_length in zip(feature_rows, comment_lengths):

        yield float(feature) / float(comment_length)
train['comment_length'] = [comment_length for comment_length in comment_lengths(train['comment_text'])]

#длина комментария

train['number_of_capitals'] = [capitals for capitals in comments_capitals_counter(train['comment_text'])]

#количество заглавных букв

train['ratio_capitals_length'] = [ratio for ratio in ratio_feature_vs_length(train['number_of_capitals'], train['comment_length'])]

#отношение количества заглавных букв к длине комментария
def comments_char_counter(comments, char):

    for comment in comments:

        yield comment.count(char)

        

        

def comments_chars_counter(comments, chars):

    for comment in comments:

        yield sum(comment.count(char) for char in chars)

        

        

def all_words_counter(comments):

    for comment in comments:

        yield len(comment.split())

        

        

def unique_words_counter(comments):

    for comment in comments:

        yield len(set(word for word in comment.split()))
train['carriage_returns'] = [carriage_returns for carriage_returns in comments_chars_counter(train['comment_text'], '\n\r')]

#количество символов переноса строки

train['ratio_carriage_returns_length'] = [ratio for ratio in ratio_feature_vs_length(train['carriage_returns'], train['comment_length'])]

#отношение количества символов переноса строки к длине комментария

train['spaces'] = [spaces for spaces in comments_char_counter(train['comment_text'], ' ')]

#количество пробелов

train['ratio_spaces_length'] = [ratio for ratio in ratio_feature_vs_length(train['spaces'], train['comment_length'])]

#отношение количества пробелов к длине комментария
train['exclamation_marks'] = [exclamation_marks for exclamation_marks in comments_char_counter(train['comment_text'], '!')]

#количество восклицательных знаков

train['ratio_exclamation_marks_length'] = [ratio for ratio in ratio_feature_vs_length(train['exclamation_marks'], train['comment_length'])]

#отношение количества восклицательных знаков к длине комментария

train['question_marks'] = [question_marks for question_marks in comments_char_counter(train['comment_text'], '?')]

#количество вопросительных знаков

train['ratio_question_marks_length'] = [ratio for ratio in ratio_feature_vs_length(train['question_marks'], train['comment_length'])]

#отношение количества вопросительных знаков  к длине комментария
train['special_chars'] = [special_chars for special_chars in comments_chars_counter(train['comment_text'], '#*$')]

#количество специальных символов вроде $*#

train['ratio_special_chars_length'] = [ratio for ratio in ratio_feature_vs_length(train['special_chars'], train['comment_length'])]

#отношение количества специальных символов к длине комментария
train['all_words'] = [all_words for all_words in all_words_counter(train['comment_text'])]

#количество всех слов

train['unique_words'] = [unique_words for unique_words in unique_words_counter(train['comment_text'])]

#количество уникальных слов

train['ratio_all_vs_unique_words'] = [ratio for ratio in ratio_feature_vs_length(train['unique_words'], train['all_words'])]

#их отношение
new_features = ['comment_length', 'number_of_capitals', 'ratio_capitals_length', 

                'exclamation_marks', 'ratio_exclamation_marks_length', 'question_marks', 'ratio_question_marks_length',

                'carriage_returns', 'ratio_carriage_returns_length', 'special_chars', 'spaces',

                'ratio_spaces_length', 'ratio_special_chars_length', 'all_words', 'unique_words', 

                'ratio_all_vs_unique_words']



rows = [{column:train[feature].corr(train[column]) for column in output_classes} for feature in new_features]

train_correlations = pd.DataFrame(rows, index=new_features)

display(train_correlations)

graph.heatmap(train_correlations, vmin=-0.3, vmax=0.3, center=0.0)
remove_features = ['spaces', 'special_chars', 'question_marks', 'ratio_special_chars_length']

new_features = [feature for feature in new_features if feature not in remove_features]

train.drop(remove_features, axis=1, inplace=True, errors = 'ignore')
train_output_correlations = train_output.corr()

display(train_output_correlations)

graph.heatmap(train_output_correlations, vmin=0, vmax=1, center=0.5)
import string

def ascii_chars_from_text(text, ascii_chars):

    for char in text:

        if char in ascii_chars:

            yield char





def clean_text(comments):

    train_comments_cleaned = []

    ascii_chars = set(string.printable)

    

    for comment in comments:

        # переведем в нижний регистр.

        comment = comment.lower().strip(' ')

        # удалим символы переноса строки.

        comment = re.sub("\\n", " ", comment)

        # удалим никнеймы.

        comment = re.sub("\[\[.*\]", "", comment)

        comment = re.sub("[\$\*&%#@\"]", " ", comment)

        comment = re.sub('\W', ' ', comment)

        # уберем не ascii символы.

        comment = ''.join([char for char in ascii_chars_from_text(comment, ascii_chars)])

        comment = re.sub("fck", "fuck", comment)

        comment = re.sub("f ck", "fuck", comment)

        comment = re.sub("fagget", "faggot", comment)

        comment = re.sub("you re", "you are", comment)

        comment = re.sub("\d", " ", comment)



        train_comments_cleaned.append(comment)

        

    return pd.Series(train_comments_cleaned).astype(str)



def clean(train_perspective):

    classes = ['comment', 'is_toxic']

    train = train_perspective.loc[:, classes]



    train.comment = clean_text(train.comment)

    train.is_toxic = train.is_toxic.astype(np.int64)

    

    return train


train.comment_text = clean_text(train.comment_text)

test.comment_text = clean_text(test.comment_text)



words_frequency = GetWordFrequency(train.comment_text)



print("\nКоличество уникальных слов: ", len(words_frequency))

print(words_frequency[:10])
X_train, X_test, y_train, y_test = train_test_split(train.comment_text, 

                                                    train[output_classes],

                                                    random_state=45, 

                                                    test_size=0.30)
import numpy as np



from sklearn.metrics import roc_auc_score

from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer





def calculate_predictions(classifier, X_train, X_test, y_train, y_test, output_classes):

    roc_auc = []

    predictions = np.zeros(shape=(len(y_test), len(output_classes)))

    predictions_int = np.zeros(shape=(len(y_test), len(output_classes)))

    

    for i, output_class in enumerate(output_classes):

        classifier.fit(X_train, y_train[output_class])

        

        predictions[:, i] = classifier.predict_proba(X_test)[:, 1]

        predictions_int[:, i] = classifier.predict(X_test)

        auc = roc_auc_score(y_test[output_class], predictions[:, i])

        roc_auc.append(auc)

        print("\nКласс: ", output_class)

        print("ROC AUC: ", auc)



    print("\nМногоклассовая ROC AUC: ", np.mean(roc_auc))

    

    return roc_auc, predictions, predictions_int





def apply_best_model(dataset_name, train, test):

    vectorizer = TfidfVectorizer(sublinear_tf=True)

    X_train_vect = vectorizer.fit_transform(train.comment)

    X_test_vect = vectorizer.transform(test.comment)



    svm = LinearSVC(C=0.22)

    classifier = CalibratedClassifierCV(svm)



    classifier.fit(X_train_vect, train.is_toxic)



    predictions = classifier.predict_proba(X_test_vect)[:, 1]

    auc_roc = roc_auc_score(test.is_toxic, predictions)

    print(dataset_name + " AUC ROC: ", auc_roc)

    



def multi_class_rocauc(y_test, predictions):

    total_roc_auc = 0

    number_of_classes = len(y_test[0])

    for j in range(number_of_classes):

        total_roc_auc += roc_auc_score(y_test[:, j], predictions[:, j])



    return total_roc_auc / number_of_classes
def plot_auc_roc(output_classes, y_test, predictions, auc_roc):

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

    left  = 0.125  

    right = 0.9    

    bottom = 0.1   

    top = 0.9     

    wspace = 0.6   

    hspace = 0.5   

    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)



    for i, label in enumerate(output_classes):

        fpr, tpr, threshold = roc_curve(y_test.values[:, i], predictions[:, i])



        row = int(i / 3)

        col = i % 3



        axes[row, col].set_title('ROC AUC  ' + label)

        axes[row, col].plot(fpr, tpr, 'b', label = 'ROC AUC = %0.2f' % auc_roc[i])

        axes[row, col].legend(loc = 'lower right')

        axes[row, col].plot([0, 1], [0, 1],'r--')

        plt.xlim([0, 1])

        plt.ylim([0, 1])

        axes[row, col].set_xlabel('False Positive Rate')

        axes[row, col].set_ylabel('True Positive Rate')
start_time = time.time()

vectorizer = TfidfVectorizer(norm='l1')

X_train_vect = vectorizer.fit_transform(X_train)



X_test_vect = vectorizer.transform(X_test)



classifier = MultinomialNB(alpha=0.005)



auc_roc_nb, predictions, predictions_int = calculate_predictions(classifier, X_train_vect, X_test_vect, y_train, y_test, output_classes)

end_time = time.time()

total_time_nb = end_time - start_time

print("\nВремя выполнение: ", total_time_nb, "секунд")
plot_auc_roc(output_classes, y_test, predictions, auc_roc_nb)
start_time = time.time()

vectorizer = TfidfVectorizer(sublinear_tf=True, norm='l2')

X_train_vect = vectorizer.fit_transform(X_train)



X_test_vect = vectorizer.transform(X_test)



svm = LinearSVC(C=0.12)

classifier = CalibratedClassifierCV(svm)



auc_roc_svm, predictions, predictions_int = calculate_predictions(classifier, X_train_vect, X_test_vect, y_train, y_test, output_classes)

end_time = time.time()

total_time_svm = end_time - start_time

print("\nВремя выполнения: ", total_time_svm, "секунд")
plot_auc_roc(output_classes, y_test, predictions, auc_roc_svm)
start_time = time.time()

vectorizer = TfidfVectorizer(sublinear_tf=True)

X_train_vect = vectorizer.fit_transform(X_train)



X_test_vect = vectorizer.transform(X_test)



classifier = SGDClassifier(loss='log', penalty='l2', alpha=0.000004)



auc_roc_sgd, predictions, predictions_int = calculate_predictions(classifier, X_train_vect, X_test_vect, y_train, y_test, output_classes)

end_time = time.time()

total_time_sgd = end_time - start_time

print("\nВремя выполнения: ", total_time_sgd, "секунд")
plot_auc_roc(output_classes, y_test, predictions, auc_roc_sgd)
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(train[new_features], 

                                                    train[output_classes], 

                                                    random_state=45, 

                                                    test_size=0.50)



params = {

    'n_estimators': 300,

    'objective': 'binary:logistic',

    'learning_rate': 0.1,

    'subsample': 0.5,

    'colsample_bytree': 0.5,

    'eval_metric': 'auc',

    'seed': 1024

}



classifier = xgb.XGBClassifier(**params)



auc_roc_xgb, predictions, predictions_int = calculate_predictions(classifier, X_train, X_test, y_train, y_test, output_classes)

end_time = time.time()

total_time_xgb = end_time - start_time

print("\nВремя выполнения: ", total_time_xgb, "секунд")



xgb.plot_importance(classifier)

plt.show()
plot_auc_roc(output_classes, y_test, predictions, auc_roc_xgb)
models = {

    'Модель': ['Linear SVM Regularized', 'Stochastic Gradient Descent', 'Multinomial Naive Bayes', 'Extreme Gradient Boosting Tree'],

    'Точность': [np.mean(auc_roc_svm), np.mean(auc_roc_sgd), np.mean(auc_roc_nb), np.mean(auc_roc_xgb)],

    'Время': [total_time_svm, total_time_sgd, total_time_nb, total_time_xgb]

}

summary = pd.DataFrame(models)

display(summary)
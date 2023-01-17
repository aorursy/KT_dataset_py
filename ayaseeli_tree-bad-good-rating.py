import pandas as pd

import numpy as np

import os

os.listdir("../input")
imdb = pd.read_csv("../input/movie_metadata.csv")

imdb = imdb.reindex(np.random.permutation(imdb.index))



# check dimension dataframe

print(imdb.shape)

# 0 to drop index with na value, 0 to drop column with na value

df = imdb.dropna(axis = 0)

print(df.shape)
# Make histogram split 25 bins

import matplotlib.pyplot as plt

plt.hist(df['imdb_score'], bins=25)

plt.title("Distribution of IMDB score")

plt.show()
# change all object data type to integer category code

olist = list(df.select_dtypes(['object']))

for col in olist:

    df[col] = df[col].astype('category').cat.codes
# split dataset to training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop("imdb_score",1), df["imdb_score"],test_size=0.20, random_state=42)
import pylab

import scipy.stats as stats



# function to visualize model accuracy

def plot_model(x,y,model):

    print("R^2: %f" % mod.score(x,y))

    

    y_fitted = model.predict(x)

    residual = y - y_fitted



    plt.figure(figsize=(7,5))



    plt.subplot(221)

    plt.scatter(y_fitted, y)

    plt.title("fitted value vs actual value")

    plt.xlabel("fitted value")

    plt.ylabel("actual value")



    plt.subplot(222)

    plt.hist(residual, bins=50)

    plt.title("residual histogram")





    plt.subplot(223)

    stats.probplot(residual, dist="norm", plot=pylab)





    plt.subplot(224)

    plt.scatter(y_fitted,residual)

    plt.title("fitted value vs residual")

    plt.xlabel("fitted value")

    plt.ylabel("residual")



    plt.tight_layout()

    plt.show()
from sklearn import linear_model

mod = linear_model.LinearRegression()

mod.fit(X_train, y_train)

plot_model(X_train,y_train,mod)
y_test_fitted = mod.predict(X_test)

plt.scatter(y_test_fitted, y_test)

plt.title("predicted value vs actual value")

plt.xlabel("predicted value")

plt.ylabel("actual value")

plt.show()



print("Score: %f" % mod.score(X_test,y_test))

print("SSE: %f" % sum((y_test_fitted-y_test)**2))
imdb_score = np.array(df['imdb_score'])

percent25 = np.percentile(imdb_score,33)

percent75 = np.percentile(imdb_score,67)



clean_list = (imdb_score>percent75) + (imdb_score<percent25)

classifier_clean_data = df[clean_list]

classifier_clean_data = classifier_clean_data.drop("imdb_score",1)



imdb_level = list(df['imdb_score'][clean_list]>percent75)

imdb_level = [int(i) for i in imdb_level]



x_train, x_test, y_train, y_test = train_test_split(classifier_clean_data, imdb_level, 

                                                    test_size=0.25, random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

import itertools



# function to plot testing result

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
from sklearn.model_selection import cross_val_score



avg_score_list = []

for i in range(1,20):

    mod = DecisionTreeClassifier(max_depth = i)

    scores = cross_val_score(mod, x_train, y_train, cv=20)

    avg_score_list.append(np.mean(scores))

    

plt.plot(range(1,20),avg_score_list,'--',linewidth=3)

plt.axvline(avg_score_list.index(max(avg_score_list))+1, linewidth=3)

plt.show()



print("max score reached with depth %d" % (avg_score_list.index(max(avg_score_list))+1))
def plot_test(x,y,model):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, [i[1] for i in model.predict_proba(x)])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.plot(false_positive_rate, true_positive_rate, "b", label='AUC %0.2f' % (roc_auc))

    plt.title("AUC Curve")

    plt.show()



    auc_score = roc_auc_score(y, [i[1] for i in mod.predict_proba(x)])

    cm = confusion_matrix(y,mod.predict(x))

    plot_confusion_matrix(cm,classes = ["bad movie","good movie"],normalize=False)

    print("AUC Score: %f" % auc_score)

    print("Accuracy: %f" % (sum(mod.predict(x) == y)/float(len(y))))





mod = DecisionTreeClassifier(max_depth = 7)

mod.fit(x_train,y_train)

plot_test(x_test,y_test,mod)
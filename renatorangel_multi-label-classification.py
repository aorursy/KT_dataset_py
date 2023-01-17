import re

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, classification_report



from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from sklearn.svm import LinearSVC



from sklearn.pipeline import Pipeline

import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer



from sklearn.model_selection import cross_val_score, GridSearchCV, learning_curve

from sklearn.feature_extraction.text import TfidfVectorizer



from skmultilearn.problem_transform import ClassifierChain, LabelPowerset

from sklearn.metrics import f1_score



from matplotlib.pyplot import figure, show

plt.style.use('ggplot')

from seaborn import countplot, kdeplot



pd.set_option('display.max_rows', 150)
# got this code from here https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, 

                        title, 

                        X, 

                        y,

                        ylim=None, 

                        cv=None,

                        n_jobs=None, 

                        train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=(12,6))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator,

                                                            X,

                                                            y,

                                                            cv=cv,

                                                            scoring="f1_macro",

                                                            n_jobs=n_jobs,

                                                            train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
df = pd.read_csv("/kaggle/input/mpst-movie-plot-synopses-with-tags/mpst_full_data.csv")
df.head()
df.shape
df["tags"].str.split(",").head()
mlb = MultiLabelBinarizer()

tags = mlb.fit_transform(df["tags"].str.split(", "))

categories = mlb.classes_
df = pd.concat([df, pd.DataFrame(tags, columns=mlb.classes_)], axis=1)
df.shape
df.head()
counts = []

categories = mlb.classes_

for i in categories:

    counts.append((i, df[i].sum()))

df_stats = pd.DataFrame(counts, columns=['category', 'number_of_synopsis'])
df_stats.sort_values('number_of_synopsis', ascending=False).plot(x='category', y='number_of_synopsis', kind='bar', legend=False, grid=True, figsize=(24, 6))

plt.title("Total de sinopses por categoria")

plt.ylabel('Quantidade', fontsize=12)

plt.xlabel('categoria', fontsize=12)
rowsums = df.iloc[:,6:].sum(axis=1)

x = rowsums.value_counts()



plt.figure(figsize=(12,6))

ax = sns.barplot(x.index, x.values)

plt.title("Tags por sinopse")

plt.ylabel('Quantidade', fontsize=12)

plt.xlabel('Total de categorias', fontsize=12)
figure(figsize=(12,6))

kdeplot(df["plot_synopsis"].str.len())

show()
print('Numero de dados faltantes nas sinopses:')

sum(df['plot_synopsis'].isna())
print(len(df.columns))

categories = df_stats.loc[df_stats["number_of_synopsis"] > 400, "category"].tolist()

df.drop(df_stats.loc[df_stats["number_of_synopsis"] < 400, "category"].tolist(), axis=1, inplace=True)

len(df.columns)

df = df[df.sum(axis=1) != 0]
train = df[(df["split"] == "train") | (df["split"] == "val")]

test = df[df["split"] == "test"]
X_train = train.plot_synopsis

X_test = test.plot_synopsis

print(X_train.shape)

print(X_test.shape)
pipe= Pipeline(steps=[("preprocessing", TfidfVectorizer(stop_words=stop_words, min_df=10, max_features=15000, max_df=.8)),

                      ("classifier", ClassifierChain())])





search_space = [{"classifier__classifier": [(LinearSVC(max_iter=3000))],

                 "classifier__classifier__C": [0.1, 1, 10, 100]}]



gridsearch = GridSearchCV(pipe, 

                          search_space, 

                          cv=5, 

                          n_jobs=-1, 

                          scoring = 'f1_macro') 



gridsearch.fit(X_train, train[categories])



model = gridsearch.best_estimator_







print(gridsearch.best_estimator_)

print(gridsearch.best_score_)
title = "Learning Curves SVM"



lc_svm = plot_learning_curve(model, title, X_train, train[categories], ylim=(0.0,1.0), cv=5, n_jobs=-1)

lc_svm.show()
predictions = model.predict(X_test)
f1_score(test[categories], predictions, average="macro")
def predict_synopsis(series_synopsis, model, categories):

    tags = pd.DataFrame(gridsearch.predict(series_synopsis).todense(), columns=categories) 

    return tags.loc[:,(tags == 1.0).values.tolist()[0]].columns.tolist()

    
tags = predict_synopsis(pd.Series(test.iloc[96]["plot_synopsis"]), model, categories)



tags
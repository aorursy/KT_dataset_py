import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

from pandas import ExcelWriter

from pandas import ExcelFile

from sklearn.model_selection import GridSearchCV
df = pd.read_csv("C:/Users/egor/Desktop/выгрузка.csv", sep=';', skiprows=10)

df.head(20)
df.isnull().describe()
df.drop(columns = ["Примечание","№\nдокумента","№ п/п","Дата\nрегистрации","Исх. №\nДата"], inplace = True)
df.head()
df_train = pd.DataFrame()

df_train['Адресат'] = df['Адресат']

df_train['текст'] = df['Автор'] + ' ' + df['Краткое\nсодержание']
df_train.head()
len(df_train)
df_train.dropna(inplace = True)
from sklearn.feature_extraction.text import TfidfVectorizer    

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df_train['текст'])

# print(vectorizer.get_feature_names())



print(X.shape)
from sklearn import preprocessing

encoders = {}

encoders["Адресат"] = preprocessing.LabelEncoder()

df_train["Адресат"] = encoders["Адресат"].fit_transform(df_train["Адресат"])
df_train["Адресат"]
df_train_tfidf = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names() )
df_train_tfidf
X = df_train_tfidf.values
X.shape
y = df_train["Адресат"].values
y.shape
feature_names = df_train_tfidf.columns
feature_names[:5]
X_train = X[ : int(len(X)*0.8) ]

X_test = X[int(len(X)*0.8) : ]



y_train = y[ : int(len(y)*0.8) ]

y_test = y[int(len(y)*0.8) : ]

X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.ensemble import RandomForestClassifier
# knn_grid = {'n_neighbors': np.array(np.linspace(1, 100, 100), dtype='int')}



param_grid = {'max_depth': [i for i in range(2, 10)],

              'min_samples_split': [i for i in range(2, 10)],

#               'max_features': [2, len(X_train[0])-1]

              }



alg = RandomForestClassifier()

gs = GridSearchCV(alg, param_grid, cv=5)

gs.fit(X_train, y_train)



# best_params_ содержит в себе лучшие подобранные параметры, best_score_ лучшее качество

print()

gs.best_params_, gs.best_score_



    
def plot_feature_importances(model, columns, features_to_display):

    imp = pd.Series(data = model.best_estimator_.feature_importances_, 

                    index=columns).sort_values(ascending=False)

    plt.figure(figsize=(7,5))

    plt.title("Feature importance")

    ax = sns.barplot(y=imp.index[:features_to_display], x=imp.values[:features_to_display], orient='h')
# encoded_data.columns

plot_feature_importances(gs, feature_names, 30)
alg = RandomForestClassifier()



alg.fit(X_train, y_train)
preds = alg.predict(X_test)
from sklearn.metrics import classification_report



print(classification_report(y_test, preds))
encoders["Адресат"].inverse_transform([18, 143, 57])
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
# От такие у нас данные
df_train = pd.read_csv("../input/train.csv")
df_train.head()
# Заимпортим все шо нада
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
# Вот мы сделаем объект типа векторайзер
vect = TfidfVectorizer()
# А воть классификатор
clf = SGDClassifier(loss = 'modified_huber')
# А вот создали модель, которая будет вначале векторайзить, а потом классифицировать
model = Pipeline([('vect', vect), ('clf', clf)])
# Посмотрим какие у нашей модели есть параметры
model.get_params().keys()
# Некоторые из них переберем, прям как в прошлых лабах
# Перебор будет овер долгим, поэтому переберем чуть-чуть параметров

model_grid = {
    'vect__min_df': [0, 0.0001, 0.001],
    'vect__max_df': [1.0, 0.9, 0.8],
    'clf__tol': [1e-3, 1e-4, 1e-2]
}
model_search = GridSearchCV(model, model_grid, cv=StratifiedKFold(4, random_state=3), 
                            n_jobs=4, verbose=True)

X = df_train['question_text'].values
y = df_train['target'].values
model_search.fit(X, y)
# Посмотри на параметры лучшей модельки
model_search.best_params_
# Сохраним лучшую моделю
best_model = model_search.best_estimator_
%%time

# Теперь сделаем предсказания, причем предсказывать будем вероятности, они нам интереснее
model_preds = cross_val_predict(best_model, X, y,
                          cv=StratifiedKFold(4, random_state=3), n_jobs=4,
                          method='predict_proba')
# Переберем пороги для определения, плохой вопрос или норм. Таким образом найдел лучший порог
from sklearn.metrics import f1_score

thresholds = np.arange(0.05, 0.95, 0.05)
opt_threshold = 0
opt_score = 0
for threshold in thresholds:
    tmp_pred = list(map(lambda x: 1 if x[1]>threshold else 0, model_preds))
    tmp_score = f1_score(y, tmp_pred)
    if tmp_score > opt_score:
        opt_score = tmp_score
        opt_threshold = threshold
    print(f"F1-score = {tmp_score} with threshold = {threshold}")

print('----------------------------------------------------')
print(f"Optimal threshold is {opt_threshold}")
# Итак, теперь обучим моделю и сделаем предсказания на тестовых данных

best_model.fit(X, y)
test_data = pd.read_csv("../input/test.csv")["question_text"].values
test_id = pd.read_csv("../input/test.csv")["qid"].values
predictions_prob = best_model.predict_proba(test_data)
final_predictions = list(map(lambda x: 1 if x[1]>opt_threshold else 0, predictions_prob))
# Сделаем ответы в правильном виде
answers = pd.DataFrame(np.transpose([test_id, final_predictions]),
                                   columns = ["qid", "prediction"])
answers.head()
answers.to_csv("submission.csv", index=False)
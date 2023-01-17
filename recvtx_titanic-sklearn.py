import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Загрузка данных из таблиц
data_train = pd.read_csv('../input/titanic/train.csv')
data_test = pd.read_csv('../input/titanic/test.csv')
data_all = data_train.append(data_test, sort=False)

# Подготовка данных
# ID пасажиров понадобятся в выходных данных, где им будут сопаставлена предсказанная выживаемость
ids = data_all[data_all.Survived.isna()].PassengerId

# Удалим поля с большим числом пропусков и с малополезными данными
data_all.drop(columns=['PassengerId', 'Ticket', 'Cabin'], inplace=True)

# Много пропусков у признака Age. заменим их на медианные значение
data_all.Age.fillna (data_all.Age.median(), inplace = True )
# Aналогично для Fara
data_all.Fare.fillna(data_all.Fare.median(), inplace = True )
# Пару пропусков Embarked заменим на моду
data_all.Embarked.fillna(data_all.Embarked.mode()[0], inplace = True ) 

# Приведём значения принаков к числовому виду
data_all.Sex = data_all.Sex.astype('category').cat.codes
data_all.Embarked = data_all.Embarked.astype('category').cat.codes

# Обращения вида Mr, Master, Ms, Miss, Mrs и т.п. могут нести дополнительную информацию, выдерним их
def get_title(name):
    Titles = ['Capt', 'Rev.', 'Col.', 'Sir.', 'Mr.', 'Master.', 'Dr.', 'Ms.', 'Mrs.', 'Miss.']
    for i in range(len(Titles)):
        if Titles[i] in name:
            return i
    return len(Titles)

# Добавим обращения в новое поле title, а сторое (Name) удалим
data_all['title'] = data_all['Name'].apply(get_title)
data_all.drop(columns=['Name'], inplace=True)

Xtrain = data_all[~data_all.Survived.isna()].drop(columns=['Survived'], inplace=False).values
Ytrain = data_all[~data_all.Survived.isna()].Survived.values

Xtest = data_all[data_all.Survived.isna()].drop(columns=['Survived'], inplace=False).values
 

# Объединяем данные, это нужно, чтобы нормирование признаков было одинаковым для всей выборки   
XAll = np.concatenate((Xtrain,Xtest))
# Нормируем данные (приводим к единой шкале)
XAll = scale (XAll)           

# Cнова разделим на тестовую и обучающую выборку
i,j = Xtrain.shape
Xtrain = XAll[:i, :]
Xtest = XAll[i:, :]

# Предсказывать будем двумя алгоритмами - Случайный лесной классификатор и Метод K соседей
# Параметры этих моделей заранее подогнаны для лучшего результата с помощьюю функции, описанной выше

# Метод K соседей
kN = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=14, p=1, weights='uniform')

# Случайный лесной классификатор
forest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=0.1,
                       min_weight_fraction_leaf=0.0, n_estimators=26,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

# Подбор параметрив классификатора
# Подбор параметров занимает довольно длительное время, поэтому вызов этих функций закомментирован, 
# а результат их работы приведен ниже в комментариях.
# Наилучшие параметры и стоит подставить в модель, для большей точности.

def find_best_parameters(forest, Xtrain, Ytrain, CV, params):
    cv = GridSearchCV(forest, params, verbose=2, cv=CV, n_jobs=-1, refit=True)
    best_params = cv.fit(Xtrain, Ytrain)
    print(f"Accuracy: {best_params.best_score_:.4f}")
    print("Подобранные параметры")
    print(best_params.best_estimator_)
    
params_RFC = [{"min_samples_leaf": list(range(1, 30)), "min_samples_split":list(np.arange(0.1,1,0.1)), "n_estimators":list(range(1,30))}]
params_kN = [{"n_neighbors": list(range(1, 30)), "leaf_size":list(range(1,60)), "weights":["uniform", "distance"], "algorithm":["auto", "ball_tree", "kd_tree", "brute"], "p":[1,2]}]                                                                                

# Нахождение наилучших параметров для kN
CV_kN = RepeatedStratifiedKFold(n_splits=7, n_repeats = 5)
#find_best_parameters(kN,Xtrain, Ytrain, CV_kN, params_kN)

# Нахождение наилучших параметров для RFC
CV_RFC = RepeatedStratifiedKFold(n_splits=7, n_repeats = 5)
#find_best_parameters(forest,Xtrain, Ytrain, CV_RFC, params_RFC)
                                                                                
# Accuracy: 0.8236
# Подобранные параметры
# KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
#                      metric_params=None, n_jobs=None, n_neighbors=14, p=1,
#                      weights='uniform')

# Accuracy: 0.8296
# Подобранные параметры
# RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=None, max_features='auto',
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=3, min_samples_split=11,
#                        min_weight_fraction_leaf=0.0, n_estimators=8,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)
# Accuracy: 0.8314
# Подобранные параметры
# RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=None, max_features='auto',
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=2, min_samples_split=13,
#                        min_weight_fraction_leaf=0.0, n_estimators=8,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)
# Accuracy: 0.8202
# Подобранные параметры
# RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=None, max_features='auto',
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=0.1,
#                        min_weight_fraction_leaf=0.0, n_estimators=26,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)

# Применение модели
Ytest = np.zeros( shape=(Xtest.shape[0], 2), dtype=int )  # для id и предсказаний
Ytest[:, 0] = ids
Ytest2 = Ytest

# Применение первой модели
clf = forest
clf.fit(Xtrain, Ytrain)

# Применение второй модели
kNPredict = kN
kN.fit(Xtrain, Ytrain)

# Записываем предсказания
for i in range(Xtest.shape[0]):
    Ytest[i,1]=clf.predict([ Xtest[i] ])

for i in range(Xtest.shape[0]):
    Ytest2[i,1]=kNPredict.predict([ Xtest[i] ])
    
# Сохраняем в файлы
np.savetxt('submission_forest.csv', Ytest, fmt='%d', delimiter=',', header="PassengerId,Survived", comments='')
np.savetxt('submission_kN.csv', Ytest2, fmt='%d', delimiter=',', header="PassengerId,Survived", comments='')

# Альтернативное сохранение
#pd.DataFrame(Ytest).to_csv("submission_forest.csv", index=True)
#pd.DataFrame(Ytest2).to_csv("submission_kN.csv", index=True)
print("Done")
# Пример с применением XGBClassifier
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
paramsXGB = {"learning_rate":1e-3, "booster":"gbtree"} #gblinear or dart or gbtree.
clf = xgb.XGBClassifier()
clf.set_params(**paramsXGB)
metLearn= CalibratedClassifierCV(clf, method='isotonic', cv=2)
metLearn.fit(Xtrain, Ytrain)

testPredictions = metLearn.predict(Xtest)

Ytest3 = np.zeros(shape=(Xtest.shape[0], 2), dtype=int )  # для id и предсказаний
Ytest3[:, 0] = ids

for i in range(Xtest.shape[0]):
    Ytest3[i,1]=metLearn.predict([ Xtest[i] ])

# Сохраняем в файл
np.savetxt('submission_XGBoost.csv', Ytest3, fmt='%d', delimiter=',', header="PassengerId,Survived", comments='')
print("Done")
from sklearn.feature_selection import RFECV
clf = forest

print("Используемый классификатор")
print(clf.get_params())

print("\nИспользуемей набор признаков")
print(data_all.columns[1:])

CV = RepeatedStratifiedKFold(n_splits=7, n_repeats = 5)
score = cross_val_score(clf, Xtrain, Ytrain, cv=CV)
print(f"\nAccuracy для этого набора признаков: {score.mean():.3f} ± {score.std():.3f}")
Scores = []
Scores.append([score.mean()])


rfecv = RFECV(estimator=clf, cv = CV, scoring='accuracy', step=1, n_jobs=-1)
rfecv.fit(Xtrain, Ytrain)

print("Accuracy для разных наборов признаков")
print(rfecv.grid_scores_)

print("Полезные и бесполезные признаки")
print(rfecv.get_support())  # какие признаки сохранить?
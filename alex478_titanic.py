# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#from sclearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib
from sklearn.tree import DecisionTreeClassifier
titanic_data = pd.read_csv('/kaggle/input/titanic/train.csv')#, index_col = 'PassengerId')
titanic_data.head(3)
titanic_data.info()
#titanic_data.shape # (891, 12)
titanic_data.isnull().sum() # сколько пропущенных значений
#titanic_data.shape # (891, 12)
#titanic_data.Cabin.value_counts() #  проверка с добавкой новых колонок из Cabin / стало хуже
#print(titanic_data.Cabin.unique())
# подготовка новой колонки Cabin_new
#Cabin = titanic_data.Cabin
#Cabin = np.array(Cabin)
#for i in range(len(Cabin)):
#    if 'C' in str(Cabin[i]):
#        Cabin[i] = 'C'
#    if 'A' in str(Cabin[i]):
#        Cabin[i] = 'A'
#    if 'B' in str(Cabin[i]):
#        Cabin[i] = 'B'
#    if 'G' in str(Cabin[i]):
#        Cabin[i] = 'G'
#    if 'D' in str(Cabin[i]):
#        Cabin[i] = 'D'
#    if 'E' in str(Cabin[i]):
#        Cabin[i] = 'E'
#    if 'F' in str(Cabin[i]):
#        Cabin[i] = 'F'
#    if 'T' in str(Cabin[i]):
#        Cabin[i] = 0
#Cabin
#titanic_data['Cabin_new'] = Cabin
#titanic_data[titanic_data['Cabin_new'].isin(['T'])].median()[3] # 45 столько значений T
#titanic_data[titanic_data['Embarked'].isin(['0'])].median() # Embarked
#titanic_data.head(3)
#print(titanic_data.Cabin.unique())
#titanic_data.loc[titanic_data.Age == 'female'].median()
#titanic_data[titanic_data['Age'].isin(['NaN'])] # все строки где такие значения
#female_median = titanic_data[titanic_data['Sex'].isin(['female'])].median()[3] # 27
#female_median # медианный возраст у женщин
#male_median = titanic_data[titanic_data['Sex'].isin(['male'])].median()[3] # 29
#male_median # медианный возраст у мужчин
#mask = titanic_data.query("Sex == 'male' & Age == 'NaN'")# = 29 # 53 + 124
#titanic_data.loc[mask, 'Age'] = 29
#titanic_data.query("Sex == 'male' & Age == 'NaN'")['Age'] = 29
#titanic_data.query("Sex == 'male' & Age == 'NaN'")
#print(titanic_data.Cabin.unique())
#titanic_data.shape # (891, 13)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import KFold
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
titanic_data.head(3)
X = titanic_data.drop(['Name', 'Cabin', 'Ticket', 'Survived'], axis = 1)
# появилась разница в кол-ве колонок / надо их удалить выше - Cabin_new_T, Embarked_0 , 
y = titanic_data['Survived']
#X.shape # (891, 9)
X = X.fillna(0)
X = pd.get_dummies(X) # уберем категор значения 
X = X.fillna({'Age':X.Age.median()}) # можно разделить на медиан.знач для мужч и женщин
X = X.fillna({'Fare':X.Fare.median()}) 
X = X.drop(['Embarked_0'], axis = 1)
#X.shape # (891, 12)
X.head(3)
import scipy.stats as stats
import matplotlib.pyplot as plt
#age = X.Age                            # узнаем норм.распределение этого признака
#age = age.sort() 
#age.hist()
#hmean_age = np.mean(age)
#hstd_age = np.std(age)
#pdf_age = stats.norm.pdf(age, hmean_age, hstd_age)
#plt.plot(age, pdf_age)

# если нормальное рапределение не нормальное - стоит сделать новый признак x_i = log(x_i + 1)
# надо проверить
# можно признак Name разделить на катег.знач. - мистер мисс миссис и тд
# еще  надо бы проверить у катег признака кол-ва у кажд значения. у признака Cabin например 
fare = X.Fare     # узнаем норм.распределение этого признака
#fare.sort()
hmean_fare = np.mean(fare) # 
hstd_fare = np.std(fare)
pdf_fare = stats.norm.pdf(fare, hmean_fare, hstd_fare)
print(hmean_fare, hstd_fare)#, pdf_fare)
#plt.plot(fare, pdf_fare)
fare_log = np.log(fare) + 1
#plt.plot(fare, fare_log)
#X.Fare.hist()
X['Fare_log'] = np.log(X.Fare)
X.Fare_log.iloc[[179, 263, 271, 277,302, 413, 466, 481, 597, 633, 674, 732, 806, 815, 822]] = 2.6709850297651974# 2.6709850297651974 
X.Fare_log[130:190]
#X.Fare_log[X.Fare_log == '-inf'].count()
#X.Fare_log.unique()
X.Fare_log.hist() # новая колонка 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier(criterion= 'entropy', max_depth= 3) # на тесте - 0.8203389830508474
max_depth_values = range(1, 100)
# поиск лучших параметров
scores_data = pd.DataFrame()
for i in max_depth_values:
    clf = DecisionTreeClassifier(criterion= 'entropy', max_depth= i)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    mean_cross_val_score = cross_val_score(clf, X_train, y_train, cv= 5).mean()
    
    temp_score_data = pd.DataFrame({'max_depth':[i], 
                                    'train_score':[train_score], 
                                    'test_score':[test_score],
                                   'cross_val_score':[mean_cross_val_score]})
    scores_data = scores_data.append(temp_score_data)
scores_data.head(3)
# поиск лучших параметров
scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], 
                           value_vars=['train_score', 'test_score', 'cross_val_score'],
                          var_name= 'set_type', value_name= 'score')
scores_data_long.head(3)
# поиск лучших параметров
scores_data_long.query("set_type == 'cross_val_score'").head(2)
# поиск лучших параметров
sns.lineplot(data = scores_data_long, x = 'max_depth', y = 'score', hue='set_type')
#
clf = DecisionTreeClassifier(criterion= 'entropy', max_depth= 4)
#cross_val_score(clf, X_train, y_train, cv= 5) # применим крос-валидацию
cross_val_score(clf, X_train, y_train, cv= 5).mean() # 0.7886274509803922
best_clf = DecisionTreeClassifier(criterion= 'entropy', max_depth= 10) # 
#cross_val_score(clf, X_test, y_test, cv= 5).mean() # на тестовых данных 

cross_val_score(best_clf, X_test, y_test, cv= 5).mean() # на тестовых данных  0.7796610169491525
best_clf.fit(X_train, y_train)
best_clf.score(X_test, y_test) # 0.8033898305084746
# 
from sklearn.model_selection import GridSearchCV # можно добавить разные метрики
clf = DecisionTreeClassifier()
parameters = {'criterion':['gini','entropy'], 'max_depth':range(1,30)}
grid_Search_CV_clf = GridSearchCV(clf, param_grid= parameters, cv= 5)
grid_Search_CV_clf
grid_Search_CV_clf.fit(X_train, y_train)
grid_Search_CV_clf.best_params_ # показать лучшие параметры
best_clf = grid_Search_CV_clf.best_estimator_
best_clf.score(X_test, y_test)
from sklearn.metrics import precision_score # precision - точность 
from sklearn.metrics import recall_score # recall - полнота
from sklearn.metrics import f1_score

y_pred = best_clf.predict(X_test)
print('precision - точность -',precision_score(y_test, y_pred))
print('recall - полнота - ', recall_score(y_test, y_pred))
print('f1_score - ', f1_score(y_test, y_pred))
y_predicted_prob = best_clf.predict_proba(X_test)
pd.Series(y_predicted_prob[:,1]).hist() # вероятности попадания в тот или другой класс
np.where(y_predicted_prob[:,1] > 0.8, 1 , 0) # перекодируем в класс 1 или 0 в зависимости от вероятности - тут 0,8
y_pred = np.where(y_predicted_prob[:,1] > 0.6, 1 , 0)
print('precision - точность -',precision_score(y_test, y_pred))
print('recall - полнота - ', recall_score(y_test, y_pred))
# нарисуем ROC кривую
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])
roc_auc= auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#best_clf = DecisionTreeClassifier(criterion='entropy', max_depth=8)
clf = DecisionTreeClassifier(criterion='gini', 
                             max_depth=3, 
                             min_samples_split= 10, 
                             min_samples_leaf= 5)
#clf = DecisionTreeClassifier(criterion='entropy', 
#                             max_depth=3, 
#                             min_samples_split= 12, 
#                             min_samples_leaf= 5) # тоже значение
clf.fit(X_train, y_train)
clf.score(X_test, y_test) # 0.823728813559322 / 0.8135593220338984

from sklearn.ensemble import RandomForestClassifier # 0.77990
from sklearn.svm import SVC # 0.73923
from sklearn.ensemble import GradientBoostingClassifier # 0.80143 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
clf_rf =  RandomForestClassifier()
params = {'n_estimators':range(18, 21), 
          'max_depth':range(10, 11)}
grid_Search_CV_clf = GridSearchCV(clf_rf, param_grid= params, cv= 5)
grid_Search_CV_clf.fit(X_train, y_train)
grid_Search_CV_clf.best_params_
best_clf = grid_Search_CV_clf.best_estimator_
best_clf.score(X_test, y_test) # 0.8338983050847457 / 0.8203389830508474
best_clf.feature_importances_ # признаки / какой вклад признаки вносят
feature_importances = best_clf.feature_importances_
feature_importances_df = pd.DataFrame({'feature':X_test.columns, 
                                       'feature_importances':feature_importances})
feature_importances_df
feature_importances_df.sort_values('feature_importances', ascending=False) # отсорируем по значению
#clf_svc = SVC(kernel= 'linear', random_state=241, C= 10000) # 0.73923 стало хуже
#clf_svc.fit(X_train, y_train)
#clf_svc.score(X_test, y_test) # 0.8203389830508474 при С = 10000 / 0.8033898305084746 при C=1
#
clf_gb = GradientBoostingClassifier(n_estimators=320, random_state=42, 
                                    learning_rate=0.1,
                                    max_depth= 3) # 0.8372881355932204
clf_gb.fit(X_train, y_train)
clf_gb.score(X_test, y_test) # 0.8271186440677966
# 0.8169491525423729 с введением нового признака Fare_log - стало хуже
print(clf_gb.score(X_test, y_test))

#for i in range(1, 10):
#    clf_gb = GradientBoostingClassifier(n_estimators=320, random_state=42, learning_rate=0.1, 
#                                        max_depth= i)
#    clf_gb.fit(X_train, y_train)
#    print(i, '-', clf_gb.score(X_test, y_test))
    
# n_estimators=320 / learning_rate= 0.1 / min_samples_split = 2 / min_samples_leaf= 2 / max_depth=8
#kf = KFold(n_splits=5, random_state=1, shuffle=True)
#w = cross_val_score(clf_gb,X_test, y_test, cv= 10) # результат хуже стал
#print(w.mean())
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')#, index_col = 'PassengerId')
#titanic_test.shape # (418, 11)
# 
# подготовка новой колонки Cabin_new / результат ухудшился / убираем
#Cabin = titanic_test.Cabin
#Cabin = np.array(Cabin)
#for i in range(len(Cabin)):
#    if 'C' in str(Cabin[i]):
#        Cabin[i] = 'C'
#    if 'A' in str(Cabin[i]):
#        Cabin[i] = 'A'
#    if 'B' in str(Cabin[i]):
#        Cabin[i] = 'B'
#    if 'G' in str(Cabin[i]):
#        Cabin[i] = 'G'
#    if 'D' in str(Cabin[i]):
#        Cabin[i] = 'D'
#    if 'E' in str(Cabin[i]):
#        Cabin[i] = 'E'
#    if 'F' in str(Cabin[i]):
#        Cabin[i] = 'F'
#Cabin
#titanic_test['Cabin_new'] = Cabin
#titanic_test.shape # (418, 12)
#print(titanic_test.Cabin_new.unique())
#titanic_test.loc[:, titanic_test.isnull().any()].copy() # показать столбец где есть НАН
# чистка test выборки

titanic_test = titanic_test.drop(['Name', 'Cabin', 'Ticket'], axis = 1)
titanic_test = titanic_test.fillna(0)

titanic_test = pd.get_dummies(titanic_test) # уберем категор значения 
titanic_test = titanic_test.fillna({'Age':titanic_test.Age.median()}) # можно разделить на медиан.знач для мужч и женщин
#titanic_test.Fare.isnull().all(0)
#nans = titanic_test.loc[:, titanic_test.isnull().any()].copy() # показать столбец где есть НАН
titanic_test = titanic_test.fillna({'Fare':titanic_test.Fare.median()})
titanic_test['Fare_log'] = np.log(titanic_test.Fare) # добавим новую колонку


titanic_test.head(3)
#titanic_test.shape # (418, 19)
print(X.columns)
print(X.shape)
print(titanic_test.columns)
print(titanic_test.shape)
# появилась разница в кол-ве колонок / надо их удалить выше - Cabin_new_T, Embarked_0 , 

#y_pred_test = best_clf.predict(titanic_test) # 0.7709
#y_pred_test = clf.predict(titanic_test) # 0.77990
#y_pred_test = best_clf.predict(titanic_test) # 0.7709 с кросс валидацией
#y_pred_test = clf.predict(titanic_test) # без колонок Embarked - 77990
# y_pred_test = clf.predict(titanic_test) # с колонками образованными из Cabin  - 0.76794
#y_pred_test = clf_svc.predict(titanic_test) # SVC - 0.73923
y_pred_test = clf_gb.predict(titanic_test) # GradientBoosting - 0.80143
output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': y_pred_test})
output.to_csv('my_submission.csv', index=False)
print("+++")
# 0.80143 / 0.79186 после всяких ....
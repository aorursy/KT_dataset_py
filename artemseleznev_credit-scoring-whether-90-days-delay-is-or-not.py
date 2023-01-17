#общие импорты
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import missingno as msno
import seaborn as sns
import os
import catboost

#теперь сделть
#что будем использовать
#liner regression - плюсы: высокое время обучение,
#logistic regression - плюсы: высокое время обучение,
#decision tree - плюсы:высокая точность,
#random forest - плюсы: высокая точность, 
#SMV - плюсы: хорошо работает с большими наборами данных,
#knn -

#есть модели, которые подходят, но что такое Скоринг
# Скоринг - это классификация (бинарная: 1|0)
#импортируем нужное
from sklearn.linear_model import LogisticRegression  #
from sklearn.linear_model import LinearRegression  #
from sklearn.neighbors import KNeighborsClassifier #
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
#Scoring
#90 days of delay
#загружаем дату
scoring_data = pd.DataFrame(pd.read_csv('../input/credit_scoring_train.csv')).head(7500) #в примере работаю только с 10%, просто долго с 75000
#рассмотрим данные
scoring_data.head()
#информация по колонкам
scoring_data.info()
#посмотрим пустоты
msno.matrix(scoring_data, color = (0.1, 0.5, 0.75), figsize=(10,10));
#мы так же видим много странных значений, возможно это выбросы, давайте выведим графики
#почему я решил это сделать?
#согласитесь, что есть много странных данных в DIR, которые сильно выделяются (нужны ли они нам?)
#решим это
#график покажет, есть ли аномально большие значения или нет
scoring_data.DIR.plot.box();
plt.title('DIR'); #заголовочек
#кажется, как-будто аномальные DIR == INCOME
#проверим по среднему

#DIR = debt to income ratio = (зар.плата) /  сумма всех кредитов, заемов и ипотек в месяц (т.е. обязательных платежей)
#DIR не должна быть больше 1
scoring_data.DIR[scoring_data.DIR > 1].mean()

#DIR = debt to income ratio 
#BalanceToCreditLimit = (сумма всех балансов) / (сумма всех лимитов по картам)
#средняя по ЗП
scoring_data.Income[scoring_data.Income != False].mean()
#узнаем, совпадает ли кол-во пустых ячеек с тем, где аномалия в DIR
print(scoring_data.DIR[scoring_data.DIR > 1].count())
print(scoring_data.Income[scoring_data.Income != False].count())
#можно попробовать заменить Income на DIR, но это может быть не правильным решением
#например, где мы точно уверены, что нет Income и DIR больше минимального значения
scoring_data_replaced = scoring_data

#настоящий DIR
DIR_mean = scoring_data_replaced[scoring_data_replaced.DIR < 1].DIR.mean()

def rep(d, real_dir = DIR_mean):
    if d.isnull()[2] == True:
        #print(d[1])
        if d[1] > 100: #малоли кто и как посчитал %, может быть это все таки %
            scoring_data_replaced.Income[scoring_data_replaced.client_id == d[0]] = scoring_data_replaced.DIR[scoring_data_replaced.client_id == d[0]].values[0]
            scoring_data_replaced.DIR[scoring_data_replaced.client_id == d[0]] = real_dir
            #print(scr.DIR[scr.client_id == d[0]].values[0])
    
scoring_data_replaced.loc[:,('client_id', 'DIR', 'Income')].apply(rep, axis = 1) #рекомендуется использовать loc, без него будет очень долго
#но остальные NA, все равно лучше удалить
scoring_data_replaced.dropna(inplace=True)
scoring_data_replaced.head()
#получиться как-то так
#лучше или хуже, определим позже
#выведем DIR и посмотрим, все ли убралось
scoring_data_replaced.DIR.plot.box();
plt.title(scoring_data_replaced['DIR'].count()); #заголовочек, так как много файлов
#Все равно, DIR выше 0 придеться удалить, так как он являеться не правильным
scoring_data_replaced = scoring_data_replaced[scoring_data_replaced.DIR < 1]
#что сталось
scoring_data_replaced.DIR.plot.box();
plt.title(scoring_data_replaced['DIR'].count()); #заголовочек, так как много файлов
#интересно посмотреть, как изменятеся время задолженности 
plt.figure(figsize=(15,10))
pd.plotting.parallel_coordinates(scoring_data.head(1000),'client_id', ['Num30-59Delinquencies', 'Num60-89Delinquencies', 'Delinquent90']).legend_.remove()
#что это, опять анамалия?
scoring_data['Num30-59Delinquencies'].plot.box();
scoring_data['Num30-59Delinquencies'][scoring_data['Num30-59Delinquencies'] > 20].count()
#не факт, что это выброс
#давайте вычистим данные от пустых, а дальше посмотрим опять на выбросы
scoring_data.dropna(inplace = True)
#посмотрим пустоты
msno.matrix(scoring_data, color = (0.1, 0.5, 0.75), figsize=(10,10));
scoring_data.DIR.plot.box();
#интересно посмотреть, как изменятеся время задолженности 
plt.figure(figsize=(15,10))
pd.plotting.parallel_coordinates(scoring_data.head(1000),'client_id', ['Num30-59Delinquencies', 'Num60-89Delinquencies', 'Delinquent90']).legend_.remove()
#эти значения ещё остались
scoring_data['Num30-59Delinquencies'].plot.box();
scoring_data['Num30-59Delinquencies'][scoring_data['Num30-59Delinquencies'] > 20].count()
#не факт, что это выброс
#сколько таких данных в replaced
scoring_data_replaced['Num30-59Delinquencies'][scoring_data['Num30-59Delinquencies'] > 20].count()
#удалим их из replaced
scoring_data_replaced = scoring_data_replaced[scoring_data['Num30-59Delinquencies'] <= 20]

#У нас две даты
#scoring_data - простой дата сет с данными, просто вычещенным NA
#scoring_data_replaced - дата сет с данным, вычещенный от выбросов, с изменными DIR и INCOM, без задержек более 20 раз
#начнем
#To Do
#1 сделать два отвельных дата сета (с измененными данными, как выше и просто с выбросом NaN и неподходящих данных)
#2 Сдлеать бинарную (перевести в 1 и 0) каждый вариант = будет 4 дата сета (два обычных, два бинарных)
#добавляем ещё бинарную замену для каждого дата сета
#IncomeType - уровень ЗП 0,1,2
#IsFamaly - есть семья или нет 1,0
#IsDelinquencies - есть вообще задолжности 1 - да, 0 - нет
#bad_DIR - заменит заработок - 0 хорошо, 1 плохо
def income_med(count):
    if 8200 > count > 3500:
        return 1
    else:
        return 0
    
def income_good(count):
    if count > 8200:
        return 1
    else:
        return 0
    
def income_bad(count):
    if count < 3500:
        return 1
    else:
        return 0

def famaly(count):
    if count == 0:
        return 0
    else:
        return 1

def deli30(c):
    if c == 0:
        return 0
    else:
        return 1

def deli60(c):
    if c == 0:
        return 0
    else:
        return 1

def bad_DIR(count):
    if (count * 100) < 18:
        return 0
    else:
        return 1

def good_DIR(count):
    if (count * 100) > 36:
        return 0
    else:
        return 1
    
# есть кредиты
# есть ипотеки
# баланс по карте хороший (выше 0.7)
def isLoan(count):
    if count == 0:
        return 0
    else:
        return 1

def isEstim(count):
    if count == 0:
        return 0
    else:
        return 1

def goodBalance(count):
    if count > 0.15:
        return 1
    else:
        return 0
    
def badBalance(count):
    if count < 0.02:
        return 1
    else:
        return 0
    
def ager(count):
    if count >= 55:
        return 1
    else:
        return 0
#замена для scoring_data
scoring_data_bin = pd.DataFrame()

scoring_data_bin["IncomeTypeGood"] = scoring_data["Income"].apply(income_good)
scoring_data_bin["IncomeTypeMed"] = scoring_data["Income"].apply(income_med)
scoring_data_bin["IncomeTypeBad"] = scoring_data["Income"].apply(income_bad)
scoring_data_bin["IsFamaly"] = scoring_data["NumDependents"].apply(famaly)
scoring_data_bin["IsDeli30"] = scoring_data["Num30-59Delinquencies"].apply(deli30)
scoring_data_bin["IsDeli60"] = scoring_data["Num60-89Delinquencies"].apply(deli60)
scoring_data_bin["BadDIR"] = scoring_data["DIR"].apply(bad_DIR)
scoring_data_bin["GoodDIR"] = scoring_data["DIR"].apply(good_DIR)
scoring_data_bin["IsLoan"] = scoring_data["NumLoans"].apply(isLoan)
scoring_data_bin["IsEstim"] = scoring_data["NumRealEstateLoans"].apply(isEstim)
scoring_data_bin["GoodBalance"] = scoring_data["BalanceToCreditLimit"].apply(goodBalance)
scoring_data_bin["BadBalance"] = scoring_data["BalanceToCreditLimit"].apply(badBalance)
scoring_data_bin["AgeType"] = scoring_data["Age"].apply(ager)
#добавим колонку которую будем определяь
scoring_data_bin["Delinquent90"] = scoring_data["Delinquent90"]

scoring_data_bin.head()
#замена для scoring_data
scoring_data_replaced_bin = pd.DataFrame()

scoring_data_replaced_bin["IncomeTypeGood"] = scoring_data_replaced["Income"].apply(income_good)
scoring_data_replaced_bin["IncomeTypeMed"] = scoring_data_replaced["Income"].apply(income_med)
scoring_data_replaced_bin["IncomeTypeBad"] = scoring_data_replaced["Income"].apply(income_bad)
scoring_data_replaced_bin["IsFamaly"] = scoring_data_replaced["NumDependents"].apply(famaly)
scoring_data_replaced_bin["IsDeli30"] = scoring_data_replaced["Num30-59Delinquencies"].apply(deli30)
scoring_data_replaced_bin["IsDeli60"] = scoring_data_replaced["Num60-89Delinquencies"].apply(deli60)
scoring_data_replaced_bin["BadDIR"] = scoring_data_replaced["DIR"].apply(bad_DIR)
scoring_data_replaced_bin["GoodDIR"] = scoring_data_replaced["DIR"].apply(good_DIR)
scoring_data_replaced_bin["IsLoan"] = scoring_data_replaced["NumLoans"].apply(isLoan)
scoring_data_replaced_bin["IsEstim"] = scoring_data_replaced["NumRealEstateLoans"].apply(isEstim)
scoring_data_replaced_bin["GoodBalance"] = scoring_data_replaced["BalanceToCreditLimit"].apply(goodBalance)
scoring_data_replaced_bin["BadBalance"] = scoring_data_replaced["BalanceToCreditLimit"].apply(badBalance)
scoring_data_replaced_bin["AgeType"] = scoring_data_replaced["Age"].apply(ager)
#добавим колонку которую будем определяь
scoring_data_replaced_bin["Delinquent90"] = scoring_data_replaced["Delinquent90"]

scoring_data_replaced_bin.head()
#все наборы
sets = [scoring_data, #обычный
       scoring_data_bin, #обычный, бинарный
       scoring_data_replaced, #с хорошей датой
       scoring_data_replaced_bin] #с хорошей датой, бинарный
set_name = ['Обычный набор',
            'Обычный набор с бинарными параметрами',
            'Набор с инсправленной датой',
            'Набор с инсправленной датой с бинарными параметрами',] #я не нашел, как получить имена переменных интересным и простым способо, лучше я назову списки сам
#X, y - наборы
Xs = [x.drop('Delinquent90', axis=1) for x in sets]
Ys = [y['Delinquent90'] for y in sets]
#списки масштабируемых Хs
X_train_scaled_set = list()
X_test_scaled_set = list()
#списки Ys
Y_train_set = list()
Y_test_set = list()



for num in range(len(Xs)):
    print(num+1, '-', set_name[num])
    #получем Трайн и Тест
    X_train, X_test, y_train, y_test = train_test_split(Xs[num], Ys[num], test_size=0.3, random_state = 28)
    Y_train_set.append(y_train)
    Y_test_set.append(y_test)
    #отмасштабируем выборку
    scaler = StandardScaler()
    X_train_scaled_set.append(scaler.fit_transform(X_train))
    X_test_scaled_set.append(scaler.transform(X_test))
#Фитим модельки
#Логистическая регрессия
params1 = {
    'C': [1e-5, 1e-2, 1]
}


model1 = LogisticRegression()
gs1 = GridSearchCV(model1, params1, scoring='roc_auc', cv=5)
gs1_fits = []

for step in range(len(X_train_scaled_set)):
    gs1_fits.append(gs1.fit(X_train_scaled_set[step], Y_train_set[step]))
    print(set_name[step], ' - ', gs1.best_score_)
#LinearRegression
params1_1 = {
    'n_jobs': [-1]}

model1_1 = LinearRegression()
gs1_1 = GridSearchCV(model1_1, params1_1,scoring='roc_auc', cv=5)
gs1_1_fits = []

for step in range(len(X_train_scaled_set)):
    gs1_1_fits.append(gs1_1.fit(X_train_scaled_set[step], Y_train_set[step]))
    print(set_name[step], ' - ', gs1_1.best_score_)
#метод соседей
params2 = {
    'n_neighbors': [1, 5, 8]
}

model2 = KNeighborsClassifier()
gs2 = GridSearchCV(model2, params2, scoring='roc_auc', cv=5)
gs2_fits = []

for step in range(len(X_train_scaled_set)):
    gs2_fits.append(gs2.fit(X_train_scaled_set[step], Y_train_set[step]))
    print(set_name[step], ' - ', gs2.best_score_)

#RandomForestClassifier

params3 = {
    'n_estimators': [10, 50]
}

model3 = RandomForestClassifier()
gs3 = GridSearchCV(model3, params3, scoring='roc_auc', cv=5)
gs3_fits = []

for step in range(len(X_train_scaled_set)):
    gs3_fits.append(gs3.fit(X_train_scaled_set[step], Y_train_set[step]))
    print(set_name[step], ' - ', gs3.best_score_)
#по рекомендации XGB
class Stacking_def:
    def __init__(self, models, final_model, n_folds):
        self.n_folds = n_folds
        self.models = models
        self.final_model = final_model
    
    def fit_predict(self, X_train, y_train, X_test, params):
        # Переводим в numpy-массивы, чтобы удобно было 
        # обращаться к строкам
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        
        # Инициализруем X_train_2 и X_test_2 -- матрицы признаков
        # для финальной модели
        n_examples = len(X_train)
        n_models = len(self.models)
        X_train_2 = np.zeros((n_examples, n_models))
        X_test_2 = np.zeros((len(X_test), n_models))
        
        # Строим разбиение X_train на фолды
        kfold = KFold(n_splits=self.n_folds, shuffle=True, \
                      random_state=28)
        folds = list(kfold.split(range(len(X_train))))
        
        for model_id, model in enumerate(self.models):
            #print('=' * 30)
            #print('Model_id: ', model_id)
            for k, (train_ind, test_ind) in enumerate(folds):
#                print('-' * 30)                
#                print('Fold number: ', k)
#                print('Train indexes: ', train_ind)
#                print('Test indexes: ', test_ind)
                
                # Обучаем модель model на X_train[train_ind]
                model.best_estimator_.fit(X_train[train_ind], y_train[train_ind])
                
                # Строим прогноз по тестовому фолду
                y_tmp = model.best_estimator_.predict_proba(X_train[test_ind])[:, 1]
                X_train_2[test_ind, model_id] = y_tmp
                
                # Строим прогноз по X_test
                y_tmp = model.predict_proba(X_test)[:, 1]
                X_test_2[:, model_id] += y_tmp
                # То же самое, что и
                #X_test_2[:, model_id] = X_test_2[:, model_id] + y_tmp
        
        X_test_2 = X_test_2 / self.n_folds
        
        print('Feature matrices calculation finished.')
        gs = GridSearchCV(self.final_model, params, scoring='roc_auc', cv = 3)
        gs.fit(X_train_2, y_train)
        print('Final model score on train: ', gs.best_score_)
        y_test = gs.best_estimator_.predict_proba(X_test_2)[:, 1]
        
        return y_test
class Stacking_cat:
    def __init__(self, models, final_model):
        self.models = models
        self.final_model = final_model
    
    def fit_predict(self, X_train, y_train, X_test, y):        
        print('Feature matrices calculation finished.')

        self.final_model.fit(X_train, y_train, eval_set=(X_test, y), 
                             logging_level='Silent', 
                             use_best_model=True)
        
        #nums = self.final_model.get_feature_importance(X_train, y_train)
        #print(nums)
        
        y_test = self.final_model.predict_proba(X_test)[:, 1]
        print('Final model score on train: ', roc_auc_score(y, y_test))
        
        return self.final_model
        
#Делаем XGB
from xgboost import XGBClassifier

#gs1.best_estimator_
models_set = [
    gs1_fits,
    #gs1_1_fits, исключаем этот метод, так как он не обладает predict_proba
    gs2_fits,
    gs3_fits,
]

final_model = XGBClassifier()
params = {
    'n_estimators': [10, 20, 30]
}

for num_model in range(3):
    s = Stacking_def(models_set[num_model], final_model, n_folds=5)
    for step in range(4):
        y_test = s.fit_predict(X_train_scaled_set[step], Y_train_set[step], X_test_scaled_set[step], params)
#Делаем Cat
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn.metrics import roc_auc_score

final_model = CatBoostClassifier()

#в catboost нет best_estimator (или я не нашел)
for num_model in range(3):
    s = Stacking_cat(models_set[num_model], final_model)
    for step in range(4):
        _model = s.fit_predict(X_train_scaled_set[step], Y_train_set[step], X_test_scaled_set[step], Y_test_set[step])
        print(_model.get_feature_importance(Pool(X_train_scaled_set[step])))

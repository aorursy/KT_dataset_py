import numpy as np
import pandas as pd

import joblib
import time
import warnings

from plotly.subplots import make_subplots

from sklearn.metrics  import f1_score
from sklearn.metrics  import precision_score
from sklearn.metrics  import recall_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
from termcolor import colored
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns

from scipy import stats as st

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import KNNImputer

from sklearn.metrics import classification_report
#Определяем болд
def bold(): 
    return "\033[1m"

def bold_end(): 
    return "\033[0m"

#Ставим формат для нумериков
pd.options.display.float_format = '{: >10.2f}'.format

#Убираем ворнинги
warnings.simplefilter(action='ignore', category=FutureWarning)
#**Функция print_basic_info, для вывода информации о массиве, и его переменных.**

#* base - название базы данных
#* info - 1: вывод информации о массиве, другое: не вывод
#* describe - 1: вывод описания переменных массива, другое: не вывод        
#* duplicat - 1: вывод количества полных дублей
#* head - n: вывод примера базы (вывод n - строк), n < 1: не вывод

def print_basic_info(base, info, describe, duplicat, head):
    if info == 1:
        print("\n", bold(), colored('info','green'), bold_end(), "\n")
        print( base.info())  
    if head >= 1:
        print("\n", bold(),colored('head','green'),bold_end())
        display(base.head(head))
    if describe == 1:
        print("\n", bold(),colored('describe','green'),bold_end(),"\n")
        for i in base.columns:
            print("\n", bold(), colored(i,'blue'),bold_end(),"\n", base[i].describe())
    if duplicat == 1:
        print("\n", bold(),colored('duplicated','green'),bold_end(),"\n")
        print(base[base.duplicated() == True][base.columns[0]].count())
#__Функция ft_namecount__, для вывода названия переменной, частотной нормированной таблицы и описания переменной.

#5 входных параметров:

#* *base* - название базы данных
#* *index* - название переменной в базе
#* *table* - 1: вывод частотной нормированной таблицы, 0: не вывод
#* *sort* - 1: сортировка таблицы по лейблам переменной, 0: не сортировка
#* *describe* - 1: вывод описания переменной, 0: не вывод

def ft_name_count (base, name , table, sort, describe):
    print(bold(), colored(name,'blue') , bold_end(), "\n")
    if table != 0:
        s = (base[name].value_counts(normalize=True))
        if sort != 0:
            s.sort_index(inplace=True)
        print(s)
    if describe != 0:
        print(base[name].describe())
#Работаю локально, онлайн путь другой

contest_train = pd.read_csv('../input/mf-accelerator/contest_train.csv', sep=',',decimal='.' , index_col= 'ID')
contest_test = pd.read_csv('../input/mf-accelerator/contest_test.csv', sep=',',decimal='.', index_col = 'ID')
print_basic_info(contest_train,1,0,1,3)
print_basic_info(contest_test,1,0,1,3)
ft_name_count(contest_train, 'TARGET' , 1, 1, 0)
# Заполним пропуски на основе 4 ближайших сеседей.
imputer = KNNImputer(n_neighbors=4)

contest_train_after = imputer.fit_transform(contest_train.drop(['TARGET'], axis=1))

contest_test_aftet = imputer.transform(contest_test)
print(contest_train_after.shape, contest_train.shape, contest_test_aftet.shape)
pd_contest_train_after = pd.DataFrame(contest_train_after, index = contest_train.index,
                                      columns = contest_train.drop(['TARGET'], axis=1).columns.values)

pd_contest_train_after['TARGET'] = contest_train['TARGET']

pd_contest_test_after = pd.DataFrame(contest_test_aftet, index = contest_test.index ,
                                      columns = contest_test.columns.values)
for i in pd_contest_train_after.columns:
    if pd_contest_train_after[pd_contest_train_after[i].isnull() == True]['TARGET'].sum() > 0:
        print(i, pd_contest_train_after[pd_contest_train_after[i].isnull() == True]['TARGET'].sum())
        
for i in pd_contest_test_after.columns:
    if pd_contest_test_after[pd_contest_test_after[i].isnull() == True]['FEATURE_0'].sum() > 0:
        print(i, pd_contest_test_after[pd_contest_test_after[i].isnull() == True]['FEATURE_0'].sum())
        
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

cl_contest_train = clean_dataset(pd_contest_train_after)
cl_contest_test = clean_dataset(pd_contest_test_after)
features = cl_contest_train.drop(['TARGET'], axis=1)

target = contest_train['TARGET']

features_test = cl_contest_test
#разбиваем, стратифицируем
features_train_big, features_valid, target_train_big, target_valid = train_test_split(features, target , test_size=0.20, 
                                                                              random_state=515093, stratify = target)

features_train, features_valid = np.array(features_train_big), np.array(features_valid)
def resample(features, target, repeat_down, repeat_up):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    features_two = features[target == 2]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    target_two = target[target == 2]

    features_resampled = pd.concat([features_zeros.sample(n =(int(len(features_zeros)*repeat_down)), 
                                                            replace=False,random_state=2)] + [features_ones] 
                                   + [features_two.sample(n =(int(len(features_two)*repeat_up)), 
                                                            replace=True,random_state=2)])
    
    target_resampled = pd.concat([target_zeros.sample(n =(int(len(target_zeros)*repeat_down)),
                                                        replace=False)] + [target_ones] 
                                 + [target_two.sample(n =(int(len(target_two)*repeat_up)), 
                                                            replace=True)])
    features_resampled, target_resampled = shuffle(features_resampled, target_resampled, random_state=12345)
    features_resampled.reset_index(drop = True, inplace = True)
    target_resampled.reset_index(drop = True, inplace = True)
    return features_resampled, target_resampled

base_up = pd.DataFrame()

model_RFC = RandomForestClassifier(n_estimators = 50, random_state = 2)

for i in np.arange(0.37, 0.46, 0.02):
    for j in np.arange(2.2, 2.5, 0.05):
        features_resampled, target_resampled = resample(features_train_big, target_train_big, i , j)
        model_RFC.fit(features_resampled, target_resampled)
        base_up.loc[str(round(i,2)) + " " +str(round(j,2)) ,'Precision'] = precision_score(target_valid, model_RFC.predict(features_valid), average='macro')
        base_up.loc[str(round(i,2)) + " " +str(round(j,2)) ,'Recall'] = recall_score(target_valid, model_RFC.predict(features_valid), average='macro')
        base_up.loc[str(round(i,2)) + " " +str(round(j,2)),'F1'] = f1_score(target_valid, model_RFC.predict(features_valid), average='macro')
#строим график
sns.set(style="whitegrid")
plt.figure(figsize = (20,5)) 
sns.lineplot(data=base_up, palette="tab20", linewidth=2.5)
plt.title("Точность и Полнота модели с различной выборкой положительного типа \n", fontsize=15)
plt.ylabel("accuracy_score %%")
plt.xlabel("sample size")
#plt.ylim((0.1, 0.6)) 
plt.show()
features_resampled, target_resampled = resample(features_train_big, target_train_big, 0.39, 2.2)

features_resampled, target_resampled = np.array(features_resampled), target_resampled
target_resampled.value_counts(normalize=True)
cv = StratifiedKFold(n_splits=2, random_state=1234, shuffle=True)
#сделаем фунукцию, котрая будет записывать время обучения, скорость предсказания, и качество предсказания(RMSE)
def put_in_base_test(base_res, model, features_train, target_train, features_valid, target_valid):
    model.fit(features_train, target_train)
    target = target_valid
    features = features_valid 
    
    proba = pd.DataFrame(model.predict_proba(features_valid), columns = ['0','1','2'])

    proba['predict'] = pd.Series(model.predict(features_valid))    
    #proba['predict'] = proba['predict'].where(proba['0'] > 0.125, 0)
    
    prediction = proba['predict']
    accuracy, precision, recall, f1 = [], [], [], []
    accuracy.append(accuracy_score(target, prediction))
    precision.append(precision_score(target, prediction, average='macro'))
    recall.append(recall_score(target, prediction, average='macro'))
    f1.append(f1_score(target, prediction, average='macro'))
    target_names = ['Seg - 0', 'Seg - 1', 'Seg - 2']
    print(classification_report(target, prediction, target_names=target_names))
    base_res.loc[str(model).split('(')[0],'accuracy'] = np.mean(accuracy)
    base_res.loc[str(model).split('(')[0],'precision'] = np.mean(precision)
    base_res.loc[str(model).split('(')[0],'recall'] = np.mean(recall)
    base_res.loc[str(model).split('(')[0],'f1'] = np.mean(f1)
    
    return base_res, prediction, model

# Выведем графиики для наглядности
def param_bars(base_name, name):
    sns.set(style="whitegrid")
    plt.figure(figsize = (10,3)) 
    df = base_name
    sns.barplot(data=df, palette="tab20", linewidth=2.5)
    plt.title("Показатель эффективности модели - " + str(name), fontsize=15)
    plt.ylabel("%%")
    plt.xlabel("Параметры")
    plt.ylim((0, 1.2)) 
    c = 0
    for i in df.columns:
        plt.text( c - 0.1 , df[i].mean() + 0.1, "{0:.0%}".format(df[i].mean()))
        c = c + 1
    plt.show()
features_train, target_train = features_resampled, target_resampled
#Будем подбирать параметры гридсерчем
parameters = {'max_depth':[i for i in range(15,26,1)] , 'n_estimators':[i for i in range(50,500,50)], 
              'max_features':[80,90,95,100], 'random_state':[1234]}

clf = GridSearchCV(RandomForestClassifier(), cv = cv, param_grid = parameters, scoring = 'f1_macro')
#clf.fit(features_train, target_train)
#print(clf.best_params_)
grid = {'n_estimators' : [i for i in range(5,25,5)], 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],
       'algorithm' : ['SAMME', 'SAMME.R']}

clf = AdaBoostClassifier(RandomForestClassifier())
rs = GridSearchCV(clf, grid, cv=cv, scoring = 'f1_macro')

#rs.fit(features_train, target_train)
#AdaBoostRegressor_best_params1 = rs.best_params_
info_test = pd.DataFrame()

model = AdaBoostClassifier(RandomForestClassifier(max_depth = 19, n_estimators = 500, max_features = 90, random_state = 2),
    n_estimators=200, learning_rate=0.2, algorithm = 'SAMME.R') 

info_test, predictions, final_model = put_in_base_test(info_test, model, features_resampled, target_resampled, features_valid, target_valid)
model = final_model
param_bars(info_test,'Final Test')
output = pd.DataFrame({'ID': contest_test.index, 
                       'Predicted': pd.Series(final_model.predict(features_test), dtype='int32')})

output.to_csv('sub_kiseleva.csv', index=False)
print("Your submission was successfully saved!")

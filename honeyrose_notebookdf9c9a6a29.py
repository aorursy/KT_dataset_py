import warnings

warnings.filterwarnings('ignore')

# ---



%matplotlib inline

import pandas as pd

pd.options.display.max_columns = 100

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import numpy as np

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV



#соединим обучающую и тестовую выборки чтобы результат ьыл точнее

def get_combined_data():

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    #создаём датасет на основе целевого вектора и удаляем его из исходной таблицы

    targets = train.Survived

    train.drop('Survived',1,inplace=True)

    # добавляем к обучающий выборки тестовую

    data = train.append(test)

    # пронумеруем заново индексы

    data.reset_index(inplace=True)

    #удалим столбец index

    data.drop('index',inplace=True,axis=1)

    return data

combined = get_combined_data()

combined.head()



def get_titles():



    global combined

    

    # формируем столбец title из столбца имён (сначала разделяем по запятой ,затем по точке и копируем строку без имени)

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    # словарь титулов

    Title_Dictionary = {

                        "Capt":       "Officer",

                        "Col":        "Officer",

                        "Major":      "Officer",

                        "Jonkheer":   "Royalty",

                        "Don":        "Royalty",

                        "Sir" :       "Royalty",

                        "Dr":         "Officer",

                        "Rev":        "Officer",

                        "the Countess":"Royalty",

                        "Dona":       "Royalty",

                        "Mme":        "Mrs",

                        "Mlle":       "Miss",

                        "Ms":         "Mrs",

                        "Mr" :        "Mr",

                        "Mrs" :       "Mrs",

                        "Miss" :      "Miss",

                        "Master" :    "Master",

                        "Lady" :      "Royalty"



                        }

    #поставим соответствие каждому титулу из словаря

    combined['Title'] = combined.Title.map(Title_Dictionary)

get_titles()



#группируем по полу,титулу и классу чтобы посмотреть взаимосвязь признаков

grouped = combined.groupby(['Sex','Pclass','Title'])

grouped.median()

#получаем среднее значение по столбцам





#пропуски в столбце age поэтому заполним  на основе данных из предыдущей таблицы

def process_age():

    global combined

    def fillAges(row):

        if row['Sex']=='female' and row['Pclass'] == 1:

            if row['Title'] == 'Miss':

                return 30

            elif row['Title'] == 'Mrs':

                return 45

            elif row['Title'] == 'Officer':

                return 49

            elif row['Title'] == 'Royalty':

                return 39



        elif row['Sex']=='female' and row['Pclass'] == 2:

            if row['Title'] == 'Miss':

                return 20

            elif row['Title'] == 'Mrs':

                return 30



        elif row['Sex']=='female' and row['Pclass'] == 3:

            if row['Title'] == 'Miss':

                return 18

            elif row['Title'] == 'Mrs':

                return 31



        elif row['Sex']=='male' and row['Pclass'] == 1:

            if row['Title'] == 'Master':

                return 6

            elif row['Title'] == 'Mr':

                return 41.5

            elif row['Title'] == 'Officer':

                return 52

            elif row['Title'] == 'Royalty':

                return 40



        elif row['Sex']=='male' and row['Pclass'] == 2:

            if row['Title'] == 'Master':

                return 2

            elif row['Title'] == 'Mr':

                return 30

            elif row['Title'] == 'Officer':

                return 41.5



        elif row['Sex']=='male' and row['Pclass'] == 3:

            if row['Title'] == 'Master':

                return 6

            elif row['Title'] == 'Mr':

                return 26

    

    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)

process_age()



def process_names():

    global combined

    # убираем столбец Name

    combined.drop('Name',axis=1,inplace=True)

    # осущствим векторризацию признаков

    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')

    combined = pd.concat([combined,titles_dummies],axis=1)

    # удаляем столбец title

    combined.drop('Title',axis=1,inplace=True)

process_names()

combined.head()



def process_fares():

    global combined

    # заполним один пропуск в таблице средним значением по столбцу

    combined.Fare.fillna(combined.Fare.mean(),inplace=True)

process_fares()



combined.info()

#видим,что осталиь пропуски в столбце cabin and embarked



def process_embarked():

    global combined

    #так как в столбце embarked всего 2 пропуска ,заполним их наиболее часто встречающимся значением S

    combined.Embarked.fillna('S',inplace=True)

    # векторизируем признак

    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')

    combined = pd.concat([combined,embarked_dummies],axis=1)

    combined.drop('Embarked',axis=1,inplace=True)

process_embarked()



def process_cabin():

    global combined

    # заполняем пропуски значением U

    combined.Cabin.fillna('U',inplace=True)

    #в каждом значении столбца cabin оставим только букву кабины

    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    #векторизируем признак

    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')

    combined = pd.concat([combined,cabin_dummies],axis=1)

    combined.drop('Cabin',axis=1,inplace=True)

process_cabin()

combined.head()



def process_sex():

    global combined

    #сделаем значения столбца пол либо 0 или 1 

    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})

process_sex()



def process_pclass():

    global combined

    #векторизация признаков

    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")

    combined = pd.concat([combined,pclass_dummies],axis=1)

    combined.drop('Pclass',axis=1,inplace=True)

process_pclass()



def process_ticket():

    global combined

    #отделяем префикс  билета

    def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        if (not ticket[0].isdigit()):

            return ticket[0]

        else: 

            return 'XXX'

    #векторизация признака

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')

    combined = pd.concat([combined, tickets_dummies],axis=1)

    combined.drop('Ticket',inplace=True,axis=1)

process_ticket()



def process_family():

    global combined

    #Добавляем столбец - число членов семьи,вместе с пассажиром ,для этого нужно сложить значение в столбце sibSp(число родных) и parch(число родителей и друзей)

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    #создадим вектор-столбцы основываясь на количестве человек в семье

    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)

process_family()



def scale_all_features():

    global combined

    # создаём список

    features = list(combined.columns)

    #удаляем элемент

    features.remove('PassengerId')

    #нормализация признаков,чтобы каждый признак вносил одинаковый вклад в оценку

    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)

    print( 'Features scaled successfully !')

scale_all_features()

combined.head()







def fun():

    global combined

    train0 = pd.read_csv('../input/train.csv')

    y_train = train0.Survived

    X_train = combined.ix[0:890]

    X_test = combined.ix[891:]

    return X_train,X_test,y_train



train,test,y = fun()

#график ,показывающий значимость признаков и их корреляци

import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif

predictors=combined.columns

selector = SelectKBest(f_classif, k=5)

selector.fit(train,y)

scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()



#оставим только те признаки,от которых в наибольшей степени зависит результат

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(train, y)

features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort(['importance'],ascending=False)



#обновляем обучающую и тестовую выборки

model = SelectFromModel(clf, prefit=True)

train_new = model.transform(train)

test_new = model.transform(test)
cv = StratifiedKFold(y, n_folds=7, shuffle=True, random_state=1)

alg_frst_model = RandomForestClassifier(random_state=1)

alg_frst_params = [{

    "n_estimators": [350, 400, 450],

    "min_samples_split": [6, 8, 10],

    "min_samples_leaf": [1, 2, 4]

}]

alg_frst_grid = GridSearchCV(alg_frst_model, alg_frst_params, cv=cv, refit=True, verbose=1, n_jobs=-1)

alg_frst_grid.fit(train_new, y)

alg_frst_best = alg_frst_grid.best_estimator_

print("Accuracy (random forest auto): {} with params {}"

      .format(alg_frst_grid.best_score_, alg_frst_grid.best_params_))

alg_test = alg_frst_best



alg_test.fit(train_new,y)





output = alg_frst_best.predict(test_new)

df_output = pd.DataFrame()

df_output['PassengerId'] = test['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('output2.csv',index=False)
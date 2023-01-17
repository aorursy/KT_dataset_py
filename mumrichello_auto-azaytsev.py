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
url_models_list=['https://auto.ru/cars/bmw/02/used/',
'https://auto.ru/cars/bmw/1er/used/',
'https://auto.ru/cars/bmw/2er/used/',       
'https://auto.ru/cars/bmw/2activetourer/used/',
'https://auto.ru/cars/bmw/2grandtourer/used/',
'https://auto.ru/cars/bmw/2000_c_cs/used/',
'https://auto.ru/cars/bmw/3er/used/',
'https://auto.ru/cars/bmw/3_15/used/',
'https://auto.ru/cars/bmw/321/used/',
'https://auto.ru/cars/bmw/326/used/',
'https://auto.ru/cars/bmw/340/used/',
'https://auto.ru/cars/bmw/4/used/',
'https://auto.ru/cars/bmw/5er/used/',
'https://auto.ru/cars/bmw/6er/used/',
'https://auto.ru/cars/bmw/7er/used/',
'https://auto.ru/cars/bmw/8er/used/',
'https://auto.ru/cars/bmw/e3/used/',
'https://auto.ru/cars/bmw/i3/used/',
'https://auto.ru/cars/bmw/i8/used/',
'https://auto.ru/cars/bmw/m2/used/',
'https://auto.ru/cars/bmw/m3/used/',
'https://auto.ru/cars/bmw/m4/used/',
'https://auto.ru/cars/bmw/m5/used/',
'https://auto.ru/cars/bmw/m6/used/',
'https://auto.ru/cars/bmw/x1/used/',
'https://auto.ru/cars/bmw/x2/used/',
'https://auto.ru/cars/bmw/x3/used/',
'https://auto.ru/cars/bmw/x3_m/used/',
'https://auto.ru/cars/bmw/x4/used/',
'https://auto.ru/cars/bmw/x4_m/used/',
'https://auto.ru/cars/bmw/x5/used/',
'https://auto.ru/cars/bmw/x5_m/used/',
'https://auto.ru/cars/bmw/x6/used/',
'https://auto.ru/cars/bmw/x6_m/used/',
'https://auto.ru/cars/bmw/x7/used/',
'https://auto.ru/cars/bmw/z1/used/',
'https://auto.ru/cars/bmw/z3/used/',
'https://auto.ru/cars/bmw/z3m/used/',
'https://auto.ru/cars/bmw/z4/used/',
'https://auto.ru/cars/bmw/z8/used/']
# Функция для нахождения кол-ва страниц по модели.

from bs4 import BeautifulSoup    
import requests

def numPage(adress):
    response=requests.get(adress)
    page=BeautifulSoup(response.text, 'html.parser')
    total =  page.find_all(class_="Button Button_color_whiteHoverBlue Button_size_s Button_type_link Button_width_default ListingPagination-module__page")
    if total == []:
        return 1
    else:
        ccc = str(total[-1])
        ppp = ccc.find('<span class="Button__text">')
        uuu = ccc.find('</span>')
        return int(ccc[ppp+27:uuu])
# Функция для нахождения всех url текухищ машин BMW находящихся в продаже.

url_list=[]

def allCarsUrls(adress):
    print(adress)
    num = numPage(adress)
    for i in range(1,num):
        if i% 5 == 0: print(i)
        adress2 = adress+'?page=' +str(i)
        response=requests.get(adress2)
        page=BeautifulSoup(response.text, 'html.parser')
        total =  page.find_all(class_="Link ListingItemTitle-module__link")
        for i in total:
            car=str(i)
            b = (car.find('href="'))
            c = (car.find('target="'))
            url_list.append(car[b+6:c-2])

for j in url_models_list:
    allCarsUrls(j)
url_list_df = pd.DataFrame(url_list)
url_list_df.to_csv('all_urls.csv',encoding='utf-8-sig', index=False)
# url_list = (pd.read_csv('all_urls.csv'))
# url_list = url_list.values.tolist()

# for i in range(len(url_list)):
#     pppp =  str(url_list[i]).replace("['",'')
#     pppp = pppp.replace("']",'')
#     url_list[i] = pppp
from bs4 import BeautifulSoup    
import requests

dfForTraining = pd.DataFrame(columns=['bodyType','color','fuelType','model','name','productionDate','vehicleTransmission',
                                                     'engineDisplacement','enginePower','mileage','Привод','Руль','Состояние',
                                                     'Владельцы','ПТС','Таможня','Владение','Price'])

# dfForTraining = pd.read_csv('training_new18000.csv')

def carsData(adress):
    dfForFill = pd.DataFrame(index=range(1),columns=['bodyType','color','fuelType','model','name','productionDate','vehicleTransmission',
                                                     'engineDisplacement','enginePower','mileage','Привод','Руль','Состояние',
                                                     'Владельцы','ПТС','Таможня','Владение','Price'])
    response=requests.get(adress)
    response.encoding = 'utf-8'
    page=BeautifulSoup(response.text, 'html.parser')
    
    if (page.find_all(class_="CardInfo__row CardInfo__row_bodytype"))!=-1:
    
        # Достаем тип кузова
        bodyType = page.find_all(class_="CardInfo__row CardInfo__row_bodytype")
        bodyType = str(bodyType)
        b1 = bodyType.find('/">')
        if b1!=-1:
            b2 = bodyType.find("</a>")
            dfForFill.bodyType = str.lower(bodyType[b1+3:b2])


            # Достаем цвет авто
            color = page.find_all(class_="CardInfo__row CardInfo__row_color")
            color = str(color)
            b1 = color.find('/">')
            b2 = color.find("</a>")
            dfForFill.color = str.lower(color[b1+3:b2])

            # Достаем топливо и л.с.
            fuelType = page.find_all(class_="CardInfo__row CardInfo__row_engine")
            fuelType = str(fuelType)
            if (fuelType.find('Электро')==-1 and fuelType.find('электро')==-1 ):
                b1 = fuelType.find('/">')
                b2 = fuelType.find("</a>")
                dfForFill.fuelType = str.lower(fuelType[b1+3:b2])
                c1 = fuelType.find('<div>')
                pp = fuelType[c1:]
                c2 = pp.find('/')
                dfForFill.engineDisplacement = pp[5:c2-3]
                ppp = pp[c2+2:]
                c3 = ppp.find('/')
                if c3!=-1:
                    dfForFill.enginePower = int(ppp[:c3-5])
                else: dfForFill.enginePower = int('0')

            # Достаем модель и название
            all_items = page.find_all(class_='Link Link_color_gray CardBreadcrumbs__itemText')
            model = str(all_items[1])
            pp1 = model.find('">')
            pp2 = model.find('<!--')
            dfForFill.model = model[pp1+2:pp2]
            name = str(all_items[4])
            pp1 = name.find('">')
            pp2 = name.find('<!--')
            dfForFill.name = name[pp1+2:pp2]

            # Достаем год выпуска
            productionDate = page.find_all(class_="CardInfo__row CardInfo__row_year")
            productionDate = str(productionDate)
            b1 = productionDate.find('/">')
            b2 = productionDate.find("</a>")
            dfForFill.productionDate = int(productionDate[b1+3:b2])

            # Достаем тип трансмиссии
            vehicleTransmission = page.find_all(class_="CardInfo__row CardInfo__row_transmission")
            vehicleTransmission = str(vehicleTransmission)
            b1 = vehicleTransmission.rfind('CardInfo__cell')
            b2 = vehicleTransmission.rfind('</span>')
            dfForFill.vehicleTransmission = str.lower(vehicleTransmission[b1+16:b2])

            # Достаем пробег
            mileage = page.find_all(class_="CardInfo__row CardInfo__row_kmAge")
            mileage = str(mileage)
            b1 = mileage.rfind('CardInfo__cell')
            b2 = mileage.rfind('</span>')
            mileage2 = mileage[b1+16:b2-3]
            pp = mileage2.split()
            if len(pp)>1:
                cc=mileage2[len(pp[0])]
                dfForFill.mileage = int(mileage2.replace(cc,''))
            else: 
                dfForFill.mileage = int(mileage2)

            # Достаем Привод
            Привод = page.find_all(class_="CardInfo__row CardInfo__row_drive")
            Привод = str(Привод)
            b1 = Привод.rfind('CardInfo__cell')
            b2 = Привод.rfind('</span>')
            dfForFill.Привод = str.lower((Привод[b1+16:b2]))

            # Достаем Руль
            Руль = page.find_all(class_="CardInfo__row CardInfo__row_wheel")
            Руль = str(Руль)
            b1 = Руль.rfind('CardInfo__cell')
            b2 = Руль.rfind('</span>')
            dfForFill.Руль = str.lower((Руль[b1+16:b2]))

            # Достаем Состояние
            Состояние = page.find_all(class_="CardInfo__row CardInfo__row_state")
            Состояние = str(Состояние)
            b1 = Состояние.rfind('CardInfo__cell')
            b2 = Состояние.rfind('</span>')
            dfForFill.Состояние = str.lower((Состояние[b1+16:b2]))

            # Достаем Владельцов
            Владельцы = page.find_all(class_="CardInfo__row CardInfo__row_ownersCount")
            if Владельцы != []:
                Владельцы = str(Владельцы)
                b1 = Владельцы.rfind('CardInfo__cell')
                b2 = Владельцы.rfind('</span>')
                Владельцы2 = Владельцы[b1+16:b2-3]
                if Владельцы2[2]=='в':
                    dfForFill.Владельцы = int(Владельцы2[0])
                else: 
                    dfForFill.Владельцы = int('4')
            else: dfForFill.Владельцы = int('0')

            # Достаем ПТС
            ПТС = page.find_all(class_="CardInfo__row CardInfo__row_pts")
            ПТС = str(ПТС)
            b1 = ПТС.rfind('CardInfo__cell')
            b2 = ПТС.rfind('</span>')
            dfForFill.ПТС = str.lower((ПТС[b1+16:b2]))

            # Достаем инфо по таможне
            Таможня = page.find_all(class_="CardInfo__row CardInfo__row_customs")
            Таможня = str(Таможня)
            b1 = Таможня.rfind('CardInfo__cell')
            b2 = Таможня.rfind('</span>')
            dfForFill.Таможня = str.lower((Таможня[b1+16:b2]))

            # Достаем инфо по времени владения
            Владение = page.find_all(class_="CardInfo__row CardInfo__row_owningTime")
            if (Владение==[]):
                dfForFill.Владение = np.NaN                
            else:
                Владение = str(Владение)
                b1 = Владение.rfind('CardInfo__cell')
                b2 = Владение.rfind('</span>')
                dfForFill.Владение = str.lower((Владение[b1+16:b2]))

            # Достаем Цену
            Price = page.find_all(class_="OfferPriceCaption__price")
            if (Price==[]):
                dfForFill.Price = np.NaN  
            else:
                Price = str(Price)
                Price = (Price[40:Price.find('</span>')-2])
                if len(Price.split())==1:
                    dfForFill.Price = int(Price)
                else:
                    aa = Price.find('\xa0')
                    dfForFill.Price = int(Price.replace(Price[aa],''))

        return(dfForFill)
    else: return(dfForFill)

for i in range(len(dfForTraining),len(url_list)):
    if i%100==0: print(i)
    ppp = carsData(url_list[i])
    dfForTraining = dfForTraining.append(ppp, ignore_index=True)
    if i %1000 ==0:
        dfForTraining.to_csv(('training_new'+str(i)+'.csv'),encoding='utf-8-sig', index=False)
dfForTraining.dropna(subset=['bodyType','fuelType','Price'],axis=1,inplace = True)

def modifyFeat(line):
    if pd.isna(line):
        return line
    else:
        if line.rfind('>')==-1:
            return line
        else:
            pp = line.rfind('>')
            return(line[pp+1:])
        
for i in dfForTraining.columns:
    print(i)
    print(dfForTraining[i].unique())
            
            
dfForTraining.bodyType = dfForTraining.bodyType.apply(modifyFeat)
dfForTraining.fuelType = dfForTraining.fuelType.apply(modifyFeat)
dfForTraining.to_csv('training_last_new.csv',encoding='utf-8-sig', index=False)
dfForTraining = pd.read_csv('training_last_new.csv')

def first(line):
    return(line.split(' ')[0])

def ch3to4(line):
    if line==4: return 3
    else: return line

def mult10(line):
    return(line*10)
    
def preprocTrain(df):
    df.name = df.name.apply(first)
    df.Владельцы = df.Владельцы.apply(ch3to4)
    df.drop('model',axis = 1, inplace=True)
    df.engineDisplacement= df.engineDisplacement.apply(mult10)
    for feature in ['productionDate', 'enginePower','engineDisplacement','mileage', 'Владельцы','Price']:
        df[feature]=df[feature].astype('int32')
    return df

dfForTraining = preprocTrain(dfForTraining)

# dfForTraining.head()
dfForTraining.info()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

le = LabelEncoder()

le.fit(dfForTraining['bodyType'])
dfForTraining['bodyType Code'] = le.transform(dfForTraining['bodyType'])
dfForTraining.drop(['bodyType'],axis = 1, inplace = True)


le.fit(dfForTraining['color'])
dfForTraining['Color Code'] = le.transform(dfForTraining['color'])
dfForTraining.drop(['color'],axis = 1, inplace = True)

le = LabelEncoder()
le.fit(dfForTraining['fuelType'])
dfForTraining['Fuel Code'] = le.transform(dfForTraining['fuelType'])
dfForTraining.drop(['fuelType'],axis = 1, inplace = True)

le = LabelEncoder()
le.fit(dfForTraining['name'])
dfForTraining['name Code'] = le.transform(dfForTraining['name'])
dfForTraining.drop(['name'],axis = 1, inplace = True)

le.fit(dfForTraining['vehicleTransmission'])
dfForTraining['Transmission Code'] = le.transform(dfForTraining['vehicleTransmission'])
dfForTraining.drop(['vehicleTransmission'],axis = 1, inplace = True)

le.fit(dfForTraining['Привод'])
dfForTraining['Привод Code'] = le.transform(dfForTraining['Привод'])
dfForTraining.drop(['Привод'],axis = 1, inplace = True)

le.fit(dfForTraining['Руль'])
dfForTraining['Руль Code'] = le.transform(dfForTraining['Руль'])
dfForTraining.drop(['Руль'],axis = 1, inplace = True)

le.fit(dfForTraining['Состояние'])
dfForTraining['Состояние Code'] = le.transform(dfForTraining['Состояние'])
dfForTraining.drop(['Состояние'],axis = 1, inplace = True)

le.fit(dfForTraining['ПТС'])
dfForTraining['ПТС Code'] = le.transform(dfForTraining['ПТС'])
dfForTraining.drop(['ПТС'],axis = 1, inplace = True)

le.fit(dfForTraining['Таможня'])
dfForTraining['Таможня Code'] = le.transform(dfForTraining['Таможня'])
dfForTraining.drop(['Таможня'],axis = 1, inplace = True)
def spl(stra):
    return(stra.split()[0])

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

dfForTest = pd.read_csv('test.csv')

def preprocTest(df):
    df.at[823,'engineDisplacement']='2.0'
    df.engineDisplacement = df.engineDisplacement.apply(spl).astype('float64')
    df.engineDisplacement= df.engineDisplacement.apply(mult10)
    df.enginePower = df.enginePower.apply(spl).astype('float64')
    df.Владельцы = df.Владельцы.apply(spl).astype('float64')
    df.name = df.name.apply(first)
    df.drop(['brand'], axis = 1, inplace = True)
    df.drop(['modelDate'], axis = 1, inplace = True)
    df.drop(['numberOfDoors'], axis = 1, inplace = True)
    df.drop(['vehicleConfiguration'], axis = 1, inplace = True)
    df.drop(['description'], axis = 1, inplace = True)
    df.drop(['Комплектация'], axis = 1, inplace = True)
    df.drop(['Владение'], axis = 1, inplace = True)
    df.drop(['id'], axis = 1, inplace = True)
    for feature in ['productionDate', 'enginePower','engineDisplacement','mileage', 'Владельцы']:
        df[feature]=df[feature].astype('int32')
    
    return df

dfForTest = preprocTest(dfForTest)
dfForTest.info()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

le = LabelEncoder()

le.fit(dfForTest['bodyType'])
dfForTest['bodyType Code'] = le.transform(dfForTest['bodyType'])
dfForTest.drop(['bodyType'],axis = 1, inplace = True)


le.fit(dfForTest['color'])
dfForTest['Color Code'] = le.transform(dfForTest['color'])
dfForTest.drop(['color'],axis = 1, inplace = True)

le = LabelEncoder()
le.fit(dfForTest['fuelType'])
dfForTest['Fuel Code'] = le.transform(dfForTest['fuelType'])
dfForTest.drop(['fuelType'],axis = 1, inplace = True)

le = LabelEncoder()
le.fit(dfForTest['name'])
dfForTest['name Code'] = le.transform(dfForTest['name'])
dfForTest.drop(['name'],axis = 1, inplace = True)

le.fit(dfForTest['vehicleTransmission'])
dfForTest['Transmission Code'] = le.transform(dfForTest['vehicleTransmission'])
dfForTest.drop(['vehicleTransmission'],axis = 1, inplace = True)

le.fit(dfForTest['Привод'])
dfForTest['Привод Code'] = le.transform(dfForTest['Привод'])
dfForTest.drop(['Привод'],axis = 1, inplace = True)

le.fit(dfForTest['Руль'])
dfForTest['Руль Code'] = le.transform(dfForTest['Руль'])
dfForTest.drop(['Руль'],axis = 1, inplace = True)

le.fit(dfForTest['Состояние'])
dfForTest['Состояние Code'] = le.transform(dfForTest['Состояние'])
dfForTest.drop(['Состояние'],axis = 1, inplace = True)

le.fit(dfForTest['ПТС'])
dfForTest['ПТС Code'] = le.transform(dfForTest['ПТС'])
dfForTest.drop(['ПТС'],axis = 1, inplace = True)

le.fit(dfForTest['Таможня'])
dfForTest['Таможня Code'] = le.transform(dfForTest['Таможня'])
dfForTest.drop(['Таможня'],axis = 1, inplace = True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics 

from catboost import CatBoostRegressor

VAL_SIZE   = 0.33
N_FOLDS    = 10

# CATBOOST
ITERATIONS = 4000
LR         = 0.05

RANDOM_SEED = 42
VERSION=3

X = dfForTraining.drop(['Price'],axis=1)
y = dfForTraining.Price

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size = VAL_SIZE, random_state=RANDOM_SEED)

cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()

model = CatBoostRegressor(iterations = ITERATIONS,
                          learning_rate = LR,
                          random_seed = RANDOM_SEED,
                          eval_metric='MAPE',
                          custom_metric=['R2', 'MAE']
                         )
model.fit(X_train, y_train,
         cat_features=cat_features_ids,
         eval_set=(X_test, y_test),
         verbose_eval=100,
         use_best_model=True,
         plot=True
         )
# Посмотрим важность данных для модели.
# Выглядит все весьма логично для меня. Дата на первом месте, если бы была дата модели было бы наверное еще лучше, 
# но в обучающей выборке этоих данных не было (позднее на странице auto.ru была найдена возможность достать данные по модели,
# но как говорилось выше повторный парсинг провести не было возможности :-( ).

features_importances = pd.DataFrame(data = model.feature_importances_, index = X.columns, columns = ['FeatImportant'])
features_importances.sort_values(by = 'FeatImportant', ascending = False).head(20)
predict_submission = model.predict(dfForTest)
predict_submission
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['price'] = predict_submission
sample_submission.to_csv(f'submission_v3.csv', index=False)
sample_submission.head(10)
def cat_model(y_train, X_train, X_test, y_test):
    model = CatBoostRegressor(iterations = ITERATIONS,
                              learning_rate = LR,
                              eval_metric='MAPE',
                              random_seed = RANDOM_SEED,)
    model.fit(X_train, y_train,
              cat_features=cat_features_ids,
              eval_set=(X_test, y_test),
              verbose=False,
              use_best_model=True,
              plot=False)
    
    return(model)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))

submissions = pd.DataFrame(0,columns=["sub_1"], index=sample_submission.index) # куда пишем предикты по каждой модели
score_ls = []
splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))

for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS,):
    # use the indexes to extract the folds in the train and validation data
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
    # model for this fold
    model = cat_model(y_train, X_train, X_test, y_test,)
    # score model on test
    test_predict = model.predict(X_test)
    test_score = mape(y_test, test_predict)
    score_ls.append(test_score)
    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")
    # submissions
    submissions[f'sub_{idx+1}'] = model.predict(dfForTest)
    model.save_model(f'catboost_fold_{idx+1}.model')
    
print(f'Mean Score: {np.mean(score_ls):0.3f}')
print(f'Std Score: {np.std(score_ls):0.4f}')
print(f'Max Score: {np.max(score_ls):0.3f}')
print(f'Min Score: {np.min(score_ls):0.3f}')

sample_submission = pd.read_csv('sample_submission.csv')
submissions['blend'] = (submissions.sum(axis=1))/len(submissions.columns)
sample_submission['price'] = submissions['blend'].values
sample_submission.head(10)
sample_submission.price = round(sample_submission.price/10000,0)*10000*0.92
sample_submission.to_csv(f'submission_blend_v3.csv', index=False)
sample_submission.head(10)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from collections import defaultdict

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'RMSE = {rmse:.2f}, MAE = {mae:.2f}, R-sq = {r2:.2f}, MAPE = {mape:.2f} ')
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor

def compute_meta_feature(clf, X_train, X_test, y_train, cv):

    X_meta_train = np.zeros_like(y_train, dtype = np.float32)
    X_meta_test = np.zeros(len(X_test), dtype=np.float32)
    for train_fold_index, predict_fold_index in cv.split(X_train):
        X_fold_train, X_fold_predict = X_train.iloc[train_fold_index], X_train.iloc[predict_fold_index]
        y_fold_train = y_train[train_fold_index]
        folded_clf = clone(clf)
        
        if type(clf).__name__ == 'CatBoostRegressor':
            folded_clf.fit(X_fold_train, y_fold_train, cat_features=cat_features_ids, verbose_eval = 100)
        else:
            folded_clf.fit(X_fold_train, y_fold_train)
            
        X_meta_train[predict_fold_index] = folded_clf.predict(X_fold_predict)
        print_regression_metrics(X_meta_train[predict_fold_index], y_train.iloc[predict_fold_index])
        X_meta_test += folded_clf.predict(X_test)
    X_meta_test = X_meta_test / cv.n_splits

    return X_meta_train, X_meta_test



def generate_meta_features(classifiers, X_train, X_test, y_train, cv):
    features = [
        compute_meta_feature(clf, X_train, X_test, y_train, cv)
        for clf in tqdm(classifiers)
    ]

    stacked_features_train = np.stack([
        features_train for features_train, features_test in features
        ],axis=-1)

    stacked_features_test = np.stack([
        features_test for features_train, features_test in features
        ],axis=-1)

    return stacked_features_train, stacked_features_test
X = dfForTraining.drop(['Price'],axis=1)
y = dfForTraining.Price

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size = VAL_SIZE, random_state=RANDOM_SEED)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
model_rf = RandomForestRegressor(n_estimators=10, random_state=RANDOM_SEED)
model_bet = BaggingRegressor(ExtraTreeRegressor(random_state=RANDOM_SEED))
model_cb = CatBoostRegressor( 
#                           learning_rate = 0.1,
                          random_seed = 42,
                          eval_metric='MAPE')
X_train.reset_index(drop=True,inplace = True)
y_train.reset_index(drop=True, inplace = True)
stacked_features_train, stacked_features_test = generate_meta_features([model_rf,model_bet, model], X_train, 
                                                                       dfForTest, y_train, cv)


from sklearn.linear_model import Ridge
final_model = Ridge(alpha=20).fit(stacked_features_train, y_train)
y_pred = np.round((final_model.predict(stacked_features_test)/1000))*1000

display(len(y_pred))
# print_regression_metrics(y_test, y_pred)
y_pred_stck = np.round((final_model.predict(stacked_features_test)/1000))*1000

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['price'] =  y_pred_stck
sample_submission.to_csv(f'submission_stack_v1.csv', index=True)

sample_submission.head(10)
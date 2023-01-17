# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from pandas import Series,DataFrame





import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline





from sklearn.linear_model import LinearRegression
train_df  = pd.read_csv("../input/train.csv")

store_df = pd.read_csv("../input/store.csv")

test_df  = pd.read_csv("../input/test.csv")



train_df.head()


fig, (axis1) = plt.subplots(1,1,figsize=(12,4))

sns.countplot(x='Open',hue='DayOfWeek', data=train_df, palette="husl", ax=axis1)



print(test_df["Open"].value_counts())

print(test_df["Open"].shape)



# заполняем пропущенные значения если не воскресенье

test_df["Open"][test_df["Open"] != test_df["Open"]] = (test_df["DayOfWeek"] != 7).astype(int)

# очевидно что большинство не работает в воксресенье


# Создаем колонки месяц и год

train_df['Year']  = train_df['Date'].apply(lambda x: int(str(x)[:4]))

train_df['Month'] = train_df['Date'].apply(lambda x: int(str(x)[5:7]))



test_df['Year']  = test_df['Date'].apply(lambda x: int(str(x)[:4]))

test_df['Month'] = test_df['Date'].apply(lambda x: int(str(x)[5:7]))



# Меняем формат даты в год-месяц

train_df['Date'] = train_df['Date'].apply(lambda x: (str(x)[:7]))

test_df['Date']     = test_df['Date'].apply(lambda x: (str(x)[:7]))





average_sales    = train_df.groupby('Date')["Sales"].mean()

pct_change_sales = train_df.groupby('Date')["Sales"].sum().pct_change()







average_sales.plot(legend=True,marker='o',title="Average Sales",figsize=(12,4))
# распределние продаж и посетителей по годам

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(12,4))



sns.barplot(x='Year', y='Sales', data=train_df, ax=axis1)

sns.barplot(x='Year', y='Customers', data=train_df, ax=axis2)
#разбираемся со StateHoliday

sns.barplot(x='StateHoliday', y='Sales', data=train_df)

print(train_df["StateHoliday"].value_counts())
# у нас есть разные нули((( ну и пусть каникулы не отличаются

train_df["StateHoliday"] = train_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

test_df["StateHoliday"]  = test_df["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
# а теперь с SchoolHoliday

sns.barplot(x='SchoolHoliday', y='Sales', data=train_df)

# тут все ок
# Переходим к информации из store_df







# добавляем к ней данные из train_df чтобы посмотреть зависимости продаж от типов магазинов и прочего

average_sales_customers = train_df.groupby('Store')[["Sales", "Customers"]].mean()

sales_customers_df = DataFrame({'Store':average_sales_customers.index,

                      'Sales':average_sales_customers["Sales"], 'Customers': average_sales_customers["Customers"]}, 

                      columns=['Store', 'Sales', 'Customers'])

store_df = pd.merge(sales_customers_df, store_df, on='Store')



store_df.head()
sns.barplot(x='StoreType', y='Sales', data=store_df)

# тип b самый доходный
# теперь посмотрим по ассортименту

sns.barplot(x='Assortment', y='Customers', data=store_df)

# опять тип b самый крутой
# теперь посмотрим  Promo2 это вроде реклама магазина

sns.barplot(x='Promo2', y='Sales', data=store_df)

# получается покупают больше когда не рекламируется хмм
# CompetitionDistance разберемся с конкурентами



# fill NaN values да тут куча пустых данных

store_df["CompetitionDistance"].fillna(store_df["CompetitionDistance"].median())



# посмотрим как у нас зависят продажи

store_df.plot(kind='scatter',x='CompetitionDistance',y='Sales',figsize=(12,4))

# похоже что продажи не шибко зависят от дальности конкурентов, видимо где много конурентов там и более проходное место
# что произойдет сос редними продажами т-го магазина когда откроется рядом конкурент?



store_id = 6

store_data = train_df[train_df["Store"] == store_id]





average_store_sales = store_data.groupby('Date')["Sales"].mean()



# Находим день и месяц когда появился конкурент

y = store_df["CompetitionOpenSinceYear"].loc[store_df["Store"]  == store_id].values[0]

m = store_df["CompetitionOpenSinceMonth"].loc[store_df["Store"] == store_id].values[0]





ax = average_store_sales.plot(legend=True,figsize=(12,4),marker='o')

ax.set_xticks(range(len(average_store_sales)))

ax.set_xticklabels(average_store_sales.index.tolist(), rotation=90)



# очевидно что если конкурент откурылся до 2013 года то он не вносит вклад в продажи( все устаканилось)

# ну а еще там может быть пусто

if y >= 2013 and y == y and m == m:

    plt.axvline(x=((y-2013) * 12) + (m - 1), linewidth=3, color='grey')
store_df["CompetitionOpenSinceYear"].fillna(0, inplace = True)




mask= np.zeros(train_df.shape[0])

for i in range(train_df.shape[0]):

    if store_df.loc[i%1115,"CompetitionOpenSinceYear"] == 0:

        mask[i] = 1

        continue

    if i%10000 == 0:

        print(i)

    z = (train_df.loc[i,'Year']> store_df.loc[i%1115,"CompetitionOpenSinceYear"])

    if z:

            mask[i] = 1

    if (train_df.loc[i,'Year'] == store_df.loc[i%1115,"CompetitionOpenSinceYear"]) and (train_df.loc[i,'Month'] >= store_df.loc[i%1115,"CompetitionOpenSinceMonth"]):

            mask[i] = 1

        

    
mask_true = (mask==1)



train_df = train_df[mask_true]


        
# в тесте у нас только 9 и 10 месяц 2015 года



train_df.drop(["Year", "Month"], axis=1, inplace=True)

test_df.drop(["Year", "Month"], axis=1, inplace=True)

# удаляем столбцы которые уже не нужны



# создаем dummy для дней недели( воскресенье не нужно)

day_dummies_train  = pd.get_dummies(train_df['DayOfWeek'], prefix='Day')

day_dummies_train.drop(['Day_7'], axis=1, inplace=True)



day_dummies_test  = pd.get_dummies(test_df['DayOfWeek'],prefix='Day')

day_dummies_test.drop(['Day_7'], axis=1, inplace=True)



train_df = train_df.join(day_dummies_train)

test_df     = test_df.join(day_dummies_test)



train_df.drop(['DayOfWeek'], axis=1,inplace=True)

test_df.drop(['DayOfWeek'], axis=1,inplace=True)

train_df.head()
# удаляем все что закрыто

train_df = train_df[train_df["Open"] != 0]

# удаляем так как не нужно будет далее

train_df.drop(["Open", "Customers", "Date"], axis=1, inplace=True)
# сохраняем номера закрытых магазинов в тесте, они всеравно ничего не продадут - проставим им потом нули

closed_store_ids = test_df["Id"][test_df["Open"] == 0].values



# удаляем все закрытые магазины

test_df = test_df[test_df["Open"] != 0]



# drop unnecessary columns, these columns won't be useful in prediction

test_df.drop(['Open', 'Date'], axis=1,inplace=True)
train_dic  =  dict(list(train_df.groupby('Store')))

test_dic     = dict(list(test_df.groupby('Store')))

submission   = Series()
print(train_df['Store'].value_counts().shape)
# Изучали регрессию - будет регрессия

# пробежимся по всем магазинам





for i in test_dic:

    

    

    store = train_dic[i]

    X_train = store.drop(["Sales","Store"],axis=1)

    Y_train = store["Sales"]

    X_test  = test_dic[i].copy()

    

    store_ids = X_test["Id"]

    X_test.drop(["Id","Store"], axis=1,inplace=True)

    

    

    lreg = LinearRegression()

    lreg.fit(X_train, Y_train)

    Y_pred = lreg.predict(X_test)

   



    submission = submission.append(Series(Y_pred, index=store_ids))





submission = submission.append(Series(0, index=closed_store_ids))





submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})

submission.to_csv('otvet1.csv', index=False)


import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
# Исходная обучающая выборка

data_train_b = pd.read_csv('../input/train.csv')

# Исходна тестовая выборка

data_test_b = pd.read_csv('../input/test.csv')
# Обработаем обучающие данные

data_train_b.head()
# Изменим формат даты в данных

data_train = data_train_b.copy()

data_train['Year'] = data_train.Date.apply(lambda x: x.split('-'))

data_train['Month'] = data_train.Year.apply(lambda x: int(x[1]))

data_train['Day'] = data_train.Year.apply(lambda x: int(x[2]))

data_train['Year'] = data_train.Year.apply(lambda x: int(x[0]))

data_train = data_train.drop(["Date"], axis = 1)

data_train.head()
data_test = data_test_b.copy()

data_test['Year'] = data_test.Date.apply(lambda x: x.split('-'))

data_test['Month'] = data_test.Year.apply(lambda x: int(x[1]))

data_test['Day'] = data_test.Year.apply(lambda x: int(x[2]))

data_test['Year'] = data_test.Year.apply(lambda x: int(x[0]))

data_test = data_test.drop(["Date"], axis = 1)

data_test.head()
# слишком много значений, округлим

# дерево глубины 10 уже справится

print (len(data_train.Sales.apply(lambda x: 50*int(round(x / 50.0))).unique()))
def sf(x):

    if x == '0':

        return 0

    elif x == 'a':

        return 1

    elif x == 'b':

        return 2

    elif x == 'c':

        return 3

    else:

        return x

    

def sfa(x):

    if x == 'a':

        return 1

    else:

        return 0

def sfb(x):

    if x == 'b':

        return 1

    else:

        return 0

def sfc(x):

    if x == 'c':

        return 1

    else:

        return 0
data_train_post = data_train.copy()

data_train_post['PublicHoliday'] = data_train_post.StateHoliday.apply(sfa)

data_train_post['EasternHoliday'] = data_train_post.StateHoliday.apply(sfb)

data_train_post['XmasHoliday'] = data_train_post.StateHoliday.apply(sfc)

#data_train_post.Sales = data_train.Sales.apply(lambda x: 40*int(round(x / 40.0)))

data_train_post = data_train_post.drop(["StateHoliday", "Day", "Customers"], axis=1)

print (len(data_train_post.Sales.unique()))
data_train_post.head()
data_test.head()
print (data_test.Id.unique())

print (len(data_test.Id.unique()))

print (sum(data_test.Id.unique()!=np.arange(1, 41088+1)))

# Id можно дропунуть, номера по порядку
def lxnan(x):

    if (x != 1.0 and x != 0.0):

        return 1.0

    else:

        return x



for i in data_test.values:

    if (i[3] != 1.0 and i[3] != 0):

        print (i)

# Магазин явно открыт
data_test_post = data_test.copy()

data_test_post.head()

data_test_post['PublicHoliday'] = data_test_post.StateHoliday.apply(sfa)

data_test_post['EasternHoliday'] = data_test_post.StateHoliday.apply(sfb)

data_test_post['XmasHoliday'] = data_test_post.StateHoliday.apply(sfc)

data_test_post = data_test_post.drop(["StateHoliday", "Day"], axis=1)
data_train_post.head()
data_test_post.head()
# проверим работоспособность дерева

data_train1 = data_train_post[:1000000].copy()

data_test1 = data_train_post[1000000:].copy()
data_train_t = data_train1.drop(['Sales'], axis=1)

data_train_c = data_train1['Sales']

data_test_t = data_test1.drop(['Sales'], axis=1)

data_test_c = data_train_b[1000000:]['Sales']
data_train_t.head()
data_train_c.head()
data_test_t.head()
data_test_c.head()
from sklearn.ensemble import RandomForestRegressor
data_train_t
par=[]

for i in range(40, 41):

    model = RandomForestRegressor(max_depth=i, n_jobs=-1, n_estimators=15)

    model.fit(data_train_t, data_train_c)

    c = model.predict(data_test_t)

    parpar = 0

    for k, l in zip(data_test_c, c):

        if k != 0:

             parpar += np.square((k - round(l))/float(k))

    parpar = np.sqrt(parpar/float(len(c)))

    print (i,parpar)
data_train_t = data_train_post.drop(["Sales"], axis=1)

data_train_c = data_train_post["Sales"]
data_train_t.head()
data_train_c.head()
model = RandomForestRegressor(max_depth=40, n_jobs=-1, n_estimators=15)

model.fit(data_train_t, data_train_c)
data_test_post.fillna(value=1,axis='index', inplace=True)

data_test_post=data_test_post.drop(['Id'],axis=1)
data_test_c = model.predict(data_test_post)
df = pd.DataFrame({"Id": data_test_b.Id, "Sales": data_test_c})
df.index=df.Id

df=df.drop(['Id'],axis=1)

df.head()
df.to_csv('predictions.csv')
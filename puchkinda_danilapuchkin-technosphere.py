import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')
# Исходная обучающая выборка

data_train_b = pd.read_csv('../input/train.csv')

data_train_b
# Исходна тестовая выборка

data_test_b = pd.read_csv('../input/test.csv')

data_test_b
# Обработаем обучающие данные

data_train_b.head()
type(data_train_b.Date[0])
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
# Проверим данные на корректность

print(data_train.Store.unique())

print(sum(data_train.Store.unique()!=np.arange(1, 1115+1)))

# Номера магазинов от 1 до 1115, в порядке
print(data_train.DayOfWeek.unique())

print(data_train.Open.unique())

print(data_train.Promo.unique())

print(data_train.Sales.unique()) # править

print(data_train.StateHoliday.unique()) # править

print(data_train.Customers.unique())# дропнуть, в тестовой выборке нет такой колонки

print(data_train.SchoolHoliday.unique())

print(data_train.Year.unique())

print(data_train.Month.unique())

print(data_train.Day.unique())
print(len(data_train.Sales.unique()))

# слишком много значений, округлим
# дерево глубины 10 уже справится

print(len(data_train.Sales.apply(lambda x: 100*int(round(x / 100.0))).unique()))
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
data_train_post = data_train.copy()

data_train_post.StateHoliday = data_train_post.StateHoliday.apply(sf)

data_train_post.Sales = data_train.Sales.apply(lambda x: 10*int(round(x / 10.0)))

data_train_post = data_train_post.drop(["Year", "Month", "Customers"], axis=1) # ниже показано, почему можно дропнуть

print(data_train_post.StateHoliday.unique())

print(len(data_train_post.Sales.unique()))
data_train_post.head()
data_test.head()
print(data_test.Id.unique())

print(len(data_test.Id.unique()))

print(sum(data_test.Id.unique()!=np.arange(1, 41088+1)))

# Id можно дропунуть, номера по порядку
print(data_test.DayOfWeek.unique())

print(data_test.Open.unique()) #править

print(data_test.Promo.unique()) 

print(data_test.StateHoliday.unique()) # править

print(data_test.SchoolHoliday.unique()) 

print(data_test.Year.unique()) # стоит дропнуть

print(data_test.Month.unique()) # стоит дропнуть

print(data_test.Day.unique())
def lxnan(x):

    if (x != 1.0 and x != 0.0):

        return 1.0

    else:

        return x



for i in data_test.values:

    if (i[3] != 1.0 and i[3] != 0):

        print(i)

# Магазин явно открыт
data_test_post = data_test.copy()

data_test_post.Open = data_test_post.Open.apply(lxnan)

data_test_post = data_test_post.drop(["Id", "Year", "Month"], axis=1)

data_test_post.StateHoliday = data_train_post.StateHoliday.apply(sf)
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
from sklearn.tree import DecisionTreeClassifier
# больше 20 начинает лететь память

# возьмем 20

for i in range(9, 20+1):

    model = DecisionTreeClassifier(max_depth=i)

    model.fit(data_train_t, data_train_c)

    c = model.predict(data_test_t)

    parpar = 0

    for k, l in zip(data_test_c, c):

        if k != 0:

            parpar += np.square((k - round(l))/float(k))

    parpar = np.sqrt(parpar/float(len(c)))

    print(i, parpar)
data_train_t = data_train_post.drop(["Sales"], axis=1)

data_train_c = data_train_post["Sales"]
data_train_t.head()
data_train_c.head()
model = DecisionTreeClassifier(max_depth=20)

model.fit(data_train_t, data_train_c)
data_test_c = model.predict(data_test_post)
df = pd.DataFrame({"Id": data_test_b.Id, "Sales": data_test_c})
df.head()
df.to_csv('predictions.csv')
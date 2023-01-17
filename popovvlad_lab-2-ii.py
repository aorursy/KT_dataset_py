import numpy as np
# Создание массива (ndarray) из списка (list)

list_1 = [0,1,2,3,4]

arr_1d = np.array(list_1)

print(arr_1d)

# Сделайте print полученного массива и его типа

# Ваш код
# Прибавление и вычитание числа



for i in range(len(list_1)):

    list_1[i] += 1

    

arr_1d += 1



print(list_1)

print(arr_1d)



for i in range(len(list_1)):

    list_1[i] -= 1

    

arr_1d -= 1



print(list_1)

print(arr_1d)



# Также работает умножение и деление
# Массивы могут быть созданы различными способами

zeros = np.zeros(10)

ones = np.ones(10)

arange = np.arange(10)

linspace = np.linspace(0, 1, 5)



print('zeros', zeros)

print('ones', ones)

print('arange', arange)

print('linspace', linspace)
# Массивы могут складываться, вычитаться, умножаться, делится



print(ones+arange)

print(ones*arange)

print(ones/arange) # имеется возможность для деления на 0 - приводит к inf, отсутствующие значения к np.nan
# Индексация и двумерные массивы

print(np.arange(10)[5:9])



arange = np.arange(10).reshape(2, 5) # команда reshape в данном случае позволяет получуть двумерный массив

print(arange)

print(arange[:, 3:])



# также можно проводить экстракцию по условию

print(arange[arange%2 == 1])
my_array = np.linspace(5,100,20)

my_array

my_array = my_array[1:20:2]

my_array
my_array = my_array.reshape(5,2)

my_array
my_second_array = my_array - 5

my_second_array
result = np.concatenate((my_array, my_second_array), axis= 1)

my_array = np.delete(my_array.T,[4], axis = 1)

result



result[1] = result[1]/my_array[0]

result[2] = result[2]/my_array[1]

result
result = result[np.where(result>3)]

result

result = result.reshape(3,4)

result1 = result.copy()

result[0] = result1[2]

result[2] = result1[0]

result
print("Среднее") 

print(np.mean(result, axis = 1))

print("Медиана")  

print(np.median(result, axis = 1))

print("Стандартное отклонение")  

print(np.std(result, axis = 1))
result[0] = result[0]/result[0].max()

result[1] = result[1]/result[1].max()

result[2] = result[2]/result[2].max()

result
nall = np.random.rand(1000,2)

arr = np.sqrt(pow(nall[:,0],2) + pow(nall[:,1],2))

inin = np.where(arr<1,arr,0)

pi = 4 * inin/arr

pi.sum()
import pandas as pd

df = pd.DataFrame({'int_col' : [1,2,6,8,-1], 'float_col' : [0.1, 0.2,0.2,10.1,None], 'str_col' : ['a','b',None,'c','a']})
df
# индексация

df.loc[:,['float_col','int_col']]
# загрузка csv таблицы

df = pd.read_csv('../input/lab-2-2sim/churn.csv')

df.head()
df.shape
df.columns
# описание переменных можно получить методом info

df.info()
# получить быстро различные статистики можно получить методом describe

df.describe()
# статистики по другим переменным

df.describe(include=['object', 'bool'])
# сортировка

df = df.sort_values(by='total day charge', ascending=False)

df.head()
# индексирование iloc

df.iloc[:15]
# индексирование loc

df.loc[:15]
# сортировка по разным переменным

df.sort_values(by=['account length', 'total day charge'],

        ascending=[True, False]).head()
# отбор по условию и вывод средних значений

print(df[df['account length'] == 1].mean())

# отбор по нескольким условиям и получение максимального значения одного признака

print(df[(df['account length'] == 1) & (df['international plan'] == 'no')]['total intl minutes'].max())
# группирование осуществляется методом groupby



columns_to_show = ['total day minutes', 'total eve minutes', 

                   'total night minutes']



df.groupby(['churn'])[columns_to_show].describe(percentiles=[])
data = pd.read_csv('../input/adyulttxt/adult.txt', sep=', ')

data.head() 
sexm = data['sex'] == 'Male'

sexf = data['sex'] == 'Female'

print(sexm.sum(), ',', sexf.sum())
agem=data.query("sex=='Male'")['age'].mean()

agef=data.query("sex=='Female'")['age'].mean()

print(agem,',', agef)
countrycub = data['native-country']=='Cuba'

countcub = countrycub.sum()

countryall = data['native-country'].count()

proccub = countcub/countryall *100

print(proccub.round(decimals=1),'%')
meanage = data.query("salary=='<=50K'")['age'].mean()

meanstd = data.query("salary=='<=50K'")['age'].std()

meanage1 = data.query("salary=='>50K'")['age'].mean()

meanstd1 = data.query("salary=='>50K'")['age'].std()

print(meanage,',', meanstd,' ',' Меньше 50К','\n',meanage1,',', meanstd1, ' ',' Больше 50К')

arreducat = data.query("salary=='>50K'")['education']

people = arreducat.isin(['Bachelors','Prof-school','Assoc-acdm','Assoc-voc','Masters or Doctorate feature']).sum()

allpeople = data.query("salary=='>50K'")['education'].count()

people = people/allpeople * 100

print('Процент людей которые зарабатывают больше 50К, имея высшее образование равен',people.round(decimals=0),'%')
stat = data.groupby(['race','sex'])

stat.age.describe()
male =  data.query("salary=='>50K'")['marital-status']

marmale = male.isin(['Married-civ-spouse','Married-spouse-absent', 'Married-AF-spouse']).sum()

notmarmale = male.isin(['Never-married', 'Divorced', 'Separated','Widowed']).sum()

print(' Количество женатых мужчин равно', marmale,'\n','Количество не женатых мужчин равно', notmarmale)
counthours = data['hours-per-week'].max()

maxcounthours = data['hours-per-week'] == counthours

allpeople = data.query("salary=='>50K'")[maxcounthours]

countrichpeople= allpeople['hours-per-week'].count()

procrichpeople = countrichpeople/maxcounthours.sum() * 100

print(' Максимальное значение по количеству рабочих часов в неделю равно',counthours,'\n','Количество людей которые работают 99 часов в неделю равно',

maxcounthours.sum(),'\n','Процент тех кто работает 99 часов в неделю и зарабатывает больше 50К равен',procrichpeople.round(decimals=0),'%')
counthours = data.query("salary=='<=50K'")['hours-per-week'].groupby(data['native-country']).mean()

counthours1 = data.query("salary=='>50K'")['hours-per-week'].groupby(data['native-country']).mean()

print(' Среднее количество рабочих часов в неделю для тех кто зарабатывает мало','\n',counthours,'\n','Среднее количество рабочих часов в неделю для тех кто зарабатывает много','\n',counthours1)
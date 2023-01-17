# ОПИСАТЕЛЬНАЯ СТАТИСТИКА
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

file = open('/kaggle/input/adult-data-5/train_data.csv', 'r') # считываем файл

def chr_int(a): # метод, необходимый для того, чтобы считать цифры в файле в int
    if a.isdigit(): return int(a)
    else: return 0

data = [] # перезапишем данные в список
for line in file: 
    data1 = line.split(', ')
    if len(data1) == 15:
        data.append([chr_int(data1[0]), data1[1], chr_int(data1[2]), data1[3], chr_int(data1[4]), data1[5], data1[6],data1[7],data1[8],data1[9], chr_int(data1[10]), chr_int(data1[11]), chr_int(data1[12]), data1[13], data1[14]]) # считываем: chr_int - там, где в файле будут цифры
    
print(data[1:2]) # проверяем прочитанный файл

df = pd.DataFrame(data) # поместим список в dataframe, пометив оси
df.columns = ['age', 'type_employer','fnlwgt', 'education', 'education_num', 'marital', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hr_per_week', 'country', 'income']
df.shape # проверим, сколько получилось образцов данных в строках и столбцах

print('Count of subjects: \n', df.groupby('country').size().head(), '\n') # подсчитаем количество предметов на страну

ml = df[(df.sex == 'Male')] # переменные с делением м/ж, а также м/ж с высоким доходом
ml1 = df[(df.sex == 'Male') & (df.income == '>50K\n')]
fm = df[(df.sex == 'Female')]
fm1 = df[(df.sex == 'Female') & (df.income == '>50K\n')]

df1 = df[(df.income=='>50K\n')] # профессионалы с высоким доходом в целом
print ('The rate of people with high income is: ', int(len(df1)/float(len(df))*100), '%.')
print ('The rate of men with high income is: ', int(len(ml1)/float(len(ml))*100), '%.')
print ('The rate of women with high income is: ', int(len(fm1)/float(len(fm))*100), '%.', '\n')

print ('The average age of men is: ', ml['age'].mean()) # оценим их средний возраст
print ('The average age of women is: ', fm['age'].mean())
print ('The average age of high-income men is: ', ml1['age'].mean())
print ('The average age of high-income women is: ', fm1['age'].mean(), '\n')

ml_mu = ml['age'].mean()
ml_var = ml['age'].var()
ml_std = ml['age'].std()

fm_mu = fm['age'].mean()
fm_var = fm['age'].var()
fm_std = fm['age'].std()

print ('Statistics of age for men: \n mu:', ml_mu, '\n var:', ml_var, '\n std:', ml_std) # статистики для количества часов в неделю
print ('Statistics of age for women: \n mu:', fm_mu, '\n var:', fm_var, '\n std:', fm_std, '\n')

ml_median = ml['age'].median() # средний возраст
fm_median = fm['age'].median()
print ("Median age per men and women: ", ml_median, ', ', fm_median)
ml_median_age = ml1['age'].median()
fm_median_age = fm1['age'].median()
print ("Median age per men and women with high-income: ", ml_median_age, ', ', fm_median_age, '\n')

print('Quantity of men and woman: \n') # распределение количества мужчин и женщин по возрасту
ml_age = ml['age']
ml_age.hist(density=0 , histtype = 'stepfilled', bins = 20)
fm_age = fm['age']
fm_age.hist(density=0 , histtype = 'stepfilled', bins = 20)
print('\n')
import seaborn as sns # гистограмма возраста работающих
fm_age.hist(density=0, histtype = 'stepfilled', alpha = .5, bins = 20)
ml_age.hist(density=0, histtype = 'stepfilled', alpha = .5, color = sns.desaturate("indianred", .75), bins = 20)
fm_age.hist(density=1, histtype = 'stepfilled', alpha = .5, bins = 20) # нормализированная
ml_age.hist(density=1, histtype = 'stepfilled', alpha = .5, bins = 20, color = sns.desaturate("indianred",.75))
ml_age.hist(density=1, histtype = 'step', cumulative = True, linewidth = 3.5, bins = 20) # CDF
fm_age.hist(density=1, histtype = 'step', cumulative = True, linewidth = 3.5, bins = 20, color = sns.desaturate("indianred", .75))
# статистики очищенные от выбросов
df2 = df.drop(df.index[(df.income == '>50K\n') & (df['age'] > df['age'].median() + 35) & (df['age'] > df['age'].median() -15)])
ml1_age = ml1['age']
fm1_age = fm1['age']
ml2_age = ml1_age.drop(ml1_age.index[ (ml1_age > df['age'].median() + 35) & (ml1_age > df['age'].median() - 15) ])
fm2_age = fm1_age.drop(fm1_age.index[ (fm1_age > df['age'].median() + 35) & (fm1_age > df['age'].median() - 15) ])
                                                                                                                                                                    
print ("Men statistics:")
print ("Mean:", ml2_age.mean(), "Std:", ml2_age.std())
print ("Median:", ml2_age.median())
print ("Min:", ml2_age.min(), "Max:", ml2_age.max(), '\n')
print ("Women statistics:")
print ("Mean:", fm2_age.mean(), "Std:", fm2_age.std())
print ("Median:", fm2_age.median())
print ("Min:", fm2_age.min(), "Max:", fm2_age.max(), '\n')
# посмотрим, сколько выбросов удаляется
import matplotlib.pyplot as plt
plt.figure(figsize = (13.4, 5))
df.age[(df.income == '>50K\n')].plot(alpha = .25, color = 'blue')
df2.age[(df2.income == '>50K\n')].plot(alpha = .45, color = 'red')
# разница в средних значениях с выбросами и без
print ('The mean difference with outliers is: %4.2f. '% (ml_age.mean() - fm_age.mean()))
print ('The mean difference without outliers is: %4.2f.'% (ml2_age.mean() - fm2_age.mean()))
countx, divisionx = np.histogram(ml2_age)
county, divisiony = np.histogram(fm2_age)
plt.plot([(divisionx[i] + divisionx[i+1])/2 for i in range(len(divisionx) - 1)], countx - county, 'o-')
def skewness(x):
    res = 0
    m = x.mean ()
    s = x.std ()
    for i in x:
        res += (i-m) * (i-m) * (i-m)
        res /= (len(x) * s * s * s)
    return res

print ("Skewness of the male population = ", skewness(ml2_age))
print ("Skewness of the female population is = ", skewness(fm2_age), '\n')

def pearson(x):
    return 3*(x.mean() - x.median())*x.std()

print ("Pearson’s coefficient of the male population = ", pearson(ml2_age))
print ("Pearson’s coefficient of the female population = ", pearson(fm2_age))
# рассматриваем ядро Гаусса
import scipy.stats # для работы функции нормировки scipy.stats.norm.pdf
x1 = np.random.normal(-1, 0.5, 15)
x2 = np.random.normal(6, 1, 10)
y = np.r_[x1, x2]
x = np.linspace(min(y), max(y), 100)
s = 0.4
plt.plot(x, np.transpose([scipy.stats.norm.pdf(x, yi, s) for yi in y]))
plt.plot(x, np.transpose([scipy.stats.norm.pdf(x, yi, s) for yi in y]).sum(1))
plt.plot(y, np.zeros(len(y)), 'bo', ms = 10)
from scipy.stats import kde
density = kde.gaussian_kde(y)
xgrid = np.linspace(x.min(), x.max(), 200)
plt.hist(y, bins = 28, density = True)
plt.plot(xgrid, density(xgrid), 'r-')

err = 0.0
for i in range(1000):
    x = np.random.normal(0.0, 1.0, 1000)
    err += (x.mean()-0.0)**2
print ('MSE: ', err / 200)
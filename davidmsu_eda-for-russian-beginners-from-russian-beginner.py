# Библиотеки:
# Таблицами и массивы:
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats # Для 5-го этапа.

# Фиксации seed'ов и более удобного вывода:
import random
import time
from tqdm import tqdm_notebook as tqdm

# Для форматирования таблиц в более естественный вид:
from datetime import date, timedelta

# Для 4 этапа - визуализации данных:
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

# Для декомпзиции временных рядов:
import statsmodels.api as sm
from pandas.core.nanops import nanmean as pd_nanmean #  Вычисляет ср-ее, игнорируя np.nan.
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# Фуксируем seed:
seed_num = 1

random.seed(seed_num)
np.random.seed(seed_num)

# Отключаем предупреждения:
import warnings
warnings.filterwarnings('ignore')
# #Доступ к диску:
# from google.colab import drive
# drive.mount('/content/drive')
calen_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
sales_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
price_df = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

pd.options.display.max_rows = 1000 # Увеличиваем максимальное кол-во строк, чтобы лучше посмотреть на датасет.
calen_df
sales_df
price_df
main_df = sales_df.drop(['item_id','dept_id','cat_id','store_id','state_id'], axis=1) # Убираем лишнии стобцы.
main_df = main_df.rename({'id': 'date'}, axis=1).set_index('date').T                  # Теперь индексы
main_df.index = [date(2011,1,29) + timedelta(delta) for delta in range(1913)]         #  строк - даты, а столбцов
main_df.columns = list(map(lambda line: line[:-11], main_df.columns))                 #  id название товаров. 
id_dict = {prev_name: "_" + str(new_name) for new_name, prev_name in enumerate(main_df.columns)} # Переименнуем стобцы, но
name_id = dict(reversed(item) for item in id_dict.items())                                       #  сохраним старые название в виде двух словарей, чтобы 
main_df.columns = id_dict.values()                                                               #   было возможно переходить от id к названиям и от названий к id.
right = main_df.reset_index()
left = calen_df.drop(['wm_yr_wk','weekday','wday',\
               'month','year','d','event_name_1',\
               'event_name_2','event_type_2'], axis=1)
main_df = left.join(right, rsuffix='date').dropna(thresh=10).drop(['index'], axis=1).rename({'event_type_1': 'event'}).set_index('date')
main_df.iloc[[0,1,500,-1]]
# Переписать оптимальней!

price_sales_df = price_df.groupby('item_id').mean().loc[:,['sell_price']]
means = []
for item in tqdm(price_sales_df.index):
  cur_keys = [key for key in id_dict.keys() if item in key]
  means.append(main_df.loc[:,[id_dict[key] for key in cur_keys]].mean().mean())
price_sales_df.loc[:,'mean_sales'] = means
price_sales_df.head(10)
subcat_df = sales_df.drop(['id','item_id','cat_id','store_id','state_id'], axis=1) # Лишнии столбцы.
subcat_df = subcat_df.groupby('dept_id').mean().T                                  # Cтолбцы <-> строки.
subcat_df.index = [date(2011,1,29) + timedelta(delta) for delta in range(1913)]    # Переименовываем.
subcat_df.iloc[[0,1,1000,-1]]

right = subcat_df.reset_index()
left = calen_df.drop(['wm_yr_wk','weekday','wday',\
               'month','year','d','event_name_1',\
               'event_name_2','event_type_2'], axis=1)
subcat_df = left.join(right, rsuffix='date').dropna(thresh=10).drop(['index'], axis=1).rename({'event_type_1': 'event'}).set_index('date')
subcat_df.event_type_1 = subcat_df.event_type_1.fillna('Ordinary')
subcat_df.columns.name = ''                                                        # Просто так красивее)
subcat_df.head()
cat_df = sales_df.drop(['id','item_id','dept_id','store_id','state_id'], axis=1) # Лишнии столбцы.
cat_df = cat_df.groupby('cat_id').sum().T                                        # Cтолбцы <-> строки.
cat_df.index = [date(2011,1,29) + timedelta(delta) for delta in range(1913)]     # Переименовываем.
cat_df.columns.name = ''                                                         # Просто так красивее)
cat_df.iloc[[0,1,1000,-1]]
cat_df.plot();
store_df = sales_df.drop(['id','item_id','cat_id','dept_id','state_id'], axis=1) # Лишнии столбцы.
store_df = store_df.groupby('store_id').sum().T                                  # Cтолбцы <-> строки.
store_df.index = [date(2011,1,29) + timedelta(delta) for delta in range(1913)]   # Переименовываем.
store_df.columns.name = ''                                                       # Просто так красивее)
store_df.iloc[[0,1,1000,-1]]
desс_pr_df = price_df.drop(['store_id','wm_yr_wk'], axis=1).groupby('item_id').describe()
desс_pr_df.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] # Избавляемся от мультииндексов.
desс_pr_df.index.name = ''                                                       # Просто так красивее)
desс_pr_df
sales_df.groupby(['state_id','cat_id']).sum().T.agg(['mean', 'std']).T
desс_pr_df.head()
plt.rc('figure', figsize=(20,15))       # Устанавливаем размеры графиков.
font = {'family' : 'monospace'}
plt.rc('font', **font)
names = ['Понедельный спрос на товар.(+1 если товар был продан на недели, +0 иначе)', 'Средняя цена на товар.',
         'Стандартное отклонение от среднего.', 'Минимальная цена на товар.',
         '0.25 квантиль распределения цены.', 'Медианная цена на товар.',
         '0.75 квантиль распределения цены.', 'Максимальная цена на товар.']
plt.rc('figure', figsize=(20,15))
fig, axes = plt.subplots(4, 2)

for id_ in range(8):
  row_id = int(id_ / 2)
  col_id = id_ % 2
  axes[row_id,col_id].hist(desс_pr_df.iloc[:,id_].to_list(), bins=25, color='b', alpha=0.7);
  axes[row_id,col_id].set_title(names[id_]) 
  axes[row_id,col_id].set_xlabel('Цена в доллар/товар')
  axes[row_id,col_id].set_ylabel('Кол-во')
  axes[row_id,col_id].grid()
axes[1,0].set_xlabel('Условные единицы')

plt.subplots_adjust(wspace=0.3, hspace=0.5) # Задаем пустое пространство между графиками. 

plt.show()
plt.close()
plt.rc('figure', figsize=(30,10))
fig, axes = plt.subplots(2, 4)
bin_list = [30,100,300,600]

for id_ in range(4):
  row_id = int(id_ / 2)
  col_id = id_ % 2
  axes[row_id,col_id].hist(desс_pr_df['std'].to_list(), bins=bin_list[id_], color='r', alpha=1);
  axes[row_id,col_id].set_title('Стандартное отклонение от среднего.') 
  axes[row_id,col_id].set_xlabel('Условные единицы')
  axes[row_id,col_id].set_ylabel('Кол-во')


axes[0,2].scatter(desс_pr_df['mean'].to_list(), desс_pr_df['25%'].to_list(), color='b', alpha=1)
axes[0,2].set_title('Корреляция среднего и 0.25 квантиля.')
axes[0,2].set_xlabel('mean')
axes[0,2].set_ylabel('25%')

axes[0,3].scatter(desс_pr_df['mean'].to_list(), desс_pr_df['50%'].to_list(), color='b', alpha=1)
axes[0,3].set_title('Корреляция среднего и медианы.')
axes[0,3].set_xlabel('mean')
axes[0,3].set_ylabel('median')

axes[1,2].scatter(desс_pr_df['mean'].to_list(), desс_pr_df['75%'].to_list(), color='b', alpha=1)
axes[1,2].set_title('Корреляция среднего и 0.75 квантиля.')
axes[1,2].set_xlabel('mean')
axes[1,2].set_ylabel('75%')

axes[1,3].scatter(desс_pr_df['mean'].to_list(), desс_pr_df['max'].to_list(), color='b', alpha=1)
axes[1,3].set_title('Корреляция среднего и максимума.')
axes[1,3].set_xlabel('mean')
axes[1,3].set_ylabel('max')

plt.subplots_adjust(wspace=0.2, hspace=0.2) # Задаем пустое пространство между графиками.

plt.show()
plt.close()
plt.rc('figure', figsize=(30,10))
fig, axes = plt.subplots(3,1)
names = ['FOODS', 'HOBBIES', 'HOUSEHOLD']

for id_ in range(3):
  tmp = desс_pr_df.loc[[names[id_] in word for word in desс_pr_df.reset_index().iloc[:,0].to_list()]]['std']
  axes[id_].hist(tmp, bins=300, color='r', alpha=0.7, range=(0,5), label=names[id_]);
  axes[id_].set_title('Стандартное отклонение ' + names[id_] + ' от среднего.') 
  axes[id_].set_xlabel('Условные единицы')
  axes[id_].set_ylabel('Кол-во')
  axes[id_].grid()
plt.subplots_adjust(hspace=0.5)

plt.show()
plt.close()
plt.rc('figure', figsize=(30,7))
fig, axes = plt.subplots(1,1)
names = ['FOODS', 'HOUSEHOLD', 'HOBBIES']
colors = ['r','g','b']

for id_ in range(3):
  tmp = desс_pr_df.loc[[names[id_] in word for word in desс_pr_df.reset_index().iloc[:,0].to_list()]]['std']
  axes.hist(tmp, bins=75, color=colors[id_], alpha=0.7, range=(0,2), label=names[id_]);

axes.legend()
axes.set_title('Стандартное отклонение от среднего.') 
axes.set_xlabel('Условные единицы')
axes.set_ylabel('Кол-во')
axes.grid()
plt.subplots_adjust(hspace=0.4)

plt.show()
plt.close()
fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(111)

sns.set(style="ticks", color_codes=True)
sns.heatmap(ax=ax1, data=desс_pr_df.corr(), cmap='Blues')
# pair_corr = sns.pairplot(data=desс_pr_df, diag_kind='kde') # Как еще один вариант исследования корреляции.
# pair_corr.fig.set_size_inches(7,7) 

plt.show()
plt.close()
# Для поиска средних товаров, фильтруя по названиям.

def mean_time_series(name_1, name_2=''):

  if not name_2:
    name_2 = name_1
  
  com = np.zeros(main_df['_1'].to_numpy().shape)

  rand_num = [name_ for name_ in id_dict.keys() if name_1 in name_ and name_2 in name_]
  rand_num = [int(id_dict[id_][1:]) for id_ in rand_num if id_ in id_dict.keys()]

  for id_ in range(len(rand_num)):
    com += main_df['_'+str(rand_num[id_])].to_numpy()
  com /= len(rand_num)

  return com
type_names = ['HOBBIES_1','HOBBIES_2',
              'FOODS_1','FOODS_2','FOODS_3', 
              'HOUSEHOLD_1','HOUSEHOLD_2']

state_names = ['CA', 'TX', 'WI']     
               
data_series_types  = {type_name:  mean_time_series(type_name)  for type_name  in type_names}
data_series_states = {state_name: mean_time_series(state_name) for state_name in state_names}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime as dt
sns.set(style='whitegrid', rc={'figure.facecolor': 'snow', 'axes.facecolor': 'gainsboro'})
# plt.style.use('ggplot')  # Красивые графики

fig, axes = plt.subplots(len(data_series_types),2, figsize=(20,15))
fig.suptitle('Временные ряды / покупки товаров по субкатегориям:',
             fontsize = 25,            
             fontfamily = 'monospace',
             fontstyle  = 'oblique')

for id_y, type_name in enumerate(type_names):

  sns.kdeplot(ax=axes[id_y,0],           data=data_series_types[type_name], shade=True, cut=0, bw=.0, label="bw: 0.0")
  graphic = sns.kdeplot(ax=axes[id_y,0], data=data_series_types[type_name], shade=True, cut=0, bw=.1, label="bw: 0.1")
  graphic.set_xlim(0, 5)
  graphic.set_ylim(0, 6)
  axes[id_y,0].set_title(type_name,
                         loc='left',
                         fontsize=12,
                         fontfamily = 'serif',
                         fontstyle  = 'italic',
                         pad=3)
  
  time_interval = [dt.datetime.strptime(i, "%Y-%m-%d") for i in main_df.index] # Получаем интервалы времени в нужном виде.

  graphic = sns.lineplot(ax=axes[id_y,1], x=time_interval, y=data_series_types[type_name], color='r') # Рисуем плотность распределения.
  graphic.set_ylim(0, 3)
  axes[id_y,1].set_title(type_name,
                         loc='right',
                         fontsize=12,
                         fontfamily = 'serif',
                         fontstyle  = 'italic',
                         pad=3)    

plt.subplots_adjust(wspace=0.2, hspace=0.7)

plt.show()
plt.close()
plt.style.use('bmh')  # Красивые графики

fig, axes = plt.subplots(len(data_series_states),2, figsize=(21,9))
fig.suptitle('Временные ряды / покупки товаров по штатам:',
             fontsize = 20,            
             fontfamily = 'monospace',
             fontstyle  = 'oblique')

for id_, state_name in enumerate(state_names):

  sns.kdeplot(ax=axes[id_,0],            data=data_series_states[state_name], shade=True, cut=0, bw=.00, label="bw: 0.0")  
  graphic = sns.kdeplot(ax=axes[id_,0],  data=data_series_states[state_name], shade=True, cut=0, bw=.05, label="bw: 0.05")  
  graphic.set_xlim(0, 3)
  graphic.set_ylim(0, 3)
  axes[id_,0].set_title(state_name,
                        loc='left',
                        fontsize=12,
                        fontfamily = 'serif',
                        fontstyle  = 'italic',
                        pad=3)
  
  time_interval = [dt.datetime.strptime(i, "%Y-%m-%d") for i in main_df.index] # Получаем интервалы времени в нужном виде.  
  
  graphic = sns.lineplot(ax=axes[id_,1], x=time_interval, y=data_series_states[state_name], color='r')
  axes[id_,1].set_title(state_name,
                        loc='right',
                        fontsize=12,
                        fontfamily = 'serif',
                        fontstyle  = 'italic',
                        pad=3)    
      
plt.subplots_adjust(wspace=0.4, hspace=0.3)

plt.show()
plt.close()
plt.style.use('seaborn-white')
fig = plt.figure(figsize=(22,6));
ax = fig.add_subplot(111)
diagr = subcat_df.drop(['snap_CA','snap_TX','snap_WI'], axis=1).groupby('event_type_1').mean().plot.bar(ax=ax,stacked=True);

diagr.set_title('Столб. диаграмма ср. покупок по субкатегориям сгруппированная по событиям.',
             fontsize=20,
             fontfamily = 'serif',
             fontstyle  = 'italic',
             pad=15);

diagr.set_ylim(0, 7)
plt.grid(color='k',
         linestyle='dotted',
         linewidth=1.5)
plt.style.use('seaborn-white')
fig = plt.figure(figsize=(22,6));
ax = fig.add_subplot(111)
diagr = subcat_df.loc[:,['event_type_1','snap_CA','snap_TX','snap_WI']].groupby('event_type_1').mean().plot.bar(ax=ax);

diagr.set_title('Столб. диаграмма ср. SNAP выплат сгруппированная по событиям.',
             fontsize=20,
             fontfamily = 'serif',
             fontstyle  = 'italic',
             pad=15);

diagr.set_ylim(0, 1)
plt.grid(color='k',
         linestyle='dotted',
         linewidth=1.5)
foods_state = {'CA': mean_time_series('FOODS_3', 'CA'),
               'TX': mean_time_series('FOODS_3', 'TX'),
               'WI': mean_time_series('FOODS_3', 'WI')}
# Убедимся, что наши данные имеют нормальное распределение, воспользовавшись методом Q-Q.

fig = plt.figure(figsize=(30,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
accumulate_data = mean_time_series('FOODS_3')
ax1.hist(accumulate_data, bins=50);
ax2 = stats.probplot(accumulate_data, dist="norm", plot=plt);

ax1.grid()
plt.grid()

plt.show()
plt.close()
plt.hist(foods_state['TX'], bins=50, alpha=0.7); 
plt.hist(foods_state['CA'], bins=50, alpha=0.7); 
plt.hist(foods_state['WI'], bins=50, alpha=0.7); 
stats.shapiro(foods_state['TX'])
plt.grid()
print('Стандартная ошибка среднего по FOODS_3 в штате CA: %.4f' % stats.sem(foods_state['CA']))
print('Стандартная ошибка среднего по FOODS_3 в штате TX: %.4f' % stats.sem(foods_state['TX']))
print('Стандартная ошибка среднего по FOODS_3 в штате WI: %.4f' % stats.sem(foods_state['WI']))
print('Выборочное стандартное отклонение по FOODS_3 в штате CA: %.4f' % np.std(foods_state['CA']))
print('Выборочное стандартное отклонение по FOODS_3 в штате TX: %.4f' % np.std(foods_state['TX']))
print('Выборочное стандартное отклонение по FOODS_3 в штате WI: %.4f' % np.std(foods_state['WI']))
numpyed_dict = np.array([list(arr) for arr in foods_state.values()])
x = np.mean(numpyed_dict, axis=1)
y = np.arange(len(['CA','TX','WI']))
e = np.std(numpyed_dict, axis=1)

fig,ax = plt.subplots(0,0,figsize=(10,5))
ax = plt.errorbar(x, y, xerr=e, linestyle='None', marker='o');
plt.grid()
F, p = stats.f_oneway(foods_state['CA'], foods_state['WI'], foods_state['TX'])

print('Вероятность ошибочну отвергнуть верную гипотезу = %.3f, при уровне значимости p = 0.05.\n' % p)
print("порядок p:", p)
numpyed_dict = np.array([list(arr) for arr in foods_state.values()])
x = np.mean(numpyed_dict, axis=1)
y = np.arange(len(['CA','TX','WI']))
e = np.std(numpyed_dict, axis=1)


fig,ax = plt.subplots(0,0,figsize=(10,5))
ax = plt.errorbar(x, y, xerr=e, linestyle='None', marker='o')
ax = plt.errorbar(np.mean(x), y.shape[0], xerr=np.std(x), linestyle='None', marker='o');
plt.grid()
plt.show()
plt.close()
_, p_CA_WI = stats.ttest_ind(foods_state['CA'], foods_state['WI']) # FOODS_3 товары покупают в штатах CA и WI одинаково. CA!=WI
_, p_WI_TX = stats.ttest_ind(foods_state['WI'], foods_state['TX']) # FOODS_3 товары покупают в штатах WI и TX одинаково. WI!=TX
_, p_TX_CA = stats.ttest_ind(foods_state['TX'], foods_state['CA']) # FOODS_3 товары покупают в штатах TX и CA одинаково. TX!=CA

print("Вероятность ошибочно отвергнуть гипотезу СA!=WI:  %.3f" % p_CA_WI)
print("Вероятность ошибочно отвергнуть гипотезу WI!=TX:  %.3f" % p_WI_TX)
print("Вероятность ошибочно отвергнуть гипотезу TX!=CA:  %.3f" % p_TX_CA)
_, p_CA = stats.ttest_1samp(a = foods_state['CA'], popmean=2.0)
_, p_TX = stats.ttest_1samp(a = foods_state['TX'], popmean=2.0)
_, p_WI = stats.ttest_1samp(a = foods_state['WI'], popmean=2.0)

print("Вероятность ошибочно отвергнуть гипотезу H0 об CA.mean = 2:  %.20f, с уровнем значимости p = 0.05" % p_CA)
print("Вероятность ошибочно отвергнуть гипотезу H0 об TX.mean = 2:  %.20f, с уровнем значимости p = 0.05" % p_TX)
print("Вероятность ошибочно отвергнуть гипотезу H0 об WI.mean = 2:  %.20f, с уровнем значимости p = 0.05" % p_WI)
price_sales_df.describe().T
plt.rc('figure', figsize=(20,6))       

mean_price = price_sales_df.sell_price.mean()
print('Средняя цена на товары, вне зависимости от категорий: %.2f$' % mean_price)

below_mean = price_sales_df.query('sell_price <  @mean_price').mean_sales.to_numpy()
above_mean = price_sales_df.query('sell_price >= @mean_price').mean_sales.to_numpy()

plt.hist(below_mean, bins=100, range=(0,20), alpha=0.8, label = '> mean');
plt.hist(above_mean, bins=100, range=(0,20), alpha=0.8, label = '< mean');

plt.legend()
plt.grid()
plt.show()
plt.close()
stats.chi2_contingency(below_mean, above_mean)
x = [np.mean(below_mean), np.mean(above_mean)]
y = np.arange(len(x))
e = [np.std(below_mean), np.std(above_mean)]

fig,ax = plt.subplots(0,0,figsize=(8,3))
ax = plt.errorbar(x, y, xerr=e, linestyle='None', marker='o')
ax = plt.errorbar(np.mean(x), y.shape[0], xerr=np.std(x), linestyle='None', marker='o');
plt.grid()
plt.show()
plt.close()

print('Среднее кол-во покупок товаров, цены которых ниже средней: %.2f' % x[0])
print('Среднее кол-во покупок товаров, цены которых выше средней: %.2f\n' % x[1])
print('Стандартная ошибка среднего спроса дешевых товаров: %.4f' % stats.sem(below_mean))
print('Стандартная ошибка среднего спроса дорогих товаров: %.4f' % stats.sem(above_mean))
below_mean_samples = []
above_mean_samples = []

for _ in range(20000):
  bootstrap_sample = np.random.choice(below_mean, below_mean.shape[0])
  below_mean_samples.append(np.mean(bootstrap_sample))
  
  bootstrap_sample = np.random.choice(above_mean, above_mean.shape[0])
  above_mean_samples.append(np.mean(bootstrap_sample))
plt.rc('figure', figsize=(20,3))    
plt.hist(above_mean_samples, bins=100, alpha=0.9, label='> mean price');
plt.hist(below_mean_samples, bins=100, alpha=0.9, label='< mean price');
plt.legend();
plt.grid()
plt.show()
plt.close()
import statsmodels.stats.api as sms #  Для доверительных интервалов.
cm = sms.CompareMeans(sms.DescrStatsW(below_mean_samples),
                      sms.DescrStatsW(above_mean_samples))
print(cm.tconfint_diff(alpha=0.05)) # Указываем явно уровень значимости p = 0.05
state_name = 'TX'
FOODS_1 = ['event_type_1', 'snap_'+state_name] + [id_dict[col] for col in id_dict.keys() if 'FOODS_1' in col and state_name in col]
FOODS_2 = ['event_type_1', 'snap_'+state_name] + [id_dict[col] for col in id_dict.keys() if 'FOODS_2' in col and state_name in col]
FOODS_3 = ['event_type_1', 'snap_'+state_name] + [id_dict[col] for col in id_dict.keys() if 'FOODS_3' in col and state_name in col]
fig = plt.figure(figsize=(30,5))

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

main_df.loc[:,FOODS_1].groupby('snap_'+state_name).mean().T.iloc[:,0].hist(bins=50, range=(0,20), alpha=0.5, label='without snap', ax=ax1)
main_df.loc[:,FOODS_1].groupby('snap_'+state_name).mean().T.iloc[:,1].hist(bins=50, range=(0,20), alpha=0.5, label='with snap', ax=ax1)
ax1.set_title('FOODS_1')

main_df.loc[:,FOODS_2].groupby('snap_'+state_name).mean().T.iloc[:,0].hist(bins=50, range=(0,20), alpha=0.5, label='without snap', ax=ax2)
main_df.loc[:,FOODS_2].groupby('snap_'+state_name).mean().T.iloc[:,1].hist(bins=50, range=(0,20), alpha=0.5, label='with snap', ax=ax2)
ax2.set_title('FOODS_2')

main_df.loc[:,FOODS_3].groupby('snap_'+state_name).mean().T.iloc[:,0].hist(bins=50, range=(0,20), alpha=0.5, label='without snap', ax=ax3)
main_df.loc[:,FOODS_3].groupby('snap_'+state_name).mean().T.iloc[:,1].hist(bins=50, range=(0,20), alpha=0.5, label='with snap', ax=ax3)
ax3.set_title('FOODS_3')

plt.show()
plt.close()
without_snap_1 = main_df.loc[:,FOODS_1].groupby('snap_'+state_name).mean().T.iloc[:,0].to_numpy()
with_snap_1    = main_df.loc[:,FOODS_1].groupby('snap_'+state_name).mean().T.iloc[:,1].to_numpy()

without_snap_2 = main_df.loc[:,FOODS_2].groupby('snap_'+state_name).mean().T.iloc[:,0].to_numpy()
with_snap_2    = main_df.loc[:,FOODS_2].groupby('snap_'+state_name).mean().T.iloc[:,1].to_numpy()

without_snap_3 = main_df.loc[:,FOODS_3].groupby('snap_'+state_name).mean().T.iloc[:,0].to_numpy()
with_snap_3    = main_df.loc[:,FOODS_3].groupby('snap_'+state_name).mean().T.iloc[:,1].to_numpy()
# Двухвыборочный критерий знаков для связных выборок.

from statsmodels.stats.descriptivestats import sign_test

_, p_FOODS_1 = sign_test(without_snap_1, np.median(with_snap_1))
_, p_FOODS_2 = sign_test(without_snap_2, np.median(with_snap_2))
_, p_FOODS_3 = sign_test(without_snap_3, np.median(with_snap_3))

print('Вероятность ошибочно отвергнуть верную гипотезу HO для FOODS_1 равна %.2f, с уровнем значимости p = 0.05'   % p_FOODS_1)
print('Вероятность ошибочно отвергнуть верную гипотезу HO для FOODS_2 равна %.2f, с уровнем значимости p = 0.05'   % p_FOODS_2)
print('Вероятность ошибочно отвергнуть верную гипотезу HO для FOODS_3 равна %.2f, с уровнем значимости p = 0.05\n' % p_FOODS_3)

print('p_FOODS_1 = ', p_FOODS_1)
print('p_FOODS_2 = ', p_FOODS_2)
print('p_FOODS_3 = ', p_FOODS_3)
# Двухвыборочный рагновый критерий для связных выборок.

_, p_FOODS_1 = stats.wilcoxon(without_snap_1, with_snap_1)
_, p_FOODS_2 = stats.wilcoxon(without_snap_2, with_snap_2)
_, p_FOODS_3 = stats.wilcoxon(without_snap_3, with_snap_3)

print('Вероятность ошибочно отвергнуть верную гипотезу HO для FOODS_1 равна %.2f, с уровнем значимости p = 0.05'   % p_FOODS_1)
print('Вероятность ошибочно отвергнуть верную гипотезу HO для FOODS_2 равна %.2f, с уровнем значимости p = 0.05'   % p_FOODS_2)
print('Вероятность ошибочно отвергнуть верную гипотезу HO для FOODS_3 равна %.2f, с уровнем значимости p = 0.05\n' % p_FOODS_3)

print('p_FOODS_1 = ', p_FOODS_1)
print('p_FOODS_2 = ', p_FOODS_2)
print('p_FOODS_3 = ', p_FOODS_3)
without_snap_1_samples = []
with_snap_1_samples    = []

without_snap_2_samples = []
with_snap_2_samples    = []

without_snap_3_samples = []
with_snap_3_samples    = []

for _ in range(20000):
  bootstrap_sample = np.random.choice(without_snap_1, without_snap_1.shape[0])
  without_snap_1_samples.append(np.mean(bootstrap_sample))
  
  bootstrap_sample = np.random.choice(with_snap_1, with_snap_1.shape[0])
  with_snap_1_samples.append(np.mean(bootstrap_sample))

  bootstrap_sample = np.random.choice(without_snap_2, without_snap_2.shape[0])
  without_snap_2_samples.append(np.mean(bootstrap_sample))
  
  bootstrap_sample = np.random.choice(with_snap_2, with_snap_2.shape[0])
  with_snap_2_samples.append(np.mean(bootstrap_sample))

  bootstrap_sample = np.random.choice(without_snap_3, without_snap_3.shape[0])
  without_snap_3_samples.append(np.mean(bootstrap_sample))
  
  bootstrap_sample = np.random.choice(with_snap_3, with_snap_3.shape[0])
  with_snap_3_samples.append(np.mean(bootstrap_sample))
fig = plt.figure(figsize=(30,5))    
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.hist(without_snap_1_samples, bins=400, alpha=0.8, label='days without SNAP', range=(0.7,3.7));
ax1.hist(with_snap_1_samples,    bins=400, alpha=0.8, label='days with SNAP',    range=(0.7,3.7));
ax1.grid()
ax1.legend()
ax1.set_title('FOODS_1')

ax2.hist(without_snap_2_samples, bins=400, alpha=0.8, label='days without SNAP', range=(0.7,3.7));
ax2.hist(with_snap_2_samples,    bins=400, alpha=0.8, label='days with SNAP',    range=(0.7,3.7));
ax2.grid()
ax2.legend()
ax2.set_title('FOODS_2')

ax3.hist(without_snap_3_samples, bins=400, alpha=0.8, label='days without SNAP', range=(0.7,3.7));
ax3.hist(with_snap_3_samples,    bins=400, alpha=0.8, label='days with SNAP',   range=(0.7,3.7));
ax3.grid()
ax3.legend()
ax3.set_title('FOODS_3')

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()
plt.close()
cm2 = sms.CompareMeans(sms.DescrStatsW(with_snap_2_samples),
                       sms.DescrStatsW(without_snap_2_samples))

cm3 = sms.CompareMeans(sms.DescrStatsW(with_snap_3_samples),
                       sms.DescrStatsW(without_snap_3_samples))

print("Доверительные интервалы для изменения средних FOODS_2: (%.3f, %.3f)" % cm2.tconfint_diff(alpha=0.05))
print("Среднии для FOODS_2: %.2f и %.2f без SNAP и с SNAP соответственно.\n" % (np.mean(without_snap_2_samples), np.mean(with_snap_2_samples)))

print("Доверительные интервалы для изменения средних FOODS_3: (%.3f, %.3f)" % cm3.tconfint_diff(alpha=0.05))
print("Среднии для FOODS_3: %.2f и %.2f без SNAP и с SNAP соответственно.\n" % (np.mean(without_snap_3_samples), np.mean(with_snap_3_samples)))

print('Средний спрос на товары субкатегории FOODS_1: %.2f' % mean_time_series('FOODS_1', state_name).mean())
for FOODS_id in ['FOODS_1', 'FOODS_2', 'FOODS_3']:
  mean_price = price_sales_df.loc[list(filter(lambda x: x[:7] == FOODS_id, list(price_sales_df.reset_index().item_id.to_numpy())))]['sell_price'].mean()
  print('Средняя цена на товары субкатегории', FOODS_id, ': %.2f$' % mean_price)
state_name = 'TX' # Чтобы исследования были корректны, рассмотрим штат Техас.
FOODS_1 = [id_dict[col] for col in id_dict.keys() if 'FOODS_1' in col and state_name in col]
tmp_df = main_df.loc[:,['event_type_1']]
tmp_df['mean_sales'] = main_df.loc[:,FOODS_1].mean(axis=1)
tmp_df = tmp_df.fillna('Ordinary')
tmp_df.groupby('event_type_1').mean()
from itertools import combinations

corr_dict = {}

for subcat_names in list(combinations(subcat_df.columns[4:], 2)):
  cur_corr = stats.spearmanr(subcat_df[subcat_names[0]], subcat_df[subcat_names[-1]])
  corr_dict.update({cur_corr[0]: [subcat_names[0], subcat_names[1]]})
for idx, pair in enumerate(sorted(corr_dict.items(), key=lambda x: x[0], reverse=True)[:5]):
  print('Топ %.f:\t' % idx, pair[1][0], '\t', pair[1][1], '\tcorr: %.4f' % pair[0])
for idx, pair in enumerate(sorted(corr_dict.items(), key=lambda x: x[0])[:5]):
  print('Топ -%.f:\t' % (idx+1), pair[1][0], '\t', pair[1][1], '\tcorr: %.4f' % pair[0])
fig, axes = plt.subplots(3,2, figsize=(30,10))

subcat_df.plot.scatter(ax=axes[0,0], x='HOUSEHOLD_1', y='HOUSEHOLD_2')
subcat_df.plot.scatter(ax=axes[1,0], x='HOBBIES_1',   y='HOUSEHOLD_2')
subcat_df.plot.scatter(ax=axes[2,0], x='FOODS_1',     y='HOUSEHOLD_1')

subcat_df.plot.scatter(ax=axes[0,1], x='FOODS_3',     y='HOBBIES_2')
subcat_df.plot.scatter(ax=axes[1,1], x='HOBBIES_1',   y='HOBBIES_2')
subcat_df.plot.scatter(ax=axes[2,1], x='FOODS_1',     y='HOBBIES_2')

plt.subplots_adjust(wspace=0.4, hspace=0.5)

plt.show()
plt.close()
tmp_df = sales_df.drop(['id','dept_id','cat_id','store_id', 'state_id'], axis=1).groupby('item_id').mean().T
tmp_df
corr_dict = {}

items = list(combinations(tmp_df.columns, 2))

for id_ in tqdm(np.random.randint(0, len(items), 100000)):

  item_names = items[id_]
  cur_corr = stats.spearmanr(tmp_df[item_names[0]], tmp_df[item_names[-1]])
  corr_dict.update({cur_corr[0]: [item_names[0], item_names[1]]})
sorted_dict = sorted(corr_dict.items(), key=lambda x: x[0], reverse=True)
for idx, pair in enumerate(sorted_dict[:5]):
  print('Топ %.f:\t' % idx, pair[1][0], '\t', pair[1][1], '\tcorr: %.4f' % pair[0])
plus_one = [sorted_dict[0][1][0], sorted_dict[0][1][1]]
sorted_dict = sorted(corr_dict.items(), key=lambda x: x[0])
for idx, pair in enumerate(sorted_dict[:5]):
  print('Топ %.f:\t' % idx, pair[1][0], '\t', pair[1][1], '\tcorr: %.4f' % pair[0])
minus_one = [sorted_dict[0][1][0], sorted_dict[0][1][1]]
sorted_dict = sorted(corr_dict.items(), key=lambda x: x[0]**2)
for idx, pair in enumerate(sorted_dict[:5]):
  print('Топ %.f:\t' % idx, pair[1][0], '\t', pair[1][1], '\tcorr: %.4f' % pair[0])
zero_one = [sorted_dict[0][1][0], sorted_dict[0][1][1]]
fig = plt.figure(figsize=(28,6))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.scatter(x=tmp_df[plus_one[0]], y=tmp_df[plus_one[1]])
ax1.set_xlabel(plus_one[0])
ax1.set_ylabel(plus_one[1])
ax1.set_title('corr  ~  +1')

ax2.scatter(x=tmp_df[minus_one[0]], y=tmp_df[minus_one[1]])
ax2.set_xlabel(minus_one[0])
ax2.set_ylabel(minus_one[1])
ax2.set_title('corr  ~  0')

ax3.scatter(x=tmp_df[zero_one[0]], y=tmp_df[zero_one[1]])
ax3.set_xlabel(zero_one[0])
ax3.set_ylabel(zero_one[1])
ax3.set_title('corr  ~  -1')

plt.show()
plt.close()
subcat_df.loc[:,['snap_CA','snap_TX','snap_WI']]
tmp = subcat_df.loc[:,['snap_CA','snap_TX','snap_WI']].to_numpy()
snap_dict = {'CA': tmp[:,0], 'TX': tmp[:,1], 'WI': tmp[:,2]}
from sklearn.metrics import matthews_corrcoef

corr_CA_TX = matthews_corrcoef(snap_dict['CA'], snap_dict['TX'])
corr_TX_WI = matthews_corrcoef(snap_dict['WI'], snap_dict['TX'])
corr_WI_CA = matthews_corrcoef(snap_dict['CA'], snap_dict['WI'])
print('Уровень корреляции между выплатами SNAP в Калифорнии и Техасе     =\t%.4f' % corr_CA_TX)
print('Уровень корреляции между выплатами SNAP в Техасе и Висконсине     =\t%.4f' % corr_TX_WI)
print('Уровень корреляции между выплатами SNAP в Висконсине и Калифорнии =\t%.4f' % corr_WI_CA)
print(stats.chi2_contingency(snap_dict['CA']+1, snap_dict['TX']+1))
print(stats.chi2_contingency(snap_dict['WI']+1, snap_dict['TX']+1))
print(stats.chi2_contingency(snap_dict['CA']+1, snap_dict['WI']+1))
# Посмотрим на наш очень сырой, относительно работы с временными рядами, датасет.
main_df.head(2)
# Уберем не нужные ранговые стобцы.
date_df = main_df.drop(['event_type_1', 'snap_CA', 'snap_TX', 'snap_WI'], axis=1).reset_index()
date_df.head(2)
date_df['Datetime'] = pd.to_datetime(date_df.date, format='%Y-%m-%d %H:%M:%S')
date_df.index  = date_df.Datetime
date_df.drop(['date', 'Datetime'], axis = 1, inplace = True)
date_df.head(2)
id_='_1'
date_df[id_].plot(figsize=(15,5), title=name_id['_0'], fontsize=14);
fig, axes = plt.subplots(2, 1, sharey=False, sharex=False)
fig.set_figwidth(14)
fig.set_figheight(8)

window = 30
date_df[id_].plot(ax=axes[0], label='Original')
date_df[id_].rolling(window=window).mean().plot(ax=axes[0], label='1-Month Rolling Mean')
axes[0].set_title("1-Month Rolling Mean")
axes[0].legend(loc='best')

window = 30*6
date_df[id_].plot(ax=axes[1], label='Original')
date_df[id_].rolling(window=window).mean().plot(ax=axes[1], label='6-Month Rolling Mean')
axes[1].set_title("6-Month Rolling Mean")
axes[1].legend(loc='best')

plt.tight_layout()
plt.show()
plt.rcParams['figure.figsize'] = 20, 5
date_df['year']  = date_df.index.year
date_df['month'] = date_df.index.month
date_df['dow']   = date_df.index.dayofweek
date_df['day']   = date_df.index.day
date_df_pivot = pd.pivot_table(date_df.loc[:,[id_,'year', 'month','dow','day']], values = id_, columns = "year", index = "month")
date_df.loc[:,[id_,'year', 'month','dow', 'day']].head(2)
date_df_pivot.head(2)
date_df_pivot.plot();
date_df.drop(['year','month','dow','day'], axis=1, inplace=True)
plt.rcParams['figure.figsize'] = 12, 7
result = sm.tsa.seasonal_decompose(date_df[id_], model='additive', freq=365)
result.plot()
plt.show()
plt.rcParams['figure.figsize'] = 12, 7
date_df[id_]+=1
result = sm.tsa.seasonal_decompose(date_df[id_]+1, model='mul', freq= 365)
result.plot()
plt.show()
plt.figure(figsize=(16,2))
date_df[id_] += 1
plt.plot(date_df[id_])
plt.show()
plt.figure(figsize=(16,2))

MA = date_df[id_].rolling(window=365).mean()

plt.plot(MA)
plt.show()
detrend_ = date_df[id_] / MA
plt.figure(figsize=(16,2))

plt.plot(detrend_)
plt.show()
def seasonal_mean(x, period):
    """
    Return means for each period in x. period is an int that gives the
    number of periods per cycle. E.g., 12 for monthly. NaNs are ignored
    in the mean.
    """
    return np.array([pd_nanmean(x[i::period], axis=0) for i in range(period)])
s_a = seasonal_mean(detrend_, 365)
#s_a -= np.mean(s_a, axis=0)

nobs = detrend_.shape[0]
seasonal = np.tile(s_a.T, nobs // 365 + 1).T[:nobs]
plt.figure(figsize=(16,2))

plt.plot(seasonal)
plt.show()
residuals = date_df[id_] / (MA * seasonal)
plt.figure(figsize=(16,2))

plt.plot(residuals)
plt.show()
w_years = 365
train = date_df.loc[:,[id_]].iloc[:-w_years]
val   = date_df.loc[:,[id_]].iloc[-w_years:]
plt.figure(figsize=(16,2))

plt.plot(train[id_])
plt.plot(val[id_])

plt.show()
preds = [train[id_].mean()] * w_years
# Метрики качества:

from sklearn.metrics import mean_squared_error
from math import sqrt
    
def rmse(y_actual, y_predicted):
    rmse = sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse

# mean_absolute_percentage_error    
def mape(y_true, y_pred): 
    y_true = np.asarray(y_true)+1
    y_pred = np.asarray(y_pred)+1
    return np.mean(np.abs((y_true - y_pred) /y_true)) * 100

# Root Mean Squared Scaled Error
def rmsse(y_train, y_true, y_pred):

  y_train, y_true, y_pred = np.asarray(y_train), np.asarray(y_true), np.asarray(y_pred)

  above = np.mean((y_true-y_pred)**2)
  below = ((y_train - np.roll(y_train, 1))[1:]**2).mean() + 1e-10

  return sqrt(above/below)
print('RMSE  - ', rmse(val[id_], preds))
print('MAPE  - ', mape(val[id_], preds))
print('RMSSE - ', rmsse(train[id_], val[id_], preds))
print('RMSE - ', rmse(val[id_], np.around(preds)))
print('MAPE - ', mape(val[id_], np.around(preds)))
print('RMSSE - ', rmsse(train[id_], val[id_], np.around(preds)))
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), preds)), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), np.around(preds))), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
preds = train.iloc[-w_years:][id_].values
print('RMSE  - ', rmse(val[id_], preds))
print('MAPE  - ', mape(val[id_], preds))
print('RMSSE - ', rmsse(train[id_], val[id_], preds))
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), preds)), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
#Оцениваем по последним N годам.
n = 4
tr_sample = train.iloc[-n*w_years:]
preds_arr = np.array(tr_sample[id_]).reshape(n,-1)

preds = preds_arr.mean(axis = 0)
print('RMSE  - ', rmse(val[id_], preds))
print('MAPE  - ', mape(val[id_], preds))
print('RMSSE - ', rmsse(train[id_], val[id_], preds))
print('RMSE  - ', rmse(val[id_], np.around(preds)))
print('MAPE  - ', mape(val[id_], np.around(preds)))
print('RMSSE - ', rmsse(train[id_], val[id_], np.around(preds)))
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), preds)), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), np.around(preds))), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
W = np.array([0.1, 0.2, 0.3, 0.4])
preds_arr_W = np.array([preds_arr[x]*W[x] for x in range(len(W))])
preds = preds_arr_W.sum(axis = 0)
print('RMSE - ', rmse(val[id_], preds))
print('MAPE - ', mape(val[id_], preds))
print('RMSSE - ', rmsse(train[id_], val[id_], preds))
print('RMSE  - ', rmse(val[id_], np.around(preds)))
print('MAPE  - ', mape(val[id_], np.around(preds)))
print('RMSSE - ', rmsse(train[id_], val[id_], np.around(preds)))
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), preds)), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), np.around(preds))), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
model = SimpleExpSmoothing(train).fit(smoothing_level=0.7, optimized=False)
preds = model.forecast(val.shape[0])
print('RMSE - ', rmse(val[id_], preds))
print('MAPE - ', mape(val[id_], preds))
print('RMSSE - ', rmsse(train[id_], val[id_], preds))
print('RMSE  - ', rmse(val[id_], np.around(preds)))
print('MAPE  - ', mape(val[id_], np.around(preds)))
print('RMSSE - ', rmsse(train[id_], val[id_], np.around(preds)))
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), preds)), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), np.around(preds))), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
model = Holt(train).fit(smoothing_level = 0.4, smoothing_slope = 0.2)
preds = model.forecast(val.shape[0])
print('RMSE - ', rmse(val[id_], preds))
print('MAPE - ', mape(val[id_], preds))
print('RMSSE - ', rmsse(train[id_], val[id_], preds))
print('RMSE  - ', rmse(val[id_], np.around(preds)))
print('MAPE  - ', mape(val[id_], np.around(preds)))
print('RMSSE - ', rmsse(train[id_], val[id_], np.around(preds)))
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), preds)), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), np.around(preds))), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
model = ExponentialSmoothing(train[id_],
                             seasonal_periods=365,
                             trend=None,
                             seasonal='add').fit()

preds = model.forecast(val.shape[0])
# forecast.index = val.index
print('RMSE - ', rmse(val[id_], preds))
print('MAPE - ', mape(val[id_], preds))
print('RMSSE - ', rmsse(train[id_], val[id_], preds))
print('RMSE  - ', rmse(val[id_], np.around(preds)))
print('MAPE  - ', mape(val[id_], np.around(preds)))
print('RMSSE - ', rmsse(train[id_], val[id_], np.around(preds)))
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), preds)), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
plt.figure(figsize=(16,2))

plt.plot(np.hstack((train[id_].to_numpy(), np.around(preds))), color='r')
plt.plot(train[id_].to_numpy())

plt.show()
# Как должено было бы работать 3-ое эксп. сглаживание:

x = np.linspace(0,int(np.pi*100),1000)
data_ = x*np.sin(x)+x*1.7+10

tr_ = data_[:-200]
vl_ = data_[-200:]

pred = model = ExponentialSmoothing(tr_,
                                    seasonal_periods=20,
                                    trend='add',
                                    seasonal='mul').fit().forecast(200)

plt.plot(np.hstack((tr_,pred)), color='r')
plt.plot(tr_);
id_ = '_1'
sample_data = date_df[id_]
sample_data = sample_data - sample_data.mean()
sample_data/= (sample_data.max() - sample_data.min())/2
sample_data
size_ = 1912
corr_history = []
for k in range(size_):  
    corr_history.append(stats.spearmanr(sample_data.values, np.roll(sample_data, k))[0])
plt.plot(corr_history);
import statsmodels.api as sm
fig, ax = plt.subplots(figsize=(20,8))
sm.graphics.tsa.plot_pacf(sample_data.values, lags=50, ax = ax)
plt.show()
# Если вдруг таблица на шагах выше испортилась, то сформируем её лишний раз.
date_df = main_df.drop(['event_type_1', 'snap_CA', 'snap_TX', 'snap_WI'], axis=1).reset_index()
date_df['Datetime'] = pd.to_datetime(date_df.date, format='%Y-%m-%d %H:%M:%S')
date_df.index  = date_df.Datetime
date_df.drop(['date', 'Datetime'], axis = 1, inplace = True)
date_df.head(2)
from sklearn.model_selection import TimeSeriesSplit 

W = 7 # Лучше чем брать 2,30,60,180,365. Проверили циклом.

errors_rmse,   errors_mape,   errors_rmsse   = [], [], []
errors_rmse_r, errors_mape_r, errors_rmsse_r = [], [], []
    
tscv = TimeSeriesSplit(n_splits=6) 

for id_ in tqdm(list(id_dict.values())[::100]):

  error_rmse, error_mape, error_rmsse       = [], [], []
  error_rmse_r, error_mape_r, error_rmsse_r = [], [], []

  for train_idx, val_idx in tscv.split(date_df[id_]):

      train = date_df.loc[:,id_].iloc[train_idx]
      val   = date_df.loc[:,id_].iloc[val_idx]

      model = ExponentialSmoothing(train,
                                  seasonal_periods=W,
                                  trend = None,
                                  seasonal='add').fit()

      preds = model.forecast(val_idx.shape[0])

      error_rmse.append(rmse(val, preds))
      error_mape.append(mape(val, preds))
      error_rmsse.append(rmsse(train, val, preds))

      error_rmse_r.append(rmse(val, np.around(preds)))
      error_mape_r.append(mape(val, np.around(preds)))
      error_rmsse_r.append(rmsse(train, val, np.around(preds)))

  errors_rmse.append(np.mean(error_rmse))
  errors_mape.append(np.mean(error_mape))
  errors_rmsse.append(np.mean(sorted(error_rmse)[1:-1]))

  errors_rmse_r.append(np.mean(error_rmse_r))
  errors_mape_r.append(np.mean(error_mape_r))
  errors_rmsse_r.append(np.mean(sorted(error_rmse_r)[1:-1]))    
  

print('W: ', W)
print('RMSE  - %.4f' % np.mean(errors_rmse))
print('MAPE  - %.4f' % np.mean(errors_mape))
print('RMSSE - %.4f' % np.median(errors_rmsse), end='\n\n')

print('RMSE_R  - %.4f' % np.mean(errors_rmse_r))
print('MAPE_R  - %.4f' % np.mean(errors_mape_r))
print('RMSSE_R - %.4f' % np.median(errors_rmsse_r))
# W = 7
# test_size = 28
# cols = ['F'+str(i) for i in range(1,29)]
# submission = pd.DataFrame([])

# for id_ in tqdm(list(id_dict.values())):
#     train = date_df.loc[:,id_]

#     model = ExponentialSmoothing(train,
#                                 seasonal_periods=W,
#                                 trend = None,
#                                 seasonal='add').fit()

#     preds = model.forecast(test_size)
#     submission.loc[int(id_[1:]),cols] = preds.to_numpy()
def one_prediction(train):

  return ExponentialSmoothing(train,
                              seasonal_periods=W,
                              trend = None,
                              seasonal='add').fit().forecast(28)
%%time
all_pred = np.apply_along_axis(one_prediction, arr=date_df.to_numpy().T, axis=1)
submission = pd.DataFrame(data=np.array(all_pred).reshape(28,30490)).T
submission = pd.concat((submission, submission), ignore_index=True)

sample_submission = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')        
submission[["id"]] = sample_submission[["id"]]

cols = list(submission.columns)
cols = cols[-1:] + cols[:-1]

submission = submission[cols]
submission.columns = ["id"] + [f"F{i}" for i in range (1,29)]
submission.to_csv("submission.csv", index=False)
submission.head(3)
import torch # Подключаем PyTorch - библиотеку для работы с нейронными сетями.
##**Рекурентный прогноз:**
log_step = 7 # Предполагаем, что сезонность имеет порядок в неделю.

class SATnet(torch.nn.Module):
  def __init__(self, n_hidden_neurons):
    super(SATnet, self).__init__()
    self.fc1  = torch.nn.Linear(log_step , n_hidden_neurons)
    self.act1 = torch.nn.ReLU()
    self.fc2  = torch.nn.Linear(n_hidden_neurons , n_hidden_neurons)
    self.act2 = torch.nn.ReLU()
    self.fc4  = torch.nn.Linear(n_hidden_neurons, 1)
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.act2(x)
    x = self.fc4(x)
    
    return x
  
SERIESnet = SATnet(50)
optimizer = torch.optim.Adam(SERIESnet.parameters(), lr=0.006)

def loss(pred, target):
  squares = (pred - target) ** 2
  return squares.mean()
# Функция для предикта test_size элементов.

def get_nn_predict(net, y_train, test_size):

  result = torch.Tensor([])
  prev_serie = y_train[-log_step:].T
  
  for _ in range(test_size):
    next_val = net.forward(prev_serie).data
    prev_serie = torch.cat((prev_serie[:,1:].T, next_val)).detach().T
    result = torch.cat((result, next_val))

  return result
# Прогноз сразу на сезон вперед, чтобы избежать увелечения ошибки на перспективе.

class SATnet(torch.nn.Module):
  def __init__(self, n_hidden_neurons):
    super(SATnet, self).__init__()
    self.fc1  = torch.nn.Linear(log_step , n_hidden_neurons)
    self.act1 = torch.nn.ReLU()
    self.fc2  = torch.nn.Linear(n_hidden_neurons , n_hidden_neurons)
    self.act2 = torch.nn.ReLU()
    self.fc3  = torch.nn.Linear(n_hidden_neurons , n_hidden_neurons)
    self.act3 = torch.nn.ReLU()
    self.fc4  = torch.nn.Linear(n_hidden_neurons , n_hidden_neurons)
    self.act4 = torch.nn.ReLU()
    self.fc5  = torch.nn.Linear(n_hidden_neurons , n_hidden_neurons)
    self.act5 = torch.nn.ReLU()
    self.fc6  = torch.nn.Linear(n_hidden_neurons , n_hidden_neurons)
    self.act6 = torch.nn.ReLU()
    self.fc7  = torch.nn.Linear(n_hidden_neurons , n_hidden_neurons)
    self.act7 = torch.nn.ReLU()
    self.fc8  = torch.nn.Linear(n_hidden_neurons, log_step)
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.act1(x)
    x = self.fc2(x)
    x = self.act2(x)
    x = self.fc3(x)
    x = self.act3(x)
    x = self.fc4(x)
    x = self.act4(x)
    x = self.fc5(x)
    x = self.act5(x)
    x = self.fc6(x)
    x = self.act6(x)
    x = self.fc7(x)
    x = self.act7(x)
    x = self.fc8(x)
    
    return x
  
SERIESnet = SATnet(50)
# Функция для тренировки нейронной сети.

def train_NN(SERIESnet, X_train, y_train, N_number):

  min_smape = 201
  SERIESnet = SATnet(N_number)
  optimizer = torch.optim.Adam(SERIESnet.parameters(), lr=0.01)
  flag = True
  i=0
  k=0
  while i < 300:
    
    i+=1
    k+=1

    optimizer.zero_grad()
    
    y_pred = SERIESnet.forward(X_train)
    loss_test = loss(y_pred, y_train)
    loss_test.backward()
    
    optimizer.step()

    y_test_pred = get_nn_predict(SERIESnet, X_val, y_val.shape[0])
    cur_smape = smape(y_test_pred, y_val)

    if cur_smape < min_smape:
      min_smape = cur_smape
      best_SERIESnet = SERIESnet

    if cur_smape < 15 and flag:
      optimizer = torch.optim.Adam(SERIESnet.parameters(), lr=0.007)
      flag = False
      
    if cur_smape < 3.5:
      break

    if i > 250 and cur_smape > 15:
      i = 0
      SERIESnet = SATnet(N_number)
      optimizer = torch.optim.Adam(SERIESnet.parameters(), lr=0.01)          

    if k > 2000:
      break

  return best_SERIESnet
# Пока на этом остановимся, в перспективе реализовать прогноз временного ряда, операясь на множество лаговых статистик
#  значения парных корреляция и т.п. Использовать LSTM - рекурентную сеть с памятью.
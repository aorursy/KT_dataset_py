# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
bread_prise = 10 #хлеб по 10 лир за буханку
bread_quantity = 400 #Каждый день вы продаете по 400 буханок хлеба
bread_quantity_year = bread_quantity * 365

baking_materials = 6000 #6000 лир в неделю на материалы для выпекания
rent_and_electricity = 30000 #30 тыс. лир в месяц на аренду и электричество
salary_and_other = 30000 #30 тыс. лир в месяц на зарплату и прочие расходы

#определяем расходы за год
expense = baking_materials * (365 / 7) + rent_and_electricity * 12 + salary_and_other * 12
proceeds = bread_quantity_year * bread_prise #определяем выручку за год

margin_lir = (proceeds - expense)/bread_quantity_year
print ('Маржа с буханки хлеба:', round(margin_lir, 2), 'lir')
print('Прибыль предприятия относительно расхода', round((proceeds - expense)/expense, 3)*100, '%')
monthly_profit = (proceeds - expense)/12
print('Месячная прибыль предприятия была:', round(monthly_profit, 2), 'lir')

#Рассмотрим изменнения с продажей наборов
bread_quantity_1 = 300 #Теперь ежедневно магазин продает по 300 буханок хлеба
set_quantity_1 = 200 #ежедневно магазин проедет 200 наборов для завтрака
set_prise = 18 #Стоимость одного набора составляет 18 лир

#Расчитываем изменения объемов продаж хлеба,
#чтобы пропорционально скорректировать стоимость его материалов
growth_bread_quantity = (bread_quantity_1 + set_quantity_1)/bread_quantity

#Создание наборов привело к росту зарплатных и прочих расходов на 30 тыс. в месяц.
salary_and_other_1 = salary_and_other + 30000 

#Масло покупается раз в три дня партиями по 650 брикетов за 2860 лир.
butter_size = 650
butter_cost = 2860
butter_day = butter_size//set_quantity_1

#определяем расходы и выручку за год
expense_1 = (baking_materials * (365 / 7) * growth_bread_quantity) + (rent_and_electricity * 12) + (salary_and_other_1 * 12) + butter_cost * (365//butter_day)
proceeds_1 = (bread_quantity_1 * 365 * bread_prise) + (set_quantity_1 * 365 * set_prise)

monthly_profit_1 = (proceeds_1 - expense_1)/12
print('Месячная прибыль предприятия стала:', round(monthly_profit_1, 2), 'lir')
print('Месячная прибыль предприятия выросла на:', round(monthly_profit_1 - monthly_profit, 2), 
      'lir,', 'что составило увеличение на', round((monthly_profit_1  /monthly_profit - 1) * 100, 1), '%')



def monthly_profit_params (bread_quantity, set_quantity, baking_materials, growth_demand=1, butter_count=1, butter_day=3):

    bread_prise = 10
    set_prise = 18

    bread_quantity_demand = int(bread_quantity * growth_demand)
    set_quantity_demand = int(set_quantity * growth_demand)
    print('bread_quantity_demand:', bread_quantity_demand)
    print('set_quantity_demand:', set_quantity_demand)

    butter_size = 650
    butter_cost = 2860
    
    if butter_size * butter_count < set_quantity_demand * butter_day:
        set_quantity_next = butter_size * butter_count
        bread_quantity_next = int(bread_quantity_demand * butter_day  + (set_quantity_demand * butter_day - set_quantity_next) / 2)
    else:
        set_quantity_next = set_quantity_demand * butter_day
        bread_quantity_next = bread_quantity_demand * butter_day
    print('set_quantity in butter_day:', set_quantity_next)
    print('bread_quantity in butter_day:', bread_quantity_next)

    #изменения объемов хлеба, что пропорционально влияет на стоимость его материалов
    growth_bread_quantity_next = (bread_quantity_next + set_quantity_next)/(bread_quantity * butter_day + set_quantity * butter_day)

    baking_materials_next = baking_materials * growth_bread_quantity_next
    print('baking_materials with growth_bread_quantity:', round(baking_materials_next, 2))
    #остальные затраты не меняем, так как не указано, что они меняются вместе с объемом хлеба
    rent_and_electricity = 30000 
    salary_and_other = 60000


    #определяем расходы, выручку за год
    expense = baking_materials_next * (365/7) + rent_and_electricity * 12 + salary_and_other * 12 + butter_cost * butter_count * (365//butter_day)
    proceeds = (bread_quantity_next * (365//butter_day) * bread_prise) + (set_quantity_next * (365//butter_day) * set_prise)

    monthly_profit = (proceeds - expense)/12
    print('Месячная прибыль предприятия составит:', round(monthly_profit, 2), 'lir')
monthly_profit_params (300, 200, 7500, growth_demand=1.1, butter_count=1, butter_day=3)
monthly_profit_params (330, 220, 8225, growth_demand=1.1, butter_count=2, butter_day=5)
monthly_profit_params (330, 220, 8225, growth_demand=1.1, butter_count=1, butter_day=3)
monthly_profit_params (363, 242, 9047.5, growth_demand=1.1, butter_count=1, butter_day=3)
monthly_profit_params (363, 242, 9047.5, growth_demand=1.1, butter_count=2, butter_day=5)
monthly_profit_params (363, 242, 9047.5, growth_demand=1.1, butter_count=3, butter_day=7)
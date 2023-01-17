import numpy as np # линейная алгебра

import pandas as pd # обработка данных



# Файлы с данными доступны в директории "../input/"

# Например, выполнив данный код (нажав run или нажав Shift+Enter) вы увидите список файлов в директории input



import os

print(os.listdir("../input"))
# Загружаем данные из test.csv

test = pd.read_csv('../input/test.csv')

# Получаем значения столбца photoId

ids = test['photoId'].values

# Создаём DataFrame, с колонками photoId и withSmile, в котором на одном строке с чётным

# photoId стоит значение withSmile равное 1, с нечётным photoId - 0

answer = pd.DataFrame({'photoId': ids,

                       'withSmile': [i%2 for i in ids]})

# Записываем DataFrame в файл answer.csv

answer.to_csv('answer.csv', index=False)
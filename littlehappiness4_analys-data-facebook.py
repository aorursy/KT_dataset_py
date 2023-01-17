#Проанализировать массив данных при помощи языка Python (допускается и рекомендуется использование дополнительных библиотек): 
 #   вычисление среднего, максимального/минимального значений, медианы, моды числовых значений как для всего массива в целом, так и для каждого типа 
  #  контента (столбец Type) в отдельности. Найти самый популярный объект в выборке, объяснить почему. Решение предоставить в виде .py/.ipynb файла на github. 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#input data
df = pd.read_csv(os.path.join('../input', 'dataset_Facebook.csv'), sep = ';')
df.shape
#посмотрим на данные
df.head(10)
#define type fo each columns
df.dtypes
# define mean, max, min, 50% - mediana, (среднее, максимальное, минимальное, медиана)
df_all=df.describe()
df_all= df_all.drop(['count', 'std', '25%', '75%'], axis = 0) #исключим лишниe первый и третий квартили и число строк
df_all
#moda, найдем моду 
df.mode().head(1)
#определим какие типы бывают в данных
df.Type.value_counts()
#distribute data by type
df_photo = df[df['Type'] == 'Photo']
df_status = df[df['Type'] == 'Status']
df_link = df[df['Type'] == 'Link']
df_video = df[df['Type'] == 'Video']
#Type = photo
#аналогичено для каждого типа выводим среднее значение, минимальное, медиана = 50%, максимальное
df_photo1=df_photo.describe()
df_photo1 = df_photo1.drop(['count', 'std', '25%', '75%'], axis = 0)
df_photo1
#мода для Photo
df_photo.mode().head(1)
#Type = status
df_status.describe().drop(['count', 'std', '25%', '75%'], axis = 0)
#мода для Status
df_status.mode().head(1)
#Type = link
df_link.describe().drop(['count', 'std', '25%', '75%'], axis = 0)
#мода для Link
df_link.mode().head(1)
#Type = Video
df_video.describe().drop(['count', 'std', '25%', '75%'], axis = 0)
#мода для Video
df_video.mode().head(1)
#определим самый популярный объект
# Отсортируем все объекты по убыванию в порядке значимости критерия популярности
    #1. лайки
    #2. Количество комментариев
    #3. Количество взаимодействий с объектом
    # посмотрим на отсортированные данные, для того чтобы определить нужно ли проводить отбор по другим критерием
    # объект с индексом 244 (photo) является самым популярным, из-за наибольшего количества лайков (максимальное количество по всем объектам), 
    #комментариев (тоже максимальное количество по всем объектам), и просто взаимодействий с ним.
df1 = df.sort_values(['like','comment', 'Total Interactions'],ascending = False)
#Самый популярный объект
df1.head(1)

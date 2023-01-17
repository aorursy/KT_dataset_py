# Імпорт Pandas і Numpy
import numpy as np
import pandas as pd

#Вхідні дані доступні в "../input/"
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Будь-які результати доступні у папці "../output/kaggle/working"
data = pd.read_csv("/kaggle/input/uci-adult/adult.data.csv")
data.shape
data.head()
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
# Ваш код тут
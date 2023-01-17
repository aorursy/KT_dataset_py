import numpy as np
import pandas as pd
from scipy import stats
data = np.array((79725,12862,18022,76712,256440,14013,46083,6808,85781,1251,6081,50397,11020,13633,1064,496433,25308,6616,11210,13900))
data
data = pd.DataFrame(data=data)
data
data.describe()
std = 117539.291236
var = std**2
print("Varyans : ",var)
print("Varyans : ", np.var(data))
print("Standart Sapma : ", np.std(data))
shapiro_test = stats.shapiro(data)
shapiro_test
print("shapiro_test.statistic : ", shapiro_test[0])
print("shapiro_test.p_value : ", shapiro_test[1])
print('Ho : Veriler Normal Dağılım göstermektedir.')
print('H1 : Veriler Normal Dağılım göstermemektedir.')
alfa = 0.05 # Hata payı

p_value = shapiro_test[1]
if p_value < alfa:
    print('Veriler Normal Dağılım Göstermemektedir.')
else:
    print('Veriler Normal dağılım göstermektedir.')


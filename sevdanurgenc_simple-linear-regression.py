import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
rng = np.random.RandomState()
x = 10 * rng.rand(50)

y = 2 * x - 5 + rng.randn(50)
# 2 değeri slop yani eğitimi verir.
# -5 değeri bias yani intercept değeri içinidir. 

plt.figure(dpi=200)
plt.scatter(x, y)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', rotation=0, fontsize=18)

print('X\n', x, '\n')
print('y\n', y, '\n')
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)

#intercept bias değerini aktif etmiş oluyoruz.
#dot production olduğu için modeli fit edebilmek adına newaxis kullanıyoruz.

model.fit(x[:, np.newaxis], y)
#nplinspace 0 ve 10 arasındaki değişkenlerde 1000 tane değer alır ve linear bir çizgi oluşturur.
#newaxis boyutu ayarlanıyor.

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.figure(dpi=200)
plt.scatter(x, y)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', rotation=0, fontsize=18)
plt.plot(xfit, yfit);
print("Model eğimi:    ", model.coef_[0])
#çizginin ve modelin eğimidir.
#eğim iki iken model 1.93 yakın gelmiştir. 
print("Model kesişimi:", model.intercept_)
#intercept bias değer -4 gelmiş.
#modelde bias değerini -5 vermiştik, gelen değer -4 ve yakındır.
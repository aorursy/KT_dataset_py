import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
bh_data = load_boston()
print(bh_data.keys())
boston = pd.DataFrame(bh_data.data, columns=bh_data.feature_names)
print(bh_data.DESCR)
boston['MEDV'] = bh_data.target
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT','RM'])

Y = boston['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=9)
lin_reg_mod = LinearRegression()
lin_reg_mod.fit(X_train, y_train)
pred = lin_reg_mod.predict(X_test)
test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))



test_set_r2 = r2_score(y_test, pred)
print(test_set_rmse)

print(test_set_r2)
import matplotlib.pyplot as plt 



def estimate_coef(x, y): 

	# gözlem / puan sayısı

	n = np.size(x) 



	# x ve y vektörünün ortalaması 

	m_x, m_y = np.mean(x), np.mean(y) 



	# x ile ilgili çapraz sapma ve sapmanın hesaplanması 

	SS_xy = np.sum(y*x) - n*m_y*m_x 

	SS_xx = np.sum(x*x) - n*m_x*m_x 



	# regresyon katsayılarının hesaplanması 

	b_1 = SS_xy / SS_xx 

	b_0 = m_y - b_1*m_x 



	return(b_0, b_1) 



def plot_regression_line(x, y, b): 

    # SADECE GORSEL AMACLI

	# gerçek noktaları dağılım grafiği olarak çizmek

	plt.scatter(x, y, color = "m", 

			marker = "o", s = 30) 



	# öngörülen yanıt vektörü 

	y_pred = b[0] + b[1]*x 



	# regresyon çizgisinin çizilmesi 

	plt.plot(x, y_pred, color = "g") 



	# etiket koymak

	plt.xlabel('x') 

	plt.ylabel('y') 



	# gösterme işlevi 

	plt.show() 

    

    

    

""" düşük statü oranı ('LSTAT') ve oda sayısı ('RM')"""

X2 = X.LSTAT

plt.title("düşük statü oranı ('LSTAT')")

b = estimate_coef(X2, Y) 

plot_regression_line(X2, Y, b)



X3 = X.RM

plt.title("oda sayısı ('RM')")

b = estimate_coef(X3, Y) 

plot_regression_line(X3, Y, b)
#Laboratório 3...Análise de Dados
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
X, y = load_boston(return_X_y= True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=1)
ssc = StandardScaler()
X_train = ssc.fit_transform(X_train)
X_test = ssc.transform(X_test)
L =[i for i in range(X.shape[1])]
errs =[]
# здесь должен быть цикл, в котором на каждой итерации из L
# исключается номер самого худшего признака
for k in range(X.shape[1]-1, 0, -1):
  lr = LinearRegression()
  lr.fit(X_train[:, L], y_train)
  idx = np.argmin(np.abs(lr.coef_)) #номер худшего признака
  val = np.min(np.abs(lr.coef_))  #значение коэфициэнта при худшем признаке
  y_pred = lr.predict(X_test[:, L])
  err = np.mean((y_pred - y_test) ** 2)
  errs.append(err)
  L.pop(idx)
errs[::-1] # построить график (и отметить на графике минимум -- опиционально)
plt.plot(errs[::-1])
lr = LinearRegression()
errs =[]
# здесь должен быть цикл, в котором на каждой итерации 
# исключается несколько самых худших признаков
for k in range(1, X.shape[1]+1):
  rfe = RFE(lr, k)  # мы передаем линейную регресси.(модель) метод подбора модели 
  rfe.fit(X_train, y_train) # берем обьект рфе 
  y_pred = rfe.predict(X_train)
  err = np.mean((y_pred - y_train) ** 2)
  errs.append(err)
errs[::] # построить график (и отметить на графике минимум -- опиционально)
plt.plot(errs[::])
xe = np.argmin(errs)
ye =np.min(errs)
print(xe,ye);
pf  = PolynomialFeatures(2)
X_train  = pf.fit_transform(X_train)
X_test = pf.transform(X_test)
X_train.shape
lr = LinearRegression()
errs =[]
# здесь должен быть цикл, в котором на каждой итерации 
# исключается несколько самых худших признаков
for k in range(1, X_train.shape[1]+1):
    rfe = RFE(lr, k)
    rfe.fit(X_train, y_train)
    y_pred = rfe.predict(X_train)
    err = np.mean((y_pred - y_train) ** 2)
    errs.append(err)
# построить график (и отметить на графике минимум -- опиционально)
plt.plot(errs)  #рисуем график
xe = np.argmin(errs)  #дает значение по х
ye =np.min(errs)  #дает значение по у
print(xe,ye);
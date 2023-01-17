import numpy as np
import pandas as pd
df = pd.read_csv('../input/advertising/Advertising.csv', index_col = 0)
df.head()
df.describe().T
df.info()
import seaborn as sns
sns.jointplot(x = 'TV',y = 'sales', data = df, kind ='reg');
df.head()
from sklearn.linear_model import LinearRegression
X = df[['TV']]
X.head()
y = df[['sales']]
y.head()
reg_model = LinearRegression()
reg_model.fit(X,y)
reg_model
reg_model.intercept_
reg_model.coef_
reg_model.score(X,y)
7.03 + 0.04*40
import seaborn as sns
import matplotlib.pyplot as plt
g = sns.regplot(df['TV'], df['sales'],ci = None, scatter_kws={'color':'r','s':9})
g.set_title('Model Denklemi: Sales = 7.03 + TV*0.05')
g.set_ylabel('Satis Sayisi')
g.set_xlabel('TV Harcamalari')
plt.xlim(-10,310)
plt.ylim(bottom=0);
reg_model.intercept_ + reg_model.coef_*165
reg_model.predict([[165]])
new_data = [[5],[15],[30]]
reg_model.predict(new_data)
X = df.drop('sales', axis=1)
y = df[['sales']]
reg_model = LinearRegression()
reg_model.fit(X,y)
reg_model.intercept_
reg_model.coef_
2.94 + 30*0.04 + 10*0.19 - 40*0.001
new_dat = [[300],[120],[400]]
new_dat = pd.DataFrame(new_dat).T
reg_model.predict(new_dat)
from sklearn.metrics import mean_squared_error
y.head()
reg_model
reg_model.predict(X)
y_pred = reg_model.predict(X)
mse = mean_squared_error(y, y_pred)
mse
import numpy as np
rmse = np.sqrt(mse)
rmse
df.describe().T
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

X_train.head()
y_train.head()
X_test.head()
y_test.head()
X_train.shape
X_test.shape
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
from sklearn.model_selection import cross_val_score
reg_model = LinearRegression()
#FIRST METHOD:
-cross_val_score(reg_model,X,y,cv=10, scoring = 'neg_mean_squared_error')
np.mean(-cross_val_score(reg_model,X,y,cv=10, scoring = 'neg_mean_squared_error'))
np.std(-cross_val_score(reg_model, X,y,cv=10,scoring = 'neg_mean_squared_error'))
np.sqrt(np.mean(-cross_val_score(reg_model,X,y,cv=10, scoring = 'neg_mean_squared_error')))
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.20, random_state = 1)
reg_model = LinearRegression()
reg_model.fit(X_train,y_train)
np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv=10, scoring = 'neg_mean_squared_error')))
#Test
y_pred = reg_model.predict(X_test)
#Test_Error
np.sqrt(mean_squared_error(y_test,y_pred))

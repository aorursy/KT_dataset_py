# 导入必要的python 拓展库

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



# 读取数据集文件

df = pd.read_csv("../input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")



# 把timstamp格式转换成date格式

df['Date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date



# 按Date分组并取均值

df= df.groupby(df['Date']).mean()



print("length of data set:", len(df))

df.head()
# 使用加权平均价表示当日价格

df.rename(columns={'Weighted_Price':'cur_price'}, inplace = True)



# 把当日价格上移一行，作为明天的价格加入数据集中

df[['next_price']] = df[['cur_price']].shift(-1)

# 计算价格趋势，1代表上涨，0代表下降

df.loc[(df['cur_price']<df['next_price']), 'price_trend'] = 1

df.loc[(df['cur_price']>df['next_price']), 'price_trend'] = 0



df.drop(df.index[-1], axis=0, inplace=True) # 删除最后一行



df[['price_trend']] = df[['price_trend']].astype(int) # 修改数据类型为int



print("length:", len(df))

df.head()
# 生成画热力图用数据集

data_heatmap = df[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)','Volume_(Currency)','next_price']]



# 绘制热力图

import seaborn as sns

plt.figure(figsize=(10,10))

sns.heatmap(data_heatmap.corr(), vmin=0.5,annot=True, fmt=".3")

plt.show()
# 去除无关属性，生成数据集

data_base_model = df[['Open','High', 'Low', 'Close', 'price_trend']]



data_base_model.head()
# 拆分数据

from sklearn.model_selection import train_test_split

y = data_base_model['price_trend']

x = data_base_model.iloc[:,:4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)



len(x_train),len(x_test), len(y_train), len(y_test)
# 使用lbgfs即默认参数，建立模型

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='lbfgs')

clf.fit(x_train, y_train)
# 测试并评分

y_pred=clf.predict(x_test)

score = clf.score(x_test, y_test)

print("Accurancy of model:",score)
from sklearn.metrics import confusion_matrix



M = confusion_matrix(y_test, y_pred)

sns.heatmap(M, annot=True, fmt='d')

plt.show()
# 计算OHLCW五个价格的趋势

df['Weight'] = df['cur_price']

OHLCW = ['Open', 'High', 'Low', 'Close','Weight']

# 把昨日的OHLCW价格下移一行，作为昨日价格加入数据集中

df[['last_Open', 'last_High', 'last_Low', 'last_Close', 'last_Weight']] = df[OHLCW].shift(1)

# 比较价格得到趋势，1代表上涨，0代表下降

for price in OHLCW:

    df.loc[(df['last_'+price]<df[price]), price+'_trend'] = 1

    df.loc[(df['last_'+price]>df[price]), price+'_trend'] = 0



df.drop(df.index[0], axis=0, inplace=True) # 删除第一行



# 调整列的顺序便于观察

new_order = ['Volume_(BTC)','Volume_(Currency)']

for price in OHLCW:

    new_order.extend([price, 'last_'+price, price+'_trend'])

new_order.extend(['cur_price', 'next_price', 'price_trend'])

df = df[new_order]



# 修改数据类型为int

df[['Open_trend']] = df[['Open_trend']].astype(int)

df[['High_trend']] = df[['High_trend']].astype(int)

df[['Low_trend']] = df[['Low_trend']].astype(int) 

df[['Close_trend']] = df[['Close_trend']].astype(int) 

df[['Weight_trend']] = df[['Weight_trend']].astype(int) 



# 离散法优化用数据集

data_label = df[['Open_trend', 'High_trend', 'Low_trend', 'Close_trend','Weight_trend', 'price_trend']]



data_label
# 使用离散化的特征建立模型

y = data_label['price_trend']

x = data_label.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

clf = LogisticRegression(solver='lbfgs')

clf.fit(x_train, y_train)



y_pred=clf.predict(x_test)

score = clf.score(x_test, y_test)

score
# 拆分数据

from sklearn.model_selection import train_test_split

y = data_base_model['price_trend']

x = data_base_model.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)



# 使用坐标轴下降法，建立模型

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='liblinear')

clf.fit(x_train, y_train)



# 测试并评分

y_pred=clf.predict(x_test)

score = clf.score(x_test, y_test)

score
# 拆分数据

from sklearn.model_selection import train_test_split

y = data_base_model['price_trend']

x = data_base_model.iloc[:,:-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)



# 使用坐标轴下降法加精度，建立模型

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver='liblinear', tol=4e-6)

clf.fit(x_train, y_train)



# 测试并评分

y_pred=clf.predict(x_test)

score = clf.score(x_test, y_test)

score
# 读取数据

Bitcoin = data_base_model[['Open', 'Close', 'High', 'Low', 'price_trend']]

Vibration=Bitcoin.High-Bitcoin.Low

Bitcoin['vibration'] = Vibration

#Bitcoin.drop(['High','Low'], inplace=True, axis = 1)

Bitcoin.head(5)
# application module sklearn 

from sklearn import model_selection

# Sortir tous les attributs Variable indépendante取出所有自变量

# 日期本身不作为自变量

predictors = ['Open','Close','High', 'Low']

#predictors = ['Open','Close','vibration']

# Divisez le jeu en jeu d'apprentissage 75 pourcents et jeu de test 25 pourcents 

X_train, X_test, y_train, y_test = model_selection.train_test_split(Bitcoin[predictors], Bitcoin.price_trend, 

                                                                    test_size = 0.25, random_state = 1234)
# SVM Linaire

from sklearn import svm

import numpy as np



clf=svm.LinearSVC(C=0.1)

clf.fit(X_train,y_train)

from sklearn import metrics



# Classification sur le jeu de test

pred_linear_svc = clf.predict(X_test)

# Accuracy

metrics.accuracy_score(y_test, pred_linear_svc)
# 使用 sklearn 

from sklearn import model_selection

# 取出所有自变量

# 日期本身不作为自变量

predictors = Bitcoin.columns[:4]

# 以3比1的比例划分训练集和测试集

X_train, X_test, y_train, y_test = model_selection.train_test_split(Bitcoin[predictors], Bitcoin.price_trend, 

                                                                    test_size = 0.25, random_state = 1234)
# 使用 GridSearchCV 创建最优决策树

from sklearn.model_selection import GridSearchCV

from sklearn import tree

# 初始化参数

max_depth = [2,3,4,5,6]

min_samples_split = [2,4,6,8]

min_samples_leaf = [2,4,8,10,12]



parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}

# 测试不同参数下的结果

grid_dtcateg = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = parameters, cv=10)

# 拟合决策树模型

grid_dtcateg.fit(X_train, y_train)

# 确定最优参数

grid_dtcateg.best_params_
# metrics est la méthode d'estimation du modèle sklearn

from sklearn import metrics

# Construire l'arbre avec les paramètres optimisés

CART_Class = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf = 8, min_samples_split=2)

# création du modèle sur le jeu d'apprentissage

decision_tree = CART_Class.fit(X_train, y_train)

# Application du modèle sur le jeu de test et classfication

pred = CART_Class.predict(X_test)

# Accuracy

print('Accuracy：',metrics.accuracy_score(y_test, pred))
from sklearn import ensemble

# Construction du Forêt aléatoire

RF_class = ensemble.RandomForestClassifier(n_estimators=200, random_state=1234)

# Réalisation du forêt avec jeu d'apprentissage

RF_class.fit(X_train, y_train)

# Classification sur le jeu de test

RFclass_pred = RF_class.predict(X_test)

# Précision

print('Accuracy ：',metrics.accuracy_score(y_test, RFclass_pred))
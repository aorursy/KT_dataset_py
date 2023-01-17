import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

%matplotlib inline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
dtrain = pd.read_csv('../input/data_test.txt', delimiter = ",")
dtest = pd.read_csv('../input/data_train.txt', delimiter = ",")
dtrain.shape
dtest.shape
dtrain_1 = dtrain[dtrain['Occupancy']==1].shape[0]
dtrain_0 = dtrain[dtrain['Occupancy']==0].shape[0]
print("кол-во 1 {} и 0 {}".format(dtrain_1, dtrain_0))
#корреляционная матрица
dtrain.corr()
#целевой признак коррелирует с освещенностью, температурой, CO2 и HumRatio 
sbn.set(style = "ticks")
sbn.heatmap(dtrain.corr())
sbn.pairplot(dtrain, hue = "Occupancy")
dtrain.columns[dtrain.isnull().values.any()].tolist()
#в сете нет пропущенных значений
dtrain = dtrain.drop(('Humidity'), axis = 1)
dtest = dtest.drop(('Humidity'), axis = 1)
arr_name = []
arr_train = []
arr_val = []

# используем предварительно отобранные признаки
cols_x = ['Temperature', 'Light', 'CO2', 'HumidityRatio']   
# целевой признак
col_y = 'Occupancy'

# функция тестирования классификатора
def test_classifier(classifier, classifier_name):
    # обучаем классификатор
    classifier.fit(dtrain[cols_x], dtrain[col_y])

    # проверяем классификатор
    y_train = classifier.predict(dtrain[cols_x])
    
    # определяем точность
    y_train_acc = accuracy_score(dtrain[col_y], y_train)
    # для валидационной выборки
    y_val = classifier.predict(dtest[cols_x])
    y_val_acc = accuracy_score(dtest[col_y], y_val)

    # сохранение информации в массивы
    arr_name.append(classifier_name)
    arr_train.append(y_train_acc)
    arr_val.append(y_val_acc)
    
    # вывод промежуточных результатов
    print('Точность для алгоритма {} на обучающей выборке = {}, \
    на валидационной выборке = {}'\
          .format(classifier_name,\
                  round(y_train_acc, 3),\
                  round(y_val_acc, 3)))
    
    # возвращаем обученный классификатор
    return classifier

%time classifier = test_classifier(LogisticRegression(),'LR')
def sigmoid(x):
    ret = x.apply(lambda t: 1.0 / (1.0 + np.exp(-t)))
    return ret
alpha = 0.0001
def GradientDescent(x, y, count = 1001, alpha = 0.0001):
    w = pd.Series(data = ((np.random.randn(np.shape(x)[1])) / 300), index = x.columns, dtype = np.float64)
    i = 0
    res = []
    cs = []
    for j in range (10000):        
        i = i + 1
        #print(x.dot(w).head())
        y_pred = sigmoid(x.dot(w))
        #mi = (alpha/x.shape[0]) * np.matmul(x.transpose(), (y_pred - y))
        #print(mi)
        w = w - (alpha/x.shape[0]) * np.matmul(x.transpose(), (y_pred - y)) #y - правильный рез-т
        #print (w)
        lolkek = accuracy_score(y, ClassificationResult(x, w))
        #print(lolkek)
        #print(CostFunc(y, x.dot(w)))
        #print(y_pred.head())
        #print(lolkek)
        res.append(lolkek)
        cs.append(CostFunc(y, sigmoid(x.dot(w))))
        if lolkek > 0.98 or i > count:
            print('i = {}'.format(i))
            break
    return w, res, cs
def ClassificationResult(x, w):
    probability = sigmoid(x.dot(w))
    probability = probability.apply(lambda x: 0 if x < 0.5 else 1)
    #print (probability.describe())
    return probability
def CostFunc (y_true, y_pred):
    #print(np.shape(y_pred))
    #print(y_pred.describe())
    func = (-1.0/np.shape(y_pred)[0]) * (y_true.dot(np.log(y_pred))+(1-y_true).dot(np.log(1-y_pred)))
    return func
#w = np.ones(1, np.shape(dtrain)[1])
#w.shape
print(dtrain.columns)
#x_train = dtrain.drop(('Humidity'), axis = 1)
x_train = dtrain.drop(('Occupancy'), axis = 1)
x_train = x_train.drop(('date'), axis = 1)
y_train = dtrain['Occupancy']
tmp, gr, gr1 = GradientDescent(x_train, y_train, 300, 0.0000015)
#print(tmp)
#print('y')
#print(y_train.head())
y_t = sigmoid(x_train.dot(tmp))
#print(y_t.head())
print(CostFunc(y_train, y_t))
print(accuracy_score(y_train, ClassificationResult(x_train, tmp)))
plt.plot(range(len(gr)), gr)
plt.show()
plt.plot(gr1)
plt.show()
x_tr = dtest.drop(('date'), axis = 1)
print(accuracy_score(dtest['Occupancy'], ClassificationResult(x_tr.drop(('Occupancy'), axis = 1), tmp)))
dtrain.describe()
x_tr = dtest.drop(('date'), axis = 1)
print(accuracy_score(dtest['Occupancy'], ClassificationResult(x_tr.drop(('Occupancy'), axis = 1), tmp)))
dtrain.drop(('Occupancy'), axis = 1).columns
tmp #0.963904840033
y_t

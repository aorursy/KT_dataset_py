import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

%matplotlib inline

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,8)

# Для кириллицы на графиках
font = {'family': 'Verdana',
        'weight': 'normal'}
plt.rc('font', **font)
DATA_DIR = os.path.join('data')
IMG_DIR = os.path.join(DATA_DIR, 'images')
IMG_DIR
# filepath = os.path.join(DATA_DIR, 'table_data.csv')
# table_data = pd.read_csv(filepath)
table_data = pd.read_csv('/kaggle/input/data12/table_data.csv')
print("Число классов: %d"%table_data.species.nunique())
print(table_data.shape)
table_data.head()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
table_data.set_index('id')
np.random.seed(2020)
le = LabelEncoder()
species = le.fit(table_data['species']) #label encoding
print('количесво классов:', len(list(le.classes_)))
y = le.transform(table_data['species'])
id_s = table_data['id']
x = table_data.drop(['species', 'id'], axis='columns').to_numpy()
print('первые 10 y -' ,y[:10])
print('первые 10 признаков 1го объекта',x[1][:10])
sss = StratifiedShuffleSplit(n_splits=1, test_size=198) #ShuffleSplitting(всего 99 классов, а значит,
train_idx =0  # в тестовой выборке 198 объектов ( по паре из каждого класса)
test_idx=0
for train_index, test_index in sss.split(x, y):
    train_idx =train_index
    test_idx=test_index
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
id_s_train = id_s[train_index]
id_s_test = id_s[test_index]
# попробуем загрузить 1 изображение
# filepath = os.path.join(IMG_DIR, '33.jpg')
img = plt.imread('/kaggle/input/im1233/images/33.jpg')
print(img.shape)
plt.imshow(img, cmap='Greys')
plt.grid(None)
from skimage.transform import resize
img_resized = resize(img, (100, 100))
plt.imshow(img_resized, cmap='Greys')
plt.grid(None)
massive = []
leaves_for_picture = []
for i in range(1, 1585):
    img = plt.imread(f'/kaggle/input/im1233/images/{i}.jpg')
    img_resized = np.ravel(resize(img, (100, 100)))
    massive.append(np.array(img_resized))
    leaves_for_picture.append(np.array(resize(img, (100, 100))))
massive = np.array(massive)
print(massive.shape)
def get(a, b):
    c = []
    for i in b: # для каждого из b(id) подбираем пару в массиве а и заносим в новый массив
        c.append(a[i - 1])
    return c
imgs_train = get(massive, id_s_train)
imgs_test =get(massive, id_s_test)
leaves_for_pictures = get(leaves_for_picture, id_s_train)
xs = np.linspace(-3, 3, 1000)
for i in range(1, 26): # числа от 1 до 25
    plt.subplot(5, 5, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
                         # третье - номер текущей картинки, если ситать слева направо, сверху вниз
    plt.plot(xs, xs**i)
    # plt.axis("off") # отключить оси, получится просто 25 линий
sample = np.random.normal(size=1000) # гистограмма строится по одномерной выборке - вектору чисел
_ = plt.hist(sample, bins=100) # то, что возвращает функция, сохранять никуда не нужно. bins=100 - число столбиков.
a = set()
b = []
for index, i in enumerate(y_train):
    if i not in a:
        a.add(i)
        b.append(index)
    if len(b) >= 99:
        break
leafs = get(leaves_for_picture, b)
for i in range(1, 101): # числа от 1 до 100
    plt.subplot(10, 10, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
                         # третье - номер текущей картинки, если ситать слева направо, сверху вниз
    if i < 100:
        plt.imshow(leafs[i - 1], cmap='Greys')
    else:
        plt.plot()
    plt.grid(None)
    plt.axis("off") # отключить оси, получится просто 25 линий
a = []
np.random.seed(50)
f = np.random.choice(list(range(10000)), 100)
data = pd.DataFrame(imgs_train)
data.head()
# import warnings
# warnings.simplefilter('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'svg' 
from pylab import rcParams
rcParams['figure.figsize'] = 15, 18
import pandas as pd
for i in range(1, 101): # числа от 1 до 100
    plt.subplot(10, 10, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
    plt.hist(data[f[i-1]], bins=10)
#     plt.axis("off")
rcParams['figure.figsize'] = 5, 8
plt.imshow(data.mean().to_numpy().reshape(100,100), cmap='Greys')
plt.grid(None)
plt.imshow(data.std().to_numpy().reshape(100,100), cmap='Greys')
plt.grid(None)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=9, random_state=0).fit(imgs_train)
kmeans.predict(imgs_test)
for i in range(1, 10): # числа от 1 до 9
    plt.subplot(3, 3, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
    plt.imshow(kmeans.cluster_centers_[i-1].reshape(100,100), cmap='Greys')
    plt.grid(None)
    plt.axis("off")
kmeans = KMeans(n_clusters=25, random_state=0).fit(imgs_train)
kmeans.predict(imgs_test)
for i in range(1, 26): # числа от 1 до 25
    plt.subplot(5, 5, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
    plt.imshow(kmeans.cluster_centers_[i-1].reshape(100,100), cmap='Greys')
    plt.grid(None)
    plt.axis("off")
kmeans = KMeans(n_clusters=100, random_state=0).fit(imgs_train)
kmeans.predict(imgs_test)
for i in range(1, 101): # числа от 1 до 100
    plt.subplot(10, 10, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
    plt.imshow(kmeans.cluster_centers_[i-1].reshape(100,100), cmap='Greys')
    plt.grid(None)
    plt.axis("off")
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import  MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
clf = NearestCentroid()
clf.fit(imgs_train, y_train)
accuracy_score(clf.predict(imgs_test), y_test)
clf.centroids_
rcParams['figure.figsize'] = 15, 18
for i in range(1, 100): # числа от 1 до 100
    plt.subplot(10, 10, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
    plt.imshow(clf.centroids_[i-1].reshape(100, 100), cmap='Greys')
    plt.grid(None)
    plt.axis("off")
clf = MultinomialNB()
clf.fit(imgs_train, y_train)
accuracy_score(clf.predict(imgs_test), y_test)
clf.feature_log_prob_
for i in range(1, 100): # числа от 1 до 100
    plt.subplot(10, 10, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
    plt.imshow(clf.feature_log_prob_[i-1].reshape(100, 100), cmap='Greys')
    plt.grid(None)
    plt.axis("off")
clf = LogisticRegression()
clf.fit(imgs_train, y_train)
accuracy_score(clf.predict(imgs_test), y_test)
clf.coef_
for i in range(1, 100): # числа от 1 до 100
    plt.subplot(10, 10, i) # первое число - сколько картинок по вертикали, второе - сколько по горизонтали, 
    plt.imshow(clf.coef_[i-1].reshape(100, 100), cmap='Greys')
    plt.grid(None)
    plt.axis("off")
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import RandomForestClassifier
rcParams['figure.figsize'] = 5, 8
randomforest = RandomForestClassifier()
randomforest.fit(imgs_train, y_train)
randomforest.predict(imgs_test)
print(accuracy_score(randomforest.predict(imgs_test), y_test))
plt.imshow(randomforest.feature_importances_.reshape(100,100), cmap='Greys')
plt.grid(None)
plt.axis("off")
randomforest.feature_importances_
from sklearn.decomposition import PCA
num_components = list(2**np.arange(10))
train_accur = []
test_accur = []
const = []
print(num_components)
for i in num_components:
    red = PCA(n_components=i)
    data = red.fit_transform(massive)
    X_train1 = get(massive, train_idx)
    X_test1 =get(massive, test_idx)
    randomforest = RandomForestClassifier()
    const.append(0.001) # всего 99 классов , то есть рандомный классификатор с точностью примерно 0.001 (p*q) даст верный ответ
    randomforest.fit(X_train1, y_train)
    test_accur.append(accuracy_score(randomforest.predict(X_test1), y_test))
    train_accur.append(accuracy_score(randomforest.predict(X_train1), y_train))
plt.plot(num_components, train_accur, label='train')
plt.plot(num_components, test_accur, label='test')
plt.plot(num_components, const, label='const')
plt.xlabel('num_components')
plt.xscale("symlog")
plt.legend()
print(max(test_accur))
num_components[test_accur.index(max(test_accur))]
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
accuracy_score(clf.predict(X_test), y_test)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

rfcl = RandomForestClassifier()
parameters = {'criterion':['gini', 'entropy'], 'n_estimators':range(50, 100, 10), 'min_samples_leaf': 
             range(1, 5, 1), 'max_features': ['sqrt', 'log2']}
clf = GridSearchCV(rfcl, parameters, cv=4, scoring='accuracy', n_jobs=-1)
clf.fit(X_train, y_train)
accuracy_score(clf.predict(X_test), y_test)
clf.best_params_
err = []
a = clf.predict(X_test)
for i in range(len(a)):
    t = []
    if a[i] != y_test[i]:
        t.append(a[i])
        t.append(y_test[i])
        err.append(t)
        t = []
print(err)

print(len(y))

    

print('слева - ошибка, справа - нужный класс')
for i in err:
    pic_id = ''
    pic_id_real = ''
    f = False
    ff = False
    for ii in range(len(y)):
        if y[ii] == i[0] and not f:
            pic_id = id_s[ii]
            f = True
        elif y[ii] == i[1] and not ff:
            pic_id_real = id_s[ii]
            ff = True
        if (ff and f):
            break
    plt.subplot(1, 2, 1)
    plt.grid(None)
    plt.axis("off")
    plt.imshow(leaves_for_picture[pic_id], cmap='Greys')
    plt.subplot(1, 2, 2)
    plt.imshow(leaves_for_picture[pic_id_real], cmap='Greys')
    plt.grid(None)
    plt.axis("off")
    plt.show()
# Dataset

from sklearn import datasets



import sys

sys.path.insert(0, '../')



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

%matplotlib inline

%load_ext autoreload
!pip install MiniSom

from minisom import MiniSom
# carregar dados

iris = datasets.load_iris()

data = iris.data

# data normalization

data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)
classes = iris.target_names

features = iris.feature_names

iris_target = iris.target



df_iris = pd.DataFrame(data, columns=features)

df_iris["class_id"] = iris_target

df_iris["class"] = 0



# Creating numeric class identifiers (0,1,2) 

df_iris.loc[df_iris["class_id"]==0, 'class'] = str(classes[0])

df_iris.loc[df_iris["class_id"]==1, 'class'] = str(classes[1])

df_iris.loc[df_iris["class_id"]==2, 'class'] = str(classes[2])
sns.pairplot(df_iris.drop("class_id", axis=1), hue="class", height=3, diag_kind="kde")
# Inicialização e treinamento

som3x3 = MiniSom(x= 3, y = 3, input_len = 4, sigma=3, learning_rate=0.5,

             neighborhood_function='triangle', random_seed=10)



# Inicialização e treinamento

som7x7 = MiniSom(x= 7, y = 7, input_len = 4, sigma=3, learning_rate=0.5,

             neighborhood_function='triangle', random_seed=10)



# Inicialização e treinamento

som9x9 = MiniSom(x= 9, y = 9, input_len = 4, sigma=3, learning_rate=0.5,

             neighborhood_function='triangle', random_seed=10)
som3x3.pca_weights_init(data)

print("Training...3x3")

som3x3.train_batch(data, 1000, verbose=True)

print("\n...ready!3x3")



som7x7.pca_weights_init(data)

print("Training...7x7")

som7x7.train_batch(data, 1000, verbose=True)

print("\n...ready!7x7")



som9x9.pca_weights_init(data)

print("Training...9x9")

som9x9.train_batch(data, 1000, verbose=True)

print("\n...ready!9x9")
import matplotlib.patches as mpatches
t = np.zeros(len(iris_target),dtype=int)

t[iris_target == 0] = 0

t[iris_target == 1] = 1

t[iris_target == 2] = 2

# use different coalors and markers for each label

markers = ['o','s','D']

colors = ['r','g','b']



red_patch = mpatches.Patch(color='red', label='setosa')

blue_patch = mpatches.Patch(color='blue', label='versicolor')

green_patch = mpatches.Patch(color='green', label='virginica')
plt.figure(figsize=(8,5))

plt.title('Som 3x3')

plt.pcolor(som3x3.distance_map().T) # distance map as background

plt.colorbar()



for cnt,xx in enumerate(data):

    w = som3x3.winner(xx) # getting the winner

    plt.plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',

             markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)

    

plt.axis([0,som3x3.get_weights().shape[0],0,som3x3.get_weights().shape[1]])

plt.legend(handles=[red_patch, blue_patch,green_patch])

plt.show() # show the figure
plt.figure(figsize=(8,5))

plt.title('Som 7x7')

plt.pcolor(som7x7.distance_map().T) # distance map as background

plt.colorbar()



for cnt,xx in enumerate(data):

    w = som7x7.winner(xx) # getting the winner

    plt.plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',

             markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)

    

plt.axis([0,som7x7.get_weights().shape[0],0,som7x7.get_weights().shape[1]])

plt.legend(handles=[red_patch, blue_patch,green_patch])

plt.show() # show the figure
plt.figure(figsize=(8,5))

plt.title('Som 9x9')

plt.pcolor(som9x9.distance_map().T) # distance map as background

plt.colorbar()



for cnt,xx in enumerate(data):

    w = som9x9.winner(xx) # getting the winner

    plt.plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',

             markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)

    

plt.axis([0,som9x9.get_weights().shape[0],0,som9x9.get_weights().shape[1]])

plt.legend(handles=[red_patch, blue_patch,green_patch])

plt.show() # show the figure
plt.figure(figsize=(5,5))

plt.title('Heatmap para 3x3')

sns.heatmap(som3x3.distance_map(), annot=True)

plt.show()
plt.figure(figsize=(8,8))

plt.title('Heatmap para 7x7')

sns.heatmap(som7x7.distance_map(), annot=True)

plt.show()
plt.figure(figsize=(8,8))

plt.title('Heatmap para 9x9')

sns.heatmap(som9x9.distance_map(), annot=True)

plt.show()
som3x3.pca_weights_init(data)

max_iter = 10000

q_error_pca_init = []

iter_x = []

for i in range(max_iter):

    percent = 100*(i+1)/max_iter

    rand_i = np.random.randint(len(data)) # Corresponde ao treinamento randomico

    som3x3.update(data[rand_i], som3x3.winner(data[rand_i]), i, max_iter)

    if (i+1) % 100 == 0:

        error = som3x3.quantization_error(data)

        q_error_pca_init.append(error)

        iter_x.append(i)

        sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')



plt.title('Erro de quantização 3x3')

plt.plot(iter_x, q_error_pca_init)

plt.ylabel('quantization error')

plt.xlabel('iteration index')

plt.show()
som7x7.pca_weights_init(data)

max_iter = 10000

q_error_pca_init = []

iter_x = []

for i in range(max_iter):

    percent = 100*(i+1)/max_iter

    rand_i = np.random.randint(len(data)) # Corresponde ao treinamento randomico

    som7x7.update(data[rand_i], som7x7.winner(data[rand_i]), i, max_iter)

    if (i+1) % 100 == 0:

        error = som7x7.quantization_error(data)

        q_error_pca_init.append(error)

        iter_x.append(i)

        sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')



plt.title('Erro de quantização 7x7')

plt.plot(iter_x, q_error_pca_init)

plt.ylabel('quantization error')

plt.xlabel('iteration index')

plt.show()
som9x9.pca_weights_init(data)

max_iter = 10000

q_error_pca_init = []

iter_x = []

for i in range(max_iter):

    percent = 100*(i+1)/max_iter

    rand_i = np.random.randint(len(data)) # Corresponde ao treinamento randomico

    som9x9.update(data[rand_i], som9x9.winner(data[rand_i]), i, max_iter)

    if (i+1) % 100 == 0:

        error = som9x9.quantization_error(data)

        q_error_pca_init.append(error)

        iter_x.append(i)

        sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')



plt.title('Erro de quantização 9x9')

plt.plot(iter_x, q_error_pca_init)

plt.ylabel('quantization error')

plt.xlabel('iteration index')

plt.show()
class_assignments = som3x3.labels_map(data, df_iris['class'])



def classify(som, data, class_assignments):

    winmap = class_assignments

    default_class = np.sum(list(winmap.values())).most_common()[0][0]

    result = []

    for d in data:

        win_position = som3x3.winner(d)

        if win_position in winmap:

            result.append(winmap[win_position].most_common()[0][0])

        else:

            result.append(default_class)

    return result
X_train, X_test, y_train, y_test = train_test_split(data, df_iris['class'])



som3x3.pca_weights_init(X_train)

som3x3.train_random(X_train, 5000, verbose=False)

class_assignments = som3x3.labels_map(X_train, y_train)



print(classification_report(y_test, classify(som3x3, X_test, class_assignments)))
X_train, X_test, y_train, y_test = train_test_split(data, df_iris['class'])



som7x7.pca_weights_init(X_train)

som7x7.train_random(X_train, 5000, verbose=False)

class_assignments = som7x7.labels_map(X_train, y_train)



print(classification_report(y_test, classify(som7x7, X_test, class_assignments)))
X_train, X_test, y_train, y_test = train_test_split(data, df_iris['class'])



som9x9.pca_weights_init(X_train)

som9x9.train_random(X_train, 5000, verbose=False)

class_assignments = som9x9.labels_map(X_train, y_train)



print(classification_report(y_test, classify(som9x9, X_test, class_assignments)))
import sys

sys.path.insert(0, '../')



from minisom import MiniSom



import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



%load_ext autoreload
%autoreload 2

from sklearn import datasets

from sklearn.preprocessing import scale



# Carregar base de dados do sklearn

digits = datasets.load_digits(n_class=10)

data = digits.data  # matrix onde cada linha é o vetor que representa um digito

data = scale(data)

num = digits.target  # num[i] é o digito representado em data[i]



som = MiniSom(30, 30, 64, sigma=4,

              learning_rate=0.5, neighborhood_function='triangle')

som.pca_weights_init(data)

print("Treinando...")

som.train_random(data, 5000, verbose=True)  # treinamento randomico

print("\n...pronto!")
plt.figure(figsize=(8, 8))

plt.title('SOM 30x30 para Digitos')

wmap = {}

im = 0

for x, t in zip(data, num):  # scatterplot

    w = som.winner(x)

    wmap[w] = im

    plt. text(w[0]+.5,  w[1]+.5,  str(t),

              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})

    im = im + 1

plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])

plt.show()
plt.figure(figsize=(10, 10), facecolor='white')

plt.title('SOM 30x30 para Resultado')

cnt = 0

for j in reversed(range(20)):  # images mosaic

    for i in range(20):

        plt.subplot(20, 20, cnt+1, frameon=False,  xticks=[],  yticks=[])

        if (i, j) in wmap:

            plt.imshow(digits.images[wmap[(i, j)]],

                       cmap='Greys', interpolation='nearest')

        else:

            plt.imshow(np.zeros((8, 8)),  cmap='Greys')

        cnt = cnt + 1



plt.tight_layout()

plt.show()
import sys

sys.path.insert(0, '../')



from minisom import MiniSom



import numpy as np

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

%matplotlib inline



%load_ext autoreload

%autoreload 2
from sklearn.datasets import make_blobs

from sklearn.preprocessing import scale
outliers_percentage = 0.35

inliers = 300

outliers = int(inliers * outliers_percentage)





data = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[.3, .3],

                  n_samples=inliers, random_state=0)[0]





data = scale(data)

data = np.concatenate([data, 

                       (np.random.rand(outliers, 2)-.5)*4.])





som = MiniSom(2, 1, data.shape[1], sigma=1, learning_rate=0.5,

              neighborhood_function='triangle', random_seed=10)





som.train_batch(data, 100, verbose=True)  # random training
quantization_errors = np.linalg.norm(som.quantization(data) - data, axis=1)

error_treshold = np.percentile(quantization_errors, 

                               100*(1-outliers_percentage)+5)

is_outlier = quantization_errors > error_treshold
plt.hist(quantization_errors)

plt.axvline(error_treshold, color='k', linestyle='--')

plt.xlabel('error')

plt.ylabel('frequency')
plt.figure(figsize=(8, 8))

plt.scatter(data[~is_outlier, 0], data[~is_outlier, 1],

            label='inlier')

plt.scatter(data[is_outlier, 0], data[is_outlier, 1],

            label='outlier')

plt.legend()

plt.show()
from sklearn.datasets import make_circles

data = make_circles(noise=.1, n_samples=inliers, random_state=0)[0]

data = scale(data)

data = np.concatenate([data, 

                       (np.random.rand(outliers, 2)-.5)*4.])





som = MiniSom(5, 5, data.shape[1], sigma=1, learning_rate=0.5,

              neighborhood_function='triangle', random_seed=10)





som.train_batch(data, 100, verbose=True)  

quantization_errors = np.linalg.norm(som.quantization(data) - data, axis=1)

error_treshold = np.percentile(quantization_errors, 

                               100*(1-outliers_percentage)+5)

is_outlier = quantization_errors > error_treshold
plt.figure(figsize=(8, 8))

plt.scatter(data[~is_outlier, 0], data[~is_outlier, 1],

            label='inlier')

plt.scatter(data[is_outlier, 0], data[is_outlier, 1],

            label='outlier')

weights = som._weights.reshape(5*5, 2)

plt.scatter(weights[:, 0], weights[:,1],

           marker='+', s=320, c='g', label='weights')

plt.legend()

plt.show()
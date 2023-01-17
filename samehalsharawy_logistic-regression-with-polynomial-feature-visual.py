# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from pylab import rcParams



from sklearn.linear_model import LogisticRegression



from sklearn.preprocessing import scale

from sklearn.model_selection import train_test_split

from sklearn import metrics



from sklearn import datasets



from sklearn.decomposition import PCA

from sklearn.preprocessing import PolynomialFeatures

%matplotlib inline

rcParams['figure.figsize']=10,5
iris = datasets.load_iris()

df = pd.DataFrame(iris.data)

df[4] = iris.target

df
df[[0,1,2,3]] = scale(df[[0,1,2,3]])
pca = PCA()

df[[0,1,2,3]] = pca.fit_transform(df[[0,1,2,3]])

plt.bar(range(4),pca.explained_variance_ratio_)
df.drop(columns= [2,3],inplace=True)

plt.scatter(df[0],df[1])
x= df[[0,1]]

y= df[4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 5)
classifier = LogisticRegression()

classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

print('confusion matrix : \n', metrics.confusion_matrix(y_test, y_pred))

print('Accuracy : ', metrics.accuracy_score(y_test,y_pred))
def visualize(x_test,y_test,classifier):

    from matplotlib.colors import ListedColormap



    x1, x2 = np.meshgrid(np.arange(start = -4,stop =4, step = 0.01),

                        np.arange(start = -3, stop = 5, step = 0.01))

    plt.contourf(x1, x2, classifier.predict((np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),

                alpha= 0.75, cmap = ListedColormap(('red','green','blue')))



    plt.xlim(x1.min(), x1.max())

    plt.ylim(x2.min(), x2.max())



    plt.scatter(x_test[0],x_test[1],c=y_test)



    plt.title('classifier')

    plt.xlabel('pc1')

    plt.ylabel('pc2')

    plt.legend()

    plt.show()
visualize(x_test,y_test,classifier)
poly = PolynomialFeatures(degree=3, interaction_only= False, include_bias= False)

x_poly = poly.fit_transform(x_train)
poly.get_feature_names()
lrc = LogisticRegression()

lrc.fit(x_poly,y_train)

y_pred2 = lrc.predict(poly.transform(x_test))

print('confusion matrix : \n', metrics.confusion_matrix(y_test, y_pred2))

print('Accuracy : ', metrics.accuracy_score(y_test,y_pred2))
def visualize_poly(x_test,y_test,classifier,poly):

    from matplotlib.colors import ListedColormap

    x1, x2 = np.meshgrid(np.arange(start = -4,stop =4, step = 0.01),

                        np.arange(start = -3, stop = 5, step = 0.01))

    plt.contourf(x1, x2, classifier.predict(poly.fit_transform(np.array([x1.ravel(),x2.ravel()]).T)).reshape(x1.shape),

                alpha= 0.75, cmap = ListedColormap(('red','green','blue')))



    plt.xlim(x1.min(), x1.max())

    plt.ylim(x2.min(), x2.max())



    plt.scatter(x_test[0],x_test[1],c=y_test)



    plt.title('classifier')

    plt.xlabel('pc1')

    plt.ylabel('pc2')

    plt.legend()

    plt.show()
visualize_poly(x_test,y_test,lrc,poly)
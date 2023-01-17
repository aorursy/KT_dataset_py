import numpy as np

import pandas as pd

import seaborn as sb

from sklearn.preprocessing import Normalizer

sb.set_style("dark")

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix

import time



%pylab inline
%%time

train_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")
train_df.isnull().sum().describe()
train_df.describe()
train_df.shape
target=train_df['label']

train_df=train_df.drop('label',axis=1)
figure(figsize(5,5))

for digit_num in range(0,64):

    subplot(8,8,digit_num+1)

    grid_data = train_df.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")

    xticks([])

    yticks([])
target.hist()
norm = Normalizer().fit(train_df)

train_df = norm.transform(train_df)

test_df = norm.transform(test_df)
train_df = pd.DataFrame(train_df)

test_df= pd.DataFrame(test_df)
pca = PCA(n_components=784, random_state=0, svd_solver='randomized')

pca.fit(train_df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.ylim(0.9, 1.0)

plt.grid()
def pca(X_tr, X_ts, test,n):

    pca = PCA(n)

    pca.fit(X_tr)

    X_tr_pca = pca.transform(X_tr)

    X_ts_pca = pca.transform(X_ts)

    test_pca = pca.transform(test)

    return X_tr_pca, X_ts_pca, test_pca
X_train, X_test, y_train, y_test = train_test_split(train_df, target,

    test_size=0.1, random_state=2)
test_num=y_test[(y_test==9)].index

num=len(test_num)
X_num9_test=X_test.loc[test_num]
%%time

X_train_pca, X_test_pca,test_num9_pca = pca(X_train, X_test, X_num9_test,100)
%%time

model = KNeighborsClassifier(n_neighbors = 4, weights='distance')

model.fit(X_train_pca, y_train)

score = model.score(X_test_pca, y_test)

print ('KNN ', score)

#pred_submit = model.predict(test_df_pca)

pred_homework=model.predict(X_test_pca)
confusion_matrix(y_test,pred_homework) 
y_9=[9]*num
%%time

for i in range(1,20):

    model=KNeighborsClassifier(n_neighbors = i, weights='distance')

    model.fit(X_train_pca, y_train)

    score = model.score(test_num9_pca , y_9)

    print ('The accuracy of number 9''s {}NN score is :{} '.format(i,score))
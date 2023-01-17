dic = dt.load_digits()

dic.keys()
dic.data
dic.data.shape
dic.images.shape
import matplotlib.pyplot as plt

plt.imshow(dic.images[500])
X = dic.data

y = dic.target
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train.shape
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(n_neighbors = 7)

modelo = Knn.fit(X_train,y_train)



y_pred = modelo.predict(X_test)

y_score = modelo.score(X_test,y_test)

y_score
compara =pd.DataFrame(y_test)

compara['pred']= y_pred   #crinado coluna no dataframe

compara.head(100)
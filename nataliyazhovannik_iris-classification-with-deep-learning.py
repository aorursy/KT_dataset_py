import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #vizualization
import matplotlib.pyplot as plt #vizualization
from matplotlib import cm

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
iris = pd.read_csv("../input/Iris.csv")
iris.head(5)
iris.info()
iris.columns
iris['Species'].unique()
sns.pairplot(data=iris[iris.columns[1:6]], hue='Species')
plt.show()
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Versicolor', ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='yellow', label='Virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal length depending on Width")
fig = plt.gcf()
fig.set_size_inches(8,5)
plt.show()
plt.figure(figsize=(8,5)) 
sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r') 
plt.show()
plt.subplots(figsize = (10,8))
from pandas.tools import plotting

cmap = cm.get_cmap('summer') 
plotting.andrews_curves(iris.drop("Id", axis=1), "Species", colormap = cmap)
plt.show()
iris.drop('Id',axis=1, inplace=True) #
df_norm = iris[iris.columns[0:4]].apply(lambda x:(x - x.min())/(x.max() - x.min()))
df_norm.sample(n=5)
target = iris[['Species']].replace(iris['Species'].unique(), [0,1,2])
df = pd.concat([df_norm, target], axis=1)
df.sample(n=5)
import keras

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler, LabelBinarizer
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

X = StandardScaler().fit_transform(X)
y = LabelBinarizer().fit_transform(y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

model = Sequential()
model.add(Dense( 12, input_dim=4, activation = 'relu'))
model.add(Dense( units = 15, activation= 'relu'))
model.add(Dense( units = 8, activation= 'relu'))
model.add(Dense( units = 10, activation= 'relu'))
model.add(Dense( units = 3, activation= 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(x_train, y_train, epochs = 120, validation_data = (x_test, y_test))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.legend(['train', 'test'])
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train','Test'])
plt.show()

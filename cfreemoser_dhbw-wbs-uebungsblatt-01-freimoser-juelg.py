%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
# read data
df = pd.read_csv("../input/20nm_lucas_train.csv", index_col=0, low_memory=False)

# drop NaN values
df.dropna(how="any", inplace=True)

SolutionCol = df['Label']
features = df.loc[:, '400.0':'2480.0']
classes = {0: "Sand", 1: "LoamySand", 2: "SandyLoam", 3: "Loam", 4: "SiltLoam", 5: "Silt", 6: "SandyClayLoam",
           7: "ClayLoam", 8: "SiltyClayLoam", 9: "SandyClay", 10: "SiltyClay", 11: "Clay"}
df.head(5)
df.tail(5)
df.describe()
df.info()
#total = df.isnull().sum().sort_values(ascending=False)
#percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
len(df)
labels = df.Label.unique().sort()
labels
#sns.countplot(df['Label'])
df['Label'].value_counts().sort_index().plot.bar()
sns.distplot(df['Label'], bins=12, kde=False)
dataframe = df.loc[:, '400.0':'2480.0']

corrmat = dataframe.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, '2480.0')['2480.0'].index
cm = np.corrcoef(dataframe[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
pca = PCA(2)  # project from 104 to 2 dimensions
projected = pca.fit_transform(features)
df['pca-one'] = projected[:,0]
df['pca-two'] = projected[:,1] 

fg = sns.FacetGrid(data=df[['pca-one','pca-two', 'Label']], hue='Label', hue_order=labels, aspect=1.61)
fg.fig.set_size_inches(15,15)
fg.map(plt.scatter, 'pca-one', 'pca-two').add_legend()
pca = PCA(n_components=60)
pca.fit(features)
features = pca.transform(features)
from sklearn.manifold import TSNE
n_sne = 7000

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
projected = tsne.fit_transform(features)
df['x-tsne'] = projected[:,0]
df['y-tsne'] = projected[:,1]

fg = sns.FacetGrid(data=df[['x-tsne','y-tsne', 'Label']], hue='Label', hue_order=labels, aspect=1.61)
fg.fig.set_size_inches(15,15)
fg.map(plt.scatter, 'x-tsne', 'y-tsne').add_legend()
minMaxScaler = preprocessing.MinMaxScaler()
features = minMaxScaler.fit_transform(features)
x_train, x_test, y_train, y_test = train_test_split(features, SolutionCol, test_size=0.1, random_state=1)
x_rfc_train = x_train
y_rfc_train = y_train
x_rfc_test = x_test
y_rfc_test = y_test
rfc = RandomForestClassifier(n_jobs=-1, n_estimators=100)
scores = cross_val_score(rfc, x_rfc_train, y_rfc_train, cv=10)
np.average(scores)
x_knn_train = x_train
y_knn_train = y_train
x_knn_test = x_test
y_knn_test = y_test
knn= neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance', n_jobs=-1)
scores = cross_val_score(knn, x_knn_train, y_knn_train, cv=10)
np.average(scores)
'''
from keras.utils.np_utils import to_categorical
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
'''
def reduceClasses(label):
    return {
        0 : 0,
        1 : 1,
        2 : 2,
        3 : 3,
        4 : 4,
        6 : 5,
        7 : 6,
        8 : 7,
        10 : 8,
        11 : 9
    }[label]
def reduceClassesReverse(label):
    return {
        0 : 0,
        1 : 1,
        2 : 2,
        3 : 3,
        4 : 4,
        5 : 6,
        6 : 7,
        7 : 8,
        8 : 10,
        9 : 11
    }[label]
from keras.utils.np_utils import to_categorical
def makeLabelVector(y):
    labels = []
    for data in y:
        labels.append(reduceClasses(data))
    return to_categorical(labels)

def makeClass(y):
    labels = []
    for data in y:
        labels.append(reduceClasses(data))
    return labels

def makeLabelReverse(y):
    labels = []
    for data in y:
        labels.append(reduceClassesReverse(data))
    return labels
x_ann_train = x_train
y_ann_train = y_train
x_ann_test = x_test
y_ann_test = y_test
y_ann_test_metric = makeClass(y_test)
y_ann_train = makeLabelVector(y_ann_train)
y_ann_test = makeLabelVector(y_ann_test)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(40, 80, 10), random_state=1, learning_rate_init=.001, activation='relu', max_iter=400)
scores = cross_val_score(mlp, x_ann_train, y_ann_train, cv=10)
np.average(scores)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras import regularizers
def kerasAnn():
    classifier = Sequential()
        # Adding the input layer and the first hidden layer
    classifier.add(Dense(256, activation='relu', input_dim=60))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    classifier.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return classifier
trainingModel = kerasAnn()
hist = trainingModel.fit(x_ann_train, y_ann_train, batch_size=350, epochs=1200, validation_split=0.1, verbose=0)
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
rfc.fit(x_rfc_train, y_rfc_train)
knn.fit(x_knn_train, y_knn_train)
mlp.fit(x_ann_train, y_ann_train)
kerasAnn = kerasAnn()
kerasAnn.fit(x_ann_train, y_ann_train, batch_size=350, epochs=1200, verbose=0)
predictions_rfc = rfc.predict(x_rfc_test)
acc_rfc =accuracy_score(y_rfc_test, predictions_rfc)
acc_rfc
predictions_knn = knn.predict(x_knn_test)
acc_knn =accuracy_score(y_knn_test, predictions_knn)
acc_knn
# Needed for mapping
lb = preprocessing.LabelBinarizer()
lb.fit(y_ann_test_metric)

predictions_ann_sklearn = lb.inverse_transform(mlp.predict(x_ann_test))
mlp_acc =accuracy_score(y_ann_test_metric, predictions_ann_sklearn)
mlp_acc

predictions_ann_keras = kerasAnn.predict_classes(x_ann_test)
acc_ann_keras =accuracy_score(y_ann_test_metric, predictions_ann_keras)
acc_ann_keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
target_names = ['Sand [0]', 'LoamySand [1]', 'SandyLoam [2]' , 'Loam [3]' , 'SiltLoam [4]' , 'SandyClayLoam [6]', 'ClayLoam [7]', 'SiltyClayLoam [8]', 'SiltyClay [10]', 'Clay [11]' ]
print(classification_report(y_rfc_test, predictions_rfc, target_names=target_names))
print(classification_report(y_knn_test, predictions_knn, target_names=target_names))
print(classification_report(y_ann_test_metric, predictions_ann_sklearn, target_names=target_names))
print(classification_report(y_ann_test_metric, predictions_ann_keras, target_names=target_names))
dfResult = pd.DataFrame({'Real Label' : y_test})
dfResult['Predictions RFC'] = predictions_rfc
dfResult['Predictions KNN'] = predictions_knn
dfResult['Predictions ANN Sklearn'] = makeLabelReverse(predictions_ann_sklearn)
dfResult['Predictions ANN Keras'] = makeLabelReverse(predictions_ann_keras)

dfResult.sample(frac=0.02)
from yellowbrick.classifier import ClassificationReport

def visClassificationReport(classifier, xTest, yTest):
    visualizer = ClassificationReport(classifier, classes=target_names)
    visualizer.score(xTest, yTest) 
    g = visualizer.poof()     
from yellowbrick.classifier import ClassPredictionError

def visClassPredictionError(classifier, xTest, yTest):
    visualizer = ClassPredictionError(classifier, classes=target_names)
    visualizer.score(xTest, yTest)
    visualizer.size = (1200 ,600)
    g = visualizer.poof()
from yellowbrick.classifier import ROCAUC

def visRocauc(classifier, xTest, yTest):
    visualizer = ROCAUC(classifier, classes=target_names)
    visualizer.score(xTest, yTest)
    g = visualizer.poof() 
visRocauc(rfc, x_rfc_test, y_rfc_test)
visClassificationReport(rfc, x_rfc_test, y_rfc_test)
visClassPredictionError(rfc, x_rfc_test, y_rfc_test)
visRocauc(knn, x_knn_test, y_knn_test)
visClassificationReport(knn, x_knn_test, y_knn_test)

visClassPredictionError(knn, x_knn_test, y_knn_test)
# visRocauc(mlp, x_ann_test, y_ann_test)
visClassificationReport(mlp, x_ann_test, y_ann_test)
# visClassPredictionError(mlp, x_ann_test, y_ann_test) not working
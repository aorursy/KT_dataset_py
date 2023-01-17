import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
%matplotlib inline
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
train_data.head()
train_data.shape
model_train_data_unscaled = train_data.drop(['label'], axis=1)
model_train_label = train_data['label']
plt.imshow(np.array(model_train_data_unscaled.loc[10]).reshape(28, 28), cmap='Greys')
print(model_train_label[10])
from sklearn.preprocessing import MinMaxScaler

std_scaler = MinMaxScaler()
model_train_data = std_scaler.fit_transform(model_train_data_unscaled.astype(np.float64))
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve

def model_def(model, model_name, m_train_data, m_train_label):
    model.fit(m_train_data, m_train_label)
    s = "predict_"
    p = s + model_name
    p = model.predict(m_train_data)
    cm = confusion_matrix(m_train_label, p)
    precision = np.diag(cm)/np.sum(cm, axis=0)
    recall    = np.diag(cm)/np.sum(cm, axis=1)
    F1 = 2 * np.mean(precision) * np.mean(recall)/(np.mean(precision) + np.mean(recall))
    cv_score = cross_val_score(model, m_train_data, m_train_label, cv=3, scoring='accuracy')
    print("Precision Is      :", np.mean(precision))
    print("Recall Is         :", np.mean(recall))
    print("F1 Score IS       :", F1)
    print("Mean CV Score     :", cv_score.mean())
    print("Std Dev CV Score  :", cv_score.std())
from sklearn.linear_model import LogisticRegression

softmax = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', C=0.05)
model_def(softmax, "softmax", model_train_data, model_train_label)
from sklearn.decomposition import PCA

pca = PCA(n_components = 200)
model_train_data2D = pca.fit_transform(model_train_data)
print("Explained Variance Ratio:", np.sum(pca.explained_variance_ratio_))
from sklearn.svm import SVC

poly6 = SVC(C=2, kernel='poly', degree=3, gamma='auto', random_state=42)
model_def(poly6, "poly6", model_train_data2D, model_train_label)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
def build_classifier():
    classifier = Sequential([Dense(128, kernel_initializer='random_uniform', activation='relu', input_shape=(200,)),
                             Dropout(rate=0.2),
                             Dense(128, kernel_initializer='random_uniform', activation='relu'),
                             Dropout(rate=0.2),
                             Dense(10, kernel_initializer='random_uniform', activation='softmax')])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

ANN_classifier = KerasClassifier(build_fn=build_classifier, batch_size=100, epochs=20)
ANN_classifier.fit(model_train_data2D, model_train_label)
cv_score = cross_val_score(ANN_classifier, model_train_data2D, model_train_label, cv=5, scoring='accuracy')
print("Mean CV Score Is:", cv_score.mean())
print("Mean CV Score Is:", cv_score.mean())
test_data = pd.read_csv("../input/digit-recognizer/test.csv")

model_test_data = std_scaler.transform(test_data.astype(np.float64))
model_test_data2D = pca.transform(model_test_data)
predict_test_poly6 = poly6.predict(model_test_data2D)
predict_test_softmax = softmax.predict(model_test_data)
predict_test_ANN = ANN_classifier.predict(model_test_data2D)

Id = DataFrame(np.arange(1,28001))
Id.columns = ['ImageId']

prediction = DataFrame(predict_test_poly6)
prediction.columns = ['Label']

result = pd.concat([Id, prediction], axis=1)
result.to_csv("Submission_Poly.csv", index=False)

Id = DataFrame(np.arange(1,28001))
Id.columns = ['ImageId']

prediction = DataFrame(predict_test_softmax)
prediction.columns = ['Label']

result = pd.concat([Id, prediction], axis=1)
result.to_csv("Submission_Softmax.csv", index=False)

Id = DataFrame(np.arange(1,28001))
Id.columns = ['ImageId']

prediction = DataFrame(predict_test_ANN)
prediction.columns = ['Label']

result = pd.concat([Id, prediction], axis=1)
result.to_csv("Submission_ANN.csv", index=False)
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
X_train = np.array(model_train_data_unscaled).reshape(42000, 28, 28, 1)
y_train = model_train_label
X_test = np.array(test_data).reshape(28000, 28, 28, 1)
CNN_model = Sequential([Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(28,28,1),
                            padding='valid', activation='relu'),
                         MaxPooling2D(pool_size=(2, 2)),
                         Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                            padding='valid', activation='relu'),
                         MaxPooling2D(pool_size=(2, 2)),
                         Flatten(),
                         Dense(128, kernel_initializer='random_uniform', activation='relu'),
                         Dropout(rate=0.2),
                         Dense(128, kernel_initializer='random_uniform', activation='relu'),
                         Dropout(rate=0.2),
                         Dense(10, kernel_initializer='random_uniform', activation='softmax')])

CNN_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

CNN_model.fit(X_train, y_train, batch_size=100, epochs=20)
y_test = CNN_model.predict(X_test)
y_test.shape
predict_test_CNN = np.argmax(y_test, axis=1)
plt.imshow(np.array(test_data.loc[0]).reshape(28, 28), cmap='Greys')
print(predict_test_CNN[0])
Id = DataFrame(np.arange(1,28001))
Id.columns = ['ImageId']

prediction = DataFrame(predict_test_CNN)
prediction.columns = ['Label']

result = pd.concat([Id, prediction], axis=1)
result.to_csv("Submission_CNN.csv", index=False)

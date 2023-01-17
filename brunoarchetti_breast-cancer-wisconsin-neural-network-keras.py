import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns # geração de gráficos

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')
df1 = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df1.sample(10)
from sklearn.preprocessing import LabelEncoder
diagnosis = LabelEncoder()
df1['diagnosis'] = diagnosis.fit_transform(df1['diagnosis']) 
df1.head() # 0 = B; 1 = M 
df1.drop(columns =['id','Unnamed: 32'], axis=1, inplace=True)
df1.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df1.loc[:,df1.columns != 'diagnosis'],
                                                    df1['diagnosis'],
                                                    test_size=0.3)
x_train.head()
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
df_pivot = pd.DataFrame({'types': x_train.dtypes,
                         'nulls': x_train.isna().sum(),
                          '% nulls': x_train.isna().sum() / x_train.shape[0],
                          'size': x_train.shape[0],
                          'uniques': x_train.nunique()})
df_pivot
fig, ax = plt.subplots(figsize=(12,7))
sns.heatmap(x_train.corr(), vmin=-1, vmax=1,
            cmap=sns.diverging_palette(20, 220, as_cmap=True), 
            yticklabels=True) # show all y values

plt.show()
df1.corr()['diagnosis'].sort_values(ascending=False).head()
plt.scatter(x_train['concave points_worst'], x_train['perimeter_worst'], c=y_train, cmap=plt.cm.Spectral)
plt.colorbar()
plt.xlabel('concave points_worst')
plt.ylabel('perimeter_worst')
plt.title('concave points_worst x perimeter_worst')
plt.show()

sns.boxplot(y=x_train['concave points_worst'], x=y_train)
plt.show()
sns.boxplot(y=x_train['perimeter_worst'], x=y_train)
plt.show()
# Positive Correlated
plt.xlabel('area_se')
plt.ylabel('radius_se')
plt.scatter(x_train['area_se'], x_train['radius_se'])
plt.show()
from sklearn import preprocessing
preprocessParams = preprocessing.StandardScaler().fit(x_train)
x_train_normalized = preprocessParams.transform(x_train)
x_test_normalized = preprocessParams.transform(x_test)

x_train_normalized[:1] 
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
NumerOfClasses = len(y_train.unique())
NumerOfClasses
RN = Sequential() # create network structure
RN.add(Dense(10, input_shape = x_train_normalized.shape[1:], activation ='sigmoid'))
RN.add(Dense(NumerOfClasses, activation ='sigmoid'))
RN.summary()
# training
from keras.utils import to_categorical
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)
RN.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
trainedRN = RN.fit(x_train_normalized, to_categorical(y_train), epochs=100, verbose=1)
score = RN.evaluate(x_test_normalized, to_categorical(y_test),verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])
#Predict
from sklearn.metrics import confusion_matrix
y_test_predicted = RN.predict(x_test_normalized)
y_test_predicted_index = np.argmax(y_test_predicted, axis=1)
y_test_index = y_test.values
#Confusion Matrix
confMatrix = pd.DataFrame(confusion_matrix(y_test_predicted_index, y_test_index),
                           index=['0 - Benigno','1 - Maligno'],columns=['0 - Benigno','1 - Maligno'])

confMatrix.index.name = 'Actual'
confMatrix.columns.name= 'Predicted'
print(confMatrix)
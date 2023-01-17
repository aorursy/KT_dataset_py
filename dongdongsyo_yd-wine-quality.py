import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, glm
# importing data
red_wine = pd.read_csv('/kaggle/input/wine-quality/wineQualityReds.csv')
red_wine["type"] = "1"
red_wine.head()
white_wine = pd.read_csv('/kaggle/input/wine-quality/wineQualityWhites.csv')
white_wine["type"] = "0"
white_wine.head()
wine = pd.concat([red_wine, white_wine], axis=0)
wine.head()
#변수별 요약통계
wine.describe()
# quality열에서 유니크한 값 
print(wine.quality.unique())
print(sorted(wine.quality.unique()))

# quality열에서 유일한 값별 관측값 개수를 내럼차순으로 정렬하여 출력
# quality6가 가장 많음. 
print(wine.quality.value_counts())

# 와인 종류에 따른 기술 통계를 출력하기
wine.groupby('type')[['quality']].describe()

wine.groupby('type')[['quality']].describe().unstack('type') #가로방향으로 재구조화

# 와인종류에 따른 품질의 분포 확인하기
red_wine = wine.loc[wine['type']=='1', 'quality']
white_wine = wine.loc[wine['type']=='0', 'quality']
print(red_wine.head())
print(white_wine.head())

import seaborn as sns
import matplotlib.pyplot as plt
#빈도분포대신 밀도 분포로 표시
sns.set_style("dark")
print(sns.distplot(red_wine, \
		norm_hist=True, kde=False, color="red", label="Red wine"))
print(sns.distplot(white_wine, \
		norm_hist=True, kde=False, color="white", label="White wine"))
plt.xlabel("Quality Score")
plt.ylabel("Density")
plt.title("Distribution of Quality by Wine Type")
plt.legend()
plt.show()

import seaborn as sns
corr = wine.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corr, 
            vmax=0.4, vmin=-0.4,linewidths=1, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

# Split Train and Test

from sklearn.model_selection import train_test_split
import numpy as np

# Specify the data 
X= wine[['fixed.acidity', 'volatile.acidity', 'citric.acid', 'residual.sugar', 'chlorides', 'free.sulfur.dioxide', 'total.sulfur.dioxide', 'density', 'pH', 'sulphates', 'alcohol','type']].values  # 12 dimensions

# Specify the target labels and flatten the array 
y=np.ravel(wine.quality)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sc = StandardScaler()
X_norm = sc.fit_transform(X)
# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.33, random_state=42)
#X_train=X_train.astype('float')
#X_train
#X_test=X_test.astype('float')
#X_test
#y_train=y_train.astype('float')
#y_train
#y_test=y_test.astype('float')
#y_test
#sns.boxplot(data = X_train) 

#sns.boxplot(data = X_test)

# Data Model

#import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(128, activation='relu', input_shape=(12,)))

#model.add(Dropout(0.3))

# Add one hidden layer 
model.add(Dense(16, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='relu'))

#X_train
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test), 1)
#y_train
model.input_shape

model.output_shape

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
                   
history = model.fit(X_train, y_train,epochs=20, batch_size=10, validation_data=(X_test, y_test), verbose=1)


print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate(X_test, y_test,verbose=1)
print(score)


import pandas as pd

base = pd.read_csv('../input/autos.csv', encoding= 'ISO-8859-1')
base.head()
base.drop(['dateCrawled','dateCreated','postalCode','lastSeen','nrOfPictures'], axis = 1, inplace=True)
len(base.columns)
base.describe(include=['O'])
base.drop('name', axis=1, inplace=True)
base['seller'].value_counts()
base.drop('seller',axis=1, inplace=True)
base['offerType'].value_counts()
base.drop(['offerType'], axis=1, inplace=True)
base['abtest'].value_counts()
import seaborn as sns

ax = sns.countplot(base['abtest'],label="Count")
base['vehicleType'].value_counts()
ax = sns.countplot(base['vehicleType'],label="Count")
base['gearbox'].value_counts()
ax = sns.countplot(base['gearbox'],label="Count")
base['model'].value_counts()
ax = sns.countplot(base['model'], label="count")
base['fuelType'].value_counts()
ax = sns.countplot(base['fuelType'],label="Count")
base['brand'].value_counts()
base['notRepairedDamage'].value_counts()
i1 = base.loc[base.price <= 10]

len(i1)
base.price.mean()
base = base[ base.price > 10]
i1 = base.loc[base.price > 500000]

i1.head()
base = base[ base.price < 500000]
base.isnull().sum().sort_values(ascending=False)
base['notRepairedDamage'].value_counts()
base['notRepairedDamage'].fillna('nein', inplace=True)
base['vehicleType'].value_counts()
base['vehicleType'].fillna('limousine', inplace=True)
base['fuelType'].value_counts()
base['fuelType'].fillna('benzin', inplace=True)
base['model'].value_counts()
base['model'].fillna('golf', inplace=True)
base['gearbox'].value_counts()
base['gearbox'].fillna('manuell',inplace=True)
base.isnull().sum().sort_values(ascending=False)
X = base.drop('price',axis=1)

y = base['price']
X.head()
y.head()
X.describe(include=['O'])
X = pd.get_dummies(X)

len(X.columns)
X.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25)
from keras.models import Sequential

from keras.layers import Dense



regressor = Sequential()

#(316 + 1)/2 = 158

regressor.add(Dense(units=158, activation='relu', input_dim = 316))

regressor.add(Dense(units=158, activation='relu'))

#activation='linear' não faz nenhum calculo adicional

regressor.add(Dense(units=1, activation='linear'))

#mean_absolute_error = considera a média dos erros com valores absolutos.

regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','acc'])

#batch_size = numero de execuções antes de atualizar os erros.

model = regressor.fit(X_train, y_train,batch_size=300, epochs = 10)
print(model.history.keys())
print(model.history.keys())

# summarize history for accuracy

plt.plot(model.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(model.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#Fazer a predição dos valores da base X_test

predict = regressor.predict(X_test)



error = (sum(abs(a - b) for a, b in zip(predict, y_test))/ len(predict))[0]

print("O erro médio na predição é: {}".format(error))
import matplotlib.pyplot as plt

plt.figure(figsize=(20,7))

plt.plot(predict[0:300])

plt.plot(y_test[0:300].values)

plt.title('Predicted vs Real')

plt.ylabel('Price')

plt.xlabel('Id')

plt.legend(['Predicted', 'Real'], loc='upper left')



plt.show()
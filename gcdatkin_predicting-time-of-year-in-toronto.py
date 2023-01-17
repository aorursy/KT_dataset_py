import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression
data = pd.read_csv('../input/sundown-and-sunset-data-20102020/Sunup_Sundown_Data.csv')
data
data.info()
data['Month'] = data['Date'].apply(lambda date: date[5:7])
data['SunupHour'] = data['Sunup'].apply(lambda time: time[:2])

data['SunupMinute'] = data['Sunup'].apply(lambda time: time[-2:])



data['SundownHour'] = data['Sundown'].apply(lambda time: time[:2])

data['SundownMinute'] = data['Sundown'].apply(lambda time: time[-2:])
data['City'].unique()
data = data.drop(['City', 'Date', 'Sunup', 'Sundown'], axis=1)
data
data = data.astype(np.int)
corr = data.corr()



plt.figure(figsize=(12, 10))

sns.heatmap(corr, annot=True, vmin=-1.0, vmax=1.0, cmap='rocket')

plt.show()
plt.figure(figsize=(16, 5))

for column in data.columns:

    sns.kdeplot(data[column], shade=True)

plt.show()
y = data['Month'].copy()

X = data.drop('Month', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=122)
models = []

Cs = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]



for C in Cs:

    model = LogisticRegression(C=C)

    model.fit(X_train, y_train)

    models.append(model)
model_acc = [model.score(X_test, y_test) for model in models]



print(f"   Model Accuracy (C={Cs[0]}):", model_acc[0])

print(f"    Model Accuracy (C={Cs[1]}):", model_acc[1])

print(f"    Model Accuracy (C={Cs[2]}):", model_acc[2])

print(f"   Model Accuracy (C={Cs[3]}):", model_acc[3])

print(f"  Model Accuracy (C={Cs[4]}):", model_acc[4])

print(f" Model Accuracy (C={Cs[5]}):", model_acc[5])

print(f"Model Accuracy (C={Cs[6]}):", model_acc[6])
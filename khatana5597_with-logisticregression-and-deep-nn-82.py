import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import train_test_split
data= pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
print ("Rows     : " ,data.shape[0])

print ("Columns  : " ,data.shape[1])

print ("\nFeatures : \n" ,data.columns.tolist())

print ("\nMissing values :  ", data.isnull().sum().values.sum())

print ("\nUnique values :  \n",data.nunique())
#dropout customerID becouse it is not usefull for Churn

data=data.drop(['customerID'],axis=1)

data.head()
# Checking the data types of all the columns

data.info()
import seaborn as sns

sns.set(style="ticks", color_codes=True)

df=data

fig, axes = plt.subplots(nrows = 3,ncols = 5,figsize = (25,15))

sns.countplot(x = "gender", data = df, ax=axes[0][0])

sns.countplot(x = "Partner", data = df, ax=axes[0][1])

sns.countplot(x = "Dependents", data = df, ax=axes[0][2])

sns.countplot(x = "PhoneService", data = df, ax=axes[0][3])

sns.countplot(x = "MultipleLines", data = df, ax=axes[0][4])

sns.countplot(x = "InternetService", data = df, ax=axes[1][0])

sns.countplot(x = "OnlineSecurity", data = df, ax=axes[1][1])

sns.countplot(x = "OnlineBackup", data = df, ax=axes[1][2])

sns.countplot(x = "DeviceProtection", data = df, ax=axes[1][3])

sns.countplot(x = "TechSupport", data = df, ax=axes[1][4])

sns.countplot(x = "StreamingTV", data = df, ax=axes[2][0])

sns.countplot(x = "StreamingMovies", data = df, ax=axes[2][1])

sns.countplot(x = "Contract", data = df, ax=axes[2][2])

sns.countplot(x = "PaperlessBilling", data = df, ax=axes[2][3])

ax = sns.countplot(x = "PaymentMethod", data = df, ax=axes[2][4])

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.show(fig)
# Converting Total Charges to a numerical data type

data['TotalCharges'] = pd.to_numeric(data.TotalCharges, errors='coerce')

data.isnull().sum()
data = data.dropna()

data.isnull().sum()
fig, (ax1, ax2, ax3) = plt.subplots(3)

sns.kdeplot(data["tenure"], shade=True, color="b",ax = ax1)

sns.kdeplot(data["MonthlyCharges"], shade=True, color="r", ax = ax2)

sns.kdeplot(data["TotalCharges"], shade=True, color="g", ax = ax3)

fig.tight_layout()

plt.show(fig)
data.info()
data=pd.get_dummies(data,drop_first=True)

data.head()
data.info()
X = data.drop(['Churn_Yes'],axis=1)

Y = data['Churn_Yes']

print(X.shape,'\n',Y.shape)
X = X.astype('float32')

Y = Y.astype('float32')
from sklearn.model_selection import train_test_split

X_train , X_test , Y_train , Y_test = train_test_split(X ,Y , test_size =.10 ,random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=0)

dtc.fit(X_train,Y_train)
y_pred = dtc.predict(X_test)

# Print the prediction accuracy

print (metrics.accuracy_score(Y_test, y_pred))

print(classification_report(Y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

rfc =RandomForestClassifier()

rfc.fit(X_train,Y_train)
y_pred = rfc.predict(X_test)

# Print the prediction accuracy

print (metrics.accuracy_score(Y_test, y_pred))

print(classification_report(Y_test, y_pred))
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

print (metrics.accuracy_score(Y_test, y_pred))

print(classification_report(Y_test, y_pred))
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train)

Y_test = to_categorical(Y_test)
Nx=X_train.shape[1:][0]

Ny=Y_train.shape[1:][0]

print(Nx,Ny)
from tensorflow.keras.layers import Input, Dense

from keras.models import Sequential



input_layer = Input(shape = X_train.shape[1:])

hidden_layer = Dense(10, activation = 'relu',)(input_layer)

hidden_layer = Dense(10, activation = 'relu',)(hidden_layer)

output_layer = Dense(2, activation = 'sigmoid')(hidden_layer)
from tensorflow.keras.models import Model

from tensorflow.keras import optimizers

model = Model(inputs=[input_layer], outputs=[output_layer])

model.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, Y_train, epochs=50)
from sklearn.metrics import confusion_matrix

loss, accuracy = model.evaluate(X_test, Y_test,verbose=0)  # Evaluate the model

print('Accuracy :%0.3f'%accuracy)
history.history.keys()

import matplotlib.pyplot as plt

plt.plot(range(len(history.history['acc'])), history.history['acc'],c='blue')

plt.plot(range(len(history.history['loss'])), history.history['loss'],c='red')

plt.show()
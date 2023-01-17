from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import collections
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv",sep=',')
data.head()
data.dtypes
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"],errors="coerce")
data.isnull().sum()
summary = data.describe(include=["O"])
summary
data.dropna(how="any",inplace=True)
data.isnull().sum()
data_cont = ['tenure','MonthlyCharges', 'TotalCharges']
data_cat = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies',
            'Contract', 'PaperlessBilling','PaymentMethod']
sns.catplot(x="Churn", kind="count", data=data,orient='h')
fig , ax = plt.subplots(4,4,figsize=(20,20))
sns.set(style="ticks", color_codes=True)
for axis,col in zip(ax.flat,data_cat):
    sns.countplot(x=data["Churn"],hue=data[col],ax=axis)
fig,ax = plt.subplots(1, 3, figsize=(20,5))
sns.boxplot(x=data["tenure"], orient="h", color="purple",ax=ax[0])
sns.boxplot(x=data["MonthlyCharges"], orient="h", color="olive",ax=ax[1])
sns.boxplot(x=data["TotalCharges"] , orient="h", color="green",ax=ax[2])
fig,ax = plt.subplots(1,3,figsize=(15,5))

sns.distplot(data["tenure"],bins=10,kde=False,color="purple",ax=ax[0])
sns.distplot(data["MonthlyCharges"],bins=10,kde=False,color="olive",ax=ax[1])
sns.distplot(data["TotalCharges"],bins=10,kde=False,color="green",ax=ax[2])
NumHistTenure = sns.FacetGrid(data,col="Churn",height=6,aspect=1)
NumHistTenure = NumHistTenure.map(plt.hist, "tenure",bins=20,color="purple")
NumHistMC = sns.FacetGrid(data,col="Churn",height=6,aspect=1)
NumHistMC = NumHistMC.map(plt.hist, "MonthlyCharges",bins=20,color="olive")
NumHistTC = sns.FacetGrid(data,col="Churn",height=6,aspect=1)
NumHistTC = NumHistTC.map(plt.hist, "TotalCharges",bins=20,color="green")
scaler = StandardScaler()
data_continuous = scaler.fit_transform(data[data_cont])
for cols in data_cat:
    data.loc[:,cols] = LabelEncoder().fit_transform(data.loc[:,cols])

onehotencoder = OneHotEncoder(sparse=False)
data_categorical = onehotencoder.fit_transform(data[data_cat])
features = np.concatenate([data_continuous, data_categorical], axis=1)

target = data.iloc[:,20:].values
target = LabelEncoder().fit_transform(target)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 0)
classifier = Sequential()
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_shape=(46,)))
classifier.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
classifier.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
classifier.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
print(classification_report(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm.T,square=True, annot=True, fmt='d', cbar=False,xticklabels=['No','Yes'],yticklabels=['No','Yes'] )
plt.xlabel('true label')
plt.ylabel('predicted label')
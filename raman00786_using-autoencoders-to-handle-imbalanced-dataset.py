import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv')
df.head()
df.isnull().sum() ## Checking for missing values
df.info()
df = df.drop(['RowNumber','CustomerId','Surname'],axis=1)
df.head()
cat_features = df.select_dtypes('object')

cat_features
df['CreditScoreByAge'] = df['CreditScore']/df['Age']
gender_dummies = pd.get_dummies(df['Gender'],drop_first=True)

gender_dummies.columns=['gender_male']

df = pd.concat([df,gender_dummies],axis=1)

df = df.drop('Gender',axis=1)
df['Geography'] = np.where(df['Geography']=='Spain',0,1)
### detecting outlier



outlier=[]



def detect_outlier(dataframe,feature):

    thresh = 3

    mean = np.mean(dataframe[feature])

    std = np.std(dataframe[feature])

    

    for i in dataframe[feature]:

        z_score = (i-mean)/std

        if z_score > thresh:

            outlier.append(i)

    return outlier
detect_outlier(df,'Age')
df['Age'] = np.where(df['Age'] >=71,71,df['Age']) ##Replacing outlier with border values
df.head()
df['Salary/Age'] = df['EstimatedSalary']/df['Age']
df['NumOfProducts'] = np.where(df['NumOfProducts']>1,1,0) ##Multiple products (>1)
cat_features =['Geography','NumOfProducts',

       'HasCrCard', 'IsActiveMember', 

       'gender_male']
cont_features = ['Age', 'Balance','EstimatedSalary', 'CreditScoreByAge', 'Salary/Age']
for feature in cat_features: 

    df[feature] = np.where(df[feature]==0,-1,df[feature]) ###To decorrelate them we will replace 0 with -1
df.head()
X = df.drop(['Exited','CreditScore'],axis=1).values

y = df['Exited'].values
X.shape
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)

scaled_X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import SGD
encoder = Sequential()

encoder.add(Dense(11,activation="relu"))

encoder.add(Dense(5,activation="relu"))

encoder.add(Dense(2,activation="relu"))
decoder = Sequential()

decoder.add(Dense(5,activation="relu"))

decoder.add(Dense(11,activation="relu"))
autoencoder= Sequential([encoder,decoder])

autoencoder.compile(loss="binary_crossentropy",optimizer="SGD",metrics=["accuracy"])
autoencoder.fit(scaled_X_train,y_train,epochs=40,validation_data=(scaled_X_test,y_test))
lower_layer = Sequential()

lower_layer.add(Dense(1,activation="sigmoid"))
final_model = Sequential([encoder,lower_layer])

final_model.compile(loss="binary_crossentropy",optimizer="SGD",metrics=["accuracy"])
final_model.fit(scaled_X_train,y_train,epochs=10,validation_data=(scaled_X_test,y_test))
losses= final_model.history.history
losses = pd.DataFrame(losses)
losses[["loss","val_loss"]].plot()
predictions = final_model.predict_classes(scaled_X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve,balanced_accuracy_score
print(roc_auc_score(y_test,predictions))

print(confusion_matrix(predictions,y_test))
print(accuracy_score(y_test,predictions))
print(balanced_accuracy_score(predictions,y_test))
print(classification_report(predictions,y_test))
fpr,tpr,threshold  =roc_curve(predictions,y_test)
sns.countplot(df["Exited"])
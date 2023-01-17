import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None); 

pd.set_option('display.max_rows', None);

data = pd.read_csv("/kaggle/input/churn-predictions-personal/Churn_Predictions.csv")

data.head()
print("Data Shape: " , data.shape)

print(data.isna().sum())
data.info()
data[data["Balance"]==0].groupby(["NumOfProducts"]).agg({"EstimatedSalary":"mean"})
df2=data[(data["NumOfProducts"]==4) & (data["Exited"]==1)]

df2
data["CreditScore"].plot.hist()
data["Age"].plot.hist()
data["EstimatedSalary"].plot.hist()

print("Minimum Salary ",data["EstimatedSalary"].min())

print("Average Salary ",data["EstimatedSalary"].median())

print("Maximum Salary ",data["EstimatedSalary"].max())
a=data["Geography"].nunique()

print(a,"Countries \n")

count=1

for i in data["Geography"].unique():

    print(count,".",i)

    count=count+1
print(data["Geography"].value_counts())
data["Gender"].value_counts()
data["NumOfProducts"].value_counts()
data["HasCrCard"].value_counts()
data["IsActiveMember"].value_counts()
data["IsActiveMember"].value_counts()
data.groupby(["Geography","Gender","Exited"]).agg({"Exited":"count"})
f, ax = plt.subplots(1, 2, figsize = (15, 7))

f.suptitle("Churn ?", fontsize = 18.)

_ = data.Exited.value_counts().plot.bar(ax = ax[0], rot = 0, color = (sns.color_palette()[0], 

sns.color_palette()[2])).set(xticklabels = ["No", "Yes"])



_ = data.Exited.value_counts().plot.pie(labels = ("No", "Yes"), autopct = "%.2f%%", label = "", fontsize = 13., ax = ax[1],\

colors = (sns.color_palette()[0], sns.color_palette()[2]), wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), 

ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")
data.drop(["CustomerId", "Surname"], axis = 1, inplace = True)

data.info()
# Correlation matrix graph of the data set

f, ax = plt.subplots(figsize= [15,10])

sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax )

ax.set_title("Correlation Matrix", fontsize=20)

plt.show()
data.groupby(["NumOfProducts","Exited"]).agg({"Exited":"count"})
cat_df = data[["Geography","Gender"]]

print(cat_df)
bool_df = data[["IsActiveMember","HasCrCard"]]
data.drop(["IsActiveMember","HasCrCard","Geography","Gender"], axis = 1, inplace = True)
cat_df = pd.get_dummies(cat_df, drop_first=True)



Y = data["Exited"]

X = data.drop(["Exited"], axis = 1)

cols = X.columns

index = X.index
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X = scaler.fit_transform(X)

X = pd.DataFrame(X,columns=cols,index=index)

X = pd.concat([X,bool_df,cat_df],axis=1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, 

                                                    test_size=0.20, 

                                                    random_state=12345)

#Dengesiz bir veri seti olduğu için örneklem sayısını arttıracağız

from imblearn.combine import SMOTETomek



smk = SMOTETomek()

X_train, y_train = smk.fit_sample(X_train, y_train)



X_test, y_test = smk.fit_sample(X_test, y_test)
from sklearn.linear_model import LogisticRegression  

from sklearn.neighbors import KNeighborsClassifier  

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score, GridSearchCV







models = []

models.append(('LR', LogisticRegression( random_state = 12345)))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier( random_state = 12345)))

models.append(('RF', RandomForestClassifier( random_state = 12345)))

models.append(('SVM', SVC(gamma='auto', random_state = 12345)))

models.append(('XGB', GradientBoostingClassifier( random_state = 12345)))

models.append(("LightGBM", LGBMClassifier( random_state = 12345)))



# evaluate each model in turn

results = []

names = []
for name, model in models:

    base = model.fit(X_train,y_train)

    y_pred = base.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)

    results.append(acc_score)

    names.append(name)

    msg = "%s: %f" % (name, acc_score)

    print(msg)
models2 = []

models2.append(('CART', DecisionTreeClassifier( random_state = 12345)))

models2.append(('RF', RandomForestClassifier( random_state = 12345)))

models2.append(('XGB', GradientBoostingClassifier( random_state = 12345)))

models2.append(("LightGBM", LGBMClassifier( random_state = 12345)))
for name, model in models2:

        base = model.fit(X_train,y_train)

        y_pred = base.predict(X_test)

        acc_score = accuracy_score(y_test, y_pred)

        feature_imp = pd.Series(base.feature_importances_,

                        index=X.columns).sort_values(ascending=False)



        sns.barplot(x=feature_imp, y=feature_imp.index)

        plt.xlabel('Değişken Önem Skorları')

        plt.ylabel('Değişkenler')

        plt.title(name)

        plt.show()
import tensorflow.keras

from keras.models import Sequential

from keras.layers import Dense
# Initiate the sequential model

model = Sequential()

# add input layers

model.add(Dense(units=25,activation="tanh"))

model.add(Dense(units=25,activation="tanh"))

model.add(Dense(units=1,activation="sigmoid"))
model.compile(optimizer="adam",loss="binary_crossentropy", metrics=['accuracy'])
epochs_hist= model.fit(X_train,y_train,epochs=100,batch_size=25)
loss_history = epochs_hist.history['loss']

acc_history = epochs_hist.history['accuracy']

epochs = [(i + 1) for i in range(100)]



ax = plt.subplot(211)

ax.plot(epochs, loss_history, color='red')

ax.set_xlabel('Epochs')

ax.set_ylabel('Error Rate\n')

ax.set_title('Error Rate per Epoch\n')



ax2 = plt.subplot(212)

ax2.plot(epochs, acc_history, color='blue')

ax2.set_xlabel('Epochs')

ax2.set_ylabel('Accuracy\n')

ax2.set_title('Accuracy per Epoch\n')



plt.subplots_adjust(hspace=0.8)

plt.show()
y_pred=model.predict(X_test)

y_pred
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix,accuracy_score



cm=confusion_matrix(y_pred,y_test)

print(cm)
accuracy_score(y_pred,y_test)
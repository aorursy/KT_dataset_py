import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
data = pd.read_csv("../input/creditcard.csv")

data.describe()
print(data.head())
corr = data.corr()

plot.figure(figsize=(30,30))
sns.heatmap(corr, annot=True)
corr1 = corr[corr>0.1]
plot.figure(figsize=(20,20))
sns.heatmap(corr1, annot=True)
from sklearn.decomposition import PCA

#dropping the solution
pca_data = data.drop("Class", 1)

pca = PCA(n_components=5)
pca.fit(pca_data)

pca_data = pd.DataFrame(pca.transform(pca_data))
print(pca_data.shape)
means = []
stds = []

for col in range(pca_data.shape[1]):
        mn = np.mean(pca_data.iloc[:,col])
        st = np.mean(pca_data.iloc[:,col].std())
        
        #storing statistical data for later
        means.append(mn)
        stds.append(st)
        
        pca_data.iloc[:,col] = (pca_data.iloc[:,col]-mn)/st
        pca_data.iloc[:,col] = np.nan_to_num(pca_data.iloc[:,col])
        
pca_data.describe()
test_ratio=0.2

#combining with solutions to keep order
new_data = pd.concat([pca_data, data["Class"]],1)

test_data = new_data.sample(frac=test_ratio)
train_data = new_data.drop(test_data.index)

test_sols = test_data["Class"]
test_data = test_data.drop("Class", 1)

train_sols = train_data["Class"]
train_data = train_data.drop("Class", 1)



#get dummies for better classification
train_sols = pd.get_dummies(train_sols, prefix="Class")
test_sols = pd.get_dummies(test_sols, prefix="Class")

print(new_data.shape)
print(train_data.shape)
print(test_data.shape)

print(train_sols.head())
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, LeakyReLU
from keras.callbacks import ModelCheckpoint

inp = Input(shape=(5,))
x = Dense(64)(inp)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(32)(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(8)(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(2, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
weight0 = 1.0/train_sols["Class_0"].sum()
weight1 = 1.0/train_sols["Class_1"].sum()

_sum = weight0+weight1

weight0 /= _sum
weight1 /= _sum

print(weight0)
print(weight1)
callback = [ModelCheckpoint("check.h5", save_best_only=True, monitor="val_acc", verbose=0)]
model.fit(train_data, train_sols, batch_size=500, epochs=100, verbose=0, callbacks=callback, validation_split=0.2, shuffle=True, 
         class_weight={0:weight0, 1: weight1})
best_model = load_model("check.h5")
score = model.evaluate(test_data, test_sols)
print(score[1])
test_data2 = pd.concat([test_data, test_sols],1)
test_data2 = test_data2.loc[test_data2["Class_1"]==1]

test_sols2 = test_data2[["Class_0", "Class_1"]]
test_data2 = test_data2.drop(["Class_0", "Class_1"],1)

score2 = model.evaluate(test_data2, test_sols2)
print(score2[1])
from sklearn.ensemble import GradientBoostingClassifier

xgmodel = GradientBoostingClassifier(n_estimators=200)

#weighting the samples
xgweight0 = train_sols["Class_0"].values*weight0
xgweight1 = train_sols["Class_1"].values*weight1

xgweights = xgweight0+xgweight1
print(xgweights.shape)
print(train_sols["Class_1"].shape)
#fit
xgmodel.fit(train_data, train_sols["Class_1"], xgweights)

xgscore = xgmodel.score(test_data, test_sols["Class_1"])
xgscore2 = xgmodel.score(test_data2, test_sols2["Class_1"])
print("Overall score:\t\t"+str(xgscore))
print("Rare case score:\t"+str(xgscore2))
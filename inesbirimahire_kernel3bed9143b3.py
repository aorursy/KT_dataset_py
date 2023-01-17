%matplotlib inline

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

import os

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.manifold import Isomap

from sklearn.preprocessing import normalize, Normalizer # data normalizers

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.ensemble import AdaBoostClassifier

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings("ignore")
# path to save figures

if not os.path.exists('./Figures'):

    os.mkdir('./Figures')
# reading the data using pandas routine

data = pd.read_csv('../input/Churn_Modelling.csv')



#printing 5 first rows od the data

data.head()
# general information about the data

data.info()
# Drop the irrelevant columns  as shown above

data = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

# One-Hot encoding our categorical attributes

"""As Gender will gibve us two columns we can drop male column and renaim the Gender_Female to Gender 

such that 1 corresponds to the female customer while 0 corresponds to male customer"""

cat_attr = ['Geography', 'Gender']

data = pd.get_dummies(data, columns = cat_attr, prefix = cat_attr)



data = data.drop(["Gender_Male"], axis = 1)

data.rename(columns={"Gender_Female": "Gender(1:F,0:M)"}, inplace=True)
data.head()
# five point descriptive statistics, and std

data.describe()
# Checking for unique value in the data attributes

data.nunique()
# Now lets check the class distributions

sns.countplot("Exited",data=data)

plt.title('Histogram')

plt.savefig('./Figures/Class distribution')

# Now lets check the class distributions

sns.countplot("Gender(1:F,0:M)",data=data)

plt.title('Histogram')

plt.savefig('./Figures/Gender distribution')

"""Data are unmbalanced from the histogram. let us chetch the class percentage"""

N,D = data.shape

no_exited_pct = np.sum(data.iloc[:,-5] == 0)/N

exited_pct = np.sum(data.iloc[:,-5] == 1)/N



print('No exited customer  Class precentage: {:2f}'.format(100*no_exited_pct))

print('Exited customer Class precentage: {:2f}'.format(100*exited_pct))





# group the data by the target variable 

data_grp= data.groupby('Exited')
# seeing the count per Exited customers data

data_grp.count()
# general describtive per Exited customers data

data_grp.describe()
# general data correlation

data.corr()


# general data correlation heatmap

sns.heatmap(data.corr(),cmap='YlGnBu')

plt.title('Correlation')

plt.savefig('./Figures/corr_matrix_plot')
#arranging features with are highly correlated with the target (ascending order)

cor = data.corr()

corr_t = (cor ["Exited"]).abs()



print("The features which are most correlated with Exited feature:\n", corr_t.sort_values(ascending=False)[1:14].index)

#plotting all features 

#sns.pairplot(data,hue='Exited',palette="dark")

sns.pairplot(data,palette="dark")

plt.title('Pair plot')

plt.legend(['Not churn', 'churn'])

plt.savefig('./Figures/All in one plot')
#  churned customers class distributions on gender

churn     = data[data["Exited"] == 1]

not_churn = data[data["Exited"] == 0]

sns.countplot("Gender(1:F,0:M)",data=churn)

plt.title('Histogram_churn_Customers')

plt.savefig('./Figures/Gender_churn distribution')

#  not churned customers class distributions on gender

churn     = data[data["Exited"] == 1]

not_churn = data[data["Exited"] == 0]

sns.countplot("Gender(1:F,0:M)",data=not_churn)

plt.title('Histogram_churn_Customers')

plt.savefig('./Figures/Gender_not_churn distribution')

# dimentionality reduction



pca = PCA(2) # to get the independent representation

tsne = TSNE(2) # state of the art / the data is nonlinear

iso = Isomap() # follwing the assumption that the data lives in nonlinear manifold.
# fit the dimensionality reduction algorithm 



pca_data = pca.fit_transform(normalize(data))

tsne_data = tsne.fit_transform(normalize(data))

iso_data = iso.fit_transform(normalize(data))
# ploting 

f, (ax1, ax2, ax3) = plt.subplots(3, 1,)





ax1.set_title('PCA')

ax1.scatter(pca_data[:,0],pca_data[:,1], c = data['Exited'])

plt.title('PCA plot')





ax2.set_title('TSNE')

ax2.scatter(tsne_data[:,0],tsne_data[:,1], c = data['Exited'])

plt.title('TSNE')





ax3.set_title('Isomap')

ax3.scatter(iso_data[:,0],iso_data[:,1], c = data['Exited'])

plt.title('ISOMAP')

plt.savefig('./Figures/PCA,TSNE,ISOMAP')
from sklearn.model_selection import train_test_split



X = data.drop(["Exited"],axis=1)

y = data.Exited

train_data,test_data, target_train, target_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data.head()
print(len(train_data))

print(len(test_data))



np.random.seed(6)

X=normalize(train_data)

y=target_train

# search for an optimal value of K for KNN with cross validation

# range of k we want to try

k_range = range(2, 39)

# empty list to store scores

k_scores = []

# 1. we will loop through reasonable values of k

for k in k_range:

    # 2. run KNeighborsClassifier with k neighbours

    knn = KNeighborsClassifier(n_neighbors=k)

    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours

    scores = cross_val_score(knn, X, y, cv = 10, scoring='accuracy')

    #scores = cross_val_score(knn, inputs, label, cv = 10, scoring='accuracy')

    # 4. append mean of scores for k neighbors to k_scores list

    k_scores.append(scores.mean())



print("k different averages:", k_scores)

print("\nMax average, index of Max:", max(k_scores),"||", k_scores.index(max(k_scores)))

pos = k_scores.index(max(k_scores))
# plot how accuracy changes as we vary k

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-validated accuracy')

plt.title('Cross-Validate Accuracy Vs K')

plt.savefig('./Figures/Cross-Validate Accuracy Vs K')


model = KNeighborsClassifier(3)

print("\t %%% Now lets fit our model first and then we test with our test set %%%")

model = model.fit(X,y)

print("\n Training score: ",model.score(X, y)) 

pred = model.predict(normalize(test_data))

score = metrics.accuracy_score(pred, target_test)

print("\nThe accuracy score that we get is: ",score)  

print("\n Confusion Matrix: ", confusion_matrix(target_test, pred))

print(metrics.classification_report(target_test, pred))


ETC = ExtraTreesClassifier(random_state=5)

ETC = ETC.fit(X, y)

print("\n Training score: ",ETC.score(X, y)) #evaluating the training error

pred = ETC.predict(normalize(test_data))

score = metrics.accuracy_score(pred,target_test)

print("\nThe accuracy score that we get is: ",score)

print("\n Confusion Matrix: ", confusion_matrix(target_test, pred))

print(metrics.classification_report(target_test, pred))
input_train, label_input = X, y

input_test, label_test = normalize(test_data), target_test



RF = RandomForestClassifier(max_depth=11, random_state=5)

RF = RF.fit(input_train, label_input)

print("\n Training score: ",RF.score(input_train, label_input)) #evaluating the training error

pred = RF.predict(input_test)

score = metrics.accuracy_score(pred,label_test)

print("\nThe accuracy score that we get is: ",score)

print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))

print(metrics.classification_report(label_test, pred))
# getting featrure importance

RF.feature_importances_


dc = DummyClassifier(strategy="uniform",random_state=5)

dc = dc.fit(input_train, label_input)

print("\n Training score: ",dc.score(X, y)) #evaluating the training error

pred = dc.predict(input_test)

score = metrics.accuracy_score(pred,label_test)

print("\nThe accuracy score that we get is: ",score)

print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))

print(metrics.classification_report(label_test, pred))



ABC = AdaBoostClassifier(random_state=5)

ABC = ABC.fit(input_train, label_input)

print("\n Training score: ",ABC.score(input_train, label_input)) #evaluating the training error

pred = ABC.predict(input_test)

score = metrics.accuracy_score(pred,label_test)

print("\nThe accuracy score that we get is: ",score)

print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))

print(metrics.classification_report(label_test, pred))


GBC = GradientBoostingClassifier(random_state=5)

GBC = GBC.fit(input_train, label_input)

print("\n Training score: ",GBC.score(input_train, label_input)) #evaluating the training error

pred = GBC.predict(input_test)

score = metrics.accuracy_score(pred,label_test)

print("\nThe accuracy score that we get is: ",score)

print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))

print(metrics.classification_report(label_test, pred))


dt = tree.DecisionTreeClassifier(random_state=5)

dt = dt.fit(input_train, label_input)

print("\n Training score: ",dt.score(input_train, label_input)) #evaluating the training error

pred = dt.predict(input_test)

score = metrics.accuracy_score(pred,label_test)

print("\nThe accuracy score that we get is: ",score)

print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))

print(metrics.classification_report(label_test, pred))


sgdc = SGDClassifier(loss="hinge", penalty="l2", max_iter=25,random_state=5)

sgdc = sgdc.fit(input_train, label_input)

print("\n Training score: ",sgdc.score(input_train, label_input)) #evaluating the training error

pred = sgdc.predict(input_test)

score = metrics.accuracy_score(pred,label_test)

print("\nThe accuracy score that we get is: ",score)

print("\n Confusion Matrix: ", confusion_matrix(label_test, pred))

print(metrics.classification_report(label_test, pred))
import tensorflow as tf

from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler


#Scaling data

scaler = MinMaxScaler()

scaler.fit(X)

dfx = scaler.transform(X)

scaler.fit(test_data)

dfx_test = scaler.transform(test_data)
#Neural network



model0 = keras.Sequential([

    keras.layers.Flatten(input_shape=dfx.shape[1:]),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(1, activation=tf.nn.sigmoid)

])





model0.compile(optimizer='adam', 

              loss='binary_crossentropy',

              metrics=['accuracy']);





model0.fit(dfx, y, epochs=100);
test_loss, test_acc = model0.evaluate(dfx_test, target_test)



print('Test accuracy:', test_acc)





y_hat = model0.predict_classes(dfx_test)



print(metrics.classification_report(target_test, y_hat))
# helper ploting function 



def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()

from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy = 'auto', kind = 'regular',random_state=5)



data1 = data.drop(["Exited"],axis=1)

inputs,label = sm.fit_sample(data1, data['Exited'])

print("Original dataset: ",data['Exited'].value_counts())





compt = 0

for i in range(len(label)):

    if label[i]==1:

        compt += 1

print("\nNumber of 1 in the new  dataset: ",compt)
plot_2d_space(inputs, label, 'balanced data scatter plot')

plt.savefig("Scatter plot balanced data")



Xc = inputs

yc = label

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
print('-----------------Gradient Boost Classifier--------------------')

GBC = GradientBoostingClassifier(random_state=5)

GBC = GBC.fit(Xc_train, yc_train)

pred = GBC.predict(Xc_test)

score = metrics.accuracy_score(pred,(yc_test))

print("Accuracy---", accuracy_score(yc_test,pred))

print(metrics.classification_report(yc_test, pred))





print('-----------------Ada Boost Classifier--------------------')

ABC = AdaBoostClassifier(random_state=5)

ABC = ABC.fit(Xc_train, yc_train)

pred = ABC.predict(Xc_test)

print("Accuracy---", accuracy_score(yc_test,pred))

print(classification_report(yc_test,pred))





print('-----------------RandomForestClassifier--------------------')

model  = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0).fit(Xc_train, yc_train)

pred = model.predict(Xc_test)

print("Accuracy---", accuracy_score(yc_test,pred))

print(classification_report(yc_test,pred))



#Try the neural network also

#Scaling data

scaler = MinMaxScaler()

scaler.fit(Xc_train)

dfx = scaler.transform(Xc_train)

scaler.fit(Xc_test)

dfx_test = scaler.transform(Xc_test)
model0.fit(dfx, yc_train, epochs=100);

test_loss, test_acc = model0.evaluate(dfx_test, yc_test)



print('Test accuracy:', test_acc)

y_hat = model0.predict_classes(dfx_test)



print(metrics.classification_report(yc_test, y_hat))
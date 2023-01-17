import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import preprocessing

import seaborn as sns

from sklearn import preprocessing

data = pd.read_csv('../input/datat3rr/DataT3RR.csv')

print('Size of weather data frame is :',data.shape)

data.info()

data[0:10]
data.count().sort_values()
data = data.drop(columns=['Stasiun','Tanggal'],



                 axis=1)

data = data.dropna(how='any')

print(data.shape)
print(data.shape)

data.head()
cnt_pro = data['Besok_hujan'].value_counts()

plt.figure(figsize=(6,4))

sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Kelas', fontsize=12)

plt.xticks(rotation=90)

plt.show();
sns.set_style("whitegrid")

sns.pairplot(data,hue="Besok_hujan",size=3);

plt.show()
# Calculate the correlation matrix

corr = data.corr()

corr1 = pd.DataFrame(abs(corr['Besok_hujan']),columns = ['Besok_hujan','Tn','Tx','Tavg'])

nonvals = corr1.loc[corr1['Besok_hujan'] < 0.005]

print('Var correlation < 0.5%',nonvals)

nonvals = list(nonvals.index.values)



# We extract variables with correlation less than 0.5%

data1 = data.drop(columns=nonvals,axis=1)

print('Data Final',data1.shape)
data = data[['Tn','Tx','Tavg','RR','Hari_hujan','Besok_hujan']] #Subsetting the data

cor = data.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
from sklearn.model_selection import train_test_split

Y = data1['Besok_hujan']

X = data1.drop(columns=['Besok_hujan'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=9)
print('X train shape: ', X_train.shape)

print('Y train shape: ', Y_train.shape)

print('X test shape: ', X_test.shape)

print('Y test shape: ', Y_test.shape)
from sklearn.neighbors import KNeighborsClassifier



# We define the model

knncla = KNeighborsClassifier(n_neighbors=5,n_jobs=-1)



# We train model

knncla.fit(X_train, Y_train)



# We predict target values

Y_predict6 = knncla.predict(X_test)
# The confusion matrix

from sklearn.metrics import confusion_matrix

import seaborn as sns



knncla_cm = confusion_matrix(Y_test, Y_predict6)

f, ax = plt.subplots(figsize=(5,5))

sns.heatmap(knncla_cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")

plt.title('KNN Classification Confusion Matrix')

plt.xlabel('Y predict')

plt.ylabel('Y test')

plt.show()
test_acc_knncla = round(knncla.fit(X_train,Y_train).score(X_test, Y_test)* 100, 2)

train_acc_knncla = round(knncla.fit(X_train, Y_train).score(X_train, Y_train)* 100, 2)
#Accuracy

model1 = pd.DataFrame({

    'Model': ['KNN'],

    'Train Score': [train_acc_knncla],

    'Test Score': [test_acc_knncla]

})

model1.sort_values(by='Test Score', ascending=False)
#Precision, Recall

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y_test, Y_predict6)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import matplotlib.pyplot as plt



disp = plot_precision_recall_curve(knncla,X_train, Y_train)

disp.ax_.set_title('2-class Precision-Recall curve: '

                   'AP={0:0.2f}'.format(average_precision))
Y1 = data['Besok_hujan']

X1 = data.drop(columns=['Besok_hujan','Hari_hujan'])



from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel



lsvc = LinearSVC(C=0.05, penalty="l1", dual=False,random_state=9).fit(X1, Y1)

model = SelectFromModel(lsvc, prefit=True)

X_new = model.transform(X1)

cc = list(X1.columns[model.get_support(indices=True)])

print(cc)

print(len(cc))
# Principal component analysis

from sklearn.decomposition import PCA



pca = PCA().fit(X1)

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Hujan_besok, Tn, Tx, RR')

plt.ylabel('% Variance Explained')

plt.title('PCA Analysis')

plt.grid(True)

plt.show()
# Percentage of total variance explained

variance = pd.Series(list(np.cumsum(pca.explained_variance_ratio_)), 

                        index= list(range(1, 5))) 

print(variance[30:70])
X1 = data[cc] 

from sklearn.model_selection import train_test_split

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.3, random_state=9)
# K-Nearest Neighbor classification

knncla.fit(X1_train, Y1_train)

Y1_predict6 = knncla.predict(X1_test)

knncla_cm = confusion_matrix(Y1_test, Y1_predict6)

score1_knncla= knncla.score(X1_test, Y1_test)
fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3, 3, 1) 

ax1.set_title('KNN Classification')

sns.heatmap(data=knncla_cm, annot=True, linewidth=0.7, linecolor='cyan',cmap="BuPu" ,fmt='g', ax=ax1)

plt.show()
Testscores1 = pd.Series([score1_knncla], index=[ 'K-Nearest Neighbour Score']) 

print(Testscores1)
#Precision, Recall

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(Y1_test, Y1_predict6)



print('Average precision-recall score: {0:0.2f}'.format(

      average_precision))
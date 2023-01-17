from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

low_memory=False

df1 = pd.read_csv("/kaggle/input/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

df2=pd.read_csv("/kaggle/input/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

df3=pd.read_csv("/kaggle/input/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv")

df4=pd.read_csv("/kaggle/input/MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")

df5=pd.read_csv("/kaggle/input/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")

df6=pd.read_csv("/kaggle/input/MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

df7=pd.read_csv("/kaggle/input/MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv")

df8=pd.read_csv("/kaggle/input/MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv")



df1.head()
df2.head()
df3.head()
df4.head()
df5.head()
df6.head()
df7.head()
df8.head()
nRowsRead = None # Čita sve redove, moguće je pročitati samo određeni broj





df = pd.concat([df1,df2])

del df1,df2

df = pd.concat([df,df3])

del df3

df = pd.concat([df,df4])

del df4

df = pd.concat([df,df5])

del df5

df = pd.concat([df,df6])

del df6

df = pd.concat([df,df7])

df = pd.concat([df,df8])





df_model = pd.concat([df7,df8]) #utorak i sreda

del df7

del df8

nRow, nCol = df.shape

print(f'U tabeli ima {nRow} redova i {nCol} kolona')
df.head(5)
df_model.head()
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

plotPerColumnDistribution(df, 10, 5)
df[' Label'].value_counts()
#Podela skupa podataka na podatke za treniranje i testiranje

from sklearn.model_selection import train_test_split

train, test=train_test_split(df_model,test_size=0.25, random_state=10)





train.describe()

test.describe()
#Skaliranje numeričkih atributa

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



#Izvlačenje numeričkih atribute i skaliranje tako da imaju nultu sredinu i jedinicu varijanse 

cols = train.select_dtypes(include=['float64','int64']).columns

sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))

sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))



#Vraćanje rezultata natrag u okvir podataka

sc_traindf = pd.DataFrame(sc_train, columns = cols)

sc_testdf = pd.DataFrame(sc_test, columns = cols)
from sklearn.preprocessing import OneHotEncoder 



# Kreiranje one hot encoder objekta 

onehotencoder = OneHotEncoder() 



trainDep = train[' Label'].values.reshape(-1,1)

trainDep = onehotencoder.fit_transform(trainDep).toarray()

testDep = test[' Label'].values.reshape(-1,1)

testDep = onehotencoder.fit_transform(testDep).toarray()
train_X=sc_traindf

train_y=trainDep[:,0]



test_X=sc_testdf

test_y=testDep[:,0]
#Eliminacija rekurzivnih karakteristika

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFE

import itertools



rfc = RandomForestClassifier()





rfe = RFE(rfc, n_features_to_select=20)

rfe = rfe.fit(train_X, train_y)





feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_X.columns)]

selected_features = [v for i, v in feature_map if i==True]



selected_features



a = [i[0] for i in feature_map]

train_X = train_X.iloc[:,a]

test_X = test_X.iloc[:,a]
#Podela podataka

X_train,X_test,Y_train,Y_test = train_test_split(train_X,train_y,train_size=0.75, random_state=2)



#Fitting Models

from sklearn.svm import SVC

from sklearn.naive_bayes import BernoulliNB 

from sklearn import tree

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



# Treniranje KNeighborsClassifier modela

KNN_Classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=3, metric='minkowski', algorithm='auto') 

#n_jobs=-1 znaci da koristi sve procesore,  algorithm='auto' sam primenjuje odgovarajuci algoritam na osnovu prosleđenih vrednosti, n_neighbors=3 uzima u obzir 3 najbliza suseda, minkowski udaljenost

KNN_Classifier.fit(X_train, Y_train); 



# Treniranje Naive Baye modela

BNB_Classifier = BernoulliNB()

BNB_Classifier.fit(X_train, Y_train)



# Treniranje Decision Tree modela

DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)

#Funkcija za merenje kvaliteta podele podrzava kriterijume „gini“, za nečistoću i „entropy“ za dobijanje informacija,

DTC_Classifier.fit(X_train, Y_train)
#Evaluacija modela

from sklearn import metrics



models = []

models.append(('Naive Baye Classifier', BNB_Classifier))

models.append(('Decision Tree Classifier', DTC_Classifier))

models.append(('KNeighborsClassifier', KNN_Classifier))



for i, v in models:

    scores = cross_val_score(v, X_train, Y_train, cv=10)

    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))

    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))

    classification = metrics.classification_report(Y_train, v.predict(X_train))

    print()

    print('============================== {} Model Evaluation =============================='.format(i))

    print()

    print ("Cross Validation Mean Score:" "\n", scores.mean())

    print()

    print ("Model Accuracy:" "\n", accuracy)

    print()

    print("Confusion matrix:" "\n", confusion_matrix)

    print()

    print("Classification report:" "\n", classification) 

    print()
#Rezultat

for i, v in models:

    

    accuracy = metrics.accuracy_score(Y_test, v.predict(X_test))

    confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(X_test))

    classification = metrics.classification_report(Y_test, v.predict(X_test))

    print()

    print('============================== {} Model Test Results =============================='.format(i))

    print()

    print ("Model Accuracy:" "\n", accuracy)

    print()

    print("Confusion matrix:" "\n", confusion_matrix)

    print()

    print("Classification report:" "\n", classification) 

    print()        
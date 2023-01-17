import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
data.isnull().sum()
data.describe()
print('No Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')

print('Fraudulent Transactions', data['Class'].value_counts()[1])
import seaborn as sns



colors = ["#0101DF", "#DF0101"]



sns.countplot('Class', data=data, palette=colors)
from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))



data.drop(['Time','Amount'], axis=1, inplace=True)



amount = data['scaled_amount']

time = data['scaled_time']



data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

data.insert(0, 'amount', amount)

data.insert(1, 'time', time)
data.head()
data = data.sample(frac=1)



fraud_data = data.loc[data['Class']==1]

nfraud_data = data.loc[data['Class']==0][:492]



normal_distributed_df = pd.concat([fraud_data, nfraud_data])



# Shuffle dataframe rows

ndata = normal_distributed_df.sample(frac=1, random_state=42)



ndata.head()
print('Fraudulent Transactions', data['Class'].value_counts()[1])

print('Non-Fraudulent Transactions', data['Class'].value_counts()[0])

colors = ["#0101DF", "#DF0101"]



sns.countplot('Class', data=ndata, palette=colors)
X= data.iloc[:, data.columns != 'Class']

y = data.iloc[:, data.columns == 'Class']
X.corrwith(ndata.Class).plot.bar(figsize= (20,10),title="Corr", fontsize=10, grid=True)
plt.figure(figsize = (20,10))

sns.heatmap(ndata.corr(), annot=True,cmap="YlGnBu")
dataframe = pd.DataFrame(data=ndata)
dataframe
X= dataframe.iloc[:, ndata.columns != 'Class']

y = dataframe.iloc[:, ndata.columns == 'Class']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=1)
X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'gini', random_state= 0 )

classifier.fit(X_train, y_train.ravel())
y_pre = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

cm_grid = confusion_matrix(y_test,y_pre)

sns.heatmap(cm_grid, annot=True)
print(classification_report(y_test,y_pre))
classifier.score(X_test,y_test)
from sklearn import tree

text_representation = tree.export_text(classifier)

print(text_representation)
with open("decistion_tree.log", "w") as fout:

    fout.write(text_representation)
fig = plt.figure(figsize=(100,40))

_ = tree.plot_tree(classifier, filled=True)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators= 10, criterion= 'entropy', random_state=0)

clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

cm_grid = confusion_matrix(y_test,y_pred)

sns.heatmap(cm_grid, annot=True)
print(classification_report(y_test,y_pred))
clf.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

clsf= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

clsf.fit(X_train, y_train.ravel())
y_pr = clsf.predict(X_test)

cm_grid = confusion_matrix(y_test,y_pr)

sns.heatmap(cm_grid, annot=True)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout





#Initializing ANN

classifier = Sequential()





#Input Layer

classifier.add(Dense(30, activation='relu'))

#2nd layer

classifier.add(Dense(16, activation='relu'))

classifier.add(Dense(16, activation='relu'))



#Output layer

classifier.add(Dense(1, activation='sigmoid'))



#Compling the ANN

classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])



#Fitting the dataset into ANN

classifier.fit(X_train, y_train, batch_size=100, epochs=100)





#Predicting the results

y_pred = classifier.predict(X_test)













classifier.summary()
import pandas as pd

from pylab import rcParams

import matplotlib.pyplot as plt





def plot_densities(data):

    '''

    Plot features densities depending on the outcome values

    '''

    # change fig size to fit all subplots beautifully 

    rcParams['figure.figsize'] = 15, 60



    # separate data based on outcome values 

    outcome_0 = data[data['Class'] == 0]

    outcome_1 = data[data['Class'] == 1]



    # init figure

    fig, axs = plt.subplots(30, 1)

    fig.suptitle('Features densities for different outcomes 0/1')

    plt.subplots_adjust(left = 0.25, right = 0.9, bottom = 0.1, top = 0.95,

                        wspace = 0.2, hspace = 0.9)



    # plot densities for outcomes

    for column_name in names[:-1]: 

        ax = axs[names.index(column_name)]

        #plt.subplot(4, 2, names.index(column_name) + 1)

        outcome_0[column_name].plot(kind='density', ax=ax, subplots=True, 

                                    sharex=False, color="red", legend=True,

                                    label=column_name + ' for Outcome = 0')

        outcome_1[column_name].plot(kind='density', ax=ax, subplots=True, 

                                     sharex=False, color="green", legend=True,

                                     label=column_name + ' for Outcome = 1')

        ax.set_xlabel(column_name + ' values')

        ax.set_title(column_name + ' density')

        ax.grid('on')

    plt.show()

    fig.savefig('densities.png')



# load your data 

#data  = pd.read_csv('diabetes.csv')

names = list(data.columns)



# plot correlation & densities

plot_densities(ndata)
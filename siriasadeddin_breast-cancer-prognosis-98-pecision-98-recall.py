import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from sklearn import metrics
df=pd.read_csv('../input/breast-cancer-wisconsin-prognostic-data-set/data 2.csv')
df.T
columns=df.columns

columns_new=[]

for i in columns:

    columns_new.append(any(df[i].isnull()|df[i].isnull()))

df=df.drop(columns[columns_new],axis=1)
{'unique patients':len(df.id.unique()), 'records':len(df.id)}
ax = sns.countplot(df.diagnosis,label="Count")       # M = 212, B = 357

df.diagnosis.value_counts()
train_features, test_features, train_labels, test_labels=train_test_split(

    df.drop(['id','diagnosis'], axis=1),

    df[['diagnosis']],

    test_size=0.3,

    random_state=41)
corrMatrix = train_features.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corrMatrix, annot=True,ax=ax)

plt.show()
correlated_features = set()

for i in range(len(corrMatrix .columns)):

    for j in range(i):

        if abs(corrMatrix.iloc[i, j]) > 0.85:

            colname = corrMatrix.columns[i]

            correlated_features.add(colname)

print(correlated_features)
train_features.drop(labels=correlated_features, axis=1, inplace=True)

test_features.drop(labels=correlated_features, axis=1, inplace=True)
corrMatrix = train_features.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corrMatrix, annot=True,ax=ax)

plt.show()
ax = sns.boxplot( palette="Set2", orient="h",data=train_features)
constant_filter = VarianceThreshold(threshold=0.0)

constant_filter.fit(train_features)

train_features = constant_filter.transform(train_features)

test_features = constant_filter.transform(test_features)



train_features.shape, test_features.shape
mm_scaler = preprocessing.StandardScaler()

train_features = pd.DataFrame(mm_scaler.fit_transform(train_features))

test_features=pd.DataFrame(mm_scaler.transform(test_features))
X = train_features

y = train_labels.replace({'B':0,'M':1})

# define the keras model

model = Sequential()

model.add(Dense(12, input_dim=train_features.shape[1], activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

model.fit(X, y, epochs=50, batch_size=8)

# evaluate the keras model

_, accuracy = model.evaluate(X, y)
# make class predictions with the model

y_pred = model.predict_classes(test_features)
def conf_matrix(matrix,pred):

    class_names= [0,1]# name  of classes

    fig, ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names)

    plt.yticks(tick_marks, class_names)

    # create heatmap

    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')

    ax.xaxis.set_label_position("top")

    plt.tight_layout()

    plt.title('Confusion matrix', y=1.1)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    plt.show()
cnf_matrix = metrics.confusion_matrix(y_pred,test_labels.replace({'B':0,'M':1}),normalize='true')

conf_matrix(cnf_matrix,test_labels)
# calculate prediction

report = classification_report(y_pred,test_labels.replace({'B':0,'M':1}))

print(report)
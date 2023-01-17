import os # accessing directory structure

from mpl_toolkits.mplot3d import Axes3D



from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing



import matplotlib.pyplot as plt # plotting



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix

from numpy.random import seed

from tensorflow import set_random_seed



set_random_seed(2)

seed(1)
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
df = pd.read_csv('../input/data.csv', delimiter=',')

nRow, nCol = df.shape # how many rows and columns do we have

df1 = df[0:761]

df2 = df[761:]

c = list(df2)

for x in range(12):

    c[x+1] = c[x+3]

df2.columns = c

df2 = df2.iloc[:, :-2]

df2.rename(columns={'campaign_id': 'reporting_start','fb_campaign_id': 'reporting_end'}, inplace=True)

df2.insert(3, 'campaign_id',np.NaN)

df2.insert(4,'fb_campaign_id',np.NaN)

df = df1.append(df2, ignore_index=True) # final dataframe

df.drop(columns=['reporting_start','reporting_end','total_conversion','ad_id','campaign_id','fb_campaign_id'],inplace=True)

#lets encode labels and normalized values for ranges

le_gender = preprocessing.LabelEncoder()

le_ages = preprocessing.LabelEncoder()

impressions_normalized = preprocessing.minmax_scale(df.impressions)

spent_normalized = preprocessing.minmax_scale(df.spent)

gender_encoded = le_gender.fit_transform(df.gender)

ages_one_hot_encoding_df = pd.get_dummies(df.age,prefix="Ages")

df = pd.concat([ages_one_hot_encoding_df,df], axis=1)

df.drop(columns='age',inplace=True)

#replace columns with new columns

df['gender'], df['impressions'], df['spent'] = gender_encoded, impressions_normalized, spent_normalized

print(df.approved_conversion.value_counts())

df.loc[df.approved_conversion >= 1,'approved_conversion'] = 1 

print(df.approved_conversion.value_counts())

# make sure everything is numerical Cause computers dont understand english yet

df = df.apply(pd.to_numeric)

data = df.values

df = df.sample(frac=1)

df.dataframeName = 'data.csv'

print("Dataframe: {}".format(df.dataframeName))
max_interest_val = np.maximum(np.maximum(df.interest1.max(),df.interest2.max()),df.interest3.max())

min_interest_val = np.minimum(np.minimum(df.interest1.max(),df.interest2.max()),df.interest3.max())

plotCorrelationMatrix(df,8)
axs = scatter_matrix(df, alpha=0.2,diagonal='kde',figsize  = [15, 15])

n = len(df.columns)

for x in range(n):

    for y in range(n):

        ax = axs[x, y]

        ax.xaxis.label.set_rotation(90)

        ax.yaxis.label.set_rotation(0)

        ax.yaxis.labelpad = 50
from sklearn.model_selection import train_test_split

data = np.asarray(df.values).astype(float)

X = data[:,:11]

Y = data[:,11]

x_train,x_test,y_train, y_test = train_test_split(X,Y,shuffle=True)

print("INPUT_SHAPE:",x_train.shape)

print("OUTPUT_SHAPE:",y_train.shape)
# Simple Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import classification_report

logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

target_names = ['Not Approved', 'Approved']

print(classification_report(y_test,predictions, target_names=target_names))
logmodel = LogisticRegressionCV()

logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

target_names = ['Not Approved', 'Approved']
print(classification_report(y_test,predictions, target_names=target_names))
from sklearn.svm import SVC

clf = SVC(gamma='auto',probability=True)

clf.fit(x_train,y_train) 

predictions = clf.predict(x_test)

print(classification_report(y_test,predictions, target_names=target_names))
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=1,)

clf.fit(x_train,y_train)

predictions  = clf.predict(x_test)

print(classification_report(y_test,predictions, target_names=target_names))
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.naive_bayes import GaussianNB
clf = BernoulliNB()

clf.fit(x_train,y_train)

predictions  = clf.predict(x_test)

print(classification_report(y_test,predictions, target_names=target_names))
clf = MultinomialNB()

clf.fit(x_train,y_train)

predictions  = clf.predict(x_test)

print(classification_report(y_test,predictions, target_names=target_names))
clf = GaussianNB()

clf.fit(x_train,y_train)

predictions  = clf.predict(x_test)

print(classification_report(y_test,predictions, target_names=target_names))
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras import backend as K
input_dimensions = X.shape[1]



# Set up Model Topology

model = Sequential()

model.add(Dense(32,input_dim=input_dimensions))

model.add(Activation('relu'))

model.add(Dense(64,input_dim=32))

model.add(Activation('relu'))

model.add(Dense(128,input_dim=64))

model.add(Activation('relu'))

model.add(Dense(256,input_dim=128))

model.add(Activation('relu'))

model.add(Dense(128,input_dim=256))

model.add(Activation('relu'))

model.add(Dense(64,input_dim=128))

model.add(Activation('relu'))

model.add(Dense(32,input_dim=64))

model.add(Activation('relu'))

model.add(Dense(16,input_dim=32))

model.add(Activation('relu'))

model.add(Dense(1,input_dim=16))

model.add(Activation('sigmoid'))



# Set up Learning Processes

model.compile(optimizer='rmsprop',

              loss='binary_crossentropy', # we have a single output

              metrics=['accuracy'])



#Train the model (finally)

model.fit(x_train,y_train,epochs=75)

predictions = np.round(model.predict(x_test))

print(classification_report(y_test,predictions))

score = model.evaluate(x_test, y_test)
# 75 epochs

predictions = np.round(model.predict(x_test)) # either 0 or 1 since our training data is like that

print(classification_report(y_test,predictions,target_names=target_names))
#87 epochs

print(classification_report(y_test,predictions,target_names=target_names))
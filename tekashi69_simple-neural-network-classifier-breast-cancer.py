import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn import preprocessing



#input training data

training = pd.read_csv('../input/data.csv')

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 10000)



#drop unneccessary columns

columns2drop = ['id','diagnosis','Unnamed: 32']



def check_all(df):

    print(df.describe())

    check_na(df)

    plot_hist(df)



def plot_hist(df):

    df.plot.hist(bins=10)

    plt.show()

    plt.close()



def check_na(df):

    print("there are NA values --> " + str(df.isnull().values.any()))



def corr_matrix(df, size=10):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr, cmap='hot')

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.show()

    plt.close()



def binn(df,num_bins):

    index = (df.max() - df.min())/num_bins

    bin = [0]

    counter_index = df.min()

    x = 0

    while x < num_bins+1:

        bin.append(counter_index)

        counter_index = counter_index + index

        x=x+1

    bin.append(1000000000)



    group_names = []

    y=0

    while y<len(bin)-1:

        group_names.append(y)

        y = y+1



    return bin, group_names



training.diagnosis = pd.Categorical(training.diagnosis)

training["Cat_Diagnosis"] = training.diagnosis.cat.codes



#last line before training

training = training.drop(columns2drop, axis=1)



y = training.Cat_Diagnosis

x = training.drop('Cat_Diagnosis', axis=1)



percentage_test = 0.1

break_point_column = int(len(training) * percentage_test)

X_test = x.iloc[0:break_point_column]

X_train = x.iloc[break_point_column:]

y_test = y.iloc[0:break_point_column]

y_train = y.iloc[break_point_column:]



from sklearn import preprocessing

X_test = X_test.values

X_train = X_train.values

min_max_optimization = preprocessing.MinMaxScaler()

X_test = min_max_optimization.fit_transform(X_test)

X_train = min_max_optimization.fit_transform(X_train)



#build model

from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(12, input_dim=len(training.columns)-1, activation='relu'))

model.add(Dense(47, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','categorical_accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=10)



predictions = model.predict(X_test)

rounded = []

#set threshold for classifier, if above X% than classify as positive, etc.

threshold = 0.5



for x in predictions:

    if x>threshold:

        x = 1

    else:

        x = 0

    rounded.append(x)



rounded = pd.DataFrame(rounded, columns = ['predicted_output'])

concat = [y_test, rounded]



compare = pd.concat(concat, axis=1)

print(compare)



from sklearn.metrics import confusion_matrix

confused = confusion_matrix(y_test, rounded)



t_negative = confused[0][0]

f_positive = confused[0][1]

f_negative = confused[1][0]

t_positive = confused[1][1]



accuracy = (t_negative + t_positive) / (t_negative + f_positive + f_negative + t_positive)

recall = t_positive / (t_positive + t_negative)

precision = t_positive / (t_positive + f_positive)



print("my confusion matrix is: ")

print(confused)



print("accuracy is", round(accuracy*100),'%')

print("precision is", round(precision*100), '%')

print('recall is', round(recall*100), '%')





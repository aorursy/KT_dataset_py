import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense

df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")



#print(df.head())

# shows interesting facts about age group, missing values, and bell curves

# df.hist(figsize=(10,7))

# plt.show()



# for col in df.columns:

#     sns.scatterplot(x=col, y='Glucose', data=df, hue="Outcome")

#     plt.show()



# for col in df.columns:

#     sns.distplot(df.loc[df.Outcome == 0][col], hist=False, kde_kws={"linestyle":"-", "color":"black", "label": "No Diabetes"})

#     sns.distplot(df.loc[df.Outcome == 1][col], hist=False, kde_kws={"linestyle":"--", "color":"red", "label": "Diabetes"})

#

#     plt.show()





# edit null and 0 responses with col mean

for col in df.columns:

    if col != 'Pregnancies' and col != 'Outcome':

        df[col] = df[col].replace(0, np.nan)

        df[col] = df[col].fillna(df[col].mean())



# check null and 0 responses

# print(df.isnull().any())

#

# for col in df.columns:

#     missing_rows = df.loc[df[col]==0].shape[0]

#     print(col + ": "+ str(missing_rows))





# scale data as features have vastly different scales, e.g. mins and maxes

df_scaled = preprocessing.scale(df)

# convert back to pd datafrme

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# don't scale outcome variable

df_scaled['Outcome'] = df['Outcome']

df = df_scaled

# print(df.describe().round(2))

# for col in df.columns:

#     sns.distplot(df.loc[df.Outcome == 0][col], hist=False, kde_kws={'linestyle':'-', 'color':'blue', 'label':'No Diabetes'})

#     sns.distplot(df.loc[df.Outcome == 1][col], hist=False, kde_kws={'linestyle':'--', 'color':'red', 'label': 'Diabetes'})

#     plt.show()





# train data 80 20 and then 80 20

X = df.loc[:, df.columns != 'Outcome']

y = df.loc[:, 'Outcome']



X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.2)



# create Sequential model

model = Sequential()

# first layer

# Dense is a fully connected layer

# input_dim = 8 as we have 8 features

model.add(Dense(32, activation='relu', input_dim=8))

# second layer

model.add(Dense(16, activation='relu'))

#model.add(Dense(8, activation='relu'))



# output layer

# sigmoid as we need values between 0 and 1

model.add(Dense(1, activation='sigmoid'))

# define parameters of training model

model.compile(

              optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy']

)

# train model

model.fit(X_train, y_train, epochs=200)

# test the model accuracy

scores = model.evaluate(X_train, y_train)

print('Training Accuracy: %.f%%\n' %(scores[1]*100))

scores = model.evaluate(X_val,y_val)

print('Testing Accuracy: %.2f%%\n' % (scores[1]*100))

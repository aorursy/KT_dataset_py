import os

import pandas as pd # for reading our csv files

import matplotlib.pyplot as plt # for the amazing plots which will increase the notebook beauty

import seaborn as sns # a more detailed plotting tool used mostly for making complex plots just by simples lines
print(os.listdir("../input"))
df_red = pd.read_csv("../input/wineQualityReds.csv")

df_white = pd.read_csv("../input/wineQualityWhites.csv")

print(df_red.shape,df_white.shape)

df = pd.concat([df_red,df_white])

df = df.iloc[:,1:]

df = df.drop("quality",axis=1)

df
Type = {'Red': 1,'White': 0}



df["Type"] = [Type[item] for item in df["Type"]]

df
# The info() function is used to print a concise summary of a DataFrame. 

#This method prints information about a DataFrame including the index dtype and column dtypes,

#non-null values and memory usage. 

df.info()
#The describe() method is used for calculating some statistical data like percentile, 

#mean and std of the numerical values of the Series or DataFrame. It analyzes both 

#numeric and object series and also the DataFrame column sets of mixed data types.

df.describe()
new_df = df.drop("Type",axis=1) # i am dropinf this because dur to value between 0 and 1 for type 

#its throwing an error of one hot encoding, so i am not taking one hot encodinf here because its for learning of you people how skewness look

# but remmember we will train and perform all analysis with type coloum included

fig, axes = plt.subplots(ncols=new_df.shape[1],figsize=(40,10))

for ax, col in zip(axes, new_df.columns):

    sns.distplot(new_df[col], ax=ax)



plt.show()
# # Some of you may get this error as i got it and posted here if anyone else have please have look

#RuntimeWarning: invalid value encountered in log1p this occurs due to negative value in 

#total.sulfur.dioxide coloumn

import numpy as np

new_df['total.sulfur.dioxide'] = new_df[new_df['total.sulfur.dioxide']>=0]

new_df['free.sulfur.dioxide'] = new_df[new_df['free.sulfur.dioxide']>=0]

new_df['residual.sugar'] = np.log1p(new_df['residual.sugar'])

new_df['chlorides'] = np.log1p(new_df['chlorides'])

new_df['total.sulfur.dioxide'] = np.log1p(new_df['total.sulfur.dioxide'])

new_df['free.sulfur.dioxide'] = np.log1p(new_df['free.sulfur.dioxide'])

new_df['sulphates'] = np.log1p(new_df['sulphates'])
fig, axes = plt.subplots(ncols=new_df.shape[1],figsize=(40,10))

for ax, col in zip(axes, new_df.columns):

    sns.distplot(new_df[col], ax=ax)

plt.show()
_,ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(),annot=True,ax=ax)
x = df.iloc[:,:-1].values

x
y = df.iloc[:,-1].values

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression,LinearRegression

model1 = LogisticRegression(random_state=0)

model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

y_pred1.shape
from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.metrics import classification_report

print(r2_score(y_test, y_pred1))

print(metrics.accuracy_score(y_pred1,y_test))

print(classification_report(y_test, y_pred1))
model2 = LinearRegression()

model2.fit(X_train, y_train)
y_pred2 = model1.predict(X_test)

y_pred2.shape
from sklearn.metrics import r2_score

print(r2_score(y_test, y_pred2))

print(metrics.accuracy_score(y_pred2,y_test))

print(classification_report(y_test, y_pred2))
#most important SVR parameter is Kernel type. It can be 

#linear,polynomial or gaussian SVR. We have a non-linear condition 

#so we can select polynomial or gaussian but here we select RBF(agaussian type) kernel.

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, y_train)
y_pred3 = regressor.predict(X_test)

print(r2_score(y_test, y_pred3))

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf = clf.fit(X_train,y_train)

y_pred4 = clf.predict(X_test)
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred4))

print(classification_report(y_test, y_pred4))
# from sklearn.tree import export_graphviz

# from sklearn.externals.six import StringIO  

# from IPython.display import Image  

# import pydotplus



# dot_data = StringIO()

# export_graphviz(clf, out_file=dot_data,  

#                 filled=True, rounded=True,

#                 special_characters=True)

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# graph.write_png('diabetes.png')

# Image(graph.create_png())
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# n_estimators :-The number of trees in the forest.

clf1 = RandomForestRegressor(n_estimators=10)

clf1.fit(X_train,y_train)

predicted_Y = clf1.predict(X_test)

predicted = np.round(predicted_Y)

print(metrics.accuracy_score(predicted,y_test))

print(classification_report(y_test, predicted))
import tensorflow as tf # the libary for building our model you can use pytorch also but that would be much advanced.

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 10)

y_pred = ann.predict(X_test)
comparison_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

comparison_df
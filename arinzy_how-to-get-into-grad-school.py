# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#imports

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.style as style

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression

import sklearn
df =  pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df.describe()
df.isnull().sum()
df.drop(['Serial No.'],axis=1,inplace=True)

df.head()
df = df.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL','University Rating':'UniversityRating','Chance of Admit':'Chance'})

df.head()
df.rename(columns={ df.columns[4]: "LOR" }, inplace = True)
df.rename(columns={ df.columns[-1]: "Chance" }, inplace = True)

df.head()
df.corr()
df.corr()['Chance'].sort_values()
style.use('fivethirtyeight')

sns.heatmap(df.corr())
df_pred = df.sample(frac=1).reset_index(drop=True)
scaler = StandardScaler()

numeric_features = ['GRE','TOEFL','UniversityRating','SOP','LOR','CGPA']



df_pred[numeric_features]  = scaler.fit_transform(df_pred[numeric_features])

df_pred.head()
X_train, X_test, y_train, y_test = train_test_split(df_pred.drop('Chance',axis=1), df_pred['Chance'], test_size = .2, random_state=10)
bins = [df.Chance.min(), 0.75, 1]

labels = [0,1]

df_cat = df.copy()

df_cat['Admission'] = pd.cut(df['Chance'],bins=bins,labels=labels)

df_cat.head()
df_cat.Admission.value_counts(normalize=True)


g = sns.FacetGrid(df_cat, hue="Admission")

plt.figure(figsize=(15,20))

g.map(sns.scatterplot, "GRE", "TOEFL", alpha=.7)

g.add_legend()
plt.figure(figsize=(8,6))

sns.scatterplot(data=df_cat,x="GRE", y="CGPA",hue='Admission', alpha=.7)
plt.figure(figsize=(8,6))

sns.scatterplot(data=df_cat,x="GRE", y="TOEFL",hue='Admission', alpha=.7)
df.Research.value_counts()
labels=['1','0']

plt.figure(figsize=(6,6))

plt.title('Resarch Experience')

plt.pie(df.Research.value_counts(), labels=labels, autopct='%1.1f%%', shadow=True);
g = sns.FacetGrid(df_cat, col="Research", hue="Admission",height=5,aspect=1)

g.map(sns.scatterplot, "GRE", "CGPA", alpha=.7)

g.add_legend()


df_risky = df_cat[(df_cat.GRE > 300) & (df_cat.GRE < 320)& (df_cat.CGPA > 8.2) & (df_cat.CGPA<=9)]
df_risky.head()
g = sns.FacetGrid(df_risky, col="Research", hue="Admission",height=5,aspect=1)

g.map(sns.scatterplot, "GRE", "CGPA", alpha=.7)

g.add_legend()
df_risky['Admission'].value_counts(normalize=True).plot(kind='pie')
df_cat.groupby('Research')['Admission'].value_counts(normalize=True).unstack('Research')
ax = df_cat.groupby('Research')['Admission'].value_counts(normalize=True).unstack('Research').plot(kind='bar',figsize=(12,8),rot=0)

ax.legend(['Not Admitted',' Admitted'])

ax.set_xlabel('Research');

ax.set_ylabel('Percentages');
ax = df_risky.groupby('Research')['Admission'].value_counts(normalize=True).unstack('Admission').plot(kind='bar',figsize=(12,8),rot=0)

ax.legend(['Not Admitted',' Admitted'])

ax.set_xlabel('Research');

ax.set_ylabel('Percentages');
def get_rmse(model,X_test,y_test):

    mse = sklearn.metrics.mean_squared_error(y_test, model.predict(X_test));

    return np.sqrt(mse)
import sklearn

from sklearn.model_selection import GridSearchCV



def get_best_model_and_accuracy(model, params, X, y):

    grid = GridSearchCV(model, params, error_score=0.)

    grid.fit(X, y) # fit the model and parameters

    # our classical metric for performance

    print ("Best Accuracy: {}".format(grid.best_score_))

    # the best parameters that caused the best accuracy

    print ("Best Parameters: {}".format(grid.best_params_))

    # the average time it took a model to fit to the data (in seconds)

    print('RMSE on test set: {}'.format(get_rmse(grid,X_test,y_test)))

    return grid.best_params_



from sklearn.linear_model import ElasticNet



elastic_net = ElasticNet()

elastic_net_params = {'alpha':[0.25,0.5,1], 'l1_ratio':[0.25,0.5,1]}



get_best_model_and_accuracy(elastic_net, elastic_net_params, X_train, y_train);
linreg = LinearRegression()

linreg_params = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

linreg_best_params = get_best_model_and_accuracy(linreg, linreg_params, X_train, y_train)
import keras

import tensorflow as tf

from keras import layers

from keras.layers import Input

from keras.layers.core import  Dense, Activation

from keras.models import Model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils import plot_model

from keras.layers import Dropout

from keras.regularizers import l2

from keras.regularizers import l1

np.random.seed(10)

def model_nn(input_shape):

    



    # Define the input placeholder as a tensor with shape input_shape

    X_input = Input(input_shape)



  

    X = Dense(512)(X_input)

    X = Activation('relu')(X)

    X = Dropout(0.5,seed=10)(X)

    X = Dense(256)(X)

    X = Activation('relu')(X)

    X = Dropout(0.25,seed=10)(X)

    X = Dense(16)(X)

    X = Activation('relu')(X)

    X = Dropout(0.25)(X)

    X = Dense(1, activation='relu')(X)



    # Create model. 

    model = Model(inputs = X_input, outputs = X, name='nnModel')



    return model
nnModel = model_nn(X_train.shape)

nnModel.compile(optimizer="adam",loss="mse",metrics=['mean_squared_error'])

nnModel.fit(x = X_train, y = y_train, epochs = 250, batch_size = 32)
preds = nnModel.evaluate(x= X_test, y=y_test)

#print()

print ("Loss = " + str(preds[0]))

print ("Rmse on test set = " , np.sqrt(preds[1]))
regressor = LinearRegression(linreg_best_params)

regressor.fit(X_train,y_train);
plt.scatter(X_test.index.values, y_test, color = 'red')

plt.scatter(X_test.index.values, regressor.predict(X_test), color = 'blue')

plt.title('True values(Red) vs Prediction(Blue) (Test set) Linear Regression')

plt.xlabel('Index')

plt.ylabel('Chance')

plt.show()
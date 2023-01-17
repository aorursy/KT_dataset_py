# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()
df.rename(mapper={'Chance of Admit ':'Chance of Admit'},inplace=True,axis='columns')

df.columns
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

%matplotlib inline
corr=df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(data=corr,mask=mask,ax=ax,cmap=cmap,annot=True)
fig,ax=plt.subplots(figsize=(11,9))

sns.kdeplot(df['GRE Score'].apply(lambda x:(x-np.min(df['GRE Score']))/(np.max(df['GRE Score'])-np.min(df['GRE Score']))),ax=ax,label='GRE Score')

sns.kdeplot(df['TOEFL Score'].apply(lambda x:(x-np.min(df['TOEFL Score']))/(np.max(df['TOEFL Score'])-np.min(df['TOEFL Score']))),ax=ax,label='TOEFL Score')

sns.kdeplot(df['CGPA'].apply(lambda x:(x-np.min(df['CGPA']))/(np.max(df['CGPA'])-np.min(df['CGPA']))),ax=ax,label='CGPA')

sns.kdeplot(df['Chance of Admit'].apply(lambda x:(x-np.min(df['Chance of Admit']))/(np.max(df['Chance of Admit'])-np.min(df['Chance of Admit']))),ax=ax,label='Chance of Admit',shade=True)
# Here I have normalized the data, because when I do that, I can undersand the distribution (standard deviation: the spread of the curve)

# interestingly all seems to have a similar curves, so I plan to check the collinearity
fig,ax=plt.subplots(figsize=(8,6))

sns.scatterplot(x='CGPA',y='Chance of Admit',ax=ax,hue='Research',data=df)

# Checking if research can show any relation

# unsurpisingly, the answer is if you have research in resume, there is a good possibility that you will be admited
import patsy

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

df.rename({'TOEFL Score':'TOEFL_Score','LOR ':'LOR','Chance of Admit':'Chance_of_Admit','GRE Score':'GRE_Score'},inplace=True,axis='columns')
# checking the collinearity between different independent variables

# in general VIF>5, is considered to have collinearity, which may effect the regression model

# why CGPA? chance of admit is more correlated to CGPA. [0.88]

feature_name='CGPA'

df[['GRE_Score','TOEFL_Score','SOP','LOR','CGPA','Chance_of_Admit']].dropna()

features='+'.join(df[['GRE_Score','TOEFL_Score','SOP','LOR','CGPA','Chance_of_Admit']].columns.difference([feature_name]))

features=feature_name+'~'+features

Y,x=patsy.dmatrices(formula_like=features,data=df[['GRE_Score','TOEFL_Score','SOP','LOR','CGPA','Chance_of_Admit']],return_type='dataframe')

VIF=pd.DataFrame()

VIF['Features']=x.columns

VIF['Values']=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]

VIF
## I think none of the Variance inflation factor is greater than 5, which tells the non of the data is collinear? Hence we can use all the data except serial number

# for the regression analysis



# Here I use the keras sequential model, i am providing the link below. (I am no expert on keras, as of now)

# I am linking the keras model tutorial, using which I developed below

# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/





from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
dataset=df.values
X=dataset[:,1:-1]

Y=dataset[:,-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.15)
# Defined the keras sequntial model

def baseline_model(neurons):

    # create model

    model = Sequential()

    model.add(Dense(units=neurons, input_dim=7, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
# I have changed the scaler from standard to minmax scaler, I found the error to be less with that

# neuros=16: I have tried with varying number of neurons(7:20) and found the least error for 16

seed=7

np.random.seed(seed)

estimators = []

pipeline=[]

estimators.append(('standardize', MinMaxScaler()))

estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=10, verbose=0,neurons=16)))

pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10, random_state=seed)

# I have used the k-fold on training set 

results = cross_val_score(pipeline, X_train, Y_train, cv=kfold)

#res=np.append(res,np.asarray(results.mean()))

print("Results: %.4f (%.4f) MSE" % (results.mean(), results.std()))
# send test set as new data, which network has never seen before

pipeline.fit(X_train,Y_train)

prediction = pipeline.predict(X_test)
# plotting the first 10 (i=10) data chance of admit for actual vs predicted

i=10

plt.figure(figsize=(10,8))

ax=plt.subplot(1,1,1)



plt.xlabel('No')

plt.ylabel('Chance of Admitance')





if i<=X_test.shape[0]:

    ax.plot(range(i),prediction[0:i],label='Prediction')

    ax.plot(range(i),Y_test[0:i],label='actual')

else:

    ax.plot(range(X_test.shape[0]),prediction[0:X_test.shape[0]],label='Prediction')

    ax.plot(range(X_test.shape[0]),Y_test[0:X_test.shape[0]],label='Prediction')



ax.legend()
# as a metrics, i used the mean squared error

mean_absolute_error(prediction,Y_test)
# As you can see, this is my first model that I am presenting. you can suggest me better approach to the problem
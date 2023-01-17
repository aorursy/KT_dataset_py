import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score, f1_score

import os

#set matplotlib style

plt.style.use('ggplot')

#set seaborn

sns.set(context='notebook', palette='RdBu', style='darkgrid')

%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()
df.info()
df.describe()
df.Species.unique()
sns.countplot(x='Species',

              data=df)

plt.show()
# define the CDF function

def cdf(x):

    x=np.sort(x)

    y=np.arange(1,len(x)+1)/len(x)

    return x,y
# Check Sepal Length for different Species

setosa=df[df['Species']=='Iris-setosa']['SepalLengthCm']

versicolor=df[df['Species']=='Iris-versicolor']['SepalLengthCm']

virginica=df[df['Species']=='Iris-virginica']['SepalLengthCm']

s_x, s_y=cdf(setosa)

ve_x, ve_y=cdf(versicolor)

vi_x, vi_y=cdf(virginica)

plt.plot(s_x, s_y, label='setosa', color='red', marker='.')

plt.plot(ve_x, ve_y, label='versicolor', color='blue', marker='.')

plt.plot(vi_x, vi_y, label='virginica', color='green', marker='.')

plt.title('CDF of Sepal Length')

plt.xlabel('Sepal Length')

plt.ylabel('ECDF')

plt.legend()

plt.show()
setosa=df[df['Species']=='Iris-setosa']['SepalWidthCm']

versicolor=df[df['Species']=='Iris-versicolor']['SepalWidthCm']

virginica=df[df['Species']=='Iris-virginica']['SepalWidthCm']

s_x, s_y=cdf(setosa)

ve_x, ve_y=cdf(versicolor)

vi_x, vi_y=cdf(virginica)

plt.plot(s_x, s_y, label='setosa', color='red', marker='.')

plt.plot(ve_x, ve_y, label='versicolor', color='blue', marker='.')

plt.plot(vi_x, vi_y, label='virginica', color='green', marker='.')

plt.title('CDF of Sepal Width')

plt.xlabel('Sepal Width in cm')

plt.ylabel('ECDF')

plt.legend()

plt.show()
setosa=df[df['Species']=='Iris-setosa']['PetalLengthCm']

versicolor=df[df['Species']=='Iris-versicolor']['PetalLengthCm']

virginica=df[df['Species']=='Iris-virginica']['PetalLengthCm']

s_x, s_y=cdf(setosa)

ve_x, ve_y=cdf(versicolor)

vi_x, vi_y=cdf(virginica)

plt.plot(s_x, s_y, label='setosa', color='red', marker='.')

plt.plot(ve_x, ve_y, label='versicolor', color='blue', marker='.')

plt.plot(vi_x, vi_y, label='virginica', color='green', marker='.')

plt.title('CDF of Petal Length')

plt.xlabel('Petal Length in cm')

plt.ylabel('ECDF')

plt.legend()

plt.show()
setosa=df[df['Species']=='Iris-setosa']['PetalWidthCm']

versicolor=df[df['Species']=='Iris-versicolor']['PetalWidthCm']

virginica=df[df['Species']=='Iris-virginica']['PetalWidthCm']

s_x, s_y=cdf(setosa)

ve_x, ve_y=cdf(versicolor)

vi_x, vi_y=cdf(virginica)

plt.plot(s_x, s_y, label='setosa', color='red', marker='.')

plt.plot(ve_x, ve_y, label='versicolor', color='blue', marker='.')

plt.plot(vi_x, vi_y, label='virginica', color='green', marker='.')

plt.title('CDF of Petal width')

plt.xlabel('Petal width in cm')

plt.ylabel('ECDF')

plt.legend()

plt.show()
def to_codes(df,col):

    df[col]=pd.Categorical(df[col]).codes

    return df

pre_df=to_codes(df, 'Species')


def min_max_scale(df, col):

    df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())

    return df

for i in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:

    pre_df=min_max_scale(pre_df, i)

x=pre_df.drop(['Id', 'Species'], axis=1)

y=pre_df.Species
x_train, x_test, y_train, y_test=train_test_split(x,

                                                  y,

                                                  test_size=0.2,

                                                  random_state=123) 
def cv_score(x,y, model):

    cv=cross_val_score(model,

                       x, y,

                       cv=5,

                       scoring='accuracy',

                       n_jobs=-1)

    return np.mean(cv)

lr=LogisticRegression()

print('CV Score of Logistic Regression: ', cv_score(x_train, y_train, lr))
# prediction

lr.fit(x_train, y_train)

pred=lr.predict(x_test)

print('accuracy score of Logistic Regression: ', accuracy_score(y_test, pred))
lr.get_params
params={

    'solver':['newton-cg', 'lbfgs', 'liblinear'],

    'penalty':['l1', 'l2', 'elasticnet'],

    'C':np.arange(0.1,3,0.01)

}

gs=GridSearchCV(lr,

                param_grid=params,

                cv=5,

                scoring='accuracy', 

                n_jobs=-1)

gs.fit(x_train, y_train)

print('best parameters: ',gs.best_params_)

print('best CV score: ',gs.best_score_)
lr2=LogisticRegression(C=1.7799999999999994, penalty='l2', solver= 'newton-cg')

lr2.fit(x_train, y_train)

pred2=lr2.predict(x_test)

print('accuracy score of tuned Logistic Regression: ', accuracy_score(y_test, pred))
pd.DataFrame({

    'predicted':pred2,

    'real':y_test

})
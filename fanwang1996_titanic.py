# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

warnings.filterwarnings("ignore", message="numpy.ufunc size changed")



from pandas import Series, DataFrame



import pandas as pd

import numpy as np
data_train = pd.read_csv('../input/titanic/train.csv', engine ='python', encoding = 'UTF-8')



data_train.head()
#Overview of the dataset



data_train.info()
data_train.describe()
%matplotlib inline

import matplotlib.pyplot as plt

fig = plt.figure()

fig.set(alpha=0.2)  #set the color variable in the chart



plt.subplot2grid((2,3),(0,0))  #inludes differen small picture into a big picture

data_train.Survived.value_counts().plot(kind='bar')

plt.title("Saving condition(1 means saved)")

plt.ylabel(" numebr of people")





plt.subplot2grid((2,3),(0,1))

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel("number of people")

plt.title("passagers class dirtribution")



plt.subplot2grid((2,3),(0,2))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel("Age")

plt.grid(b=True, which='major', axis='y')

plt.title("see the save distribution base on their age(1 means saved)")



plt.subplot2grid((2,3),(1,0), colspan=2)

data_train.Age[data_train.Pclass == 1].plot(kind='kde')

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel('age') # plot on axis table

plt.title('differenct class age distribution')

plt.legend(('first class,"second class, third class'),loc='best') #sets our legend for our graph.



plt.subplot2grid((2,3),(1,2))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title("Number of people boarding at each boarding port")

plt.ylabel('numebr of people')

plt.show()
fig = plt.figure()

fig.set(alpha = 0.2)





Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df=pd.DataFrame({'unsafe':Survived_0, 'saved':Survived_1})

df.plot(kind = 'bar', stacked = True)



plt.title('different class save situation')

plt.xlabel('passager class')

plt.ylabel('number of people')

plt.show()
fig = plt.figure()

fig.set(alpha = 0.2)





Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})

df.plot(kind = 'bar', stacked = True)



plt.title('Different gender save situation')

plt.xlabel('Gender')

plt.ylabel('number of people')

plt.show()
fig = plt.figure()

fig.set(alpha=0.65) #Setting transparency

plt.title('base on gender and different class')





ax1=fig.add_subplot(141)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().sort_index().plot(kind = 'bar', label='female 1st class', color = '#FA2479')

ax1.set_xticks([0,1])

ax1.set_xticklabels(["unsave", "save"], rotation = 0)

ax1.legend(["female/1st class"], loc='best' )



ax2=fig.add_subplot(142, sharey=ax1)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().sort_index().plot(kind = 'bar', label='famale, low class', color = 'pink')

ax2.set_xticklabels(["unsave", "save"], rotation = 0)

plt.legend(["female/low class"], loc='best' )



ax3=fig.add_subplot(143, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().sort_index().plot(kind = 'bar', label='male, 1st class', color = 'lightblue')

ax2.set_xticklabels(["unsave", "save"], rotation = 0)

plt.legend(["male/1st class"], loc='best' )



ax4=fig.add_subplot(144, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().sort_index().plot(kind = 'bar', label='male, low class', color = 'steelblue')

ax2.set_xticklabels(["unsave", "save"], rotation = 0)

plt.legend(["male/low class"], loc='best' )



plt.show()
fig = plt.figure()

fig.set(alpha=0.2)



Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()



df = pd.DataFrame({'unsafe':Survived_0, 'save':Survived_1})

df.plot(kind = 'bar', stacked=True)

plt.title("Save distribution base on different port")

plt.xlabel('On-board Port')

plt.ylabel('# of people')

plt.show()
gg = data_train.groupby(['SibSp','Survived'])

df = pd.DataFrame(gg.count()['PassengerId'])

print(df)
gp = data_train.groupby(['Parch','Survived'])

df = pd.DataFrame(gp.count()['PassengerId'])

print(df)
data_train.Cabin.value_counts()
fig = plt.figure()

fig.set(alpha=0.2)  



Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()

Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()

df=pd.DataFrame({'Yes':Survived_cabin, 'No':Survived_nocabin}).transpose()



df.plot(kind='bar', stacked=True)

plt.title("Check the save distribuiton base on the Cabin")

plt.xlabel("Cabin or Non-Cabin") 

plt.ylabel("# number of peole")

plt.show()
from sklearn.ensemble import RandomForestRegressor



## Use RandomForest to fullfill the age variable##



def set_missing_ages(df):

    

    # Put the known numerical varible into the Random Forest Regressor

    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

  

    

    # Divided the customers into 2 groups:age known/ age unknown

    known_age = age_df[age_df.Age.notnull()].values

    unknown_age = age_df[age_df.Age.isnull()].values

     

    # y means generate age

    y = known_age[:, 0]

    

    # x means variable values

    X = known_age[:, 1:]

    

    # fit into the RandomForestRegressor

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

    

    # Using the model to predict the unknown age 

    predictedAges = rfr.predict(unknown_age[:, 1::])

    

    # Using the predict value to fullfill the missing value

    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    

    return df, rfr



def set_Cabin_type(df):

    df.loc[(df.Cabin.notnull()), 'Cabin' ] ="Yes"

    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"

    return df

    

data_train, rfr = set_missing_ages(data_train)

data_train = set_Cabin_type(data_train)
data_train.head(10)
data_train.info()
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix = 'Cabin')



dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix = 'Embarked')



dummies_Sex = pd.get_dummies(data_train['Sex'], prefix = 'Sex')



dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')



df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis = 1)



df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace = True)



df.head()
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()



age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)



fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), age_scale_param)



df.drop(['Age','Fare','PassengerId'], axis=1, inplace = True)



df.head()
from sklearn import linear_model



# Use regular to extract the value we want



train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Cabin_.*')

train_np = train_df.values



# y means the column 0, which is the target varible 

y = train_np[:, 0]



# X menas the column after the 1st column are the x variable

X = train_np[:, 1:]





# fit into the LogisticRegression

clf = linear_model.LogisticRegression(solver='liblinear', C=1.0, penalty='l1', tol=1e-6)

clf.fit(X, y)



clf
data_test = pd.read_csv('../input/titanic/test.csv', engine ='python', encoding = 'UTF-8')



data_test.head()
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

# Using the same feature enginering on the test_data



# Use RandomForestRegressor model fullfil the missing values

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

null_age = tmp_df[data_test.Age.isnull()].values



# Fullfill the Missing Values

X = null_age[:, 1:]

predictedAges = rfr.predict(X)

data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges



data_test = set_Cabin_type(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')





df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)

df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)



df_test.drop(['Age','Fare','PassengerId'], axis=1, inplace = True)



df_test.head()
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
filename = 'mycsvfile.csv'

result.to_csv(filename,index=False)

print('Saved file: ' + filename)
pd.DataFrame({'columns':list(train_df.columns)[1:], 'coef':list(clf.coef_.T)})
from sklearn.model_selection import cross_val_score, train_test_split



clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1',tol=1e-6)



all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*|Cabin_.*')



X = all_data.values[:,1:]



y = all_data.values[:, 0]



print(cross_val_score(clf, X, y, cv=5))
import numpy as np

import matplotlib.pyplot as plt

# from sklearn.learning_curve import learning_curve  修改以fix learning_curve DeprecationWarning

from sklearn.model_selection import learning_curve



# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 

                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):

    """

    画出data在某模型上的learning curve.

    参数解释

    ----------

    estimator : 你用的分类器。

    title : 表格的标题。

    X : 输入的feature，numpy类型

    y : 输入的target vector

    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点

    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)

    n_jobs : 并行的的任务数(默认1)

    """

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    if plot:

        plt.figure()

        plt.title(title)

        if ylim is not None:

            plt.ylim(*ylim)

        plt.xlabel("Trainning samples")

        plt.ylabel("Score")

        plt.gca().invert_yaxis()

        plt.grid()

    

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 

                         alpha=0.1, color="b")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 

                         alpha=0.1, color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label='Trainning set Score')

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label='Validation Score')

    

        plt.legend(loc="best")

        

        plt.draw()

        plt.gca().invert_yaxis()

        plt.show()

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff



plot_learning_curve(clf, "Learning Curve", X, y)
from sklearn.ensemble import BaggingRegressor



train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')

train_np = train_df.values



# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到BaggingRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

bagging_clf.fit(X, y)



test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')

predictions = bagging_clf.predict(test)

result2 = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
filename = 'mycsvfile2.csv'

result2.to_csv(filename,index=False)

print('Saved file: ' + filename)
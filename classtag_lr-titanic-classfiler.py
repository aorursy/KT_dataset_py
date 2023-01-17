# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load data

train_df = pd.read_csv("../input/train.csv",header=0)

test_df = pd.read_csv("../input/test.csv",header=0)

train_df.shape,test_df.shape
# merge train and test

#合成一个总的data

train_df['source']= 'train'

test_df['source'] = 'test'

raw_data = pd.concat([train_df, test_df],ignore_index=True)

raw_data.shape
raw_data.head()
raw_data.dtypes
# 看看缺失情况

raw_data.apply(lambda x: sum(x.isnull()))
raw_data.info()
# 接下来我们就逐个feature开始分析

# 先简单做个交叉分析吧

raw_data.columns
# 看下整体分布

raw_data.describe()
# 看看每个/多个 属性和最后的Survived之间有着什么样的关系

import matplotlib.pyplot as plt



fig = plt.figure(num=None, figsize=(9, 9), dpi=70, facecolor='w', edgecolor='k')

fig.set(alpha=0.3)  # 设定图表颜色alpha参数



plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图

raw_data.Survived.value_counts().plot(kind='bar')# plots a bar graph of those who surived vs those who did not. 

plt.title(u"survived") # puts a title on our graph

plt.ylabel(u"num.")  



plt.subplot2grid((2,3),(0,1))

raw_data.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"num.")

plt.title(u"Pclass")



plt.subplot2grid((2,3),(0,2))

plt.scatter(raw_data.Survived, raw_data.Age)

plt.ylabel(u"Age")                         # sets the y axis lable

plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs

plt.title(u"age and suvived")





plt.subplot2grid((2,3),(1,0), colspan=2)

raw_data.Age[raw_data.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age

raw_data.Age[raw_data.Pclass == 2].plot(kind='kde')

raw_data.Age[raw_data.Pclass == 3].plot(kind='kde')

plt.xlabel(u"Age")# plots an axis lable

plt.ylabel(u"") 

plt.title(u"Pclass & Age")

plt.legend((u'1P', u'2P',u'3P'),loc='best') # sets our legend for our graph.





plt.subplot2grid((2,3),(1,2))

raw_data.Embarked.value_counts().plot(kind='bar')

plt.title(u"Embarked")

plt.ylabel(u"num.")

plt.show()
#看看各乘客等级的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = raw_data.Pclass[raw_data.Survived == 0].value_counts()

Survived_1 = raw_data.Pclass[raw_data.Survived == 1].value_counts()

df=pd.DataFrame({u'Survived':Survived_1, u'Not Survived':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"Pclass-Survived")

plt.xlabel(u"Pclass") 

plt.ylabel(u"num.") 



plt.show()
#看看各登录港口的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = raw_data.Embarked[raw_data.Survived == 0].value_counts()

Survived_1 = raw_data.Embarked[raw_data.Survived == 1].value_counts()

df=pd.DataFrame({u'Survived_1':Survived_1, u'Survived_0':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"Embarked-Survived")

plt.xlabel(u"Embarked") 

plt.ylabel(u"Num.") 



plt.show()
#看看各性别的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_m = raw_data.Survived[raw_data.Sex == 'male'].value_counts()

Survived_f = raw_data.Survived[raw_data.Sex == 'female'].value_counts()

df=pd.DataFrame({u'male':Survived_m, u'female':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title(u"Sex-Survived")

plt.xlabel(u"Sex") 

plt.ylabel(u"Num.")

plt.show()
#然后我们再来看看各种舱级别情况下各性别的获救情况

fig = plt.figure(num=None, figsize=(9, 9), dpi=70, facecolor='w', edgecolor='k')

fig.set(alpha=0.65) # 设置图像透明度，无所谓

plt.title(u"Pclass-Sex-Survived")



ax1=fig.add_subplot(141)

raw_data.Survived[raw_data.Sex == 'female'][raw_data.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels([u"YES", u"NO"], rotation=0)

ax1.legend([u"female/Pclass1-2"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

raw_data.Survived[raw_data.Sex == 'female'][raw_data.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"Yes", u"No"], rotation=0)

plt.legend([u"female/Pclass3"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

raw_data.Survived[raw_data.Sex == 'male'][raw_data.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"YES", u"NO"], rotation=0)

plt.legend([u"male/Pclass1-2"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

raw_data.Survived[raw_data.Sex == 'male'][raw_data.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"Yes", u"No"], rotation=0)

plt.legend([u"male/Pclass3"], loc='best')



plt.show()
g = raw_data.groupby(['SibSp','Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

df
g = raw_data.groupby(['Parch','Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

df
#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，不纳入考虑的特征范畴

#cabin只有204个乘客有值，我们先看看它的一个分布

raw_data.Cabin.value_counts()
#cabin的值计数太分散了，绝大多数Cabin值只出现一次。感觉上作为类目，加入特征未必会有效

#那我们一起看看这个值的有无，对于survival的分布状况，影响如何吧

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_cabin = raw_data.Survived[pd.notnull(raw_data.Cabin)].value_counts()

Survived_nocabin = raw_data.Survived[pd.isnull(raw_data.Cabin)].value_counts()

df=pd.DataFrame({u'YES':Survived_cabin, u'NO':Survived_nocabin}).transpose()

df.plot(kind='bar', stacked=True)

plt.title(u"Has_Cabin-Survived")

plt.xlabel(u"Has_Cabin") 

plt.ylabel(u"Num.")

plt.show()



#似乎有cabin记录的乘客survival比例稍高，那先试试把这个值分为两类，有cabin值/无cabin值，一会儿加到类别特征好了
# 补充一个票价的缺失值

raw_data.Fare = raw_data.Fare.fillna(raw_data['Fare'].mean())
age_df = raw_data[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

# age_df.Fare = age_df.Fare.fillna(age_df['Fare'].mean())

# 乘客分成已知年龄和未知年龄两部分

known_age = age_df[age_df.Age.notnull()]

unknown_age = age_df[age_df.Age.isnull()]



# y即目标年龄

y = known_age['Age']

# X即特征属性值

X = known_age[['Fare', 'Parch', 'SibSp', 'Pclass']]



from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

rfr.fit(X, y)



# 用得到的模型进行未知年龄结果预测

predictedAges = rfr.predict(unknown_age[['Fare', 'Parch', 'SibSp', 'Pclass']])
from sklearn.ensemble import RandomForestRegressor

### 使用 RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    

    # 乘客分成已知年龄和未知年龄两部分

    known_age = age_df[age_df.Age.notnull()]

    unknown_age = age_df[age_df.Age.isnull()]

    

    # y即目标年龄

    y = known_age['Age']

    # X即特征属性值

    X = known_age[['Fare', 'Parch', 'SibSp', 'Pclass']]

    

    # fit到RandomForestRegressor之中

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

    

    # 用得到的模型进行未知年龄结果预测

    predictedAges = rfr.predict(unknown_age[['Fare', 'Parch', 'SibSp', 'Pclass']])

    

    # 用得到的预测结果填补原缺失数据

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    

    return df, rfr



def set_Cabin_type(df):

    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    return df





raw_data, rfr = set_missing_ages(raw_data)

raw_data = set_Cabin_type(raw_data)

raw_data.head()
raw_data.info()
# 因为逻辑回归建模时，需要输入的特征都是数值型特征

# 我们先对类目型的特征离散/因子化

# 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性

# 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0

# 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1

# 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示

dummies_Cabin = pd.get_dummies(raw_data['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(raw_data['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(raw_data['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(raw_data['Pclass'], prefix= 'Pclass')

df = pd.concat([raw_data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df.head()
# 接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内

# 这样可以加速logistic regression的收敛



df.Age.reshape(-1,1)

df.Fare.reshape(-1,1)

#age_scale_param = scaler.fit(df['Age'])

#df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)

#fare_scale_param = scaler.fit(df['Fare'])

#df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

df.head()
df.describe()
#df.loc[ (df.source == 1), 'source' ] = "train"

#df.loc[ (df.source == 0), 'source' ] = "test"

train_df = df[df.source == 'train']

test_df = df[df.source == 'test']

train_df.drop(["source"],axis=1,inplace=True)

test_df.drop(["source"],axis=1,inplace=True)

train_df.head()
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模

from sklearn import linear_model



train_data = train_df.filter(regex='Survived|Age|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')



# y即Survival结果

train_y = train_df.filter(regex='Survived')



# X即特征属性值

train_X = train_df.filter(regex='Age|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')



#train_X.head(10)

# fit到RandomForestRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

clf.fit(train_X, train_y)



clf
test_X = test_df.filter(regex='Age|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()

test_df.head()

predictions = clf.predict(test_X)



result = pd.DataFrame({'PassengerId':test_df['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

result.to_csv("logistic_regression_predictions.csv", index=False)
pd.read_csv("logistic_regression_predictions.csv").head()
# 要判定一下当前模型所处状态(欠拟合or过拟合)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.learning_curve import learning_curve



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

    train_sizes, train_scores, test_scores = learning_curve(estimator, 

                                                            X, 

                                                            y, 

                                                            cv=cv, 

                                                            n_jobs=n_jobs, 

                                                            train_sizes=train_sizes, 

                                                            verbose=verbose)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    if plot:

        plt.figure()

        plt.title(title)

        if ylim is not None:

            plt.ylim(*ylim)

        plt.xlabel(u"train_sample_nums")

        plt.ylabel(u"score")

        plt.gca().invert_yaxis()

        plt.grid()

    

        plt.fill_between(train_sizes, 

                         train_scores_mean - train_scores_std, 

                         train_scores_mean + train_scores_std,

                         alpha=0.1, 

                         color="b")

        plt.fill_between(train_sizes, 

                         test_scores_mean - test_scores_std, 

                         test_scores_mean + test_scores_std,

                         alpha=0.1, 

                         color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train_score")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"kfold_score")

    

        plt.legend(loc="best")

        

        plt.draw()

        plt.gca().invert_yaxis()

        plt.show()

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff



plot_learning_curve(clf, u"learning curve", train_X.as_matrix, train_y)
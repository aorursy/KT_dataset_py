# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
df=pd.concat([train,test],axis=0,ignore_index=True)
df.shape
df.head()
df.info()

#发现有一些列，Cabin,Age等有缺失值
df.describe()
import matplotlib.pyplot as plt
fig=plt.figure()

fig.set(alpha=0.2)#设定图标颜色参数
plt.subplot2grid((2,3),(0,0))#在一张图里分为两行三列，选取第一行第一列,和subplot差不多，但是可以设置图片跨越的宽度和高度

df.Survived.value_counts().plot(kind='bar')#当前图为柱状图

plt.title("Survived") #当前标题

plt.ylabel("Number of people") #当前y轴



plt.subplot2grid((2,3),(0,1))

df.Pclass.value_counts().plot(kind="bar")

plt.ylabel("Pclass")

plt.title("Passenger Class")



#plt.subplot2grid((2,3),(0,2)

#df['Age'][df.Age.isnull()==True]=df.Age.median()

#plt.scatter(df.Survived,df.Age)



plt.subplot2grid((2,3),(1,0),colspan=2)

df.Age[df.Pclass==1].plot(kind='kde')

df.Age[df.Pclass==2].plot(kind='kde')                 

df.Age[df.Pclass==3].plot(kind='kde') 

plt.xlabel("Age")

plt.xlabel("Density")

plt.title("Age distribution of different class")

plt.legend(('1','2','3'),loc='best')



plt.subplot2grid((2,3),(1,2))

df.Embarked.value_counts().plot(kind='bar')

plt.title('number of people ')

plt.ylabel('number of people')

plt.show()
#看看各乘客登机的获救情况

fig=plt.figure()

fig.set(alpha=0.2)#设定图标颜色alpha参数



Survived_0=df.Pclass[df.Survived==0].value_counts()

Survived_1=df.Pclass[df.Survived==1].value_counts()

df_pclass=pd.DataFrame({'Survived':Survived_1,'NotSurvived':Survived_0})

df_pclass.plot(kind='bar',stacked=True)#把两种数据堆积

plt.title('Survival of different pclass')

plt.xlabel('Pclass')

plt.ylabel('Number of people')

plt.show()
#明显等级为1的乘客，获救的概率高很多。所以这个一定是影响获救结果的一个特征
#看看各性别的获救情况

fig=plt.figure()

fig.set(alpha=0.2)#设定图标颜色参数



Survived_m=df.Survived[df.Sex=='male'].value_counts()

Survived_f=df.Survived[df.Sex=='female'].value_counts()

df_sex=pd.DataFrame({'male':Survived_m,'female':Survived_f})

df_sex.plot(kind='bar',stacked=True)

plt.title('Survival of different sex')

plt.xlabel('Survived')

plt.ylabel('Number of people')

plt.show()
#女性存货的比例明显较多，性别无疑也要作为重要特征加入最后的模型中
#详细版本：各种舱级别情况下各性别的获救情况

fig=plt.figure()

fig.set(alpha=0.65)#设置图像透明度

plt.title("Survival of different class")



ax1=fig.add_subplot(141)#返回轴ax1，1，4，1分别代表子图总行数，子图总列数，子图位置

df.Survived[df.Sex=='female'][df.Pclass!=3].value_counts().plot(kind='bar',label="female highclass",color='red')

ax1.set_xticklabels(["Survived","NotSurvived"],rotation=0)#xticklabels为坐标轴刻度

ax1.legend(["female/highclass"],loc='best')  #legend为图例



ax2=fig.add_subplot(142,sharey=ax1)

df.Survived[df.Sex=='female'][df.Pclass==3].value_counts().plot(kind='bar',label="female lowclass",color='blue')

ax2.set_xticklabels(["Survived","NotSurvived"],rotation=0)#xticklabels为坐标轴刻度

ax2.legend(["female/lowclass"],loc='best')  #legend为图例



ax3=fig.add_subplot(143,sharey=ax1)

df.Survived[df.Sex=='male'][df.Pclass!=3].value_counts().plot(kind='bar',label="female highclass",color='pink')

ax3.set_xticklabels(["Survived","NotSurvived"],rotation=0)#xticklabels为坐标轴刻度

ax3.legend(["male/highclass"],loc='best')  #legend为图例



ax4=fig.add_subplot(144,sharey=ax1)

df.Survived[df.Sex=='male'][df.Pclass==3].value_counts().plot(kind='bar',label="female lowclass",color='green')

ax4.set_xticklabels(["Survived","NotSurvived"],rotation=0)#xticklabels为坐标轴刻度

ax4.legend(["male/lowclass"],loc='best')  #legend为图例



plt.show()
#坚定了之前的判断
#再看看各登船港口的获救情况
fig=plt.figure()

fig.set(alpha=0.2)



Survived_0=df.Embarked[df.Survived==0].value_counts()

Survived_1=df.Embarked[df.Survived==1].value_counts()

df_embarked=pd.DataFrame({'Survived':Survived_1,'NotSurvived':Survived_0})

df_embarked.plot(kind='bar',stacked=True)

plt.title('Survival of different Embarked')

plt.xlabel('Embarked')

plt.ylabel('Number of people')



plt.show()
#看一下，兄弟姐妹/父母孩子有几人，是否对获救有影响

g=df.groupby(['SibSp','Survived'])

df_id=pd.DataFrame(g.count()['PassengerId'])

df_id
g=df.groupby(['Parch','Survived'])

df_id=pd.DataFrame(g.count()['PassengerId'])

df_id
#这里没有特别明显的规律，先作为备选特征

#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴

#cabin只有204个乘客有值，先看看它的分布

df.Cabin.value_counts()
#猜一下，也许前面的ABCDE是指的甲板位置，然后编号是房间号？

#cabin的属性，应该是类目型的，缺失值多还不集中。如果啊按照类目处理的话，太分散了。、

#加上有很多缺失值，要不先把Cabin缺失与否作为条件（虽然这部分信息缺失可能并非未登记，可能只是丢失了而已，所以这样做未必妥当）

#先在有无Cabin信息这个粗粒度上看看Survived的情况
fig=plt.figure()

fig.set(alpha=0.2)



Survived_cabin=df.Survived[pd.notnull(df.Cabin)].value_counts()

Survived_nocabin=df.Survived[pd.isnull(df.Cabin)].value_counts()

df_cabin=pd.DataFrame({'cabin':Survived_cabin,'nocabin':Survived_nocabin}).transpose()#矩阵转置操作

df_cabin.plot(kind='bar',stacked=True)

plt.title('Survival of Cabin')

plt.xlabel('cabin or not')

plt.ylabel('number of people')

plt.show()
#有cabin记录的貌似获救率高一点，先放一放吧
#之前是对感兴趣的属性有了大概的了解，下面就要处理这些数据，为建模做准备了

#特征工程

#首先从最突出的属性开始，cabin和age

#对于cabin，暂时按照刚才说的，按照cabin的有无数据，将这个属性处理成yes和no两种

#对于Age,通常遇到缺失值的情况，会有几种常见的处理方式

#通常遇到缺值的情况，我们会有几种常见的处理方式
#本例中，后两种处理方式可行，先试试拟合补全（虽然没有很多的背景可以供我们拟合，这不一定是一个很好的选择）

#这里用sklearn中的RandomForest来拟合一下缺失的年龄数据：randomforest是一个用在原始数据中做不同采样，

#建立多棵decision tree，再用average等来降低过拟合现象，提高结果的机器学习算法。
df['Fare'][df.Fare.isnull()]=df.Fare.mean()
df['Embarked'][df.Embarked.isnull()]='S'
from sklearn.ensemble import RandomForestRegressor

#使用RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    #把已有的数值型特征取出来丢进Random Forest Regression中

    age_df=df[['Age','Fare','Parch','SibSp','Pclass']]

    

    #乘客分为已知年龄和未知年龄两部分

    known_age=age_df[age_df.Age.notnull()].as_matrix()

    unknown_age=age_df[age_df.Age.isnull()].as_matrix()

    #Convert the frame to its Numpy-array representation.这里数据狂变为数组

    

    #y即目标年龄

    y=known_age[:,0]

    

    #x即特征属性值

    x=known_age[:,1:]

    

    #fit到RandomForestRegressor中

    rfr=RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)

    #n_jobs=-1，then the number of jobs is set to the number of cores.

    #random_state:the seed used by the random number generator

    rfr.fit(x,y)

    

    #用得到的模型进行位置年龄结果预测

    predictedAges=rfr.predict(unknown_age[:,1:])

    

    #用得到的预测结果填补缺失数据

    df.loc[(df.Age.isnull()),'Age']=predictedAges

    

    return df,rfr



def set_Cabin_type(df):

    df.loc[(df.Cabin.notnull()),'Cabin']="Yes"

    df.loc[(df.Cabin.isnull()),'Cabin']="No"

    return df



df,rfr=set_missing_ages(df)

df=set_Cabin_type(df)

    
df.isnull().sum()
#到这里，完成了“Age”"Cabin"的填充

#因为逻辑斯特回归建模的时候，需要输入的特征都是数值型特征，通常会先对类目型的特征因子化

#什么是因子化？以Cabin为例，原本一个属性维度，其值可以取['yes','no']

#而将其平展为'Cabin_yes''Cabin_no'两个属性，原本为yes的在此处'Cabin_yes'下取值为1，'Cabin_no'下取值为0

#同理，原本为no的在此处'Cabin_yes'下取值为0，'Cabin_no'下取值为1

#使用pandas的“get——dummies来完成这个工作，并拼接在原来的“df”之上
dummies_Cabin=pd.get_dummies(df['Cabin'],prefix='Cabin')#prefix列名称前面加上某个字符串

dummies_Embarked=pd.get_dummies(df['Embarked'],prefix='Embarked')

dummies_Sex=pd.get_dummies(df['Sex'],prefix='Six')

dummies_Pclass=pd.get_dummies(df['Pclass'],prefix='Pclass')



df=pd.concat([df,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)

df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)

df
#这样成功地把这些类目属性全都转化成1，0的数值属性

#但是'Age','Fare'两个属性，数值幅度变化太大了

#对于逻辑回归和梯度下降，各属性之间的scale差距太大，将会对收敛速度造成损害，甚至不收敛

#所以现需要用sklearn里面的preprocessing模块对这两个特征进行scaling，也就是将变化幅度较大的值特征化到[-1,1]之间
import sklearn.preprocessing as preprocessing

scaler=preprocessing.StandardScaler()

age_scale_param=scaler.fit(df[['Age']])

df['Age_scaled']=scaler.fit_transform(df[['Age']],age_scale_param)

fare_scale_param=scaler.fit(df[['Fare']])

df['Fare_scaled']=scaler.fit_transform(df[['Fare']],fare_scale_param)

df
#数据处理完成，需要把需要的属性值抽取出来，转成scikit-learn里面LogisticRegression可以处理的格式
#逻辑回归建模，把需要的特征字段取出来，转成numpy格式，使用sklearn里面的LogisticRegression建模

from sklearn import linear_model



#注：这里其实是整个数据集，包括训练集和测试集，所以先分离出来

test_df=df[df.Survived.isnull()]

train_df=df[df.Survived.notnull()]
#用正则取出我们想要的属性值'

train_final=train_df.filter(regex='Survived|Age_.*|SibSp|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

#|表示或.表示任意字符*表示任意个

train_np=train_final.as_matrix()#训练是必须先从DataFrame转换成Numpy



#y即Survival结果

y=train_np[:,0]



#x即特征属性值

x=train_np[:,1:]
#fit到RandomForestRegressor之中

clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)

clf.fit(x,y)



clf
#这样得到了一个model
#一般这里如果train和test是分开处理，则需要再进行对test进行上面train的出来

test_final=test_df.filter(regex='Age_.*|SibSp|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

#这里不用把test转换为numpy格式？



#预测

predictions=clf.predict(test_final)

result=pd.DataFrame({'PassengerId':test_df['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})

#这里把ID和预测结果写入DataFrame



result.to_csv("logistic_regression_predictions.csv",index=False)
from sklearn import cross_validation

#通常情况下这么做交叉验证：把train。csv分成两部分，一部分用于训练我们需要的模型，另一部分数据上看我们预测算法的效果
clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)

all_data=train_df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

x=all_data.as_matrix()[:,1:]

y=all_data.as_matrix()[:,0]

print (cross_validation.cross_val_score(clf,x,y,cv=5))
#做数据分割，按照训练数据：cv数据=7：#的比例

split_train,split_cv=cross_validation.train_test_split(train_df,test_size=0.3,random_state=0)

train_df_cv=split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

#生成模型

clf=linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)

clf.fit(train_df_cv.as_matrix()[:,1:],train_df_cv.as_matrix()[:,0])
#对cross validation数据进行预测

cv_df=split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions=clf.predict(cv_df.as_matrix()[:,1:])



original_data_train=pd.read_csv("../input/train.csv")

original_data_train
bad_cases=original_data_train.loc[original_data_train['PassengerId'].isin(split_cv[predictions!=cv_df.as_matrix()[:,0]]['PassengerId'].values)]

bad_cases
import numpy as np

import matplotlib.pyplot as plt

from sklearn.learning_curve import learning_curve



#用sklearn的learning_curve得到training_score和cv_score,使用matplotlib画出learning curve

def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):

   #画出data在某模型上的learning curve.

    #参数解释：estimator:你用的分类器，title:表格的标题，x:输入的feature,numpy类型，y:输入的target vector

    #ylim:tuple格式的（ymin,ymax）,设定图像中纵坐标的最低点和最高点

    #cv:做cross_validation的时候，数据分成的份数，其中一份作为cv集，其余n-1作为training（默认为3份）

    #n_jobs:并行的任务数，默认为1

    

    train_sizes,train_scores,test_scores=learning_curve(estimator,title,x,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes,verbose=verbose)

    

    train_scores_mean=np.mean(train_scores,axis=1)

    train_scores_std=np.std(train_scores,axis=1)

    test_scores_mean=np.mean(test_scores,axis=1)

    test_scores_std=np.std(test_scores,axis=1)

    

    if plot:

        plt.figure()

        plt.title(title)

        if ylim is not None:

            plt.ylim(*ylim)

        plt.xlabel("number of samples")

        plt.ylabel("score")

        plt.gca().invert_yaxis()

        plt.grid()

        

        plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="b")

        plt.fill_between(test_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="b")

        plt.plot(train_sizes,train_scores_mean,'o-',color="b",label="score in train")

        plt.plot(test_sizes,test_scores_mean,'o-',color="r",label="score in cv")

        

        plt.legend(loc="best")

        

        plt.draw()

        plt.show()

        plt.gca().invert_yaxis()

        

    midpoint=((train_scores_mean[-1]+train_scores_std[-1])+(test_score_mean[-1]-test_scores_std[-1]))/2

    diff=(train_scores_mean[-1]+train_scores_std[-1])+(test_score_mean[-1]-test_scores_std[-1])

    return minpoin,diff
plot_learning_curve(clf,"learning curve",x,y)
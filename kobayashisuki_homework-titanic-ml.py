# 这个ipython notebook主要是我解决Kaggle Titanic问题的思路和过程



import pandas as pd #数据分析

import numpy as np #科学计算

from pandas import Series,DataFrame



data_train = pd.read_csv("../input/titanic/train.csv",encoding='UTF-8')

#data_train.columns

data_train.head() 

#data_train[data_train.Cabin.notnull()]['Survived'].value_counts()
data_test = pd.read_csv("../input/titanic/test.csv",encoding='UTF-8')

data_test.head() 
data_train.info()
data_test.info()
data_train.describe()
data_test.describe()
%matplotlib inline

import matplotlib.pyplot as plt



#plt.rcParams['font.sans-serif']=['SimHei']

#plt.rcParams['axes.unicode_minus'] = False    #显示中文标题



fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图

data_train.Survived.value_counts().plot(kind='bar')# plots a bar graph of those who surived vs those who did not. 

plt.title(u"Survived") # puts a title on our graph

plt.ylabel(u"Number of People")  



plt.subplot2grid((2,3),(0,1))

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel(u"Number of People")

plt.title(u"Pclass")



plt.subplot2grid((2,3),(0,2))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel(u"Age")                         # sets the y axis lable

plt.grid(b=True, which='major', axis='y') # formats the grid line style of our graphs

plt.title(u"Survived")





plt.subplot2grid((2,3),(1,0), colspan=2)

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # plots a kernel desnsity estimate of the subset of the 1st class passanges's age

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"Age")# plots an axis lable

plt.ylabel(u"Density") 

plt.title(u"Pclass-Age")

plt.legend((u'Pclass_1', u'Pclass_2',u'Pclass_3'),loc='best') # sets our legend for our graph.





plt.subplot2grid((2,3),(1,2))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title(u"Embarked")

plt.ylabel(u"Number of People")  



plt.tight_layout()    #控制子图间距

plt.show()

 
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



def bar_chart(feature):

    survived=data_train[data_train['Survived']==1][feature].value_counts()

    dead=data_train[data_train['Survived']==0][feature].value_counts()

    df=pd.DataFrame([survived,dead])

    df.index=['survived','dead']

    df.plot(kind='bar',stacked=True, figsize=(7,4))

    

bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Embarked')
bar_chart('SibSp')
bar_chart('Parch')
#看看各乘客等级的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df=pd.DataFrame({u'Survived':Survived_1, u'Dead':Survived_0})

df.plot(kind='bar', stacked=True)

plt.title(u"Survived")

plt.xlabel(u"Pclass") 

plt.ylabel(u"Number of People") 



plt.show()
#看看各性别的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

df=pd.DataFrame({u'Male':Survived_m, u'Female':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title(u"Survived")

plt.xlabel(u"Sex") 

plt.ylabel(u"Number of People")



plt.show()
#然后我们再来看看各种舱级别情况下各性别的获救情况

fig=plt.figure()

fig.set(alpha=0.65) # 设置图像透明度，无所谓

plt.title(u"Survived")



ax1=fig.add_subplot(141)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels([u"1", u"0"], rotation=0)

ax1.legend([u"Female/12"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"0", u"1"], rotation=0)

plt.legend([u"Female/3"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"0", u"1"], rotation=0)

plt.legend([u"Male/12"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"0", u"1"], rotation=0)

plt.legend([u"Male/3"], loc='best')



plt.tight_layout()    #控制子图间距

plt.show()
gs = data_train.groupby(['SibSp','Survived'])

df = pd.DataFrame(gs.count()['PassengerId'])

df
gp = data_train.groupby(['Parch','Survived'])

df = pd.DataFrame(gp.count()['PassengerId'])

df
#ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，不纳入考虑的特征范畴

#cabin只有204个乘客有值，我们先看看它的一个分布

data_train.Cabin.value_counts()
#cabin的值计数太分散了，绝大多数Cabin值只出现一次。感觉上作为类目，加入特征未必会有效

#那我们一起看看这个值的有无，对于survival的分布状况，影响如何吧

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()

Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()

df=pd.DataFrame({u'Yes':Survived_cabin, u'No':Survived_nocabin}).transpose()

df.plot(kind='bar', stacked=True)

plt.title(u"Survived")

plt.xlabel(u"Cabin") 

plt.ylabel(u"Nummber of Peoole")



plt.show()



#似乎有cabin记录的乘客survival比例稍高，那先试试把这个值分为两类，有cabin值/无cabin值，一会儿加到类别特征好了
data_train['Title'] = data_train.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

data_train.Title.value_counts()
List=data_train.Title.value_counts().index[4:].tolist()

mapping={}

for s in List:

    mapping[s]='Else'

data_train['Title']=data_train['Title'].map(lambda x: mapping[x] if x in mapping else x)

data_train.Title.value_counts()
from sklearn.ensemble import RandomForestRegressor

 

### 使用 RandomForestClassifier 填补缺失的年龄属性

def set_missing_ages(df):

    

    # 把已有的数值型特征取出来丢进Random Forest Regressor中

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]



    # 乘客分成已知年龄和未知年龄两部分

    known_age = age_df[age_df.Age.notnull()].as_matrix()

    unknown_age = age_df[age_df.Age.isnull()].as_matrix()



    # y即目标年龄

    y = known_age[:, 0]



    # X即特征属性值

    X = known_age[:, 1:]



    # fit到RandomForestRegressor之中

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(X, y)

    

    # 用得到的模型进行未知年龄结果预测

    predictedAges = rfr.predict(unknown_age[:, 1::])

    

    # 用得到的预测结果填补原缺失数据

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df, rfr



def set_Cabin_type(df):

    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"

    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    return df



#data_train, rfr = set_missing_ages(data_train)



mean_ages = np.zeros(5)

mean_ages[0]=np.average(data_train[data_train['Title'] == 'Miss']['Age'].dropna())

mean_ages[1]=np.average(data_train[data_train['Title'] == 'Mrs']['Age'].dropna())

mean_ages[2]=np.average(data_train[data_train['Title'] == 'Mr']['Age'].dropna())

mean_ages[3]=np.average(data_train[data_train['Title'] == 'Master']['Age'].dropna())

mean_ages[4]=np.average(data_train[data_train['Title'] == 'Else']['Age'].dropna())



data_train.loc[ (data_train.Age.isnull()) & (data_train.Title == 'Miss') ,'Age'] = mean_ages[0]

data_train.loc[ (data_train.Age.isnull()) & (data_train.Title == 'Mrs') ,'Age'] = mean_ages[1]

data_train.loc[ (data_train.Age.isnull()) & (data_train.Title == 'Mr') ,'Age'] = mean_ages[2]

data_train.loc[ (data_train.Age.isnull()) & (data_train.Title == 'Master') ,'Age'] = mean_ages[3]

data_train.loc[ (data_train.Age.isnull()) & (data_train.Title == 'Else') ,'Age'] = mean_ages[4]



data_train = set_Cabin_type(data_train)

data_train.Ticket.value_counts()
#看看各性别的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_m = data_train.Survived[(data_train.Ticket == 'CA. 2343')&(data_train.Sex == 'male')].value_counts()

Survived_f = data_train.Survived[(data_train.Ticket == 'CA. 2343')&(data_train.Sex == 'female')].value_counts()

df=pd.DataFrame({u'Male':Survived_m, u'Female':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title(u"Survived")

plt.xlabel(u"Sex") 

plt.ylabel(u"Number of People")



plt.show()
#看看各性别的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_m = data_train.Survived[(data_train.Ticket == '347082')&(data_train.Sex == 'male')].value_counts()

Survived_f = data_train.Survived[(data_train.Ticket == '347082')&(data_train.Sex == 'female')].value_counts()

df=pd.DataFrame({u'Male':Survived_m, u'Female':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title(u"Survived")

plt.xlabel(u"Sex") 

plt.ylabel(u"Number of People")



plt.show()
pd.crosstab(pd.cut(data_train.Age,8)[:len(data_train)],data_train.Survived).plot.bar(stacked=True)
data_train['Family']=data_train['SibSp']+data_train['Parch']

data_train['Sex_Pclass'] = data_train.Sex + "_" + data_train.Pclass.map(str)



data_train['AgeRank']=data_train['Age']

data_train.loc[ (data_train.Age<=10) ,'AgeRank'] = 'child'

data_train.loc[ (data_train.Age>60),'AgeRank'] = 'aged'

data_train.loc[ (data_train.Age>10) & (data_train.Age <=30) ,'AgeRank'] = 'adult'

data_train.loc[ (data_train.Age>30) & (data_train.Age <=60) ,'AgeRank'] = 'senior'



data_train.info()
data_train["Fname"] = data_train.Name.apply(lambda name: name.split(",")[0])

data_train.Fname.value_counts()
#有女性死亡的家庭，除了1岁以下的婴儿外，家庭成员全部死亡。



dead_train = data_train[data_train["Survived"] == 0]

dead_fname_ticket = dead_train[(dead_train["Sex"] == "female") & (dead_train["Family"] != 0)][["Fname", "Ticket"]]

data_train["dead_family"] = np.where(data_train["Fname"].isin(dead_fname_ticket["Fname"])\

                                & data_train["Ticket"].isin(dead_fname_ticket["Ticket"]), 1, 0)



#家庭中若有大于18岁男性存活，则该家庭全部存活。



live_train = data_train[data_train["Survived"] == 1]

live_fname_ticket = live_train[(live_train["Sex"] == "male") & (live_train["Family"] !=0) & ((live_train["Age"] >= 18) )][["Fname", "Ticket"]]

data_train["live_family"] = np.where(data_train["Fname"].isin(live_fname_ticket["Fname"])\

                                & data_train["Ticket"].isin(live_fname_ticket["Ticket"]), 1, 0)
#看看各性别的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived = data_train.Survived[(data_train["Fname"].isin(dead_fname_ticket["Fname"]))&(data_train["Ticket"].isin(dead_fname_ticket["Ticket"]))].value_counts()

#Survived = data_train.Survived[(data_train["Fname"].isin(live_fname_ticket["Fname"]))&(data_train["Ticket"].isin(live_fname_ticket["Ticket"]))].value_counts()

df=pd.DataFrame({u'Number of People':Survived})

df.plot(kind='bar', stacked=True)

plt.title(u"Survived")

plt.xlabel(u"dead_family") 

plt.ylabel(u"Number of People")



plt.show()
#看看各性别的获救情况

fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



#Survived = data_train.Survived[(data_train["Fname"].isin(dead_fname_ticket["Fname"]))&(data_train["Ticket"].isin(dead_fname_ticket["Ticket"]))].value_counts()

Survived = data_train.Survived[(data_train["Fname"].isin(live_fname_ticket["Fname"]))&(data_train["Ticket"].isin(live_fname_ticket["Ticket"]))].value_counts()

df=pd.DataFrame({u'Number of People':Survived})

df.plot(kind='bar', stacked=True)

plt.title(u"Survived")

plt.xlabel(u"live_family") 

plt.ylabel(u"Number of People")



plt.show()
data_train.describe()

#data_train
# 因为逻辑回归建模时，需要输入的特征都是数值型特征

# 我们先对类目型的特征离散/因子化

# 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性

# 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0

# 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1

# 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')



dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')



dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')



dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')



dummies_Sex_Pclass = pd.get_dummies(data_train['Sex_Pclass'], prefix= 'Sex_Pclass')



dummies_AgeRank = pd.get_dummies(data_train['AgeRank'], prefix= 'AgeRank')



dummies_Title = pd.get_dummies(data_train['Title'], prefix= 'Title')



df = pd.concat([data_train, dummies_Cabin,  dummies_Sex, dummies_Pclass, dummies_AgeRank, dummies_Title, dummies_Sex_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'AgeRank', 'Title', 'Sex_Pclass' ,'Fname'], axis=1, inplace=True)



#df['Sex_Pclass_male_4']=df['Sex_Pclass_male_2']+df['Sex_Pclass_male_3']

#df.drop(['Sex_Pclass_male_2','Sex_Pclass_male_3'], axis=1, inplace=True)



#df['Sex_Pclass_female_4']=df['Sex_Pclass_female_2']+df['Sex_Pclass_female_3']

#df.drop(['Sex_Pclass_female_2','Sex_Pclass_female_3'], axis=1, inplace=True)



df.columns
# 接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内

# 这样可以加速logistic regression的收敛

import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()



age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)



fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)



df.columns
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模

from sklearn import linear_model



train_df = df.filter(regex='Survived|Family|dead_family|live_family|AgeRank|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Title_.*')

#train_np = train_df.as_matrix()

train_np = train_df.values

# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到RandomForestRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', solver='liblinear', tol=1e-6)

clf.fit(X, y)

    

clf
X.shape
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0



# 接着我们对test_data做和train_data中一致的特征变换

# 首先用同样的RandomForestRegressor模型填上丢失的年龄



data_test['Family']=data_train['SibSp']+data_train['Parch']

data_test['Sex_Pclass'] = data_test.Sex + "_" + data_test.Pclass.map(str)



data_test['Title'] = data_test.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

List=data_test.Title.value_counts().index[4:].tolist()

mapping={}

for s in List:

    mapping[s]='Else'

data_test['Title']=data_test['Title'].map(lambda x: mapping[x] if x in mapping else x)



#tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

#null_age = tmp_df[data_test.Age.isnull()].as_matrix()



# 根据特征属性X预测年龄并补上

#X = null_age[:, 1:]

#predictedAges = rfr.predict(X)

#data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges



mean_ages = np.zeros(5)

mean_ages[0]=np.average(data_test[data_test['Title'] == 'Miss']['Age'].dropna())

mean_ages[1]=np.average(data_test[data_test['Title'] == 'Mrs']['Age'].dropna())

mean_ages[2]=np.average(data_test[data_test['Title'] == 'Mr']['Age'].dropna())

mean_ages[3]=np.average(data_test[data_test['Title'] == 'Master']['Age'].dropna())

mean_ages[4]=np.average(data_test[data_test['Title'] == 'Else']['Age'].dropna())



data_test.loc[ (data_test.Age.isnull()) & (data_test.Title == 'Miss') ,'Age'] = mean_ages[0]

data_test.loc[ (data_test.Age.isnull()) & (data_test.Title == 'Mrs') ,'Age'] = mean_ages[1]

data_test.loc[ (data_test.Age.isnull()) & (data_test.Title == 'Mr') ,'Age'] = mean_ages[2]

data_test.loc[ (data_test.Age.isnull()) & (data_test.Title == 'Master') ,'Age'] = mean_ages[3]

data_test.loc[ (data_test.Age.isnull()) & (data_test.Title == 'Else') ,'Age'] = mean_ages[4]



data_test['AgeRank']=data_test['Age']

data_test.loc[ (data_test.Age<=10) ,'AgeRank'] = 'child'

data_test.loc[ (data_test.Age>60),'AgeRank'] = 'aged'

data_test.loc[ (data_test.Age>10) & (data_test.Age <=30) ,'AgeRank'] = 'adult'

data_test.loc[ (data_test.Age>30) & (data_test.Age <=60) ,'AgeRank'] = 'senior'



data_test = set_Cabin_type(data_test)



data_test["Fname"] = data_test.Name.apply(lambda name: name.split(",")[0])

#data_train.Fname.value_counts()



#有女性死亡的家庭，除了1岁以下的婴儿外，家庭成员全部死亡。（分数第一次有大的提升，也是添加了该特征之后）

dead_fname_ticket = dead_train[(dead_train["Sex"] == "female") & (dead_train["Family"] != 0)][["Fname", "Ticket"]]

data_test["dead_family"] = np.where(data_test["Fname"].isin(dead_fname_ticket["Fname"])\

                                & data_test["Ticket"].isin(dead_fname_ticket["Ticket"]), 1, 0)



#家庭中若有大于18岁男性存活，或年龄为nan的男性存活，则该家庭全部存活。（加入该特征后，公分达到了0.8）

live_fname_ticket = live_train[(live_train["Sex"] == "male") & (live_train["Family"] !=0) & ((live_train["Age"] >= 18) )][["Fname", "Ticket"]]

data_test["live_family"] = np.where(data_test["Fname"].isin(live_fname_ticket["Fname"])\

                                & data_test["Ticket"].isin(live_fname_ticket["Ticket"]), 1, 0)



dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

dummies_Sex_Pclass = pd.get_dummies(data_test['Sex_Pclass'], prefix= 'Sex_Pclass')

dummies_AgeRank = pd.get_dummies(data_test['AgeRank'], prefix= 'AgeRank')

dummies_Title = pd.get_dummies(data_test['Title'], prefix= 'Title')



df_test = pd.concat([data_test, dummies_Cabin, dummies_Sex, dummies_Pclass, dummies_AgeRank, dummies_Title, dummies_Sex_Pclass], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'AgeRank', 'Title', 'Sex_Pclass', 'Fname'], axis=1, inplace=True)



#df_test['Sex_Pclass_male_4']=df_test['Sex_Pclass_male_2']+df_test['Sex_Pclass_male_3']

#df_test.drop(['Sex_Pclass_male_2','Sex_Pclass_male_3'], axis=1, inplace=True)



#df_test['Sex_Pclass_female_4']=df_test['Sex_Pclass_female_2']+df_test['Sex_Pclass_female_3']

#df_test.drop(['Sex_Pclass_female_2','Sex_Pclass_female_3'], axis=1, inplace=True)



df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)

df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)



df_test.columns
test = df_test.filter(regex='Family|dead_family|live_family|AgeRank|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Title_.*')

predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions_submission01.csv", index=False)

pd.read_csv("logistic_regression_predictions_submission01.csv")
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
# from sklearn import cross_validation

# 参考https://blog.csdn.net/cheneyshark/article/details/78640887 ， 0.18版本中，cross_validation被废弃

# 改为下面的从model_selection直接import cross_val_score 和 train_test_split

from sklearn.model_selection import cross_val_score, train_test_split



 #简单看看打分情况

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)

all_data = df.filter(regex='Survived|Family|dead_family|live_family|AgeRank|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Title_.*')

X = all_data.values[:,1:]

y = all_data.values[:,0]



cv_scores=cross_val_score(clf, X, y, cv=10)

print(cv_scores,"\n")

print("Average score:",np.mean(cv_scores))
# 分割数据，按照 训练数据:cv数据 = 3:1的比例

# split_train, split_cv = cross_validation.train_test_split(df, test_size=0.25, random_state=0)

split_train, split_cv = train_test_split(df, test_size=0.25, random_state=0)



train_df = split_train.filter(regex='Survived|Family|dead_family|live_family|AgeRank|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Title_.*')

# 生成模型

clf = linear_model.LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)

clf.fit(train_df.values[:,1:], train_df.values[:,0])



# 对cross validation数据进行预测



cv_df = split_cv.filter(regex='Survived|Family|dead_family|live_family|AgeRank|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Title_.*')

predictions = clf.predict(cv_df.values[:,1:])



# 去除预测错误的case看原始dataframe数据

#split_cv['PredictResult'] = predictions

origin_data_train = pd.read_csv("../input/titanic/train.csv",encoding='UTF-8')

bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]

bad_cases.head(10)
import numpy as np

import matplotlib.pyplot as plt

# from sklearn.learning_curve import learning_curve

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

        plt.xlabel(u"Number of Sample")

        plt.ylabel(u"Score")

        plt.gca().invert_yaxis()

        plt.grid()

    

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 

                         alpha=0.1, color="b")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 

                         alpha=0.1, color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"Test Score")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"Cross Val Score")

    

        plt.legend(loc="best")

        

        plt.draw()

        plt.gca().invert_yaxis()

        plt.show()

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff



plot_learning_curve(clf, u"Learning Curve", X, y)
from sklearn.ensemble import BaggingRegressor



train_df = df.filter(regex='Survived|Family|dead_family|live_family|AgeRank|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Title_.*|Embarked_.*')

train_np = train_df.values



# y即Survival结果

y = train_np[:, 0]



# X即特征属性值

X = train_np[:, 1:]



# fit到BaggingRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', solver='liblinear', tol=1e-6)

bagging_clf = BaggingRegressor(clf, n_estimators=10, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

bagging_clf.fit(X, y)



test = df_test.filter(regex='Survived|Family|dead_family|live_family|AgeRank|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Title_.*|Embarked_.*')

predictions = bagging_clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

result.to_csv("logistic_regression_predictions_submission02.csv", index=False)
pd.read_csv("logistic_regression_predictions_submission02.csv").head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score



rf=RandomForestClassifier()

scores=cross_val_score(rf, X, y, cv=10)

print(scores)

print(scores.mean())



#print(classification_report(y_pred_rf, gender_submission['Survived']))

#print(y_pred_rf)



gender_submission=pd.read_csv('../input/titanic/gender_submission.csv')



y_pred_rf=rf.fit(X, y).predict(test)

data={'PassengerId': gender_submission['PassengerId'],'Survived':y_pred_rf}

result_rf=pd.DataFrame(data)

result_rf.to_csv('logistic_regression_predictions_submission03.csv', index=False)

result_rf=pd.read_csv('logistic_regression_predictions_submission03.csv')

result_rf.head()
print(classification_report(y_pred_rf, gender_submission['Survived']))

#print(y_pred_rf)
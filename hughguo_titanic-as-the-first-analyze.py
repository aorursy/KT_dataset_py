#先import一些包
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
#再做一些简单处理
train_file = '../input/train.csv'
test_file = '../input/test.csv'
train = pd.read_csv(train_file)
train = train.set_index('PassengerId').drop(['Name','Ticket'],axis = 1)
test = pd.read_csv(test_file)
test = test.set_index('PassengerId').drop(['Name','Ticket'],axis = 1)
#看看训练集情况
train.head()
#看看测试集情况
test.head()
train.describe()
train.info()
### Combine

fig = plt.figure()
fig = plt.figure(figsize = (16,8))
plt.style.use('ggplot')

plt.subplot2grid((2,3),(0,0))
train.Survived.value_counts().plot(kind='bar')# 柱状图 
plt.title('Survived=1,0') # 标题
plt.ylabel('#')  

plt.subplot2grid((2,3),(0,1))
train.Pclass.value_counts().plot(kind="bar")
plt.ylabel('#')
plt.title('PClass')

plt.subplot2grid((2,3),(0,2))
plt.scatter(train.Survived, train.Age)
plt.ylabel('Age') # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y') 
plt.title('Survived by Age')


plt.subplot2grid((2,3),(1,0), colspan=2)
train.Age[train.Pclass == 1].plot(kind='kde')   
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
plt.xlabel('Age')# plots an axis lable
plt.ylabel('P') 
plt.title('Age by PClass')
plt.legend(('L1','L2','L3'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts().plot(kind='bar')
plt.title('Embarked')
plt.ylabel('#')  
plt.show()
##Survive by LEVEL

Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df=pd.DataFrame({'Surv':Survived_1, 'No_Surv':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('Survive by Level')
plt.xlabel('level') 
plt.ylabel('#') 
plt.show()
##Survive by Gender

#Survived_m = train.Survived[train.Sex == 'male'].value_counts()
#Survived_f = train.Survived[train.Sex == 'female'].value_counts()
#df=pd.DataFrame({'Male':Survived_m, 'Female':Survived_f})
#df.plot(kind='bar', stacked=True)
#plt.title('Survive by Gender')
#plt.xlabel('Gender') 
#plt.ylabel('#')
#plt.show()

Survived_0 = train.Sex[train.Survived == 0].value_counts()
Survived_1 = train.Sex[train.Survived == 1].value_counts()
df=pd.DataFrame({'Surv':Survived_1, 'No_Surv':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title('Survive by Gender')
plt.xlabel('Gender') 
plt.ylabel('#') 
plt.show()
###Level + Gender

fig = plt.figure(figsize = (16,8))
#plt.style.use('ggplot')
plt.title('Level vs Gender')
plt.axis('off')
ax1=fig.add_subplot(161)
train.Survived[train.Sex == 'female'][train.Pclass == 1].value_counts().plot(kind='bar', label="female L1", color='red')
plt.legend(['Female | L1'], loc='best')

ax2=fig.add_subplot(162, sharey=ax1)
train.Survived[train.Sex == 'female'][train.Pclass == 2].value_counts().plot(kind='bar', label="female L2", color='salmon')
plt.legend(['Female | L2'], loc='best')

ax3=fig.add_subplot(163, sharey=ax1)
train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts().plot(kind='bar', label='female, L3', color='pink')
plt.legend(['Female | L3'], loc='best')

ax4=fig.add_subplot(164, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass == 1].value_counts().plot(kind='bar', label='male, L1',color='lightblue')
plt.legend(['Male | L1'], loc='best')

ax5=fig.add_subplot(165, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass == 2].value_counts().plot(kind='bar', label='male, L2',color='cornflowerblue')
plt.legend(['Male | L2'], loc='best')

ax6=fig.add_subplot(166, sharey=ax1)
train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts().plot(kind='bar', label='male L3', color='steelblue')
plt.legend(['Male | L3'], loc='best')

plt.show()
### put them in bin
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
level = ['0~10', '11~20', '21~30', '31~40', '41~50', '51~60', '61~70', '71~80','81+']
train['Age_Group'] = pd.cut(train['Age'], bins=bins, labels=level)

###
titanic_data_bucket = train.groupby(['Age_Group', 'Survived'])['Survived'].count().unstack().fillna(0)
titanic_data_bucket.rename(columns={1:'yes', 0:'no'}, inplace=True)

fig = plt.figure(figsize=(15, 7))
ax_1 = fig.add_subplot(121)
titanic_data_bucket['yes'].plot(kind='bar',alpha=0.5)              # 各年龄段乘客存活/未存活数分布，通过柱状图展示
titanic_data_bucket['no'].plot(kind='bar',alpha=0.5)
ax_1.set_xlabel('Age')
ax_1.set_ylabel('Survive / Total #')
ax_1.set_title('Survived shared Age')
plt.xticks(rotation=45)
ax_2 = fig.add_subplot(122)
titanic_data_bucket[['no_percent','yes_percent']] \
= titanic_data_bucket.apply(lambda x: x / x.sum()*100, axis=1)
print (titanic_data_bucket)

titanic_data_bucket['yes_percent'].plot(kind='bar', stacked=True)  # 各年龄段乘客存活百分比
ax_2.set_xlabel('Age')
ax_2.set_ylabel('Survive%')
ax_2.set_title('Survived shared %')
plt.xticks(rotation=45)
plt.show()
Si_b_Sp = train.groupby(['SibSp','Survived']).count()['Pclass']
Si_b_Sp
Par_ch = train.groupby(['Parch','Survived']).count()['Pclass']
Par_ch
train.Cabin.value_counts()
Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()
df=pd.DataFrame({'Y':Survived_cabin, 'N':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title('Survive with Carbin')
plt.xlabel('Y/N') 
plt.ylabel('#')
plt.show()
#补Age
from sklearn.ensemble import RandomForestRegressor

def missing_ages(df):
    age = df[['Age','Fare','Parch','SibSp','Pclass']]
    #分为有/没有Age两部分
    age_y = age[age.Age.notnull()].values
    age_n = age[age.Age.isnull()].values
    #y = target
    y = age_y[:,0]
    #N = profile
    X = age_y[:,1:]
    
    #fit
    rf = RandomForestRegressor(random_state = 0, n_estimators = 100, n_jobs = -1)
    rf.fit(X,y)
    
    predict_age = rf.predict(age_n[:,1:])
    
    df.loc[(df.Age.isnull()), 'Age'] = predict_age
    
    return df, rf

#改Cabin
def missing_cabin(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Y'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'N'
    return df

train, rfr = missing_ages(train)
train_data = missing_cabin(train)   
train_data.info()
train_data.head()
#快乐的dummy一下，变为特征因子化
dummy_Pclass = pd.get_dummies(train_data['Pclass'], prefix = 'Pclass')
dummy_Sex = pd.get_dummies(train_data['Sex'], prefix = 'Sex')
dummy_Cabin = pd.get_dummies(train_data['Cabin'], prefix ='Cabin')
dummy_Embarked = pd.get_dummies(train_data['Embarked'], prefix = 'Embarked')

df = pd.concat([train_data, dummy_Pclass, dummy_Sex, dummy_Cabin, dummy_Embarked], axis = 1)
df = df.drop(['Pclass','Sex','Cabin','Embarked'], axis = 1)
df.head()
#Age和Fare快乐的归一化


from sklearn.preprocessing import StandardScaler
#机智的把需要归一化的数据找出来单独成一个Data_Frame
#temp = df.loc[:,['Age','Fare']].reset_index().drop(['PassengerId'], axis = 1)
#temp.head()
##并不机智...我们用归一化模板吧因为一会儿还要用
scaler = StandardScaler()
age_scale = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scale'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale)
fare_scale = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scale'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale)
#min_max = sklearn.preprocessing.MinMaxScaler()
#temp = scaler.fit_transform(temp)
#temp_df = pd.DataFrame(temp)
#temp.head()

#df = df.reset_index()
#df['Age_scale'] = temp_df[0]
#df['Fare_scale'] = temp_df[1]
df = df.drop(['Age','Fare'], axis = 1)
df.head()
from sklearn import linear_model

# 用正则取出我们要的feature
#train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#train_np = train_df.as_matrix()
#as_matrix已经要被停用了，聪明的人都用values ^_^
train_np = df.values
# y = Survival结果
y = train_np[:, 0]

# X = 特征属性值
X = train_np[:, 1:]

# fit到LogisticRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

clf

### training model is DONE~ ###
test.head()
##def missing_ages(df) ==> rfr
##def missing_cabin(df) ==> data_frame

#Fare
test.loc[ (test.Fare.isnull()), 'Fare' ] = 0
test_temp = test[['Age','Fare','Parch','SibSp','Pclass']]

#Age
age_null = test_temp[test.Age.isnull()].values
X = age_null[:,1:] #同一规则
pred_age = rfr.predict(X)
test.loc[(test.Age.isnull()),'Age'] = pred_age
#refresh the cabin~
test = missing_cabin(test)
#dummies the str
dummies_Pclass = pd.get_dummies(test['Pclass'], prefix= 'Pclass')
dummies_Sex = pd.get_dummies(test['Sex'], prefix= 'Sex')
dummies_Cabin = pd.get_dummies(test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(test['Embarked'], prefix= 'Embarked')

#update the test table with same rules / fit as train
df_test = pd.concat([test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test = df_test.drop(['Pclass', 'Sex', 'Cabin', 'Embarked'], axis=1)

df_test['Age_scale'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale)
df_test['Fare_scale'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale)

df_test = df_test.drop(['Age','Fare'], axis = 1)
df_test.head()
test_test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test_test)
test = test.reset_index()
result = pd.DataFrame({'PassengerId':test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv('submission.csv', index=False)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
train_heatmap = train.astype('str')
for column in train_heatmap.columns:
    train_heatmap[column] = LabelEncoder().fit_transform(train_heatmap[column])

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_heatmap.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
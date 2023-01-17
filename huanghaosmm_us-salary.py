import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import warnings

import xgboost

from sklearn import model_selection

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
import tensorflow as tf
#Set columns names

col_Name = ['Age', 'Work_Class', 'Fnlwgt', 'Education', 'Education_Num', 'Marital_Status', 'Occupation',

           'Relationship', 'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_Per_Week', 'Native_Country','Salary'] 

train = pd.read_csv('../input/adult.data.csv',sep = ',', header= None)

train.columns = col_Name

test1 = pd.read_csv('../input/adult.test.csv', sep = ',', header = None)          

test1.columns = col_Name
#train null value and info

train.info()

print(train.isnull().sum())

print()



"""

尽管显示空值为0，但是经过观察有很多值为?，也要视为空值，并尝试填充

"""
#Get all data and clean

data_cleaner = [train, test1]

train['Salary'] = train['Salary'].apply(lambda x : 1  if x == ' >50K' else 0)

test1['Salary'] = test1['Salary'].apply(lambda x : 1  if x == ' >50K.' else 0)



train.Salary.value_counts()
train1 = train.copy()

def cc(x):

    return sum(x==' ?')



train.apply(cc)

train.loc[train['Work_Class'] == ' ?'].apply(cc)
## EDA

# Age versus Salary

train['Age'].describe()
# Age Distribution

bins = np.arange(0,90,5)

sns.distplot(train['Age'], bins=bins)

plt.title('Age Distribution')

plt.show()

"""

Age为一个左尾正态分布，从20到30上升过大

"""
# Age versus Salary

AgePlot = sns.FacetGrid(data = train , hue = 'Salary', aspect=3)

AgePlot.map(sns.distplot, 'Age', bins = bins)

AgePlot.set(ylim = (0, 0.05))

AgePlot.set(xlim = (10,90))

AgePlot.add_legend()

plt.show()

"""

明显Age和Salary会有关系，没钱的更集中在25岁左右，

"""

# 工作与工资比例

plt.figure(figsize=(14,7))

sns.barplot(y = 'Salary', x = 'Work_Class',data = train)

"""

Without-pay, Never-worked没有工资，Self-emp-inc 和 Federal-gov工资比例最高，

"""
#发现Work_Class变量前面有空格，先删除

for all_data in data_cleaner:

    all_data['Work_Class'] = all_data['Work_Class'].apply(lambda x : x.strip())
#Work_Class分布图

plt.figure(figsize=(10,7))

plt.hist(x = [train[train['Salary']==0]['Work_Class'],train[train['Salary']==1]['Work_Class']], stacked=True,

        color = ['g', 'r'], label=['<50K', '>50K'])

plt.legend()

"""

绝大多数人为Private职业，但是工资1占比不高,withou-pay 和 Never-worked的人太少，而且工资都为0

"""
#Work_class数量统计

print('Workclass总数:','\n', train['Work_Class'].value_counts())

print('-'*50)

print('工资为0数目:', '\n', train[train['Salary']==0]['Work_Class'].value_counts())
#Work_Class中？很多， 看一下他在年龄中的分布情况

bins = np.arange(0,95,1)

work_age = train[train['Work_Class'] =='?']

sns.distplot(work_age['Age'],bins = bins)

plt.title('Missing Values in Work Class with Age')

"""

看的出来年龄在24和62左右空值最多，按经验来看24可能是学生为Never-worked团体，而60岁以上为退休集体可能Without-pay

"""
print(train[train['Work_Class']=='Never-worked']['Age'].describe())

"""

可以看到'never-worked'人群平均年龄只有20,那么Work-Class可考虑填充为Never-worked

"""
print(train[train['Work_Class']=='Without-pay']['Age'])

plt.hist(train[train['Work_Class']=='Without-pay']['Age'])

"""

without-pay大部分也是20左右年轻人和60以上老年人，考虑用without-pay进行填补"""
#尝试填补年轻人Work_Class再观察

train1 = train.copy()

train1.loc[(train1['Age']<=24) & (train1['Work_Class']=='?'),'Work_Class']='Never-worked'
#再看分布图

bins = np.arange(0,95,1)

age_work = train1[train1['Work_Class'] == '?']

sns.distplot(age_work['Age'],bins = bins)

plt.xlim(24,95)

"""

超过60岁还有缺失值，按常理来说应该很多是Without-pay"""
#观察有无工作与年龄的关系

age_with_work = train1[train1['Work_Class']!='?'].groupby('Age').count().max(1)/train1['Age'].value_counts()

age_without_work =  train1[train1['Work_Class']=='?'].groupby('Age').count().max(1)/train1['Age'].value_counts()

age_with_work = (age_with_work - age_with_work.mean())/age_with_work.std()

age_without_work = (age_without_work - age_without_work.mean())/age_without_work.std()
age_with_work=age_with_work.fillna(0)

age_without_work=age_without_work.fillna(0)

diff = age_with_work-age_without_work

diff.loc[(85,86,88),]=0
age = np.arange(17,90,1)

plt.bar(age,diff)

"""

在考虑了年龄的分布后，可以看出大于60岁后，Work_class为空值的都比非空的多，所以可以将60岁以上Work-class为空的设定为retired"""
train1.loc[(train1['Work_Class']=='?') & (train1['Age'] >= 60),'Work_Class'] = 'Retired' 
bins = np.arange(0,95,1)

work_age1 = train1[train1['Work_Class'] =='?']

sns.distplot(work_age1['Age'],bins = bins)
#Occupation vs salary

plt.figure(figsize = (18,7))

sns.barplot(x = 'Occupation', y = 'Salary' , data = train1)
#Education与Salary

print(train1['Education'].value_counts())

print(train1['Education_Num'].value_counts())

"""

这2个为相同的东西，删去后一个变量"""
for all_data in data_cleaner:

    all_data['Education'] = all_data['Education'].apply(lambda x : x.strip())
train1 = train.copy()

train1['Education'] = train1['Education'].apply(lambda x : x.strip())
salary_ratio = []

for x in np.unique(train1['Education']):

    salary_ratio.append(train1[(train1['Education'] == x)&(train1['Salary']==1)].count().max()/

                       train1[train1['Education'] == x].count().max())
salary_class_ratio ={'Education':np.unique(train1['Education']), 

                    'Salary' : salary_ratio}

salary_class_ratio = pd.DataFrame(salary_class_ratio)

salary_class_ratio = salary_class_ratio.sort_values(['Salary']).reset_index(drop=True)
#各个Education

plt.figure(figsize=(18,7))

salary_class_ratio['Salary'] = salary_class_ratio['Salary'].apply(lambda x: np.round(x,4))

ax = sns.barplot(y = 'Salary', x = 'Education', data = salary_class_ratio )

for i,v in enumerate(salary_class_ratio['Salary'].iteritems()):

    ax.text(i ,v[1], "{:,}".format(v[1]), color='m', va ='bottom', rotation=45)

plt.tight_layout()

plt.title('Salary with Education')

plt.show()

"""

看的出来文化水平越高工资比例越高"""
plt.figure(figsize=(18,7))

Education = pd.DataFrame(train1['Education'].value_counts()).sort_values(by='Education')

sns.barplot(x= Education.index, y =Education['Education'])

plt.ylabel('counts')

plt.title('Education Bar Plot')

plt.show()

"""

可以看到高中毕业最多，考虑将小学生， 高中生分别合并"""
#Marital_Status与Salary

train1['Marital_Status'].value_counts()
plt.figure(figsize=(13,8))

sns.barplot(y = 'Salary', x = 'Marital_Status', data = train1)

plt.title('Salary with Marital_Status')

"""

结婚了的工资相对较高

"""
#Salary vs Marital_status

plt.figure(figsize=(11,7))

plt.hist(x = [train1[train1['Salary']==0]['Marital_Status'],train1[train1['Salary']==1]['Marital_Status']],

        color = ['g', 'r'], label=['<50K', '>50K'])

plt.legend()

"""

大部分人未婚，但是他们工资比例较低，一般稳定结婚了的比例更高，将MCVS和MAFS结合一起，结婚后只剩一个人的也可考虑结合在一起"""
for all_data in data_cleaner:

    all_data['Marital_Status'] = all_data['Marital_Status'].apply(lambda x : x.strip())
#查看Never-married和年龄的关系

train1 = train.copy()

Marital_age = train1[train1['Marital_Status'] == 'Never-married']

ax = sns.FacetGrid(data = Marital_age, hue = 'Salary', aspect=3)

ax.map(sns.distplot,'Age' )

ax.add_legend()

plt.show()

"""

显示出来大部分年轻人没有结婚，工资也低，而38岁左右不结婚也会有比较高工资

"""
#Occupation缺失值查和Work_class的差不多，看看哪里不同

train.ix[((train['Work_Class']=='?'))]
print(train[(train['Work_Class']=='?')&(train['Occupation']==' ?')].count().max())

"""

Work_class为空的时候Occupation也为空

"""
print(train[(train['Work_Class']!='?')&(train['Occupation']==' ?')])

"""

Occupation为空时，Work_class都为Never-Worked,其他都处理方式和Work_class相同

"""
#relationship vs salry

train1['Relationship'].value_counts()
plt.figure(figsize=(14,8))

sns.barplot(y = 'Salary', x = 'Relationship', data = train1)

"""

结了婚工资普遍偏高"""
#relationshion vs age

h = sns.FacetGrid(data = train1, col = 'Relationship')

h.map(sns.distplot, 'Age',color = 'r')

h.add_legend()

"""

这个数据有些不科学，own-child居然是20岁左右最多，而unmaried居然平均在40左右，由于有sex变量考虑将husband 和 wife合并，他们基本相同分布，并且Salary比例也很相似"""
#看是否有同性恋

train1.loc[(train1['Relationship'] == ' Husband')&(train1['Sex']==' Female'),]

"""应该是错误数据，删去"""
train = train.drop(index = 7109)
#race vs salary(

plt.figure(figsize =(10,7))

sns.barplot(x = 'Race', y = 'Salary', data = train1)
Race_dist = train1.groupby('Race').count().max(1).sort_values()

Race_dist.plot(kind = 'bar')

"""

亚洲人Salary的比例最高，白人人数最多工资工资比例也高"""
#Sex vs Salary

sns.barplot(x = 'Sex', y = 'Salary', data = train1)

"""男性工资普遍比女性高"""
#Native_Country vs Salary

salary_ratio1 = []

for x in np.unique(train1['Native_Country']):

    salary_ratio1.append(train1[(train1['Native_Country'] == x)&(train1['Salary']==1)].count().max()/

                       train1[train1['Native_Country'] == x].count().max())

salary_country_ratio ={'Native_Country':np.unique(train1['Native_Country']), 

                    'Salary' : salary_ratio1}

salary_country_ratio = pd.DataFrame(salary_country_ratio)

salary_country_ratio = salary_country_ratio.sort_values(['Salary'],ascending = 0).reset_index(drop=True)

plt.figure(figsize=(18,7))

salary_country_ratio['Salary'] = salary_country_ratio['Salary'].apply(lambda x: np.round(x,4))

ax = sns.barplot(x = 'Salary', y = 'Native_Country', data = salary_country_ratio )

plt.tight_layout()

plt.title('Native Country with Salary')

plt.show()

#plt.figure(figsize=(10,15))

#ax = sns.barplot(y = 'Native_Country', x = 'Salary', data = train1)

train1['Native_Country'].value_counts().plot(kind = 'bar')

"""

美国占的数目太多，考虑利用发达国家，发展中国家和贫穷国家进行分割,数据中还有很多空值，但是很难通过其他变量来进行填补"""
train1['Native_Country'] = train1['Native_Country'].apply(lambda x:x.strip())
developed_country = ['United-States', 'Germany', 'Canada','England','Italy', 'Japan','Taiwan', 'Portugal',

                    'Greece', 'France', 'Hong', 'Yugoslavia', 'Scotland']

developing_country = ['Mexico', 'Philippines', 'India', 'Cuba', 'China', 'Poland', 'Ecuador', 'Ireland',

                     'Iran','Thailand','?', 'Hungary']

poor_country = ['Puerto-Rico', 'El-Salvador', 'Jamaica', 'South', 'Dominican-Republic', 'Vietnam',

               'Guatemala', 'Columbia', 'Haiti', 'Nicaragua', 'Peru', 'Cambodia', 'Trinadad&Tobago',

               'Laos', 'Outlying-US(Guam-USVI-etc)','Honduras', 'Holand-Netherlands' ]
train2 = train1.copy()

train2.loc[train2['Native_Country'].isin(developed_country),'Native_Country'] = 'Developed_country'

train2.loc[train2['Native_Country'].isin(developing_country),'Native_Country'] = 'Developing_country'

train2.loc[train2['Native_Country'].isin(poor_country),'Native_Country'] = 'poor_country'
sns.barplot(y = 'Salary', x = 'Native_Country', data = train2)

"""

现在再来看Salary分布比较平均，也比较符合实际"""
#Hours_Per_Week

plt.figure(figsize=(18,7))

sns.barplot(train1['Hours_Per_Week'], train1['Salary'])

plt.title('Hours Per Week with Salary')

"""

时长越高工资越高"""
plt.figure(figsize=(14,7))

sns.distplot(train1['Hours_Per_Week'])

"""

分布基本平均在40小时两边，考虑切割为3部分大于40，小于40，等于40，或者利用fulltime，parttime进行分类"""
train1 = train.copy()

train1['Hours_Per_Week'] = train1['Hours_Per_Week'].apply(lambda x:'<40' if x<40 else('=40' if x==40 else('>40')))
#分割完之后

fig, (axis1,axis2) = plt.subplots(1,2, figsize = (14,7))

train1['Hours_Per_Week'].value_counts().plot(kind = 'bar', ax = axis1)

axis1.set_title('Hourse Per Week distribution')



sns.barplot(x = 'Hours_Per_Week', y = 'Salary', data= train1, ax = axis2)

axis2.set_title('Salary vs House_Per_Week')

"""

分割完之后分布更加平稳，也更显示出了Salary与Hours_per_week的关系"""
#Capital_Gain&Capital_Loss

plt.figure(figsize = (14,7))

plt.subplot(121)

sns.distplot(train1['Capital_Gain'])

plt.title('Capital Gain Distribution')



plt.subplot(122)

sns.distplot(train1['Capital_Loss'])

plt.title('Capital Loss Distribution')



#scatter plot between Gain and Loss

sns.scatterplot(x = 'Capital_Gain', y = 'Capital_Loss', data = train1)

plt.title('Gain & Loss Scatter Plot')

"""

基本2者没有同时非零点则考虑将2个变量改为0，1变量"""
train1.loc[(train1['Capital_Gain']>0) & (train1['Capital_Loss']>0),].count().max()

"""

确实没有既有Gain又有Loss的值"""
##Feature Engineer

df = train.copy()

test = test1.copy()
#消除所有object变量之前的空格

data_Cleaner = [df, test]

for all_data in data_Cleaner:

    for i in col_Name:

        if all_data[i].dtype == 'object':

            all_data[i] = all_data[i].apply(lambda x : x.strip())
#空值填补

for all_data in data_Cleaner:

    all_data.loc[(all_data['Work_Class']=='?') & (all_data['Age'] >= 60), 'Work_Class'] = 'Retired' 

    all_data.loc[(all_data['Age']<=24) & (all_data['Work_Class']=='?'),'Work_Class'] = 'Never-worked'



    all_data.loc[(all_data['Work_Class'] == 'Never-worked') & (all_data['Occupation'] == '?'), 'Occupation'] = 'None'

    all_data.loc[(all_data['Occupation'] == '?') & (all_data['Age'] >= 60), 'Occupation'] = 'Retired'



    all_data.loc[all_data['Work_Class'] == '?', 'Work_Class'] = 'Unknown'

    all_data.loc[all_data['Occupation'] == '?', 'Occupation'] = 'Unknown'

    all_data.loc[all_data['Native_Country'] == '?', 'Native_Country'] = 'Unknown'

#Work_Class

for all_data in data_Cleaner:

    all_data.loc[all_data['Work_Class'].isin(['Never-worked', 'Without-pay']), 'Work_Class'] = 'Others'
#Education

Primary = ['1st-4th', '5th-6th', 'Preschool']

Secondary = ['7th-8th', '9th', '10th', '11th', '12th']

Teriary = ['HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm']

Quaternary = ['Prof-school', 'Doctorate']

for all_data in data_Cleaner:

    all_data.loc[all_data['Education'].isin(Primary), 'Education'] = 'Primary'

    all_data.loc[all_data['Education'].isin(Secondary), 'Education'] = 'Secondary'

    all_data.loc[all_data['Education'].isin(Teriary), 'Education'] = 'Teriary'

    all_data.loc[all_data['Education'].isin(Quaternary), 'Education'] = 'Quaternary'
#Marital_Status

Married = ['Married-civ-spouse', 'Married-AF-spouse']

Solo = ['Divorced', 'Separated', 'Widowed','Married-spouse-absent']

for all_data in data_Cleaner:

    all_data.loc[all_data['Marital_Status'].isin(Married), 'Marital_Status'] = 'Married'

    all_data.loc[all_data['Marital_Status'].isin(Solo), 'Marital_Status'] = 'Solo'

#Relationship

Family = ['Husband', 'Wife']

for all_data in data_Cleaner:

    all_data.loc[all_data['Relationship'].isin(Family), 'Relationship'] = 'Family'
#Native_Country

for all_data in data_Cleaner:

    all_data.loc[all_data['Native_Country'].isin(developed_country),'Native_Country'] = 'Developed_country'

    all_data.loc[all_data['Native_Country'].isin(developing_country),'Native_Country'] = 'Developing_country'

    all_data.loc[all_data['Native_Country'].isin(poor_country),'Native_Country'] = 'poor_country'
#Hours_Per_Week

for all_data in data_Cleaner:

    all_data['Hours_Per_Week'] = all_data['Hours_Per_Week'].apply(lambda x : 'Part_Time' if x<=35 

                                                                      else('Full_Time' if (x>35)&(x<=40)

                                                                      else('Much_Work' if (x>40)&(x<=50)

                                                                      else('Over_Work'))))
Remove_Columns = ['Fnlwgt', 'Education_Num']

for all_data in data_Cleaner:

    all_data.drop(Remove_Columns, axis = 1, inplace = True)
df1 = df.copy()

test2 = test.copy()

data_Cleaner1 = [df1, test2]

from sklearn import preprocessing

for all_data in data_Cleaner1:

    for column in all_data:

        le = preprocessing.LabelEncoder()

        all_data[column] = le.fit_transform(all_data[column])

    
from sklearn.ensemble import RandomForestClassifier

select = RandomForestClassifier()

train_select = df1.drop(columns='Salary', axis = 1)

test_select = df1.Salary

param_grid = {'criterion': ['gini','entropy'],

               'max_depth':[4,6,8,10],

               'n_estimators':[50,100,200,300]

             }

select1 = model_selection.GridSearchCV(select, param_grid=param_grid, cv = 5, scoring='accuracy')   

select1.fit(train_select, test_select)
select1.best_params_
selected = RandomForestClassifier(criterion='gini', max_depth=10, n_estimators=100)

selected.fit(train_select, test_select)
importances = selected.feature_importances_

indices = np.argsort(importances)[::-1]

features = train_select.columns

for f in range(train_select.shape[1]):

    print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importances[indices[f]])))
#Capital_Gain&Capital_Loss

gain = train1.loc[train1['Capital_Gain']!=0, 'Capital_Gain']

loss = train1.loc[train1['Capital_Loss']!=0, 'Capital_Loss']

print('Gain quantile(0.3):', gain.quantile(0.3))

print('Gain quantile(0.7):', gain.quantile(0.7))

print('Loss quantile(0.3):', loss.quantile(0.3))

print('Loss quantile(0.7):', loss.quantile(0.7))
for all_data in data_Cleaner:

    all_data['Capital_Total'] = 'Zero'

    all_data['Capital_Gain'] = all_data['Capital_Gain'].apply(lambda x : 'Low_Gain' if (x>0)&(x<=3942)

                                           else('Med_Gain' if (x>3942)&(x<=8614)

                                           else('High_Gain' if x>8614

                                           else('Zero'))))



    all_data['Capital_Loss'] = all_data['Capital_Loss'].apply(lambda x : 'Low_Loss' if (x>0)&(x<=1740)

                                           else('Med_Loss' if (x>1740)&(x<=1977)

                                           else('High_Loss' if x>1977

                                           else('Zero'))))

    

    all_data['Capital_Total'].loc[all_data['Capital_Gain']!='Zero'] = all_data['Capital_Gain']

    all_data['Capital_Total'].loc[all_data['Capital_Loss']!='Zero'] = all_data['Capital_Loss']

    
Remove_Columns = ['Capital_Gain', 'Capital_Loss']

for all_data in data_Cleaner:

    all_data.drop(Remove_Columns, axis = 1, inplace = True)
Dummy = ['Work_Class', 'Education', 'Marital_Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Hours_Per_Week',

        'Native_Country', 'Capital_Total']

dummies1 = pd.get_dummies(df[Dummy],prefix=Dummy)

df = pd.concat([df, dummies1], axis = 1)
dummies2= pd.get_dummies(test[Dummy],prefix=Dummy)

test = pd.concat([test, dummies2], axis = 1)
Drop_Coluns = ['Work_Class', 'Education', 'Marital_Status', 'Occupation','Relationship', 'Race', 

               'Sex', 'Hours_Per_Week', 'Native_Country', 'Capital_Total']

df.drop(Drop_Coluns, axis = 1, inplace = True)



test.drop(Drop_Coluns, axis = 1, inplace = True)
test.to_csv('D_test.csv')
target = ['Salary']

X = df.drop(target, axis = 1)

Y = df[target]
#XGBoost

cv_split = model_selection.ShuffleSplit(n_splits=3, test_size=0.3, train_size = 0.7, random_state=0)

Xgboost = XGBClassifier(max_depth = 7, leaning_rate = 0.1)

base_Result = model_selection.cross_validate(Xgboost, X, Y, cv  = cv_split, return_train_score=True)
#train&test score with no params change xgboost

print('xgboost parameters:', Xgboost.get_params())

print('train score mean:{:.4f}'.format(base_Result['train_score'].mean()))

print('test score mean:{:.4f}'.format(base_Result['test_score'].mean()))
param_Grid = {

              'max_depth':[5,7,9,12],

              'leaning_rate':[0.1,0.15,0.2]

             }

tune_Model = model_selection.GridSearchCV(XGBClassifier(), param_grid=param_Grid, scoring='roc_auc', cv = cv_split)

tune_Model.fit(X,Y)
tune_Model.best_params_
Xgboost = XGBClassifier(max_depth = 5, leaning_rate = 0.1)

Xgboost.fit(X,Y)

importance = Xgboost.feature_importances_

indices = np.argsort(importance)[::-1]

features = X.columns

for f in range(X.shape[1]):

    print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importance[indices[f]])))
test_x = test.drop(target, axis =1)

test_y = test[target]
predictions1 = Xgboost.predict(test_x)

accuracy_score(test_y, predictions1)
#random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(oob_score= True)
param_Grid1 = {'criterion': ['gini','entropy'],

               'max_depth':[4,6,8,10],

               'n_estimators':[50,100,200,300]

               

}
tune_Model1 = model_selection.GridSearchCV(rf, param_grid=param_Grid1, scoring='accuracy', cv = cv_split)

tune_Model1.fit(X,Y)
tune_Model1.best_params_
rf1 = RandomForestClassifier(criterion='gini', max_depth=10,n_estimators=200)

rf1.fit(X, Y)

predictions2 = rf1.predict(test_x)

accuracy_score(test_y, predictions2)
rf1
#random forest importances

indices = np.argsort(importance)[::-1]

features = X.columns

for f in range(X.shape[1]):

    print(("%2d) %-*s %f" % (f + 1, 30, features[indices[f]], importance[indices[f]])))
#svm

import scipy

from sklearn.svm import SVC 

sv = SVC()

param_Grid2 = { 'C' : [0.5, 1, 1.5],

              'gamma':[0.1,'auto']}

tune_Model2 = model_selection.GridSearchCV(sv, param_grid=param_Grid2, cv=cv_split, scoring='accuracy')

tune_Model2.fit(X,Y)
tune_Model2.best_params_
sv1

sv1 = SVC(C=1.5, gamma=0.1)

sv1.fit(X, Y)

predictions3 = sv1.predict(test_x)

accuracy_score(test_y, predictions3)

# gdbt

from sklearn.ensemble import GradientBoostingClassifier

gdbt = GradientBoostingClassifier()

gdbt.fit(X,Y)

predictions4 = gdbt.predict(test_x)

accuracy_score(test_y, predictions4)

gdbt
from vecstack import stacking
# stacking

clfs = [XGBClassifier(max_depth = 7, leaning_rate = 0.1),

        SVC(C=1.5, gamma=0.1),

        RandomForestClassifier(criterion='gini', max_depth=10,n_estimators=100),    

        GradientBoostingClassifier()

]



X_1, X_2, y_1, y_2 = train_test_split(X, Y, test_size=0.33, random_state=2019)



S_train, S_test = stacking(clfs, 

                           X, Y, test_x,

                           regression = False,

                           mode = 'oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds = 5,

                           stratified=True,

                           shuffle=True,

                           verbose=2,

                           random_state=2019

                         )

model = XGBClassifier()

model.fit(S_train, Y)

y_pred = model.predict(S_test)

accuracy_score(test_y, y_pred)
# dnn
print(X.shape[1], Y.shape[1])
#adam optimizer

training_epochs = 1500

learning_rate = 0.01

hidden_layers = X.shape[1] - 1



x = tf.placeholder(tf.float32, [None, 63])

y = tf.placeholder(tf.float32, [None, 1])



is_training = tf.Variable(True,dtype=tf.bool)



initializer = tf.contrib.layers.xavier_initializer()

h0 = tf.layers.dense(x, 120, activation=tf.nn.relu, kernel_initializer=initializer)

h1 = tf.layers.dense(h0, 120, activation=tf.nn.relu)

h2 = tf.layers.dense(h1, 1, activation=None)



cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=h2)

cost = tf.reduce_mean(cross_entropy)



#Momentum = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)

#RMSprop=tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9).minimize(cost)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#gradient=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)



predicted = tf.nn.sigmoid(h2)

correct_pred = tf.equal(tf.round(predicted), y)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    cost_history = np.empty(shape = 1, dtype = float)

    for step in range(training_epochs + 1):

        sess.run(gradient, feed_dict={x: X, y: Y})

        loss, _, acc = sess.run([cost, gradient, accuracy], feed_dict={

                                 x: X, y: Y})

        Cost_History = np.append(Cost_History, acc)

        if step % 200 == 0:

            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(

                step, loss, acc))

    print('Test Accuracy by gradient:', sess.run([accuracy, tf.round(predicted)], feed_dict={x: test_x, y: test_y}))

    

    sess.run(tf.global_variables_initializer())

    cost_history = np.empty(shape = 1, dtype = float)

    for step in range(training_epochs + 1):

        sess.run(Momentum, feed_dict={x: X, y: Y})

        loss, _, acc = sess.run([cost, Momentum, accuracy], feed_dict={

                                 x: X, y: Y})

        Cost_History = np.append(Cost_History, acc)

        if step % 200 == 0:

            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(

                step, loss, acc))

    print('Test Accuracy by momentum:', sess.run([accuracy, tf.round(predicted)], feed_dict={x: test_x, y: test_y}))

    

    sess.run(tf.global_variables_initializer())

    cost_history = np.empty(shape = 1, dtype = float)

    for step in range(training_epochs + 1):

        sess.run(RMSprop, feed_dict={x: X, y: Y})

        loss, _, acc = sess.run([cost, RMSprop, accuracy], feed_dict={

                                 x: X, y: Y})

        Cost_History = np.append(Cost_History, acc)

        if step % 200 == 0:

            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(

                step, loss, acc))

    print('Test Accuracy by RMSprop:', sess.run([accuracy, tf.round(predicted)], feed_dict={x: test_x, y: test_y}))

    

    sess.run(tf.global_variables_initializer())

    cost_history = np.empty(shape = 1, dtype = float)

    for step in range(training_epochs + 1):

        sess.run(adam, feed_dict={x: X, y: Y})

        loss, _, acc = sess.run([cost, adam, accuracy], feed_dict={

                                 x: X, y: Y})

        Cost_History = np.append(Cost_History, acc)

        if step % 200 == 0:

            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(

                step, loss, acc))

    print('Test Accuracy by adm:', sess.run([accuracy, tf.round(predicted)], feed_dict={x: test_x, y: test_y}))
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    cost_history = np.empty(shape = 1, dtype = float)

    for step in range(training_epochs + 1):

        sess.run(adam, feed_dict={x: X, y: Y})

        loss, _, acc = sess.run([cost, adam, accuracy], feed_dict={

                                 x: X, y: Y})

        cost_history = np.append(cost_history, acc)

        if step % 200 == 0:

            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(

                step, loss, acc))

    summary_writer = tf.summary.FileWriter('./log/', sess.graph)

    print('Test Accuracy by adm:', sess.run([accuracy, tf.round(predicted)], feed_dict={x: test_x, y: test_y}))
table = {'Test Accuracy':[86.364, 86.204, 86.167, 85.831, 85.805, 85.547],

        'Model':['Stacking', 'XGBoost', 'GDBT', 'SVM', 'RF', 'ANN']}

d = ['86.364%', '86.204%', '86.167%', '85.831%', '85.805%', '85.547%']

sns.barplot(y = table['Model'], x = table['Test Accuracy'])

plt.xlim(85, 86.5)

for i in range(6):

    plt.text(x = table['Test Accuracy'][i], y=i, s=d[i])

 



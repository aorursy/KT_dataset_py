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
#加载所需的库文件
# NumPy
import numpy as np

# Dataframe operations
import pandas as pd

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
#2、加载数据
train = pd.read_csv("../input/train.csv")
test =  pd.read_csv("../input/test.csv")
train.head()
test.head()
train.shape,test.shape
#删除数据集中的Id
test_ID = test['PassengerId']

#这里先不删除，后面需要用到
#train.drop("PassengerId",axis=1,inplace=True)
#test.drop("PassengerId",axis=1,inplace=True)
train.shape,test.shape
#看训练集中是否有离群点
fig,ax = plt.subplots()
ax.scatter(x=train['Fare'],y=train['Survived'])  #其中Fare可换位 Age Pclass等
plt.xlabel("Fare",fontsize=13)
plt.ylabel("Survived",fontsize=13)
plt.show()
#可以看出这里Fare中明显有一个离群点，删除
train = train.drop(train[(train['Survived'] == 1) & (train['Fare'] > 400)].index)

fig,ax = plt.subplots()
ax.scatter(x=train['Fare'],y=train['Survived'])  #其中Fare可换位 Age Pclass等
plt.xlabel("Fare",fontsize=13)
plt.ylabel("Survived",fontsize=13)
plt.show()
train.shape
#由于是分类任务，所以感觉没必要对目标变量镜像处理，这一块有待考证
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Survived.values

all_data = pd.concat((train,test)).reset_index(drop=True)
#先不删除
#all_data.drop(['Survived'],axis=1,inplace=True)
all_data.shape
all_data
#处理缺失的数据
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({"Missing ratio":all_data_na})
missing_data
#查看缺失的个数
all_data_na_number = all_data.isnull().sum()
all_data_na_number = all_data_na_number.drop(all_data_na_number[all_data_na_number == 0].index).sort_values(ascending=False)
print(all_data_na_number)
print("缺失的个数：",len(all_data_na_number))

#画出缺失值的bar图，数据的可视化处理
f,ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index,y=all_data_na)
plt.xlabel('Features',fontsize=15)
plt.ylabel('Percent of missing values',fontsize=15)
plt.title('Percent missing data by feature',fontsize=15)
#开始处理数据了，首先让我们来看看特征与目标变量的大概关系把。
#查看每个特征与SalePrice特征之间的关联
corrmat = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,vmax=1,square=True,linewidths=1,cmap="YlGnBu",annot=True,fmt='f',vmin=0)
#开始处理缺失的数据 Fare -- 1 Cabin -- 1013 Embark -- 2 Age -- 263
f,ax = plt.subplots(figsize=(15,12))
#plt.xticks(rotation='90')
sns.barplot(x=all_data.Fare.index,y=all_data.Fare)
#plt.xlabel('Features',fontsize=15)
#plt.ylabel('Percent of missing values',fontsize=15)
#plt.title('Percent missing data by feature',fontsize=15)
#Fare,用中位数来填充
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())
#制作Fare的区间
all_data['FareBin'] = pd.qcut(all_data['Fare'],5)

#由与船舱缺失值太多，删除之
all_data.drop(['Cabin'],axis=1,inplace=True)

#Embark 用临近的值来进行填充
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].ffill())

#目前只有Age未进行填充，年龄我们从头衔来获取
#因此我们要先生成一个头衔的特征变量
all_data['Title'] = all_data['Name']
for name_string in all_data['Name']:
    all_data['Title'] = all_data['Name'].str.extract('([A-Za-z]+)\.')
all_data.Title.value_counts(),len(all_data.Title.value_counts())
#用常见的人名取代不常见的,18种人名
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
all_data.replace({'Title': mapping}, inplace=True)
#all_data.Title.value_counts()
#从头衔从推断年龄
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = all_data.groupby('Title')['Age'].median()[titles.index(title)]
    all_data.loc[(all_data['Age'].isnull()) & (all_data['Title'] == title),'Age'] = age_to_impute
    
#重现检查看是否还有缺失值
all_data.isnull().sum()
#特征的构建开始
#all_data
#添加家庭的人数
all_data['Family_Size'] = all_data['Parch'] + all_data['SibSp'] + 1

#all_data
#创建家庭的存活率
all_data['Last_Name'] = all_data['Name'].apply(lambda x: str.split(x, ",")[0])
default_survial_value = 0.5
all_data['Family_Survival'] = default_survial_value
for grp,grp_df in all_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age']].groupby(['Last_Name', 'Fare']):
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      all_data.loc[all_data['Family_Survival']!=0.5].shape[0])
for _, grp_df in all_data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(all_data[all_data['Family_Survival']!=0.5].shape[0]))
#删除Id survial
all_data.drop(['PassengerId','Survived'],axis=1,inplace=True)
#all_data
#创建年龄的bin
all_data['AgeBin'] = pd.qcut(all_data['Age'],4)
#all_data
label = LabelEncoder()
all_data['AgeBin'] = label.fit_transform(all_data['AgeBin'])
all_data['FareBin'] = label.fit_transform(all_data['FareBin'])
#all_data
#删除不需要的列
all_data.drop(['Name', 'SibSp', 'Parch', 'Ticket'],axis=1,inplace=True)
all_data.drop(['Last_Name','Fare','Age'],axis=1,inplace=True)
all_data
#all_data.drop(['Title'],axis=1,inplace=True)
#all_data
all_data.info()
from scipy import stats
from scipy.stats import norm, skew

#检查数值特征中是否由倾斜的
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
# skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

# from scipy.special import boxcox1p
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     #all_data[feat] += 1
#     all_data[feat] = boxcox1p(all_data[feat], lam)
    
# #all_data[skewed_features] = np.log1p(all_data[skewed_features])
#感觉不需要调整数据
all_data = pd.get_dummies(all_data)
all_data
train = all_data[:ntrain]
test = all_data[ntrain:]
train.shape,test.shape
X_train = train
X_test = test
#标准缩放
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
#Grid Search CV
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc")
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)
knn = KNeighborsClassifier(algorithm='auto', leaf_size=11, metric='minkowski', 
                           metric_params=None, n_jobs=-1, n_neighbors=8, p=2, 
                           weights='uniform')
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)

temp = pd.DataFrame({'PassengerId':test_ID,"Survived":y_pred1})
temp.to_csv("gender_submission1.csv",index=False)
gd.best_estimator_.fit(X_train, y_train)
y_pred = gd.best_estimator_.predict(X_test)
temp = pd.DataFrame({'PassengerId':test_ID,"Survived":y_pred})
temp.to_csv("gender_submission2.csv",index=False)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


cv_params = {
             #'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
             #'min_child_weight': [1, 2, 3, 4, 5, 6]
            #'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1]
            #'subsample': [0.6, 0.7, 0.8, 0.9], 
            #'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
            #'reg_alpha': [0.05, 0.1, 1, 2, 3],
            #'reg_lambda': [0.05, 0.1, 1, 2, 3]
            'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2],
            'n_estimators':np.arange(10,300,20)
            }
other_params = {
                'learning_rate': 0.1, 
                'n_estimators': 50, 
                'max_depth': 5, 
                'min_child_weight': 4,
                'seed': 0,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'gamma': 0.8,
                'reg_alpha': 0.1, 
                'reg_lambda': 0.1
               }
# param_grid = {
#     "n_estimators":[50,100,150，210, 215, 220, 225, 230], 
#     "learning_rate":[0.01, 0.02, 0.03, 0.04, 0.05], 
#     "max_depth":[3, 4, 5, 6, 7]
# }
model_xgb = XGBClassifier(**other_params)
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=7)
grid_search_xgb = GridSearchCV(model_xgb,param_grid=cv_params,scoring="roc_auc",n_jobs=-1,cv=kfold,verbose=2)
grid_search_xgb.fit(train,y_train)
print(grid_search_xgb.best_params_)
print(grid_search_xgb.best_score_)
print(grid_search_xgb.best_estimator_)
grid_search_xgb.best_estimator_.fit(train,y_train)
pred2 = grid_search_xgb.best_estimator_.predict(test)
temp = pd.DataFrame({'PassengerId':test_ID,"Survived":pred2})
temp.to_csv("gender_submission3.csv",index=False)

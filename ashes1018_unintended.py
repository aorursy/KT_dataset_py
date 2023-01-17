#coding:utf-8

import matplotlib

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#coding: utf-8

def get_data():

#     读取Windows下文件，注意转义符号的使用

    train_data = pd.read_csv('../input/train.csv')

    test_data = pd.read_csv('../input/test.csv')

    return train_data,test_data

train_data,test_data = get_data()

train_data.info()

train_data.head()

# test_data.info()

# test_data.head()
target_column = 'Survived'

continuous_column_list = ['Age','SibSp','Fare','Parch']

discrete_column_list = ['Pclass','Sex','Embarked']

text_column_list = ['Name', 'Ticket', 'Cabin']



continuous_train_data = train_data.filter(continuous_column_list)

discrete_train_data = train_data.filter(discrete_column_list)

text_train_data = train_data.filter(text_column_list)

continuous_train_data.describe()

# 卡方检验

from sklearn.feature_selection import chi2

print(chi2(train_data.filter(['Fare']),train_data['Survived']))

print(chi2(train_data.filter(['Parch']),train_data['Survived']))

print(chi2(train_data.filter(['SibSp']),train_data['Survived']))

# 可以看出SibSp和Survived的相关性更大
feature = 'Parch'

feature_data = train_data.filter([feature, 'Survived'])

survived_data = feature_data[feature][feature_data.Survived == 1].value_counts()

unsurvived_data = feature_data[feature][feature_data.Survived == 0].value_counts()

df = pd.DataFrame({'Survived':survived_data,'Unsurvived':unsurvived_data})

df.plot(kind = 'bar',stacked = True)

plt.title('survived_'+feature)

plt.xlabel('Parch_numbers')

plt.ylabel('number')

plt.show()



# print(feature_data.count())

# print(survived_data)

# print(unsurvived_data)
feature = 'SibSp'

feature_data = train_data.filter([feature, 'Survived'])

survived_data = feature_data[feature][feature_data.Survived == 1].value_counts()

unsurvived_data = feature_data[feature][feature_data.Survived == 0].value_counts()

df = pd.DataFrame({'Survived':survived_data,'Unsurvived':unsurvived_data})

df.plot(kind = 'bar',stacked = True)

plt.title('survived_'+feature)

plt.xlabel('Sibsp_numbers')

plt.ylabel('number')

plt.show()
for d in  discrete_train_data.columns:

    print(discrete_train_data[d].value_counts())

    print('.....')



from sklearn.feature_selection import chi2

from sklearn.preprocessing import LabelEncoder,LabelBinarizer

# 将Sex数据标称为二值型

sex_label_data = LabelBinarizer().fit_transform(train_data['Sex'])

# 将embarked按照0,1，2,3......标记成不同的类别

embarked_label_data = LabelEncoder().fit_transform(train_data['Embarked'].fillna('S'))

print('Embarked',chi2(pd.DataFrame(embarked_label_data), train_data['Survived']))

print('Sex', chi2(sex_label_data, train_data['Survived']))

print('Pclass', chi2(train_data.filter(['Pclass']), train_data['Survived']))
def print_stacked_hist(feature):

    feature_data = train_data.filter([feature, 'Survived'])

    survived_data = feature_data[feature][feature_data.Survived == 1].value_counts()

    unsurvived_data = feature_data[feature][feature_data.Survived == 0].value_counts()

    df = pd.DataFrame({'Survived':survived_data,'Unsurvived':unsurvived_data})

    df.plot(kind = 'bar',stacked = True)

    plt.title('survived_'+feature)

    plt.xlabel(feature+'_numbers')

    plt.ylabel('number')

    plt.show()

print_stacked_hist('Pclass')

print_stacked_hist('Sex')

print_stacked_hist('Embarked')
filled_data = train_data.copy()

# 对年龄的处理，age_新 = （age_旧-start）/step，对于原来缺失的年龄值，直接赋为0

filled_data.loc[np.isnan(train_data['Age']),'Age'] = 0

def transform_category(data,start,step,category):

    result = ((data - start)/step).astype(int)+category

    return result

step = 5

filled_data['Age'] = transform_category(filled_data['Age'],0,step,0)

# 对船舱进行0/1标称化处理

filled_data.loc[filled_data['Cabin'].notnull(),'Cabin'] = 1

filled_data.loc[filled_data['Cabin'].isnull(),'Cabin'] = 0

# 对港口处理方式， 对缺失数据赋最常见港口

def get_most_common_category(series):

    return series.value_counts().axes[0][0]

most_common = get_most_common_category(filled_data['Embarked'])

filled_data.loc[filled_data['Embarked'].isnull(),'Embarked'] = most_common





newdata = filled_data

newdata
# 离散化

# dummy coding

dummy_cabin = pd.get_dummies(newdata['Cabin'],prefix = 'Cabin')

dummy_sex = pd.get_dummies(newdata['Sex'],prefix = 'Sex')

dummy_embarked = pd.get_dummies(newdata['Embarked'],prefix = 'Embarked')



dummied_data = pd.concat([newdata,dummy_cabin,dummy_sex,dummy_embarked],axis =1)

dummied_data.drop(['Cabin', 'Sex', 'Embarked'],axis = 1, inplace = True)

dummied_data
from sklearn.preprocessing import StandardScaler

# 对Fare字段进行特征提取

dummied_data['Fare'] = StandardScaler().fit_transform(dummied_data.filter(['Fare']))

dummied_data
unused_column = ['PassengerId','Name','Ticket']

target_prepared_y = dummied_data['Survived']

train_prepared_data = dummied_data.drop(unused_column+['Survived'],axis =1)

train_prepared_data
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn import cross_validation,metrics





def model_fit(alg,X,Y,performCV = True, printFeatureImportance = True, cv_folds = 5):

    dtrain_predictions = alg.predict(X)

    dtrain_predprob = alg.predict_proba(X)[:,1]

    if performCV:

        cv_score = cross_validation.cross_val_score(alg, X, Y, cv = cv_folds, scoring = 'roc_auc')

        

    print('model report:')

    print('Accuracy:%.4g' % metrics.accuracy_score(Y.values, dtrain_predictions))

    print('AUC score:%f' % metrics.roc_auc_score(Y,dtrain_predprob))

    

    if performCV:

        print('CV score :mean - %.7g|std - %.7g|min - %.7g|max - %.7g|' % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    

    if printFeatureImportance:

        feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending = False)

        feat_imp.plot(kind = 'bar',title = 'feature importance')

        plt.ylabel('feature importance score')

    

def train_model(model_class, print_coef = False, *args, ** kwargs):

    kf = KFold(n_splits = 10)

    best_lr = None

    best_score = 0

    for train_index, test_index in kf.split(train_prepared_data):

        train_sub_data, target_sub_data = train_prepared_data.loc[train_index],target_prepared_y.loc[train_index]

        test_sub_data, test_target_sub_data = train_prepared_data.loc[test_index],target_prepared_y.loc[test_index]

        lr  = model_class(*args, **kwargs)

        lr.fit(train_sub_data,target_sub_data)

        score = lr.score(test_sub_data,test_target_sub_data)

        if score > best_score:

            best_lr = lr

            best_score = score

    print(best_lr)

    print(best_score)

    

    model_fit(best_lr,train_prepared_data,target_prepared_y,printFeatureImportance = False)

    if print_coef:

        columns = list(train_prepared_data.columns)

#         画出各个因素的相关性

        plot_df = pd.DataFrame(best_lr.coef_.ravel(),index=columns)

        plot_df.plot(kind = 'bar')

    return best_lr



train_model(LogisticRegression,print_coef = True)



        
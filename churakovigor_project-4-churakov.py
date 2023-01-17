import pandas as pd

from pandas import Series

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.feature_selection import f_classif, mutual_info_classif

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression





from sklearn.metrics import confusion_matrix

from sklearn.metrics import auc, roc_auc_score, roc_curve

from sklearn.metrics import f1_score, accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/sf-dst-scoring/train.csv')

test= pd.read_csv('/kaggle/input/sf-dst-scoring/test.csv')
train.info()
test.info()
train.sample(5)
test.sample(5)
train.shape, test.shape
train.app_date = pd.to_datetime(train.app_date, format='%d%b%Y')

test.app_date = pd.to_datetime(train.app_date, format='%d%b%Y')

print(train.app_date.sample(5))

print(test.app_date.sample(5))
train.app_date.max(), test.app_date.max()
train.app_date.min(), test.app_date.min()
min_day=train.app_date.min()

train['days'] = (train.app_date - min_day).dt.days.astype('int')

test['days'] = (test.app_date - min_day).dt.days.astype('int')
train['month'] = train.app_date.dt.month

test['month'] = test.app_date.dt.month
train['weekday'] = train.app_date.dt.weekday

test['weekday'] = test.app_date.dt.weekday
train['weekday'].value_counts().plot.barh()
train['month'].value_counts().plot.barh()
train.education = train.education.astype(str).apply(lambda x: None if x.strip()=='' else x)

test.education = test.education.astype(str).apply(lambda x: None if x.strip()=='' else x)

train['education'].value_counts(), test['education'].value_counts()
train.education = train.education.replace('nan', 'SCH') 

test.education = test.education.replace('nan', 'SCH') 
train['education'] = train['education'].apply(lambda x: x.replace('SCH','1')) 

train['education'] = train['education'].apply(lambda x: x.replace('GRD','2')) 

train['education'] = train['education'].apply(lambda x: x.replace('UGR','3')) 

train['education'] = train['education'].apply(lambda x: x.replace('PGR','4')) 

train['education'] = train['education'].apply(lambda x: x.replace('ACD','5')) 



test['education'] = test['education'].apply(lambda x: x.replace('SCH','1')) 

test['education'] = test['education'].apply(lambda x: x.replace('GRD','2')) 

test['education'] = test['education'].apply(lambda x: x.replace('UGR','3')) 

test['education'] = test['education'].apply(lambda x: x.replace('PGR','4')) 

test['education'] = test['education'].apply(lambda x: x.replace('ACD','5')) 
train.score_bki.describe()
train.score_bki.hist();
def outliers(data):

    quartile_1, quartile_3 = np.percentile(data, [25, 75])

    iqr = quartile_3 - quartile_1  #находим межквартильное расстояние

    lower_bound = quartile_1 - (iqr * 1.5)  #нижняя граница коробки

    upper_bound = quartile_3 + (iqr * 1.5)  #верхняя граница коробки

    return data[((data > upper_bound) | (data < lower_bound))]
len(outliers(train.score_bki))/len(train.score_bki), len(outliers(test.score_bki))/len(test.score_bki)
#quartile_1, quartile_3 = np.percentile(train.score_bki, [25, 75])

#iqr = quartile_3 - quartile_1

#lower_bound = quartile_1 - (iqr * 1.5)

#upper_bound = quartile_3 + (iqr * 1.5)

#train = train.loc[train.score_bki.between(lower_bound, upper_bound)]
#quartile_1, quartile_3 = np.percentile(test.score_bki, [25, 75])

#iqr = quartile_3 - quartile_1

#lower_bound = quartile_1 - (iqr * 1.5)

#upper_bound = quartile_3 + (iqr * 1.5)

#test = test.loc[test.score_bki.between(lower_bound, upper_bound)]
train['region_rating'].value_counts().plot.barh()
train['region_rating'].unique(), test['region_rating'].unique()
train['sna'].value_counts().plot.barh()
test['sna'].value_counts().plot.barh()
test['first_time'].value_counts().plot.barh()
train['first_time'].value_counts().plot.barh()
bin_cols = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']

cat_cols = ['month', 'education', 'home_address', 'work_address', 'sna', 'first_time','weekday']

num_cols = ['days', 'age', 'decline_app_cnt', 'score_bki', 'bki_request_cnt', 'region_rating', 'income']

num_cols_2 = ['days', 'age', 'decline_app_cnt', 'bki_request_cnt', 'region_rating', 'income']
len(outliers(train.days))/len(train.days), len(outliers(test.days))/len(test.days)
len(outliers(train.age))/len(train.age), len(outliers(test.age))/len(test.age)
len(outliers(train.decline_app_cnt))/len(train.decline_app_cnt), len(outliers(test.decline_app_cnt))/len(test.decline_app_cnt)
len(outliers(train.bki_request_cnt))/len(train.bki_request_cnt), len(outliers(test.bki_request_cnt))/len(test.bki_request_cnt)
#quartile_1, quartile_3 = np.percentile(train.bki_request_cnt, [25, 75])

#iqr = quartile_3 - quartile_1

#lower_bound = quartile_1 - (iqr * 1.5)

#upper_bound = quartile_3 + (iqr * 1.5)

#train = train.loc[train.bki_request_cnt.between(lower_bound, upper_bound)]
#quartile_1, quartile_3 = np.percentile(test.bki_request_cnt, [25, 75])

#iqr = quartile_3 - quartile_1

#lower_bound = quartile_1 - (iqr * 1.5)

#upper_bound = quartile_3 + (iqr * 1.5)

#test = test.loc[test.bki_request_cnt.between(lower_bound, upper_bound)]
len(outliers(train.income))/len(train.income), len(outliers(test.income))/len(test.income)
#quartile_1, quartile_3 = np.percentile(train.income, [25, 75])

#iqr = quartile_3 - quartile_1

#lower_bound = quartile_1 - (iqr * 1.5)

#upper_bound = quartile_3 + (iqr * 1.5)

#train = train.loc[train.income.between(lower_bound, upper_bound)]
#quartile_1, quartile_3 = np.percentile(test.income, [25, 75])

#iqr = quartile_3 - quartile_1

#lower_bound = quartile_1 - (iqr * 1.5)

#upper_bound = quartile_3 + (iqr * 1.5)

#test = test.loc[test.income.between(lower_bound, upper_bound)]
for i in num_cols_2:

 train[i] = np.log(train[i] + 1)

 plt.figure()

 sns.distplot(train[i][train[i] > 0].dropna(), kde = False, rug=False, color='b')

 plt.title(i)

 plt.show()
for i in num_cols_2:

 test[i] = np.log(test[i] + 1)

 plt.figure()

 sns.distplot(test[i][test[i] > 0].dropna(), kde = False, rug=False, color='b')

 plt.title(i)

 plt.show()
sns.heatmap(train[num_cols].corr().abs(), vmin=0, vmax=1)
sns.heatmap(test[num_cols].corr().abs(), vmin=0, vmax=1)
label_encoder = LabelEncoder()

for column in bin_cols:

    train[column] = label_encoder.fit_transform(train[column])

for column in bin_cols:

    test[column] = label_encoder.fit_transform(test[column])  

train.head()
train[bin_cols].head()
x_cat = OneHotEncoder(sparse = False).fit_transform(train[cat_cols].values)

x_cat_test = OneHotEncoder(sparse = False).fit_transform(test[cat_cols].values)



print(x_cat.shape)

print(x_cat_test.shape)
imp_cat = pd.Series(mutual_info_classif(train[bin_cols + cat_cols],

                                        train['default'], discrete_features =True),

                    index = bin_cols + cat_cols)

imp_cat.sort_values(inplace = True)

imp_cat.plot(kind = 'barh')
imp_num = Series(f_classif(train[num_cols], train['default'])[0], index = num_cols)

imp_num.sort_values(inplace = True)

imp_num.plot(kind = 'barh')
x_num = StandardScaler().fit_transform(train[num_cols])

x_num_test = StandardScaler().fit_transform(test[num_cols])

print(x_num)

print(x_num_test)
X = np.hstack([x_num, train[bin_cols].values, x_cat])

Y = train['default'].values

id_test = test.client_id

test = np.hstack([x_num_test, test[bin_cols].values, x_cat_test])
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV



# Добавим типы регуляризации

penalty = ['l1', 'l2']



# Зададим ограничения для параметра регуляризации

C = np.logspace(0, 4, 10)



#solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

class_weight = ['balanced', None]



# Создадим гиперпараметры

hyperparameters = dict(C=C, penalty=penalty,class_weight=class_weight)



model = LogisticRegression()

model.fit(X_train, y_train)



# Создаем сетку поиска с использованием 5-кратной перекрестной проверки

clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)



best_model = clf.fit(X_train, y_train)



# View best hyperparameters

print('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])

print('Лучшее C:', best_model.best_estimator_.get_params()['C'])

#print('Лучшее solver:', best_model.best_estimator_.get_params()['solver'])

print('Лучшее class_weight:', best_model.best_estimator_.get_params()['class_weight'])
lgr = LogisticRegression(penalty = 'l2', C=166, class_weight='None', solver ='saga')

lgr.fit(X_train, y_train)

probs = lgr.predict_proba(X_test)

probs = probs[:,1]





fpr, tpr, threshold = roc_curve(y_test, probs)

roc_auc = roc_auc_score(y_test, probs)



plt.figure()

plt.plot([0, 1], label='Baseline', linestyle='--')

plt.plot(fpr, tpr, label = 'Regression')

plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.legend(loc = 'lower right')

plt.show()
lgr = LogisticRegression(penalty = 'l2', C=166, class_weight='None',solver ='saga')

lgr.fit(X, Y)

probs = lgr.predict_proba(test)

probs = probs[:,1]
my_submission = pd.DataFrame({'client_id': id_test, 

                            'default': probs})

my_submission.to_csv('submission.csv', index=False)



my_submission
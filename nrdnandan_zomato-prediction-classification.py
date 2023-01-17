import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams

import seaborn as sns

import sys
data=pd.read_csv('../input/zomato.csv')
data.head()
data.shape
data.index
data.columns
data.info()
del data['url']

del data['phone']

del data['address']

del data['location']
data.head()
data.isnull().sum()

data['rate'] = data['rate'].replace('NEW',np.NaN)

data['rate'] = data['rate'].replace('-',np.NaN)

data=data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',

                         'listed_in(city)':'city'})
X=data.copy()
X.online_order=X.online_order.apply(lambda x: '1' if str(x)=='Yes' else '0')

X.book_table=X.book_table.apply(lambda x: '1' if str(x)=='Yes' else '0')
X.rate.dtype
X.rate=X.rate.astype(str)

X.rate=X.rate.apply(lambda x : x.replace('/5',''))

X.rate=X.rate.astype(float)
X.rate.dtype
X.cost.dtype
X.cost=X.cost.astype(str)

X.cost=X.cost.apply(lambda y : y.replace(',',''))

X.cost=X.cost.astype(float)
X.cost.dtype
X.online_order=X.online_order.astype(float)

X.book_table=X.book_table.astype(float)

X.votes=X.votes.astype(float)
X.info()
X.isnull().sum()
X_del=X.copy()

X_del.dropna(how='any',inplace=True)
X_del.isnull().sum()
X_del.info()
X_del.drop_duplicates(keep='first',inplace=True)
X_del.head()
sns.countplot(X_del['online_order'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Restaurants delivering online or Not')
sns.countplot(X_del['book_table'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Restaurants allowing table booking or not')
plt.rcParams['figure.figsize'] = (15, 9)

x = pd.crosstab(X_del['rate'], X_del['online_order'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])

plt.title('online order vs rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

y = pd.crosstab(X_del['rate'], X_del['book_table'])

y.div(y.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])

plt.title('table booking vs rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
sns.countplot(X_del['city'])

sns.countplot(X_del['city']).set_xticklabels(sns.countplot(X_del['city']).get_xticklabels(), rotation=90, ha="right")

fig = plt.gcf()

fig.set_size_inches(15,15)

plt.title('Location')
loc_plt=pd.crosstab(X_del['rate'],X_del['city'])

loc_plt.plot(kind='bar',stacked=True);

plt.title('Location - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Location',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');

plt.legend().remove();

sns.countplot(X_del['rest_type'])

sns.countplot(X_del['rest_type']).set_xticklabels(sns.countplot(X_del['rest_type']).get_xticklabels(), rotation=90, ha="right")

fig = plt.gcf()

fig.set_size_inches(15,15)

plt.title('Restuarant Type')
loc_plt=pd.crosstab(X_del['rate'],X_del['rest_type'])

loc_plt.plot(kind='bar',stacked=True);

plt.title('Rest type - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Rest type',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');

plt.legend().remove();

sns.countplot(X_del['type'])

sns.countplot(X_del['type']).set_xticklabels(sns.countplot(X_del['type']).get_xticklabels(), rotation=90, ha="right")

fig = plt.gcf()

fig.set_size_inches(15,15)

plt.title('Type of Service')
type_plt=pd.crosstab(X_del['rate'],X_del['type'])

type_plt.plot(kind='bar',stacked=True);

plt.title('Type - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Type',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');
sns.countplot(X_del['cost'])

sns.countplot(X_del['cost']).set_xticklabels(sns.countplot(X_del['cost']).get_xticklabels(), rotation=90, ha="right")

fig = plt.gcf()

fig.set_size_inches(15,15)

plt.title('Cost of Restuarant')


cost_for_two = pd.cut(X_del['cost'],bins = [0, 200, 500, 1000, 5000, 8000],labels = ['<=200', '<=500', '<=1000', '<=3000', '<=5000',])

cost_plt=pd.crosstab(X_del['rate'],cost_for_two)

cost_plt.plot(kind='bar',stacked=True);

plt.title('Avg cost - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Average Cost',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');
Y=X_del.copy()
dummy_rest_type=pd.get_dummies(Y['rest_type'])

dummy_type=pd.get_dummies(Y['type'])

dummy_city=pd.get_dummies(Y['city'])

dummy_cuisines=pd.get_dummies(Y['cuisines'])

dummy_dishliked=pd.get_dummies(Y['dish_liked'])

#dummy_reviewslist=pd.get_dummies(Y['reviews_list']) #Too much memory allocation
Y=pd.concat([Y,dummy_rest_type,dummy_type,dummy_city,dummy_cuisines,dummy_dishliked,#dummy_reviewslist

            ],axis=1)
del Y['rest_type']

del Y['type']

del Y['city']

del Y['cuisines']

del Y['dish_liked']

#del Y['reviews_list']
Y.head()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=Y.drop(['name',#'dish_liked',

          'reviews_list',

          'menu_item',#'cuisines'

         ],axis=1);
x_fit=scaler.fit_transform(x)

x=pd.DataFrame(x_fit,columns=x.columns)

x.info()
x.head()
#corr_x=x.corr().abs()
#corr_x
from sklearn.feature_selection import SelectKBest
#col_r=list(x)

#col_r
#col_r.insert(0, col_r.pop(col_r.index('rate')))

#col_r
#x_1 = x.loc[:, col_r]
#x_1.head()
#x_1.info()
#for i,a in enumerate(x_1.columns.values[0:1809]):

    #print('%s is %d' % (a,i))
#column_names=x_1.columns.values

#column_names[34]='Delivery_A'

#column_names[35]='Delivery_B'

#column_names[81]='Delivery_remove_1'

#column_names[82]='Delivery_remove_2'
#x_1.columns=column_names
#x_1.head()
#del x_1['Delivery_remove_1']

#del x_1['Delivery_remove_2']
#x_1.info()
X_init=x.drop(['rate'],axis=1)

split_x=X_init.iloc[:,:]

split_x.info()

split_x.shape

split_x
Y_init=x.drop(x.columns.difference(['rate']),axis=1)

split_y=Y_init.iloc[:,:]

split_y=split_y.astype(float)

split_y.shape

split_y
bestfeatures=SelectKBest(k='all')

fit=bestfeatures.fit(split_x,split_y)
fit
scores=pd.DataFrame(fit.scores_)

columns_=pd.DataFrame(split_x.columns)
featurescore=pd.concat([columns_,scores],axis=1)
featurescore.columns = ['Features','Score']
print(featurescore.nlargest(20,'Score'))
col_select=featurescore.nlargest(800,'Score')
col_select.drop('Score',axis=1,inplace=True)
col_select_list=list(col_select.Features)
col_select_list
x_select=split_x.loc[:,col_select_list]
x_select.head()
x_select.info()
x_select = x_select.loc[:, ~x_select.columns.duplicated()]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x_select,split_y,test_size=0.05,random_state=42)
from sklearn.linear_model import LinearRegression

linreg=LinearRegression()
linreg.fit(X_train,y_train)
Y_linreg_pred=linreg.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,Y_linreg_pred)
acc_len=linreg.score(X_train,y_train)
acc_len
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators=100,random_state=42)
rf_reg.fit(X_train,y_train)
Y_rgreg_pred=rf_reg.predict(X_test)
r2_score(y_test,Y_rgreg_pred)
acc_rfreg=rf_reg.score(X_train,y_train)
acc_rfreg
from sklearn.linear_model import RidgeCV

ridge=RidgeCV(alphas=[1e-10,1e-8,1e-6,1e-2,1e-1,1,10,20,100])
fit_ridge=ridge.fit(X_train,y_train)
fit_ridge.alpha_
Y_ridge_pred=ridge.predict(X_test)
r2_score(y_test,Y_ridge_pred)
acc_ridge=ridge.score(X_train,y_train)
acc_ridge
from sklearn.linear_model import LassoCV

lasso=LassoCV(alphas=[1e-3,1e-2,1e-1,1,10,20,100],max_iter=1e2)
fit_lasso=lasso.fit(X_train,y_train)
fit_lasso.alpha_
Y_lasso_pred=lasso.predict(X_test)
acc_lasso=lasso.score(X_train,y_train)
r2_score(y_test,Y_lasso_pred)
acc_lasso
from sklearn.neural_network import MLPRegressor

mlp=MLPRegressor(random_state=42)
mlp.fit(X_train,y_train)
acc_mlp=mlp.score(X_train,y_train)
Y_mlp_pred=mlp.predict(X_test)
r2_score(y_test,Y_mlp_pred)
acc_mlp
from sklearn.ensemble import ExtraTreesRegressor
rf_extrareg=ExtraTreesRegressor(n_estimators=100,random_state=42)
rf_extrareg.fit(X_train,y_train)
Y_extra_rgreg_pred=rf_extrareg.predict(X_test)
r2_score(y_test,Y_extra_rgreg_pred)
acc_extra_reg_score=rf_extrareg.score(X_train,y_train)
acc_extra_reg_score
from sklearn.neighbors import KNeighborsRegressor

knn=KNeighborsRegressor(n_jobs=-1)
knn.fit(X_train,y_train)
Y_knn_pred=knn.predict(X_test)
r2_score(y_test,Y_knn_pred)
acc_knn_score=knn.score(X_train,y_train)
acc_knn_score
from sklearn.svm import LinearSVR

svr=LinearSVR(random_state=42)
svr.fit(X_train,y_train)
Y_svr_pred=svr.predict(X_test)
r2_score(y_test,Y_svr_pred)
acc_svr=svr.score(X_train,y_train)
acc_svr
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

gbr=HistGradientBoostingRegressor(random_state=42)
gbr.fit(X_train,y_train)
Y_gbr_pred=gbr.predict(X_test)
r2_score(y_test,Y_gbr_pred)
acc_gbr=gbr.score(X_train,y_train)
acc_gbr
from sklearn.linear_model import ElasticNet

en=ElasticNet(random_state=42,alpha=0.0001,precompute=True)
en.fit(X_train,y_train)
Y_en_pred=en.predict(X_test)
r2_score(y_test,Y_en_pred)
acc_en=en.score(X_train,y_train)
acc_en
from sklearn.linear_model import BayesianRidge

bay=BayesianRidge()
bay.fit(X_train,y_train)
Y_bay_pred=bay.predict(X_test)
r2_score(y_test,Y_bay_pred)
acc_bay=bay.score(X_train,y_train)
acc_bay
from sklearn.linear_model import SGDRegressor

sgd=SGDRegressor(loss='squared_epsilon_insensitive',random_state=42,learning_rate='adaptive',max_iter=3000)
sgd.fit(X_train,y_train)
Y_sgd_pred=sgd.predict(X_test)
r2_score(y_test,Y_sgd_pred)
acc_sgd=sgd.score(X_train,y_train)
acc_sgd
data_class=Y.copy()

data_class.head()
del data_class['name']

del data_class['menu_item']

del data_class['reviews_list']
x_class=data_class.drop(['online_order','book_table'],axis=1)

y_class=data_class.drop(data_class.columns.difference(['online_order','book_table']),axis=1)
x_class
bestfeatures_class=SelectKBest(k='all')

fit_class_oo=bestfeatures_class.fit(x_class,y_class.online_order)

fit_class_bt=bestfeatures_class.fit(x_class,y_class.book_table)
fit_class_oo
fit_class_oo.scores_
fit_class_bt.scores_
class_score=pd.DataFrame(fit_class_oo.scores_)

class_columns_=pd.DataFrame(x_class.columns)
featureclass_score=pd.concat([class_columns_,class_score],axis=1)
featureclass_score.columns=['Features','Score']
print(featureclass_score.nlargest(1000,'Score'))
feature_select=featureclass_score.nlargest(90,'Score')
feature_select_list=list(feature_select.Features)
x_class_select=x_class.loc[:,feature_select_list]
x_class_select.info()
x_class_select = x_class_select.loc[:, ~x_class_select.columns.duplicated()]
X_class_train,X_class_test,y_class_train,y_class_test=train_test_split(x_class_select,y_class,test_size=0.05,random_state=42)
from sklearn.ensemble import RandomForestClassifier

forest_class=RandomForestClassifier(n_estimators=100,random_state=42)
forest_class.fit(X_class_train,y_class_train)
y_predict_class_forest=forest_class.predict(X_class_test)
y_predict_class_forest
y_train_score_forest=forest_class.score(X_class_train,y_class_train)
y_train_score_forest
y_test_score_forest=forest_class.score(X_class_test,y_class_test)
y_test_score_forest
from sklearn.neighbors import KNeighborsClassifier

knn_class=KNeighborsClassifier(n_jobs=-1)
knn_class.fit(X_class_train,y_class_train)
y_predict_class_knn=knn_class.predict(X_class_test)
y_predict_class_knn
y_train_score_knn=knn_class.score(X_class_train,y_class_train)
y_train_score_knn
y_test_score_knn=knn_class.score(X_class_test,y_class_test)
y_test_score_knn
from sklearn.neural_network import MLPClassifier

mlp_class=MLPClassifier()
mlp_class.fit(X_class_train,y_class_train)
y_predict_class_mlp=mlp_class.predict(X_class_test)
y_predict_class_mlp
y_train_score_mlp=mlp_class.score(X_class_train,y_class_train)
y_train_score_mlp
y_test_score_mlp=mlp_class.score(X_class_test,y_class_test)
y_test_score_mlp
!pip install scikit-multilearn
from skmultilearn.problem_transform import BinaryRelevance

from sklearn.linear_model import LogisticRegression
logit_class_bin=BinaryRelevance(LogisticRegression())
logit_class_bin.fit(X_class_train,y_class_train)
y_predict_class_logit_bin=logit_class_bin.predict(X_class_test)
y_predict_class_logit_bin
y_train_score_logit_bin=logit_class_bin.score(X_class_train,y_class_train)
y_train_score_logit_bin
y_test_score_logit_bin=logit_class_bin.score(X_class_test,y_class_test)
y_test_score_logit_bin
from skmultilearn.problem_transform import ClassifierChain

logit_class_chain=ClassifierChain(LogisticRegression())

logit_class_chain.fit(X_class_train,y_class_train)
y_predict_class_logit_chain=logit_class_chain.predict(X_class_test)
y_predict_class_logit_chain
y_train_score_logit_chain=logit_class_chain.score(X_class_train,y_class_train)
y_train_score_logit_chain
y_test_score_logit_chain=logit_class_chain.score(X_class_test,y_class_test)
y_test_score_logit_chain
from skmultilearn.problem_transform import LabelPowerset

logit_class_power=LabelPowerset(LogisticRegression())
logit_class_power.fit(X_class_train,y_class_train)
y_predict_class_logit_power=logit_class_power.predict(X_class_test)
y_predict_class_logit_power
y_train_score_logit_power=logit_class_power.score(X_class_train,y_class_train)
y_train_score_logit_power
y_test_score_logit_power=logit_class_power.score(X_class_test,y_class_test)
y_test_score_logit_power
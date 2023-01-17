import pandas as pd

import numpy as np

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")  # To ignore warnings

sns.set(rc={"figure.figsize":(12,8)})  # Set figure size to 12,8



pd.options.display.max_columns=150 # to display all columns 
# to run the code line by line

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"#run single line code
col_names = ['surgery','Age','Hospital Number','rectal temperature','pulse','respiratory rate','temperature of extremities','peripheral pulse',

             'mucous membranes','capillary refill time','pain','peristalsis','abdominal distension','nasogastric tube','nasogastric reflux', 'nasogastric reflux PH','rectal examination - feces','abdomen','packed cell volume','total protein','abdominocentesis appearance',

             'abdomcentesis total protein','outcome','surgical lesion','type of lesion_1','type of lesion_2','type of lesion_3','cp_data']
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data",sep=" ",na_values="?",engine="python",index_col=False,names=col_names)
test = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.test",sep=" ",na_values="?",engine="python",index_col=False,names=col_names)

test
data
df=data.copy()
tst=test.copy()
df.shape
tst.shape
df.dtypes
df.dtypes.value_counts()
df.isnull().sum()
dele_col=df.columns[(df.isnull().sum()*100)/df.shape[0]>40]

dele_col
df=df.drop(dele_col,axis=1)
tst=tst.drop(dele_col,axis=1)
num_cols=['Hospital Number', 'rectal temperature', 'pulse', 'respiratory rate',  'packed cell volume',

 'total protein',  'type of lesion_1','type of lesion_2','type of lesion_3']#'nasogastric reflux PH','abdomcentesis total protein',
num_df=df[num_cols]

num_df
num_tst=tst[num_cols]

num_tst
cat_cols= ['surgery','Age', 'temperature of extremities', 'peripheral pulse', 'mucous membranes', 'capillary refill time', 'pain', 'peristalsis', 'abdominal distension', 'nasogastric tube', 'nasogastric reflux', 'rectal examination - feces', 'abdomen',  'outcome', 'surgical lesion', 'cp_data']#'abdominocentesis appearance',

cat_df=df[cat_cols]

cat_df
cat_tst=tst[cat_cols]

cat_tst
cat_df.isnull().sum()
cat_tst.isnull().sum()
## fill na values 

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='most_frequent')



imputed_data = imputer.fit_transform(cat_df)



imputed_data=pd.DataFrame(imputed_data,columns=cat_df.columns)



print(imputed_data.isnull().sum())
imputed_tst = imputer.fit_transform(cat_tst)



imputed_tst=pd.DataFrame(imputed_tst,columns=cat_tst.columns)



print(imputed_tst.isnull().sum())
cols= imputed_data.columns



imputed_data[cols] = imputed_data[cols].astype("category")



print(imputed_data.dtypes)
cols= imputed_tst.columns



imputed_tst[cols] = imputed_tst[cols].astype("category")



print(imputed_tst.dtypes)
imput=SimpleImputer(strategy="mean")



imput_data = imput.fit_transform(num_df)



imput_data=pd.DataFrame(imput_data,columns=num_df.columns)



print(imput_data.isnull().sum())
imput_tst = imput.fit_transform(num_tst)



imput_tst=pd.DataFrame(imput_tst,columns=num_tst.columns)



print(imput_tst.isnull().sum())
for col in imputed_data.columns.values:

    imputed_data[col]=imputed_data[col].astype('category').cat.codes
for col in imputed_tst.columns.values:

    imputed_tst[col]=imputed_tst[col].astype('category').cat.codes
df_merge=pd.concat([imputed_data,imput_data],axis=1)
df_merge
df_merge.isnull().sum()
dff=df_merge.copy()
dff.columns
dff.shape
tst_merge=pd.concat([imputed_tst,imput_tst],axis=1)

tst_merge
tst_merge.isnull().sum()
# convert target variable into x,y

y=dff['outcome']

X=dff.drop(['outcome'],axis=1)
tst_merge=tst_merge.drop(['outcome'],axis=1)
from sklearn.model_selection import train_test_split
X_train,  X_val, y_train,y_val = train_test_split(X, y, stratify=y,test_size = 0.30, random_state = 222)



X_train.shape, X_val.shape, y_train.shape, y_val.shape

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
LRC = LogisticRegression()#solver='newton-cg',max_iter=500



LRC.fit(X_train, y_train)

y_pred_LRC = LRC.predict(X_val)



print(classification_report(y_val, y_pred_LRC))
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()#criterion = 'entropy', max_features = 'sqrt', max_depth = 15, random_state = 0



DTC.fit(X_train, y_train)

y_pred_DT = DTC.predict(X_val)



print(classification_report(y_val, y_pred_DT))
from sklearn.ensemble import RandomForestClassifier#rfc_65
rfc11 = RandomForestClassifier()#n_estimators = 1500, class_weight="balanced"



rfc11.fit(X_train, y_train)

y_pred_test_RF1 = rfc11.predict(X_val)



print(classification_report(y_val, y_pred_test_RF1))
rfc111 = RandomForestClassifier(n_estimators = 2500, class_weight="balanced")#



rfc111.fit(X_train, y_train)

y_pred_test_RF11 = rfc111.predict(X_val)



print(classification_report(y_val, y_pred_test_RF11))
from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(learning_rate=0.1, n_estimators = 500, max_features="sqrt")

GBC.fit(X_train, y_train)



y_pred_GBC = GBC.predict(X_val)

print(classification_report(y_val, y_pred_GBC))
from sklearn.ensemble import AdaBoostClassifier
estimator_model = AdaBoostClassifier(base_estimator = RandomForestClassifier(),

                                    n_estimators = 500,

                                    learning_rate=0.1,

                                    random_state = 0)

estimator_model.fit(X_train, y_train)

y_pred_adaboost = estimator_model.predict(X_val)



print(classification_report(y_val, y_pred_adaboost))



from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import MultiLabelBinarizer

import xgboost as xgb

from xgboost.sklearn import XGBClassifier  

from sklearn.model_selection import GridSearchCV
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
y_pred_xg = xgb.predict(X_val)



print(classification_report(y_val, y_pred_xg))


xgb11 = XGBClassifier(n_estimators=1000,max_depth=5,min_child_weight=1,seed=27)
xgb11.fit(X_train, y_train)
y_pred_xgb = xgb11.predict(X_val)



print(classification_report(y_val, y_pred_xgb))
from mlxtend.classifier import StackingClassifier



model1 = LogisticRegression(solver='newton-cg')

model2 =DecisionTreeClassifier(criterion = 'gini', max_features = 'sqrt', max_depth = 10, random_state = 0)

model3 = RandomForestClassifier()

model4 = XGBClassifier(n_estimators=100,max_depth=5,min_child_weight=1,seed=27)
meta_model = RandomForestClassifier()



stack = StackingClassifier(classifiers=[ model2, model4], meta_classifier=meta_model)#model1, model2,



stack.fit(X_train, y_train)

y_pred_stacked = stack.predict(X_val)



print(classification_report(y_val, y_pred_stacked))



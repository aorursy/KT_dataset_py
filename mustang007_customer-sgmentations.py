import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyoff
import plotly.express as  px
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from catboost import *
train = pd.read_csv('/kaggle/input/Coustomer_sgmentation/Train_aBjfeNk.csv')
test = pd.read_csv('/kaggle/input/Coustomer_sgmentation/Test_LqhgPWU.csv')
sub = pd.read_csv('/kaggle/input/Coustomer_sgmentation/sample_submission_wyi0h0z.csv')
train.head()
data = pd.concat([train,test], axis=0)
data.head()
for i in data.columns:
    if data[i].isnull().sum() !=0:
        print('this columns {} contain '.format(i),data[i].isnull().sum(),' null values')
data.select_dtypes(include='O').columns.tolist() # object dtypes columns
# Convert Numeric NaNs to -999
features = ["Age", "Work_Experience", "Family_Size"]
data[features] = data[features].fillna(-999)
    
marry = {'Yes':0, 'No':1}
data['Ever_Married'] = data['Ever_Married'].fillna(2)

for i, j in marry.items():
    data['Ever_Married'] = data['Ever_Married'].replace(i,j)
gender_mapping = {"Male": 0, "Female": 1}
data["Gender"] = data["Gender"].fillna(2)

for gender, label in gender_mapping.items():
    data["Gender"] = data["Gender"].replace(gender, label)
    
grad = {'Yes':0, 'No':1}
data['Graduated'] = data['Graduated'].fillna(1)

for i,j in grad.items():
    data['Graduated'] = data['Graduated'].replace(i,j)
# Nan in profession could mean that person is unemployed
prof_mapping = {"Artist": 0, "Doctor": 1, "Engineer": 2, "Entertainment": 3, "Executive": 4, "Healthcare": 5, "Homemaker": 6, "Lawyer": 7, "Marketing": 8}
data["Profession"] = data["Profession"].fillna(9)

for prof, label in prof_mapping.items():
    data["Profession"] = data["Profession"].replace(prof, label)

ss_mapping = {"Low": 0, "Average": 1, "High": 2}
data["Spending_Score"] = data["Spending_Score"].fillna(ss_mapping["Low"])

for ss, label in ss_mapping.items():
    data["Spending_Score"] = data["Spending_Score"].replace(ss, label)
 
# NaN in Var1 is just another category
var1_mapping = {"Cat_1": 0, "Cat_2": 1, "Cat_3": 2, "Cat_4": 3, "Cat_5": 4, "Cat_6": 5, "Cat_7": 6}
data["Var_1"] = data["Var_1"].fillna(7)

for var1, label in var1_mapping.items():
    data["Var_1"] = data["Var_1"].replace(var1, label)
 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Prof+Grad"] = data["Profession"]+data["Graduated"]
# data["Prof+Grad"] = le.fit_transform(data["Prof+Grad"])
data['prof+var'] = data['Profession'] + data['Var_1']
data['spend+family'] = data['Spending_Score'] + data['Family_Size']
temp = data.groupby(['Age']).agg({'Spending_Score':['count','mean','sum'],
                                   'Work_Experience':['count','sum','min','max','mean'],
                                   'Profession':['max','count'],
                                           'Graduated':['count'],
                                   'Ever_Married':['count'],
                                    'Gender':['count'], 
                                       'Family_Size':['count','sum','max'],
                                       'Age':['count'],
                                    'Var_1':['count','max','min']})
temp.columns = ['_'.join(x) for x in temp.columns]
temp
temp.skew()
data = pd.merge(data,temp,on=['Age'],how='left')

data
sa
Train = data.iloc[:len(train), :]
Test = data.iloc[len(train):, :]
# Finally label encode segmentation
seg_mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
seg_mapping_rev = {0: "A", 1: "B", 2: "C", 3: "D"}
for seg, label in seg_mapping.items():
    Train["Segmentation"] = Train["Segmentation"].replace(seg, label)
# 1 -> young, 2 -> middle-aged, 3 -> old, 4 -> retired and old
Train["Age_group"] = [1 if i<=33 else 2 if i>33 and i<65 else 3 if i>=65 and i<74 else 4 for i in Train["Age"].values]
Test["Age_group"] = [1 if i<=33 else 2 if i>33 and i<65 else 3 if i>=65 and i<74 else 4 for i in Test["Age"].values]
# Bin the ID column and add as feature
est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')
Train["Binned_ID"] = est.fit_transform(np.reshape(Train["ID"].values, (-1,1)))
Test["Binned_ID"] = est.transform(np.reshape(Test["ID"].values, (-1,1)))

Train
# feature to specify as categoreis
cat_feats = ['Gender', 'Ever_Married', 'Graduated', "Var_1", 'Profession', 'Age_group', "Binned_ID"]
cat_feats_inds = [Train.columns.get_loc(c) for c in cat_feats]
cat_feats_inds
X = Train.drop(columns=['ID','Segmentation'])
Y = Train['Segmentation']

Test.drop(columns=['ID','Segmentation'], inplace=True)
# CV: 1
kfold = KFold(n_splits=5, random_state=27, shuffle=True)
scores = list()
for train, test in kfold.split(X):
    x_train, x_test = X.iloc[train], X.iloc[test]
    y_train, y_test = Y[train], Y[test]
    
    model = LGBMClassifier(random_state=100, max_depth=3, n_estimators=200, learning_rate=0.1)
    model.fit(x_train, y_train, categorical_feature=cat_feats_inds)
    preds = model.predict(x_test)
    
    score = accuracy_score(y_test, preds)
    scores.append(score)
    print("Score: ", score)
first_fold = sum(scores)/len(scores)
print("\nAverage Score: ", first_fold, "\n\n")

# CV: 2
oof = []
kfold = StratifiedKFold(n_splits=10, random_state=27, shuffle=True)
scores = list()
for train, test in kfold.split(X, Y):
    x_train, x_test = X.iloc[train], X.iloc[test]
    y_train, y_test = Y[train], Y[test]
#     model = CatBoostClassifier(random_state=27,verbose = 0)
    model = LGBMClassifier(class_weight = 'balanced', max_depth=8, n_estimators=200, learning_rate=0.3)
    model.fit(x_train, y_train)
   
    preds = model.predict(x_test)
    
    score = accuracy_score(y_test, preds)
    scores.append(score)
    print("Score: ", score)
    oof.append(model.predict_proba(Test))
    
second_fold = sum(scores)/len(scores)
print("\nAverage Score: ", second_fold)
print("\n\nFinal Average: ", first_fold*0.5 + second_fold*0.5)
oof_cat = oof
oof_lgb = oof

oof_lgb[3]
oof_lgb = np.mean(oof_lgb, 0 )
#     oof_cat = np.mean(oof_cat, 0 )
# ensemble Model
final = oof_lgb
preds = [np.argmax(x) for x in final]
# from scipy.stats import chi2_contingency
# target_related_cols=[]
# for i in cat_col:
#    if i in train.columns:
#       cross_table=pd.crosstab( train.loc[:,i],train["Segmentation"])
#       obs=cross_table.values
#       chi2, p, dof, ex = chi2_contingency(obs, correction=False)
#       if p < 0.05:
#         print("Null statement:" ,i ," is rejected ",np.round(p,5)," which means it has some association with the target variable")
#         target_related_cols.append(i)
# target_related_cols.pop()
# print(target_related_cols)
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# vif=pd.DataFrame()
# vif["vif"]=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
# vif["features"]=X.columns
# req_col=list(vif.query("vif<5")["features"])
# req_col

# from imblearn.over_sampling import SMOTE
# smote = SMOTE(sampling_strategy='auto')
# x_train, y_train = smote.fit_sample(x_train, y_train)
# print(x_train.shape,y_train.shape)
# from sklearn.model_selection import GridSearchCV
# param_test1 = {
# #     'colsample_bytree' :[0.5,0.6,0.7,0.8,0.9],
#      'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
# #     'max_depth':[1,2,3,4,5,6,7,8,9],
#     'learning_rate':[0.001,0.01,0.03,0.1,0.3]
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier(),param_grid = param_test1,verbose=True,n_jobs=-1, cv=5)
# gsearch1.fit(x_train, y_train)
# gsearch1.best_params_
len(preds)
sub['Segmentation'] = preds
sub['Segmentation'] = sub['Segmentation'].map({0:'A',
                                              1:'B',
                                              2:'C',
                                              3:'D'})
sub.to_csv('sub.csv')
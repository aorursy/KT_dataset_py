import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
train =pd.read_csv('../input/janata-hack-customer-segmentation/train.csv')
test =pd.read_csv('../input/janata-hack-customer-segmentation/test.csv')
train['data']='train'
test['data']='test'
df=pd.concat([train,test],ignore_index=True, sort=False)
df.shape
df.ID = df.ID-458982
df.ID.nunique()
train.ID.nunique()
test.ID.nunique()
df.drop('ID',axis=1,inplace=True)
trainset = df[df.data=='train']
trainset.drop('data', axis =1 ,inplace = True)
trainset.sample(5)
df.isnull().sum()
df.info()
df[df.Ever_Married.isnull()].isnull().sum()
df[df.Ever_Married.isnull()].Spending_Score.value_counts()
df.loc[ (pd.isnull(df['Ever_Married'])) & (df['Spending_Score'] != 'Low'), 'Ever_Married'] = 'Yes'
df[df.Ever_Married.isnull()].isnull().sum()
# lawer ,married
# healthcare ,unmarried
df.loc[ (pd.isnull(df['Ever_Married'])) & (df['Profession'] == 'Lawyer'), 'Ever_Married'] = 'Yes'
df.loc[ (pd.isnull(df['Ever_Married'])) & (df['Profession'] == 'Healthcare'), 'Ever_Married'] = 'No'
df[df.Ever_Married.isnull()].isnull().sum()
# graduated married
df.loc[ (pd.isnull(df['Ever_Married'])) & (df['Graduated'] == 'Yes'), 'Ever_Married'] = 'Yes'
df.loc[ (pd.isnull(df['Ever_Married'])) & (df['Graduated'] == 'No'), 'Ever_Married'] = 'No'
df[df.Ever_Married.isnull()].isnull().sum()
# artists graduated
df.loc[ (pd.isnull(df['Graduated'])) & (df['Profession'] == 'Artist'), 'Graduated'] = 'Yes'
df[df.Graduated.isnull()].isnull().sum()
# married graduated
df.loc[ (pd.isnull(df['Graduated'])) & (df['Ever_Married'] == 'Yes'), 'Graduated'] = 'Yes'
df.loc[ (pd.isnull(df['Graduated'])) & (df['Ever_Married'] == 'No'), 'Graduated'] = 'No'
df[df.Graduated.isnull()].isnull().sum()
df[df.Profession.isnull()].isnull().sum()
# var_1=6 , profession=artist
df.loc[ (pd.isnull(df['Profession'])) & (df['Var_1']=='Cat_6') ,'Profession'] = 'Artist'
df[df.Profession.isnull()].isnull().sum()
for i in ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist','Executive', 'Doctor', 'Homemaker', 'Marketing']:
    print(i,'\n',df[df.Profession == i]['Work_Experience'].median(),'\n')
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Healthcare'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Engineer'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Lawyer'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Entertainment'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Artist'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Executive'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Doctor'), 'Work_Experience'] = 1
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Homemaker'), 'Work_Experience'] = 8
df.loc[ (pd.isnull(df['Work_Experience'])) & (df['Profession'] == 'Marketing'), 'Work_Experience'] = 1
df['Work_Experience'].fillna(1,inplace=True)
#ever_married = yes , family size = 2
df.loc[ (pd.isnull(df['Family_Size'])) & (df['Ever_Married'] == 'Yes'), 'Family_Size'] = 2
df.loc[ (pd.isnull(df['Family_Size'])) & (df['Ever_Married'] == 'No'), 'Family_Size'] = 1
# fam siz 4 proff healthcare
df.loc[ (pd.isnull(df['Profession'])) & (df['Family_Size'] == 4), 'Profession'] = 'Healthcare'
df.Profession.fillna(method='ffill',inplace=True)
df.Var_1.fillna(method='ffill',inplace=True)


df.isnull().sum()
trainset = df[df.data=='train']
X = trainset.drop(['Segmentation','data'],axis=1)
y = trainset['Segmentation']
X.shape
y.shape
def target(x):
    if x == 'A':
        x=1
    elif x == 'B':
        x=2
    elif x == 'C':
        x=3
    elif x == 'D':
        x=4
    return(x)
y = y.apply(target)


y.value_counts()
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
cate_features_index = np.where(X.dtypes != float)[0]
cate_features_index

X.Ever_Married.value_counts()
X.columns
cat_cols = np.array([0, 1, 3, 4, 6, 8])
cat_cols
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
# 1st target
modelCat = CatBoostClassifier()
modelCat.fit(X_train,y_train,cat_features=cate_features_index,eval_set=(X_test,y_test))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
y_pred = modelCat.predict(X_test)
from sklearn import metrics  
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.f1_score(y_test, y_pred, average='weighted'))

df
def gender(x):
    if x=='Male':
        x=1
    else:
        x=0
    return(x)
df.Gender = df.Gender.apply(gender)
def yes_no(x):
    if x=='Yes':
        x=1
    else:
        x=0
    return(x)
df.Ever_Married = df.Ever_Married.apply(yes_no)
df.Graduated = df.Graduated.apply(yes_no)
df['divorce'] = df['Family_Size']- df['Ever_Married']
def divor(x):
    if x==0:
        x=1
    else:
        x=0
    return(x)
df['divorce'] = df['divorce'].apply(divor)
df['work'] = df['Work_Experience']/df['Age']
df=pd.get_dummies(df,columns=['Profession','Var_1','Spending_Score'])
df.data.value_counts()
traindata = df[df.data=='train']
X = traindata.drop(['Segmentation','data'],axis=1)
y = traindata['Segmentation']
# balancing via SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE('auto')
X_sm, y_sm = smote.fit_sample(X,y)
print(X_sm.shape, y_sm.shape)
from numpy import mean
from numpy import std
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
model = LGBMClassifier(num_leaves=11,n_estimators=200)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
n_scores = cross_val_score(model, X_sm, y_sm, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# complte train fit
model.fit(X, y)
testdata = df[df.data=='test'].drop('data',axis=1)
X_test = testdata.drop('Segmentation',axis=1)
y_pred_testset = model.predict(X_test)
y_pred_testset
# pd.DataFrame(y_pred_testset).to_csv('predictions.csv')




#seg_et_1=pd.concat([test_ids,seg_et],ignore_index=False,join='outer',axis=1)
#df=pd.get_dummies(df,columns=['Profession','Var_1','Spending_Score'])



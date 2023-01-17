# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from ipywidgets import interact,widgets
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
## Functions for reading data , sanity checks ,univariate and bivariate analysis
def read_data(path):
    df = pd.read_csv(path)
    return df


def sanity_checks(df):
    # number of columns and records
    print(f"Shape of data {df.shape}")
    
    # data type of columns 
    print(df.info())

    # check for duplicacy
    duplicacy_chk = df.shape[0]-df.drop_duplicates().shape[0]
    if duplicacy_chk>0:
        print(f"There are duplicate records {duplicacy_chk}")
    else :
        print(f"There are no duplicate records")

    # Missing Value Check
    print(df.isnull().sum())
    series_null = df.isnull().sum()*100/df.shape[0]
    plt.figure()
    plt.plot(series_null,marker='o')
    plt.xticks(rotation=90)
    plt.xlabel("Features")
    plt.ylabel("%Missing")
    plt.title("Missing Values in Data")
    plt.grid()
    
def variable_distribution(col,var_type,data):
    
    df = data_dict[f"{data}_data"]

    if var_type == 'categorical':
        if col in df.columns:
#             plt.figure()
            (df[col].value_counts()*100/df.shape[0]).plot(kind='bar')
            plt.xlabel(f"{col} Flag")
            plt.ylabel("% in each category")
            plt.title(col)
        else :
            print(f"{col} not in {data}")
    else :
        if col in df.columns :
            plt.hist(df[col].dropna())
        else :
            print(f"{col} not in {data}")
            
def bivariate_distribution(col1,col2,col1_type,col2_type,data):
    df = data_dict[f"{data}_data"]
    plt.figure(figsize=(15,5))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1,3,1)
    variable_distribution(col1,col1_type,data)
    plt.subplot(1,3,2)
    variable_distribution(col2,col2_type,data)
    plt.subplot(1,3,3)

    if col1!=col2 and col1_type=='categorical' and col2_type=='categorical':

        plt.scatter(col1,col2,data = df.groupby([col1,col2],as_index=False).PassengerId.count(),
                   s = "PassengerId"
                   )
        plt.xlabel(col1)
        plt.ylabel(col2)
    elif col1!=col2 and (col1_type!='categorical' or col2_type!='categorical'):
        if col1_type=='categorical':cat_col,cont_col = col1,col2
        else : cat_col,cont_col = col2,col1
        print(cat_col,cont_col)

        for val in df[cat_col].unique():
            plt.hist(df[df[cat_col]==val][cont_col].dropna(),label=f'val_{val}')
        plt.legend()
        
        print(df.groupby([cat_col],as_index=False).agg({cont_col:['mean','median']}))
    else :
        print("Please select different columns")
## Missing Value Imputation
def missing_value(df,cat_cols,cont_cols,data_type,impute_dict=None):
    if data_type =='train_data':
        impute_dict = {}
        for col in  cat_cols:
            mode_val = df[col].mode()[0]
            impute_dict[col] = mode_val
            df[col] = df[col].apply(lambda x:mode_val if str(x)=='nan' else x)
        for col in cont_cols:
            median_val = df[col].median()
            impute_dict[col] = mode_val
            df[col] = df[col].apply(lambda x:median_val if str(x)=='nan' else x)

    else :
        for col in cat_cols+cont_cols:
            if col in impute_dict:
                df[col] = df[col].apply(lambda x:impute_dict[col] if str(x)=='nan' else x)
            else :
                df[col] = df[col].apply(lambda x:df[col].median()if str(x)=='nan' else x)
#     df.drop(columns =cat_cols+cont_cols,inplace=True)
    return df,impute_dict

def family_cate(family_size):
    if family_size<=1:return 'single'
    elif family_size>1 and family_size<=2 : return 'couple'
    elif family_size>2 and family_size<=4 : return 'small'
    elif family_size>4 : return 'large'

def feature_creation(df):
    pass_count = df.groupby(['Ticket'],as_index=False).PassengerId.count().rename(columns={'PassengerId':'num_pass'})
    df = pd.merge(df,pass_count,on=['Ticket'])
    df['Avg_Fare'] = df['Fare']/df['num_pass']
    df['Fare_log'] = df.Fare.apply(lambda x : np.log(x) if x>0 else 0)
    df['Cabin_sub'] = df.Cabin.apply(lambda x:str(x)[0] if str(x)!='nan' else 'nan' )
    df['Name_sub'] = df.Name.apply(lambda x : x.split(",")[1].split('.')[0].strip())
    
    name_count = df['Name_sub'].value_counts()
    req_title = list(name_count[name_count>=10].index)
    
    df['Name_sub']=df['Name_sub'].apply(lambda x: x if x in req_title else 'Others')
    
    df['Family_Size'] = df['SibSp'].astype(int)+df['Parch'].astype(int)+1
    df['familyS'] = df['Family_Size'].apply(lambda x: family_cate(x))
    
    df['Ticket_sub'] = df['Ticket'].apply(lambda x : 'num' if x.isdigit() else x.split(" ")[0].replace("/","").replace(".","") )
    
    df.drop(columns=['num_pass','Family_Size','Fare','Ticket','Name','Cabin'],inplace=True)
    return df 


## Feature Transformation
def feat_transform(df,cols_dummy,cols_scale):
    # Categorical Variables
    dummy_df = pd.get_dummies(df[cols_dummy])
    dummy_df['key'] = df.index
    df['key'] = df.index
    df = pd.merge(df,dummy_df,on='key',how='left')

    # Scaling Continuous Variables
    for col in cols_scale:
        df[f'{col}_scale'] = (df[col]-min(df[col]))/(max(df[col])-min(df[col]))
        
    df.drop(columns =['key']+cols_dummy+cols_scale,inplace=False)
    
    return df

## EDA on base data
categorical = ['Survived', 'Pclass','Sex', 'SibSp','Parch', 'Embarked']
continuous = [ 'Age', 'Fare']
data_dict = {"train_data":read_data("/kaggle/input/titanic/train.csv"),
             "test_data":read_data("/kaggle/input/titanic/test.csv")}
sanity_checks(data_dict['train_data'])
sanity_checks(data_dict['test_data'])
interact(bivariate_distribution,col1=categorical+continuous,col2=categorical+continuous,
         col1_type=['categorical','continuous'],col2_type=['categorical','continuous'],data=['train','test'])

## deep dive into relation between fare and survival rate
train_data = data_dict['train_data']
per_ls = []
for percentile_ in range(10,101,5):
    per_ls.append([percentile_,np.percentile(train_data.Fare,percentile_),
          train_data[train_data.Fare>=np.percentile(train_data.Fare,percentile_)].groupby(['Ticket']).PassengerId.count().mean(),
          train_data[train_data.Fare>=np.percentile(train_data.Fare,percentile_)].Survived.sum(),
        train_data[train_data.Fare>=np.percentile(train_data.Fare,percentile_)].Survived.count(),
                  ]
         )
    
# train_data.Fare.describe()
per_df = pd.DataFrame(per_ls,columns=['percentile','value','mean_passengers','survived','total_passen'])
per_df['per_survived'] = per_df['survived']/per_df['total_passen']

plt.plot("percentile","mean_passengers",data=per_df,label='mean_passengers')
plt.plot("percentile","per_survived",data=per_df,label='per_survived')
plt.legend(loc='upper right')
plt.ylabel("% Survived/Mean Passengers")
plt.twinx()
plt.scatter("percentile","value",data=per_df,label='value')
plt.legend()
plt.ylabel("Percentile Value")
## Data Prep
data_dict = {"train_data":read_data("/kaggle/input/titanic/train.csv"),
             "test_data":read_data("/kaggle/input/titanic/test.csv")}
data_dict_cleaned = {}
for data in data_dict.keys():

    df = data_dict[data].copy()
    print(str(list(df.columns)).replace("'",""))
    
    cat_cols = [col.strip() for col in input("Input categorical columns for missing value comma sep: ").split(",") if col!=' ']
    cont_cols = [col.strip() for col in input("Input continuous columns for missing value comma sep: ").split(",") if col!=' ']
    impute_dict = {}
    df_missing,impute_dict = missing_value(df,cat_cols,cont_cols,data,impute_dict)
    df_feat_create = feature_creation(df_missing)
    
    print(str(list(df_feat_create.columns)).replace("'",""))
    cols_dummy = [col.strip() for col in input("Input col names for dummy variable creation comma sep: ").split(",") if col!=' ']
    cols_scale = [col.strip() for col in input("Input col names for scaling comma sep: ").split(",") if col!=' ']

    
    df_feat_trans = feat_transform(df_feat_create,cols_dummy,cols_scale)
    data_dict_cleaned[data] = df_feat_trans
#Sex,Embarked,Cabin_sub, Name_sub, familyS, Ticket_sub
#Age,Avg_Fare, Fare_log
# # define feature and target columns
print(str(list(data_dict_cleaned['train_data'].columns)).replace("'",""))
cols_not_req = [col.strip() for col in input("Cols to be removed from modelling : ").split(",") if col!=' ']
print(cols_not_req)
cols_model = [col for col in data_dict_cleaned['train_data'].columns if col not in cols_not_req]
target_col = 'Survived'

for col in cols_model:
    if col not in data_dict_cleaned['test_data'].columns:
        data_dict_cleaned['test_data'][col] = 0
## Experimenting with various classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

classifiers = { 'RF':RandomForestClassifier(random_state =100),
               'DT':DecisionTreeClassifier(random_state =100),
               'GB':GradientBoostingClassifier(random_state =100),
               'LR' :LogisticRegression(random_state =100),
               'ET':ExtraTreeClassifier(random_state =100),
               'SVC':SVC(random_state =100)
                }
base_cv_score = {}
test_preds = {}
model_dict = {}
for model in classifiers.keys():
    x_train = data_dict_cleaned['train_data'][cols_model]
    y_train = data_dict_cleaned['train_data'][target_col]
    tempmodel = classifiers[model]
    tempmodel.fit(x_train,y_train)
    model_dict[model] = tempmodel       
    test_preds[model] = tempmodel.predict(data_dict_cleaned['test_data'][cols_model])
    
    base_cv_score[model] = (cross_val_score(tempmodel,data_dict_cleaned['train_data'][cols_model],
                                           data_dict_cleaned['train_data'][target_col]
                                           ,cv=5).mean())
    
    print(f"{model} done!!")
plt.plot(list(base_cv_score.keys()),list(base_cv_score.values()),marker='o')
plt.grid()
from sklearn.model_selection import GridSearchCV
params_grid = {'n_estimators':[200,400],
              'max_depth' : [4,6,10]
              }
# gs = GridSearchCV(model_dict['RF'],params_grid,cv=5)
# gs.fit(data_dict_cleaned['train_data'][cols_model],data_dict_cleaned['train_data'][target_col])
print(gs.best_score_,gs.best_params_)
RF = RandomForestClassifier(n_estimators=200,max_depth=6,random_state=100)
RF.fit(data_dict_cleaned['train_data'][cols_model],data_dict_cleaned['train_data'][target_col])
print(cross_val_score(RF,data_dict_cleaned['train_data'][cols_model],data_dict_cleaned['train_data'][target_col],cv=5).mean())
## submission file
## Random Forest gives 0.8 score to be in top 8%
submission_df = data_dict_cleaned['test_data'][['PassengerId']]
submission_df['Survived'] = RF.predict(data_dict_cleaned['test_data'][cols_model])
submission_df.to_csv("/kaggle/working/submission.csv",index=False)

# from sklearn.ensemble import VotingClassifier
# x_train = data_dict_cleaned['train_data'][cols_model]
# y_train = data_dict_cleaned['train_data'][target_col]
# voting_model = VotingClassifier(estimators = [(name,model) for name,model in model_dict.items() if name in ['RF','GB']])
# voting_model.fit(x_train,y_train)
# print(cross_val_score(voting_model,x_train,y_train,cv=5).mean())

# submission_df = data_dict_cleaned['test_data'][['PassengerId']]
# submission_df['Survived'] = voting_model.predict(data_dict_cleaned['test_data'][cols_model])
# submission_df.to_csv("/kaggle/working/submission.csv",index=False)


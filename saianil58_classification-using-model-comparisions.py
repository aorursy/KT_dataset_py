# import all libraries



import pandas as pd #basic

import numpy as np #basic

# import pandas_profiling as pp #EDA



from scipy.stats import shapiro,pearsonr #Stats

import scipy.stats as stats #Stats



import plotly.express as px #visualization

import plotly.graph_objs as go#visualization

import plotly.offline as py#visualization

import plotly.figure_factory as ff#visualization

import seaborn as sns



from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split #Split data

from sklearn import preprocessing #manipulate data

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,precision_score,recall_score,f1_score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import RadiusNeighborsClassifier





from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.ensemble import RandomForestClassifier

from imblearn.ensemble import BalancedRandomForestClassifier





from imblearn.ensemble import BalancedBaggingClassifier



from sklearn.ensemble import VotingClassifier





import xgboost  as xgb

from catboost import CatBoostClassifier

import lightgbm as lgbm
train= pd.read_csv('train_ctrUa4K.csv',index_col='Loan_ID')

loans = train.copy()
print ("Rows     : " ,loans.shape[0])

print ("Columns  : " ,loans.shape[1])

print ("\nFeatures : \n" ,loans.columns.tolist())

print ("\nMissing values :  ", loans.isnull().sum().values.sum())

print ("\nUnique values :  \n",loans.nunique())
loans.describe(include='all').transpose()
df_new=loans.copy()

# pp.ProfileReport(df_new)
# loans['total_income']=loans['ApplicantIncome']+loans['CoapplicantIncome']



Id_col = ['Loan_ID']

target_col = ['Loan_Status']

exclude = []    

cat_cols = loans.nunique()[loans.nunique() < 6].keys().tolist()

cat_cols = [x for x in cat_cols if x not in target_col + exclude]

num_cols = [x for x in loans.columns if x not in cat_cols+ target_col + Id_col + exclude]

bin_cols   = loans.nunique()[loans.nunique() == 2].keys().tolist()

multi_cols = [i for i in cat_cols if i not in bin_cols+exclude]

   



def fill_cat_cols(col_name,data_frame):

    col_df=data_frame[col_name].value_counts().index

    nans = data_frame[col_name].isna()

    ##Key logic stats here

    col_df=(data_frame[col_name].value_counts())

    for x in col_df.index:

        col_df[x]=(col_df[x]/data_frame.shape[0])*100

    weights_g = col_df.values/100    

    import random

    ## use random.choices and give the respective distribution

    replacement=random.choices(col_df.index,weights=weights_g, k=data_frame[col_name].isnull().sum())

    ## use the above random values to keep in df again

    data_frame.loc[nans,col_name] = replacement



def fill_median_num_cols(col_name,data_frame):

    data_frame[col_name].fillna(value=data_frame[col_name].median(),inplace=True)



def drop_exclude_cols(col_name,data_frame):

    data_frame.drop(col_name,axis=1,inplace=True)



def change_cat_num_data(data_frame,is_test,is_cat,is_num):

    if is_test == 'Y':

        bin_cols.remove('Loan_Status')

    #Label encoding Binary columns

    if is_cat == 'Y':

        le = LabelEncoder()

        for i in bin_cols :

#             print(i)

            data_frame[i] = le.fit_transform(data_frame[i])

        data_frame = pd.get_dummies(data = data_frame,columns = multi_cols )

    if is_num == 'Y':

        std = StandardScaler()

        scaled = std.fit_transform(data_frame[num_cols])

        scaled = pd.DataFrame(scaled,columns=num_cols,index=data_frame.index)

        df_og = data_frame.copy()

        data_frame = data_frame.drop(columns = num_cols,axis = 1)

        data_frame = data_frame.merge(scaled,left_index=True,right_index=True,how = "left")

    return data_frame
for x in cat_cols:

    fill_cat_cols(x,loans)

for x in ['LoanAmount','Loan_Amount_Term']:

    fill_median_num_cols(x,loans)

for x in exclude:

    drop_exclude_cols(x,loans)   

loans=change_cat_num_data(loans,'N','Y','Y')    


##seperating dependent and independent variables



X = loans.drop('Loan_Status',axis=1)

y = loans['Loan_Status']



##Split the train and test with 30%ratio



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.3, random_state=111,stratify =y)



def knn_classifier(k):

    knn = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric_params = None, n_jobs = 1, n_neighbors = k, p = 2, weights = 'uniform')

    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)

    y_test_pred = knn.predict(X_test)

    print('Train Accuracy score for k ={}  is  {:.3f}'.format(k, accuracy_score(y_train, y_train_pred)))

    print('Test Accuracy score for k ={}  is  {:.3f}'.format(k, accuracy_score(y_test, y_test_pred)))

for k in range(1, 20, 2):

    knn_classifier(k)
knn = KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = 5, p = 2, weights = 'uniform')

knn.fit(X_train, y_train)

y_train_pred = knn.predict(X_train)

y_test_pred = knn.predict(X_test)

print('Train Accuracy score for k ={}  is  {:.3f}'.format(5, accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score for k ={}  is  {:.3f}'.format(5, accuracy_score(y_test, y_test_pred)))
## Logistic regression



log_reg = LogisticRegression(random_state=1,solver='liblinear')

log_reg.fit(X_train, y_train)

y_train_pred = log_reg.predict(X_train)

y_test_pred = log_reg.predict(X_test)

print('Train Accuracy score LogisticRegression  is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy scoreLogisticRegression is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
#GaussianNB

gb = GaussianNB()

gb.fit(X_train, y_train)

y_train_pred = gb.predict(X_train)

y_test_pred = gb.predict(X_test)

print('Train Accuracy score GaussianNB  is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy GaussianNB is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


#DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=3,min_samples_leaf=3)

dtc.fit(X_train, y_train)

y_train_pred = dtc.predict(X_train)

y_test_pred = dtc.predict(X_test)

print('Train Accuracy score DecisionTreeClassifier  is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy DecisionTreeClassifier is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
#RadiusNeighborsClassifier



rnc = RadiusNeighborsClassifier(radius=1.678,outlier_label =0)

rnc.fit(X_train, y_train)

y_train_pred = rnc.predict(X_train)

y_test_pred = rnc.predict(X_test)

print('Train Accuracy score RadiusNeighborsClassifier  is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy RadiusNeighborsClassifier is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
#BaggingClassifier

k_bgc = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5),n_estimators=100,max_samples=.7,max_features=.6,random_state=1)

k_bgc.fit(X_train,y_train)

y_train_pred = k_bgc.predict(X_train)

y_test_pred = k_bgc.predict(X_test)

print('Train Accuracy score  is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy   is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
lr_bgc = BaggingClassifier(base_estimator=log_reg,max_features=.6,n_estimators=100,max_samples=.7,random_state=1)

lr_bgc.fit(X_train,y_train)

y_train_pred = lr_bgc.predict(X_train)

y_test_pred = lr_bgc.predict(X_test)

print('Train Accuracy score   is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score  is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
gb_bgc = BaggingClassifier(base_estimator=gb,max_features=.6,n_estimators=35,max_samples=.7,random_state=1)

gb_bgc.fit(X_train,y_train)

y_train_pred = gb_bgc.predict(X_train)

y_test_pred = gb_bgc.predict(X_test)

print('Train Accuracy score   is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score  is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
dt_bgc = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5),max_features=.6,n_estimators=35,max_samples=.7,random_state=1)

dt_bgc.fit(X_train,y_train)

y_train_pred = dt_bgc.predict(X_train)

y_test_pred = dt_bgc.predict(X_test)

print('Train Accuracy score   is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score  is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
lr_abc = AdaBoostClassifier(base_estimator=LogisticRegression(random_state=1,solver='liblinear'),random_state=1,learning_rate=1,n_estimators=50)

lr_abc.fit(X_train,y_train)

y_train_pred = lr_abc.predict(X_train)

y_test_pred = lr_abc.predict(X_test)

print('Train Accuracy score   is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score  is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
gb_abc = AdaBoostClassifier(base_estimator=GaussianNB(),random_state=1,learning_rate=.5,n_estimators=75)

gb_abc.fit(X_train,y_train)

y_train_pred = gb_abc.predict(X_train)

y_test_pred = gb_abc.predict(X_test)

print('Train Accuracy score  is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
dtc_abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5),random_state=1,learning_rate=.7,n_estimators=100)

dtc_abc.fit(X_train,y_train)

y_train_pred = dtc_abc.predict(X_train)

y_test_pred = dtc_abc.predict(X_test)

print('Train Accuracy score   is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score  is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
abc = AdaBoostClassifier(random_state=1,learning_rate=.5,n_estimators=75)

abc.fit(X_train,y_train)

y_train_pred = abc.predict(X_train)

y_test_pred = abc.predict(X_test)

print('Train Accuracy score   is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score  is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


gbc = GradientBoostingClassifier(learning_rate=1,n_estimators=100,random_state=50, min_samples_leaf=9)

gbc.fit(X_train,y_train)

y_train_pred = gbc.predict(X_train)

y_test_pred = gbc.predict(X_test)

print('Train Accuracy score    is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score  is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


rc = RandomForestClassifier(n_estimators =20,random_state=1,max_depth=3, min_samples_leaf=9)

rc.fit(X_train,y_train)

y_train_pred = rc.predict(X_train)

y_test_pred = rc.predict(X_test)

print('Train Accuracy score    is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


brf = BalancedRandomForestClassifier(max_depth=3,n_estimators=50, random_state=0)

brf.fit(X_train,y_train)

y_train_pred = brf.predict(X_train)

y_test_pred = brf.predict(X_test)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


bbc_lr = BalancedBaggingClassifier(base_estimator=log_reg, random_state=0)

bbc_lr.fit(X_train,y_train)

y_train_pred = bbc_lr.predict(X_train)

y_test_pred = bbc_lr.predict(X_test)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


bbc_k = BalancedBaggingClassifier(base_estimator=knn, random_state=0)

bbc_k.fit(X_train,y_train)

y_train_pred = bbc_k.predict(X_train)

y_test_pred = bbc_k.predict(X_test)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


bbc_gb = BalancedBaggingClassifier(base_estimator=gb, random_state=0)

bbc_gb.fit(X_train,y_train)

y_train_pred = bbc_gb.predict(X_train)

y_test_pred = bbc_gb.predict(X_test)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


bbc_dt = BalancedBaggingClassifier(base_estimator=dtc, random_state=0)

bbc_dt.fit(X_train,y_train)

y_train_pred = bbc_dt.predict(X_train)

y_test_pred = bbc_dt.predict(X_test)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))


bbc_R = BalancedBaggingClassifier(base_estimator=rnc, random_state=0)

bbc_R.fit(X_train,y_train)

y_train_pred = bbc_R.predict(X_train)

y_test_pred = bbc_R.predict(X_test)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
def get_vc_estimators(n):

    est = []

    vc_df=model_performances.sort_values('Test_Accuracy_score',ascending=False).head(n)['Model_Name']

    for x in vc_df.values:

        tup=(x,x)

        est.append(tup)

    print(est)

# get_vc_estimators(10)    
vc = VotingClassifier(

    estimators=[

        ('abc', abc), ('brf', brf), ('dtc', dtc),

                ('lr_bgc', lr_bgc), ('gb_bgc', gb_bgc),

        ('dt_bgc', dt_bgc), ('lr_abc', lr_abc), ('log_reg', log_reg), 

                ('rc', rc)

               ]

    ,voting='soft')

vc.fit(X_train,y_train)

y_train_pred = vc.predict(X_train)

y_test_pred = vc.predict(X_test)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))
def model_report(

    m_type,

    model,

    model_name,

    actual_name

    ):

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)

    df = pd.DataFrame({

        'Model_Type': [m_type],

        'Model_Name': [model_name],

        'Model_act_name':[actual_name],

        'Train_Accuracy_score': [accuracy_score(y_train, y_train_pred)],

        'Test_Accuracy_score': [ accuracy_score(y_test, y_test_pred)],

        })    

    return df

model1=model_report('Base',knn,'knn','KNN-5')

model2=model_report('Base',log_reg,'log_reg','Logis Reg')

model3=model_report('Base',dtc,'dtc','Dec Tree')

model4=model_report('Base',gb,'gb','GB')

model5=model_report('Base',rnc,'rnc','RNC')



model6=model_report('Bagging',k_bgc,'k_bgc','BGC-KNN-5')

model7=model_report('Bagging',lr_bgc,'lr_bgc','BGC-LR')

model8=model_report('Bagging',gb_bgc,'gb_bgc','BGC-GB')

model9=model_report('Bagging',dt_bgc,'dt_bgc','BGC-DT')



model10=model_report('Boosting',lr_abc,'lr_abc','ABC-LR')

model11=model_report('Boosting',gb_abc,'gb_abc','ABC-GB')

model12=model_report('Boosting',dtc_abc,'dtc_abc','ABC-DT')

model13=model_report('Boosting',abc,'abc','ABC-Def')



model14=model_report('Random Forest',rc,'rc','RFC')

model15=model_report('Gradient Boost',gbc,'gbc','GBC')



model16=model_report('Balance',brf,'brf','BAL-RF')

model17=model_report('Balance',bbc_lr,'bbc_lr','BAL-LR')

model18=model_report('Balance',bbc_k,'bbc_k','BAL-KNN-5')

model19=model_report('Balance',bbc_dt,'bbc_dt','BAL-DT')

model20=model_report('Balance',bbc_gb,'bbc_gb','BAL-GB')

model21=model_report('Balance',bbc_R,'bbc_R','BAL-Rad-C')



model22=model_report('Voting',vc,'vc','Voting Classifier')





#concat all models

model_performances = pd.concat([model1,model2,model3,

                                model4,model5,model6,

                                model7,model8,model9,

                                model10,model11,model12,

                                model13,model14,model15,

                                model16,model17,model18,

                                model19,model20,model21,model22

                               ],axis = 0).reset_index()



model_performances = model_performances.drop(columns = "index",axis =1)

D_train = xgb.DMatrix(X_train, label=y_train)

D_test = xgb.DMatrix(X_test, label=y_test)

param = {

    'eta': 0.1,

    'max_depth': 3,

    'objective': 'multi:softprob',

    'num_class': 3,

    }

steps = 10  # The number of training iterations

xgb_b = xgb.train(param, D_train, steps)

preds = xgb_b.predict(D_test)

preds_train = xgb_b.predict(D_train)

best_preds_train = np.asarray([np.argmax(line) for line in preds_train])

best_preds_test = np.asarray([np.argmax(line) for line in preds])

print ('Train Accuracy score is  {:.3f}'.format(accuracy_score(y_train,best_preds_train)))

print ('Test Accuracy score is  {:.3f}'.format(accuracy_score(y_test,best_preds_test)))

model_performances=model_performances.append(pd.DataFrame({

    'Model_Type': ['XG Boost Basic'],

    'Model_Name': ['xgb_b'],

    'Model_act_name': ['xgb_b'],

    'Train_Accuracy_score': [accuracy_score(y_train,

                             best_preds_train)],

    'Test_Accuracy_score': [accuracy_score(y_test, best_preds_test)],

    }), ignore_index=True)

cat = CatBoostClassifier(

    iterations=10,

    learning_rate=.5,

    depth=5,

    eval_metric='Accuracy',

    use_best_model=True,

    random_seed=42,

    loss_function=None,

    )



# Fit model



cat_b = cat.fit(X_train, y_train, eval_set=(X_test, y_test))

y_train_pred = cat_b.predict(X_train)

y_test_pred = cat_b.predict(X_test)

print ('Train Accuracy score is  {:.3f}'.format(accuracy_score(y_train,

        y_train_pred)))

print ('Test Accuracy score is  {:.3f}'.format(accuracy_score(y_test,

        y_test_pred)))

model_performances = model_performances.append(pd.DataFrame({

    'Model_Type': ['CAT Boost Basic'],

    'Model_Name': ['cat_b'],

    'Model_act_name': ['cat_b'],

    'Train_Accuracy_score': [accuracy_score(y_train, y_train_pred)],

    'Test_Accuracy_score': [accuracy_score(y_test, y_test_pred)],

    }), ignore_index=True)
lgm_b = lgbm.LGBMClassifier(n_estimators=50,random_state=100)

lgm_b.fit(X_train, y_train)

y_test_pred = lgm_b.predict(X_test)

y_train_pred = lgm_b.predict(X_train)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))

model_performances=model_performances.append(pd.DataFrame({

    'Model_Type': ['LGBM Basic'],

    'Model_Name': ['lgm_b'],

    'Model_act_name': ['lgm_b'],

    'Train_Accuracy_score': [accuracy_score(y_train,

                             y_train_pred)],

    'Test_Accuracy_score': [accuracy_score(y_test, y_test_pred)],

    }), ignore_index=True)
# Work with missing values 
df_XGB= train.copy()

df_XGB['total_income']=df_XGB['ApplicantIncome']+df_XGB['CoapplicantIncome']

for x in cat_cols:

    fill_cat_cols(x,df_XGB)

for x in exclude:

    drop_exclude_cols(x,df_XGB)

df_XGB=change_cat_num_data(df_XGB,'N','Y','N')     



##seperating dependent and independent variables



X = df_XGB.drop('Loan_Status',axis=1)

y = df_XGB['Loan_Status']



##Split the train and test with 30%ratio



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.3, random_state=111,stratify =y)



D_train = xgb.DMatrix(X_train, label=y_train)

D_test = xgb.DMatrix(X_test, label=y_test)

param = {

    'eta': 0.1,

    'max_depth': 3,

    'objective': 'multi:softprob',

    'num_class': 3,

    }

steps = 10  # The number of training iterations

xgb_n = xgb.train(param, D_train, steps)

preds = xgb_n.predict(D_test)

preds_train = xgb_n.predict(D_train)

best_preds_train = np.asarray([np.argmax(line) for line in preds_train])

best_preds_test = np.asarray([np.argmax(line) for line in preds])

print ('Train Accuracy score is  {:.3f}'.format(accuracy_score(y_train,best_preds_train)))

print ('Test Accuracy score is  {:.3f}'.format(accuracy_score(y_test,best_preds_test)))

model_performances=model_performances.append(pd.DataFrame({

    'Model_Type': ['XGBoost Adv'],

    'Model_Name': ['xgb_n'],

    'Model_act_name': ['adv'],

    'Train_Accuracy_score': [accuracy_score(y_train,

                             best_preds_train)],

    'Test_Accuracy_score': [accuracy_score(y_test, best_preds_test)],

    }), ignore_index=True)

df_LGM= train.copy()

# df_LGM['total_income']=df_LGM['ApplicantIncome']+df_LGM['CoapplicantIncome']

# for x in exclude:

#     drop_exclude_cols(x,df_LGM)

df_LGM=change_cat_num_data(df_LGM,'N','N','N')  



##seperating dependent and independent variables

for x in cat_cols:

    df_LGM[x]=df_LGM[x].astype('category')





X = df_LGM.drop('Loan_Status',axis=1)

y = df_LGM['Loan_Status']



##Split the train and test with 30%ratio



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.3, random_state=111,stratify =y)



lgm_c = lgbm.LGBMClassifier(n_estimators=100,random_state=100,max_depth=3,learning_rate=.2,num_leaves=10,importance_type='gain')

lgm_c.fit(X_train, y_train,categorical_feature=cat_cols)

y_test_pred = lgm_c.predict(X_test)

y_train_pred = lgm_c.predict(X_train)

print('Train Accuracy score is  {:.3f}'.format( accuracy_score(y_train, y_train_pred)))

print('Test Accuracy score is  {:.3f}'.format( accuracy_score(y_test, y_test_pred)))

model_performances=model_performances.append(pd.DataFrame({

    'Model_Type': ['LGBM Cat Cols'],

    'Model_Name': ['lgm_c'],

    'Model_act_name': ['adv'],

    'Train_Accuracy_score': [accuracy_score(y_train,

                             y_train_pred)],

    'Test_Accuracy_score': [accuracy_score(y_test, y_test_pred)],

    }), ignore_index=True)

df_cat= train.copy()



df_cat=change_cat_num_data(df_cat,'N','N','N')  



df_cat['Credit_History']=df_cat['Credit_History'].fillna(30)

df_cat['Credit_History']=df_cat['Credit_History'].astype('int')

df_cat['Credit_History']=df_cat['Credit_History'].astype('str')

df_cat['Credit_History']=df_cat['Credit_History'].str.replace('30','XXX')

#for catboost only

for x in cat_cols:

    df_cat[x]=df_cat[x].fillna('XXX')



##seperating dependent and independent variables

for x in cat_cols:

    df_cat[x]=df_cat[x].astype('object')



X = df_cat.drop('Loan_Status',axis=1)

y = df_cat['Loan_Status']



##Split the train and test with 30%ratio



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.3, random_state=111,stratify =y)



cat = CatBoostClassifier(

    iterations=10,

    learning_rate=.5,

    depth=5,

    random_seed=42,

    loss_function=None,

    )



# Fit model



cat_m = cat.fit(X_train, y_train, cat_features=cat_cols)

y_train_pred = cat_m.predict(X_train)

y_test_pred = cat_m.predict(X_test)

print ('Train Accuracy score is  {:.3f}'.format(accuracy_score(y_train,

        y_train_pred)))

print ('Test Accuracy score is  {:.3f}'.format(accuracy_score(y_test,

        y_test_pred)))

model_performances = model_performances.append(pd.DataFrame({

    'Model_Type': ['CAT Boost Cat Cols'],

    'Model_Name': ['cat_m'],

    'Model_act_name': ['adv'],

    'Train_Accuracy_score': [accuracy_score(y_train, y_train_pred)],

    'Test_Accuracy_score': [accuracy_score(y_test, y_test_pred)],

    }), ignore_index=True)
model_performances['diff']=(model_performances['Train_Accuracy_score']-model_performances['Test_Accuracy_score'])*100

model_performances.sort_values('Test_Accuracy_score',ascending=False)





# good_models=model_performances[(model_performances['Model_act_name'] != 'adv')&(model_performances['Test_Accuracy_score']>=0.77)]['Model_Name'].values
validate= pd.read_csv('test_lAUu6dG.csv',index_col='Loan_ID')

#generate prediction on test data

# validate['total_income']=validate['ApplicantIncome']+validate['CoapplicantIncome']

for x in cat_cols:

    fill_cat_cols(x,validate)

for x in ['LoanAmount','Loan_Amount_Term']:

    fill_median_num_cols(x,validate)

for x in exclude:

    drop_exclude_cols(x,validate)   

validate=change_cat_num_data(validate,'Y','Y','Y')    
dfcat= pd.read_csv('test_lAUu6dG.csv',index_col='Loan_ID')



dfcat=change_cat_num_data(dfcat,'N','N','N')  



dfcat['Credit_History']=dfcat['Credit_History'].fillna(30)

dfcat['Credit_History']=dfcat['Credit_History'].astype('int')

dfcat['Credit_History']=dfcat['Credit_History'].astype('str')

dfcat['Credit_History']=dfcat['Credit_History'].str.replace('30','XXX')

#for catboost only

for x in cat_cols:

    dfcat[x]=dfcat[x].fillna('XXX')



##seperating dependent and independent variables

for x in cat_cols:

    dfcat[x]=dfcat[x].astype('object')
predc=lr_bgc.predict(validate)

output = pd.DataFrame({'Loan_ID': validate.index,

                       'Loan_Status': predc})

output['Loan_Status']=output['Loan_Status'].replace(1,'Y')

output['Loan_Status']=output['Loan_Status'].replace(0,'N')

# output.to_csv(r'C:\Users\samu0315\Desktop\Mine\Personal\gl_aiml\supervised learning\GIT Practice\Classification\Binary Classfication\AV Comp bank\sol.csv', index=False)
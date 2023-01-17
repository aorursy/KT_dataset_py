import pandas as pd
import matplotlib as plt
import seaborn as sns


#reading dataset
df1=pd.read_csv('../input/diabetes-dataset/diabetic_data.csv',keep_default_na=' ')
#no. of rows and columns
df1.shape
#column names
df1.columns
df1.head()
categorical=df1.select_dtypes(include=['object'])
numeric=df1.select_dtypes(exclude=['object'])
categorical['encounter_id']=df1['encounter_id']
#setting index
numeric.set_index(['encounter_id'])
#categorical.set_index(['encounter_id'])
def drop_col(df):
        drop_col=['patient_nbr']
        for  x in df.columns:
            if x!='encounter_id' and x.endswith('_id'):
                drop_col.append(x)
                
        
        return df.drop(drop_col,axis=1)
    
    

numeric1=drop_col(numeric).copy()
#Mean
numeric1.apply(lambda x:x.mean(),axis=0)
numeric2=numeric1.loc[:,numeric1.columns!='encounter_id']
numeric2.describe().transpose()
#Quartiles to find IQR and figure out ouliers
Q1=numeric2.apply(lambda x:x.quantile(0.25),axis=0)
Q3=numeric2.apply(lambda x:x.quantile(0.75),axis=0)
IQR=Q3-Q1
# to return outliers
def dcf(x):
    IQR=x.quantile(0.75)-x.quantile(0.25)
    b=x.quantile(0.25)-IQR
    c=x.quantile(0.75) +IQR
    outliers=[x.loc[x> c],x.loc[x< b]]
    return outliers
s=numeric2.apply(lambda x: dcf(x),axis=0)
#list of outliers
s
#Percentage of outliers
s.apply(lambda x: len(x[0])*100/len(df1))
#Null Values in columns 
numeric2.apply(lambda x:x.loc[x.isnull()],axis=1)
#Null values in rows
numeric2.apply(lambda x:x.loc[x.isnull()],axis=0)
#correlation among numerical attributes
numeric2.corr()
import numpy as np
categorical.describe(include=[np.object]).transpose().iloc[:,:]


def drop_columns_with_imbalance_distribution(categorical):
    a=categorical.describe(include=[np.object]).transpose()
    return categorical.drop(a[a['freq']>0.8*a['count']].transpose().columns,axis=1)

def drop_columns_with_null_distribution(categorical):
    a=categorical.describe(include=[np.object]).transpose()
        
        
    return categorical.drop(a[a['top']=='?'].transpose().columns,axis=1)


def drop_columns_with_high_levels(categorical):
    a=categorical.describe(include=[np.object]).transpose()
        
        
    return categorical.drop(a[a['unique']>100].transpose().columns,axis=1)
categorical1=drop_columns_with_imbalance_distribution(categorical).copy()
categorical2=drop_columns_with_null_distribution(categorical1)
categorical3=drop_columns_with_high_levels(categorical2)
categorical3
# to handle gender column
def race_bin(x):
    if x=='Caucasian':
        return 1
    elif x=='AfricanAmerican':
        return 0
    else: return -1000

def gender_bin(x):
    if x=='Male':
        return 1
    elif x=='Female':
        return 0
    else:
        return -1000

def age_bin(x):
    if x=='[70-80)':
        return 80
    elif x=='[60-70)':
        return 70
    elif x=='[50-60)':
            return 60
    elif x=='[80-90)':
        return 90
    else:
        return -1000
    
def insulin(x):
    if x=='No':
        return 0
    elif x=='Steady':
        return 1
    elif x=='Yes':
        return 1
    
    else:return -1000

def change(x):
    if x=='No':
        return 0
    elif x=='Ch':
        return 1
    else: return -1000 





# to handle categorical columns with No, steady and yes

    



def readmitted_code(x):
    if x=='NO':
        return 0
    elif x=='>30':
        return 1
    elif x=='<30':
        return 1
    
    else:return -1000

def diabetesmed(x):
    if x=='No':
        return 0
    elif x=='Yes':
        return 1
    else: return -1000
# to handle Changed column

categorical4=pd.DataFrame()
categorical4['encounter_id']=categorical3['encounter_id']   
categorical4['age']=categorical3['age'].apply(lambda x: age_bin(x))
categorical4['gender']=categorical3['gender'].apply(lambda x: gender_bin(x))
categorical4['race']=categorical3['race'].apply(lambda x: race_bin(x))
categorical4['insulin']=categorical3['insulin'].apply(lambda x: insulin(x))
categorical4['diabetesMed']=categorical3['diabetesMed'].apply(lambda x: diabetesmed(x))
categorical4['change']=categorical3['change'].apply(lambda x: change(x))
categorical4['readmitted']=categorical3['readmitted'].apply(lambda x: readmitted_code(x))



categorical4
y=categorical4['readmitted']
categorical_subset_edit_final=categorical4.drop(['readmitted'],axis=1)
dumm=categorical_subset_edit_final.drop('encounter_id',axis=1)
dumm=dumm.apply(lambda x: x.astype('category'),axis=0)
dumm=pd.get_dummies(dumm,drop_first=True)
dumm.apply(lambda x: x.astype('category'),axis=1)
#to remove dummy trap (multi collinearity caused due to dummy variable)
#dumm=dumm.drop(['age_70','gender_1','race_1','insulin_1','diabetesMed_0','change_0'],axis=1)
dumm['encounter_id']=categorical4['encounter_id']

dumm
import numpy as np
from sklearn.preprocessing import StandardScaler
#scaling the numeric columns
a=StandardScaler()

numeric2.loc[:, numeric2.columns != 'encounter_id']=a.fit_transform(numeric2.loc[:, numeric2.columns != 'encounter_id'])
numeric2['encounter_id']=dumm['encounter_id']
import pandas as pd
# merging numeric and categorical columns

cleaned_df=dumm.merge(numeric2, how='inner',on=dumm['encounter_id'])
#

'''
time_in_hospital correlated with num_medications,num_lab_procedures,num_diagnoses
num_procedures,num_medications are correlated
num_medications,num_procedurs are correlated
num_inpatient correlated with 


'''

#dropping encounter id of x and y dataframes
cleaned_df=cleaned_df.drop(['encounter_id_y','encounter_id_x'],axis=1)
#cleaned_df.shape
a=pd.DataFrame()
b=pd.DataFrame()
#a['cleaned_df']=
a['cleaned_df']=cleaned_df.columns
b['df']=df1.columns
#a['df']
a.to_csv('columns_list.csv')
b.to_csv('b.csv')
#a['cleaned_df']
cleaned_df_final=cleaned_df.rename(index=str,columns={'key_0':'encounter_id'})
#cleaned_df_final=cleaned_df_final.drop(['age_-1000','gender_-1000','race_-1000','insulin_-1000','diabetesMed_0','change_0'],axis=1)
cleaned_df_final.columns
cleaned_df_final=cleaned_df_final[['age_60', 'age_70', 'age_80', 'age_90', 'gender_0',
       'gender_1', 'race_0', 'race_1', 'insulin_0', 'insulin_1',
       'diabetesMed_1', 'change_1', 'time_in_hospital', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'number_diagnoses']]
#randomforest, decision tree, logistic, naive babe’s
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,recall_score,precision_score,roc_curve
train,validate,y_train,y_validate=train_test_split(cleaned_df_final,y,train_size=0.7,shuffle=True)
validate
len(train.columns)
#Global Model Building
model=GaussianNB()
model.fit(train,y_train)
predictions=model.predict(validate)
confusion_matrix(predictions,y_validate)
#roc_curve(y_validate, predictions, pos_label=2)

lg=LogisticRegression(C=0.1,penalty='l2',solver='newton-cg')
deci=DecisionTreeClassifier(max_depth=6,min_samples_split=0.5,min_samples_leaf=1,max_features=0.6)
#rf=RandomForestClassifier(bootstrap,)
nb=GaussianNB()
#Logistic Regression
log_metrics=pd.DataFrame()
log_results=pd.DataFrame()
c_array=[]
confusion=[]
solver_array=[]
log_results['y_validate']=y_validate
c_list=np.arange(0.1,0.7,0.1)
solver_list=['newton-cg']
log_results['y_validate']=y_validate
penalty='l2'
for solver in solver_list:
    for c in c_list:
        lg=LogisticRegression(C=c,penalty=penalty,solver=solver,verbose=True,n_jobs=5)
        model=lg.fit(train,y_train)
        log_results['predictions']=model.predict(validate)
        c_array.append(c)
        solver_array.append(solver)
        confusion.append(confusion_matrix(log_results['predictions'],y_validate))
#log_results['c']=c_array
#log_results['solver']=solver_array
log_metrics['confusion']=confusion
log_metrics['confusion']

lg=LogisticRegression(C=c,penalty=penalty,verbose=True,solver='newton-cg')
lg.fit(train,y_train)
predictions=lg.predict(validate)
confusion_matrix(predictions,y_validate)
#accuracy 62 %

# Naive Baye's
model1=nb.fit(train,y_train)
predictions_nb=model1.predict(validate)

confusion_matrix(predictions_nb,y_validate)

# Random Forest
'''
rf_results=pd.DataFrame()
results=[]
rf_metrics=pd.DataFrame()
max_leaf_nodes_array=[]
confusion=[]
min_impurity_decrease_array=[]
rf_results['y_validate']=y_validate
max_depth_list=np.arange(3,6,1)
max_leaf_nodes_list=np.arange(2,7,1)

min_impurity_decrease_list=[0.01,0.05,0.1]
oob_score=True
min_fraction_leaf_list=np.arange(0.4,0.8,0.1)
for max_depth in max_depth_list:
    for max_leaf_nodes in max_leaf_nodes_list:
        for min_impurity_decrease in min_impurity_decrease_list:
            rf=RandomForestClassifier(n_estimators=150,bootstrap=True,max_features='auto',max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease,min_samples_split=3,oob_score=True)
            model=rf.fit(train,y_train)
            results.append(model.predict(validate))
            max_leaf_nodes_array.append(max_leaf_nodes)
            min_impurity_decrease_array.append(min_impurity_decrease)
        
        confusion.append(confusion_matrix(predictions,y_validate))
rf_metrics['min_impurity_decrease_array']=min_impurity_decrease_array
rf_metrics['max_leaf_nodes_array']=max_leaf_nodes_array
#
'''

rf=RandomForestClassifier(oob_score=True,verbose=True,n_jobs=1,n_estimators=150)
rf.fit(train,y_train)
predictions_rf=rf.predict(validate)
confusion_matrix(predictions_rf,y_validate)


from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,Conv2D,MaxPooling1D,BatchNormalization,Dropout
from keras.optimizers import Adam
opt = Adam(lr=0.1)

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

lrr = ReduceLROnPlateau(monitor='loss', min_delta=1e-4, factor=0.02,  patience=10)
early_stop = EarlyStopping(monitor='loss', min_delta=1e-2, patience=10, verbose=1)
error=[]
model=Sequential()
model.add(Dense(50,input_dim=20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train,y_train,epochs=100,callbacks=[early_stop],verbose=0)
#a=model.evaluate(validate,y_validate)
predictions=model.predict(validate)
confusion_matrix(predictions,y_validate)
train.shape
#train = np.expand_dims(train, axis=2)
train.shape[0]
nrows,ncols=train.shape
train_cnn=np.array(train).reshape(nrows, ncols, 1)
train_cnn.shape
# Reshape the training set in the form of (nrows,ncols,1) and input shape=(ncols,1)

model=Sequential([
    Conv1D(16,kernel_size=1,activation='relu',input_shape=(20,1),),
MaxPooling1D(2),
Dropout(0.5),
BatchNormalization(),


Flatten(),
Dense(10,activation='relu'),
Dense(1,activation='softmax')])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_cnn,y_train,epochs=10,callbacks=[early_stop],verbose=1)
nrows,ncols=validate.shape
validate_cnn=np.array(validate).reshape(nrows, ncols, 1)
predictions=model.predict(validate_cnn)
confusion_matrix(predictions,y_validate)
validate_cnn
#a=np.array(train).reshape((-1,22,1))
#b=np.array(y_train).reshape((-1,1))
import xgboost as xgb
train.shape
params = {'eta': 0.02, 'max_depth': 5, 'subsample': 0.7, 'colsample_bytree': 0.7, 'objective': 'binary:logistic', 'seed': 99, 'silent': 1, 'eval_metric':'auc', 'nthread':4}

xg_train = xgb.DMatrix(train, label=y_train)
n_folds = 5
early_stopping = 10
cv = xgb.cv(params, xg_train, 5000, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=0)
model=xgb.train(params,xg_train)
validate
#Xgboost
xg_val=xgb.DMatrix(validate)
predictions=model.predict(xg_val)
def predict_0_or_1(x):
    if x<0.64:
        return 0
    else: return 1
predictions=pd.Series(predictions).apply(lambda x:predict_0_or_1(x))
confusion_matrix(predictions,y_validate)
from sklearn import tree
from sklearn.metrics import confusion_matrix
#Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train, y_train)
predictions=clf.predict(validate)
confusion_matrix(predictions,y_validate)
train_y.shape
vddfdffdfdddfv
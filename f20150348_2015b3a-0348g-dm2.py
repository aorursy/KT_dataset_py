#from google.colab import drive



#drive.mount('/content/drive') 
#cd drive/
#cd My Drive
#cd Mllab3
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/train.csv")

data = data_orig
data.head()

data.tail()
data.info()
df=data
from collections import Counter

#Enrolled Fill PREV MLU Reason Area

#mode MIC Worker Class MOC Hispanic MSA REG MOVE Live PREV Teen
for i in range(1, len(df.columns)):

    print(df.columns[i])

    print(Counter(df.iloc[:,i]).keys()) # equals to list(set(words))

    print(Counter(df.iloc[:,i]).values())# counts the elements' frequency

    #print(df.columns[i]," has ",df.iloc[:,i].unique(),len(df.iloc[:,i].unique()))
df = df.replace({'?': np.nan})
#df = df.drop(columns=['Enrolled', 'Fill','Worker Class','PREV' ,'MLU' ,'Reason' ,'Area','State',"Detailed", 'COB SELF', 'COB MOTHER', 'COB FATHER', 'Teen', 'Live', 'MOVE', 'MOC', 'MIC', 'REG', 'MSA']) #

df.info()
df.fillna(df.mode().iloc[0], inplace=True)
df.head()
df.columns[df.isnull().any()].tolist()

y=df['Class']
df.columns
df=df.drop(columns=['Class'])
df.columns
for i in range(1, len(df.columns)):

    print(df.columns[i]," has ",len(df.iloc[:,i].unique()))
mylist = list(df.select_dtypes(include=['object']).columns)

mylist

data1 = pd.get_dummies(df, columns=mylist)
data1.head()


'''

data = pd.read_csv("train.csv", sep=',')

df=data

from collections import Counter

train_labels=df['Class']

train=df.drop(columns=['Class'])

from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

mylist = list(df.select_dtypes(include=['object']).columns)

df = df.replace({'?': np.nan})

df.fillna(df.mode().iloc[0], inplace=True)

data1 = pd.get_dummies(df, columns=mylist)

train_labels=data1['Class']

train=data1.drop(columns=['Class'])

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train,train_labels,test_size=0)

sel = SelectFromModel(RandomForestClassifier(n_estimators=175,bootstrap=False,class_weight='balanced',max_leaf_nodes=180,criterion='entropy',min_samples_leaf=53,max_depth=17,min_samples_split=4,random_state=17,n_jobs=-1),threshold=0.0003)

sel.fit(X_train, y_train)

selected_feat= X_train.columns[(sel.get_support())]

list13=selected_feat

len(list13)

list13[:64]

list[64:]

'''



data1=data1[['ID', 'Age', 'IC', 'OC', 'Timely Income', 'Gain', 'Loss', 'Stock',

       'Weight', 'NOP', 'Own/Self', 'Vet_Benefits', 'Weaks', 'WorkingPeriod',

       'Worker Class_WC1', 'Worker Class_WC2', 'Worker Class_WC3',

       'Worker Class_WC4', 'Worker Class_WC5', 'Schooling_Edu1',

       'Schooling_Edu12', 'Schooling_Edu13', 'Schooling_Edu14',

       'Schooling_Edu15', 'Schooling_Edu16', 'Schooling_Edu2',

       'Schooling_Edu3', 'Schooling_Edu4', 'Schooling_Edu5', 'Schooling_Edu6',

       'Schooling_Edu9', 'Enrolled_Uni1', 'Enrolled_Uni2', 'Married_Life_MS1',

       'Married_Life_MS2', 'Married_Life_MS3', 'Married_Life_MS4', 'MIC_MIC_A',

       'MIC_MIC_C', 'MIC_MIC_D', 'MIC_MIC_F', 'MIC_MIC_G', 'MIC_MIC_H',

       'MIC_MIC_I', 'MIC_MIC_M', 'MIC_MIC_N', 'MIC_MIC_O', 'MIC_MIC_T',

       'MOC_MOC_A', 'MOC_MOC_B', 'MOC_MOC_C', 'MOC_MOC_D', 'MOC_MOC_E',

       'MOC_MOC_F', 'MOC_MOC_G', 'MOC_MOC_H', 'MOC_MOC_K', 'MOC_MOC_L',

       'Cast_TypeA', 'Cast_TypeD', 'Hispanic_HA', 'Hispanic_HD', 'Sex_F',

       'Sex_M','MOC_MOC_G', 'MOC_MOC_H', 'MOC_MOC_I', 'MOC_MOC_J', 'MOC_MOC_K',

       'MOC_MOC_L', 'MOC_MOC_M', 'Cast_TypeA', 'Cast_TypeD', 'Hispanic_HA',

       'Hispanic_HC', 'Hispanic_HD', 'Hispanic_HE', 'Sex_F', 'Sex_M', 'MLU_NO',

       'MLU_YES', 'Reason_JL2', 'Reason_JL4', 'Full/Part_FA', 'Full/Part_FB',

       'Full/Part_FC', 'Full/Part_FD', 'Tax Status_J1', 'Tax Status_J2',

       'Tax Status_J3', 'Tax Status_J4', 'Tax Status_J5', 'Tax Status_J6',

       'Area_S', 'State_s15', 'Detailed_D2', 'Detailed_D3', 'Detailed_D4',

       'Detailed_D5', 'Detailed_D6', 'Detailed_D7', 'Detailed_D8',

       'Summary_sum1', 'Summary_sum2', 'Summary_sum3', 'Summary_sum4',

       'Summary_sum5', 'Summary_sum6', 'MSA_StatusA', 'MSA_StatusL',

       'MSA_StatusO', 'REG_StatusA', 'REG_StatusB', 'MOVE_StatusA',

       'MOVE_StatusB', 'Live_NO', 'Live_YES', 'Teen_B', 'Teen_M',

       'COB FATHER_c24', 'COB FATHER_c36', 'COB MOTHER_c24', 'COB MOTHER_c36',

       'COB SELF_c24', 'COB SELF_c36', 'Citizen_Case1', 'Citizen_Case2',

       'Citizen_Case3']]
xtrain=data1

ytrain=y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(xtrain,y,random_state=10,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import AdaBoostClassifier

import numpy as np

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB as NB

nb = NB()
nb.fit(X_train,y_train)

nb.score(X_test,y_test)

y_pred_RF = nb.predict(X_test)

print(roc_auc_score(y_test,y_pred_RF))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_RF).ravel()

print(confusion_matrix(y_test, y_pred_RF))
from sklearn.tree import DecisionTreeClassifier

dTree1 = DecisionTreeClassifier(max_depth=5)

dTree1.fit(X_train,y_train)
from sklearn.ensemble import BaggingClassifier

bag1=BaggingClassifier(base_estimator=dTree1) 
bag1.fit(X_train,y_train)
bag1.score(X_test,y_test)
y_pred=bag1.predict(X_test)
confusion_matrix(y_test, y_pred)
X_train,X_test,y_train,y_test=train_test_split(xtrain,y,random_state=10,test_size=0)
regr=RandomForestClassifier(n_estimators=200,bootstrap=False,class_weight='balanced',max_leaf_nodes=256,criterion='entropy',min_samples_leaf=47,max_depth=15,min_samples_split=4,random_state=97,n_jobs=-1)

#(n_estimators=100,class_weight='balanced',min_samples_leaf=9,max_depth=17,min_samples_split=4,random_state=17,n_jobs=-1)regr=RandomForestClassifier(n_estimators=100,class_weight='balanced',min_samples_leaf=9,max_depth=17,min_samples_split=4,random_state=17,n_jobs=-1)
clf=AdaBoostClassifier(n_estimators = 3, base_estimator=regr, algorithm = "SAMME.R")
clf.fit(X_train ,y_train)
'''

from sklearn.model_selection import GridSearchCV

rf_temp = RandomForestClassifier() #Initialize the classifier object

parameters = {'max_depth':[15,17,19,21],'min_samples_split':[4, 5,6,7,8],'n_estimators':[50,100,150,200,250]} #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro') #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer) #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train) #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)

#{'max_depth': 17, 'min_samples_split': 4} for n=100

#

'''
data_test = pd.read_csv('../input/test.csv')

data2 = pd.read_csv('../input/test.csv')

df2= data2.replace({'?': np.nan})
#df2 = df2.drop(columns=['Enrolled', 'Fill' ,'PREV' ,'MLU' ,'Worker Class','Reason','State','Area',"Detailed", 'COB SELF', 'COB MOTHER', 'COB FATHER', 'Teen', 'Live', 'MOVE', 'MOC', 'MIC', 'REG', 'MSA']) #

df2.info()
df2.fillna(df.mode().iloc[0], inplace=True)
df2.head()
df2.columns
for i in range(1, len(df2.columns)):

    print(df2.columns[i]," has ",len(df2.iloc[:,i].unique()))
'''

df columns

['ID', 'Age', 'Worker Class', 'IC', 'OC', 'Schooling', 'Timely Income',

       'Married_Life', 'MIC', 'MOC', 'Cast', 'Hispanic', 'Sex', 'Full/Part',

       'Gain', 'Loss', 'Stock', 'Tax Status', 'State', 'Detailed', 'Summary',

       'Weight', 'MSA', 'REG', 'MOVE', 'Live', 'NOP', 'Teen', 'COB FATHER',

       'COB MOTHER', 'COB SELF', 'Citizen', 'Own/Self', 'Vet_Benefits',

       'Weaks', 'WorkingPeriod']

'''
datatest = pd.get_dummies(df2, columns=mylist)
datatest.head()
datatest=datatest[['ID', 'Age', 'IC', 'OC', 'Timely Income', 'Gain', 'Loss', 'Stock',

       'Weight', 'NOP', 'Own/Self', 'Vet_Benefits', 'Weaks', 'WorkingPeriod',

       'Worker Class_WC1', 'Worker Class_WC2', 'Worker Class_WC3',

       'Worker Class_WC4', 'Worker Class_WC5', 'Schooling_Edu1',

       'Schooling_Edu12', 'Schooling_Edu13', 'Schooling_Edu14',

       'Schooling_Edu15', 'Schooling_Edu16', 'Schooling_Edu2',

       'Schooling_Edu3', 'Schooling_Edu4', 'Schooling_Edu5', 'Schooling_Edu6',

       'Schooling_Edu9', 'Enrolled_Uni1', 'Enrolled_Uni2', 'Married_Life_MS1',

       'Married_Life_MS2', 'Married_Life_MS3', 'Married_Life_MS4', 'MIC_MIC_A',

       'MIC_MIC_C', 'MIC_MIC_D', 'MIC_MIC_F', 'MIC_MIC_G', 'MIC_MIC_H',

       'MIC_MIC_I', 'MIC_MIC_M', 'MIC_MIC_N', 'MIC_MIC_O', 'MIC_MIC_T',

       'MOC_MOC_A', 'MOC_MOC_B', 'MOC_MOC_C', 'MOC_MOC_D', 'MOC_MOC_E',

       'MOC_MOC_F', 'MOC_MOC_G', 'MOC_MOC_H', 'MOC_MOC_K', 'MOC_MOC_L',

       'Cast_TypeA', 'Cast_TypeD', 'Hispanic_HA', 'Hispanic_HD', 'Sex_F',

       'Sex_M','MOC_MOC_G', 'MOC_MOC_H', 'MOC_MOC_I', 'MOC_MOC_J', 'MOC_MOC_K',

       'MOC_MOC_L', 'MOC_MOC_M', 'Cast_TypeA', 'Cast_TypeD', 'Hispanic_HA',

       'Hispanic_HC', 'Hispanic_HD', 'Hispanic_HE', 'Sex_F', 'Sex_M', 'MLU_NO',

       'MLU_YES', 'Reason_JL2', 'Reason_JL4', 'Full/Part_FA', 'Full/Part_FB',

       'Full/Part_FC', 'Full/Part_FD', 'Tax Status_J1', 'Tax Status_J2',

       'Tax Status_J3', 'Tax Status_J4', 'Tax Status_J5', 'Tax Status_J6',

       'Area_S', 'State_s15', 'Detailed_D2', 'Detailed_D3', 'Detailed_D4',

       'Detailed_D5', 'Detailed_D6', 'Detailed_D7', 'Detailed_D8',

       'Summary_sum1', 'Summary_sum2', 'Summary_sum3', 'Summary_sum4',

       'Summary_sum5', 'Summary_sum6', 'MSA_StatusA', 'MSA_StatusL',

       'MSA_StatusO', 'REG_StatusA', 'REG_StatusB', 'MOVE_StatusA',

       'MOVE_StatusB', 'Live_NO', 'Live_YES', 'Teen_B', 'Teen_M',

       'COB FATHER_c24', 'COB FATHER_c36', 'COB MOTHER_c24', 'COB MOTHER_c36',

       'COB SELF_c24', 'COB SELF_c36', 'Citizen_Case1', 'Citizen_Case2',

       'Citizen_Case3']]
prediction = clf.predict(datatest)
zero=0

one=0
for i in range(0,len(prediction)):

  if(prediction[i]==0):

    zero=zero+1

  else:

    one=one+1
print(zero,one)
prediction[5526:5596]
answer = {"ID" : df2["ID"], "Class" : prediction}

ans = pd.DataFrame(answer, columns = ["ID","Class"])

ans.to_csv("submission.csv", index = False)
from IPython.display import HTML 

import pandas as pd 

import numpy as np 

import base64 

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)     

    b64 = base64.b64encode(csv.encode())     

    payload = b64.decode()     

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'    

    html = html.format(payload=payload,title=title,filename=filename)     

    return HTML(html) 

create_download_link(ans)
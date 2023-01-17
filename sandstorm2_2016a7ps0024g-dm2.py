# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
question_mark = ["?"]
train_orig = pd.read_csv("../input/train.csv", na_values = question_mark) #load data frm system
train = train_orig
train.head()

question_mark = ["?"]
test_orig =  pd.read_csv("../input/test.csv",na_values = question_mark) #load data frm system
test = test_orig
test.head()
null_columns = train.columns[train.isnull().any()]
null_columns
train_drop_col = train.drop(['Worker Class', 'Enrolled', 'MIC', 'MOC', 'Hispanic', 'MLU', 'Reason',
       'Area', 'State', 'MSA', 'REG', 'MOVE', 'Live', 'PREV', 'Teen',
       'COB FATHER', 'COB MOTHER', 'COB SELF', 'Fill' ], 1) # Drop Total from domain knowledge
train_drop_col.info()
test_drop_col = test.drop(['Worker Class', 'Enrolled', 'MIC', 'MOC', 'Hispanic', 'MLU', 'Reason',
       'Area', 'State', 'MSA', 'REG', 'MOVE', 'Live', 'PREV', 'Teen',
       'COB FATHER', 'COB MOTHER', 'COB SELF', 'Fill'], 1) # Drop Total from domain knowledge
test_drop_col.info()
#train_no_nonsense=train_drop_col.dropna()
#train_no_nonsense.info()
#test_no_nonsense=test_drop_col.dropna()
#test_no_nonsense.info()
def Gender_method(v):
    if v=='M':
        return 0
    elif v =='F':
        return 1
    
train_drop_col["Sex"] = train_drop_col["Sex"].apply(Gender_method)
train_drop_col["Sex"].value_counts()    
test_drop_col["Sex"] = test_drop_col["Sex"].apply(Gender_method)
test_drop_col["Sex"].value_counts() 
#Schooling
def Schooling_method(v):
     x = (v[3])
     
     return int(x)

train_drop_col["Schooling"] = train_drop_col["Schooling"].apply(Schooling_method)
train_drop_col["Schooling"].value_counts()
test_drop_col["Schooling"] = test_drop_col["Schooling"].apply(Schooling_method)
test_drop_col["Schooling"].value_counts()

def Casting_method(v):
    if(v=="TypeA"):
        return 1
    elif(v=="TypeB"):
        return 2
    elif(v=="TypeC"):
        return 3
    elif(v=="TypeD"):
        return 4
    else:
        return 5
                
        
train_drop_col["Cast"] = train_drop_col["Cast"].apply(Casting_method)
train_drop_col["Cast"].value_counts()    

test_drop_col["Cast"] = test_drop_col["Cast"].apply(Casting_method)
test_drop_col["Cast"].value_counts()    
def q_method(v):
    if(v=="FA"):
        return 1
    elif(v=="FB"):
        return 2
    elif(v=="FC"):
        return 3
    elif(v=="FD"):
        return 4
    elif(v=="FE"):
        return 5
    elif(v=="FF"):
        return 6
    elif(v=="FG"):
        return 7
    else:
        return 8
            
train_drop_col["Full/Part"] = train_drop_col["Full/Part"].apply(q_method)
train_drop_col["Full/Part"].value_counts()    

test_drop_col["Full/Part"] = test_drop_col["Full/Part"].apply(q_method)
test_drop_col["Full/Part"].value_counts() 
def Marry_method(v):
    
    x = (v[2:])
    
    return int(x)
    
train_drop_col["Married_Life"] = train_drop_col["Married_Life"].apply(Marry_method)
train_drop_col["Married_Life"].value_counts()

test_drop_col["Married_Life"] =test_drop_col["Married_Life"].apply(Marry_method)
test_drop_col["Married_Life"].value_counts()
def Tax_method(v):
    x = (v[-1:])
    
    return int(x)
                
train_drop_col["Tax Status"] = train_drop_col["Tax Status"].apply(Tax_method)
train_drop_col["Tax Status"].value_counts()   

test_drop_col["Tax Status"] = test_drop_col["Tax Status"].apply(Tax_method)
test_drop_col["Tax Status"].value_counts()
def Summary_method(v):
    x = (v[-1:])
    
    return int(x)
                
train_drop_col["Summary"] = train_drop_col["Summary"].apply(Summary_method)
train_drop_col["Summary"].value_counts()    

test_drop_col["Summary"] = test_drop_col["Summary"].apply(Summary_method)
test_drop_col["Summary"].value_counts()    
def Detail_method(v):
    x = (v[1:])
    
    return int(x)
                
train_drop_col["Detailed"] = train_drop_col["Detailed"].apply(Detail_method)
train_drop_col["Detailed"].value_counts()

test_drop_col["Detailed"] = test_drop_col["Detailed"].apply(Detail_method)
test_drop_col["Detailed"].value_counts()
def Citizen_method(v):
    x = (v[4])
    
    return int(x)
                
train_drop_col["Citizen"] = train_drop_col["Citizen"].apply(Citizen_method)
train_drop_col["Citizen"].value_counts()

test_drop_col["Citizen"] = test_drop_col["Citizen"].apply(Citizen_method)
test_drop_col["Citizen"].value_counts()

#removing Class so we can predict it
y=train_drop_col['Class']
X=train_drop_col.drop(['Class'],axis=1)
X.head()
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
X_train.head()
from sklearn.ensemble import RandomForestClassifier
score_train_RF = []
score_test_RF = []

for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
            markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed', marker='o'
                     ,markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=14, random_state = 42)
rf.fit(X_train, y_train)
rf.score(X_val,y_val)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)
print(classification_report(y_val, y_pred_RF))
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
rf_temp = RandomForestClassifier(n_estimators = 14) #Initialize the classifier 
parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]} #Dictionary 
scorer = make_scorer(f1_score, average = 'micro') #Initialize the scorer using 
grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer) #Initialize a GridSearchCV 
grid_fit = grid_obj.fit(X_train, y_train) #Fit the gridsearch object with X_train,
best_rf = grid_fit.best_estimator_ #Get the best estimator. For this, check documentation 
print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators = 14, max_depth = 10, min_samples_split = 3)
rf_best.fit(X_train, y_train)
rf_best.score(X_val,y_val)
y_pred_RF_best = rf_best.predict(X_val)
confusion_matrix(y_val, y_pred_RF_best)
print(classification_report(y_val, y_pred_RF_best))
final = rf_best.predict(test_drop_col)
final2 = pd.DataFrame({"ID":test_orig["ID"],"Class":final})
final2.to_csv("Assignment_submission.csv",index = False)
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
create_download_link(final2)
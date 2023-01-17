import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
train_data_orig = pd.read_csv('../input/train.csv', sep=',')

train_data = train_data_orig

test_data_orig = pd.read_csv('../input/test.csv',sep=',')

test_data = test_data_orig
train_data = train_data.dropna()

test_data = test_data.dropna()
train_data = train_data.replace({'?':np.nan})

test_data = test_data.replace({'?':np.nan})
#Removing columns with too many null values

train_data = train_data.drop(['Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','Fill'],axis=1)

test_data = test_data.drop(['Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','Fill'],axis=1)
#Removing categorical columns with too many categories

train_data = train_data.drop(['Schooling','Married_Life','COB FATHER','COB MOTHER','COB SELF','Hispanic','Full/Part','Tax Status','Detailed','Summary'],axis=1)

test_data = test_data.drop(['Schooling','Married_Life','COB FATHER','COB MOTHER','COB SELF','Hispanic','Full/Part','Tax Status','Detailed','Summary'],axis=1)
train_data = train_data.drop(['ID'],axis=1)

test_data = test_data.drop(['ID'],axis=1)
#getting dummies for rest of the categorical columns

train_data = pd.get_dummies(train_data,columns=['Cast','Sex','Citizen'])

test_data = pd.get_dummies(test_data,columns=['Cast','Sex','Citizen'])
import seaborn as sns

f, ax = plt.subplots(figsize=(20, 18))

corr = train_data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
#Weaks is highly correlated to NOP and IC. So drop it

train_data = train_data.drop(['Weaks'],axis=1)

test_data = test_data.drop(['Weaks'],axis=1)
y = train_data['Class']

X = train_data.drop(['Class'],axis=1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
X_test = test_data
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X_train)

X_train = pd.DataFrame(np_scaled)

np_scaled_val = min_max_scaler.transform(X_val)

X_val = pd.DataFrame(np_scaled_val)

np_scaled_test = min_max_scaler.transform(X_test)

X_test = pd.DataFrame(np_scaled_test)

X_train.head()
from sklearn.ensemble import RandomForestClassifier

score_train_RF = []

score_test_RF = []



for i in range(1,18,1):

    rf = RandomForestClassifier(n_estimators=i, class_weight='balanced',random_state = 42)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=14, random_state = 42,class_weight='balanced')

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 14)        #Initialize the classifier object



parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters





grid_obj = GridSearchCV(rf_temp, parameters)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators = 14, max_depth = 10, min_samples_split = 4,class_weight='balanced',random_state=19)

rf_best.fit(X_train, y_train)

rf_best.score(X_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_RF_best = rf_best.predict(X_val)

confusion_matrix(y_val, y_pred_RF_best)
print(classification_report(y_val, y_pred_RF_best))
y_test_RF = rf_best.predict(X_test)
y_test_RF = pd.DataFrame(y_test_RF)
test_data_final = test_data_orig
test_data_final = test_data_final['ID']
without_deleted = pd.concat([test_data_final,y_test_RF],axis=1)

without_deleted.head()
without_deleted.columns = ['ID','Class']

without_deleted.head()
final_df  = without_deleted
final_df.to_csv('RF_final.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(final_df)

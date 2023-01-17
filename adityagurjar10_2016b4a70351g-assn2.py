import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X_tr_orig = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')
X_te_orig = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',') 
X_tr_orig.head()
X_te_orig.head()
X_te_orig.info()
X_te_orig.info()
print(X_tr_orig.duplicated().sum())
print(X_te_orig.duplicated().sum())
print(X_tr_orig.isnull().any().sum())
print(X_te_orig.isnull().any().sum())
X_tr = X_tr_orig.drop(['ID'], axis=1)
X_tr.head()
test_id = X_te_orig['ID']
test_id.head()
X_te = X_te_orig.drop(['ID'], axis=1)
obj_list = list(X_tr.select_dtypes(include=['object']).columns)
obj_list
X_tr[['col2', 'col11', 'col37', 'col44', 'col56']].head()
X_tr['col2'].value_counts()
X_tr['col11'].value_counts()
X_tr['col37'].value_counts()
X_tr['col44'].value_counts()
X_tr['col56'].value_counts()
onehotlist = ['col11', 'col37', 'col44']
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoded_X_tr = X_tr.copy()
encoded_X_te = X_te.copy()

encoded_X_tr = pd.get_dummies(encoded_X_tr, columns=onehotlist, prefix = onehotlist)
encoded_X_te = pd.get_dummies(encoded_X_te, columns=onehotlist, prefix = onehotlist)
df_tr = encoded_X_tr.copy()
df_te = encoded_X_te.copy()
df_tr[['col2', 'col56']].head()
df_tr['col2'].replace({'Silver' : 25, 'Gold': 50, 'Platinum': 75, 'Diamond': 100}, inplace=True)
df_te['col2'].replace({'Silver' : 25, 'Gold': 50, 'Platinum': 75, 'Diamond': 100}, inplace=True)
df_tr['col2'].head()
df_tr['col56'].replace({'Low' : 25, 'Medium': 50, 'High': 75}, inplace=True)
df_te['col56'].replace({'Low' : 25, 'Medium': 50, 'High': 75}, inplace=True)
df_tr['col56'].head()
df_tr.info()
df_te.info()
y_tr = df_tr['Class']
y_tr.value_counts()
df_tr = df_tr.drop(['Class'], axis=1)
np.random.seed(42)
# Creating a new feature origin
df_tr['origin'] = 0
df_te['origin'] = 1
## taking a random sample from test and train data
tr = df_tr.sample(200, random_state=4)
te = df_te.sample(200, random_state=4)
## combining these random samples to form a new dataset
comb = tr.append(te)
y = comb['origin']
comb.drop('origin',axis=1,inplace=True)
df_tr.drop('origin', axis=1, inplace=True)
df_te.drop('origin', axis=1, inplace=True)
## Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
model = RandomForestClassifier(n_estimators = 50, max_depth = 5,min_samples_leaf = 5)
drop_list = []
for i in comb.columns:
    score = cross_val_score(model,pd.DataFrame(comb[i]),y,cv=2,scoring='roc_auc')
    if (np.mean(score) > 0.8):
        drop_list.append(i)
        print(i,np.mean(score))
drop_list
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_tr)
X_N = pd.DataFrame(np_scaled)
X_N.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_N, y_tr, test_size=0.10, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
score_train_RF = []
score_test_RF = []

for i in range(5,30,1):
    rf = RandomForestClassifier(n_estimators = 100, max_depth=i)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_test,y_test)
    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(5,30,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,30,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=100, max_depth = 21, random_state=42)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
rf = RandomForestClassifier(n_estimators=100, max_depth = 7, random_state=44)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
rf = RandomForestClassifier(n_estimators=100, max_depth = 15, random_state=2)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
rf = RandomForestClassifier(n_estimators=100, max_depth = 16, random_state=42)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
rf = RandomForestClassifier(n_estimators=100, max_depth = 24, random_state=42)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
y_pred_RF = rf.predict(X_test)
print(classification_report(y_test, y_pred_RF))
#df_tr.drop(columns = drop_list, inplace=True)
#df_te.drop(columns = drop_list, inplace=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

rf_temp = RandomForestClassifier(n_estimators=100)        #Initialize the classifier object

parameters = {
    'max_features': ['log2', 'auto', 'sqrt'],
    'max_depth':[6, 7, 8, 9, 10],
    'min_samples_split':[2, 3, 4, 5, 6],
    }    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(df_tr, y_tr)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
rf10 = RandomForestClassifier( n_estimators=100, max_depth = 6, min_samples_split = 2, max_features = 'log2')
    
rf10.fit(df_tr, y_tr)

preds_rf1 = rf10.predict(df_te)
unique, counts = np.unique(preds_rf1, return_counts=True)
dict(zip(unique, counts))
preds3 = pd.DataFrame({'ID': test_id, 'Class': preds_rf1})
preds3.to_csv('preds_rf_11.csv', index=False)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(preds3)

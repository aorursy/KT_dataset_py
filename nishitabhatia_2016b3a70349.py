import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')
data = data_orig
data.head()
data.duplicated().sum()
one_hot=['col11','col37','col44']
data['col2'].unique()
#using replace method
data['col2'].replace({
    'Silver':0,
    'Gold':1,
    'Platinum':2,
    'Diamond':3,
    },inplace=True)

data.head()
data['col56'].unique()
#using replace method
data['col56'].replace({
    'Low':0,
    'Medium':1,
    'High':2,
    },inplace=True)

data.head()
data = data.drop(['ID'], axis = 1)
data.head()
y=data['Class']
X=data.drop(['Class'],axis=1)
X.head()
for i in one_hot:
    X= pd.get_dummies(X, columns=[i],prefix=[i])
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

score_train_RF = []
score_test_RF = []

for i in range(5,20,1):
    rf = RandomForestClassifier(n_estimators = 2000, max_depth=i, class_weight='balanced')
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_test,y_test)
    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))
train_score,=plt.plot(range(5,20,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,20,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Score vs. Max_depth')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
rf_temp = RandomForestClassifier(n_estimators = 2000,class_weight='balanced')        #Initialize the classifier object

parameters = {'max_depth':[12],'min_samples_split':[2,3,4,5]}    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators=2000,max_depth=12,min_samples_split=3,class_weight='balanced',random_state=50)
rf_best.fit(X_train, y_train)
rf_best.score(X_test,y_test)
test_orig = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')
col_id=test_orig['ID']
test=test_orig.drop(['ID'],axis=1)
one_hot=['col11','col37','col44']
#using replace method
test['col2'].replace({
    'Silver':0,
    'Gold':1,
    'Platinum':2,
    'Diamond':3,
    },inplace=True)
#using replace method
test['col56'].replace({
    'Low':0,
    'Medium':1,
    'High':2,
    },inplace=True)

test.head()
for i in one_hot:
    test=pd.get_dummies(test, columns=[i],prefix=[i])
test.head()
prediction=rf_best.predict(test)
prediction
ID=test_orig['ID']
sol1=pd.DataFrame(data=prediction)
sol2=pd.DataFrame(data=ID)
sol=pd.concat([sol2,sol1],axis=1)
sol.rename(columns = {0:'Class'}, inplace = True) 
sol
sol.to_csv('rf10.csv',index=False)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(sol)
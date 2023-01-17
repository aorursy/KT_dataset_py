import numpy as np
import pandas as pd
df1 = pd.read_csv("../input/opcode_frequency_benign.csv", sep=',')
df_1 = df1

df2 = pd.read_csv("../input/opcode_frequency_malware.csv", sep=',')
df_2 = df2

df_test = pd.read_csv("../input/Test_data.csv",sep=',')
x_t = df_test
x_t.info()
df1['Class']=0
df2['Class']=1
df_2.head()
data = df_1.append(df_2)
data.head()
data.tail()
data = data.drop(['FileName'], axis = 1)
y=data['Class']
x=data.drop(['Class'],axis=1)
x_t = x_t.drop(['FileName'], axis = 1)
x.head()
x_t.info()
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.19, random_state=42)
x_t = x_t.drop(['Unnamed: 1809'], axis = 1)
x_t.head()
x_t.info()
from sklearn import preprocessing
min_max_sc = preprocessing.MinMaxScaler()
np_sc = min_max_sc.fit_transform(x_tr)
x_tr = pd.DataFrame(np_sc)
np_sc_val = min_max_sc.transform(x_val)
x_val = pd.DataFrame(np_sc_val)
x_tr.head()
from sklearn import preprocessing
min_max_sc2 = preprocessing.MinMaxScaler()
np_sc2 = min_max_sc2.fit_transform(x_t)
x_t = pd.DataFrame(np_sc2)
x_t.head()
from sklearn.ensemble import RandomForestClassifier
scr_tr_RF = []
scr_test_RF = []

for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42)
    rf.fit(x_tr, y_tr)
    sc_tr = rf.score(x_tr,y_tr)
    scr_tr_RF.append(sc_tr)
    sc_t= rf.score(x_val,y_val)
    scr_test_RF.append(sc_t)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
tr_score,=plt.plot(range(1,18,1),scr_tr_RF,color='blue', marker='o',
         markerfacecolor='red', markersize=5)
test_score,=plt.plot(range(1,18,1),scr_test_RF,color='red',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [tr_score,test_score],["Train Score","Test Score"])
plt.title(' Score vs Trees')
plt.xlabel('Trees')
plt.ylabel('Score')
rfc = RandomForestClassifier(n_estimators=9, random_state = 42)
rfc.fit(x_tr, y_tr)
rfc.score(x_val,y_val)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_rf = rf.predict(x_val)
confusion_matrix(y_val, y_pred_rf)
print(classification_report(y_val, y_pred_rf))
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
rf_tmp = RandomForestClassifier(n_estimators = 9)
params = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]}
scrr = make_scorer(f1_score, average = 'micro')         
grid_object = GridSearchCV(rf_tmp, params, scoring=scrr)        
grid_fit = grid_object.fit(x_tr, y_tr)       
bst_rf = grid_fit.best_estimator_         
print(grid_fit.best_params_)
rf_bst = RandomForestClassifier(n_estimators = 9, max_depth = 10, min_samples_split = 2)
rf_bst.fit(x_tr, y_tr)
rf_bst.score(x_val,y_val)
y_pred_rf_final = rf_bst.predict(x_val)
confusion_matrix(y_val, y_pred_rf_final)
print(classification_report(y_val, y_pred_rf_final))
y_pred_test_rf = rf_bst.predict(x_t)
z = y_pred_test_rf.tolist()
sub = pd.DataFrame(z)
result_rf = pd.concat([df_2['FileName'], sub], axis=1).reindex()
result_rf = result_rf.rename(columns={0: 'Class'})
result_rf.head(10)
sub.head(10)
result_rf.to_csv('submission_rf.csv', index = False,  float_format='%.f')
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html='<a download="{filename}"href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(result_rf)








import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')
data1 = data_orig
# df0 = data1[data1.Class == 0].sample(150,random_state=650)
# df1 = data1[data1.Class == 1].sample(46,random_state=650)
# df2 = data1[data1.Class == 2].sample(150,random_state=650)
# df3 = data1[data1.Class == 3].sample(150,random_state=650)
# frames = [df0, df1, df2, df3]
# data1 = pd.concat(frames)
# data3=data1
data2 = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')
data1 = data1.drop(["Class"], 1)
data = pd.concat([data1, data2])
data.head()

col = data.isnull().sum()
col = col[col > 0]
col
data.duplicated().sum()
data.info()
data = data.drop(['ID'],1)
# data=data.drop(["col2", "col11", "col37", "col44", "col56"],1)  #row1
# data=data.drop(["col26", "col32", "col50"],1)   #row2
data=data.drop(["col0", "col13", "col25", "col43", "col49", "col63"],1)  #row3
data=data.drop(["col9", "col15", "col27", "col28", "col31", "col34", "col36", "col40",],1)  #row4
data = pd.get_dummies(data, columns=["col2", "col11", "col37", "col44", "col56"])

# corr_matrix = data3.corr()
# pd.options.display.max_rows = None
# corr_matrix["Class"].sort_values()

# to_drop = []
# temp = corr_matrix["Class"]
# for i in temp.index:
#     if(temp[i]<0.2 and temp[i]>0.06 ):
#         to_drop.append(i)
# to_drop

# data = data.drop(data[to_drop], axis=1)
data.info()
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)] 
data.drop(to_drop, axis=1, inplace=True)
to_drop
data.info()
y=data_orig['Class']
X = data.head(700)
from sklearn import preprocessing

scaler=preprocessing.StandardScaler()
np_scaled=scaler.fit(X).transform(X)
np=scaler.fit(data).transform(data)
X_N=pd.DataFrame(np_scaled,columns=X.columns)
data=pd.DataFrame(np,columns=data.columns)
X_N.tail()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_N, y, test_size=0.20, random_state=201)
from sklearn.ensemble import RandomForestClassifier
score_train_RF = []
score_test_RF = []

for i in range(1,26,1):
    rf = RandomForestClassifier(n_estimators = 100, max_depth=i,random_state = 201)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_test,y_test)
    score_test_RF.append(sc_test)
plt.figure(figsize=(10,10))
train_score,=plt.plot(range(1,26,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,26,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=100, max_depth = 8, random_state = 201)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_RF = rf.predict(X_test)
confusion_matrix(y_test, y_pred_RF)
print(classification_report(y_test, y_pred_RF))
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

rf_temp = RandomForestClassifier(n_estimators = 100,random_state=201)        #Initialize the classifier object

parameters = {'max_depth':[8],'min_samples_split':[2, 3, 4, 5,6]}    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators=100, max_depth = 8,min_samples_split= 2, random_state = 201)
rf_best.fit(X_train, y_train)
rf_best.score(X_test,y_test)
y_pred_RF_best = rf_best.predict(X_test)
confusion_matrix(y_test, y_pred_RF_best)
print(classification_report(y_test, y_pred_RF_best))
testset = data[700:]
abcd = rf_best.predict(testset)
abcd
final = pd.DataFrame({"ID":data2["ID"],"Class":abcd})
final.to_csv("finalsub.csv", index=False)
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(final)

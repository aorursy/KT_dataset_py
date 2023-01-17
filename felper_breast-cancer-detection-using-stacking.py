import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
#Import the data into a dataframe
df = pd.read_csv('../input/data.csv', encoding = 'ISO-8859-1')
#Shape of the dataframe
df.shape
#Names of the columns
df.columns
#Types of the elements of the dataframe
df.dtypes
df['Unnamed: 32'].head(15)
df['Unnamed: 32'].isna().sum()
df.isnull().any()
df = df.set_index('id')
type(df['diagnosis'].iloc[2])
le = preprocessing.LabelEncoder()
le.fit(df['diagnosis'])
df['diagnosis'] = le.transform(df['diagnosis'])
X = df[df.columns[1:-1]]
y = df['diagnosis']
# Benign, malignant
sum(y == 0) , sum(y == 1)
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y , test_size = 0.3)
X_tun, X_test, y_tun, y_test = train_test_split(X_temp, y_temp,test_size=0.5)
X_tun.shape , y_tun.shape
import seaborn as sns

corr_train = X_train.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr_train,annot=True ,square=True);
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train,y_train)
knn_tun = knn.predict(X_tun)
confusion_matrix(y_tun,knn_tun)
print(classification_report(y_tun, knn_tun))
accuracy_score(y_tun, knn_tun)
nn = MLPClassifier(solver='lbfgs', activation = 'tanh', alpha = 1,
                         hidden_layer_sizes = [64, 64,32])
nn.fit(X_train, y_train)
nn_tun = nn.predict(X_tun)
confusion_matrix(y_tun,nn_tun)
print(classification_report(y_tun, nn_tun))
accuracy_score(y_tun, nn_tun)
gbt = GradientBoostingClassifier()
gbt.fit(X_train, y_train)
gbt_tun = gbt.predict(X_tun)
confusion_matrix(y_tun,gbt_tun)
print(classification_report(y_tun, gbt_tun))
accuracy_score(y_tun, gbt_tun)
rf = RandomForestClassifier(max_depth = 9 , min_samples_leaf = 5 )
rf.fit(X_train, y_train)
rf_tun = rf.predict(X_tun)
confusion_matrix(y_tun,rf_tun)
print(classification_report(y_tun, rf_tun))
accuracy_score(y_tun, rf_tun)
feat_imp_rf = pd.DataFrame([X.columns,rf.feature_importances_]).T
feat_imp_rf.sort_values(by=1,ascending=False)
feat_imp_gbt = pd.DataFrame([X.columns,gbt.feature_importances_]).T
feat_imp_gbt.sort_values(by=1,ascending=False)
L = list(feat_imp_rf.sort_values(by=1,ascending=False).iloc[0:6,0].values)
pd.DataFrame(np.array([rf_tun,knn_tun,nn_tun,gbt_tun])).T.corr()
knn_pred = knn.predict(X_test)
nn_pred = nn.predict(X_test)
gbt_pred = gbt.predict(X_test)
rf_pred = rf.predict(X_test)
predictions = np.hstack([knn_pred.reshape(-1,1) , 
                        nn_pred.reshape(-1,1) , 
                        gbt_pred.reshape(-1,1) ,
                        rf_pred.reshape(-1,1)])
tuning = np.hstack([knn_tun.reshape(-1,1) , 
                        nn_tun.reshape(-1,1) , 
                        gbt_tun.reshape(-1,1) ,
                        rf_tun.reshape(-1,1)])
df_voting = pd.DataFrame(predictions)
df_tuning = pd.DataFrame(tuning)
pred_votes = df_voting.mode(axis=1)
tun_votes = df_tuning.mode(axis=1)
tun_votes.head()
#Report for the tuning data
print(classification_report(y_tun, tun_votes[0].values))
#Report for the test data
print(classification_report(y_test, pred_votes[0].values))
confusion_matrix(y_tun,tun_votes[0].values)
confusion_matrix(y_test,pred_votes[0].values)
#Tuning data score
accuracy_score(y_tun,tun_votes[0].values)
#Test data score
accuracy_score(y_test,pred_votes[0].values)
W_train = np.hstack([rf.predict(X_train).reshape(-1,1),knn.predict(X_train).reshape(-1,1),gbt.predict(X_train).reshape(-1,1),nn.predict(X_train).reshape(-1,1),X_train[L]])
W_tun = np.hstack([rf.predict(X_tun).reshape(-1,1),knn.predict(X_tun).reshape(-1,1),gbt.predict(X_tun).reshape(-1,1),nn.predict(X_tun).reshape(-1,1),X_tun[L]])
W_test = np.hstack([rf.predict(X_test).reshape(-1,1),knn.predict(X_test).reshape(-1,1),gbt.predict(X_test).reshape(-1,1),nn.predict(X_test).reshape(-1,1),X_test[L]])
from sklearn.linear_model import LogisticRegression

lr_stack = LogisticRegression()
lr_stack.fit(W_train,y_train)
stack_tun = lr_stack.predict(W_tun)
stack_test = lr_stack.predict(W_test)
#Tuning data
print(classification_report(y_tun, stack_tun))
confusion_matrix(y_tun,stack_tun)
accuracy_score(y_tun,stack_tun)
#Test data
print(classification_report(y_test, stack_test))
confusion_matrix(y_test,stack_test)
accuracy_score(y_test,stack_test)

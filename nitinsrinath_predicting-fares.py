import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def get_datetime(date):
    
    temp=date.split('/')
    temp.extend(temp[2].split(' '))
    temp.extend(temp[4].split(':'))
    temp.remove(temp[2])
    temp.remove(temp[3])
    temp=[int(temp[i]) for i in [2, 0, 1, 3, 4]]
    tempdt=dt.datetime(temp[0], temp[1], temp[2], temp[3], temp[4])
    return temp, tempdt
spn=pd.read_csv('Spain Rail Transportation.csv')
spn=spn.dropna()

for i in spn.index:
    #print(spn.start_date[i], end=' ')
    strt, strtdt=get_datetime(spn.start_date[i])
    end, enddt=get_datetime(spn.end_date[i])
    spn.at[i, 'travel_time']=divmod((enddt-strtdt).seconds, 60)[0]
    spn.at[i, 'year']=strt[0]
    spn.at[i, 'month']=strt[1]
    spn.at[i, 'day']=strt[2]
spn=spn.sort_values(['year', 'month', 'day'])
spn=spn[['origin', 'destination', 'train_type', 'price', 'train_class', 'fare', 'travel_time' ,'month', 'day']]
traveltime=pd.DataFrame(spn.groupby(['origin', 'destination'])['travel_time'].mean()).reset_index()
traveltime['travel_time']=traveltime.travel_time.round(0)
g=nx.Graph()

for i in traveltime.index:
    g.add_edge(traveltime.origin[i], traveltime.destination[i], weight=traveltime.travel_time[i])

plt.figure(figsize=(10,10))
pos = nx.spring_layout(g, k=500)
labels = nx.get_edge_attributes(g,'weight')
nx.draw_networkx_nodes(g, pos, node_size=1000, node_color='orange')
nx.draw_networkx_labels(g, pos, font_size=10, font_family='sans-serif')
nx.draw_networkx_edges(g, pos, width=6)
nx.draw_networkx_edge_labels(g,pos,edge_labels=labels)
spn['week']=np.ceil((spn['month']*30+spn['day']-30)/7)
weeklyprices=pd.DataFrame(spn.groupby(['week', 'train_class'])['price'].mean()).reset_index()
weeklyTT=pd.DataFrame(spn.groupby(['week', 'train_class'])['travel_time'].mean()).reset_index()
weeklytickets=pd.DataFrame(spn.groupby(['week', 'train_class'])['price'].size()).reset_index()
sns.set(rc={'figure.figsize':(10, 8)})
sns.set_style("whitegrid")
sns.lineplot(x=weeklyprices.week, y=weeklyprices.price, hue=weeklyprices.train_class)
sns.set(rc={'figure.figsize':(10, 8)})
sns.set_style("whitegrid")
sns.lineplot(x=weeklyTT.week, y=weeklyTT.travel_time, hue=weeklyTT.train_class)
sns.set(rc={'figure.figsize':(10, 8)})
sns.set_style("whitegrid")
sns.lineplot(x=weeklytickets.week, y=weeklytickets.price, hue=weeklytickets.train_class)
plt.ylabel('number of tickets')
label_encoder = LabelEncoder()
for i in ['origin', 'destination', 'train_type', 'train_class', 'fare']:
    print(i, end=" ")
    spn[i+'enc']=label_encoder.fit_transform(spn[i])
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(le_name_mapping)
spn.corr()
spn.hist(figsize=(20,20))
X=spn[['originenc', 'destinationenc', 'train_typeenc', 'train_classenc', 'price', 'travel_time']]
Y=spn['fare']
chi_scores=chi2(X,Y)
p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'degree': [2,3,4], 'kernel':['poly']}
 ]

SVM = SVC(max_iter=1000)
clf = GridSearchCV(SVM, param_grid)
clf.fit(X_train, Y_train)
clf.best_params_
SVM = SVC(C=10, gamma=0.001, kernel='rbf', max_iter=1000)

SVM.fit(X_train, Y_train)
y_pred=SVM.predict(X_test)
accuracy_score(Y_test, y_pred)
DTC= DecisionTreeClassifier()
DTC.fit(X_train, Y_train)
y_pred=DTC.predict(X_test)
accuracy_score(Y_test, y_pred)
RFtest=pd.DataFrame()
for i in range(1, 16):
    RFC = RandomForestClassifier(max_depth=i, random_state=0)
    RFC.fit(X_train, Y_train)
    y_train_pred=RFC.predict(X_train)
    y_test_pred=RFC.predict(X_test)
    RFtest.at[i, 'TrainAcc']=accuracy_score(Y_train, y_train_pred)
    RFtest.at[i, 'TestAcc']=accuracy_score(Y_test, y_test_pred)
plt.plot(RFtest.index, RFtest.TrainAcc, label='Train Acc')
plt.plot(RFtest.index, RFtest.TestAcc, label='Test Acc')
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.grid()
RFC = RandomForestClassifier(max_depth=10, random_state=0)

RFC.fit(X_train, Y_train)
y_pred=RFC.predict(X_test)
accuracy_score(Y_test, y_pred)
feature_selection=pd.DataFrame()
for i in X.columns:
    use_x_train=X_train[i].values[:, np.newaxis]
    use_x_test=X_test[i].values[:, np.newaxis]
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)
feature_selection=pd.DataFrame()
use_features1=[i for i in X.columns if i!='price']
for i in use_features1:
    use_x_train=pd.DataFrame(X_train[['price', i]])
    use_x_test=pd.DataFrame(X_test[['price', i]])
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)  
RFC = RandomForestClassifier(max_depth=10, random_state=0)
X=X[['price', 'travel_time']]
cross_val_score(RFC, X, Y, cv=10)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40, random_state=42)
RFC.fit(X_train, Y_train)
y_train_pred=RFC.predict(X_train)
y_test_pred=RFC.predict(X_test)
print(accuracy_score(Y_train, y_train_pred))
print(accuracy_score(Y_test, y_test_pred))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
RFC.fit(X_train, Y_train)
y_pred=RFC.predict(X_test)
confusion_matrix(Y_test, y_pred, labels=['Flexible', 'Promo','Promo +', 'Adulto ida', 'Individual-Flexible'])
print(classification_report(Y_test, y_pred, labels=['Flexible', 'Promo','Promo +', 'Adulto ida', 'Individual-Flexible']))
sample=X.sample(frac=1).head(10)
sample['prediction']=RFC.predict(sample)
sample
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(spn.price, spn.travel_time, spn.fareenc, c=spn.fareenc)
ax.set_xlabel('Price')
ax.set_ylabel('Travel Time')
ax.set_zlabel('Fare Enc')
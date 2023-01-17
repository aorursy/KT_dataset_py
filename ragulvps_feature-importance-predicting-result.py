import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply

from sklearn.preprocessing import StandardScaler 
#from sklearn

print('Import Complete')
train_data = pd.read_csv('/kaggle/input/airline-passenger-satisfaction/train.csv',index_col=0)
test_data =pd.read_csv('/kaggle/input/airline-passenger-satisfaction/test.csv',index_col=0)
print('Shape of train Data',train_data.shape)
print('Shape of test Data',test_data.shape)
train_data.head()
train_data.describe()
train_data.info()
train_data['Arrival Delay in Minutes'].isnull().sum()
train_data.dropna(axis=0,inplace=True)
train_data.isnull().sum().any()
train_data.drop('id',axis=1).head()
test_data.info()
test_data['Arrival Delay in Minutes'].isnull().sum()
test_data.dropna(axis=0,inplace=True)
test_data.isnull().sum().any()
train_data.drop('id',axis=1, inplace=True)
test_data.drop('id',axis=1,inplace=True)
sns.countplot(train_data['satisfaction'])
plt.xticks()
train_data['satisfaction'].unique()
cols = [ 'Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service','Cleanliness']
#len(cols)
fig, ax = plt.subplots(3, 5, figsize=(30, 20))
for variable, subplot in zip(cols, ax.flatten()):
    sns.countplot(train_data[variable],hue=train_data['satisfaction'], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
cat_col =['Gender','Customer Type','Type of Travel','Class']

fig, ax = plt.subplots(1,4, figsize=(25, 7))
for variable, subplot in zip(cat_col, ax.flatten()):
    sns.countplot(train_data[variable],hue=train_data['satisfaction'], ax=subplot)
 

sns.set_palette('Set1')
plt.figure(figsize=(7,7))
sns.scatterplot(x=train_data['Flight Distance'],
                y=train_data['Departure Delay in Minutes'],
                hue=train_data['satisfaction'])
plt.figure(figsize=(7,7))
sns.scatterplot(x=train_data['Flight Distance'],
                y=train_data['Arrival Delay in Minutes'],
                hue=train_data['satisfaction'])
plt.figure(figsize=(7,7))
sns.set_palette('Set2')
sns.catplot(x='Type of Travel',y='Departure/Arrival time convenient',
            row ='Gender',hue='satisfaction',col='Class',
            data=train_data, kind= 'bar')
plt.figure(figsize=(7,7))
sns.catplot(y='Departure/Arrival time convenient',col='Type of Travel',x ='Customer Type',
            hue='satisfaction',row='Class', data=train_data, kind= 'bar',palette='coolwarm')
#sns.pairplot(train_data)
cat_col=train_data.select_dtypes('object').columns
#cat_col
for i in cat_col:
    print('Unique values of', str(i),' are:',train_data[i].unique())
from sklearn.preprocessing import LabelEncoder
 
label = LabelEncoder()
labeled_train_data= train_data.copy()
labeled_test_data= test_data.copy()
for i in cat_col:
    labeled_train_data[i]=label.fit_transform(labeled_train_data[i])
    labeled_test_data[i]=label.fit_transform(labeled_test_data[i])
    
#train_data.head()
labeled_train_data.describe()
#labeled_test_data.describe()
x_train = labeled_train_data.drop('satisfaction',axis=1)
y_train = labeled_train_data['satisfaction']
x_test = labeled_test_data.drop('satisfaction',axis=1)
y_test = labeled_test_data['satisfaction']
num_col = ['Age','Departure Delay in Minutes','Arrival Delay in Minutes','Flight Distance'] 
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_train[num_col]=scaler.fit_transform(x_train[num_col])
x_train.head()
x_test[num_col]=scaler.transform(x_test[num_col])
x_test.head()
from sklearn.ensemble import RandomForestClassifier 

clf = RandomForestClassifier(max_depth=35,min_samples_leaf= 1,min_samples_split= 2,n_estimators=1400,
         random_state= 42).fit(x_train,y_train)

y_pred = clf.predict(x_test)
#Accuracy Score
from sklearn.metrics import f1_score,confusion_matrix,plot_confusion_matrix,accuracy_score

print("Accuracy Score is :",accuracy_score(y_pred,y_test))
confusion_matrix(y_pred,y_test)
round(f1_score(y_pred,y_test),4)
from sklearn.metrics import roc_auc_score,classification_report

round(roc_auc_score(y_test,y_pred),3)
print(classification_report(y_test,y_pred))
sns.barplot(y = clf.feature_importances_,x = x_train.columns.values)
plt.xticks(rotation=90)
plt.title('Feature Importance Plot')

importances = clf.feature_importances_
std = np.std([feature.feature_importances_ for feature in clf.estimators_],
axis=0)
indices = np.argsort(importances)[::-1]
indices
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]),'-',x_train.columns[indices[f]])

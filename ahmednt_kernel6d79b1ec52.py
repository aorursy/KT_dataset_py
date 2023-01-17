import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier,BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm 
np.set_printoptions(precision=4)
%config InlineBackend.figure_format = 'retina'
%matplotlib inline
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.5)
train_data = pd.read_csv('train.csv')
test_data= pd.read_csv('test.csv')
train_data.head()
train_data.shape
test_data.head()

test_data.shape
train_data.columns
test_data.columns
#save Survived in y_train
train_y = train_data['Survived']
survived = train_data[train_data['Survived'] == 1]
not_survived = train_data[train_data['Survived'] == 0]

print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train_data)*100.0))
print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train_data)*100.0))
print ("Total: %i"%len(train_data))
# Dropping the Survived from the train data
train_data.drop('Survived',axis=1, inplace= True)
# concat the train data with test data to do cleaing
concat_df= pd.concat([train_data,test_data], axis=0, sort=True)
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 6))

sns.heatmap(concat_df.isnull(), yticklabels= False, cbar=False, cmap='viridis')
ax.set_title('check the missing value ')

total =concat_df.isnull().sum()
missing_data = pd.concat([total], axis=1, keys=['Total'])
missing_data.head(11)
concat_df.describe()
# ml search manually (make a guse [ is about know the realtionship ])
concat_df['Age']=concat_df['Age'].fillna(29.881138)
concat_df['Cabin'].unique()
concat_df['Embarked'].value_counts()
concat_df['Embarked']=concat_df['Embarked'].fillna('S')
# If there originally was a value for Cabin -- put 1
# If the value is missing/null -- put 0 
# x (the colum i call it )
concat_df['Cabin']= concat_df['Cabin'].apply(lambda x :0 if pd.isnull(x)else 1)
concat_df['Cabin'].unique()
concat_df['Fare']=concat_df['Fare'].fillna(33.295479)
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (18, 6))

sns.heatmap(concat_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
ax.set_title('check the missing value ')
concat_df = pd.get_dummies(concat_df,drop_first=True)
concat_df.head()
concat_df.corr()
#mask = np.zeros_like(concat_df, dtype=np.bool) 
#mask[np.triu_indices_from(mask)] = True
#sns.set(style="white")
#fig=plt.figure(figsize=(10,10))
#ax = fig.gca()
#cmap = sns.diverging_palette(220, 10, as_cmap=True)
#sns.heatmap(concat_df, annot=True,ax=ax,mask=mask,vmax=.3, center=0,cmap=cmap,
          #  square=True, linewidths=.5, cbar_kws={"shrink": .5})
#ax.set_title('The correlation between The Features ')
# Separating dataframe into train set
df_train = concat_df.iloc[0:891 , : ]
df_train.shape
df_train['Title']=df_train['Name'].map(lambda x: substrings_in_string(x, title_list))

#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
df_train['Title']=df_train.apply(replace_titles, axis=1)
# Separating dataframe into test set    

df_test =concat_df.iloc[891: , : ]
df_test.shape
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_ss = ss.fit_transform(df_train)
X_test_ss = ss.fit_transform(df_test)

train_y .value_counts(normalize=True)
#randomF = RandomForestClassifier(max_depth=50)
#randomF.fit(df_train, train_y)
#print('Train score :',randomF.score(df_train,train_y))
#pred_rf=randomF.predict(df_test)
#pred_rf
#knn_classifier = KNeighborsClassifier(n_neighbors=5)
#knn_classifier.fit(X_train, y_train)
#print(knn_classifier.score(X_train, y_train))
#print(knn_classifier.score(X_test, y_test))
#et = ExtraTreesClassifier(n_estimators=100)
#et.fit(df_train,train_y)
#print('Train score :',et.score(df_train,train_y))
#df_train.columns
loc=LogisticRegression()
loc.fit(X_train_ss,train_y )
loc_prdict=loc.predict(X_test_ss)
print(loc.score(X_train_ss,train_y ))
kc = KNeighborsClassifier(n_neighbors=500)  
kc.fit(X_train_ss,train_y )
kn_prdict=kc.predict(X_test_ss)
print(kc.score(X_train_ss,train_y ))
rfc=RandomForestClassifier()
rfc.fit(X_train_ss,train_y)
rfc_prdict=rfc.predict(X_test_ss)
print(rfc.score(X_train_ss,train_y))
dtc = DecisionTreeClassifier()
dtc.fit(X_train_ss,train_y)
dtc_prdict=dtc.predict(X_test_ss)
print(dtc.score(X_train_ss,train_y))
Et= ExtraTreesClassifier(n_estimators=100)
Et.fit(X_train_ss,train_y)
Et_prdict=Et.predict(X_test_ss)
print(Et.score(X_train_ss,train_y))
bag = BaggingClassifier()
bag.fit(X_train_ss,train_y)
bag_prdict=bag.predict(X_test_ss)
print(bag.score(X_train_ss,train_y))
Ad= model = AdaBoostClassifier(n_estimators=2)
Ad.fit(X_train_ss,train_y)
Ad_prdict=bag.predict(X_test_ss)
print(Ad.score(X_train_ss,train_y))
svm = svm.SVC(kernel='poly')
svm.fit(X_train_ss,train_y)
svm_prdict=svm.predict(X_test_ss)

print(svm.score(X_train_ss,train_y))

from sklearn import svm
svm_l = svm.SVC(kernel='linear', C=20)
svm_l.fit(X_train_ss,train_y)
svm_l_prdict=svm_l.predict(X_test_ss)

print('Train : ', svm_l.score(X_train_ss,train_y))

from sklearn import svm
svm_rbf = svm.SVC(kernel='rbf',C=90,gamma=30)
svm_rbf.fit(X_train_ss,train_y)
svm_rbf_prdict=svm_rbf.predict(X_test_ss)

print('Train : ',svm_rbf.score(X_train_ss,train_y))

Boosting = GradientBoostingClassifier(random_state=30)
Boosting.fit(X_train_ss,train_y)
y = Boosting.predict(X_test_ss)
#accuracy = cross_val_score(Boosting, X_train, y_train, cv=5).mean()
print('Train : ',Boosting.score(X_train_ss,train_y))

y_test_sub=loc.predict(X_test_ss)
Sub1 = [x for x in range (892,1310)]
Submission1 = {'PassengerId':Sub1,
               'Survived':y_test_sub}
df_submission = pd.DataFrame(Submission1)
df_submission.to_csv('loc_submission.csv',index=False)
len(y_test_sub),len(Sub1)
df_submission.head()
#train_2 = pd.read_csv('rfc_submission3.csv')
#train_2[train_2['PassengerId']==13]


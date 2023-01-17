!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887
!apt update && apt install -y libsm6 libxext6
%load_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics
import seaborn as sns
sns.set_style('whitegrid')
#path = 'titanic/'
!ls ../input
df = pd.read_csv('../input/titanic/train.csv')

df.head()
f,ax=plt.subplots(1,2, figsize=(18,8))
df[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Embarked')
sns.countplot('Embarked',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Sex:Survived vs Embarked')
plt.show()
df.Survived.value_counts().plot(kind='bar',legend=True)
df.Sex.value_counts().plot(kind='bar')
_=df.Pclass.value_counts().plot(kind='bar')
f,ax=plt.subplots(1,2, figsize=(18,5))
df[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
df.head()
train_cats(df)
df,y,nas=proc_df(df, 'Survived')
df.head()
m=RandomForestClassifier(n_jobs=-1)
m.fit(df, y)
m.score(df,y)
def split_vals(a, n): return a[:n].copy(), a[n:].copy()

n_valid = 209
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
def rmse(x, y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res=[rmse(m.predict(X_train),y_train), rmse(m.predict(X_valid), y_valid),
         m.score(X_train, y_train), m.score(X_valid, y_valid)]
    
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestClassifier(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, max_depth=3, bootstrap=False)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs=-1, n_estimators=1, bootstrap=False)
m.fit(X_train, y_train)
print_score(m)
m =RandomForestClassifier(n_estimators=5, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators=60, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators=200, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
X_train, X_valid = split_vals(df, n_trn)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
t = m.estimators_[0].tree_
dectree_max_depth(t)
m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators=100, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_estimators=40, n_jobs=-1, min_samples_leaf=3, oob_score=True, max_features=0.5)
m.fit(X_train, y_train)
print_score(m)
fi=rf_feat_importance(m,df); fi
feats=['Name','Ticket','PassengerId','Embarked','Age_na','Fare']
df.drop(feats, axis=1, inplace=True)

fi.plot('cols','imp',figsize=(5,6),legend=False)
X_train, X_valid = split_vals(df, n_trn)
df.head()
m = RandomForestClassifier(n_estimators=50, n_jobs=-1, min_samples_leaf=3,oob_score=True, random_state=1)
m.fit(X_train,y_train)
print_score(m)
m = RandomForestClassifier(n_estimators=50, n_jobs=-1, min_samples_leaf=3, oob_score=True, 
                           random_state=1, max_features=None)
m.fit(X_train,y_train)
print_score(m)
df_test=pd.read_csv('../input/titanic/test.csv')
df_test.head()
train_cats(df_test)
df_test,y_name,nas=proc_df(df_test, 'Name')
df_test.head()
feats=['Ticket','PassengerId','Embarked','Age_na','Fare_na','Fare']

df_test.drop(feats, axis=1, inplace=True)

df_test.head()

df.head()
Survived=m.predict(df_test)
Survived.sum()
df_sample=pd.read_csv('../input/titanic/test.csv')

df_sample['Survived']=pd.Series(Survived)
df_sample.head()
df_sample.to_csv('../input/submit.csv',columns=['PassengerId','Survived'], index=False)
submit=pd.read_csv('submit.csv')


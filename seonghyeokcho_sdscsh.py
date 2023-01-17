import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline

d = pd.read_csv('human.csv', engine='python', encoding='cp949')
#d.head()

d = d.rename(columns={'아이디':'id', '나이':'age', '노동 계급':'workclass', '학력':'education', 
              '교육 수':'education_num', '혼인 상태':'marital_status', '직업':'occupation',
              '관계':'relationship', '인종':'race', '성별':'sex', '자본 이득':'capital_gain',
              '자본 손실':'capital_loss', '주당 시간':'hours_per_week', '모국':'native_country'})
#d.head()

d.sex = (d.sex==' Male').astype(int)
d.head()
sex_frequency = d.sex.value_counts().to_frame()
sex_frequency.style.background_gradient(cmap='coolwarm')
import category_encoders as ce

cols = ['marital_status','relationship','race']
count_enc = ce.CountEncoder()
count_encoded = count_enc.fit_transform(d[cols])
d = d.join(count_encoded.add_suffix("_count"))
d.head()
cols = ['sex','marital_status_count','relationship_count','race_count']
corr_count = d[cols].corr()
corr_count = corr_count.sex.sort_values(ascending=False).to_frame()
corr_count.style.background_gradient(cmap='Dark2')
cols = ['workclass','marital_status','occupation','relationship','race']
one = pd.get_dummies(d[cols], columns=cols)
#one.head()

d = d.join(one)
d.head()
onehot = d[one.columns].join(d.sex)
#onehot

corr_onehot = onehot.corr()
corr_onehot = corr_onehot.sex.sort_values(ascending=False).to_frame()
corr_onehot.style.background_gradient(cmap='Dark2')
d['relationship'] = d['relationship'].astype('category').cat.codes
d.head()
corr_catcode = d[['sex','relationship']].corr()
corr_catcode = corr_catcode.sex.sort_values(ascending=False).to_frame()
corr_catcode.style.background_gradient(cmap='coolwarm')
drop = ['workclass','education','marital_status','occupation','race','native_country']
d = d.drop(drop, axis=1)
d.head()
cols = ['age','fnlwgt','education_num','sex','capital_gain','capital_loss','hours_per_week']
corr_others = d[cols].corr()
corr_others = corr_others.sex.sort_values(ascending=False).to_frame()
corr_others.style.background_gradient(cmap='Dark2')
others = d[cols]
others.head()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
others = others[cols].apply(encoder.fit_transform)
others.head()
corr_lab_others = others.corr()
corr_lab_others = corr_lab_others.sex.sort_values(ascending=False).to_frame()
corr_lab_others.style.background_gradient(cmap='Dark2')
# 이전 파일에서 hours_per_week 컬럼을 제거
d = d.drop('hours_per_week', axis=1)
#d.head()

d = d.join(others['hours_per_week'])
d.head()
corr_total = d.corr()
corr_total = corr_total.sex.sort_values(ascending=False).to_frame()
corr_total.style.background_gradient(cmap='Dark2')
from sklearn.model_selection import train_test_split

feature_data = d.drop(['id','sex'],axis=1)
label_data = d.sex.values

X_train,X_test,y_train,y_test = train_test_split(feature_data,label_data,test_size=.25,random_state=0)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
# decision tree 로 모델링
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=6, random_state=0); tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test); print(classification_report(y_test, pred_tree))
print("-----------------------------------")
print("Dummy model:"); print(confusion_matrix(y_test, pred_dummy))
print("Decision tree:"); print(confusion_matrix(y_test, pred_tree))
from imblearn.combine import SMOTETomek

XX, yy = SMOTETomek(random_state=0).fit_sample(X_train, y_train)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(XX, yy)
y_pred = tree.predict(X_test)
print(classification_report(y_test, y_pred))
print("-----------------------------------")
print("Dummy model:"); print(confusion_matrix(y_test, pred_dummy))
print("Decision tree:"); print(confusion_matrix(y_test, y_pred))
from imblearn.combine import SMOTETomek

X_resampled, y_resampled = SMOTETomek(random_state=0).fit_sample(feature_data, label_data)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=0)
print(feature_data.shape, X_resampled.shape, X_train.shape, X_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import glob

kfold = StratifiedKFold(n_splits=5) # 하이퍼 파라미터 지정
n_it = 12

params = {'max_features':list(np.arange(1, d.shape[1])), 'bootstrap':[False], 'n_estimators': [50], 'criterion':['gini','entropy']}
model = RandomizedSearchCV(RandomForestClassifier(), param_distributions=params, n_iter=n_it, cv=kfold, scoring='roc_auc',n_jobs=-1, verbose=1)
print('MODELING.............................................................................')
model.fit(X_train, y_train)
print('========BEST_AUC_SCORE = ', model.best_score_)
model = model.best_estimator_
print(model.score(X_test, y_test))
print('COMPLETE')
print(model)
forest = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=None, max_features=7,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
forest.fit(X_train, y_train).score(X_test, y_test)
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(base_estimator=forest, random_state=0, n_estimators=200)
bagging.fit(X_train, y_train).score(X_test, y_test)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

y_model = bagging.predict(X_test)
print("Dummy model:"); print(confusion_matrix(y_test, pred_dummy))
print("bagging:"); print(confusion_matrix(y_test, y_model))
print(classification_report(y_test, y_model))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

y_model = model.predict(X_test)
print("Dummy model:"); print(confusion_matrix(y_test, pred_dummy))
print("randomsearchCV:"); print(confusion_matrix(y_test, y_model))
print(classification_report(y_test, y_model))
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

fpr, tpr, _ = roc_curve(y_test, bagging.predict_proba(X_test)[:,1])
auc(fpr, tpr)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
auc(fpr, tpr)
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, model, color=None) :
    model = model + ' (auc = %0.3f)' % auc(fpr, tpr)
    plt.plot(fpr, tpr, label=model, color=color)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.axis([0,1,0,1])
    plt.xlabel('FPR (1 - specificity)')
    plt.ylabel('TPR (recall)')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
fpr_dummy, tpr_dummy, _ = roc_curve(y_test, 
                                    dummy.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_dummy, tpr_dummy, 'dummy model', 'hotpink')
fpr_tree, tpr_tree, _ = roc_curve(y_test, 
                                  bagging.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_tree, tpr_tree, 'bagging', 'red')
fpr_tree, tpr_tree, _ = roc_curve(y_test, 
                                  model.predict_proba(X_test)[:,1])
plot_roc_curve(fpr_tree, tpr_tree, 'randomsearchCV', 'blue')
from sklearn.metrics import precision_recall_curve

def plot_precision_recall_curve(precisions, recalls, color) :
    plt.plot(recalls, precisions, color=color)
    plt.axis([0,1,0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
precisions, recalls, _ = precision_recall_curve(y_test, 
                                    bagging.predict_proba(X_test)[:,1])
plot_precision_recall_curve(precisions, recalls, 'red')
precisions, recalls, _ = precision_recall_curve(y_test, 
                                    model.predict_proba(X_test)[:,1])
plot_precision_recall_curve(precisions, recalls, 'blue')
nd = pd.read_csv('human_new.csv', engine='python', encoding='cp949')
#nd
nd = nd.rename(columns={'아이디':'id', '나이':'age', '노동 계급':'workclass', '학력':'education', 
              '교육 수':'education_num', '혼인 상태':'marital_status', '직업':'occupation',
              '관계':'relationship', '인종':'race', '성별':'sex', '자본 이득':'capital_gain',
              '자본 손실':'capital_loss', '주당 시간':'hours_per_week', '모국':'native_country'})
#nd.head()

### Count Encoding
import category_encoders as ce
cols = ['marital_status','relationship','race']
count_enc = ce.CountEncoder()
count_encoded = count_enc.fit_transform(nd[cols])
nd = nd.join(count_encoded.add_suffix("_count"))
#nd.head()

### One Hot Encoding
cols = ['workclass','marital_status','occupation','relationship','race']
one = pd.get_dummies(nd[cols], columns=cols)
#one.head()
nd = nd.join(one)
#nd.head()

### Cat Codes
nd['relationship'] = nd['relationship'].astype('category').cat.codes
#nd.head()

### Drop Categorical Features
drop = ['workclass','education','marital_status','occupation','race','native_country']
nd = nd.drop(drop, axis=1)
#nd.head()

# Reconstruction Constant Columns
cols = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
others = nd[cols]
others.head()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
others = others[cols].apply(encoder.fit_transform)
#others.head()

# 이전 파일에서 hours_per_week 컬럼을 제거
nd = nd.drop('hours_per_week', axis=1)
#d.head()
nd = nd.join(others['hours_per_week'])
#nd.head()
# id와 함께 csv파일로 저장하기 위해 복사본 준비
fi = nd.copy()
fi = fi.rename(columns={'id':'ID'})
fi.head()
# 예측할 데이터를 train과 열일치
pred = nd.drop('id',axis=1)
pred.head()
fi['SEX'] = bagging.predict(pred)
fi['SEX'] = model.predict(pred)
fi[['ID','SEX']].to_csv('gender_bagging.csv', index=False)
check = pd.read_csv('gender_bagging.csv')
check
fi[['ID','SEX']].to_csv('gender_randomforest.csv', index=False)
check = pd.read_csv('gender_randomforest.csv')
check
# bagging - accuracy : 0.84927
check['SEX'].value_counts()
# randomsearchCV(randomforest) - accuracy : 0.85083
check['SEX'].value_counts()

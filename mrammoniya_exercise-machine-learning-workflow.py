import pandas as pd
import numpy as np
import random
df = pd.read_csv('../input/data.csv')
df.head(3)
df.dtypes
df.shape
df.describe()
df.columns
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(df['thalach'], kde = False, bins=30, color='blue')
sns.distplot(df['chol'], kde=False,color='red')
plt.show()
plt.figure(figsize=(20,14))
sns.countplot(x='age',data = df, hue = 'target',palette='GnBu')
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df, hue='target')
plt.show()
plt.figure(figsize=(8,6))
sns.scatterplot(x='target',y='thalach',data=df, hue='sex')
plt.show()
plt.figure(figsize=(8,6))
sns.swarmplot(x='target',y='thalach',data=df, hue='sex')
plt.show()
plt.figure(figsize=(8,6))
sns.boxplot(x='target',y='thalach',data=df, hue='sex')
plt.show()
plt.figure(figsize=(8,6))
sns.violinplot(x='sex',y='thalach',data=df, hue='target', split=True)
plt.show()
df.isnull().sum()
df.dropna()
df.fillna(method='bfill')
from scipy import stats 
plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df, hue='target')
plt.show()
df2 = df[(np.abs(stats.zscore(df['thalach'])) < 3)]
df2 = df2[(np.abs(stats.zscore(df2['trestbps'])) < 3)]
df2.shape[0]
plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df2, hue='target')
plt.show()
q = df['thalach'].quantile(0.7)
df3 = df[df['thalach'] < q]

q = df3['trestbps'].quantile(0.99)
df3 = df3[df3['trestbps'] < q]
plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df3, hue='target')
plt.show()
df3.shape[0]
df = df3
from sklearn import preprocessing
df.head(3)
df.describe()
label_encoder = preprocessing.LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
df.head(3)
x = df[['chol', 'thalach']].values
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(x)
df.head(5)
pd.DataFrame(df_scaled).head(5)
plt.figure(figsize=(18,18))
plt.rcParams["axes.labelsize"] = 20
sns.set(font_scale=1.4)
sns.heatmap(df.corr(), annot = True ,linewidths=.1)
plt.show()
def find_correlation(data, threshold=0.9):
    corr_mat = data.corr()
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][abs(corr_mat[col])> threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat
columns_to_drop = find_correlation(df.drop(columns=['target']) , 0.7)
df4 = df.drop(columns=columns_to_drop)
df4
corr = df.corr()
linear_features=abs(corr).target.drop('target').sort_values(ascending=False)[:5].keys()
abs(corr).target.drop('target').sort_values(ascending=False)[:5].plot(kind='barh')
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
y = df.copy(deep=True)['target']
model = rf.fit(df.drop('target', axis=1),y)
importance = rf.feature_importances_
feat_importances_act = pd.Series(importance, index=df.drop('target', axis=1).columns)
feat_importances = feat_importances_act.nlargest(20)
feat_importances.plot(kind='barh')
df.dtypes
df['sex'] = df['sex'].astype('object')
df['cp'] = df['cp'].astype('object')
df['fbs'] = df['fbs'].astype('object')
df['restecg'] = df['restecg'].astype('object')
df['exang'] = df['exang'].astype('object')
df['slope'] = df['slope'].astype('object')
df['thal'] = df['thal'].astype('object')
df.dtypes
df.head(10)
df_1 = pd.get_dummies(df, drop_first=True)
df_1.head()
df
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('target',1), df['target'], test_size = .2, random_state=10)
X_train.head()
from sklearn.metrics import accuracy_score
model = RandomForestClassifier(max_depth=20)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# cla_pred.append(accuracy_score(y_test,predictions))
print(accuracy_score(y_test,predictions))
importance = model.feature_importances_
feat_importances_act = pd.Series(importance, index=X_train.columns)
feat_importances = feat_importances_act.nlargest(20)
feat_importances.plot(kind='barh')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

classifiers=[['Logistic Regression :',LogisticRegression()],
       ['Decision Tree Classification :',DecisionTreeClassifier()],
       ['Gradient Boosting Classification :', GradientBoostingClassifier()],
       ['Ada Boosting Classification :',AdaBoostClassifier()],
       ['Extra Tree Classification :', ExtraTreesClassifier()],
       ['K-Neighbors Classification :',KNeighborsClassifier()],
       ['Support Vector Classification :',SVC()],
       ['Gaussian Naive Bayes :',GaussianNB()]]
cla_pred=[]
for name,model in classifiers:
    model=model
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    cla_pred.append(accuracy_score(y_test,predictions))
    print(name,accuracy_score(y_test,predictions))
random.seed(100)
rfmodel = RandomForestClassifier()
#Hyperparameter tuning for Logistic Regression
random.seed(100)
from sklearn.model_selection import GridSearchCV
n_estimators = [10, 20, 50, 100]
max_depth = [5,10,15,20]
hyperparameters = dict(max_depth=max_depth, n_estimators=n_estimators)
h_rfmodel = GridSearchCV(rfmodel, hyperparameters, cv=5, verbose=0)
best_logmodel=h_rfmodel.fit(df.drop('target', 1), df['target'])
print('Best Estimators:', best_logmodel.best_estimator_.get_params()['n_estimators'])
print('Best Max Depth:', best_logmodel.best_estimator_.get_params()['max_depth'])
random.seed(100)
rfmodel = RandomForestClassifier(n_estimators=100, max_depth=10)
rfmodel.fit(X_train, y_train)
predictions = rfmodel.predict(X_test)
print(accuracy_score(y_test,predictions))
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test,predictions)
plt.figure(figsize=(24,12))

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.show()
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,rfmodel.predict_proba(X_test)[:,1])
import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(y_test,rfmodel.predict_proba(X_test), figsize = (20,20))
plt.figure(figsize=(40,18))
plt.show()
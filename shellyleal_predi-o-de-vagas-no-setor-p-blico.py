# Importando as bibliotecas necessárias para o código
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import math
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
pd.options.mode.chained_assignment = None
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Trazendo os dados de uma planilha csv extraída do banco de dados da Força.
df = pd.read_csv("../input/lotacao/LOTACAO.csv",sep=',',low_memory=False)
df
df['orgcomandada'].nunique()

df['quadro'].nunique()

df = df[df.groupby('orgcomandada').orgcomandada.transform(len) > 1]
#verifica se os dados estão equilibrados
df.temvaga.value_counts(normalize=True)
sns.countplot(x = 'temvaga',data = df, palette = 'hls')
plt.show()
plt.savefig('count_plot')
%matplotlib inline
pd.crosstab(df.orgcomandada,df.temvaga).plot(kind='bar')
plt.title('Presença de vaga por tipo de organização')
plt.xlabel('Organização')
plt.ylabel('Vaga')
plt.savefig('existe_vaga_om')
%matplotlib inline
pd.crosstab(df.ano,df.temvaga).plot(kind='bar')
plt.title('Presença de vaga por ano')
plt.xlabel('Ano')
plt.ylabel('Vaga')
plt.savefig('existe_vaga_om')
%matplotlib inline
pd.crosstab(df.posto,df.temvaga).plot(kind='bar')
plt.title('Presença de vaga por posto')
plt.xlabel('Posto')
plt.ylabel('Vaga')
plt.savefig('existe_vaga_om')
%matplotlib inline
pd.crosstab(df.quadro,df.temvaga).plot(kind='bar')
plt.title('Presença de vaga por tipo de quadro')
plt.xlabel('Quadro')
plt.ylabel('Vaga')
plt.savefig('existe_vaga_quadro')
df.describe()
df.dtypes
X, y = df.iloc[:,:-1], df.iloc[:,-1:]
X
y
data = df
data
cat_vars=['orgcomandada', 'quadro', 'posto']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['orgcomandada', 'quadro', 'posto']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
data_final.columns.values
os_x = data_final.loc[:, data_final.columns != 'temvaga']
os_y = data_final.loc[:, data_final.columns == 'temvaga']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(os_x, os_y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['temvaga'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['temvaga']==0]))
print("Number of subscription",len(os_data_y[os_data_y['temvaga']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['temvaga']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['temvaga']==1])/len(os_data_X))
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
os_x=os_data_X
os_y=os_data_y['temvaga']
X_train, X_test, y_train, y_test = train_test_split(os_x, os_y, test_size=0.3, random_state=0)
logreg = LogisticRegression(solver='lbfgs', multi_class='auto',max_iter=500)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
X_pre = pd.get_dummies(df.iloc[:,:-1],drop_first=True)
X_pre
y = df.iloc[:,-1:]
y = y.values.ravel()
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from sklearn import tree
clf = DecisionTreeClassifier(max_depth=3).fit(X_pre, y)
fig, ax = plt.subplots(figsize=(30,10))
out = tree.plot_tree(clf, filled=True, rounded = True, fontsize=20, feature_names=X_pre.columns)
for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(3)
X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size=0.3,
                                                    random_state=42)
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,     ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

random_state = 2
classifiers = [
    GaussianNB(),
    LogisticRegression(C=1, solver='lbfgs', multi_class='auto',max_iter=500),
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(random_state=3),
    RandomForestClassifier(random_state=3),
    AdaBoostClassifier(random_state=3),
    ExtraTreesClassifier(random_state=3),
    GradientBoostingClassifier(random_state=3),
]
accuracy_res = []
algorithm_res = []
for clf in classifiers:
    # clf.fit(features_train, labels_train)
    # Added ravel to convert column vector to 1d array
    clf.fit(X_train, y_train.ravel())
    name = clf.__class__.__name__

    train_predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, train_predictions)
    print(name, "{:.4%}".format(accuracy))
    accuracy_res.append(accuracy)
    algorithm_res.append(name)
    print()

y_pos = np.arange(len(algorithm_res))
plt.barh(y_pos, accuracy_res, align='center', alpha=0.5)
plt.yticks(y_pos, algorithm_res)
plt.xlabel('Accuracy')
plt.title('Algorithms')
plt.show()
import numpy as np
(unique, counts) = np.unique(y_test, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)
import numpy as np
(unique, counts) = np.unique(y_train, return_counts=True)
frequencies = np.asarray((unique, counts)).T

print(frequencies)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
enc = OneHotEncoder().fit(X)
log_reg = LogisticRegression(random_state=0,max_iter=1000).fit(X=enc.transform(X_train), y=y_train)
log_reg.score(enc.transform(X_test), y_test)
from sklearn import metrics
metrics.f1_score(y_test, log_reg.predict(enc.transform(X_test)), average='weighted')
from sklearn.metrics import plot_confusion_matrix
sns.set_style("whitegrid", {'axes.grid' : False})
fig, ax = plt.subplots(figsize=(7, 7))
plot_confusion_matrix(log_reg, enc.transform(X_test), y_test,values_format='d', ax=ax)
X_pre = pd.get_dummies(df.iloc[:,:-1])
X_pre
X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size=0.3,
                                                    random_state=42)
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(
    StandardScaler(), 
    LogisticRegression(solver='lbfgs', multi_class='auto', random_state=123,max_iter=500)
)
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
y = df.temvaga
reg_log = LogisticRegression(solver='lbfgs', random_state=123,max_iter=500)
cross_val_score(reg_log, X_pre, y, cv=5, scoring='accuracy').mean()
X_train = X_train.reset_index(drop=True)
X_train.T
y_train
X_test = X_test.reset_index(drop=True)
X_test.T
y_test
columns = X_pre.columns
columns
# Simulando dado manual de teste de predição 1
ano = [2018]
org = [1, 0, 0]
quad = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
post = [0, 0, 0, 0, 1]
data = np.concatenate ([ano,org,quad,post])
data
new_test = pd.DataFrame(data,index=columns)
new_test = new_test.T
new_test
pipe.predict(new_test)
# Simulando dado manual de teste de predição 2
ano = [2016]
org = [0, 0, 1]
quad = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
post = [0, 0, 0, 1, 0]
data2 = np.concatenate ([ano,org,quad,post])
data2
new_test2 = pd.DataFrame(data2,index=columns)
new_test2 = new_test2.T
new_test2
pipe.predict(new_test2)
# Teste de predição com sample automático 
novo_X = X_test.sample(30, random_state=42)
novo_X = novo_X.reset_index(drop=True)
novo_X.T
x = np
pipe.predict(novo_X)

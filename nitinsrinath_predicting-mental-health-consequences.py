import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings('ignore')
#Function to obtain adjusted dates
def dif_between_dates(d1, d2):
    x1=date(int(d1[2]), int(d1[0]), int(d1[1]))
    x2=date(int(d2[2]), int(d2[0]), int(d2[1]))
    return ((x2-x1).days)
srv=pd.read_csv('survey.csv')
#srv.info()
srv=srv.drop('Timestamp', axis=1)
srv=srv.drop('comments', axis=1)
srv=srv.drop('Country', axis=1)
srv=srv.drop('state', axis=1)
for i in srv.columns:
    if i!='Age':
        srv[i]=pd.Series(srv[i]).str.lower()

maleforms=['male', 'm', 'maile', 'male-ish', 'something kinda male?', 'cis male', 'mal', 'male (cis)', 'make', 'guy (-ish) ^_^'
          , 'male ', 'man', 'msle', 'mail', 'malr', 'cis man' , 'ostensibly male, unsure what that really means']
femaleforms=['female', 'cis female', 'f', 'woman', 'femake', 'female ', 'cis-female/femme', 'female (cis)', 'femail']
for i in srv.index:
    if srv.Gender[i] in maleforms:
        srv.at[i, 'Gender']='M'
    elif srv.Gender[i] in femaleforms:
        srv.at[i, 'Gender']='F'
    else:
        srv.at[i, 'Gender']='O'
srv=srv.dropna(subset=['self_employed'])
srv=srv.fillna(method='ffill', axis='columns')
srv=srv[srv.Age<=110]
srv=srv[srv.Age>=1]
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
srv['work_interfere']=label_encoder.fit_transform(srv['work_interfere'].astype(str))
for i in srv.columns:
    if i!='Age':
        print(i, end=" ")
        srv[i]=label_encoder.fit_transform(srv[i])
        le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(le_name_mapping)
    
srv.corr()
srv.hist(figsize=(20,20))
X=srv.drop('mental_health_consequence', axis=1)
Y=srv['mental_health_consequence']
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
chi_scores=chi2(X,Y)
p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()
p_values
from sklearn.model_selection import train_test_split
use_features=['family_history', 'self_employed', 'treatment', 'Age', 
              'mental_health_interview', 'phys_health_consequence', 'anonymity', 'work_interfere', 
              'obs_consequence', 'leave','supervisor', 'coworkers', 'mental_vs_physical']
X=srv[use_features]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB()
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(random_state=0)
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
from sklearn.neighbors import KNeighborsClassifier
KNC = KNeighborsClassifier(n_neighbors=3)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA = LinearDiscriminantAnalysis()
from sklearn.linear_model import LogisticRegression
LRC = LogisticRegression(random_state=0)
from sklearn.neighbors import RadiusNeighborsClassifier
RNC = RadiusNeighborsClassifier(radius=5.0)
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(max_depth=2, random_state=0)
from sklearn.metrics import accuracy_score
KNtest=pd.DataFrame()
for i in range(1, 16):
    KNC = KNeighborsClassifier(n_neighbors=i)
    KNC.fit(X_train, Y_train)
    y_train_pred=KNC.predict(X_train)
    y_test_pred=KNC.predict(X_test)
    KNtest.at[i, 'TrainAcc']=accuracy_score(Y_train, y_train_pred)
    KNtest.at[i, 'TestAcc']=accuracy_score(Y_test, y_test_pred)
plt.plot(KNtest.index, KNtest.TrainAcc)
plt.plot(KNtest.index, KNtest.TestAcc)
plt.legend()
plt.xlabel('N_neighbors')
plt.ylabel('Accuracy')
plt.grid()
RFtest=pd.DataFrame()
for i in range(1, 16):
    RFC = RandomForestClassifier(max_depth=i, random_state=0)
    RFC.fit(X_train, Y_train)
    y_train_pred=RFC.predict(X_train)
    y_test_pred=RFC.predict(X_test)
    RFtest.at[i, 'TrainAcc']=accuracy_score(Y_train, y_train_pred)
    RFtest.at[i, 'TestAcc']=accuracy_score(Y_test, y_test_pred)
plt.plot(RFtest.index, RFtest.TrainAcc)
plt.plot(RFtest.index, RFtest.TestAcc)
plt.legend()
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.grid()
RNtest=pd.DataFrame()
for i in range(5, 16):
    RNC = RadiusNeighborsClassifier(radius=i)
    RNC.fit(X_train, Y_train)
    y_train_pred=RNC.predict(X_train)
    y_test_pred=RNC.predict(X_test)
    RNtest.at[i, 'TrainAcc']=accuracy_score(Y_train, y_train_pred)
    RNtest.at[i, 'TestAcc']=accuracy_score(Y_test, y_test_pred)
plt.plot(RNtest.index, RNtest.TrainAcc)
plt.plot(RNtest.index, RNtest.TestAcc)
plt.legend()
plt.xlabel('Radius')
plt.ylabel('Accuracy')
plt.grid()
KNC = KNeighborsClassifier(n_neighbors=14)
RFC = RandomForestClassifier(max_depth=7, random_state=0)
RNC = RadiusNeighborsClassifier(radius=7.0)
models=[BNB, GNB, DTC, RFC, KNC, RNC, LDA, LRC]
model_names=['BNB', 'GNB', 'DTC', 'RFC', 'KNC', 'RNC', 'LDA', 'LRC']
model_test=pd.DataFrame()
k=0
for i in models:
    i.fit(X_train, Y_train)
    Y_pred=i.predict(X_test)
    model_test.at[model_names[k], 'testAcc']=accuracy_score(Y_test, Y_pred)
    k=k+1
model_test
from sklearn.cross_validation import cross_val_score
cvscoresRFC=cross_val_score(RFC, X, srv['mental_health_consequence'], cv=5)
cvscoresGNB=cross_val_score(GNB, X, srv['mental_health_consequence'], cv=5)
cvscoresLDA=cross_val_score(LDA, X, srv['mental_health_consequence'], cv=5)

fig, ax=plt.subplots(1,3, figsize=(20,5))
ax[0].boxplot(cvscoresRFC)
ax[1].boxplot(cvscoresGNB)
ax[2].boxplot(cvscoresLDA)

ax[0].set_ylabel('Test Accuracy')
ax[1].set_ylabel('Test Accuracy')
ax[2].set_ylabel('Test Accuracy')
ax[0].set_xlabel('Random Forest')
ax[1].set_xlabel('Gaussian Naive Bayes')
ax[2].set_xlabel('LDA')
feature_selection=pd.DataFrame()
for i in use_features:
    use_x_train=X_train[i].values[:, np.newaxis]
    use_x_test=X_test[i].values[:, np.newaxis]
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)
feature_selection=pd.DataFrame()
use_features1=[i for i in use_features if i!='supervisor']
for i in use_features1:
    use_x_train=pd.DataFrame(X_train[['supervisor', i]])
    use_x_test=pd.DataFrame(X_test[['supervisor', i]])
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)    
feature_selection=pd.DataFrame()
use_features1=[i for i in use_features if i not in ['supervisor', 'phys_health_consequence']]
for i in use_features1:
    use_x_train=pd.DataFrame(X_train[['supervisor', 'phys_health_consequence', i]])
    use_x_test=pd.DataFrame(X_test[['supervisor', 'phys_health_consequence', i]])
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)
feature_selection=pd.DataFrame()
use_features1=[i for i in use_features if i not in ['supervisor', 'phys_health_consequence', 'mental_vs_physical']]
for i in use_features1:
    use_x_train=pd.DataFrame(X_train[['supervisor', 'phys_health_consequence', 'mental_vs_physical', i]])
    use_x_test=pd.DataFrame(X_test[['supervisor', 'phys_health_consequence', 'mental_vs_physical', i]])
    RFC.fit(use_x_train, Y_train)
    y_pred=RFC.predict(use_x_test)
    feature_selection.at[i, 'testAcc']=accuracy_score(Y_test, y_pred)
feature_selection.sort_values('testAcc', ascending=False)
from sklearn.cross_validation import cross_val_score
final_features=['supervisor', 'phys_health_consequence', 'mental_vs_physical']
cvscores=cross_val_score(RFC, srv[final_features], srv['mental_health_consequence'], cv=20)
plt.plot(cvscores)
plt.xlabel('Fold')
plt.ylabel('Test Accuracy')
plt.grid()
print(cvscores.mean())
X=srv[final_features]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
RFC.fit(X_train, Y_train)
y_pred=RFC.predict(X_test)
print('Test Accuracy', accuracy_score(Y_test, y_pred))
print('Train Accuracy', accuracy_score(Y_train, RFC.predict(X_train)))
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred, labels=[0,1,2])
sample=srv[final_features].sample(frac=1).head(10)
sample['prediction']=RFC.predict(sample)
sample

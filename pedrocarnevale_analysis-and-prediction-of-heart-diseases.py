import plotly.graph_objects as go
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
path="../input/heart-disease-uci/heart.csv"
data=pd.read_csv(path)
data
data.target.value_counts()
data=data.rename({'cp':'Chest pain type','trestbps':'Resting blood pressure','chol':'Cholestoral (mg/dl)',
            'fbs':'Fasting blood sugar (>120mg/dl)','restecg':'Resting electrocardiographic results',
                  'thalach':'Maximum heart rate','exang':'Exercise induced angina','ca':'Number of major vessels'},axis=1)
data
correlation_matrix=data.corr()
correlation_matrix.target.sort_values(ascending=False)
scaler=StandardScaler()
data1_scaled=scaler.fit_transform(data.drop(['target'],axis=1))
data1_scaled=pd.DataFrame(data=data1_scaled,columns=data.drop(['target'],axis=1).keys())
data_scaled=pd.concat([data.target,data1_scaled],axis=1)
data_scaled
data_violinplot = pd.melt(data_scaled, id_vars="target",
                    var_name="exams",
                    value_name="result")
data_violinplot
plt.figure(figsize=(15,10))
plt.title("Overall view of all exams")
sns.violinplot(x="exams",y="result",hue="target",data=data_violinplot,split=True)
plt.xticks(rotation=90)
plt.show()
data_plot=data.copy()
data_plot.sex=data_plot.sex.replace([0,1],["woman","man"])
data_plot['Chest pain type']=data_plot['Chest pain type'].replace([0,1,2,3],['Type 0','Type 1','Type 2','Type 3'])
data_plot['Exercise induced angina']=data_plot['Exercise induced angina'].replace([0,1],['No','Yes'])
data_plot['slope']=data_plot['slope'].replace([0,1,2],['Slope 0','Slope 1',
                                                                     'Slope 2'])
data_plot['Number of major vessels']=data_plot['Number of major vessels'].replace([0,1,2,3,4],['0 major vessels','1 major vessels',
                                                                                                        '2 major vessels','3 major vessels',
                                                                                                          '4 major vessels'])
data_plot['thal']=data_plot['thal'].replace([0,1,2,3],['Type 0','Type 1','Type 2','Type 3'])
        
data_plot
plt.figure(figsize=(15,10))
plt.title("Relation between age and target",fontsize=15)
sns.countplot(x="age",hue="target",data=data_plot)
plt.xlabel("Gender",fontsize=15)
plt.grid()
plt.show()
sns.pointplot(x="target",y="age",data=data_plot,color='grey')
plt.xlabel("target",fontsize=15)
plt.ylabel("Age",fontsize=15)
age_healthy=data_plot[(data_plot['target']==0)].age.mean()
age_not_healthy=data_plot[(data_plot['target']== 1)].age.mean()
plt.show()
print(f"Mean age of a person with target=1: {age_not_healthy:.1f}")
print(f"Mean age of a person with target=0: {age_healthy:.1f}")
plt.title("Relation between sex and target",fontsize=15)
sns.countplot(x="sex",hue="target",data=data_plot )
plt.xlabel("Gender",fontsize=15)
plt.grid()
plt.show()
man_ratio=(len(data_plot[(data_plot['sex']=="man") & (data_plot['target']==0)]))/(len(data_plot[(data_plot['sex']=="man")]))
woman_ratio=(len(data_plot[(data_plot['sex']=="woman") & (data_plot['target']==0)]))/(len(data_plot[(data_plot['sex']=="woman")]))
print(f"{man_ratio*100:.2f}% of men have target=0")
print(f"{woman_ratio*100:.2f}% of women have target=0")
plt.title("Relation between Chest pain type and target",fontsize=15)
sns.countplot(x="Chest pain type",hue="target",data=data_plot )
plt.xlabel("Chest pain type",fontsize=15)
plt.grid()
plt.show()
ratio0=(len(data_plot[(data_plot['Chest pain type']=='Type 0') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Chest pain type']=='Type 0')]))
ratio1=(len(data_plot[(data_plot['Chest pain type']=='Type 1') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Chest pain type']=='Type 1')]))
ratio2=(len(data_plot[(data_plot['Chest pain type']=='Type 2') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Chest pain type']=='Type 2')]))
ratio3=(len(data_plot[(data_plot['Chest pain type']=='Type 3') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Chest pain type']=='Type 3')]))
print(f"{ratio0*100:.2f}% of chest pain type 0 have target=0")
print(f"{ratio1*100:.2f}% of chest pain type 1 angina have target=0")
print(f"{ratio2*100:.2f}% of chest pain type 2 pain have target=0")
print(f"{ratio3*100:.2f}% of chest pain type 3 have target=0")
plt.title("Relation between angina during exercises and target",fontsize=15)
sns.countplot(x="Exercise induced angina",hue="target",data=data_plot )
plt.xlabel("Exercise induced angina",fontsize=15)
plt.grid()
plt.show()
ratio_no=(len(data_plot[(data_plot['Exercise induced angina']=="No") & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Exercise induced angina']=="No")]))
ratio_yes=(len(data_plot[(data_plot['Exercise induced angina']=="Yes") & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Exercise induced angina']=="Yes")]))
print(f"{ratio_yes*100:.2f}% of people that exercise produces engina has target=0")
print(f"{ratio_no*100:.2f}% of people that exercise doesn't produce engina has target=0")
plt.figure(figsize=(15,10))
plt.title("Relation between oldpeak and target",fontsize=15)
sns.countplot(x="oldpeak",hue="target",data=data_plot)
plt.xlabel("ST depression induced by exercise relative to rest",fontsize=15)
plt.grid()
plt.show()
sns.pointplot(x="target",y="oldpeak",data=data_plot,color='grey')
plt.xlabel("Target",fontsize=15)
plt.ylabel("ST depression induced by exercise relative to rest",fontsize=15)
oldpeak_healthy=data_plot[(data_plot['target']==0)].oldpeak.mean()
oldpeak_not_healthy=data_plot[(data_plot['target']==1)].oldpeak.mean()
plt.show()
print(f"Mean oldpeak of a person with target=1: {oldpeak_not_healthy:.1f}")
print(f"Mean oldpeak of a person with target=0: {oldpeak_healthy:.1f}")
plt.title("Relation between slope and target",fontsize=15)
sns.countplot(x="slope",hue="target",data=data_plot )
plt.xlabel("Slope of the peak exercise ST segment",fontsize=15)
plt.grid()
plt.show()
ratio0=(len(data_plot[(data_plot['slope']=='Slope 0') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['slope']=='Slope 0')]))
ratio1=(len(data_plot[(data_plot['slope']=='Slope 1') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['slope']=='Slope 1')]))
ratio2=(len(data_plot[(data_plot['slope']=='Slope 2') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['slope']=='Slope 2')]))
print(f"{ratio0*100:.2f}% of slope 0 have target=0")
print(f"{ratio1*100:.2f}% of slope 1 have target=0")
print(f"{ratio2*100:.2f}% of slope 2 have target=0")
plt.figure(figsize=(10,5))
plt.title("Relation between number of major vessels and target",fontsize=15)
sns.countplot(x="Number of major vessels",hue="target",data=data_plot )
plt.xlabel("Number of major vessels",fontsize=15)
plt.grid()
plt.ylabel("Heart Desease Diagnosis",fontsize=15)
plt.show()
ratio0=(len(data_plot[(data_plot['Number of major vessels']=='0 major vessels') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Number of major vessels']=='0 major vessels')]))
ratio1=(len(data_plot[(data_plot['Number of major vessels']=='1 major vessels') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Number of major vessels']=='1 major vessels')]))
ratio2=(len(data_plot[(data_plot['Number of major vessels']=='2 major vessels') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Number of major vessels']=='2 major vessels')]))
ratio3=(len(data_plot[(data_plot['Number of major vessels']=='3 major vessels') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Number of major vessels']=='3 major vessels')]))
ratio4=(len(data_plot[(data_plot['Number of major vessels']=='4 major vessels') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['Number of major vessels']=='4 major vessels')]))
print(f"{ratio0*100:.2f}% of people with 0 major vessels have target=0")
print(f"{ratio1*100:.2f}% of people with 1 major vessels the target=0")
print(f"{ratio2*100:.2f}% of people with 2 major vessels the target=0")
print(f"{ratio3*100:.2f}% of people with 3 major vessels the target=0")
print(f"{ratio4*100:.2f}% of people with 4 major vessels the target=0")
plt.title("Relation between thal and target",fontsize=15)
sns.countplot(x="thal",hue="target",data=data_plot )
plt.xlabel("Chest pain type",fontsize=15)
plt.grid()
plt.show()
ratio0=(len(data_plot[(data_plot['thal']=='Type 0') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['thal']=='Type 0')]))
ratio1=(len(data_plot[(data_plot['thal']=='Type 1') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['thal']=='Type 1')]))
ratio2=(len(data_plot[(data_plot['thal']=='Type 2') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['thal']=='Type 2')]))
ratio3=(len(data_plot[(data_plot['thal']=='Type 3') & (data_plot['target']==0)]))/(len(data_plot[(data_plot['thal']=='Type 3')]))
print(f"{ratio0*100:.2f}% of people with tal=0 have target=0")
print(f"{ratio1*100:.2f}% of people with tal=1 have target=0")
print(f"{ratio2*100:.2f}% of people with tal=2 have target=0")
print(f"{ratio3*100:.2f}% of people with tal=3 have target=0")
data_prediction=data.copy()
data_prediction.sex=data_prediction.sex.replace([0,1],["woman","man"])
data_prediction['Chest pain type']=data_prediction['Chest pain type'].replace([0,1,2,3],['Type 0','Type 1','Type 2','Type 3'])
data_prediction['Exercise induced angina']=data_prediction['Exercise induced angina'].replace([0,1],['No','Yes'])
data_prediction['slope']=data_prediction['slope'].replace([0,1,2],['Slope 0','Slope 1','Slope 2'])
data_prediction['Number of major vessels']=data_prediction['Number of major vessels'].replace([0,1,2,3,4],['0 major vessels','1 major vessels',
                                                                                                        '2 major vessels','3 major vessels',
                                                                                                          '4 major vessels'])
data_prediction['thal']=data_prediction['thal'].replace([0,1,2,3],['Type 0','Type 1','Type 2','Type 3'])

data_prediction=pd.get_dummies(data_prediction)
data_prediction
data_prediction=data_prediction.drop(['thal_Type 0','Number of major vessels_4 major vessels'],axis=1)
def fit_and_predict(model,X,y):
    model.fit(X,y)
    scores = cross_validate(model, X, y, cv=5,
                            scoring=('accuracy'),   
                            return_train_score=True)
    print(f"Train score:{scores['train_score'].mean()}")
    print(f"Test score:{scores['test_score'].mean()}")
    return scores['test_score'].mean()
X=data_prediction.drop(['target'],axis=1)
y=data_prediction['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,train_size=0.7)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled=pd.DataFrame(data=X_scaled,columns=X.keys())
X_scaled
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)
baseline_model=DummyClassifier(strategy='most_frequent')
fit_and_predict(baseline_model,X_train,y_train)
results={}
parameters={'max_iter':[1000,2000,3000,4000,5000]}
linear_svc=GridSearchCV(LinearSVC(random_state=42),parameters)
linearsvc_score=fit_and_predict(linear_svc,X_train_scaled,y_train)
results[linearsvc_score]=linear_svc
parameters2={'n_neighbors':np.arange(2,21)}
kneighbors=GridSearchCV(KNeighborsClassifier(),parameters2)
kneighbors_score=fit_and_predict(kneighbors,X_train_scaled,y_train)
results[kneighbors_score]=kneighbors
parameters3={'learning_rate':[0.01,0.05,0.1],'n_estimators':[100,200,300],'max_depth':np.arange(1,5),
             'subsample':np.arange(0.1,1.5,0.1),'eta':np.arange(0.1,1,0.1)}
xgboost=RandomizedSearchCV(XGBClassifier(),parameters3,n_iter=30)
xgboost_score=fit_and_predict(xgboost,X_train_scaled,y_train)
results[xgboost_score]=xgboost
parameters4={'n_estimators':[100,200,300,400,500],'max_depth':np.arange(1,10)}
forest=RandomizedSearchCV(RandomForestClassifier(random_state=42),parameters4,n_iter=4)
forest_score=fit_and_predict(forest,X_train_scaled,y_train)
results[forest_score]=forest
parameters5={'max_depth':np.arange(1,10),'min_samples_split':np.arange(1,5)}
decision=GridSearchCV(DecisionTreeClassifier(),parameters5)
decision_score=fit_and_predict(decision,X_train_scaled,y_train)
results[decision_score]=decision
parameters6={'C':np.arange(0.1,2,0.1)}
logistic=GridSearchCV(LogisticRegression(),parameters6)
logistic_score=fit_and_predict(logistic,X_train_scaled,y_train)
results[logistic_score]=logistic
best_result = max(results)
best_model = results[best_result]
print("Best Model: ")
print(best_model)
best_model.fit(X_train_scaled,y_train)
prediction=best_model.predict(X_test_scaled)
print(f"\nAccuracy score of the best model: {accuracy_score(y_test,prediction)}")
plt.title("Best Model Confusion Matrix")
best_model_matrix=confusion_matrix(y_test,prediction)
sns.heatmap(best_model_matrix,annot=True,fmt='g')
plt.show()
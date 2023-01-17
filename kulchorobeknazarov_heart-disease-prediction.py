import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix,roc_curve,classification_report,roc_auc_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
df=pd.read_csv('../input/heart-patients/US_Heart_Patients.csv')
df.describe()
df.isnull().sum()
no_m_v=df.dropna(axis=0)
data_w_d=pd.get_dummies(no_m_v,drop_first=True)
data_w_d.columns
y=data_w_d['TenYearCHD']
x=data_w_d.drop(['TenYearCHD'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
models={'Logistic Regression':LogisticRegression(),'KNN':KNeighborsClassifier(),'RandomForest':RandomForestClassifier()}
def fit_and_plot(models,x_train,x_test,y_train,y_test):
    model_scores={}
    for name,model in models.items():
        model.fit(x_train,y_train)
        model_scores[name]=model.score(x_test,y_test)
    return model_scores
scores=fit_and_plot(models,x_train,x_test,y_train,y_test)
scores
# We  can see that from three tools above , RandomForestClassifier showed the best result - 85,2% 
train_scores=[]
test_scores=[]

knn=KNeighborsClassifier()

neighbors=range(1,41)

for i in neighbors:
    knn.fit(x_train,y_train)
    knn.set_params(n_neighbors=i)
    train_scores.append(knn.score(x_train,y_train))
    test_scores.append(knn.score(x_test,y_test))
plt.plot(neighbors,train_scores,label='train',color='orange')
plt.plot(neighbors,test_scores,label='test',color='darkblue')
plt.legend()
plt.show()

print(f'Test maximum score:{max(test_scores)*100:.2f}%')
#From the hypertuning of the KNN , we were able to increase KNN score from 82% to 84,84% . But it still lower than
#logistic regression score.
# Now we will hypertune logistic regression and random forest classifier
log_grid={'C':np.logspace(-4,4,20),'solver':['liblinear']}
random_grid={'n_estimators':[50,100,150],'max_depth':[None,5,10],'max_features':['auto','sqrt'],'min_samples_split':[2,4,6],'min_samples_leaf':[1,2,3]}
rs_log_search=RandomizedSearchCV(LogisticRegression(),param_distributions=log_grid,n_iter=5,cv=5,verbose=True)
rs_random_search=RandomizedSearchCV(RandomForestClassifier(),param_distributions=random_grid,n_iter=5,cv=5,verbose=True)
rs_log_search.fit(x_train,y_train)
rs_random_search.fit(x_train,y_train)
print(rs_log_search.score(x_test,y_test))
print(rs_random_search.score(x_test,y_test))

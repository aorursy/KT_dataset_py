#Importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split


%matplotlib inline
df=pd.read_csv('../input/seattleWeather_1948-2017.csv')
df.shape
df.head()
df.describe()
sns.boxplot(data=df[['TMAX','TMIN','PRCP']])
plt.title("Box plot with Outliers")
tmin_Q=df['TMIN'].quantile([0.25,0.75])
tmax_Q=df['TMAX'].quantile([0.25,0.75])
prcp_Q=df['PRCP'].quantile([0.25,0.75])
Q3_tmin=tmin_Q.get_values()[1]
Q1_tmin=tmin_Q.get_values()[0]

Q3_tmax=tmax_Q.get_values()[1]
Q1_tmax=tmax_Q.get_values()[0]

Q3_prcp=prcp_Q.get_values()[1]
Q1_prcp=prcp_Q.get_values()[0]
iqr_tmin=Q3_tmin-Q1_tmin
iqr_tmax=Q3_tmax-Q1_tmax
iqr_prcp=Q3_prcp-Q1_prcp
df=df.drop(df[df['TMIN']< Q1_tmin-1.5*iqr_tmin].index)
df=df.drop(df[(df['TMAX']< Q1_tmax-1.5*iqr_tmax) | (df['TMAX']> Q3_tmax+1.5*iqr_tmax)].index)
df=df.drop(df[(df['PRCP']< 0) | (df['PRCP']> Q3_prcp+1.5*iqr_prcp)].index)

df.shape
sns.boxplot(data=df[['TMAX','TMIN','PRCP']])
plt.title("Box Plot after removal of outliers")
#Some user-defined functions

def Rates(tn,fp,fn,tp):
    TPR=float(tp/(tp+fn))
    TNR=float(tn/(tn+fp))
    FPR=float(fp/(tn+fp))
    FNR=float(fn/(tp+fn))
    print("True Positive Rate or Sensitivity = %f" %(TPR*100))
    print("True Negative Rate or Specificity = %f" %(TNR*100))
    print("False Positive Rate or Fall-out = %f" %(FPR*100))
    print("False Negative Rate or Missclassification rate = %f" %(FNR*100))


def tran(Y):
    if Y==True:
        temp=1
    else:
        temp=0
    return temp        
       
Y=df['RAIN'].map(tran)    
df=df.drop(['DATE','RAIN'],axis=1)
df.head()
df.isna().sum()
im=Imputer()
df=im.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(df, Y, test_size=0.3, stratify = Y, random_state=42)
sd=StandardScaler()
X_train = sd.fit_transform(X_train)
X_test = sd.transform(X_test)
tuned_parameters = [{'C': [10**-4, 10**-3,10**-2,10**-1,10**0,10**1,10**2,10**3,10**4]}]
#Using GridSearchCV
model = GridSearchCV(LogisticRegression(), tuned_parameters,
                     scoring = 'accuracy', cv=5,n_jobs=-1)
model.fit(X_train, y_train)
cv_scores=[x[1] for x in model.grid_scores_]

#Calculating misclassification error
MSE = [1 - x for x in cv_scores]

#Finding best K
val=list(tuned_parameters[0].values())
optimal_value=val[0][MSE.index(min(MSE))]

print("\n The optimal value of C in Logistic Regression is %f ." %optimal_value)

# plot misclassification error vs C 
plt.plot(val[0], MSE)

for xy in zip(val[0], np.round(MSE,3)):
    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')

plt.xlabel('Value of C')
plt.ylabel('Misclassification Error')
plt.show()
lr=LogisticRegression(C=optimal_value)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
lr.score(X_test,y_test)
tn, fp, fn, tp =confusion_matrix(y_test, y_pred).ravel()
Rates(tn,fp,fn,tp)
x=confusion_matrix(y_test, y_pred)
cm_df=pd.DataFrame(x,index=[0,1],columns=[0,1])

sns.set(font_scale=1.4,color_codes=True,palette="deep")
sns.heatmap(cm_df,annot=True,annot_kws={"size":16},fmt="d",cmap="YlGnBu")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Value")
plt.ylabel("True Value")
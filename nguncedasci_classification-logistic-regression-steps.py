import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
#import the data and indicate X and y
diabetes=pd.read_csv("../input/diabetes.csv")
df=diabetes.copy()
df=df.dropna()
df.head()
# We will try to improve ML model to predict that when the new patient has come, he/she has diabetes or not.
df.info()
df["Outcome"].value_counts()
df["Outcome"].value_counts().plot.barh();
df.describe().T
y=df["Outcome"]
X=df.drop("Outcome", axis=1)
#set and fit the model with statsmodels
loj=sm.Logit(y,X)
loj_model=loj.fit()
loj_model.summary()
#We didn't split the data, first we will apply log. reg. to whole data for easy learning. 
#After understand the issue, we will apply the rules on splitted data
#set and fit the model with sklearn
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X,y)
loj_model
loj_model.intercept_   #beta0
loj_model.coef_       #all of the other betas(beta1...-beta8)
#Model Tuning
# NOTE: There isn't ANY hiperparameter(external parameter) for logistic reg. Therefore, model tuning means confirmation
# So, we can only optimize beta0.
y_pred=loj_model.predict(X)
confusion_matrix(y,y_pred)
accuracy_score(y,y_pred)  # (True Pozitive+True Negative)/ All. This is primitive, we haven't verified it
print(classification_report(y,y_pred))
loj_model.predict(X)[0:5]    #predictions
loj_model.predict_proba(X)[0:5]   #first column give us the probability of being in class 0
                                  #second column gives us the probabilty of being in class 1 so we consider this column
y[0:5]   #real values
#we will specify a threshold(ex. 0.5) and if the proba exceeds the threshold we will say that it is from class 1
y_probs=loj_model.predict_proba(X)
y_probs=y_probs[:,1]
y_probs[0:5]
y_pred=[1 if i>0.5 else 0 for i in y_probs]
y_pred[0:5]   #we found classes manually
print(classification_report(y,y_pred))   #confussion_matrix
logit_roc_auc = roc_auc_score(y, loj_model.predict(X))

fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()


# Easy learning trial is finished. 
# Now, we apply rules on splitted data and find the RIGHT accuracy rate and gain the REAL test error.
#Normally, we follow these rules.. 
#Model Accuracy
#split data train test
#gain an error with cv
# 1) split the data into train-test
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.30, random_state=42)
# 2) set and fit the model
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)
loj_model
# 3) Gain the test error
y_pred= loj_model.predict(X_test)
accuracy_score(y_test,y_pred)
# 4) CV-Cross Validated score (it means verified error)
cross_val_score(loj_model,X_test,y_test,cv=10).mean()


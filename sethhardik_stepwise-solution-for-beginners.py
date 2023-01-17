
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,roc_curve,classification_report,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split,cross_val_score
%matplotlib inline
data = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
data.head()
data.shape
data.isnull().sum()
# number of nulls are drawn using heatmap we can also print the values ( caution may not show small values like in this case )
sns.heatmap(data.isnull(),yticklabels=False,cbar=False) 
plt.show()
sns.countplot(x="Property_Area",data=data)
plt.show()
sns.countplot(x="Gender",data=data)
sns.countplot(x="Loan_Status",hue="Married",data=data)
# it can be seen that married people are more likely to be given loan
sns.countplot(x="Loan_Status",hue="Credit_History",data=data,palette = "rainbow")
plt.show()
# it can be seen that credit history  is one of the most important feature for loan status
sns.countplot(x="Loan_Status",data=data)
data.hist(bins=50,figsize=(10,10),grid=False)
plt.tight_layout()
plt.show()
sns.boxplot(x="Credit_History",y="LoanAmount",data=data,palette="winter")
sns.countplot(x="Dependents",data=data)
plt.show()
sns.countplot(x="Self_Employed",data=data)
plt.show()
sns.countplot(x="Education",hue="Loan_Status",data=data)
plt.show()
# from the boxplot i can see that mean loan amount for both credit history is approximately same so i will fill nan using mean 
# value only
data.LoanAmount.fillna(data.LoanAmount.mean(),inplace=True)
# from the countplot of gender w can see that male is  more than female so we have high chances of nan being male 
data.Gender.fillna("Male",inplace=True)
data.Dependents.fillna('0',inplace=True)
data.Self_Employed.fillna("No",inplace=True)
data.Credit_History.fillna(0.0,inplace=True)
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mode()[0],inplace=True)
data.Married.fillna(data.Married.mode()[0],inplace = True)
# ouliers in loan amount dataset can be log transformed so that effect of outliers can be removed
data["log_LoanAmount"] = np.log(data.LoanAmount)
data.log_LoanAmount.hist(bins=50)
plt.show()
# rather than using differnet values of the Income combining both to get a general income and than taking log to remove 
# effect of outliers
data["totalIncome_log"] = np.log(data.ApplicantIncome + data.CoapplicantIncome)
data.totalIncome_log.hist()
data.head()
data = data.drop(["Loan_ID","ApplicantIncome","CoapplicantIncome","LoanAmount"],axis=1)
data.head()
data.info()
from sklearn.preprocessing import LabelEncoder
categorical_column = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in categorical_column:
    data[i] = le.fit_transform(data[i])
data.head()
data.info()
data.info()
train = data
train.head()
train.Loan_Status.value_counts()
from sklearn.utils import resample
data_major = train[train.Loan_Status==1]
data_minor = train[train.Loan_Status==0]
data_upscale = resample(data_minor,replace= True,n_samples=422)
train=pd.concat([data_major,data_upscale])
train.shape
def roc_curve_do(c_name,classifier,x_test,y_test):
    probs = classifier.predict_proba(x_test)  
    probs = probs[:, 1]  
    fper, tper, thresholds = roc_curve(y_test, probs)
    plt.plot(fper, tper)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC curve'.format(c_name))
  # show the plot
    plt.show()

def kfold(classifier,X,Y,cv):
    score=cross_val_score(classifier,X,Y,cv=cv)
    print("Individual Score:",score)
    print("Mean Score:",score.mean()*100,"%")
  # plot
    plt.plot(np.arange(cv), score, 'o-', linewidth=1)
    plt.title("Accuracy: %f%% and Deviation (%f%%)" % (score.mean()*100, score.std()*100))
    plt.xlabel('number of Folds')
    plt.ylabel('Accuracy score')
    plt.show()

def all_score(classifier,x_test,y_test,x_train,y_train):
    predict=classifier.predict(x_test)
    print("testing accuracy:",accuracy_score(y_test,predict))
    print("training accuracy:",accuracy_score(y_train,classifier.predict(x_train)))
    print(confusion_matrix(y_test,predict))
    print("Classification report:\n",classification_report(y_test,predict))
xtrain = train.drop(["Loan_Status"],axis=1)
ytrain = train["Loan_Status"]
x_train,x_test,y_train,y_test = train_test_split(xtrain,ytrain,test_size=0.2)
model = LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
roc_curve_do("Logistic Regression",model,x_test,y_test)
kfold(model,xtrain,ytrain,5)
all_score(model,x_test,y_test,x_train,y_train)
model1 = RandomForestClassifier(n_estimators=200)
model1.fit(x_train,y_train)
roc_curve_do("Logistic Regression",model1,x_test,y_test)
kfold(model1,xtrain,ytrain,5)
all_score(model1,x_test,y_test,x_train,y_train)
parameters =  ['Gender', 'Married', 'Dependents', 'Education','Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','log_LoanAmount','totalIncome_log']
featimp = pd.Series(model1.feature_importances_, index=parameters).sort_values(ascending=False)
print (featimp)
new_train = train.drop(['Property_Area','Married','Education','Gender','Self_Employed','Dependents','Loan_Amount_Term'],axis=1)
new_train.head()
new_xtrain = new_train.drop(["Loan_Status"],axis=1)
new_ytrain = new_train["Loan_Status"]
new_xtrain.head()
x_newtrain,x_newtest,y_newtrain,y_newtest = train_test_split(new_xtrain,new_ytrain,test_size=0.2)
model2 = RandomForestClassifier(n_estimators=150)
model2.fit(x_newtrain,y_newtrain)
roc_curve_do("Random Forest",model2,x_newtest,y_newtest)
kfold(model2,new_xtrain,new_ytrain,5)
all_score(model2,x_newtest,y_newtest,x_newtrain,y_newtrain)

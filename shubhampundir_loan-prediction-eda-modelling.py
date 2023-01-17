import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
Loan_data=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

Loan_data.head()
Loan_data.shape 
Loan_data.info()
Loan_data.isnull().sum().sort_values(ascending=False)
Loan_data.drop("Loan_ID",axis=1,inplace=True)
Loan_data.describe(include="all")
Loan_data.columns
Loan_data['Credit_History']=Loan_data["Credit_History"].astype("object")
Loan_data.duplicated().any()
#Loan Status.

figsize=(7,5)

sns.countplot(Loan_data["Loan_Status"])
#Credit_History

grid=sns.FacetGrid(Loan_data,col="Loan_Status",size=3.5)

grid.map(sns.countplot,"Credit_History")
#Gender

grid=sns.FacetGrid(Loan_data,col="Loan_Status",size=3.5)

grid.map(sns.countplot,"Gender")
#Marriage

grid=sns.FacetGrid(Loan_data,col="Loan_Status",size=3.5)

grid.map(sns.countplot,"Married")
#dependents 

plt.figure(figsize=(8,5))

sns.countplot(x="Dependents",hue="Loan_Status",data=Loan_data,saturation=5)
#Education

plt.figure(figsize=(7,6))

sns.countplot(x="Education",hue="Loan_Status",data=Loan_data,saturation=5)
#Self employed

plt.figure(figsize=(8,5))

sns.countplot(x="Self_Employed",hue="Loan_Status",data=Loan_data,saturation=5)
#property area 

plt.figure(figsize=(7,5))

sns.countplot(x="Property_Area",hue="Loan_Status",data=Loan_data,saturation=5)
Loan_mean_amount = Loan_data.groupby(['Property_Area','Education'])['LoanAmount'].mean().reset_index()



plt.figure(figsize=(9,6))

sns.barplot(x='Property_Area',y='LoanAmount',hue='Education',data=Loan_data)

plt.xlabel("Property Area of Education")

plt.ylabel("Average loan amount")

plt.show()
cateagorical_data=[]

Numerical_data=[]



for i,c in enumerate(Loan_data.dtypes):

  if c==object:

    cateagorical_data.append(Loan_data.iloc[:,i])

  else:

      Numerical_data.append(Loan_data.iloc[:,i])
cateagorical_data=pd.DataFrame(cateagorical_data).transpose()

Numerical_data=pd.DataFrame(Numerical_data).transpose()
cateagorical_data.head()
Numerical_data.head()
cateagorical_data=cateagorical_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

cateagorical_data.isnull().sum().any()
Numerical_data=Numerical_data.apply(lambda x:x.fillna(x.mean()))

Numerical_data.isnull().sum().any()
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()



Label_value={"Y":0,"N":1}

Label=cateagorical_data["Loan_Status"]

cateagorical_data.drop("Loan_Status",axis=1,inplace=True)

Label=Label.map(Label_value)
for i in cateagorical_data:

  cateagorical_data[i]=LE.fit_transform(cateagorical_data[i])
cateagorical_data.head()
Label
Loan_data=pd.concat([cateagorical_data,Numerical_data,Label],axis=1)
plt.figure(figsize=(10,7))

plt.title("Correlation Matrix")

sns.heatmap(Loan_data.corr(),annot=True,)
from sklearn.model_selection import train_test_split

X=pd.concat([cateagorical_data,Numerical_data],axis=1)

Y=Label
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=28)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



models={"LogisticRegression":LogisticRegression(random_state=28),

        "KNeighborsClassifier":KNeighborsClassifier(),

        "DecisionTreeClassifier":DecisionTreeClassifier(max_depth=1,random_state=28),

        "RandomForestClassifier":RandomForestClassifier(n_estimators=400,oob_score=True,random_state=28,n_jobs=-1),

        "AdaBoostClassifier":AdaBoostClassifier(random_state=28)

        }
from sklearn.metrics import precision_score,accuracy_score



def loss(y_true,y_pred,retu=False):

  pre=precision_score(y_true,y_pred)

  acc=accuracy_score(y_true,y_pred)





  if retu:

    return pre,acc

  else:

      print(' pre: %.3f\n  acc: %.3f\n '%(pre,acc))
def train_pred_score(models,x,y):

  for name,model in models.items():

    print(name," :")

    model.fit(x,y)

    loss(y,model.predict(x))

    print("-"*30)



train_pred_score(models,x_train,y_train)
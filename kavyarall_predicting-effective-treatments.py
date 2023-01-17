from IPython.display import Image

Image("../input/infographic/INFOGRAPHIC.jpg")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/diabetes/diabetic_data.csv')

data.shape
data.columns
data.info()
data.isnull().values.any()
data.race.value_counts().plot(kind = 'bar' )
data.payer_code.value_counts().plot(kind = 'bar' )
data.medical_specialty.value_counts()
data.max_glu_serum.value_counts().plot(kind = 'bar' )
data.A1Cresult.value_counts().plot(kind = 'bar' )
data.change.value_counts().plot(kind = 'bar' )
data.diabetesMed.value_counts().plot(kind = 'bar' )
data.readmitted.value_counts().plot(kind = 'bar' )
data.age.value_counts().plot(kind = 'bar')
data=data[data.diabetesMed=='Yes']

data.shape
data=data[data.readmitted=='NO']

data.shape
data=data[~data.discharge_disposition_id.isin([11,13,14,19,20])]

data.shape
data = data.drop(['medical_specialty','payer_code','weight'],axis=1)
data['race']=data.race.replace('?',np.nan)

data['race'].fillna(data['race'].mode()[0], inplace=True)

data.race.isnull().sum()
data.shape
data.columns
treatments = data[['encounter_id','metformin', 'repaglinide', 'nateglinide',

       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',

       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',

       'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',

       'insulin', 'glyburide-metformin', 'glipizide-metformin',

       'glimepiride-pioglitazone', 'metformin-rosiglitazone',

       'metformin-pioglitazone']].copy()
treatments.head()
treatments=treatments.replace(['No','Steady','Up','Down'],[0,1,1,1])

treatments.set_index('encounter_id',inplace=True)

treatments.head()
treatments.sum(axis=1).value_counts()
i1 = treatments[treatments['insulin']==1].sum(axis = 1).replace([1,2,3,4,5,6],['insulin','io','io','io','io','io'])
i1.value_counts()
i0=treatments[treatments['insulin']==0].sum(axis=1).replace([0,1,2,3,4,5,6],['no med','other','other','other','other','other','other'])
i0.value_counts()
treatments=pd.concat([i1,i0])

treatments = pd.DataFrame({'treatments':treatments})
treatments.head()
data=data.join(treatments,on='encounter_id') #setting index as encounter_id
data.head()
data = data.drop(['metformin', 'repaglinide', 'nateglinide',

       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',

       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',

       'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',

       'insulin', 'glyburide-metformin', 'glipizide-metformin',

       'glimepiride-pioglitazone', 'metformin-rosiglitazone',

       'metformin-pioglitazone'],axis=1)
data=data[data.treatments!='other']

data.shape
data.columns
data = pd.get_dummies(data, columns=['race', 'gender','max_glu_serum', 'A1Cresult', 'change',

       'diabetesMed', 'readmitted'])
data.head()
data.age.value_counts()
labels = data['age'].astype('category').cat.categories.tolist()

replace_age = {'age' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}



print(replace_age)
data.replace(replace_age, inplace=True)
data.age.value_counts()
data.num_lab_procedures.plot(kind='hist')
import seaborn as sns

sns.distplot(data.time_in_hospital)
import matplotlib.pyplot as plt

age_count = data['age'].value_counts()

sns.set(style="darkgrid")

sns.barplot(age_count.index, age_count.values, alpha=0.9)

plt.title('Frequency Distribution of age')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Age', fontsize=12)

plt.show()
labels = data['age'].astype('category').cat.categories.tolist()

counts = data['age'].value_counts()

sizes = [counts[var_cat] for var_cat in labels]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot

ax1.axis('equal')

plt.show()
data.columns
data = data.drop(['diag_1','diag_2','diag_3'],axis = 1)
from IPython.display import Image

Image("../input/correlation/Picture1.png")
import pandas as pd

import numpy as np

import scipy.stats as stats

from scipy.stats import chi2_contingency



class ChiSquare:

    def __init__(self, dataframe):

        self.df = dataframe

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)
data['dummyCat'] = np.random.choice([0, 1], size=(len(data),), p=[0.5, 0.5])



data.dummyCat.value_counts()
#Initialize ChiSquare Class

cT = ChiSquare(data)



#Feature Selection

testColumns = ['encounter_id', 'patient_nbr', 'age', 'admission_type_id',

       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',

       'num_lab_procedures', 'num_procedures', 'num_medications',

       'number_outpatient', 'number_emergency', 'number_inpatient','number_diagnoses',

       'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 'race_Hispanic',

       'race_Other', 'gender_Female', 'gender_Male',

       'max_glu_serum_>200', 'max_glu_serum_>300', 'max_glu_serum_None',

       'max_glu_serum_Norm', 'A1Cresult_>7', 'A1Cresult_>8', 'A1Cresult_None',

       'A1Cresult_Norm', 'change_Ch', 'change_No', 'diabetesMed_Yes',

       'readmitted_NO', 'dummyCat']

for var in testColumns:

    cT.TestIndependence(colX=var,colY="treatments" ) 
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
X = data.drop(['encounter_id','patient_nbr','num_lab_procedures','number_outpatient','number_emergency',

                      'race_Asian','race_Other','diabetesMed_Yes','max_glu_serum_>200','A1Cresult_>8','A1Cresult_Norm',

                      'readmitted_NO','dummyCat','treatments'],axis=1)

Y = data['treatments']

print(X.shape)

print(Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
y_p=[]

for i in range(y_test.shape[0]):

    y_p.append(y_test.mode()[0])#Highest class is assigned to a list which is compared with ytest

len(y_p) 
y_pred=pd.Series(y_p)
print("Accuracy : ",accuracy_score(y_test,y_pred))
#Logistic Regression

m1=LogisticRegression()

m1.fit(X_train,y_train)

y_pred_lr=m1.predict(X_test)

Train_Score_lr = m1.score(X_train,y_train)

Test_Score_lr = accuracy_score(y_test,y_pred_lr)





print('Training Accuracy is:',Train_Score_lr)

print('Testing Accuracy is:',Test_Score_lr)

print(classification_report(y_test,y_pred_lr))
m2 = KNeighborsClassifier()

m2.fit(X_train,y_train)

y_pred_knn = m2.predict(X_test)

Train_Score_knn = m2.score(X_train,y_train)

Test_Score_knn = accuracy_score(y_test,y_pred_knn)



print('Training Accuracy is :',Train_Score_knn)

print('Testing Accuracy is:',Test_Score_knn)

print(classification_report(y_test,y_pred_knn))

m3=BernoulliNB()

m3.fit(X_train,y_train)

y_pred_bnb=m3.predict(X_test)

Train_Score_bnb = m3.score(X_train,y_train)

Test_Score_bnb = accuracy_score(y_test,y_pred_bnb)



print('Training Accuracy :',Train_Score_bnb)

print('Testing Accuracy  :',Test_Score_bnb)

print(classification_report(y_test,y_pred_bnb))
m4 = DecisionTreeClassifier()

m4.fit(X_train,y_train)

y_pred_dt=m4.predict(X_test)

Train_Score_dt = m4.score(X_train,y_train)

Test_Score_dt = accuracy_score(y_test,y_pred_dt)



print('Training Accuracy :',Train_Score_dt)

print('Testing Accuracy :',Test_Score_dt)

print(classification_report(y_test,y_pred_dt))
m5 = RandomForestClassifier()

m5.fit(X_train,y_train)

y_pred_rf=m5.predict(X_test)

Train_Score_rf = m5.score(X_train,y_train)

Test_Score_rf = accuracy_score(y_test,y_pred_rf)



print('Training Accuracy :',Train_Score_rf)

print('Testing Accuracy :',Test_Score_rf)

print(classification_report(y_test,y_pred_rf))
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 3



# parameters to build the model on

parameters = {'max_depth': range(5, 15, 5),

    'min_samples_leaf': range(50, 150, 50),

    'min_samples_split': range(50, 150, 50),

    'criterion': ["entropy", "gini"]}



# instantiate the model

dtree = DecisionTreeClassifier(random_state = 100)



# fit tree on training data

tree = GridSearchCV(dtree, parameters, 

                    cv=n_folds, 

                   scoring="accuracy")

tree.fit(X_train, y_train)
tree.best_params_
m6 = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_leaf=50,min_samples_split=50)

m6.fit(X_train,y_train)

y_pred_tdt=m6.predict(X_test)

Train_Score_tdt = m6.score(X_train,y_train)

Test_Score_tdt = accuracy_score(y_test,y_pred_tdt)



print('Training Accuracy :',Train_Score_tdt)

print('Testing Accuracy  :',Test_Score_tdt)

print(classification_report(y_test,y_pred_tdt))

#Gridsearch CV to find Optimal K value for KNN model

grid = {'n_neighbors':np.arange(1,50)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,grid,cv=3)

knn_cv.fit(X_train,y_train)





print("Tuned Hyperparameter k: {}".format(knn_cv.best_params_))
m7 = KNeighborsClassifier(n_neighbors=19)

m7.fit(X_train,y_train)

y_pred_tknn=m7.predict(X_test)

Train_Score_tknn = m7.score(X_train,y_train)

Test_Score_tknn = accuracy_score(y_test,y_pred_tknn)





print('Training Accuracy :',Train_Score_tknn)

print('Testing Accuracy  :',Test_Score_tknn)

print(classification_report(y_test,y_pred_tknn))
parameter={'n_estimators':np.arange(1,101)}

gs = GridSearchCV(m5,parameter,cv=3)

gs.fit(X_train,y_train)

gs.best_params_



m8 = RandomForestClassifier(n_estimators=73)

m8.fit(X_train,y_train) 

y_pred_trf=m8.predict(X_test)

Train_Score_trf = m8.score(X_train,y_train)

Test_Score_trf = accuracy_score(y_test,y_pred_trf)





print('Training Accuracy :',Train_Score_trf)

print('Testing Accuracy  :',Test_Score_trf)

print(classification_report(y_test,y_pred_trf))
data.treatments.replace(['insulin','io'],[0,1],inplace = True)
a = data.drop(['age','treatments'],axis=1)

b = data.treatments
cate_features_index = np.where(a.dtypes != int)[0]

xtrain,xtest,ytrain,ytest = train_test_split(a,b,train_size=.70,random_state=2)

from catboost import CatBoostClassifier, Pool,cv

#let us make the catboost model, use_best_model params will make the model prevent overfitting

model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)
model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))
#show the model test acc, but you have to note that the acc is not the cv acc,

#so recommend to use the cv acc to evaluate your model!

print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))

test_score_catboost = accuracy_score(ytest,model.predict(xtest))

print("the train accuracy is :",model.score(xtrain,ytrain))

train_score_catboost = model.score(xtrain,ytrain)
model.predict(xtest)
Model_Scores=pd.DataFrame({'Models':['Logistic Regression','KNN','Bernauli Naives Bayes','Decision Tree','Random Forest','Tuned Decison Tree','Tuned KNN','Tuned Random Forest','Cat Boost'],

             'Training Accuracy':[Train_Score_lr,Train_Score_knn,Train_Score_bnb,Train_Score_dt,Train_Score_rf,Train_Score_tdt,Train_Score_tknn,Train_Score_trf,train_score_catboost],

             'Testing Accuracy':[Test_Score_lr,Test_Score_knn,Test_Score_bnb,Test_Score_dt,Test_Score_rf,Test_Score_tdt,Test_Score_tknn,Test_Score_trf,test_score_catboost],

                })



Model_Scores.sort_values(by=('Testing Accuracy'),ascending=False)
%load_ext autoreload

%autoreload 2

%matplotlib inline
from IPython.display import display



from sklearn.tree import export_graphviz

import graphviz
dot_data = export_graphviz(

    m6,

    out_file=None,

    feature_names=X.columns,

    class_names=['insulin', 'Insulin+others'],

    filled=True,

    rounded=True,

    special_characters=True)

graph = graphviz.Source(dot_data)

graph
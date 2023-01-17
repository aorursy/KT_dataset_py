# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import scipy



import pandas as pd

import matplotlib 

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.manifold import TSNE

from IPython.core.interactiveshell import InteractiveShell

import warnings

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

warnings.filterwarnings('ignore') # to supress seaborn warnings

pd.options.display.max_columns = None # Remove pandas display column number limit

#InteractiveShell.ast_node_interactivity = "all" # Display all values of a jupyter notebook cell

import sys

#import savReaderWriter as sav

#import the evaluatation metric

from sklearn.metrics import balanced_accuracy_score

#pandas library for reading data

import pandas as pd

#numpy library for computation with matrices and arrays

import numpy as np

#matplotlib library for visualization

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action='ignore')

#command for displaying visualizations within the notebook

%matplotlib inline



from sklearn.metrics import confusion_matrix, recall_score, precision_score

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score

from sklearn.linear_model import LogisticRegression

#from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn import preprocessing, metrics

from xgboost import XGBClassifier

warnings.filterwarnings('ignore') # to supress warnings

from sklearn.model_selection import learning_curve, GridSearchCV

#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import learning_curve, GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score

from sklearn.linear_model import LogisticRegression

#from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn import preprocessing, metrics

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore') # to supress warnings

#Import libraries:

import pandas as pd

import numpy as np

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

#from sklearn import cross_validation, metrics   #Additional scklearn functions

#from sklearn.grid_search import GridSearchCV   #Perforing grid search

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2D, Flatten, BatchNormalization

from tensorflow.keras.layers import Embedding, Masking, LSTM, Dropout

from tensorflow.keras.utils import to_categorical

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

import seaborn as sns
LOANDATA=pd.read_csv("/kaggle/input/loandata1/LOAN.csv")

LOANDATA.head()
import pandas_profiling as pp

pp.ProfileReport(LOANDATA)
LOANDATA.info()
#LOANDATA1=LOANDATA.copy

LOANDATA.Class.value_counts()
LOANDATA1=LOANDATA.copy()
LOANDATA1.isna().sum()
#replacing spaced words in culumns (' ', '_')  with underscore for ease 

LOANDATA1.columns =LOANDATA1.columns.str.strip().str.replace(' ', '_')

LOANDATA1.head(1)
LOANDATA1.drop('Unnamed:_0', axis=1,inplace=True)
LOANDATA1.set_index(['Customer_Id'],inplace=True)
LOANDATA1=LOANDATA1.fillna(value={"Province":"Umujyi wa Kigali","Class":"Normal"})
LOANDATA1["Class"]=LOANDATA1["Class"].map({"Normal": "Normal","Acceptable Risk":"Acceptable Risk","Special Mention": "Special Mention", "Doubtful": "Doubtful","Substandard":"Substandard","Loss":"Doubtful"})
LOANDATA1.Class.value_counts()
LOANDATA1["Class"]=LOANDATA1["Class"].map({"Normal": 1,"Acceptable Risk":2,"Special Mention": 3, "Doubtful": 4,"Substandard":5}).astype(int)
LOANDATA1["CreditScoreGroup"]=LOANDATA1["CreditScoreGroup"].map({"A": 0, "B": 1, "C": 2}).astype(int)
LOANDATA1["Province"]=LOANDATA1["Province"].map({"Umujyi wa Kigali": 0, "Iburasirazuba": 1, "Iburengerazuba": 2,'Amajyepfo':3,"Amajyaruguru":4,"Diaspora - A":5}).astype(int)
LOANDATA1["PaymentStatus"]=LOANDATA1["PaymentStatus"].str.strip().str.replace(' ', '_')

LOANDATA1["PaymentStatus"]=LOANDATA1["PaymentStatus"].map({"Completely_Repaid":0,"Partially_Repaid":1,"In_arrears":2,"Not_yet":3}).astype(int)
LOANDATA1["ReturningCustomer"]=(LOANDATA1["ReturningCustomer"]).astype(int)
LOANDATA4=LOANDATA1.drop(['District','Effective_Date','Maturity_Date','Date_of_Birth'], axis=1)
LOANDATA1.isnull().sum()
#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = LOANDATA1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_target = abs(cor["Class"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.06]

relevant_features
col=["Principal_amt","Paid_Penalty","Total_Remaining_prinipal","Remaining_principal","Due_Principal","Previous_Days","Overdue_Days",

     "CreditScoreGroup","PaymentStatus","Duration","ReturningCustomer"]
col1=["Principal_amt","Paid_Penalty","Total_Remaining_prinipal","Remaining_principal","Due_Principal","Previous_Days","Overdue_Days",

     "CreditScoreGroup","PaymentStatus","Duration","ReturningCustomer","Class"]
LOANDATA3=LOANDATA1[col1]
X=pd.get_dummies(LOANDATA3.drop('Class',1))

Y=LOANDATA3.Class
train_ratio = 0.7

validation_ratio = 0.15

test_ratio = 0.15

kfold = 5



x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)

StratifiedKFold(n_splits=kfold, random_state=42)

# test is now 15% of the initial data set

# validation is now 15% of the initial data set

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
# list of different classifiers we are going to test

clfs = {

'LogisticRegression' : LogisticRegression(),

'GaussianNB': GaussianNB(),

'RandomForest': RandomForestClassifier(),

'DecisionTreeClassifier': DecisionTreeClassifier(),

'SVM': SVC(),

'KNeighborsClassifier': KNeighborsClassifier(),

'GradientBoosting': GradientBoostingClassifier(),

'XGBClassifier': XGBClassifier()

}
# code block to test all models in clfs and generate a report

models_report = pd.DataFrame(columns = ['Model', 'Precision_score', 'Recall_score','F1_score', 'Accuracy'])



for clf, clf_name in zip(clfs.values(), clfs.keys()):

    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_test)

    y_score = clf.score(x_test,y_test)

    

    #print('Calculating {}'.format(clf_name))

    t = pd.Series({ 

                     'Model': clf_name,

                     'Precision_score': metrics.precision_score(y_test, y_pred,average='macro'),

                     'Recall_score': metrics.recall_score(y_test, y_pred,average='macro'),

                     'F1_score': metrics.f1_score(y_test, y_pred,average='macro'),

                     'Accuracy': metrics.accuracy_score(y_test, y_pred)}

                   )



    models_report = models_report.append(t, ignore_index = True)



models_report
def gridsearch(model, params,x_train, x_test, y_train, y_test, kfold):

    gs = GridSearchCV(model, params, scoring='accuracy', n_jobs=-1, cv=kfold)

    gs.fit(x_train, y_train)

    print ('Best params: ', gs.best_params_)

    print ('Best AUC on Train set: ', gs.best_score_)

    print( 'Best AUC on Test set: ', gs.score(x_test, y_test))



# Function to generate confusion matrix

def confmat(pred, y_test):

    conmat = np.array(confusion_matrix(y_test, pred, labels=[1,2,3,4,5]))

    conf = pd.DataFrame(conmat, index=["Normal","Acceptable Risk","Special Mention","Doubtful","Substandard"],

                             columns=["Pred Normal","Pred Acceptable Risk","Pred Special Mention","Pred Doubtful","Pred Substandard"])

    print( conf)



# Function to plot roc curve

def roc(prob, y_test):

    y_score = prob

    fpr = dict()

    tpr = dict()

    roc_auc=dict()

    fpr[1], tpr[1], _ = roc_curve(y_test, y_score)

    roc_auc[1] = auc(fpr[1], tpr[1])

    plt.figure(figsize=[7,7])

    plt.plot(fpr[1], tpr[1], label='Roc curve (area=%0.2f)' %roc_auc[1], linewidth=4)

    plt.plot([1,0], [1,0], 'k--', linewidth=4)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.0])

    plt.xlabel('False Positive rate', fontsize=15)

    plt.ylabel('True Positive rate', fontsize=15)

    plt.title('ROC curve for Credit Default', fontsize=16)

    plt.legend(loc='Lower Right')

    plt.show()

    

def model(md, x_train, y_train,x_test, y_test):

    md.fit(x_train, y_train)

    pred = md.predict(x_test)

    #prob = md.predict_proba(x_test)[:,1]

    print( ' ' )

    print ('Accuracy on Train set: ', md.score(x_train, y_train))

    print( 'Accuracy on Test set: ', md.score(x_test, y_test))

    print( ' ')

    print(classification_report(y_test, pred))

    print( ' ')

    print('Confusion Matrix',confmat(pred, y_test))

    

    #roc(prob, y_test)

    return md
# feature selection with the best model from grid search

gb = GradientBoostingClassifier(learning_rate= 0.02, max_depth= 8,n_estimators=100, max_features = 0.9,min_samples_leaf = 100)

model_gb = model(gb, x_train, y_train,x_test, y_test)
XGB=XGBClassifier(max_depth=3,

    learning_rate=0.3,

    n_estimators=200,

    n_jobs=1,

    gamma=0,

    min_child_weight=3,

    missing=None,) 

model_xgb = model(XGB, x_train, y_train,x_test, y_test)
# # Use gridsearch to fine tune the parameters

# xgb = XGBClassifier()

# xgb_params = {'n_estimators':[200,300],'learning_rate':[0.05,0.02], 'max_depth':[4],'min_child_weight':[0],'gamma':[0]}

# gridsearch(xgb, xgb_params,x_train, x_test, y_train, y_test,5)
LOANDATA3['PREDICTED_Class']=np.int_(model_gb.predict(LOANDATA3.drop(['Class'],axis = 1)))
LOANDATA3.PREDICTED_Class.value_counts()
LOANDATA3.Class.value_counts()


#____________________________



from sklearn import svm

from sklearn import linear_model

from sklearn import neighbors

from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib

from sklearn.metrics import classification_report

import json



class MLModel:

  def __init__(self, target_model, save_path):

    self.save_path = save_path

    self.models = {"LogisiticRegression": LogisticRegression(),

            "DecisionTreeClassifier": DecisionTreeClassifier(),

            "svm":svm.SVC(), 

            "knn":neighbors.KNeighborsClassifier(5), 

            "randomforest":RandomForestClassifier(n_estimators=10),

           }



    if target_model in self.models.keys():

      self.model = self.models[target_model]

    else:

      raise NotImplementedError

    

    self.clf = None

  

  def fit(self, X, y):

    #put your code down here

    self.clf = self.model.fit(X, y)



  def save(self, mode="pickle"):

    if mode == "pickle":

      if self.clf is not None:

        joblib.dump(self.clf, self.save_path)

      else:

        raise ValueError("train before saving the classifier")



    elif mode == "json":

      if self.clf is not None:

        model_dict = {}

        model_dict["clf"] = self.clf

        json_clf = json.dumps(model_dict, indent=4)

        with open(self.save_path+"/model.json", 'w') as file:

            file.write(json_clf)

        file.close()

      else:

        raise ValueError("Train before saving the classifier")



    else:

      raise NotImplementedError



  def load(self, path, mode):

    if mode == "pickle":

      self.clf = joblib.load(path)

      print(self.clf)

    elif mode == "json":

      with open(path, 'r') as file:

          model_dict = json.load(file)

      self.clf = model_dict["clf"]

    else:

      raise NotImplementedError

    return self.clf



  def predict(self, X):

    predictions = self.clf.predict(X)

    return predictions



  def evaluate(self, y_true, y_pred, target_names):

    # put your code down here add eval

    print(classification_report(y_true, y_pred, target_names=target_names))



    
model_randomforest=MLModel('randomforest','./randomforest.pkl')

model_randomforest.fit(x_train,y_train)

predictions=model_randomforest.predict(x_test)

print(predictions)
sum(predictions==y_test)*1.0/len(y_test)
model_randomforest.evaluate(predictions, y_test,target_names= [str(i) for i in np.unique (y_test)])
model = Sequential()

model.add(Dense(128, activation="relu", input_shape = (x_train.shape[1],))) # Hidden Layer 1 that receives the Input from the Input Layer



model.add(Dense(64, activation="relu")) # Hidden Layer 2

model.add(Dropout(0.2))



model.add(Dense(32, activation="relu")) # Hidden Layer 3

model.add(Dropout(0.2))



model.add(Dense(16, activation="relu")) # Hidden Layer 4

model.add(Dropout(0.2))





model.add(Dense(1, activation="softmax")) # Outout Layer



model.summary()
model.compile(optimizer='adam', loss = "binary_crossentropy", metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 64, epochs = 20)
validation_loss, validation_accuracy = model.evaluate(x_test, y_test, batch_size=32)

print("Loss: "+ str(np.round(validation_loss, 3)))

print("Accuracy: "+ str(np.round(validation_accuracy, 3)))
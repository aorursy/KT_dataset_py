import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import  ensemble,neighbors
from xgboost import XGBClassifier
import statsmodels.api as sm
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
#Reading train and test pet data

train_pet_adoption=pd.read_csv("../input/train/train.csv")

test_pet_adoption=pd.read_csv("../input/test/test.csv")
#Looking at training data set
train_pet_adoption.head()
#Printing column names of train data set
train_pet_adoption.columns = train_pet_adoption.columns.str.replace(' ', '')

map(str,list(train_pet_adoption.columns))
#Selecting required columns
train_sel=train_pet_adoption[['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize',
                             'FurLength','Vaccinated','Dewormed','Sterilized','Health','Fee','State','AdoptionSpeed']]

test_sel=train_pet_adoption[['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize',
                             'FurLength','Vaccinated','Dewormed','Sterilized','Health','Fee','State']]
#Preparing data for machine learining 
y=train_sel[['AdoptionSpeed']]
X=train_sel[['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize',
                             'FurLength','Vaccinated','Dewormed','Sterilized','Health','Fee','State']]
models = [
    
    #Ensemble Methods
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
          
           
    #xgboost
    XGBClassifier()    
    ]
for model in models:
    
    
    model.fit(X, y.values.ravel())
    
  
            
    feat_imp = pd.DataFrame({'importance':model.feature_importances_})    
    feat_imp['feature'] = X.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:X.shape[1]]
    
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=model.__class__.__name__, figsize=(15,15))
    plt.xlabel('Feature Importance Score')
    plt.show()
#Predicting using GBC
gbc = ensemble.GradientBoostingClassifier(min_samples_split=0.1,max_depth=5,n_estimators =20,learning_rate=0.01)

gbc.fit(X, y)
print(gbc.score(X, y))
#Predicting data
pred_gbc = gbc.predict(test_sel)
pred_gbc=pd.DataFrame(pred_gbc)
test_pet_adoption['AdoptionSpeed']=pred_gbc
#Submitting results

sub=test_pet_adoption[['PetID','AdoptionSpeed']]

sub.to_csv('submission.csv',index=False)

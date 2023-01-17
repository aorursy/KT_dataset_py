# Filter out warnings
import warnings
warnings.filterwarnings('ignore')

# For dataframe displaying purposes
from IPython.display import display

# Data analysis and processing
import pandas as pd
import numpy as np
import re

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Original, unprocessed data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Save for submission and training respectively
passenger_id = test['PassengerId']
target = train['Survived']
# Drop PassengerId on both copies
for df in [train,test]:
    df.drop('PassengerId',axis=1,inplace=True)
all_data = pd.concat([train.drop('Survived',axis=1),test]).reset_index(drop=True)
# Save indexes for splitting all_data later on
ntrain = train.shape[0]
ntest = test.shape[0]
# Create FamSize feature and drop 
all_data['FamSize'] = 1 + all_data['SibSp'] + all_data['Parch']
all_data.drop(['SibSp','Parch'],axis=1,inplace=True)
total_miss = all_data.isnull().sum()
percent_miss = (total_miss/all_data.isnull().count()*100)

# Creating dataframe from dictionary
missing_data = pd.DataFrame({'Total missing':total_miss,'% missing':percent_miss})

missing_data.sort_values(by='Total missing',ascending=False).head()
all_data[all_data['Embarked'].isnull()]
sns.boxplot(x='Pclass',y='Fare',
            data=all_data[(all_data['Pclass']==1)&(all_data['Fare']<=200)],
            hue='Embarked')
_ = all_data.set_value(61,'Embarked',value='C')
_ = all_data.set_value(829,'Embarked',value='C')
display(all_data[all_data['Fare'].isnull()])
display(all_data[all_data['Fare']==0])
_ = all_data.set_value(1043,'Fare',value=0)
splits = 5
# Intervals for discretizing fare values
for i in range(splits):
    print(f'Group {i+1}:',pd.qcut(all_data['Fare'],splits).sort_values().unique()[i])
def discretize_fare(val):
    
    fare_group = pd.qcut(all_data['Fare'],splits).sort_values().unique()
    
    for i in range(splits):
        
        if val in fare_group[i]:
            return i+1
        elif np.isnan(val):
            return val
all_data['Fare'] = all_data['Fare'].apply(discretize_fare)
all_data['Fare'] = all_data['Fare'].fillna(5).astype(int)
# Intervals for discretizing each age value
for i in range(splits):
    print(f'Group {i+1}:',pd.cut(all_data['Age'].dropna(), splits).unique()[i])
def discretize_age(val):
    
    age_group = pd.cut(all_data['Age'],splits).sort_values().unique()
    
    for i in range(splits):
        
        if val in age_group[i]:
            return i+1
        elif np.isnan(val):
            return 0
all_data['Age'] = all_data['Age'].apply(discretize_age).astype(int)
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
all_data['Title'] = all_data['Name'].apply(get_title)
# Looking at unique titles
all_data['Title'].unique()
all_data['Title'] = all_data['Title'].replace(['Ms','Mlle'],'Miss')
all_data['Title'] = all_data['Title'].replace('Mme','Mrs')
all_data['Title'] = all_data['Title'].replace(['Don','Dona','Lady','Sir',
                                                 'Countess','Jonkheer'],'Royal')
all_data['Title'] = all_data['Title'].replace(['Rev','Major','Col','Capt','Dr'],'Other')
plt.figure(figsize=(10,4))
sns.stripplot(x='Title',y='Age',data=all_data[all_data['Age']!=0],
              hue='Pclass',dodge=True)
plt.legend(loc=1)
def impute_age(row):
    
    # Features from row
    pclass = row['Pclass']
    title = row['Title']
    age = row['Age']
    
    if age == 0:
        return int(round(all_data.loc[(all_data['Age']!=0)&
                                      (all_data['Pclass']==pclass)&
                                      (all_data['Title']==title)]['Age'].mean(),1))
    else:
        return age
all_data['Age'] = all_data.apply(impute_age,axis=1)
_ = all_data.rename({'Cabin':'Deck'},axis=1,inplace=True)
all_data['Deck'] = all_data['Deck'].fillna('N')
def cabin_to_deck(row):
    return row['Deck'][0]
all_data['Deck'] = all_data.apply(cabin_to_deck,axis=1)
ticket_list = []
for ticket_id in list(all_data['Ticket'].unique()):
    
    count = all_data[all_data['Ticket']==ticket_id].count()[0]
    decks = all_data[all_data['Ticket']==ticket_id]['Deck']
    empty_decks = (decks=='N').sum()
    
    if (count > 1) and (empty_decks > 0) and (empty_decks < len(decks)):
        ticket_list.append(ticket_id)

print(ticket_list)
# Show dataframes with the previous specifications
for ticket in ticket_list:
    display(all_data[all_data['Ticket']==ticket])
# ticket ID, information

# 2668, 2 siblings (sharing with mother)
_ = all_data.set_value(533,'Deck',value=all_data.loc[128]['Deck'])
_ = all_data.set_value(1308,'Deck',value=all_data.loc[128]['Deck'])

# PC 17755, maid to Mrs. Cardeza
_ = all_data.set_value(258,'Deck',all_data.loc[679]['Deck'])

# PC 17760, manservant to Mrs White 
_ = all_data.set_value(373,'Deck',value='C')

# 19877, maid to Mrs Cavendish
_ = all_data.set_value(290,'Deck',value=all_data.loc[741]['Deck'])

# 113781, maid and nurse to the Allisons
_ = all_data.set_value(708,'Deck',value=all_data.loc[297]['Deck'])
_ = all_data.set_value(1032,'Deck',value=all_data.loc[297]['Deck'])

# 17421, maid to Mrs Thayer
_ = all_data.set_value(306,'Deck',value='C')

# PC 17608, governess (teacher) to Master Ryerson
_ = all_data.set_value(1266,'Deck',value=all_data.loc[1033]['Deck'])

# 36928, parents (sharing with daughters)
_ = all_data.set_value(856,'Deck',value=all_data.loc[318]['Deck'])
_ = all_data.set_value(1108,'Deck',value=all_data.loc[318]['Deck'])

# PC 17757, maid and manservant to the Astors
_ = all_data.set_value(380,'Deck',value='C')
_ = all_data.set_value(557,'Deck',value='C')

# PC 17761, maid to Mrs Douglas, occupied room with another maid
_ = all_data.set_value(537,'Deck',value='C')

# 24160, maid to Mrs. Robert, testimony that she was on deck E
_ = all_data.set_value(1215,'Deck',value='E')

# S.O./P.P. 3, very little information, will assume on deck E with Mrs. Mack
_ = all_data.set_value(841,'Deck',value=all_data.loc[772]['Deck'])
fig,ax = plt.subplots(1,2,figsize = (10,4))
plt.tight_layout(w_pad=2)
ax = ax.ravel()

sns.countplot(x='Pclass',data=all_data[all_data['Deck']!='N'],hue='Deck',ax=ax[0])
ax[0].legend(loc=1)
ax[0].set_title('Pclass count for known Deck')
sns.countplot(x='Pclass',data=all_data[all_data['Deck']=='N'],hue='Deck',ax=ax[1])
ax[1].set_title('Pclass count for unkown Deck')
decks_by_class = [[],[],[]]
for i in range(3):
    decks_by_class[i] = list(all_data[all_data['Pclass']==i+1]['Deck'].unique())
    print(f'Pclass = {i+1} decks:',decks_by_class[i])
# Removing null ('N') entries and single 'T' cabin
for i in range(3):
    if 'N' in decks_by_class[i]:
        decks_by_class[i].remove('N')
    if 'T' in decks_by_class[i]:
        decks_by_class[i].remove('T')
weights_by_class = [[],[],[]]

for i,deck_list in enumerate(decks_by_class):
    for deck in deck_list:
        if i == 0:
            class_total = all_data[(all_data['Deck']!='N')&(all_data['Pclass']==i+1)].count()[0]-1
        else:
            class_total = all_data[(all_data['Deck']!='N')&(all_data['Pclass']==i+1)].count()[0]
        deck_total = all_data[(all_data['Deck']==deck)&(all_data['Pclass']==i+1)].count()[0]
        weights_by_class[i].append(deck_total/class_total)
    print(f'Pclass = {i+1} weights:',np.round(weights_by_class[i],3))
# Store tickets that were already looped with cabin position
ticket_dict = {}
def impute_deck(row):
    
    ticket = row['Ticket']
    deck = row['Deck']
    pclass = row['Pclass']
    
    if (deck == 'N') and (ticket not in ticket_dict):
        
        if pclass == 1:
            deck = list(np.random.choice(decks_by_class[0],size=1,
                                         p=weights_by_class[0]))[0]
        elif pclass ==2:
            deck = list(np.random.choice(decks_by_class[1],size=1,
                                         p=weights_by_class[1]))[0]
        elif pclass ==3:
            deck = list(np.random.choice(decks_by_class[2],size=1,
                                         p=weights_by_class[2]))[0]
        
        ticket_dict[ticket] = deck
        
    elif (deck == 'N') and (ticket in ticket_dict):
        deck = ticket_dict[ticket]
    
    return deck
all_data['Deck'] = all_data.apply(impute_deck,axis=1)
all_data.head(1)
all_data = all_data.drop(['Name','Ticket','Title'],axis=1)
all_data['Deck'] = all_data['Deck'].map({'F':0,'C':1,'E':2,
                                             'G':3,'D':4,'A':5,
                                             'B':6,'T':7}).astype(int)
all_data['Embarked'] = all_data['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
all_data['Sex'] = all_data['Sex'].map( {'female':0,'male':1}).astype(int)
all_data['Alone'] = 0
all_data.loc[all_data['FamSize']==1,'Alone'] = 1
all_data.head()
all_data.shape[0] == ntrain + ntest
# Cross-validation
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score

# Estimators
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
# These are for using with CV while testing parameters
def rmse_cv(model,train): 
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train)
    return np.sqrt(-cross_val_score(model,train,target,scoring='neg_mean_squared_error',cv=kf))

def logloss_cv(model,train):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train)
    return -cross_val_score(model,train,target,scoring='neg_log_loss',cv=kf)

def accuracy_cv(model,train):
    kf = KFold(n_folds,shuffle=True,random_state=42).get_n_splits(train.values)
    return cross_val_score(model,train,target,scoring='accuracy',cv=kf)
# These are for using with predictions and target
def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def accuracy(y_true,y_pred):
    return accuracy_score(y_true,y_pred)
rf = RandomForestClassifier(n_estimators=700,max_depth=4,
                            min_samples_leaf=1,n_jobs=-1,
                            warm_start=True,
                            random_state=42)

et = ExtraTreesClassifier(n_estimators=550,max_depth=4,
                          min_samples_leaf=1,n_jobs=-1,
                          random_state=42)

ada = AdaBoostClassifier(n_estimators=550,learning_rate=0.001,
                         random_state=42)

svc = SVC(C=2,probability=True,random_state=42)
# Class is inheriting methods from these classes
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
# Metrics for measuring our fit
from sklearn.metrics import mean_squared_error, accuracy_score
class StackerLvl1(BaseEstimator, ClassifierMixin, TransformerMixin):
    
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
    # Get OOF predictions
    def oof_pred(self, X, y):
        
        self.base_models_ = [list() for x in self.base_models]
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            
            for train_index, test_index in kfold.split(X, y):
                
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.loc[train_index], y.loc[train_index])
                y_pred = instance.predict(X.loc[test_index])
                out_of_fold_predictions[test_index, i] = y_pred
            
        return out_of_fold_predictions

    # Fit meta model using OOF predictions
    def fit(self, X, y):
        
        self.meta_model_ = clone(self.meta_model)
        self.meta_model_.fit(self.oof_pred(X,y), y)
        return self
    
    # Predict off of meta features using meta model
    def predict(self, test):
        self.meta_features_ = np.column_stack([
            np.column_stack([model.predict(test) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(self.meta_features_)
train = all_data[:ntrain]
test = all_data[ntrain:]
# Create our stack object and fit it
stack_model  = StackerLvl1(base_models=(rf,et,svc),meta_model = ada)
stack_model.fit(train,target)

# Get metrics from cv (note that we are fitting to train data and comparing to target!)
print('Accuracy:',accuracy(stack_model.predict(train),target)) 
print('RMSE:',rmse(stack_model.predict(train),target)) 
stack_model_pred = stack_model.predict(test)
sub = pd.DataFrame({'PassengerId':passenger_id, 
                    'Survived':stack_model_pred})
sub.to_csv('submission.csv',index=False)
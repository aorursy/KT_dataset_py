import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')


pd.DataFrame(train.dtypes, columns=['Type'])
train.head(10)
train.describe()
train.isna().mean().round(4) * 100
test.isna().mean().round(4) * 100
colors = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(train.isnull(), cmap=sns.color_palette(colours))
# create new df so we can keep the original one and manipulate a new one
train_model = train.copy()
test_model = test.copy()
#add the median for age column
train_model['Age'] = train_model['Age'].fillna(train_model['Age'].median())
test_model['Age'] = test_model['Age'].fillna(train_model['Age'].median())

#the train dataset had missing values for fare, so we will do the same but for fare feature on test set
test_model['Fare'] = test_model['Fare'].fillna(train_model['Fare'].median())

test.head(10)
train_model = train_model.drop(['Cabin'],axis=1)
test_model = test_model.drop(['Cabin'],axis=1)
test_model.head(10)
#The test dataset didn't have missing values for Embarked, so we will apply the imputation only for the train dataset
train_model['Embarked'] = train_model['Embarked'].fillna(train_model['Embarked'].mode()[0])

train_model.isnull().mean().round(4) * 100
test_model.isnull().mean().round(4) * 100
import matplotlib.pyplot as plt


f, ax = plt.subplots(figsize=(10, 8))
ax = sns.violinplot(x="Survived", y="Fare", data=train_model)
ax.set_xticklabels(['no','yes']);


f, ax = plt.subplots(figsize=(10, 8))
ax = sns.violinplot(x="Sex", y="Age", hue="Survived", data=train_model,split=True);

pclass_total = train_model.groupby(['Pclass']).sum()
pclass_total = pclass_total['Survived']
plot = pclass_total.plot.pie(figsize=(8, 6),autopct='%1.1f%%')
plt.title('Percentage of survivals by class')
#plot distribution for weekly sales
f, ax = plt.subplots(figsize=(10, 8))
ax = sns.distplot(train_model['Fare'])
plt.ylabel('Distribution');
f, ax = plt.subplots(figsize=(10, 8))
ax = sns.boxplot(x="Embarked", y="Fare", hue="Survived",
                 data=train_model, palette="Set3");
def plot_bar(variable):
    rating_probs_die = pd.DataFrame(train_model[train_model.Survived == 0].groupby(variable).size().div(len(train_model)))
    rating_probs_survived = pd.DataFrame(train_model[train_model.Survived == 1].groupby(variable).size().div(len(train_model)))
    df = rating_probs_die.merge(rating_probs_survived,how='outer',left_index=True, right_index=True)
    df.columns = ['died', 'survived']
    ax = df.plot.bar(rot=0,colormap='Paired')
    
plot_bar("SibSp")    

plot_bar("Parch")
## Family_size seems like a good feature to create
train_model['FamilySize'] = train_model.SibSp + train_model.Parch+1
test_model['FamilySize'] = test_model.SibSp + test_model.Parch+1

plot_bar("FamilySize")  
## get the most important variables. 
corr = train_model.corr()**2
corr.Survived.sort_values(ascending=False)

mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style('whitegrid')
plt.subplots(figsize = (15,12))
sns.heatmap(train.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"
            linewidths=.9, 
            linecolor='white',
            fmt='.2g',
            center = 0,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40);

def has_SibSp_Parch(dataset):
    ''' this function creates a new feature if the person has 1 sibling 
    or spouse aboard the Titanic '''
    dataset['has_SibSp'] = dataset['SibSp'].apply(lambda x: 'True' if x==1 else 'False')
    dataset['has_Parch'] = dataset['Parch'].apply(lambda x: 'True' if x==1 else 'False')
    
    return dataset
    
train_model = has_SibSp_Parch(train_model)
test_model = has_SibSp_Parch(test_model)
    
## get the title from the name
train_model["title"] = [i.split('.')[0] for i in train_model.Name]
train_model["title"] = [i.split(',')[1] for i in train_model.title]

test_model["title"] = [i.split('.')[0] for i in test_model.Name]
test_model["title"] = [i.split(',')[1] for i in test_model.title]

train_model["title"].value_counts()
import re



def has_gender(dataset):
    ''' this function adds a new feature if it is mr, mrs, or miss. 
    It uses regular expressios in the case of Mr'''
    dataset['Mr'] = dataset['title'].apply(lambda x: 'True'  if re.search(r'Mr\b',x)  else 'False')
    dataset['Mrs'] = dataset['title'].apply(lambda x: 'True'  if 'Mrs' in x else 'False')
    dataset['Miss'] = dataset['title'].apply(lambda x: 'True'  if 'Miss' in x else 'False')
    dataset['Master'] = dataset['title'].apply(lambda x: 'True'  if 'Master' in x else 'False')
    dataset['Rare'] = dataset['title'].apply(lambda x: 'True'  if not re.search(r'Mr\b',x)  and 'Mrs' not in x and 'Miss' not in x else 'False')
    dataset = dataset.drop(['title'],axis=1)
    
    return dataset
    

train_model = has_gender(train_model)
test_model = has_gender(test_model)
def family_size(dataset):
    ''' this function creates a new feature if the person has 1 sibling 
    or spouse aboard the Titanic '''
    dataset['alone'] = dataset['FamilySize'].apply(lambda x: 'True' if x==1 else 'False')
    dataset['small_family'] = dataset['FamilySize'].apply(lambda x: 'True' if x in np.arange(2, 4) else 'False')
    dataset['big_family'] = dataset['FamilySize'].apply(lambda x: 'True' if x>=4 else 'False')
    
    return dataset

#train_model = family_size(train_model)
#test_model = family_size(test_model)
def clean(dataset):
    dataset = dataset.drop(['Name'],axis=1)
    dataset = dataset.drop(['PassengerId'],axis=1)
    dataset = dataset.drop(['Ticket'],axis=1)
    
    return dataset

train_model = clean(train_model)
test_model = clean(test_model)

train_model = pd.get_dummies(train_model)
test_model = pd.get_dummies(test_model)
!pip install pycaret
from pycaret.classification import * #this is not a good practice, but since we are only testing, I will import everything
model_setup = setup(data=train_model, target='Survived',session_id=1,silent = True)

compare_models()
model = create_model('gbc',fold=10)
tuned_model = tune_model('gbc',fold=10)
# plotting a model
plot_model(tuned_model,plot = 'auc')
# plotting a model
plot_model(tuned_model,plot = 'feature')

# plotting a model
plot_model(tuned_model,plot = 'pr')

# plotting a model
plot_model(tuned_model,plot = 'confusion_matrix')
y_pred = predict_model(tuned_model, data=test_model)


# blending all models
blend_all = blend_models(method='hard')
y_pred_blend = predict_model(blend_all, data=test_model)
# create individual models for stacking

lightgbm = create_model('lightgbm')
xgboost = create_model('xgboost')
catboost = create_model('catboost')
lda = create_model('lda')
ridge = create_model('ridge')
gbc = create_model('gbc')



# stacking models
stacker = stack_models(estimator_list = [lightgbm,xgboost,ridge,catboost,lda,gbc],
                       meta_model = xgboost,method='hard')
y_pred_stack = predict_model(stacker, data=test_model)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred['Label']
    })
#submission.to_csv("submission.csv", index=False)
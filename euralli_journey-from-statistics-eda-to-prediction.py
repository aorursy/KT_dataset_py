# Initialize Notebook

import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat
%matplotlib inline
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, Normalizer

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.describe()
test.describe()
train.describe(include=['O'])
test.describe(include=['O'])
train.head()
f,ax = plt.subplots(3,4,figsize=(20,16))
sns.countplot('Pclass',data=train,ax=ax[0,0])
sns.countplot('Sex',data=train,ax=ax[0,1])
sns.boxplot(x='Pclass',y='Age',data=train,ax=ax[0,2])
sns.countplot('SibSp',hue='Survived',data=train,ax=ax[0,3],palette='husl')
sns.distplot(train['Fare'].dropna(),ax=ax[2,0],kde=False,color='b')
sns.countplot('Embarked',data=train,ax=ax[2,2])

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[1,0],palette='husl')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1,1],palette='husl')
sns.distplot(train[train['Survived']==0]['Age'].dropna(),ax=ax[1,2],kde=False,color='r',bins=5)
sns.distplot(train[train['Survived']==1]['Age'].dropna(),ax=ax[1,2],kde=False,color='g',bins=5)
sns.countplot('Parch',hue='Survived',data=train,ax=ax[1,3],palette='husl')
sns.swarmplot(x='Pclass',y='Fare',hue='Survived',data=train,palette='husl',ax=ax[2,1])
sns.countplot('Embarked',hue='Survived',data=train,ax=ax[2,3],palette='husl')

ax[0,0].set_title('Total Passengers by Class')
ax[0,1].set_title('Total Passengers by Gender')
ax[0,2].set_title('Age Box Plot By Class')
ax[0,3].set_title('Survival Rate by SibSp')
ax[1,0].set_title('Survival Rate by Class')
ax[1,1].set_title('Survival Rate by Gender')
ax[1,2].set_title('Survival Rate by Age')
ax[1,3].set_title('Survival Rate by Parch')
ax[2,0].set_title('Fare Distribution')
ax[2,1].set_title('Survival Rate by Fare and Pclass')
ax[2,2].set_title('Total Passengers by Embarked')
ax[2,3].set_title('Survival Rate by Embarked')
train['Cabin'].value_counts().head()
g = sns.FacetGrid(col='Embarked',data=train)
g.map(sns.pointplot,'Pclass','Survived','Sex',palette='viridis',hue_order=['male','female'])
g.add_legend()
f,ax = plt.subplots(1,2,figsize=(15,3))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])
# Set Figure Size
fig = plt.figure(figsize=(15,5))

# 1st Subplot
ax = fig.add_subplot(1, 2, 1)

# Box Plot for Age by Pclass for Train Data
ax.set_title('Train Dataset')

# Settings to display median values
box_plot_train = sns.boxplot(x='Pclass',y='Age',data=train)
ax_train = box_plot_train.axes
lines_train = ax_train.get_lines()
categories_train = ax_train.get_xticks()

for cat in categories_train:
    # Median line is the 4th line in a range of 6 lines:
    # 0: 25th percentile, 1: 75th percentile, 2: lower whisker, 3: upper whisker, 4: median, 5: upper extreme value
    y = round(lines_train[4+cat*6].get_ydata()[0],1) 

    ax_train.text(cat, y, f'{y}', ha='center', va='center', fontweight='bold',
                  size=10, color='white', bbox=dict(facecolor='#445A64'))

# 2nd Subplot
ax = fig.add_subplot(1, 2, 2)

# Box Plot for Age by Pclass for Test Data
ax.set_title('Test Dataset')

# Settings to display median values
box_plot_test = sns.boxplot(x='Pclass',y='Age',data=test)
ax_test = box_plot_test.axes
lines_test = ax_test.get_lines()
categories_test = ax_test.get_xticks()

for cat in categories_test:
    # Median line is the 4th line in a range of 6 lines:
    # 0: 25th percentile, 1: 75th percentile, 2: lower whisker, 3: upper whisker, 4: median, 5: upper extreme value
    y = round(lines_test[4+cat*6].get_ydata()[0],1) 

    ax_test.text(cat, y, f'{y}', ha='center', va='center', fontweight='bold',
                  size=10, color='white', bbox=dict(facecolor='#445A64'))

test.groupby('Pclass')['Age'].median()
# Histograms for Age

# Set Figure Size
fig = plt.figure(figsize=(15,5))

# 1st Subplot
ax = fig.add_subplot(1, 2, 1)

# Histogram for Age: Train Dataset
ax.set_title('Train Dataset')

sns.distplot(train['Age'].dropna(), kde=True, bins=5)

# 2nd Subplot
ax = fig.add_subplot(1, 2, 2)

# Histogram for Age: Test Dataset
ax.set_title('Test Dataset')

sns.distplot(test['Age'].dropna(), kde=True, bins=5)

def fill_age_train(cols):
    Age = cols[0]
    PClass = cols[1]
    
    if pd.isnull(Age):
        if PClass == 1:
            return 37
        elif PClass == 2:
            return 29
        else:
            return 24
    else:
        return Age

def fill_age_test(cols):
    Age = cols[0]
    PClass = cols[1]
    
    if pd.isnull(Age):
        if PClass == 1:
            return 42
        elif PClass == 2:
            return 26.5
        else:
            return 24
    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(fill_age_train,axis=1)
test['Age'] = test[['Age','Pclass']].apply(fill_age_test,axis=1)
test['Fare'].fillna(stat.mode(test['Fare']),inplace=True)
train['Embarked'].fillna('S',inplace=True)
train['Cabin'].fillna('No Cabin',inplace=True)
test['Cabin'].fillna('No Cabin',inplace=True)
f,ax = plt.subplots(1,2,figsize=(15,3))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])
train.drop('Ticket',axis=1,inplace=True)
test.drop('Ticket',axis=1,inplace=True)
train.head()
# Combine Dataset 1st for Feature Engineering

train['IsTrain'] = 1
test['IsTrain'] = 0
df = pd.concat([train,test])
# Scaler Initiation

scaler = MinMaxScaler()
df['Title'] = df['Name'].str.split(', ').str[1].str.split('.').str[0]
df['Title'].value_counts()
df['Title'].replace('Mme','Mrs',inplace=True)
df['Title'].replace(['Ms','Mlle'],'Miss',inplace=True)
df['Title'].replace(['Dr','Rev','Col','Major','Dona','Don','Sir','Lady','Jonkheer','Capt','the Countess'],'Others',inplace=True)
df['Title'].value_counts()
df.drop('Name',axis=1,inplace=True)
df.head()
sns.distplot(df['Age'],bins=5)
df['AgeGroup'] = df['Age']
df.loc[df['AgeGroup']<=19, 'AgeGroup'] = 0
df.loc[(df['AgeGroup']>19) & (df['AgeGroup']<=30), 'AgeGroup'] = 1
df.loc[(df['AgeGroup']>30) & (df['AgeGroup']<=45), 'AgeGroup'] = 2
df.loc[(df['AgeGroup']>45) & (df['AgeGroup']<=63), 'AgeGroup'] = 3
df.loc[df['AgeGroup']>63, 'AgeGroup'] = 4
sns.countplot(x='AgeGroup',hue='Survived',data=df[df['IsTrain']==1],palette='husl')
df.drop('Age',axis=1,inplace=True)
df.head()
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 #himself
df['IsAlone'] = 0
df.loc[df['FamilySize']==1, 'IsAlone'] = 1
# Visual Inspection of Survival Rates

f,ax = plt.subplots(1,2,figsize=(15,6))
sns.countplot(df[df['IsTrain']==1]['FamilySize'],hue=train['Survived'],ax=ax[0],palette='husl')
sns.countplot(df[df['IsTrain']==1]['IsAlone'],hue=train['Survived'],ax=ax[1],palette='husl')

df.drop(['SibSp','Parch','FamilySize'],axis=1,inplace=True)
df.head()
df.head()
df['Deck'] = df['Cabin']
df.loc[df['Deck']!='No Cabin','Deck'] = df[df['Cabin']!='No Cabin']['Cabin'].str.split().apply(lambda x: np.sort(x)).str[0].str[0]
df.loc[df['Deck']=='No Cabin','Deck'] = 'N/A'
sns.countplot(x='Deck',hue='Survived',data=df[df['IsTrain']==1],palette='husl')
df.loc[df['Deck']=='N/A', 'Deck'] = 0
df.loc[df['Deck']=='G', 'Deck'] = 1
df.loc[df['Deck']=='F', 'Deck'] = 2
df.loc[df['Deck']=='E', 'Deck'] = 3
df.loc[df['Deck']=='D', 'Deck'] = 4
df.loc[df['Deck']=='C', 'Deck'] = 5
df.loc[df['Deck']=='B', 'Deck'] = 6
df.loc[df['Deck']=='A', 'Deck'] = 7
df.loc[df['Deck']=='T', 'Deck'] = 0
df.drop('Cabin',axis=1,inplace=True)
df.head()
df[['Fare','Pclass','Deck']] = scaler.fit_transform(df[['Fare','Pclass','Deck']])
df.head()
def process_dummies(df,cols):
    for col in cols:
        dummies = pd.get_dummies(df[col],prefix=col,drop_first=True)
        df = pd.concat([df.drop(col,axis=1),dummies],axis=1)
    return df
df = process_dummies(df,['Embarked','Sex','Title','AgeGroup'])
df.head()
dataset = df[df['IsTrain']==1]
dataset.drop(['IsTrain','PassengerId'],axis=1,inplace=True)
holdout = df[df['IsTrain']==0]
test_id = holdout['PassengerId']
holdout.drop(['IsTrain','PassengerId','Survived'],axis=1,inplace=True)
int(np.sum(dataset['Survived'])), dataset.shape[0]
df.to_csv('titanic_dataset_preprocessed.csv',index=False)
X = dataset.drop(['Survived'],axis=1)
y = dataset['Survived'].astype('int')
#X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=101)
%%time

model = RandomForestClassifier()
splits = 5
kf = KFold(n_splits=splits,shuffle=True,random_state=101)
score = 0
train_indices, validation_indices = [],[]
total_score = []
for curr_train_indices, curr_validation_indices in kf.split(X):
    result = model.fit(X.iloc[curr_train_indices], y.iloc[curr_train_indices])
    curr_score = result.score(X.iloc[curr_validation_indices],y.iloc[curr_validation_indices])
    print(curr_score)
    total_score.append(curr_score)
        
    if(curr_score > score):
        score = curr_score
        train_indices = curr_train_indices
        validation_indices = curr_validation_indices
print('Best Score: ',score)
print('Average Score: ', sum(total_score)/splits)

%%time

param_grid = [{'n_estimators':[10,20,50,100,200], 'max_depth':[1,2,3,4,5,7,10,None], 'max_features':[3,5,7,9,10,'auto']}]
grid = GridSearchCV(model, param_grid, n_jobs=-1) # n_jobs = -1 to use all processors
grid.fit(X, y)
grid.best_params_, grid.best_score_
predictions = grid.predict(holdout)
fullpredictions = grid.predict(X)
len(fullpredictions)
# Confusion Matrix

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y, fullpredictions)
print(conf_mat)
# Accuracy, Recall and Precision

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

accuracy = accuracy_score(y, fullpredictions)
recall = recall_score(y, fullpredictions)
precision = precision_score(y, fullpredictions)

print('Accuracy: ', '{:.0%}'.format(accuracy))
print('Recall: ', '{:.0%}'.format(recall))
print('Precision: ', '{:.0%}'.format(precision))
# Model

forest = RandomForestClassifier(max_depth=7, max_features=10, n_estimators=20, random_state=101)
forest.fit(X, y)

# Feature DataFrame

list(X.columns)
# Feature Importance (Raw)

forest.feature_importances_.tolist()
# Define Function for Random Forest Feature Importance

def RF_FeatureImportance(df, model):
    """
    df: input DataFrame with features, excluding column to be predicted
    model: input fitted RandomForest model
    """
    importancedf = pd.DataFrame(list(zip(df.columns.tolist(), model.feature_importances_.tolist())),
                         columns=['Features','Importance'])
    importancedf.sort_values(by=['Importance'], inplace=True, ascending=False)
    importancedf['Importance'] = importancedf['Importance'].map(lambda x: '{:.0%}'.format(x))
    return importancedf

# Random Forest Feature Importance

RF_FeatureImportance(X,forest)

# Check Correlation Heatmap for Feature Importance Analysis

cut_off = 0.3
corr_mat = X.corr() # Insert DataFrame X
corr_mat[(corr_mat > -cut_off) & (corr_mat < cut_off) ] = np.nan # Use NaN instead of 0 so that they will not be shown in the heatmap
plt.figure(figsize=(25.6,16))

sns.set_style(style = 'whitegrid')
mask = np.zeros_like(corr_mat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(h_neg=240, h_pos=10, s=80, l=55, n=9, 
                             center='light', as_cmap=True)
sns.heatmap(corr_mat, annot=True, cmap='magma',
            mask=mask, fmt='.2f', linewidths=0.5,
            square=True, vmin=-1, vmax=1, 
            annot_kws={"size": 15});

RF_FeatureImportance(X,forest)
# Spearman's Correlation (Raw)

from scipy.stats import spearmanr

spearman, pvalue = spearmanr(X)
spearman

# Display Spearman's Correlations in DataFrame

spearmandf = pd.DataFrame(spearman, columns=X.columns, index=X.columns)
spearmandf

# Spearman's Correlation Heatmap

cut_off = 0.3
corr_mat = spearmandf # Insert DataFrame for Spearman's Correlation, from above
corr_mat[(corr_mat > -cut_off) & (corr_mat < cut_off) ] = np.nan # Use NaN instead of 0 so that they will not be shown in the heatmap
plt.figure(figsize=(25.6,16))

sns.set_style(style = 'whitegrid')
mask = np.zeros_like(corr_mat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(h_neg=240, h_pos=10, s=80, l=55, n=9, 
                             center='light', as_cmap=True)
sns.heatmap(corr_mat, annot=True, cmap='coolwarm',
            mask=mask, fmt='.2f', linewidths=0.5,
            square=True, vmin=-1, vmax=1, 
            annot_kws={"size": 15});

# Define Function for Random Forest Tree Visualization

def RF_TreeViz(df, y, model, n):
    """
    df: input DataFrame with features, excluding column to be predicted
    y: input Series with target variable/ classification which is being predicted
    model: input fitted RandomForest model
    n: tree number in the Forest
    """
    
    # Extract one tree, n
    estimator = model.estimators_[n]

    from sklearn.tree import export_graphviz

    # Export tree as dot file
    export_graphviz(estimator, out_file='tree.dot', feature_names = list(df.columns), class_names = list(y.name),
                    rounded = True, proportion = False, precision = 2, filled = True)

    # Convert dot file to png using system command (requires graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    # Display in Jupyter Notebook
    from IPython.display import Image, display
    display(Image(filename = 'tree.png'))

# Visualize A Random Forest Tree

RF_TreeViz(X, y, forest, 10) # Recall that forest is our fitted RandomForestClassifier model and here, I randomly selected tree # 10

submission = pd.DataFrame({
    'PassengerId': test_id,
    'Survived': predictions
})

submission.to_csv('submission.csv',index=False)
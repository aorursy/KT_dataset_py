# NumPy
import numpy as np

# Dataframe operations
import pandas as pd

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_view = pd.read_csv("../input/train.csv")
train_view.head()
train_view.info()
train_view['Sex'].value_counts().plot.pie()
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal') #most sponsored projects are teacher-led
plt.title("gender distribution", fontsize=20)
sns.violinplot(data = train_view, x = "Sex", y = "Survived")
sns.distplot(train_view['Age'].dropna()).set_title('Age distribution',fontsize=20)
g = sns.FacetGrid(train, col='Sex')
g = g.map(sns.distplot, "Age")
g = sns.FacetGrid(train_view, col='Survived')
g = g.map(sns.distplot, "Age")
train_view['Age'].loc[train_view['Age'] <= 15] = 1 
train_view['Age'].loc[(train_view['Age'] > 15) & (train_view['Age'] <= 25)] = 2
train_view['Age'].loc[(train_view['Age'] > 25) & (train_view['Age'] <= 35)] = 3
train_view['Age'].loc[(train_view['Age'] > 35) & (train_view['Age'] <= 48)] = 4
train_view['Age'].loc[(train_view['Age'] > 48) & (train_view['Age'] <= 65)] = 5
train_view['Age'].loc[(train_view['Age'] > 65) & (train_view['Age'] <= 75)] = 6
train_view['Age'].loc[train_view['Age'] > 75] = 7
sns.violinplot(data = train_view, x = "Age", y = "Survived")
train_view['Pclass'].value_counts().plot.pie()
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal') #most sponsored projects are teacher-led
plt.title("class distribution", fontsize=20)
sns.violinplot(data = train_view, x = "Pclass", y = "Survived")
sns.boxplot(data = train_view, x = "Pclass", y = "Age");
plt.figure()
sns.boxplot(data = train_view, x = "Pclass", y = "Fare");
plt.figure()
train_view['Embarked'].value_counts().plot.pie()
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal') #most sponsored projects are teacher-led
plt.title("embark distribution", fontsize=20)
sns.violinplot(data = train_view, x = "Embarked", y = "Survived")
sns.boxplot(data = train_view, x = "Embarked", y = "Age");
plt.figure()
sns.boxplot(data = train_view, x = "Embarked", y = "Fare");
plt.figure()
g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
sns.jointplot(x="SibSp", y="Age", data=train_view, kind="kde", color='r');
plt.figure()
sns.jointplot(x="SibSp", y="Fare", data=train_view, kind="kde", color='r');
plt.figure()
g = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
sns.jointplot(x="Parch", y="Age", data=train_view, kind="kde", color='r');
plt.figure()
sns.jointplot(x="Parch", y="Fare", data=train_view, kind="kde", color='r');
plt.figure()
sns.distplot(train_view['Fare'].dropna()).set_title('Age distribution(1)',fontsize=20)
train_view[train_view['Fare']<100]['Fare'].plot.hist().set_title('fare distribution(2)',fontsize=20)
train_view['Fare'].loc[train_view['Fare'] <= 14] = 1 
train_view['Fare'].loc[(train_view['Fare'] > 14) & (train_view['Fare'] <= 20)] = 2
train_view['Fare'].loc[(train_view['Fare'] > 20) & (train_view['Fare'] <= 40)] = 3
train_view['Fare'].loc[(train_view['Fare'] > 40) & (train_view['Fare'] <= 60)] = 4
train_view['Fare'].loc[(train_view['Fare'] > 60) & (train_view['Fare'] <= 80)] = 5
train_view['Fare'].loc[(train_view['Fare'] > 80) & (train_view['Fare'] <= 100)] = 6
train_view['Fare'].loc[train_view['Fare'] > 100] = 7
sns.violinplot(data = train_view, x = "Fare", y = "Survived")
train_view['Title'] = train_view['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
train_view['Title'].value_counts().plot.bar()
train_view["Title"] = train_view["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_view["Title"] = train_view["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
train_view["Title"] = train_view["Title"].astype(int)
sns.violinplot(data = train_view, x = "Title", y = "Survived")
sns.boxplot(data = train_view, x = "Title", y = "Fare")
sns.boxplot(data = train_view, x = "Title", y = "Age")
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train_view,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="Embarked", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="SibSp", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g = sns.factorplot(x="Parch", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
h = sns.FacetGrid(train_view, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()
h = sns.FacetGrid(train_view, row = 'Embarked', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()
h = sns.FacetGrid(train_view, row = 'Sex', col = 'Embarked', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()
h = sns.FacetGrid(train_view, row = 'Embarked', col = 'Title', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()
h = sns.FacetGrid(train_view, row = 'Pclass', col = 'Title', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()
grid = sns.FacetGrid(train_view, col = "Embarked", row = "Title", hue = "Survived", palette = 'seismic')
grid = grid.map(plt.scatter, "PassengerId", "Fare")
grid.add_legend()
grid
grid = sns.FacetGrid(train_view, col = "Sex", row = "Title", hue = "Survived", palette = 'seismic')
grid = grid.map(plt.scatter, "PassengerId", "Fare")
grid.add_legend()
grid
grid = sns.FacetGrid(train_view, col = "Pclass", row = "Title", hue = "Survived", palette = 'seismic')
grid = grid.map(plt.scatter, "PassengerId", "Fare")
grid.add_legend()
grid
grid = sns.FacetGrid(train_view, col = "Embarked", row = "Pclass", hue = "Survived", palette = 'seismic')
grid = grid.map(plt.scatter, "PassengerId", "Fare")
grid.add_legend()
grid
sns.pairplot(train_view[['Embarked', 'Pclass', 'Title','Survived']])
corrmat = train_view.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
data_df = train_df.append(test_df)
data_df['Title'] = data_df['Name']
# Cleaning name and extracting Title
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# Replacing rare titles with more common ones
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
    
# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]

# Dropping Title feature
data_df.drop('Title', axis = 1, inplace = True)
data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)

# Making Bins
data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)

label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])

train_df['FareBin_Code'] = data_df['FareBin_Code'][:891]
test_df['FareBin_Code'] = data_df['FareBin_Code'][891:]

train_df.drop(['Fare'], 1, inplace=True)
test_df.drop(['Fare'], 1, inplace=True)
data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)

label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])

train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:891]
test_df['AgeBin_Code'] = data_df['AgeBin_Code'][891:]

train_df.drop(['Age'], 1, inplace=True)
test_df.drop(['Age'], 1, inplace=True)
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']

# Substituting Age values in TRAIN_DF and TEST_DF:
train_df['Family_Size'] = data_df['Family_Size'][:891]
test_df['Family_Size'] = data_df['Family_Size'][891:]
data_df["Alone"] = np.where(data_df['SibSp'] + data_df['Parch'] + 1 == 1, 1,0) # People travelling alone
data_df["Alone"] = np.where(data_df['SibSp'] + data_df['Parch'] + 1 == 1, 1,0) # People travelling alone

train_df['Alone'] = data_df['Alone'][:891]
test_df['Alone'] = data_df['Alone'][891:]
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])
for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))

# # Family_Survival in TRAIN_DF and TEST_DF:
train_df['Family_Survival'] = data_df['Family_Survival'][:891]
test_df['Family_Survival'] = data_df['Family_Survival'][891:]
label = LabelEncoder()
data_df['Embarked'] = data_df['Embarked'].fillna('None')
data_df['Embarked_code'] = label.fit_transform(data_df['Embarked'])

train_df['Embarked_code'] = data_df['Embarked_code'][:891]
test_df['Embarked_code'] = data_df['Embarked_code'][891:]
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)
train_df.head()
train_df = train_df.drop(columns=['PassengerId','Pclass','Name','SibSp','Parch','Ticket','Cabin','Embarked'])
test_df = test_df.drop(columns=['PassengerId','Pclass','Name','SibSp','Parch','Ticket','Cabin','Embarked'])
train_df = train_df.replace(np.inf, np.nan) # important
train_df_na = (train_df.isnull().sum() / len(train_df))
train_df_na = train_df_na.drop(train_df_na[train_df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :train_df_na})
missing_data.head()
test_df = test_df.replace(np.inf, np.nan) # important
test_df_na = (test_df.isnull().sum() / len(test_df))
test_df_na = test_df_na.drop(test_df_na[test_df_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :test_df_na})
missing_data.head()
X_train = train_df.drop('Survived', 1)
Y_train = train_df['Survived']
X_test = test_df.copy()
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
xgb = XGBClassifier(booster='gbtree',
                    gamma=0.1,
                    learning_rate = 0.01,
                    max_depth = 6,
                    min_child_weight= 1,
                    n_estimators= 100,
                    subsample= 0.5)
rf=RandomForestClassifier(random_state=42, 
                          min_samples_split=2, 
                          max_leaf_nodes=10,
                          max_features='auto', 
                          n_estimators= 500, 
                          max_depth=5, 
                          criterion='gini')
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(random_state=40,
                         algorithm = 'SAMME.R',
                         learning_rate= 0.01,
                         n_estimators = 200
                        )
dt = DecisionTreeClassifier(random_state=40,
                            criterion = 'gini',
                            max_depth= 5,
                            max_features= 'auto',
                            max_leaf_nodes= 12,
                            min_samples_split= 2)
knn = KNeighborsClassifier(algorithm='auto',
                     leaf_size=26, 
                     metric='minkowski', 
                     metric_params=None, 
                     n_jobs=-1, 
                     n_neighbors=18, 
                     p=2, 
                     weights='uniform')
from sklearn.ensemble import VotingClassifier
optimal = VotingClassifier(estimators=[('knn',knn),('ada',ada),('rf',rf),('xgb',xgb),('dt',dt)],voting='hard')
optimal
optimal.fit(X_train, Y_train)
Y_pred=optimal.predict(X_test)
Y_pred
len(Y_pred)
sub = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
sub['Survived'] = Y_pred
sub.to_csv("../working/submission.csv", index = False)
sub.head()

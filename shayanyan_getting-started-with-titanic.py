# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("Setup Complete")
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
full_train = pd.read_csv("../input/titanic/train.csv")
#full_train.head()
full_train.describe()
full_test = pd.read_csv("../input/titanic/test.csv")
#full_test.head()
full_test.describe()
## merge data for features explorations
full_train["DataType"]=1
full_test["DataType"]=2
full_data = full_train.append(full_test)
full_data.describe()
full_data.describe(include=["object"])
full_data.info()
clean_full_data = full_data.copy()

label_encoder = LabelEncoder()
clean_full_data["Sex"] = label_encoder.fit_transform(clean_full_data["Sex"])
label_encoder = LabelEncoder()
clean_full_data["Embarked_backup"] = label_data["Embarked"]
clean_full_data["Embarked"] = label_encoder.fit_transform(clean_full_data["Embarked"].astype(str))

#Embarked 0-C / 1-Q / 3-S

label_data.head()
clean_full_data[["Age", "Pclass", "Sex", "SibSp", "Parch", "Embarked", "Fare"]].corr()
## Try to extract salutation.
clean_full_data[["Salutation"]]=pd.DataFrame(clean_full_data.apply(lambda x: x.Name.split(",")[1].split(".")[0], axis=1),columns=["Salutation"])
#print(label_data.groupby(['Salutation'])['Age'].agg(['count','mean', 'median', 'min', 'max']).round(1))
#print(label_data.groupby(['Pclass', 'Salutation'])['Age'].agg(['count','mean', 'median', 'min', 'max']).round(1))
#print(label_data.groupby(['Pclass', 'Sex'])['Age'].agg(['count','mean', 'median', 'min', 'max']).round(1))
#print(label_data.groupby(['Sex', 'Salutation']).size())
fillna_data=clean_full_data.groupby(['Pclass', 'Salutation'])['Age'].agg(['mean'])
fillna_data
fillna_data2=label_data.groupby(['Pclass', 'Sex'])['Age'].agg(['mean'])
fillna_data2
clean_full_data[["Age_backup"]]=label_data[["Age"]]
clean_full_data2 = pd.merge(left=clean_full_data, right=fillna_data, how='left', left_on=['Pclass', 'Salutation'], right_on=['Pclass', 'Salutation'])

clean_full_data2["Age"] = clean_full_data2.apply(
    lambda row: row['mean'] if np.isnan(row['Age']) else row['Age'],
    axis=1
)
clean_full_data2["Age"] = clean_full_data2.apply(
    lambda row: 22.185329 if np.isnan(row['Age']) else row['Age'],
    axis=1
)


clean_full_data2.describe()
clean_full_data.groupby(['Pclass'])['Fare'].agg(['mean'])
clean_full_data2["Fare"] = clean_full_data2.apply(
    lambda row: 13.302889 if np.isnan(row['Fare']) else row['Fare'],
    axis=1
)
clean_full_data2.describe()
#label_data2.tail(10)
#Get Final Data
final_full_train=clean_full_data2.loc[clean_full_data2['DataType'] == 1]
final_full_test=clean_full_data2.loc[clean_full_data2['DataType'] == 2]

#features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]
#y = final_full_train.Survived
#X = final_full_train[features]
#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


final_full_train[["Survived", "Age", "Pclass", "Sex", "SibSp", "Parch", "Embarked", "Fare"]].corr()
#data_check = final_full_train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare"]]
#data_check[data_check.isnull().any(axis=1)]
final_full_train["cntPartners"] = final_full_train.SibSp + final_full_train.Parch
final_full_train["Age_group"] = final_full_train.Age.round(-1)
final_full_train.describe()
features_all = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare", "cntPartners", "Age_group"]

y = final_full_train.Survived
X = final_full_train[features_all]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

def model_trial(features_trial, model_a, title):
    i = 1
    #i_best = 0
    
    for feature in features_trial:
        model_a.fit(train_X[feature], train_y)
        #print(train_X[feature])
        val_predictions = model_a.predict(val_X[feature])
        val_mae = mean_absolute_error(val_predictions, val_y)
        
        print(i, "Validation MAE for ", title, " Model: ",  (val_mae))
        
        if i == 1:
            i_best = 1
            val_mae_best = val_mae
        elif val_mae < val_mae_best:
            i_best = i
            val_mae_best = val_mae
            
        i += 1
        
    print("Best feature is ", i_best)   
    
    return i_best
features_trial = []
features_trial.append(["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare"])
features_trial.append(["Pclass", "Sex", "Age", "cntPartners", "Embarked", "Fare"])
features_trial.append(["Pclass", "Sex", "Age_group", "cntPartners", "Embarked", "Fare"])
features_trial.append(["Pclass", "Sex", "Age", "cntPartners", "Embarked"])
features_trial.append(["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"])
features_trial.append(["Pclass", "Sex", "Age", "cntPartners", "Embarked"])
features_trial.append(["Pclass",  "Age_group", "cntPartners"])
features_trial.append(["Pclass", "Sex", "Age", "SibSp", "Embarked", "Fare"])
model_trial(features_trial, DecisionTreeRegressor(random_state=0), "Decision Tree")
model_trial(features_trial, RandomForestRegressor(random_state=0), "Random Forest")
model_trial(features_trial, RandomForestClassifier(random_state=0), "Random Forest")
model_trial(features_trial, RadiusNeighborsClassifier(radius=30), "KNN")
model_trial(features_trial, LogisticRegression(random_state=0), "Logistic Regression")
final_feature = features_trial[2]
final_model = RandomForestClassifier(random_state=0)
final_model.fit(X[final_feature], y)
final_val_predictions = final_model.predict(X[final_feature])
final_val_mae = mean_absolute_error(final_val_predictions, y)
        
print("Validation MAE for Final Model: ",  final_val_mae)
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X[final_feature], train_y)
    preds_val = model.predict(val_X[final_feature])
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
for max_leaf_nodes in [5, 48, 49, 50, 51, 52, 53, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: ", max_leaf_nodes," Mean Absolute Error:  ", my_mae)
my_maes=[]
for max_leaf_nodes in range(2,500,10):
    my_mae = get_mae(max_leaf_nodes + 1, train_X, val_X, train_y, val_y)
    my_maes.append(my_mae)
    
#sns.lineplot(x=my_maes, y=range(2,5000,10))
sns.lineplot(x=range(2,500,10), y=my_maes)
sns.lineplot(x=range(2,100,10), y=my_maes[:10])
my_maes=[]
for max_leaf_nodes in range(50,90):
    my_mae = get_mae(max_leaf_nodes + 1, train_X, val_X, train_y, val_y)
    my_maes.append(my_mae)
    
sns.lineplot(x=range(50,90), y=my_maes)
final_feature = features_trial[2]
final_model = RandomForestClassifier(max_leaf_nodes=84, random_state=0)
final_model.fit(X[final_feature], y)
final_val_predictions = final_model.predict(X[final_feature])
final_val_mae = mean_absolute_error(final_val_predictions, y)
        
print("Validation MAE for Final Model: ",  final_val_mae)
final_full_test["cntPartners"] = final_full_test.SibSp + final_full_test.Parch
final_full_test["Age_group"] = final_full_test.Age.round(-1)

predictions = final_model.predict(final_full_test[final_feature]).astype(int)
output = pd.DataFrame({'PassengerId': final_full_test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
print(full_train.groupby(['Pclass'])['Survived','Age'].agg(['mean', 'median']).round(1))
print(full_train.groupby(['Sex', 'Pclass'])['Survived','Age'].agg(['mean', 'median']).round(1))

sns.lineplot(x=full_train['Age'].round(), y=full_train['Survived'])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
sns.barplot(x=full_train['SibSp'].round() + full_train['Parch'].round(), y=full_train['Survived'])
sns.barplot(x=full_train['SibSp'].round() + full_train['Parch'].round(), y=full_train['Survived'])
sns.distplot(full_train['Age'])
full_train['Age'].round(-1).value_counts()
#plot_age = sns.kdeplot(full_train['Age'].loc[full_train['Survived']==1], label = 'Suvived').set_xlabel('Age')
#plot_age = sns.kdeplot(full_train['Age'].loc[full_train['Survived']==0], label = 'Not Suvived')
sns.barplot(x=full_train['Age'].round(-1), y=full_train['Survived'])
sns.barplot(x=full_train['Age'].round(-1), y=full_train['Survived'])

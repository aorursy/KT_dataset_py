import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
eda = pd.read_excel('Titanic.xlsx')
eda.head()
eda.info()
eda.describe()
sns.countplot(x='Sex',data=eda)
sns.countplot(x='Survived',data=eda)
sns.countplot(x='Pclass',data=eda)
sns.countplot(x='Parch',data=eda)
sns.countplot(x='Embarked',data=eda)
sns.distplot(eda['Age'],bins=4)
sns.distplot(eda['SibSp'],bins=4)
sns.boxplot(x='Fare',data=eda)
sns.distplot(eda['Fare'],bins=4)
sns.pairplot(eda)
sns.heatmap(eda.corr(), annot=True, linewidth=0.5)
counts = eda.groupby(['Survived', 'Pclass'], axis= 0)
counts.size()
counts = eda.groupby(['Survived', 'Sex'], axis= 0)
counts.size()
counts = eda.groupby(['Survived', 'Embarked'], axis= 0)
counts.size()
eda.isnull()
eda.isnull().sum()
cabin_only = eda[["Cabin"]].copy()
cabin_only["Cabin_Data"] = cabin_only["Cabin"].isnull().apply(lambda x: not x) 
cabin_only["Deck"] = cabin_only["Cabin"].str.slice(0,1)
cabin_only["Room"] = cabin_only["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
cabin_only[cabin_only["Cabin_Data"]]
cabin_only
cabin_only.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")
cabin_only["Deck"] = cabin_only["Deck"].fillna("N") # assign 'N' for the deck name of the null Cabin value. 
cabin_only["Room"] = cabin_only["Room"].fillna(cabin_only["Room"].mean()) # use mean to fill null Room values.
cabin_only
cabin_only=cabin_only.join(pd.get_dummies(cabin_only['Deck'], prefix='Deck'))
cabin_only=cabin_only.drop(['Deck'], axis=1)
cabin_only
eda=pd.concat([eda,cabin_only],axis=1)
eda.shape
cabin_only_test = eda[["Cabin"]].copy()
cabin_only_test["Cabin_Data"] = cabin_only_test["Cabin"].isnull().apply(lambda x: not x) # extract rows that do not contain null Cabin data.
cabin_only_test["Deck"] = cabin_only_test["Cabin"].str.slice(0,1)
cabin_only_test["Room"] = cabin_only_test["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
cabin_only_test[cabin_only_test["Cabin_Data"]]
cabin_only_test.drop(["Cabin", "Cabin_Data"], axis=1, inplace=True, errors="ignore")
cabin_only_test["Deck"] = cabin_only_test["Deck"].fillna("N") # assign 'N' for the deck name of the null Cabin value. 
cabin_only_test["Room"] = cabin_only_test["Room"].fillna(cabin_only_test["Room"].mean()) # use mean to fill null Room values.
cabin_only_test=cabin_only_test.join(pd.get_dummies(cabin_only_test['Deck'], prefix='Deck'))
cabin_only_test=cabin_only_test.drop(['Deck'], axis=1)
eda=pd.concat([eda,cabin_only_test],axis=1)
eda.head()
eda['Fare'].describe() 
eda.boxplot(column = ['Fare'])
IQR = eda['Fare'].quantile(0.75)-eda['Fare'].quantile(0.25)
print(IQR)
Upper_OutlierLimit = eda['Fare'].quantile(0.75) + 1.5*IQR
Lower_OutlierLimit = eda['Fare'].quantile(0.25) - 1.5*IQR
print(Upper_OutlierLimit)
print(Lower_OutlierLimit)
OutlierValues = eda[(eda['Fare']>=Upper_OutlierLimit)|(eda['Fare']<=Lower_OutlierLimit)]
OutlierValues
print(eda['Fare'].quantile(0.90))
eda["Fare"]=np.where(eda["Fare"] >31, 31,eda['Fare'])
sns.distplot(eda['Fare'],bins=4)
eda.boxplot(column = ['Fare'])
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
train=eda
test=pd.read_excel('Titanic.xlsx')
submission=eda
train.head()
train.Parch.value_counts()
train['Sex'] = train['Sex'].apply(lambda x : 1 if x=='male' else 0)
test['Sex'] = test['Sex'].apply(lambda x : 1 if x=='male' else 0)
train['Embarked'] = train['Embarked'].map({'S':1, 'C':2, 'Q':3})
test['Embarked'] = test['Embarked'].map({'S':1, 'C':2, 'Q':3})
my_imputer = SimpleImputer()

features = ['Pclass','Sex','Age','Parch', 'Fare', 'Embarked']

imputed_train = pd.DataFrame(my_imputer.fit_transform(train[features]))
imputed_test = pd.DataFrame(my_imputer.fit_transform(test[features]))

y = train.Survived

X = imputed_train.copy()
X_test = imputed_test.copy()
train_X, val_X, train_y, val_y = train_test_split(imputed_train, y, train_size=0.8, test_size=0.2, random_state=0)
model_1 = RandomForestRegressor(n_estimators = 50, random_state = 0)
model_2 = RandomForestRegressor(n_estimators = 100,criterion = 'mae', random_state = 0)
model_3 = RandomForestRegressor(n_estimators = 100, min_samples_split = 20, random_state = 0)
model_4 = RandomForestRegressor(n_estimators = 200, min_samples_split = 20, random_state = 0)
model_5 = RandomForestRegressor(n_estimators = 100, max_depth = 7, random_state = 0)
def score_model(model, X_train = train_X, X_val = val_X, y_train = train_y, y_val = val_y):
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    return(mean_absolute_error(y_val, predictions))

models = [model_1,model_2,model_3,model_4,model_5]

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("model %d MAE : %d"%(i+1, mae))
best_model = model_2
my_model = best_model
my_model.fit(imputed_train, y)

predict = my_model.predict(imputed_test)

submission['Survived'] = predict
submission.to_csv('submission.csv', index = False)
import os
import re
import pprint
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import svm
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sb
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
pd.set_option('display.width', 1000)
sb.set_style('whitegrid')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train_df.info())
print(train_df.describe())
train_df.isnull().sum()
test_df.isnull().sum()
print("Total Alives: %3d" % len(train_df[train_df['Survived'] == 1]))
print("Total Deads : %3d" % len(train_df[train_df['Survived'] == 0]))
sb.swarmplot(x = "Survived", y = "Age", data = train_df)
plt.show()
alives1 = len(train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 1)])
deads1 = len(train_df[(train_df['Sex'] == 'male') & (train_df['Survived'] == 0)])
alives2 = len(train_df[(train_df['Sex'] == 'female') & (train_df['Survived'] == 1)])
deads2 = len(train_df[(train_df['Sex'] == 'female') & (train_df['Survived'] == 0)])

print("Male   Alive: %3d, Dead: %3d ===> %0.4f" % (alives1, deads1, alives1 / (alives1 + deads1)))
print("Female Alive: %3d, Dead: %3d ===> %0.4f" % (alives2, deads2, alives2 / (alives2 + deads2)))

sb.swarmplot(x = "Sex", y = "Age", hue='Survived', data = train_df)
plt.show()
for val in train_df['SibSp'].unique():
    alives = len(train_df[(train_df['SibSp'] == val) & (train_df['Survived'] == 1)])
    deads = len(train_df[(train_df['SibSp'] == val) & (train_df['Survived'] == 0)])
    print("SibSp: [%d] Alive: %3d, Dead: %3d ===> %0.4f" % (val, alives, deads, alives / (alives + deads)))
for val in train_df['Parch'].unique():
    alives = len(train_df[(train_df['Parch'] == val) & (train_df['Survived'] == 1)])
    deads = len(train_df[(train_df['Parch'] == val) & (train_df['Survived'] == 0)])
    print("Parch: [%d] Alive: %3d, Dead: %3d ===> %0.4f" % (val, alives, deads, alives / (alives + deads)))
sb.swarmplot(x = "Survived", y = "Fare", data = train_df)
plt.show()
sb.factorplot("Survived", col = "Pclass", col_wrap = 3, data = train_df, kind = "count")
plt.show()
sb.swarmplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = train_df)
plt.show()
sb.factorplot("Survived", col = "Embarked", col_wrap = 3, data = train_df, kind = "count")
plt.show()
df = train_df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
corr = df.corr()   
mask = np.triu(np.ones_like(corr, dtype=np.bool))    
plt.figure(figsize=(14, 10))   
sb.heatmap(corr, mask=mask, cmap='RdBu_r', annot=True, linewidths=0.5, fmt='0.2f')
plt.show()
print(train_df[train_df['Embarked'].isna() == True])
df = train_df.copy()
df.loc[df['Embarked'].isna() == True, 'Embarked'] = 'X'  ## <-- missing value indicator
print("Total number of first class passengers whose pay about 80.00 fare")
for embarked in [ 'C', 'S', 'Q', 'X' ]:
    print("Embarked: [%s] ===> %d" % (embarked, len(df[
                             (df['Embarked'] == embarked) &
                             (df['Pclass'] == 1) &
                             (df['Fare'] >= 78) &
                             (df['Fare'] < 82)])))
train_df.loc[train_df['Embarked'].isna() == True, 'Embarked'] = 'S'
print(test_df[test_df['Fare'].isna() == True])
print(test_df.loc[test_df['Pclass'] == 3, 'Fare'].median())
test_df.loc[test_df['Fare'].isna() == True, 'Fare'] = 7.89
def preprocessingTitle(rec):
    name = rec['Name']
    sex = rec['Sex']

    name = re.search('[a-zA-Z]+\\.', name).group(0)
    name = name.replace(".", "")
    if name == 'Col' or name == 'Capt' or name  == 'Major':
        name = 0
    elif name == 'Rev' and sex == 'male':
        return 1
    elif name == 'Dr' and sex == 'female':
        name = 2
    elif name == 'Dr' and sex == 'male':
        return 3
    elif name == 'Sir' or name == 'Don':
        name = 4
    elif name == 'Mme' or name == 'Mrs':
        name = 5
    elif name == 'Lady' or name == 'Countess' or name == 'Dona':
        return 6
    elif name == 'Miss' or name == 'Ms' or name == 'Mlle' or name == 'Jonkheer':
        return 7
    elif name == 'Master':
        return 8
    elif name == 'Mr':
        return 9
        
    
    return name

train_df['Title'] = train_df.apply(preprocessingTitle, axis = 1)
test_df['Title'] = test_df.apply(preprocessingTitle, axis = 1)
def fillValues(name, *args):
    key, columns, label = ['PassengerId'], ['Title', 'Sex', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Embarked' ], [name]
    
    alldf = []
    for df in args:
        alldf.append(df)
        
    all_df = pd.concat(alldf)
    all_df = all_df[key + columns + label]
    
    all_df = pd.get_dummies(all_df, columns = ['Title', 'Sex', 'Pclass', 'Embarked'])
    
    cols = set(all_df.columns)
    cols.remove(name)
    

    all_df_in = all_df.loc[all_df[name].isna() == False, cols]
    all_df_lb = all_df.loc[all_df[name].isna() == False, label]

    model = ExtraTreesRegressor(random_state = 0)
    model.fit(all_df_in, all_df_lb)
    
    
    all_df_im = all_df.loc[all_df[name].isna() == True, cols]
       
    preds = model.predict(all_df_im)
    all_df_im[name] = preds
    
    
    for df in args:
        df.loc[df[name].isna() == True, name] = all_df_im.loc[all_df_im['PassengerId'].isin(df['PassengerId']), name]
        df[name] = df[name].astype('int64')
def postrocessingTitle(rec):
    title = rec['Title']
    sex = rec['Sex']
    age = rec['Age']
    
    if sex == 'male' and age < 16:
        return 10
    elif sex == 'female' and age < 16:
        return 11
    elif age >= 55 and sex == 'male':
        return 12
    elif age >= 55 and sex == 'female':
        return 13
    
fillValues('Age', train_df, test_df)

train_df['Title'] = train_df.apply(preprocessingTitle, axis = 1)
test_df['Title'] = test_df.apply(preprocessingTitle, axis = 1)
def captureCabinPrefix(val):
    
    if str(val) != 'nan':
        x = re.findall("[a-zA-Z]{1}", val)
        if len(x) == 0:
            x = ['X']
            
        return x[0][0]
        
    return val


train_df['CabinPrefix'] = train_df['Cabin'].apply(captureCabinPrefix)
test_df['CabinPrefix'] = test_df['Cabin'].apply(captureCabinPrefix)
print(train_df['CabinPrefix'].unique())
def captureCabinRoom(val):

    if str(val) != 'nan':
        x = re.findall("[0-9]+", val)
        if len(x) == 0:
            x = [ 0 ]

        return int(x[0])
        
    return 0


train_df['CabinRoom'] = train_df['Cabin'].apply(captureCabinRoom)
test_df['CabinRoom'] = test_df['Cabin'].apply(captureCabinRoom)
print(train_df['CabinRoom'].unique())
for cabin in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']:
    alives = len(train_df[(train_df['CabinPrefix'] == cabin) & (train_df['Survived'] == 1)])
    deads = len(train_df[(train_df['CabinPrefix'] == cabin) & (train_df['Survived'] == 0)])
    ratio = 0 if alives + deads == 0 else alives / (alives + deads)
    print("Cabin [%s] Alive: %3d, Dead: %3d ===> %0.4f" % (cabin, alives, deads, ratio))
train_df.loc[train_df['CabinPrefix'] == 'T', 'CabinPrefix'] = 'A'
train_df['CabinPrefix'] = train_df['CabinPrefix'].map({ 'A': 0, 'B': 800, 'C': 400, 
                                           'D': 1200, 'E': 1000, 'F': 600, 'G': 200 })
test_df['CabinPrefix'] = test_df['CabinPrefix'].map({ 'A': 0, 'B': 800, 'C': 400, 
                                           'D': 1200, 'E': 1000, 'F': 600, 'G': 200 })
print(train_df['CabinPrefix'].unique())
def calCabin(rec):
    prefix = rec['CabinPrefix']
    room = rec['CabinRoom']
    
    if str(prefix) != 'nan':
        return int(prefix) + int(room)
    
    return np.nan

train_df['Cabin'] = train_df.apply(calCabin, axis = 1)
test_df['Cabin'] = test_df.apply(calCabin, axis = 1)

train_df.drop(columns = ['CabinPrefix', 'CabinRoom'], inplace = True)
test_df.drop(columns = ['CabinPrefix', 'CabinRoom'], inplace = True)

fillValues('Cabin', train_df, test_df)
train_df['FamilySize'] = train_df['Parch'] + train_df['SibSp'] + 1
test_df['FamilySize'] = test_df['Parch'] + test_df['SibSp'] + 1

train_df.drop(columns = ['SibSp', 'Parch'], inplace = True)
test_df.drop(columns = ['SibSp', 'Parch'], inplace = True)
def captureTicketId(val):
    m = re.findall('[0-9]+', val)
    if len(m) == 0:
        return 1
    
    big = 0
    for num in m:
        if int(num) > big:
            big = int(num)
    
    return big

train_df['Ticket'] = train_df['Ticket'].apply(captureTicketId)
test_df['Ticket'] = test_df['Ticket'].apply(captureTicketId)

train_df['Ticket'] = np.log(train_df['Ticket'])
test_df['Ticket'] = np.log(test_df['Ticket'])

train_df['Ticket'] = train_df['Ticket'].round(4)
test_df['Ticket'] = test_df['Ticket'].round(4)
titleBinned = {}
def calTitle(df):
    
    for title in [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ]:
        alives = len(df[(df['Title'] == title) & (df['Survived'] == 1)])
        deads = len(df[(df['Title'] == title) & (df['Survived'] == 0)])
        ratio = 0 if alives + deads == 0 else alives / (alives + deads)
#        print("[%2d]: Alives {%3d} Dead: {%3d} ==> [%0.4f]" % 
#              (title, alives, deads, ratio))
        titleBinned[str(title)] = ratio
    
def binTitle(*args):
    
    for df in args:
        for title in [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ]:
            df.loc[df['Title'] == title, 'TitleP'] = titleBinned[str(title)]
            
        df['Title'] = df['TitleP'].round(4)
        df.drop(columns = ['TitleP'], inplace = True)  
        
calTitle(train_df)
binTitle(train_df, test_df)
ageBinned = {}
def calAge(df):
       
    last = 0
    for age in [ 5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 100 ]:
        alivesBoy = len(df[(df['Age'] >= last) & 
                             (df['Age'] < age) & 
                             (df['Sex'] == 'male') &
                             (df['Survived'] == 1)])
        deadsBoy = len(df[(df['Age'] >= last) & 
                            (df['Age'] < age) &
                            (df['Sex'] == 'male') & 
                            (df['Survived'] == 0)])
        alivesGirl = len(df[(df['Age'] >= last) & 
                             (df['Age'] < age) & 
                             (df['Sex'] == 'female') &
                             (df['Survived'] == 1)])
        deadsGirl = len(df[(df['Age'] >= last) & 
                            (df['Age'] < age) &
                            (df['Sex'] == 'female') & 
                            (df['Survived'] == 0)])
    
        ratioBoy = 0 if alivesBoy + deadsBoy == 0 else alivesBoy / (alivesBoy + deadsBoy)
        ratioGirl = 0 if alivesGirl + deadsGirl == 0 else alivesGirl / (alivesGirl + deadsGirl)
        print("[%3d, %3d]: Male {%2d, %2d} ==> [%0.4f], Female {%2d, %2d} ==> [%0.4f]" % 
              (last, age, alivesBoy, deadsBoy, ratioBoy, alivesGirl, deadsGirl, ratioGirl))
        
        ageBinned[str(last) + ":" + str(age) + ":male"] = round(ratioBoy, 4)
        ageBinned[str(last) + ":" + str(age) + ":female"] = round(ratioGirl, 4)
        last = age
   
        
def binAge(*args):
    
    for df in args:
        last = 0
        for age in [ 5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 100 ]:
            male = str(last) + ":" + str(age) + ":male"
            female = str(last) + ":" + str(age) + ":female"
            df.loc[(df['Age'] >= last) & (df['Age'] < age) & (df['Sex'] == 'male'), 'AgeP'] = ageBinned[male]
            df.loc[(df['Age'] >= last) & (df['Age'] < age) & (df['Sex'] == 'female'), 'AgeP'] = ageBinned[female]
            last = age
    
        df['Age'] = df['AgeP'].round(4)
        df.drop(columns = ['AgeP'], inplace = True)   

calAge(train_df)
binAge(train_df, test_df)
fareBinned = {}
def calFare(df):
          
    last = 0
    for fare in [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 200, 500, 1000 ]:
        alivesBoy = len(df[(df['Fare'] >= last) & 
                             (df['Fare'] < fare) & 
                             (df['Sex'] == 'male') &
                             (df['Survived'] == 1)])
        deadsBoy = len(df[(df['Fare'] >= last) & 
                            (df['Fare'] < fare) &
                            (df['Sex'] == 'male') & 
                            (df['Survived'] == 0)])
        alivesGirl = len(df[(df['Fare'] >= last) & 
                             (df['Fare'] < fare) & 
                             (df['Sex'] == 'female') &
                             (df['Survived'] == 1)])
        deadsGirl = len(df[(df['Fare'] >= last) & 
                            (df['Fare'] < fare) &
                            (df['Sex'] == 'female') & 
                            (df['Survived'] == 0)])
    
        ratioBoy = 0 if alivesBoy + deadsBoy == 0 else alivesBoy / (alivesBoy + deadsBoy)
        ratioGirl = 0 if alivesGirl + deadsGirl == 0 else alivesGirl / (alivesGirl + deadsGirl)
        print("[%4d, %4d]: Male {%3d, %3d} ==> [%0.4f], Female {%3d, %3d} ==> [%0.4f]" % 
              (last, fare, alivesBoy, deadsBoy, ratioBoy, alivesGirl, deadsGirl, ratioGirl))
        fareBinned[str(last) + ":" + str(fare) + ":male"] = round(ratioBoy, 4)
        fareBinned[str(last) + ":" + str(fare) + ":female"] = round(ratioGirl, 4)        
        last = fare
        
def binFare(*args):
    
    for df in args:
        last = 0
        for fare in [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 200, 500, 1000]:
            male = str(last) + ":" + str(fare) + ":male"
            female = str(last) + ":" + str(fare) + ":female"
            df.loc[(df['Fare'] >= last) & (df['Fare'] < fare) & (df['Sex'] == 'male'), 'FareP'] = fareBinned[male]
            df.loc[(df['Fare'] >= last) & (df['Fare'] < fare) & (df['Sex'] == 'female'), 'FareP'] = fareBinned[female]
            last = fare
    
        df['Fare'] = df['FareP'].round(4)
        df.drop(columns = ['FareP'], inplace = True) 

calFare(train_df)
binFare(train_df, test_df)
def prepare(*args):
    
    label_column = [ 'Survived']
    categorical_columns = [ 'Sex', 'Title', 'Pclass', 'Embarked' ]
    numeric_columns = [ 'Age', 'Fare', 'Cabin', 'FamilySize' ]
    
    alldf = []
    for df in args:
        alldf.append(df)
        
    all_df = pd.concat(alldf)    
    all_df = pd.get_dummies(all_df, columns = categorical_columns)

    train_uni = set(all_df.columns).symmetric_difference(numeric_columns + ['Name', 'PassengerId'] + label_column)

    cat_columns = list(train_uni)
    
    all_columns = numeric_columns + cat_columns 
      
    scaler = MinMaxScaler()
    all_df[numeric_columns] = scaler.fit_transform(all_df[numeric_columns])
    
    return all_df, all_columns
    
def computeRegression(model, name, df1, df2):
    
    all_df, all_columns = prepare(df1, df2)
     
    if 'Survived' in all_columns:
        all_columns.remove('Survived')

    all_df_in = all_df.loc[all_df['Survived'].isna() == False, ['PassengerId'] + all_columns]
    all_df_lb = all_df.loc[all_df['Survived'].isna() == False, 'Survived']
 
    model.fit(all_df_in[all_columns], all_df_lb)
        
    for df in [ df1, df2 ]:
        work_df = df.copy()
        work_df[all_columns] = all_df.loc[all_df['PassengerId'].isin(df['PassengerId']), all_columns]
        df[name] = model.predict(work_df[all_columns])
        df[name] = df[name].round(4)
        

computeRegression(XGBRegressor(), 'XGBoost', train_df, test_df)
computeRegression(HuberRegressor(), 'Huber', train_df, test_df)
computeRegression(RandomForestRegressor(), 'Forest', train_df, test_df)
computeRegression(TheilSenRegressor(), 'Theil', train_df, test_df)
computeRegression(MLPRegressor(), "MLP", train_df, test_df)
        
print(train_df.columns)
print(test_df.columns)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})
label_column = ['Survived']
all_features_columns = [ 'Age', 'Fare', 'Title', 'FamilySize', 'Ticket', 'Sex',
                   'Embarked', 'Cabin', 'Pclass', 'XGBoost', 'Huber', 'Forest', 'Theil', 'MLP' ]

pca = PCA(n_components = 5)
pca_train_df = pd.DataFrame(pca.fit_transform(train_df[all_features_columns]))
pca_test_df = pd.DataFrame(pca.transform(test_df[all_features_columns]))

model = svm.SVC(kernel='rbf', gamma ='auto', C=1.0)
model.fit(pca_train_df, train_df[label_column])
train_df['Prediction'] = model.predict(pca_train_df)
kfold = RepeatedStratifiedKFold(n_splits = 9, n_repeats = 5, random_state = 0)
results = cross_val_score(svm.SVC(kernel='rbf', gamma ='auto', C=1.0), 
                          pca_train_df, train_df[label_column], cv = kfold)
print("Cross Validation Accuracy: %0.4f" % results.mean())
print(confusion_matrix(train_df['Survived'], train_df['Prediction']))
print(classification_report(train_df['Survived'], train_df['Prediction']))
test_df['Survived'] = model.predict(pca_test_df)
test_df[['PassengerId','Survived']].to_csv('results.csv', index = False)
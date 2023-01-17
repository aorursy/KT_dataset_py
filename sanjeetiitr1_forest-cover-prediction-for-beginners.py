import pandas as pd
import numpy as np 
import matplotlib.pyplot as plot

testx = pd.read_csv("../input/test.csv")
testx.head()
df = pd.read_csv("../input/train.csv")
print(df.shape)
df.head()
df.corr()
df.columns.get_loc("Wilderness_Area1")
df.head()
train_y = df.pop('Cover_Type')
train_y.head()

print(type(train_y))
df.isnull().sum()
df["Wilderness"] = df.iloc[:,11:15].idxmax(axis =1)
df["Wilderness"]
df["Wilderness"] = df["Wilderness"].apply(lambda x: x[15])
df["Wilderness"]
df["Soil_Type"]= df.iloc[:,15:55].idxmax(axis =1)
df.head()
df["Soil_Type"] = df["Soil_Type"].apply(lambda x:x[9:])
df["Soil_Type"]
df.tail()
df.shape
def remove_feature(df):
    dfg = df.columns[15:55]
    dfg2 = df.columns[11:15]
    df.drop(dfg ,axis =1, inplace= True )
    df.drop(dfg2 , axis = 1, inplace= True)
    df.pop("Id")
    df.pop("Horizontal_Distance_To_Roadways")
remove_feature(df)
print(df.shape)
print(testx.shape)
df.head()
testx["Wilderness"] = testx.iloc[:,11:15].idxmax(axis =1)
testx["Wilderness"] = testx["Wilderness"].apply(lambda x: x[15])
testx["Wilderness"]
testx.shape
testx["Soil_Type"]= testx.iloc[:,15:55].idxmax(axis =1)
testx["Soil_Type"] = testx["Soil_Type"].apply(lambda x:x[9:])
testx.head()
dfg = testx.columns[15:55]
dfg2 = testx.columns[11:15]
testx.drop(dfg ,axis =1, inplace= True )
testx.drop(dfg2 , axis = 1, inplace= True)
testx.pop("Id")
testx.pop("Horizontal_Distance_To_Roadways")
testx.head()
print(testx.shape)
print(df.shape)
df.head()
train_y.shape
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(df ,train_y ,test_size = .34, random_state = 234)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer , accuracy_score
parameters = {
    'n_estimators': [4, 6, 9], 
    'max_features': ['log2', 'sqrt','auto'], 
    'criterion': ['entropy', 'gini'],
    'max_depth': [2, 3, 5, 10], 
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1,5,8]
}

acc_scorer =make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf , parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(x_train,y_train)

clf = grid_obj.best_estimator_
clf.fit(x_train, y_train)
predict = clf.predict(testx)
print(predict)
from sklearn.cross_validation import KFold 

def validation_ml(clf):
    
    kf = KFold(df.shape[0] , 10 )
    outcome =[]
    for train_i, test_i in kf :
        train_x, test_x = df.values[train_i] , df.values[test_i]
        trainy, test_y = train_y.values[train_i] , train_y.values[test_i]
        
        clf.fit(train_x,trainy)
        prediction = clf.predict(test_x)
        accuracy =accuracy_score(test_y, prediction)
        outcome.append(accuracy)
        print(accuracy)
    print(" Mean Accuracy = " , np.mean(outcome))
        
        
validation_ml(clf)
validation_ml(clf)
test_original = pd.read_csv("../input/test.csv")
ids = test_original['Id']
output = pd.DataFrame({ 'Id' : ids, 'Cover_Type' : predict })
output.shape
output.tail()
columnsTitles=["Id","Cover_Type"]
output=output.reindex(columns=columnsTitles)
output.tail()
output.to_csv('Forest-cover-prediction_new.csv', index = False)


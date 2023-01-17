import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as clf

%matplotlib inline
clf.go_offline()
#import os
#print(os.listdir("../input"))

data = pd.read_json('../input/train.json')
print(data.shape)
data.iloc[:1,:]
test = pd.read_json('../input/test.json')
print(test.shape)
test.iloc[:1,:]
#check is there any null values ??
data[data['ingredients'].isnull()]
def extract_ingredients(serie):
    list_ingredients=[]
    for lista in serie:
        for element in lista:
            if element in list_ingredients:
                pass
            elif element not in list_ingredients:
                list_ingredients.append(element)
            else:
                pass
        
    return list_ingredients 
ingredients = extract_ingredients(data['ingredients'])
print(len(ingredients))
cuisines = data['cuisine'].unique()
print(cuisines.shape)
for ingredient in ingredients:
    data[ingredient]=np.zeros(len(data["ingredients"]))
for ingredient in ingredients:
    test[ingredient]=np.zeros(len(test["ingredients"]))
def ohe(serie, dtframe):    
    ind=0
    for lista in serie:
        
        for ingredient in lista:
            if ingredient in ingredients:
                dtframe.loc[ind,ingredient]=1
            else:
                pass
        ind +=1
ohe(data['ingredients'], data)
ohe(test['ingredients'], test)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
predict = ingredients
feature = data['cuisine']
X = data[predict]
y = feature
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
log_reg = LogisticRegression(C=1)
log_reg.fit(X_train, y_train)
y_predicted = log_reg.predict(X_test)
y_predicted
accuracy_score(y_test, y_predicted)
pd.DataFrame(confusion_matrix(y_test, y_predicted, labels=cuisines), index=cuisines, columns=cuisines)
y_final_prediction = log_reg.predict(test[predict])
output = test['id']
output = pd.DataFrame(output)
output['cuisine'] = pd.Series(y_final_prediction)
output.to_csv('sample_submission.csv', index=False)

import pandas as pd
import numpy as np
import seaborn as sns
import time
np.random.seed(1988)
! dir
data = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
data.head()
#there is no NaNs
data[pd.isna(data['ingredients'])]
data.shape
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
len (ingredients)
#Types of differents cuisines:
data['cuisine'].unique().shape
cuisines = data['cuisine'].unique()
cuisines
#Create columns
t = time.time()
for ingredient in ingredients:
    data[ingredient]=np.zeros(len(data["ingredients"]))

print("It took %i seg" %(time.time()-t))
def ohe(serie, dtframe):    
    ind=0
    for lista in serie:
        
        for ingredient in lista:
            if ingredient in ingredients:
                dtframe.loc[ind,ingredient]=1
            else:
                pass
        ind +=1
t = time.time()
ohe(data['ingredients'], data)
print('it took %i segs' % (time.time()-t))
from sklearn.model_selection import train_test_split
predictors = ingredients
response = 'cuisine'
X = data[predictors]
Y = data[response]
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
del(data)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
log_reg = LogisticRegression(C=1)
log_reg.fit(x_train, y_train)
y_predicted = log_reg.predict(x_test)
y_predicted
accuracy_score(y_test, y_predicted)
pd.DataFrame(confusion_matrix(y_test, y_predicted, labels=cuisines), index=cuisines, columns=cuisines)
# Create columns on test
t = time.time()
for ingredient in ingredients:
    test[ingredient]=np.zeros(len(test["ingredients"]))

print('Takes %i seconds' %(time.time()-t))
t = time.time()
ohe(test['ingredients'], test)
print('Takes %i seconds' %(time.time()-t))
test.head()
y_final_prediction = log_reg.predict(test[predictors])
output = test['id']
output = pd.DataFrame(output)
output['cuisine'] = pd.Series(y_final_prediction)
output.head()
output.to_csv('output.csv', index=False)
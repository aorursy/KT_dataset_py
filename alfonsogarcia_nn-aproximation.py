import pandas as pd
import numpy as np
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
def ohe(serie, dtset):    
    ind=0
    for lista in serie:
        for ingredient in lista:
            if ingredient in ingredients:
                dtset.loc[ind,ingredient]=1
            else:
                pass
        ind +=1
t = time.time()
ohe(data['ingredients'], data)
print('it took %i segs' % (time.time()-t))
predictors = ingredients
response = 'cuisine'
X = data[predictors]
Y = data[response]
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
np.random.seed(1988)
len(cuisines)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

encoder = LabelEncoder()
encoder.fit(Y)
encoded_y_train = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_y_train)
encoded_y_train
dummy_y_train
# Create model:
def baseline_model():
    model = Sequential()
    model.add(Dense(100, input_dim=6714, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(20, activation='softmax')) # The output layer must create 20 output values,
    # one for each class. The output value with the largest value will be taken as the class predicted by the model.

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier
estimator = KerasClassifier(build_fn=baseline_model, epochs=5, batch_size=100)
estimator.fit(X, dummy_y_train)
# Create columns on test
t = time.time()
for ingredient in ingredients:
    test[ingredient]=np.zeros(len(test["ingredients"]))

print("It took %i seg" %(time.time()-t))
t = time.time()
ohe(test['ingredients'], test)
print('it took %i segs' % (time.time()-t))
y_pred = estimator.predict(test[predictors])
y_pred = encoder.inverse_transform(y_pred)
test['cuisine'] = y_pred
output = pd.DataFrame(test['id'])
output['cuisine'] = pd.Series(y_pred, name='cuisine')
output.to_csv('output.csv', index=False)
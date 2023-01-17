import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils.np_utils import to_categorical



from sklearn.cross_validation import train_test_split



import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")

ids = data.pop("id")



creature = data.pop("type")

color = data.pop("color")



data.head()
#add_names1 = ["bone_length_square", "rotting_flesh_square", "hair_length_square", "has_soul_square"]

#for i in range(len(add_names1)):

#    data[add_names1[i]] = data.values[:, i] ** 2



coords = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

add_names2 = ["bone_flesh", "bone_hair", "bone_soul", "flesh_hair", "flesh_soul", "hair_soul"]



for i in range(len(coords)):

    data[add_names2[i]] = data.values[:, coords[i][0]] *data.values[:, coords[i][1]]



scal = StandardScaler()

data = pd.DataFrame(scal.fit_transform(data))

data.head()
X_train, X_test, y_train, y_test = train_test_split(data, creature, test_size=0.33, random_state=42)
# encoding creatures with numbers



creature_encoder = LabelEncoder()

creature_code = creature_encoder.fit(creature)



y_train_code = creature_encoder.transform(y_train)

y_test_code = creature_encoder.transform(y_test)
X_train = X_train.values 

X_test = X_test.values 

y_train_code = to_categorical(y_train_code)

y_test_code = to_categorical(y_test_code)
def nn_model(in_dim):

    model = Sequential()

    

    model.add(Dense(300, input_dim=in_dim, init='uniform',  activation='relu'))

    model.add(Dropout(0.6))

    model.add(Dense(100, init='uniform',  activation='sigmoid'))

    model.add(Dropout(0.4))

    model.add(Dense(3, init='uniform', activation='softmax'))

    

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
np.random.seed(42)



model = nn_model(10)

model.fit(X_train, y_train_code, verbose= 0, nb_epoch= 500)



model.evaluate(X_train, y_train_code), model.evaluate(X_test, y_test_code)

#0.838709 0.83197831

#0.83870968126481582, 0.84010841255265523



#0.75806451805176278, 0.75609755903724729

#0.99193548387096775, 0.6585365848812631
np.random.seed(42)



model = nn_model(10)



y = creature_code.transform(creature)

y = to_categorical(y)



model.fit(data.values, y, verbose= 0, nb_epoch= 1000)



model.evaluate(data.values, y)
test = pd.read_csv("../input/test.csv")

test_ids = test.pop("id")

test_color = test.pop("color")



#for i in range(len(add_names1)):

#    test[add_names1[i]] = test.values[:, i] ** 2



for i in range(len(coords)):

    test[add_names2[i]] = test.values[:, coords[i][0]] *test.values[:, coords[i][1]]

     



test = scal.transform(test)

test = pd.DataFrame(scal.fit_transform(test))

test.head()
prediction_class = model.predict_classes(test.values)
result = pd.DataFrame({"id": [], "type": []})

result.id = np.array(np.array(test_ids))

result.type = creature_encoder.inverse_transform(prediction_class)
result.to_csv("nn_result.csv", index= False)
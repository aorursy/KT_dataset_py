%matplotlib notebook

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))

train_data = pd.read_csv("../input/train.csv")

train_data.head(10)
def CleanData(data):

    # Delete useless columns

    data = data.drop(columns= ["Name", "Ticket", "Cabin"])

    

    # Assign categorical values to Gender

    data = data.replace("male", 0)

    data = data.replace("female", 1)

    

    # Assign categorical values to Port of Embarking

    data = data.replace("C", 0)

    data = data.replace("Q", 1)

    data = data.replace("S", 2)

    

    data["Age"].fillna(-1, inplace=True)

    data["Embarked"].fillna(-1, inplace=True)

    

    return data



train_data = CleanData(train_data)

train_data.head(20)
pick_columns = np.arange(2,9)

feature_names = train_data.columns[pick_columns]



x_train = train_data.iloc[:, pick_columns].values.astype("float32")

y_train = train_data.iloc[:, 1].values.astype("int32")

id_train = train_data.iloc[:, 0].values.astype("int32")


x_mean = np.mean(x_train, axis = 0)

x_std = np.std(x_train, axis = 0)

x_train = (x_train - x_mean) / x_std

x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2)
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout
model = Sequential()

model.add(Dense(128, activation="relu", input_shape = (x_train.shape[1],))) # Hidden Layer 1 that receives the Input from the Input Layer



model.add(Dense(64, activation="relu")) # Hidden Layer 2

model.add(Dropout(0.2))



model.add(Dense(32, activation="relu")) # Hidden Layer 3

model.add(Dropout(0.2))



model.add(Dense(16, activation="relu")) # Hidden Layer 4

model.add(Dropout(0.2))





model.add(Dense(1, activation="sigmoid")) # Outout Layer



model.summary()
model.compile(optimizer='adam', loss = "binary_crossentropy", metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 64, epochs = 20)
validation_loss, validation_accuracy = model.evaluate(x_validation, y_validation, batch_size=32)

print("Loss: "+ str(np.round(validation_loss, 3)))

print("Accuracy: "+ str(np.round(validation_accuracy, 3)))
import eli5

from eli5.permutation_importance import get_score_importances



def Score(x, y):

    loss, accuracy = model.evaluate(x, y, batch_size=32, verbose=0)

    return accuracy



base_score, score_decreases = get_score_importances(Score, x_validation, y_validation, n_iter= 100)

feature_importances = np.mean(score_decreases, axis=0)

feature_std = np.std(score_decreases, axis=0)



sort_index = np.argsort(feature_importances)[::-1]

print("Feature Importances:")

for i in range(feature_names.size):

    

    j = sort_index[i]

    print(feature_names[j] +":  "+ str(np.round(feature_importances[j],3)) + " +- " + str(np.round(feature_std[j],3)))

import seaborn as sns

plt.figure()

sns.barplot(x = "Sex", y = "Survived", data = train_data)

plt.ylabel("Survival Rate")

plt.xlabel("Sex")

plt.xticks([0,1], ["male", "female"]) 





plt.figure()

sns.barplot(x = "Pclass", y = "Survived", data = train_data)

plt.ylabel("Survival Rate")

plt.xlabel("Passenger Class")
import shap  # package used to calculate Shap values



passenger_to_study = 40

passenger_data = np.array([x_validation[passenger_to_study]])

passenger_survival = model.predict(passenger_data)[0][0]



print("Survival change of passenger X: " + str(np.round(passenger_survival,3)))



explainer = shap.DeepExplainer(model, data= x_validation)



shap_values = explainer.shap_values(passenger_data)



print("How much each feature contributed:")

for i in range(feature_names.size):



    j = sort_index[i]





    print(feature_names[j] +" = "+ str(np.round(passenger_data[0][j]*x_std[j] + x_mean[j],3)) + " -> " + str(np.round(shap_values[0][0][j],3)))



shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0][0], passenger_data[0]*x_std + x_mean, feature_names= feature_names)

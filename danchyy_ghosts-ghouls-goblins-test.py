# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

print(train_data)

test_data = pd.read_csv("../input/test.csv")

print(test_data)
codesdummies = test_data["color"]

copy_data = test_data



id_test = test_data["id"]

one_hot = pd.get_dummies(codesdummies)

cleaned_data = test_data.drop("color", 1)

cleaned_data = cleaned_data.drop("id", 1)



#final_test_data = cleaned_data.join(one_hot)

final_test_data = cleaned_data

print(final_test_data)
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

label_enc = LabelEncoder()

dummies = train_data["color"].as_matrix()

#print(dummies)

labeled_dummies = label_enc.fit_transform(dummies)

labeled_dummies = scaler.fit_transform(labeled_dummies.reshape(-1,1))

print(labeled_dummies)

enc = OneHotEncoder()

encoded_vars = enc.fit_transform(labeled_dummies.reshape(-1,1))



test_dummies = test_data["color"].as_matrix()

test_labeled = label_enc.fit_transform(test_dummies)

test_labeled = scaler.fit_transform(test_labeled.reshape(-1,1))
codesdummies = train_data["color"]

copy_data = train_data



target = train_data["type"]

one_hot = pd.get_dummies(codesdummies)

cleaned_data = train_data.drop("color", 1)

cleaned_data = cleaned_data.drop("type", 1)

cleaned_data = cleaned_data.drop("id", 1)



#final_data = cleaned_data.join(one_hot)

final_data = cleaned_data

print(final_data)

target = target.astype("category") 

target = target.cat.codes

#print(target)
X_data = final_data.as_matrix()

y_data = target.as_matrix()

print(X_data.shape, labeled_dummies.reshape(-1,1).shape)

X_data = np.append(X_data, labeled_dummies, axis=1)

print(X_data)

print (y_data)



X_test_final = final_test_data.as_matrix()

X_test_final = np.append(X_test_final, test_labeled, axis=1)

id_array = id_test.as_matrix()

print(X_test_final)
import scipy as sp

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import zero_one_loss

from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
reduced_train = PCA(n_components=2).fit_transform(X_data)

plt.scatter(reduced_train[:,0], reduced_train[:,1], c=y_data, cmap='prism')
regularization = np.arange(1, 30, 1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

min_c, min_loss, min_degree = None, None, None

degrees = [1, 2]

count = 1

plt.figure(figsize=(12,12))

for deg in degrees:

    train_res, test_res = [], []

    poly = PolynomialFeatures(deg)

    X_train = poly.fit_transform(X_train)

    X_test = poly.fit_transform(X_test)

    for c in regularization:

        model = LogisticRegression(C=c, multi_class="ovr")

        model.fit(X_train, y_train)

        predicted_train = model.predict(X_train)

        predicted_validation = model.predict(X_test)

        train_loss = zero_one_loss(y_train, predicted_train)

        test_loss = zero_one_loss(y_test, predicted_validation)

        if not min_c or test_loss < min_loss:

            min_c = c

            min_loss = test_loss

            min_degree = deg

        train_res.append(train_loss)

        test_res.append(test_loss)

    plt.subplot(1, 2, count)

    plt.plot(regularization, train_res, color="r", label="Train error")

    plt.plot(regularization, test_res, color="b", label="Validation error")

    plt.legend(loc="upper right")

    count+=1

    plt.title("Curve for degree = " + str(deg))

    





print("Min loss: " + str(min_loss) + ", min c = " + str(min_c) + ", min degree = " + str(min_degree))



model = LogisticRegression(C=min_c)

model.fit(X_data, y_data)

predicted = model.predict(X_test_final)



predicted_string = []

for res in predicted:

    if res == 0:

        predicted_string.append("Ghost")

    elif res == 1:

        predicted_string.append("Ghoul")

    else:

        predicted_string.append("Goblin")



#print(predicted_string)


result_dict = {

    "id" : id_array.tolist(),

    "type" : predicted_string

}

frame = pd.DataFrame(result_dict)
frame.to_csv("result2.csv", index=False)
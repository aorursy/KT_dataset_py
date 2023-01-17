import numpy as np

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_json("../input/train.json")

train.info()
X = train.drop("interest_level", 1)

Y = train["interest_level"].astype("category")
X["street_address"] = X["street_address"].astype('category').cat.codes

X["created"] = X["created"].astype('category').cat.codes

X["building_id"] = X["building_id"].astype('category').cat.codes

X["description"] = X["description"].astype('category').cat.codes

X["display_address"] = X["display_address"].astype('category').cat.codes

X["manager_id"] = X["manager_id"].astype('category').cat.codes
features = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "street_address", "created", 

           "description", "display_address"]

X = X[features]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
test = pd.read_json("../input/test.json")

index = test["listing_id"]

test = test[features]
olist = list(test.select_dtypes(['object']))

for col in olist:

    test[col] = test[col].astype('category').cat.codes
favorite_clf = LinearDiscriminantAnalysis()

favorite_clf.fit(X_train, y_train)

test_predictions = favorite_clf.predict_proba(test)
labels2idx = {label: i for i, label in enumerate(favorite_clf.classes_)}

labels2idx
submission = pd.DataFrame({

        "listing_id": index,

        "high": test_predictions[:,0],

        "medium":test_predictions[:,2],

        "low":test_predictions[:,1]

    })

    

columnsTitles=["listing_id","high","medium","low"]

submission=submission.reindex(columns=columnsTitles)

submission.to_csv('submission.csv', index=False)
from keras.models import Sequential

from keras.layers import Dense
model = Sequential()

model.add(Dense(9, input_dim=9, init="normal", activation="relu"))

model.add(Dense(3, init="normal", activation="sigmoid"))
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(Y)

encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)

dummy_y = np_utils.to_categorical(encoded_Y)
# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

model.fit(X.values, dummy_y, nb_epoch=10, batch_size=10)
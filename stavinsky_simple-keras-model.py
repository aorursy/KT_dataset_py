from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.callbacks import EarlyStopping

from keras.optimizers import Adam

import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import (

    FunctionTransformer, OneHotEncoder, LabelEncoder, MinMaxScaler)

from sklearn.base import TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.cluster import MeanShift

from collections import Counter

import matplotlib.pyplot as plt

%matplotlib inline





class TransformWrapper(TransformerMixin):

    """simple wrapper for instruments like LabelEncoder.

    It is usefull if we want to encode many features at one time. 

    They have a little bit different interface compairing to OneHotEncoder for example"""

    def __init__(self, enc):

        self.enc = enc

    def fit(self, X, y=None):



        self.d = {}

        for col in X.columns:

            self.d[col] = self.enc()

            self.d[col].fit(X[col].values)

        return self

        

    def transform(self, X, y=None):



        l = list()

        for col in X.columns:

            l.append(self.d[col].transform(X[col].values).reshape(-1,1))

        result = np.hstack(l)

        return result





def write_answer(y, output='answers.csv'):

    answer = pd.DataFrame({

        'PassengerId': test.PassengerId.values,

        'Survived': y.reshape(-1)

    })

    answer.to_csv(output, index=False)

    



train = pd.read_csv("../input/train.csv", index_col="PassengerId")

test = pd.read_csv("../input/test.csv")

target_column = "Survived"

target = train[target_column]

train = train.drop(["Survived"], axis=1)

train.head()
# fare is simple float value. We just need to fill NaN and scale values

pipe_fare = Pipeline([

    ('select', FunctionTransformer(

        lambda X: X[["Fare"]], validate=False)),

    ('fillna', FunctionTransformer(

        lambda X: X.fillna(0), validate=False)),

    ('minmax', MinMaxScaler())

])



# SibSp and Parch is most like categorical features. So we will encode and binarize values. 

pipe_sibpa = Pipeline([

    ('select', FunctionTransformer(

        lambda X: X[["SibSp","Parch"]], validate=False)),

    ('fillna', FunctionTransformer(

        lambda X: X.fillna(0), validate=False)),

    ('enc', TransformWrapper(LabelEncoder)),

    ("bin", OneHotEncoder(sparse=False)),

])





# Embarked also categorical

pipe_emb = Pipeline([

    ('select', FunctionTransformer(

        lambda X: X["Embarked"], validate=False)),

    ('fillna', FunctionTransformer(

        lambda X: X.fillna('C'), validate=False)),

    ('str', FunctionTransformer(

        lambda X: pd.DataFrame(X.apply(str)), validate=False)),

    ('enc', TransformWrapper(LabelEncoder)),

    ("bin", OneHotEncoder(sparse=False)),

])



# categorical 

pipe_pclass = Pipeline([

    ('select', FunctionTransformer(

        lambda X: X["Pclass"], validate=False)),

    ('fillna', FunctionTransformer(

        lambda X: X.fillna(3), validate=False)),

    ('str', FunctionTransformer(

        lambda X: pd.DataFrame(X.apply(str)), validate=False)),

    ('enc', TransformWrapper(LabelEncoder)),

    ("bin", OneHotEncoder(sparse=False)),

])



# I'm going to make categorical feature from Age by splitting this variable to 4 classes

pipe_age = Pipeline([

    ('select', FunctionTransformer(

        lambda X: X["Age"], validate=False)),

    ('fillna', FunctionTransformer(

        lambda X: X.fillna(0).values.reshape(-1,1), validate=False)),

    ('make_cat', FunctionTransformer(

        lambda x: np.where(

            x==0, 0, np.where(

                x<18, 1, np.where(

                    x<55, 2, 3))).reshape(-1,1), validate=False)),

    ("bin", OneHotEncoder(sparse=False)),

])



# i'm trying to extract first letter from this 

# feature but look's like it doesn't change anything

pipe_cabin = Pipeline([

    ('select', FunctionTransformer(

        lambda X: X["Cabin"], validate=False)),

    ('first_letter', FunctionTransformer(

        lambda x: x.str.extract(" (?P<letter>[A-Z])", expand=True), validate=False)),

    ('fillna', FunctionTransformer(

        lambda X: X.fillna('na'), validate=False)),

    ('enc', TransformWrapper(LabelEncoder)),

    ("bin", OneHotEncoder(sparse=False)),



])

pipe_title = Pipeline([

    ('select', FunctionTransformer(

        lambda X: X["Name"], validate=False)),

    ('first_letter', FunctionTransformer(

        lambda x: x.str.extract("(?P<title>\s\w+\.)", expand=True), validate=False)),

    

    ('enc', TransformWrapper(LabelEncoder)),

    ("bin", OneHotEncoder(sparse=False)),

    

])

pipe_name_length = Pipeline([

    ('select', FunctionTransformer(

        lambda X: X["Name"], validate=False)),

    ('len', FunctionTransformer(

        lambda X: X.apply(len), validate=False)),

    ('make_cat', FunctionTransformer(

        lambda x: np.where(

            x<18, 0, np.where(

                x<40, 1, 2)).reshape(-1,1), validate=False)),

    ("bin", OneHotEncoder(sparse=False)),



])

pipe = FeatureUnion([

    ("age", pipe_age),

    ("fare", pipe_fare),

    ("emb", pipe_emb),

    ("pclass", pipe_pclass),

    ("sibpa", pipe_sibpa),

    ("cabin", pipe_cabin),

    ("title", pipe_title),

    ("name_length", pipe_name_length)

])

pipe.fit(pd.concat([train, test]))

X_train, X_test, y_train, y_test = train_test_split(

    pipe.transform(train), target, test_size=0.2, random_state=42)

X_train.shape
def make_model(features):

    """We are going to make model with 2 layers with sizes: 

    1: number of features 

    2: 30

    

    Also we added Dropout layer, because it help to prevent fast overfitting

    """

    adam = Adam(lr=0.001)

    model = Sequential()

    model.add(Dense(features.shape[1], input_shape=(features.shape[1], ), activation='linear' ))

    model.add(Dropout(0.5))

    model.add(Dense(25, activation='sigmoid'))



    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
model = make_model(X_train)



early_stop = EarlyStopping(monitor="val_loss", patience=1)

history = model.fit(

    X_train, y_train, nb_epoch=200, batch_size=10, validation_split=0.2,

    callbacks=[early_stop], 

# I was trying to use EarlyStopping but for this moment it shows not very good result

    verbose=False,

)



predict = model.predict(X_test)

predict = np.where(predict > 0.5, 1, 0)

plt.plot(history.history["val_acc"])

plt.plot(history.history["val_loss"])

# It will be interesting to look at  the dynamics of loss and accuracy 

print(accuracy_score(y_test, predict))
test_predict = model.predict(pipe.transform(test))

test_predict = np.where(test_predict>0.5, 1, 0)

write_answer(test_predict, "mlp.csv")
from sklearn.ensemble import GradientBoostingClassifier

from collections import Counter

clf = GradientBoostingClassifier(

    learning_rate=0.1, n_estimators=200, 

    max_depth=3, min_samples_leaf=2, random_state=42

)

clf.fit(X_train, y_train)

predict = clf.predict(X_test)

accuracy_score(y_test, predict)
clf_predict = clf.predict(pipe.transform(test))

write_answer(clf_predict, "grad_boost.csv")
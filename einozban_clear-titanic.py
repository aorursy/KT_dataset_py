!ls ../input/*

import numpy as np

import pandas as pd
#data

df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



#an extra column

df['Title']=df.Name.str.extract('([A-Za-z]+)\.')

df_test['Title']=df_test.Name.str.extract('([A-Za-z]+)\.')



#Plotting titles:

import seaborn as sns

from matplotlib import pyplot as plt

plt.figure(figsize=(16,3))

sns.countplot(y='Title', data=df, order = df['Title'].value_counts().index)

plt.xlabel("Title")

plt.xticks(rotation=60)

plt.show()
from sklearn.base import TransformerMixin

class TitleTransformer(TransformerMixin):



    def transform(self, df, **transform_params):

        df_new = df

        df_new['Title'].replace(

            ['Master', 'Rev', 'Major', 'Col', 'Mlle', 'Capt', 'Don',

             'Mme', 'Dr', 'Countess', 'Jonkheer', 'Lady', 'Ms'],

            ['Sir', 'Sir', 'Sir', 'Sir', 'Mrs', 'Sir', 'Sir', 'Mrs',

             'Sir', 'Mrs', 'Sir', 'Mrs', 'Miss'],inplace=True)

        return df_new



    def fit(self, X, y=None, **fit_params):

        return self
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer





num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='constant', fill_value= -1)),

    ('std_scaler', StandardScaler())

])





cat_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='constant')),

    ('one-hot', OneHotEncoder())

])





title_pipeline = Pipeline([

    ('titler', TitleTransformer()),

    ('one-hot', OneHotEncoder())

])





num_attribs = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

cat_attribs = ['Sex', 'Embarked']

title_attrib = ['Title',]

full_pipeline = ColumnTransformer([

        ("number", num_pipeline, num_attribs),

        ("category", cat_pipeline, cat_attribs),

        ("title", title_pipeline, title_attrib),

    ])
x = full_pipeline.fit_transform(df)

x_test = full_pipeline.fit_transform(df_test)

y = df['Survived']



from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(

    x, y, test_size=0.2, random_state=42)
filters = 256

kernel_size = 3

from keras.models import Sequential

from keras.layers import Dense, Conv1D, Reshape, Flatten, BatchNormalization

from keras.optimizers import Adam

model = Sequential()

model.add(Reshape((len(x[0]), 1), input_shape=x[0].shape))

model.add(Conv1D(filters, kernel_size, activation='selu', kernel_initializer='lecun_normal'))

model.add(Conv1D(filters, kernel_size, activation='selu', kernel_initializer='lecun_normal'))

model.add(Conv1D(filters, kernel_size, activation='selu', kernel_initializer='lecun_normal'))

model.add(Conv1D(filters, kernel_size, activation='selu', kernel_initializer='lecun_normal'))

model.add(Conv1D(filters, kernel_size, activation='selu', kernel_initializer='lecun_normal'))

model.add(Conv1D(filters, kernel_size, activation='selu', kernel_initializer='lecun_normal'))

model.add(Conv1D(1, kernel_size, activation='sigmoid'))

model.add(Reshape((1,)))



model.summary()

model.compile(optimizer=Adam(2e-4),

loss='binary_crossentropy',

metrics=['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks_list = [

    EarlyStopping(

        monitor='acc',

        patience=4,

    ),

    ModelCheckpoint(

        filepath='my_model.h5',

        monitor='val_loss',

        save_best_only=True,

    )

]





history = model.fit(

    x_train, y_train,

    validation_data=(x_val, y_val),

    batch_size = 16,

    epochs=8,

    callbacks=callbacks_list

)
model.load_weights('my_model.h5')

submission = pd.read_csv('../input/gender_submission.csv')

predictions = model.predict_classes(x_test)

submission.Survived = predictions

submission.to_csv('sumbission.csv',index=False)
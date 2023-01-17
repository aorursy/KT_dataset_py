import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df['Age'].mean()
train_df.head()
# get the train label data, and remove label from train data.

labels = train_df['Survived']

train_df = train_df.drop("Survived", axis=1)
# extract sub string tool funcs

def substrings_in_string(big_string, substrings):

    if type(big_string) == str:

        for substring in substrings:

            if big_string.find(substring) != -1:

                return substring

    return np.nan
# add new feature title

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']



train_df['Title']=train_df['Name'].map(lambda x: substrings_in_string(x, title_list))

test_df['Title']=test_df['Name'].map(lambda x: substrings_in_string(x, title_list))



#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

train_df['Title']=train_df.apply(replace_titles, axis=1)

test_df['Title']=train_df.apply(replace_titles, axis=1)
# add new feature deck

#Turning cabin number into Deck

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

train_df['Deck']=train_df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

train_df['Deck'] = train_df['Deck'].fillna("Unknown")

test_df['Deck']=train_df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

test_df['Deck'] = train_df['Deck'].fillna("Unknown")
#Creating new family_size column and others

train_df['Family_Size']=train_df['SibSp']+train_df['Parch']

#train_df['Age*Class']=train_df['Age']*train_df['Pclass']

train_df['Fare_Per_Person']=train_df['Fare']/(train_df['Family_Size']+1)



test_df['Family_Size']=test_df['SibSp']+test_df['Parch']

#train_df['Age*Class']=train_df['Age']*train_df['Pclass']

test_df['Fare_Per_Person']=test_df['Fare']/(test_df['Family_Size']+1)
train_df.head()
# clean data

def data_preprocessing(df):

    #category Pclass, Embarked

    dummy_columns = ['Pclass', 'Embarked', 'Sex', "Deck", "Title"]

    for col_name in dummy_columns:

        dummy_df = pd.get_dummies(df[col_name], prefix=col_name)

        df = pd.concat([df, dummy_df], axis=1)



    # remove column "name", "ticket", "cabin" since there are too many NaN value, and dummy_columns

    drop_columns = ["Cabin", 'Name', 'Ticket', 'PassengerId']+dummy_columns

    df = df.drop(drop_columns, axis=1)

    

    # since "Age" contain NaN, set it to Zero

    df['Age'] = df["Age"].fillna(df['Age'].mean())

    

    # data transformation

    # normalize data using mean normalization

    # normalize "age", "Fare"

    normal_columns = ["Age", "Fare", "Fare_Per_Person"]

    for col_name in normal_columns:

        df[col_name] = (df[col_name] - df[col_name].mean())/df[col_name].std()

        

    return df.astype(float)
train_data = data_preprocessing(train_df)

test_data = data_preprocessing(test_df)
train_data.head()
test_data.head()
train_data.corr()
import seaborn as sns

colormap = plt.cm.viridis

plt.figure(figsize=(20,20))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_data.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import xgboost as xgb



from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
# here try random forest algorithm

def randomForest(x,y):

    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

    clf.fit(x, y)

    return clf



# here try SVM

def svm(x,y):

    clf = SVC(kernel="linear", C=0.5)

    clf.fit(x, y)

    return clf



def xgboost(x, y):

    gbm = xgb.XGBClassifier(

    #learning_rate = 0.02,

     n_estimators= 2000,

     max_depth= 4,

     min_child_weight= 2,

     #gamma=1,

     gamma=0.9,                        

     subsample=0.8,

     colsample_bytree=0.8,

     objective= 'binary:logistic',

     nthread= -1,

     scale_pos_weight=1).fit(x, y)

    return gbm
classifiers = {"RandomForest": randomForest, "SVM": svm, "xgboost": xgboost}



for name, classifier in classifiers.items():

    # k-fold validation

    kf = KFold(n_splits=3, shuffle=True, random_state=40)

    for train_idx, validate_idx in kf.split(train_data):

        train_x = train_data.values[train_idx]

        train_y = labels[train_idx]

        validate_x = train_data.values[validate_idx]

        validate_y = labels[validate_idx]



        clf = classifier(train_x, train_y)



        train_pre = clf.predict(train_x)

        train_score = accuracy_score(train_y, train_pre)



        pre = clf.predict(validate_x)

        score = accuracy_score(validate_y, pre)



        print("{} - train accuracy: {:.4f}, validate accuracy: {:.4f} ".format(name, train_score, score))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras import optimizers

from keras.callbacks import EarlyStopping
def buildModel():

    model = Sequential()

    # hidden layer1

    model.add(Dense(1024, input_shape=(27,)))

    model.add(Activation("relu"))

    model.add(Dropout(0.5))



    # hidden layer2

    model.add(Dense(256))

    model.add(Activation("relu"))

    model.add(Dropout(0.5))



    # hidden layer3

    model.add(Dense(64))

    model.add(Activation("relu"))

    model.add(Dropout(0.5))



    # output

    model.add(Dense(1))

    model.add(Activation("sigmoid"))



    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=sgd)



    return model
callbacks = [

    EarlyStopping(monitor='val_loss', patience=2, verbose=0),

    #ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),

]
model = buildModel()

# k-fold validation

kf = KFold(n_splits=3, shuffle=True, random_state=40)

for train_idx, validate_idx in kf.split(train_data):

    train_x = train_data.values[train_idx]

    train_y = labels[train_idx]

    validate_x = train_data.values[validate_idx]

    validate_y = labels[validate_idx]



    model.fit(train_x, train_y, epochs=300000, validation_data=(validate_x, validate_y), callbacks=callbacks)
# train on the whloe dataset

model = buildModel()

model.fit(train_data.get_values(), labels, epochs=20, verbose=0)
predictions = model.predict(test_data.get_values())

result = np.array(predictions>0.5).astype(int)

result = np.reshape(result,(-1))

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": result

    })
submission.to_csv("titanic_submission.csv", index=False)
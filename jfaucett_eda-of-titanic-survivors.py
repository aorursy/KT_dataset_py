# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../input/train.csv")
df.head(2)
print('size: {}'.format(len(df)))
df.info()
display(np.corrcoef(df['Parch'], df['Survived']))
df['Parch'].value_counts().plot(kind='bar')

from sklearn.preprocessing import LabelEncoder

def normalize_embarked(df):
    embarked = df['Embarked'].astype('category').values.copy()
    embarked[df['Embarked'].isnull()] = 'S'

    # convert categorical to numeric encodings
    le = LabelEncoder()
    le.fit(embarked)
    return le.transform(embarked)
    
    
plt.hist(normalize_embarked(df))
plt.show()
np.corrcoef(normalize_embarked(df), df['Survived'])
print("P Classes : {}".format(np.unique(df['Pclass'].values)))
display(np.corrcoef(df['Pclass'], df['Survived']))
plt.hist(df['Pclass'], bins=3)
from sklearn.preprocessing import LabelEncoder

def normalize_sex(df):
    sex_label_encoder = LabelEncoder()
    sex_label_encoder.fit(df['Sex'].values)
    sex_array = sex_label_encoder.transform(df['Sex'].values)
    return sex_array
    
    
plt.hist(normalize_sex(df), bins=2)
plt.show()
print(np.corrcoef(sex_array, df['Survived']))

def normalize_age(df):
    age  = df['Age'].values.copy()
    
    # find mean of age for all non-null values
    age_mean = np.mean(age[~df['Age'].isnull()]) 
    
    # Set unknown values to the mean age
    age[df['Age'].isnull()] = age_mean
    
    age_max = np.max(age)
    age_min = np.min(age)
    
    return (age - age_min) / (age_max - age_min)

def age_info(df):
    display(df['Age'].describe())
    
    age_df = df[~df['Age'].isnull()]
    print('% of non-null values: {}'.format(len(age_df) / len(df)))
    plt.hist(age_df['Age'])
    plt.show()

    age_10 = age_df[age_df['Age'] <= 10]
    print(np.corrcoef(age_10['Age'], age_10['Survived']))
    print(len(age_10[age_10['Survived'] == 1]) / len(age_10))

    age_20 = age_df[np.logical_and(age_df['Age'] > 10, age_df['Age'] <= 20)]
    print(np.corrcoef(age_20['Age'], age_20['Survived']))
    print(len(age_20[age_20['Survived'] == 1]) / len(age_20))

    age_40 = age_df[np.logical_and(age_df['Age'] > 20, age_df['Age'] <= 40)]
    print(np.corrcoef(age_40['Age'], age_40['Survived']))
    print(len(age_40[age_40['Survived'] == 1]) / len(age_40))

    age_50 = age_df[np.logical_and(age_df['Age'] > 40, age_df['Age'] <= 50)]
    print(np.corrcoef(age_50['Age'], age_50['Survived']))
    print(len(age_50[age_50['Survived'] == 1]) / len(age_50))

    age_60 = age_df[np.logical_and(age_df['Age'] > 50, age_df['Age'] <= 60)]
    print(np.corrcoef(age_60['Age'], age_60['Survived']))
    print(len(age_60[age_60['Survived'] == 1]) / len(age_60))

    age_80 = age_df[np.logical_and(age_df['Age'] > 60, age_df['Age'] <= 80)]
    print(np.corrcoef(age_80['Age'], age_80['Survived']))
    print(len(age_80[age_80['Survived'] == 1]) / len(age_80))
    
plt.hist(normalize_age(df), bins=4)
plt.show()
np.corrcoef(normalize_age(df), df['Survived'])

def normalize_fares(df):
    orig_fares = df['Fare'].values.copy()
    
    # find mean of age for all non-null values
    fares_mean = np.mean(orig_fares[~df['Fare'].isnull()])
    
    # Set unknown values to the mean age
    orig_fares[df['Fare'].isnull()] = fares_mean
    
    maxf = np.max(orig_fares)
    minf = np.min(orig_fares)
    
    return (orig_fares - minf) / (maxf - minf)
    
plt.hist(df['Fare'], bins=32)
plt.show()


np.corrcoef(df['Fare'].values, df['Survived'].values)
def build_features(df):
    n = len(df)
    # Features
    age_data = normalize_age(df)
    fare_data = normalize_fares(df)
    sex_data = normalize_sex(df)
    ticket_class_data = df['Pclass'].values.copy()
    parch_data = df['Parch'].values.copy()
    embarked_data = normalize_embarked(df)
    sibsp_data = df['SibSp'].values.copy()
    
    # Targets
    survival_data = df['Survived'].values.copy() if 'Survived' in df else []
    
    features = []
    for index in range(n):
        # age, fare, sex, ticket_class, parch (# parents + children on board), the port of embarkation, # of siblings + spouses
        feature_vec = [age_data[index],fare_data[index], sex_data[index], ticket_class_data[index], parch_data[index], embarked_data[index]]
        features.append(feature_vec)
    
    features = np.array(features)
    assert len(features) == n
    
    return features, survival_data


def test_train_split(df, train=0.8, test=0.2):
    np.random.seed(42)
    
    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    train_end_idx = int(train * n)
    
    features, survival_data = build_features(df)
    
    features_final = features[indices]
    survival_data_final = survival_data[indices]
    
    x_train = features_final[:train_end_idx]
    y_train = survival_data_final[:train_end_idx]
    x_test  = features_final[train_end_idx:]
    y_test  = survival_data_final[train_end_idx:]
    
    return x_train, y_train, x_test, y_test
    
x_train, y_train, x_test, y_test = test_train_split(df)
from sklearn import svm

def build_classifier(x,y):
    clf = svm.SVC()
    clf.fit(x,y)
    return clf

clf = build_classifier(x_train,y_train)
y_preds = clf.predict(x_test)
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

def print_stats(y_true, y_pred):
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    
print_stats(y_test, y_preds)
df_test = pd.read_csv("../input/test.csv")
test_features, _survival_data = build_features(df_test)
test_preds = clf.predict(test_features)
results = pd.DataFrame({ 'PassengerId' : df_test['PassengerId'].values, 'Survived' : test_preds })
results.head(10)
results.to_csv("./predictions1.csv", index=False)
test_features[0]
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

n_cols = len(test_features[0])

early_stopping_monitor = EarlyStopping(patience=2)

def build_model():
    model = Sequential()
    model.add(Dense(512, activation='relu',input_shape=(n_cols,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    
    return model

model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train,y_train, validation_data=(x_test, y_test), callbacks=[early_stopping_monitor], epochs=10, batch_size=16)
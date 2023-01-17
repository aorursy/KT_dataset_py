import numpy as np

import pandas as pd



import xgboost as xgb

import lime

import lime.lime_tabular



from sklearn.model_selection import cross_val_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#gender_submission.csv is an example prediction file predicting all female passengers survive, and no others do

#i guess thats politically correct

#!cat /kaggle/input/titanic/gender_submission.csv
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test.head()
df_train["train"] = 1

df_test["train"] = 0

df_all = pd.concat([df_train, df_test], sort=False)

df_all.head()
df_all["Embarked"].unique()
df_all["Embarked"] = df_all["Embarked"].fillna("N")
def parse_cabin_type(x):

    if pd.isnull(x):

        return None

    #print("X:"+x[0])

    #cabin id consists of letter+numbers. letter is the type/deck, numbers are cabin number on deck

    return x[0]



def parse_cabin_number(x):

    if pd.isnull(x):

        return -1

#        return np.nan

    cabs = x.split()

    cab = cabs[0]

    num = cab[1:]

    if len(num) < 2:

        return -1

        #return np.nan

    return num



def parse_cabin_count(x):

    if pd.isnull(x):

        return np.nan

    #a typical passenger has a single cabin but some had multiple. in that case they are space separated

    cabs = x.split()

    return len(cabs)



df_all["cabin_type"] = df_all["Cabin"].apply(lambda x: parse_cabin_type(x))

df_all["cabin_num"] = df_all["Cabin"].apply(lambda x: parse_cabin_number(x))

df_all["cabin_count"] = df_all["Cabin"].apply(lambda x: parse_cabin_count(x))

df_all["cabin_num"] = df_all["cabin_num"].astype(int)

df_all.head()
df_all["family_size"] = df_all["SibSp"] + df_all["Parch"] + 1

df_all.head()
# Cleaning name and extracting Title

for name_string in df_all['Name']:

    df_all['Title'] = df_all['Name'].str.extract('([A-Za-z]+)\.', expand=True)

df_all.head()
# Replacing rare titles 

mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 

           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 

           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 

           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}

           

df_all.replace({'Title': mapping}, inplace=True)

#titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']
titles = df_all["Title"].unique()

titles
titles = list(titles)

# Replacing missing age by median age for title 

for title in titles:

    age_to_impute = df_all.groupby('Title')['Age'].median()[titles.index(title)]

    df_all.loc[(df_all['Age'].isnull()) & (df_all['Title'] == title), 'Age'] = age_to_impute
df_all[df_all["Fare"].isnull()]
df_all.loc[152]
p3_median_fare = df_all[df_all["Pclass"] == 3]["Fare"].median()

p3_median_fare
df_all["Fare"].fillna(p3_median_fare, inplace=True)
df_all.loc[152]
df_all = df_all.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)
df_all["cabin_type"].value_counts()
df_all["cabin_type"] = df_all["cabin_type"].fillna("Z")
df_all["cabin_type"].value_counts()
label_encode_cols = ["Sex", "Embarked", "Title", "cabin_type"]
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in label_encode_cols:

    le = LabelEncoder()

    label_encoders[col] = le

    df_all[col] = le.fit_transform(df_all[col])

df_all.head()
cat_cols = label_encode_cols
for col in cat_cols:

    df_all[col] = df_all[col].astype('category')
df_all.isnull().sum()
df_all["cabin_count"] = df_all["cabin_count"].fillna(1)
df_all_oh = pd.get_dummies( df_all, columns = cat_cols )

df_all_oh.head()
df_all_oh.columns
df_train = df_all[df_all["train"] == 1]

df_test = df_all[df_all["train"] == 0]
df_train_oh = df_all_oh[df_all_oh["train"] == 1]

df_test_oh = df_all_oh[df_all_oh["train"] == 0]
df_train = df_train.drop("train", axis=1)

df_test = df_test.drop("train", axis=1)

df_train.head()

df_train_oh = df_train_oh.drop("train", axis=1)

df_test_oh = df_test_oh.drop("train", axis=1)

df_train_oh.head()

target = df_train["Survived"]

target.head()
df_train = df_train.drop("Survived", axis=1)

df_test = df_test.drop("Survived", axis=1)
df_train.head()
df_train_oh = df_train_oh.drop("Survived", axis=1)

df_test_oh = df_test_oh.drop("Survived", axis=1)
df_train_oh.head()
import lightgbm as lgb



l_clf = lgb.LGBMClassifier(

                        num_leaves=1024,

                        learning_rate=0.01,

                        n_estimators=5000,

                        boosting_type="gbdt",

                        min_child_samples = 100,

                        verbosity = 0)
x_clf = xgb.XGBClassifier()
import catboost



c_clf = catboost.CatBoostClassifier()
from sklearn.model_selection import train_test_split



X = df_train

y = target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)



X_oh = df_train_oh

X_train_oh, X_val_oh = train_test_split(X_oh, test_size=0.33, random_state=42)
df_train.dtypes
df_train_oh.dtypes
#the if True parts are just to make it simpler to disable some algorithm.



if True:

    l_clf.fit(

        X_train, y_train,

        eval_set=[(X_val, y_val)],

        eval_metric='mae',

        early_stopping_rounds=5,

        verbose=False

    )

    

if True:

    c_clf.fit(

        X_train, y_train,

        eval_set=[(X_val, y_val)],

        early_stopping_rounds=5,

        cat_features=cat_cols,

        verbose=False

    )



if True:

    x_clf.fit(

        X_train_oh, y_train,

        eval_set=[(X_val_oh, y_val)],

        early_stopping_rounds=5,

        verbose=False

    )
import matplotlib.pyplot as plt



def plot_feat_importance(clf, train):

    if hasattr(clf, 'feature_importances_'):

        importances = clf.feature_importances_

        features = train.columns



        feat_importances = pd.DataFrame()

        feat_importances["weight"] = importances

        feat_importances.index = features

        feat_importances.sort_values(by="weight", ascending=False).to_csv(f"top_features.csv")

        feat_importances.nlargest(30, ["weight"]).sort_values(by="weight").plot(kind='barh', title=f"top features", color='#86bf91', figsize=(10, 8))

        # kaggle shows output image files (like this png) under "output visualizations", others (such as pdf) under "output"

        plt.savefig(f'feature-weights.png')

        plt.savefig(f'feature-weights.pdf')

        plt.show()



def plot_pimp(pimps, train):

    importances = pimps.importances_mean

    features = train.columns



    feat_importances = pd.DataFrame()

    feat_importances["weight"] = importances

    feat_importances.index = features

    feat_importances.sort_values(by="weight", ascending=False).to_csv(f"top_features.csv")

    feat_importances.nlargest(30, ["weight"]).sort_values(by="weight").plot(kind='barh', title=f"top features", color='#86bf91', figsize=(10, 8))

    # kaggle shows output image files (like this png) under "output visualizations", others (such as pdf) under "output"

    plt.savefig(f'feature-weights.png')

    plt.savefig(f'feature-weights.pdf')

    plt.show()

plot_feat_importance(l_clf, X_train)
from sklearn.inspection import permutation_importance



l_pimps = permutation_importance(l_clf, X_train, y_train, n_repeats=10, random_state=0)

dir(l_pimps)
plot_pimp(l_pimps, X_train)
plot_feat_importance(c_clf, X_train)
c_pimps = permutation_importance(c_clf, X_train, y_train, n_repeats=10, random_state=0)

plot_pimp(c_pimps, X_train)
c_pimps.importances_mean
X_train.columns[np.argmin(c_pimps.importances_mean)]
plot_feat_importance(x_clf, X_train_oh)
x_pimps = permutation_importance(x_clf, X_train_oh, y_train, n_repeats=10, random_state=0)

plot_pimp(x_pimps, X_train_oh)
from sklearn.metrics import accuracy_score, log_loss



val_pred_proba = l_clf.predict_proba(X_val)

#val_pred = np.array(val_pred[:, 1] > 0.5)

val_pred = np.where(val_pred_proba > 0.5, 1, 0)



acc_score = accuracy_score(y_val, val_pred[:,1])

acc_score
val_pred_proba = c_clf.predict_proba(X_val)

#val_pred = np.array(val_pred[:, 1] > 0.5)

val_pred = np.where(val_pred_proba > 0.5, 1, 0)



acc_score = accuracy_score(y_val, val_pred[:,1])

acc_score
val_pred_proba = x_clf.predict_proba(X_val_oh)

#val_pred = np.array(val_pred[:, 1] > 0.5)

val_pred = np.where(val_pred_proba > 0.5, 1, 0)



acc_score = accuracy_score(y_val, val_pred[:,1])

acc_score
feature_names = list(df_train.columns)

feature_names
#cat_cols was set up earliner in the notebook to contain list of categorical feature/column names

cat_cols
#the corresponding indices of the cat_cols columns in the list of features

cat_indices = [feature_names.index(col) for col in cat_cols]

cat_indices
#mapping the category values to their names. for example, {"sex"={0="female", 1="male"}}

cat_names = {}

for label_idx in cat_indices:

    label = feature_names[label_idx]

    print(label)

    le = label_encoders[label]

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    le_value_mapping = dict(zip(le.transform(le.classes_), le.classes_))

    print(le_value_mapping)

    cat_names[label_idx] = le_value_mapping

cat_names #its actually the feature index mapped to the values and their names
explainer = lime.lime_tabular.LimeTabularExplainer(df_train.values, discretize_continuous=True,

                                                   class_names=['not survived', 'survived'], 

                                                   mode="classification",

                                                   feature_names = feature_names,

                                                   categorical_features=cat_indices,

                                                   categorical_names=cat_names, 

                                                   kernel_width=10, verbose=True)
#def row_to_df(row):

#    rows = []

#    rows.append(X_val.values[0])

#    df = pd.DataFrame(rows, columns=X_val.columns)

#    for col in cat_cols:

#        df[col] = df[col].astype('category')    

#    return df



#when LIME passes synthetic data to your predict function, it gives a list of N synthetic datapoints as a numpy matrix

#this converts that matrix into a dataframe, since some algorithms choke on the pure numpy array (catboost)

def rows_to_df(rows):

    df = pd.DataFrame(rows, columns=X_val.columns)

    #set category columns first to short numeric to save memory etc, then convert to categorical for catboost

    for col in cat_cols:

        df[col] = df[col].astype('int8')

        df[col] = df[col].astype('category')

    #and finally convert all non-categoricals to their original type. since we had to create a fresh dataframe this is needed

    for col in X_val.columns:

        if col not in cat_cols:

            df[col] = df[col].astype(X_val[col].dtype)

    return df



#for one-hot encoding the values from LIME, which uses numbers in a single column to represent categories

#needed for xgboost

def rows_to_df_oh(rows):

    df = pd.DataFrame(rows, columns=X_val_oh.columns)

    for col in cat_cols:

        df[col] = df[col].astype('int8')

    return df



#the function to pass to LIME for running catboost on the synthetic data

def c_run_pred(x):

    p = c_clf.predict_proba(rows_to_df(x))

    return p



#the function to pass to LIME to run LGBM on the synthetic data

def l_run_pred(x):

    p = l_clf.predict_proba(x)

    return p



#the function to pass to LIME to run XGBoost on the synthetic data

def x_run_pred(x):

    df = rows_to_df(x)

    df = pd.get_dummies( df, columns = cat_cols )



    new_df = pd.DataFrame()

    #this look ensure the column order of the dataframe created is same as the original

    for col in X_val_oh.columns:

        if col in df.columns:

            new_df[col] = df[col]

        else:

            #sometimes it seems to happen that a specific value is missing from a category in generation,

            #which leads to missing that column. this zeroes it to ensure it exists

            #print(f"missed col:{col}")

            new_df[col] = 0

    df = new_df



    p = x_clf.predict_proba(df)

    return p



c_predict_fn = lambda x: c_run_pred(x)



l_predict_fn = lambda x: l_run_pred(x)



x_predict_fn = lambda x: x_run_pred(x)

l_clf.predict_proba([X_val.values[0]])

c_predict_fn(X_val.values)[0]
x_predict_fn(X_val.values)[0]


x_clf.predict_proba(X_val_oh)[0]
#this demonstrates the missing value branch of x_run_pred() with cabin_type=7

df = rows_to_df(X_val.values)

print(df.shape)

print(f"cat cols: {cat_cols}")

df = pd.get_dummies( df, columns = cat_cols )

missing_cols = set( X_val_oh.columns ) - set( df.columns )

print(f"missing: {missing_cols}")

# Add a missing column in test set with default value equal to 0

new_df = pd.DataFrame()

for col in X_val_oh.columns:

    if col in df.columns:

        new_df[col] = df[col]

    else:

        new_df[col] = 0

df = new_df
#p = x_clf.predict_proba(df)

#p
#x_predict_fn(X_val.values)
def explain_item(predictor, item):

    exp = explainer.explain_instance(item, predictor, num_features=10, top_labels=1)

    exp.show_in_notebook(show_table=True, show_all=False)

#this allows running the experiments N times to see if the random synthetic value generation of LIME has some effect on the results over different runs.



def explain_x_times(x, idx, invert_gender=False):

    row = X_val.values[idx]

    if invert_gender:

        if row[1] > 0:

            row[1] = 0

        else:

            row[1] = 1

    print(f"columns={X_val.columns}")

    

    for i in range(x):

        print(f"Explaining LGBM: index={idx}, row={row}")

        explain_item(l_predict_fn, row)

    for i in range(x):

        print(f"Explaining CatBoost: index={idx}, row={row}")

        explain_item(c_predict_fn, row)

    for i in range(x):

        print(f"Explaining XGBoost: index={idx}, row={row}")

        explain_item(x_predict_fn, row)

explain_x_times(2, 0)
explain_x_times(2, 0, invert_gender=True)
explain_x_times(2, 1, invert_gender=False)
explain_x_times(2, 1, invert_gender=True)
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df_train.head()
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

            'BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',

           'PoolQC','Fence','MiscFeature'):

    df_train[col]=df_train[col].fillna('None')

    df_test[col]=df_test[col].fillna('None')



for col in ('Electrical','MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    df_train[col]=df_train[col].fillna(df_train[col].mode()[0])

    df_test[col]=df_test[col].fillna(df_train[col].mode()[0])



for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',

            'GarageYrBlt','GarageCars','GarageArea'):

    df_train[col]=df_train[col].fillna(0)

    df_test[col]=df_test[col].fillna(0)



df_train['LotFrontage']=df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())

df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_train['LotFrontage'].mean())
df_train.dtypes
#removing outliers recomended by author

df_train = df_train[df_train['GrLivArea']<4000]
len_traindf = df_train.shape[0]

houses = pd.concat([df_train, df_test], sort=False)

houses = houses.fillna(0)



# turning some ordered categorical variables into ordered numerical

# maybe this information about order can help on performance

for col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",

            "FireplaceQu","GarageQual","GarageCond","PoolQC"]:

    houses[col]= houses[col].map({"Gd": 4 , "TA": 3, "Ex": 5, "Fa":2, "Po":1})

houses = houses.fillna(0)
import numbers



cdf = df_train.select_dtypes(include=np.number)

cat_names = [key for key in df_train.columns if key not in cdf.columns]

cat_names
len_traindf = df_train.shape[0]



from sklearn.preprocessing import LabelEncoder



label_encoders = {}

for col in cat_names:

    le = LabelEncoder()

    label_encoders[col] = le

    houses[col] = le.fit_transform(houses[col])



df_train = houses[:len_traindf]

df_train = df_train.drop('SalePrice', axis=1)

df_test = houses[len_traindf:]

df_test = df_test.drop('SalePrice', axis=1)



# turning categoric into numeric

houses_oh = pd.get_dummies(houses)



# separating

df_train_oh = houses_oh[:len_traindf]

df_test_oh = houses_oh[len_traindf:]
# x/y split

X_train_oh = df_train_oh.drop('SalePrice', axis=1)

y_train = df_train_oh['SalePrice']

X_test_oh = df_test_oh.drop('SalePrice', axis=1)
from hyperopt import hp, tpe, fmin



space = {'n_estimators':hp.quniform('n_estimators', 1000, 4000, 100),

         'gamma':hp.uniform('gamma', 0.01, 0.05),

         'learning_rate':hp.uniform('learning_rate', 0.00001, 0.03),

         'max_depth':hp.quniform('max_depth', 3,7,1),

         'subsample':hp.uniform('subsample', 0.60, 0.95),

         'colsample_bytree':hp.uniform('colsample_bytree', 0.60, 0.95),

         'colsample_bylevel':hp.uniform('colsample_bylevel', 0.60, 0.95),

         'reg_lambda': hp.uniform('reg_lambda', 1, 20)

        }



def objective(params):

    params = {'n_estimators': int(params['n_estimators']),

             'gamma': params['gamma'],

             'learning_rate': params['learning_rate'],

             'max_depth': int(params['max_depth']),

             'subsample': params['subsample'],

             'colsample_bytree': params['colsample_bytree'],

             'colsample_bylevel': params['colsample_bylevel'],

             'reg_lambda': params['reg_lambda']}

    

    xb_a = xgb.XGBRegressor(**params)

    score = cross_val_score(xb_a, X_train_oh, y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1).mean()

    return -score
best = fmin(fn= objective, space= space, max_evals=4, rstate=np.random.RandomState(1), algo=tpe.suggest)

#max_evals=20
X_clf = xgb.XGBRegressor(random_state=0,

                        n_estimators=int(best['n_estimators']), 

                        colsample_bytree= best['colsample_bytree'],

                        gamma= best['gamma'],

                        learning_rate= best['learning_rate'],

                        max_depth= int(best['max_depth']),

                        subsample= best['subsample'],

                        colsample_bylevel= best['colsample_bylevel'],

                        reg_lambda= best['reg_lambda']

                       )



X_clf.fit(X_train_oh, y_train)
all_cols = list(df_train.columns)

cat_indices = []

for cat_name in cat_names:

    cat_indices.append(all_cols.index(cat_name))

print(cat_indices)
houses.head()

feature_names = list(df_train.columns)

print(feature_names)
cat_names = {}

for label_idx in cat_indices:

    label = feature_names[label_idx]

    print(label)

    le = label_encoders[label]

    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    le_value_mapping = dict(zip(le.transform(le.classes_), le.classes_))

    print(le_value_mapping)

    cat_names[label_idx] = le_value_mapping
feature_names_oh = list(df_train_oh.columns)
explainer = lime.lime_tabular.LimeTabularExplainer(df_train.values, 

                                                   feature_names=feature_names, 

                                                   class_names=['price'], 

                                                   categorical_features=cat_indices,

                                                   categorical_names=cat_names,

                                                   verbose=True, 

                                                   discretize_continuous=False,

                                                   mode='regression')

def explain_xreg(row):

    df = pd.DataFrame(data=row, columns=df_train.columns)

    row = pd.get_dummies(df)

    return X_clf.predict(row)
def explain_item(item):

    exp = explainer.explain_instance(item, explain_xreg, num_features=10, top_labels=1)

    exp.show_in_notebook(show_table=True, show_all=False)

    return exp
df_test.iloc[0]
df_test.iloc[0].values.shape
explain_xreg([df_test.iloc[0].values])
exp = explain_item(df_test.iloc[0])

df_test.head()
df_test.describe()
top_features = exp.as_list()

top_features
for feat, weight in top_features:

    print(feat)
%matplotlib inline

top_names = [tup[0].split("=")[0] for tup in top_features]

df_test[top_names].hist(figsize=(15,10))
df_test[top_names].head(1)
y_train.describe()
df_test.iloc[0]["KitchenQual"]
row = df_test.iloc[0]

row["KitchenQual"] = 2
row["KitchenQual"]
exp = explain_item(row)
!ls ../input
# read the csv

cleveland = pd.read_csv('../input/heart-disease-uci/heart.csv')
# remove missing data (indicated with a "?")

data = cleveland[~cleveland.isin(['?'])]
#drop nans

data = data.dropna(axis=0)
data = data.apply(pd.to_numeric)

data.dtypes
X = np.array(data.drop(['target'], 1))

y = np.array(data['target'])
from sklearn import model_selection



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, stratify=y, random_state=42, test_size = 0.2)
# convert the data to categorical labels

from keras.utils.np_utils import to_categorical



Y_train = to_categorical(y_train, num_classes=None)

Y_test = to_categorical(y_test, num_classes=None)

print (Y_train.shape)

print (Y_train[:10])
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.layers import Dropout

from keras import regularizers



# define a function to build the keras model

def create_model():

    # create model

    model = Sequential()

    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

    

    # compile model

    adam = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model



model = create_model()



print(model.summary())
# fit the model to the training data

#verbose=1 for full output, verbose=2 for list of epochs. 0 for quiet

history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=50, batch_size=10, verbose=0)
#to see training results, exact accuracy and loss

#history.history
import matplotlib.pyplot as plt

%matplotlib inline

# Model accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# Model Losss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# convert into binary classification problem - heart disease or no heart disease

Y_train_binary = y_train.copy()

Y_test_binary = y_test.copy()



Y_train_binary[Y_train_binary > 0] = 1

Y_test_binary[Y_test_binary > 0] = 1



print(Y_train_binary[:20])
# define a new keras model for binary classification

def create_binary_model():

    # create model

    model = Sequential()

    model.add(Dense(16, input_dim=13, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(8, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(1, activation='sigmoid'))

    

    # Compile model

    adam = Adam(lr=0.001)

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model



binary_model = create_binary_model()



print(binary_model.summary())
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor



binary_model = KerasClassifier(build_fn=create_binary_model, epochs=50, batch_size=10, verbose=0)

# fit the binary model on the training data

history=binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary), epochs=50, batch_size=10, verbose=0)
import matplotlib.pyplot as plt

%matplotlib inline

# Model accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# Model Losss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()


def plot_pimp_2(pimps, features):

    importances = pimps.importances_mean



    feat_importances = pd.DataFrame()

    feat_importances["weight"] = importances

    feat_importances.index = features

    feat_importances.sort_values(by="weight", ascending=False).to_csv(f"top_features.csv")

    feat_importances.nlargest(30, ["weight"]).sort_values(by="weight").plot(kind='barh', title=f"top features", color='#86bf91', figsize=(10, 8))

    # kaggle shows output image files (like this png) under "output visualizations", others (such as pdf) under "output"

    plt.savefig(f'feature-weights.png')

    plt.savefig(f'feature-weights.pdf')

    plt.show()

cleveland.shape
df_X = data.drop(['target'], 1)

k_pimps = permutation_importance(binary_model, X_train, y_train, n_repeats=10, random_state=0)

plot_pimp_2(k_pimps, df_X.columns)
# generate classification report using predictions for categorical model

from sklearn.metrics import classification_report, accuracy_score



categorical_pred = np.argmax(model.predict(X_test), axis=1)



print('Results for Categorical Model')

print(accuracy_score(y_test, categorical_pred))

print(classification_report(y_test, categorical_pred))
#model.predict_proba(X_test)

feature_names = cleveland.columns
data.nunique()
#data["thalach"].describe()
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "ca", "thal"]

for col in cat_cols:

    print(f"{col}: {data[col].unique()}")
feature_names = list(feature_names)

cat_indices = [feature_names.index(col) for col in cat_cols]

cat_indices
#explainer = lime.lime_tabular.LimeTabularExplainer(df_train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=True,

                                                   class_names=['no risk', 'risk of heart'], 

                                                   mode="classification",

                                                   feature_names = feature_names,

                                                   categorical_features=cat_indices,

                                                   categorical_names=[], 

                                                   kernel_width=10, verbose=True)

def explain_item(item):



    #    exp = explainer.explain_instance(item, l_predict_fn, num_features=10, top_labels=1)

    exp = explainer.explain_instance(item, model.predict_proba, num_features=10, top_labels=1)

#    exp = explainer.explain_instance(item, l_clf.predict_proba, num_features=10, top_labels=1)

    exp.show_in_notebook(show_table=True, show_all=False)
def explain_item_flipped(row, flip=False):

    if flip:

        if row[2] > 0:

            row[2] = 0

        else:

            row[2] = 1

    print(f"columns={X_val.columns}")

    #    exp = explainer.explain_instance(item, l_predict_fn, num_features=10, top_labels=1)

    exp = explainer.explain_instance(row, model.predict_proba, num_features=10, top_labels=1)

#    exp = explainer.explain_instance(item, l_clf.predict_proba, num_features=10, top_labels=1)

    exp.show_in_notebook(show_table=True, show_all=False)

cat_cols
cat_indices
explain_item(X_test[0])
explain_item_flipped(X_test[0], False)
explain_item_flipped(X_test[1], False)
explain_item_flipped(X_test[0], True)
explain_item_flipped(X_test[1], True)
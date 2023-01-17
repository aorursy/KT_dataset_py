import matplotlib.pyplot as plt # For plots
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.info()
df.describe(include='all')
import pandas_profiling
report = pandas_profiling.ProfileReport(df)
display(report)
# Creating bins
df['AgeBins'] = pd.cut(df['Age'],5)
df['FareBins'] = pd.qcut(df['Fare'],5)

# Family related
df['FamilySize'] = df['Parch'] + df['SibSp']
df['IsAlone'] = df['FamilySize']==0

# Create a feature with the title extracted from the name
df['Title'] = df['Name'].apply(lambda x: x.split(' ')[1])
title_counts = df['Title'].value_counts()
min_title_count = 6
titles = title_counts[title_counts >= min_title_count].index.tolist()
df['Title'] = df['Title'].apply(lambda x: x if x in titles else 'Misc' )
df.info()
df.describe(include='all')
cat_attributes = ['Sex','Embarked','Title','AgeBins','FareBins','IsAlone']
num_attributes = ['Pclass','Age','SibSp','Parch','Fare','FamilySize']
target = ['Survived']
df = df.astype(dict(zip(cat_attributes,['category']*len(cat_attributes))))
import seaborn as sns
sns.pairplot(df[num_attributes + target], hue=target[0], diag_kind="hist", palette = 'deep')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
def get_preprocessing_pipeline(num_attributes, cat_attributes):
    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='mean')),
        ('std_scaler',StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessing_pipeline = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_attributes),
        ('cat_pipeline', cat_pipeline, cat_attributes)
    ])
    return preprocessing_pipeline
from sklearn.model_selection import train_test_split
y = df[['Survived']].copy()
X = df[num_attributes + cat_attributes].copy()
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)
preprocessing_pipeline = get_preprocessing_pipeline(num_attributes, cat_attributes)
X_train_ready = preprocessing_pipeline.fit_transform(X_train)
X_test_ready = preprocessing_pipeline.transform(X_test)
X_ready = preprocessing_pipeline.transform(X)
preprocessing_pipeline.named_transformers_['cat_pipeline'].named_steps['encoder'].get_feature_names()
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import scipy 
class ModeEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y):
        X, y = check_X_y(X,y)
        self.mode_ = scipy.stats.mode(y)[0][0]
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        prediction = np.tile(self.mode_, (X.shape[0],1))
        return prediction
# Evaluating in test set
model = ModeEstimator()
model.fit(X_train_ready,y_train.values.ravel())
score = model.score(X_test_ready,y_test)
print('accuracy: ',score)
# Evaluating using cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,X_ready,y.values.ravel(),cv=50)
#print(cv_accuracy)
print('mean_accuracy: ',scores.mean())
print('accuracy_std*3:',3*scores.std())
plt.hist(scores);
from sklearn.linear_model import LogisticRegression
# Evaluating in test set
model = LogisticRegression()
model.fit(X_train_ready,y_train.values.ravel())
score = model.score(X_test_ready,y_test)
print('test accuracy: ',score)

# Evaluating using cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,X_ready,y.values.ravel(),cv=50)
#print(cv_accuracy)
print('cv mean_accuracy: ',scores.mean())
print('cv accuracy_std*3:',3*scores.std())
plt.hist(scores);
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
models = [
    # Trees
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    # Ensemble
    RandomForestClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    GradientBoostingClassifier(),
    ExtraTreeClassifier(),
    # Linear Models
    LogisticRegression(),
    RidgeClassifier(),
    Perceptron(),
    # SVM
    SVC(probability=True),
    LinearSVC(max_iter=3000),
    # XGBoost
    XGBClassifier()
]
from sklearn.model_selection import cross_validate
training_summary = pd.DataFrame(columns=['test_score_mean','test_score_3_std','train_score_mean','train_score_3_std','model_name','model_params'])
for model in models:
    print(model.__class__.__name__)
    cv = cross_validate(model,X_ready,y.values.ravel(),cv=10,return_train_score=True)
    new_line = pd.Series({
        'test_score_mean': cv['test_score'].mean(),
        'test_score_3_std': 3*cv['test_score'].std(),
        'train_score_mean': cv['train_score'].mean(),
        'train_score_3_std': 3*cv['train_score'].std(),
        'model_name': model.__class__.__name__,
        'model_params': model.get_params()
    })
    training_summary = training_summary.append(new_line,ignore_index=True)
training_summary.sort_values(by='test_score_mean',ascending=False)
training_summary.sort_values(by='test_score_mean',ascending=False)
from sklearn.model_selection import GridSearchCV
model = SVC()
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'C': [0.1, 1, 10], 'probability': [True, False]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'C': [0.01, 0.05, 0.1, 0.5, 1, 5], 'kernel': ['rbf','linear']},
  ]
grid_search = GridSearchCV(model,param_grid=param_grid,cv=5,return_train_score=True)
grid_search.fit(X_ready,y.values.ravel())
grid_search.best_params_
best_model = grid_search.best_estimator_
cvres = grid_search.cv_results_
print("--------------MEAN-SCORE--------------------")
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print("{} {}".format(mean_score, params))

print("---------------3-*-STD--------------------")
for std_score, params in zip(cvres["std_test_score"], cvres["params"]):
    print("{} {}".format(3*std_score, params))
    
best_model
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
# Creating bins
df_test['AgeBins'] = pd.cut(df_test['Age'],5)
df_test['FareBins'] = pd.qcut(df_test['Fare'],5)

# Family related
df_test['FamilySize'] = df_test['Parch'] + df_test['SibSp']
df_test['IsAlone'] = df_test['FamilySize']==0

# Create a feature with the title extracted from the name
df_test['Title'] = df_test['Name'].apply(lambda x: x.split(' ')[1])
title_counts = df_test['Title'].value_counts()
min_title_count = 6
titles = title_counts[title_counts >= min_title_count].index.tolist()
df_test['Title'] = df_test['Title'].apply(lambda x: x if x in titles else 'Misc' )
X_test = df_test[num_attributes + cat_attributes].copy()
X_test_ready = preprocessing_pipeline.transform(X_test)
result_df = pd.DataFrame({"PassengerId":df_test['PassengerId'].values,"Survived":best_model.predict(X_test_ready)})
! rm './predictions.csv'
result_df.to_csv('./predictions.csv',header=True,index=False)
import tensorflow as tf
# Defining the model architecture based in an autoencoder
encoder = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation='selu'),
                                      tf.keras.layers.Dropout(rate=0.1),
                                      tf.keras.layers.Dense(16, activation='selu'),
                                      tf.keras.layers.Dropout(rate=0.1),
                                      tf.keras.layers.Dense(8, activation='selu')])

decoder = tf.keras.models.Sequential([tf.keras.layers.Dense(16, activation='selu'),
                                      tf.keras.layers.Dropout(rate=0.1),
                                      tf.keras.layers.Dense(30, activation='selu')])

model_top = tf.keras.models.Sequential([tf.keras.layers.Dense(8, activation='selu'),
                                        tf.keras.layers.Dense(4, activation='selu'),
                                        tf.keras.layers.Dense(1, activation='sigmoid')])

# Defining model tensors
input_ = tf.keras.layers.Input(shape=(30,))
encoder_output = encoder(input_)
output1 = decoder(encoder_output)
output2 = model_top(encoder_output)

# Defining and compiling model
model = tf.keras.models.Model(inputs=input_,outputs=[output1, output2])
model.compile(loss=["mse","binary_crossentropy"],loss_weights=[0.05,0.95], optimizer="adam")
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
callbacks = [EarlyStopping(patience=10)]
history = model.fit(X_train_ready, y_train, epochs=500, validation_split=0.15, callbacks=callbacks)
#MC dropout
y_test_est = np.array([])
for x in X_test_ready:
    y_probas = np.stack([model(np.array([x]), training=True)[-1][0] for _ in range(50)])
    y_test_est = np.append(y_test_est, 1*(y_probas.mean()>0.5))
result_df = pd.DataFrame({"PassengerId":df_test['PassengerId'].values,"Survived":y_test_est.astype(int)})
! rm './predictions.csv'
result_df.to_csv('./predictions.csv',header=True,index=False)

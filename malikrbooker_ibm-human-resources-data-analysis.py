import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import gc
url = 'https://raw.githubusercontent.com/malikrb/HumanResourcesDemonstration/master/data/hr_data.csv'
df = pd.read_csv(url)
with pd.option_context('display.max_columns', 35):
    display(df.head(6))
# Bin the Data into ['Close', 'Middle', 'Far']
temp = df.copy()
temp['DistanceFromHome'] = pd.cut(temp['DistanceFromHome'], 3, labels=['Closest', 'Middle', 'Farthest'])

temp = pd.concat([temp, pd.get_dummies(temp['Attrition'], prefix='Attrition')], axis=1)

# Visualize Attrition
display(temp.groupby(['DistanceFromHome'])[['Attrition_No', 'Attrition_Yes']].sum())
display(temp.groupby(['Department'])[['Attrition_No', 'Attrition_Yes']].sum())
print('--------------------------------------------')
display(temp.groupby(['JobRole'])[['Attrition_No', 'Attrition_Yes']].sum())
print('-------------------------------------------------------------------')
display(temp.groupby(['Department', 'JobRole'])[['Attrition_No', 'Attrition_Yes']].sum())

del temp
gc.collect();
from sklearn.preprocessing import LabelEncoder

df['Attrition'] = LabelEncoder().fit_transform(df['Attrition'])
df = df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1)

with pd.option_context('display.max_columns', 35):
    display(df.head())
plt.style.use('seaborn')

df.hist(bins=30, figsize=(22,22), edgecolor='w')

plt.show()
f, ax = plt.subplots(1, 1, figsize=(18,8))
sns.countplot(x='Age', hue='Attrition', data=df,
              edgecolor='w', linewidth=1.15)
plt.title('Attrition by Age', size=26, y=1.05)
plt.xlabel('Age', size=20)
plt.ylabel('Count', size=20)

plt.legend(['Stayed', 'Left'])

plt.show()
f, axes = plt.subplots(4, 1, figsize=(18,18), tight_layout=True)

x_vars = ['JobRole', 'MaritalStatus', 'JobLevel', 'StockOptionLevel']
for ax, x in zip(axes, x_vars):
    sns.countplot(x=x, hue='Attrition', data=df, ax=ax)
    ax.set_xlabel(x, size=16)
    ax.set_ylabel("")
    ax.legend(['Stayed', 'Left'])

f.text(x=-0.0275, y=0.5, s='Count', rotation=90, size=22)

plt.show()

del x_vars
gc.collect();
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6), tight_layout=True)

# Split the DataFrame by Attrition
attrition_0 = df.loc[df['Attrition'] == 0]
attrition_1 = df.loc[df['Attrition'] == 1]

sns.kdeplot(attrition_0['DistanceFromHome'], label='Stayed',
            shade=True, ax=ax1, alpha=0.6)
sns.kdeplot(attrition_1['DistanceFromHome'], label='Left',
            shade=True, ax=ax1, alpha=0.6)
ax1.set_xlabel('Distance From Home', size=18)

sns.kdeplot(attrition_0['YearsWithCurrManager'], label='Stayed',
            shade=True, ax=ax2, alpha=0.6)
sns.kdeplot(attrition_1['YearsWithCurrManager'], label='Left',
            shade=True, ax=ax2, alpha=0.6)
ax2.set_xlabel('Years With Current Manager', size=18)

plt.show()

del attrition_0, attrition_1
gc.collect();
# Encode all 'object' columns
columns = df.select_dtypes(include='object').columns
for col in columns:
    df[col] = LabelEncoder().fit_transform(df[col])
    
df = pd.get_dummies(df)
    
del columns
gc.collect();
df
# Splitting the DataFrame into train test
from sklearn.model_selection import train_test_split

X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from multiprocessing import cpu_count

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data, 
                       model, param_grid, cv=10, scoring_fit='accuracy',
                       do_probabilities=False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv,  
        scoring=scoring_fit,
        verbose=2,
        n_jobs=cpu_count()//2,
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    
    score = accuracy_score(pred, y_test)
    
    return fitted_model, pred, score
from xgboost import XGBClassifier

param_grid = {
    'colsample_bytree': [0.7],
    'learning_rate': [0.01],
    'max_depth': [5],
    'n_estimators': [500],
    'reg_alpha': [1.1],
    'reg_lambda': [1.2],
    'subsample': [0.8],
#     'colsample_bytree': [0.7, 0.8],
#     'learning_rate': [0.01, 0.05],
#     'n_estimators': [500, 1000],
#     'max_depth': [5, 10],
#     'reg_alpha': [1.1, 1.2, 1.3],
#     'reg_lambda': [1.1, 1.2, 1.3],
#     'subsample': [0.7, 0.8, 0.9]
}

model = XGBClassifier()

xgb_model, xgb_pred, xgb_score = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                        param_grid, cv=5, scoring_fit='accuracy')

print(xgb_model.best_score_)
print(xgb_model.best_params_)
print(f'xgb_score: {xgb_score}')
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model(activation='relu', optimizer='adam', dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(500, activation=activation, input_shape=(30, )))
    model.add(Dense(500, activation=activation))
    model.add(Dense(500, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

param_grid = {
    'batch_size':   [50],
    'epochs':       [125],
    # 'dropout_rate': [0.2, 0.3],
    # 'activation':   ['relu', 'elu'],
    # 'batch_size':   [50, 100, 150],
    # 'optimizer':    ['Adam', 'Nadam'],
    # 'epochs':       [25, 75, 125],
}

model = KerasClassifier(build_fn=build_model, verbose=0)

nn_model, nn_pred, nn_score = algorithm_pipeline(X_train, X_test, y_train, y_test, model, 
                                        param_grid, cv=5, scoring_fit='accuracy')

print(nn_model.best_score_)
print(nn_model.best_params_)
print(f'nn_score: {nn_score}')
features = xgb_model.best_estimator_.feature_importances_
columns = X.columns
sorted_features = sorted(zip(columns, features), key=lambda x: x[1], reverse=True)

## Uncomment lines below to see feature importances
## with respective column
# for col, feature in sorted_features:
#     print(f'{col}: {feature}')
plt.subplots(1, 1, figsize=(14,6), tight_layout=True)

x = [x[0] for x in sorted_features]
height = [x[1] for x in sorted_features]

plt.bar(x=x, height=height)
plt.title('Feature Importances', size=22, y=1.05)
plt.xticks(rotation=90, ha='right')
plt.show()
from sklearn.metrics import roc_curve, auc

xgb_prob = xgb_model.predict_proba(X_test)[:,1]
fpr, tpr, thresh = roc_curve(y_test, xgb_prob)
roc_auc = auc(fpr, tpr)

plt.subplots(1, 1, figsize=(6,6), tight_layout=True)

plt.title(f'Recieving Operator Characteristics', size=16, y=1.05)
plt.plot(fpr, tpr, label=f'XBGClassifier: {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--', alpha=0.6)

plt.legend(frameon=1, facecolor='w')

plt.show()
from sklearn.metrics import confusion_matrix

mxN = confusion_matrix(y_test, xgb_pred)
plt.subplots(1, 1, figsize=(6,6), tight_layout=True)

plt.title('Confusion Matrix', size=18, y=1.025)
sns.heatmap(mxN, annot=True, fmt='d',
            cmap='Blues', cbar=False,
            xticklabels=['Stayed', 'Left'],
            yticklabels=['Stayed', 'Left']
           )

plt.xlabel('Actual', size=16)
plt.ylabel('Predicted', size=16)
plt.yticks(rotation=0)

plt.show()
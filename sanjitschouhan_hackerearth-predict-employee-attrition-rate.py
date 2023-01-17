import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from IPython.display import display, Markdown, Latex
train_data = pd.read_csv("../input/Dataset/Train.csv", index_col="Employee_ID")

X_test_raw = pd.read_csv("../input/Dataset/Test.csv", index_col="Employee_ID")
y_train_raw = train_data.Attrition_rate

X_train_raw = train_data.drop("Attrition_rate", axis=1)
sns.distplot(y_train_raw)

plt.title("Distribution of Attrition Rate")
def fill_missing(data):

    df = pd.DataFrame.copy(data)

    df['Pay_Scale'] = df['Pay_Scale'].fillna(df.groupby(['Age', 'Education_Level', 'Time_of_service'])['Pay_Scale'].transform('median'))

    df['Time_of_service'] = df['Time_of_service'].fillna(df.groupby(['Age', 'Education_Level', 'Pay_Scale'])['Time_of_service'].transform('median'))

    df['Age'] = df['Age'].fillna(df.groupby(['Education_Level', 'Relationship_Status', 'Time_of_service'])['Age'].transform('median'))

    df = df.fillna(df.median())

    return df
X_train_no_na = fill_missing(X_train_raw)

X_test_no_na = fill_missing(X_test_raw)
numerical_cols = list(X_train_no_na.describe().columns)

plt.subplots(4,4, figsize=(20,20))

i = 1

for col in numerical_cols:

    plt.subplot(4, 4, i)

    try:

        plt.hist(X_train_no_na[col])

        plt.title(col)

    finally:

        i += 1
numeric_categoricals = [

 'Education_Level',

 'Time_since_promotion',

 'Travel_Rate',

 'Post_Level',

 'Work_Life_balance',

 'VAR1',

 'VAR2',

 'VAR3',

 'VAR4',

 'VAR5',

 'VAR6',

 'VAR7']



non_numeric_categoricals = [x for x in X_train_no_na.columns if x not in numerical_cols]
def convert_categoricals(data):

    df = pd.DataFrame.copy(data)

    for col in numeric_categoricals:

        df[col] = pd.Categorical(df[col])

    for col in non_numeric_categoricals:

        df[col] = pd.Categorical(df[col])

    return df
X_train_cats = convert_categoricals(X_train_no_na)

X_test_cats = convert_categoricals(X_test_no_na)


output = "| Categorial Column | Train Categories | Test Categories | Equal |"

output += "\n|:--|:--|:--|:--|"

for col in numeric_categoricals + non_numeric_categoricals:

    output += "\n|" 

    output += col 

    output += "|"

    output += str(sorted(X_train_cats[col].unique()))

    output += "|"

    output += str(sorted(X_test_cats[col].unique()))

    output += "|"

    output += ["No","Yes"][sorted(X_train_cats[col].unique())==sorted(X_test_cats[col].unique())]

    output += "|"



display(Markdown(output))
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

numeric_features = list(X_train_cats.describe().columns)



X_train_normalized = pd.DataFrame.copy(X_train_cats)

X_train_normalized[numeric_features] = scaler.fit_transform(X_train_cats[numeric_features])



X_test_normalized = pd.DataFrame.copy(X_test_cats)

X_test_normalized[numeric_features] = scaler.transform(X_test_cats[numeric_features])
X_train_normalized.describe()
X_train_one_hot = pd.get_dummies(X_train_normalized)

X_test = pd.get_dummies(X_test_normalized)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train_one_hot, y_train_raw, test_size=0.1, random_state=42)
model_type = 'boost'

if model_type == 'boost':

    from sklearn.ensemble import AdaBoostRegressor

    from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

    from sklearn.model_selection import GridSearchCV



    cls = AdaBoostRegressor(random_state=42, n_estimators=100)



    params = {

        'learning_rate': [0.001, 0.01, 0.1, 1, 10,100],

        'loss': ['linear', 'square', 'exponential']

    }



    scorer = make_scorer(mean_squared_error, greater_is_better=False)



    grid_search = GridSearchCV(cls, param_grid=params, scoring=scorer, verbose=3)

    grid_search.fit(X_train, y_train)



    best_cls = grid_search.best_estimator_

else:

    from sklearn.svm import SVR

    from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

    from sklearn.model_selection import GridSearchCV



    cls = SVR()



    params = {

        'kernel': ['linear', 'poly', 'rbf'],

        'degree': [1,2,3,4,5],

        'gamma': [10,20,30,50,100],

        'C': [0.1, 0.5, 1, 5, 10]

    }



    scorer = make_scorer(mean_squared_error, greater_is_better=False)



    grid_search = GridSearchCV(cls, param_grid=params, scoring=scorer, verbose=3)

    grid_search.fit(X_train, y_train)



    best_cls = grid_search.best_estimator_
pred_train = best_cls.predict(X_train)

print("[Training]Mean Squared Error:", mean_squared_error(y_train, pred_train))

print("[Training]Mean Absolute Error:", mean_absolute_error(y_train, pred_train))



pred_val = best_cls.predict(X_val)

print("[Validation]Mean Squared Error:", mean_squared_error(y_val, pred_val))

print("[Validation]Mean Absolute Error:", mean_absolute_error(y_val, pred_val))
RMSE = mean_squared_error(y_val, pred_val)**0.5

print("RMSE:", RMSE)

score = 100 * max(0, 1-RMSE)

print("Score:", score)
pred_test = pd.DataFrame(best_cls.predict(X_test), columns=['Attrition_rate'], index=X_test.index)

pred_test
pred_test.to_csv("Submission.csv")
import numpy as np 

import pandas as pd





import matplotlib.pyplot as plt

from catboost import CatBoostClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
# Check na values 

na_cols = df.columns[df.isna().sum() > 0]

print(na_cols)
# Convert columns with na values to object type and fill na with mode 

for col in na_cols:

    df[col] = df[col].astype('object')

    df[col].fillna(df[col].mode()[0], inplace=True)
# Split into features and target

X, y = df.drop('is_canceled', axis=1), df['is_canceled']



# Keep column names for catboost cat_features

X_cols = df.drop('is_canceled', axis=1).columns
%%time

kf = KFold(n_splits=5)

for train_index, valid_index in kf.split(X, y):

    X_train, X_valid = X.values[train_index], X.values[valid_index]

    y_train, y_valid = y.values[train_index], y.values[valid_index]

    

    X_train, X_valid = pd.DataFrame(X_train, columns=X_cols), pd.DataFrame(X_valid, columns=X_cols)

    

    cbc = CatBoostClassifier(cat_features=df.select_dtypes(include='object'), silent=True)

    cbc.fit(X_train, y_train)

    fpr, tpr, thr = roc_curve(y_valid, cbc.predict_proba(X_valid)[:, 1])

    print(f'accuracy: {accuracy_score(y_valid, cbc.predict(X_valid))}')

    plt.plot(fpr, tpr)

#     plt.show()

plt.xlabel('FPR')

plt.ylabel('TPR')

    
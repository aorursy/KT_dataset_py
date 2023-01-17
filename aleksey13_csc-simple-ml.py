# Можно зафиксировать seed рандома для numpy numpy.random.seed(0)

%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

df = df[['hotel', 'is_canceled', 'lead_time', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
    'stays_in_weekend_nights','stays_in_week_nights', 'adults', 'children', 'babies', 'meal']]
df['is_canceled'].value_counts(dropna=False) / len(df)
df['hotel'] = df['hotel'].astype('category')
df['arrival_date_month'] = df['arrival_date_month'].astype('category')
df['meal'] = df['meal'].astype('category')
df
# plt.figure(figsize=(20, 10))
# sns.heatmap(df.corr(), annot = True, fmt='.4g',cmap= 'coolwarm')
# plt.show()
# df.value_counts(dropna=False)
# df.columns
# df.info()
# df.head()
# df.describe()

# df.hist(layout=(3,4), figsize=(20,10))
# df['meal'].hist()
# plt.show()

# plt.figure(figsize=(20, 10))
# sns.heatmap(df.corr(), annot = True, fmt='.4g',cmap= 'coolwarm')
# plt.show()
# ???
# Category
prep_df = df[['hotel', 'lead_time', 'arrival_date_month','stays_in_week_nights','stays_in_weekend_nights', 'adults', 'children', 'babies', 'meal']]
target = df['is_canceled']

# One-hot encoding
prep_df = pd.get_dummies(prep_df, columns=['hotel','meal','arrival_date_month'],drop_first=True)
prep_df
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(prep_df,target, test_size=0.2, random_state=2)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def create_simple_pipeline():
    return Pipeline([
        ("fill_nan", SimpleImputer(missing_values=np.nan, strategy='mean')), 
        ("scale", StandardScaler()), 
        ("clf", LogisticRegression())
    ])

# Pipeline
simple_clf_pipline = create_simple_pipeline()

# Train
simple_clf_pipline.fit(X_train, y_train)

# Test
y_pred = simple_clf_pipline.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(f"Accuracy: {score}")

# 0.629

# 0.672
# 0.678
from sklearn.model_selection import KFold

def estimate_model(clf_pipline,data_df,target):
    k_fold = KFold(n_splits=10, shuffle=False)
    scores = []
    X = prep_df.to_numpy()
    y = target.to_numpy()
    for train_index, test_index in k_fold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train
        clf_pipline.fit(X_train, y_train)

        # Test
        y_pred = clf_pipline.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
    return scores

simple_clf_pipline = create_simple_pipeline()
scores = estimate_model(simple_clf_pipline,prep_df,target)
print("Accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores),np.std(scores)))
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

prep_df = df[['hotel', 'lead_time', 'arrival_date_month','stays_in_week_nights','stays_in_weekend_nights', 'adults', 'children', 'babies', 'meal',
             'market_segment', 'distribution_channel', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
             'reserved_room_type', 'assigned_room_type','booking_changes', 'deposit_type']]

target = df['is_canceled']

# One-hot encoding
prep_df = pd.get_dummies(prep_df, columns=['hotel','meal','arrival_date_month','market_segment','distribution_channel',
                                           'reserved_room_type','assigned_room_type','deposit_type'],drop_first=True)


from sklearn.feature_selection import VarianceThreshold

clf_pipeline = Pipeline([
        ("fill_nan", SimpleImputer(missing_values=np.nan, strategy='mean')), 
        ("scale", StandardScaler()), 
        ("select features",VarianceThreshold()),
        ("clf", LogisticRegression(C=0.01,penalty='l1', tol=0.01,solver='saga'))
    ])


scores = estimate_model(clf_pipeline,prep_df,target)
print("Accuracy: %0.4f (+/- %0.4f)" % (np.mean(scores),np.std(scores)))
# 0.629

# 0.672
# 0.678

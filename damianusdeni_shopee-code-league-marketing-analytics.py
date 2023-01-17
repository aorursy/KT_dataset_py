!pip install jcopml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease
df = pd.read_csv("../input/open-shopee-code-league-marketing-analytics/train.csv")
df.head()
plot_missing_value(df, return_df=True)
X = df.drop(columns="open_flag")
y = df.open_flag

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
import pandas as pd
from jcopml.automl import AutoRegressor
model = AutoRegressor(['user_id', 'subject_line_length', 'open_count_last_10_days', 'open_count_last_30_days', 'open_count_last_60_days', 'login_count_last_10_days', 'login_count_last_30_days', 'login_count_last_60_days', 'checkout_count_last_10_days', 'checkout_count_last_30_days', 'checkout_count_last_60_days'], ["country_code", "row_id"])
model.fit(X, y, cv=5, n_trial=10)
df.head()

# not used
# grass_date, last_open
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from jcopml.tuning import grid_search_params as rsp
preprocessor = ColumnTransformer([
    ('numeric', num_pipe(impute='mean', scaling="minmax"), ['subject_line_length', 'open_count_last_10_days', 'open_count_last_30_days', 'open_count_last_60_days', 'login_count_last_10_days', 'login_count_last_30_days', 'login_count_last_60_days', 'checkout_count_last_10_days', 'checkout_count_last_30_days', 'checkout_count_last_60_days']),
    ('categoric', cat_pipe(impute='most_frequent', encoder="onehot"), ["country_code", "grass_date"])
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', KNeighborsClassifier())
])

model = RandomizedSearchCV(pipeline, rsp.knn_params, cv=3, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))
# Save Model
save_model(model, 'model.pkl')
# Load Model
model = load_model("./model/model.pkl")
df_submit = pd.read_csv("../input/open-shopee-code-league-marketing-analytics/test.csv")
df_submit.head()
def submit(model, filename="submission1.csv"):
    df_submit = pd.read_csv("../input/open-shopee-code-league-marketing-analytics/test.csv", index_label="row_id")
    df_submit['open_flag'] = model.predict(df_submit)
    df_submit[['open_flag']].to_csv(filename, index_label="row_id")
submit(model)
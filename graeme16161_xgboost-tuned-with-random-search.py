# pandas is for data manipulation and wrangling

import pandas as pd

# XGBoost is the specific model and we want the classifier 

from xgboost import XGBClassifier

# creates feature importance plot

from xgboost import plot_importance

# Label encoding transforms non-ordinal catigorical variables

from sklearn.preprocessing import LabelEncoder

# splits data into test and training sets

from sklearn.model_selection import train_test_split

# for tuning, located in sklearn.grid_search depending on version

from sklearn.model_selection import RandomizedSearchCV

# for assessing accuracy

from sklearn.metrics import accuracy_score

# For the spliting the data

from sklearn.model_selection import StratifiedKFold



# import data set

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# view the top rows

df.head()
# Make dummy variables for catigorical variables with >2 levels

dummy_columns = ["MultipleLines","InternetService","OnlineSecurity",

                 "OnlineBackup","DeviceProtection","TechSupport",

                 "StreamingTV","StreamingMovies","Contract",

                 "PaymentMethod"]



df_clean = pd.get_dummies(df, columns = dummy_columns)



# Encode catigorical variables with 2 levels

enc = LabelEncoder()

encode_columns = ["Churn","PaperlessBilling","PhoneService",

                  "gender","Partner","Dependents"]



for col in encode_columns:

    df_clean[col] = enc.fit_transform(df[col])

    

# Remove customer ID column

del df_clean["customerID"]





# Make TotalCharges column numeric, empty strings are zeros

df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"],

    errors = 'coerce').fillna(0)
# Split data into x and y

y = df_clean[["Churn"]]

x = df_clean.drop("Churn", axis=1)



# Create test and training sets

x_train, x_test, y_train, y_test = train_test_split(x,

    y, test_size= .2, random_state= 1)
# Build XGBoost model

model = XGBClassifier()

model.fit(x_train, y_train)





# make predictions for test data

y_pred = model.predict(x_test)

predictions = [round(value) for value in y_pred]



# Find Accuracy

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



# Display feature importance

plot_importance(model)
tuned_parameters = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5, 10],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 5, 8]

        }



model = XGBClassifier(learning_rate=0.02, 

                    n_estimators=200,

                    booster = 'gbtree',

                    objective='binary:logistic',

                    silent=True, 

                    nthread=-1)





skf = StratifiedKFold(n_splits=5, shuffle = False, random_state = 22)



random_search_model = RandomizedSearchCV(estimator = model, 

                                   param_distributions=tuned_parameters, 

                                   n_iter=10, 

                                   scoring='accuracy', 

                                   n_jobs=-1, 

                                   cv=skf.split(x_train,y_train), 

                                   verbose=3, 

                                   random_state=22)



random_search_model.fit(x_train, y_train)



y_pred = random_search_model.predict(x_test)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
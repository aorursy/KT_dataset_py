import pandas as pd

import numpy as np



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC





class NumericalTransformer(BaseEstimator, TransformerMixin):

    #Class Constructor

    def __init__( self):

        self.is_fitted = False

        

    #Return self, nothing else to do here

    def fit( self, X, y = None ):

        self.is_fitted = True

        self.feature_names = X.columns

        return self 

    

    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 

    def transform(self, X, y = None):

        #Converting any infinity values in the dataset to Nan

        X = X.replace( [ np.inf, -np.inf ], np.nan )

        #returns a numpy array

        return X.values

    

    def get_feature_names(self):

        if self.is_fitted:

            return self.feature_names
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.dtypes
df.head(3)
# total charges has a few " " values, which hinders conversion

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')



# imputation

df["TotalCharges"] = df["TotalCharges"].fillna(value=df["TotalCharges"].mean())



# map Yes/No to True/False

df["Churn"] = df["Churn"].str.strip().map({"Yes":True, "No":False})



# Senior citizen is a boolean variable formatted as int (0/1)

df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)



# map Yes/No to True/False

for col in ["Partner","Dependents","PhoneService","PaperlessBilling"]:

    df[col] = df[col].str.strip().map({"Yes":True, "No":False})
# Prepare data for model training



feature_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',

                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',

                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

                'StreamingTV', 'StreamingMovies', 'PaperlessBilling',

                'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# Takin contract out because it is misleading. Of course people with month-to-month contract are more free to leave than 2-year contract holders. 

# The contract type is not what is making them leave though. The reason is maybe a bad product or bad support, contract type is just the enabler of churn.

#'Contract', 



target_col = 'Churn'



X = df[feature_cols]

y = df[target_col].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Analysis pipeline definition



# treatment of categorical and numerical features

categorical_features = df.dtypes[(df.dtypes == object) | (df.dtypes == bool)].index.to_list()

categorical_features = list(set(categorical_features).intersection(set(feature_cols)))



categorical_transformer = OneHotEncoder()





numeric_features = df.dtypes[(df.dtypes == int) | (df.dtypes == float)].index.to_list()

numeric_features = list(set(numeric_features).intersection(set(feature_cols)))



numeric_transformer = NumericalTransformer()



# pull feature preprocessing together

preprocessor = ColumnTransformer(

    transformers=[('cat', categorical_transformer, categorical_features),

                  ('num', numeric_transformer, numeric_features)]

)



# model

model = LogisticRegression(solver='liblinear')



# define pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', model)])



# fit pipeline

clf.fit(X_train, y_train)



# score pipeline

print("Train score: {:.2f}".format(clf.score(X_train, y_train)))

print("Test score: {:.2f}".format(clf.score(X_test, y_test)))
names = np.concatenate([clf.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names(categorical_features),

                        clf.named_steps['preprocessor'].named_transformers_['num'].get_feature_names()])

#names = clf.named_steps['preprocessor'].get_feature_names()

values = np.exp(clf['classifier'].coef_).flatten().round(2)



coefficients = dict(zip(names, values))
big_coefficients = {k:v for k,v in coefficients.items() if abs(v-1)>0.15}
sorted(big_coefficients.items(), key=lambda kv: kv[1])
configs = []

train_scores = []

test_scores = []



for model in [LogisticRegression(solver='liblinear'), 

              DecisionTreeClassifier(random_state=0, min_samples_leaf=100), 

              KNeighborsClassifier(),

              GaussianNB(),

              SVC(random_state=0, gamma='scale')

              # any other models to test

              ]:

    clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', model)])

    clf.fit(X_train, y_train)



    # score pipeline

    configs.append(str(model))

    train_scores.append(round(clf.score(X_train, y_train),2))

    test_scores.append(round(clf.score(X_test, y_test),2))

    

results = pd.DataFrame(list(zip(configs, train_scores, test_scores)), columns=["model","train_score","test_score"])
results
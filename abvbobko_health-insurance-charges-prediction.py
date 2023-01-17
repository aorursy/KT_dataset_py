import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
# load data
df_insurance = pd.read_csv("/kaggle/input/health-insurance-cost-prediction/insurance.csv")
df_insurance.head()
fig, ax = plt.subplots(figsize=(10, 7))

ax.set_title("BMI characteristic")
ax.yaxis.grid(True, linestyle='-', color='lightgrey', alpha=0.5)
ax.boxplot(df_insurance.bmi)

plt.show()
fig, ax = plt.subplots()

regions_names = df_insurance.region.value_counts().index
region_population = df_insurance.region.value_counts().values

ax.set_title("Regions population")
ax.pie(region_population, labels=regions_names, 
         autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()
fig, (ax_sex, ax_smoker) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax_sex.set_title("sex")
ax_smoker.set_title("smoker")

sex_names = df_insurance.sex.value_counts().index
sex_cnt = df_insurance.sex.value_counts().values

smoke_answers = df_insurance.smoker.value_counts().index
smokers_cnt = df_insurance.smoker.value_counts().values

ax_sex.bar(sex_names, sex_cnt, width=0.7)
ax_smoker.bar(smoke_answers, smokers_cnt, width=0.7)
plt.show()
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(df_insurance.age, width=4)
ax.set_title("Ages")
plt.show()
fig, ax = plt.subplots(figsize=(8, 5))

has_children = df_insurance.children.value_counts().index
children_cnt = df_insurance.children.value_counts().values
ax.set_title("Info about number of children in family")
ax.plot(has_children, children_cnt, 'o')
plt.show()
fig, ax = plt.subplots(figsize=(8, 5))

age = df_insurance.age
charges = df_insurance.charges
ax.set_xlabel("age")
ax.set_ylabel("charges")
ax.scatter(age, charges)
plt.show()
class SexEncoder:
    # 0 - male
    # 1 - female

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_result = X.copy()
        X_result.loc[X_result.sex == "male", "sex"] = 0
        X_result.loc[X_result.sex == "female", "sex"] = 1
        
        return X_result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
class SmokerEncoder:
    # 0 - no
    # 1 - yes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_result = X.copy()
        X_result.loc[X_result.smoker == "yes", "smoker"] = 1
        X_result.loc[X_result.smoker == "no", "smoker"] = 0
        
        return X_result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
class DirectionEncoder:    
    def encode_regions(self, X):
        # encode region data
        сardinal_directions = ["north", "south", "west", "east"]
        has_direction = {}
        for direction in сardinal_directions:
            has_direction[direction] = np.array(X.region.map(
                    lambda region: 1 if (direction in region) else 0
                ))

        df_directions_info = pd.DataFrame(has_direction)
        return df_directions_info
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_result = X.copy()
        df_directions_info = self.encode_regions(X_result)
        X_result = pd.concat([X_result.reset_index(drop=True), 
                              df_directions_info.reset_index(drop=True)
                             ], 
                             axis=1)        
        # charges_col = X_result.pop("charges")
        # X_result["charges"] = charges_col
        X_result = X_result.drop(["region"], axis=1)        
        return X_result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
class RegionEncoder:     
    def __init__(self):
        self.encoder = LabelEncoder()
        
    def fit(self, X, y=None):        
        self.encoder.fit(X[['region']])
        return self

    def transform(self, X):
        X_result = X.copy()
        X_result['region'] = self.encoder.transform(X_result[['region']])        
        return X_result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
class DataScaler:
    def __init__(self, scaler, cols):
        self.scaler = scaler
        self.cols = cols
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.cols], y)
        return self

    def transform(self, X):
        X_result = X.copy()
        X_result[self.cols] = self.scaler.transform(X_result[self.cols])
        return X_result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
X = df_insurance.drop(["charges"], axis=1)
y = df_insurance.charges
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
steps = [('sex_en', SexEncoder()), 
         ('smoker_en', SmokerEncoder()),          
         ('dir_en', DirectionEncoder()),
         # ('region_en', RegionEncoder()),
         ('scaler', DataScaler(MinMaxScaler(), ['age', 'bmi', 'children'])),
         ('poly', PolynomialFeatures(degree=2))
        ]
preproc_pipe = Pipeline(steps)
steps = [('preprocessor', Pipeline(steps)), 
         ('estimator', Lasso(alpha=24))]
pipe = Pipeline(steps)
X_train_preproc = preproc_pipe.fit_transform(X_train)
parameters = {'alpha': list(range(100))}
gs_pipe = GridSearchCV(Lasso(), parameters)
gs_pipe.fit(X_train_preproc, y_train)
gs_pipe.best_params_
score = cross_val_score(pipe, X, y, cv=5)
score
np.mean(score)
X_train_preproc = preproc_pipe.fit_transform(X_train)
pipe.steps[1][1].fit(X_train_preproc, y_train)
plt.scatter(range(1, len(pipe.steps[1][1].coef_) + 1), pipe.steps[1][1].coef_)
# all coefs
pipe.steps[1][1].coef_
indexes_of_max_coefs = np.argsort(pipe.steps[1][1].coef_)[-7:]
indexes_of_max_coefs
for index in indexes_of_max_coefs:
    print("{0} - {1}".format(
            pipe.steps[0][1].steps[-1][1].powers_[index],
            pipe.steps[1][1].coef_[index]
        )
    )
    
# ['age', 'sex', 'bmi', 'children', 'smoker', "north", "south", "west", "east"]
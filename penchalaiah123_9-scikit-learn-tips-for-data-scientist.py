# Load Python Package
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer

# Load data (loading Titanic dataset)
data  = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
# Make Transformer
preprocessing = make_column_transformer(
    (OneHotEncoder(), ['Pclass','Sex']),
    (SimpleImputer(), ['Age']),
    remainder='passthrough')

# Fit-Transform data with transformer
preprocessing.fit_transform(data)
# Load Python Package
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
# Load data (loading Titanic dataset)
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
# Make Transformer
preprocessing = make_column_transformer(
    (OneHotEncoder(), make_column_selector(dtype_include='object')),
    (SimpleImputer(), make_column_selector(dtype_include='int')),
    remainder='drop'
)

# Fit-Transform data with transformer
preprocessing.fit_transform(data)
# Load Python Package
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# Load data (loading Titanic dataset)
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
# Set X and y
X = data.drop('Survived',axis=1)
y = data[['Survived']]
# Split Train and Test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
# Set variables
ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
imputer = SimpleImputer(add_indicator=True, verbose=1)
scaler = StandardScaler()
clf = DecisionTreeClassifier()
# Make Transformer
preprocessing = make_column_transformer(
(make_pipeline(imputer,scaler),['Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare'])
,(ohe, ['Pclass','Sex','Name'])
,remainder='passthrough')
# Make pipeline
pipe = make_pipeline(preprocessing, clf)
# Fit model
pipe.fit(X_train, y_train.values.ravel())
print("Best score : %f" % pipe.score(X_test, y_test.values.ravel()))
# Load Python Package
from sklearn.experimental import enable_iterative_imputer, enable_hist_gradient_boosting
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
# Load Python Package
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# Load data (loading Titanic dataset)
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
# Set X and y
X = data.drop('Survived',axis=1)
y = data[['Survived']]
# Set variables
ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
imputer = SimpleImputer(add_indicator=True, verbose=1)
clf = DecisionTreeClassifier()
# Make Transformer
preprocessing = make_column_transformer(
(make_pipeline(imputer),['Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare']),
(ohe, ['Pclass','Sex','Name']),remainder='passthrough')
# Make pipeline
pipe = make_pipeline(preprocessing, clf)
# Cross-validation
cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
# Import Python Package
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# Load data (loading Titanic dataset)
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
# Set X and y
X = data.drop('Survived',axis=1)
y = data[['Survived']]
# Set variables
clf = LogisticRegression()
ohe = OneHotEncoder()
scaler = StandardScaler()
imputer = SimpleImputer()
# Make Transformer
preprocessing = make_column_transformer((make_pipeline(imputer,scaler),['Age','Siblings/Spouses Aboard','Parents/Children Aboard','Fare']),(ohe, ['Sex']),remainder='drop')
# Make pipeline
pipe = make_pipeline(preprocessing, clf)
# Set params for Grid Search
params = {}
params['logisticregression__C'] = [0.1,0.2,0.3]
params['logisticregression__max_iter'] = [200,500]
# Run grid search
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X,y.values.ravel())
print(grid.best_score_)
print(grid.best)
# Import Python Package
import pandas as pd
from sklearn.model_selection import train_test_split
# Load data (loading Titanic dataset)
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
# Set X and y
X = data.drop('Survived',axis=1)
y = data[['Survived']]
# Split Train Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

# Import Python Package
import pandas as pd
from sklearn.model_selection import train_test_split
# Load data (loading Titanic dataset)
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
# Set X and y
X = data.drop('Survived',axis=1)
y = data[['Survived']]
# Split Train, Val and Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
# Print dataFrames size
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
# Import Python Package
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
imputer = SimpleImputer()
# Load data (loading Titanic dataset)
data = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')
# Set X and y
X = data.drop('Survived',axis=1)
y = data[['Survived']]
# Write function
def lower_letter(df):
   return df.apply(lambda x : x.str.lower())
# Convert function
get_lower_letter = FunctionTransformer(lower_letter)
# Make Pipeline
preprocess = make_column_transformer((imputer, ['Age']),(get_lower_letter,['Name']),remainder='drop')
preprocess.fit_transform(X)

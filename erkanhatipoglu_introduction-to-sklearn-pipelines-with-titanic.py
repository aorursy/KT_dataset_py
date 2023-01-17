# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin 

from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from category_encoders import BinaryEncoder

import warnings

warnings.filterwarnings("ignore")



from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest
# Functions



def get_titles(df):

    '''Get the set of titles from the Name field'''

    titles=set()

    for name in df:

        if name.find('.'):

            title = name.split('.')[0].split()[-1]

            titles.add(title)

    return titles





def get_family_name(df):

    '''Get the set of family names from Name field'''

    family_names=set()

    for name in df:

        if name.find(','):

            family_name = name.split(',')[0].split()[-1]

            family_names.add(family_name)

    return family_names



def get_cabin_chars(df):

    '''Get the set of chars used in Cabin field'''

    cabin_chars = set()

    for word in df:

        cabin_chars.add(str(word)[0]) 

    return cabin_chars



def save_file (predictions):

    """Save submission file."""

    # Save test predictions to file

    output = pd.DataFrame({'PassengerId': sample_sub_file.PassengerId,

                       'Survived': predictions})

    output.to_csv('submission.csv', index=False)

    print ("Submission file is saved")

    

print("Functions loaded")
# Loading data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

sample_sub_file = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



# Make a copy to avoid changing original data

X = train_data.copy()

y = X.Survived

X_test = test_data.copy()



# Remove target from predictors

X.drop(['Survived'], axis=1, inplace=True)

print('"Survived" column dropped from training data!')



# Remove ticket column. We will not use it.

X.drop("Ticket",axis = 1, inplace = True)

X_test.drop("Ticket",axis = 1, inplace = True)

print('"Ticket" column dropped from both training and test data!')



print("\nShape of training data: {}".format(X.shape))

print("Shape of target: {}".format(y.shape))

print("Shape of test data: {}".format(X_test.shape))

print("Shape of submission data: {}".format(sample_sub_file.shape))



# Split the data for validation

X_train, X_valid, y_train, y_valid = train_test_split(X,y, random_state=2)



print("\nShape of X_train data: {}".format(X_train.shape))

print("Shape of X_valid: {}".format(X_valid.shape))

print("Shape of y_train: {}".format(y_train.shape))

print("Shape of y_valid: {}".format(y_valid.shape))



print("\nFiles Loaded")
X.head()
X.info()
# Make Lists

# Get missing values

missing_values = [col for col in X_train.columns if X_train[col].isnull().sum()]

print("Cols with missing: {}".format(missing_values))



# get numerical and categorical columns

numerical_columns = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

print("Numerical columns: {}".format(numerical_columns))

categorical_columns = [col for col in X.columns if X[col].dtype=="object"]

print("Categorical columns: {}".format(categorical_columns))



# Get titles from the name column

titles = list(get_titles(X["Name"]))

print("\nTitles: {}".format(titles))



# Get chars from the cabin column

chars = list(get_cabin_chars(X["Cabin"]))

print("\nChars in Cabin column: {}".format(chars))
# Get surnames from the name column

surname = list(get_family_name(X["Name"]))

print("Surnames of the passengers: {}".format(surname))
# Custom transformer classes

class NameColumnTransformer(BaseEstimator, TransformerMixin):

    """

    a general class for transforming Name, SibSp and Parch columns of Titanic dataset for using in the machine learning pipeline

    """

    def __init__(self):

        """

        constructor

        """

        # Will be used for fitting data

        self.titles_set = set()

        self.surname_set = set()

        # Titles captured from train data

        self.normal_titles_list = ["Mr", "Mrs", "Mme", "Miss", "Mlle", "Ms", "Master", "Dona"]

        self.titles_dict = {"Mr": ['Mr', 'Major', 'Jonkheer', 'Capt', 'Col', 'Don', 'Sir',

                                   'Rev'],

                            "Mrs": ['Mrs', 'Mme', 'Lady','Countess', 'Dona'],

                            "Miss": ['Miss', 'Mlle', 'Ms'],

                            "Master": ['Master'],

                            "Dr": ['Dr']}



    def fit(self, X, y=None, **kwargs):

        """

        an abstract method that is used to fit the step and to learn by examples

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: self: the class object - an instance of the transformer - Transformer

        """

        '''Fits the titles, family and rank from Names Column'''

        

        # Make a copy to avoid changing original data

        X_temp = X.copy()

        # Create Titles column

        if "Title" in X_temp.columns:

            X_temp.drop("Title", axis=1, inplace=True)

        else:

            pd.DataFrame.insert(X_temp, len(X_temp.columns),"Title","",False)  

            

        # Get the index values

        index_values=X_temp.index.values.astype(int)

        

        # Set state (Add to: {titles_set, surname_set} attributes) of the object

        for i in index_values:

            

            # Get the name for the ith index

            name = X_temp.loc[i,'Name']

            # Get the number of followers for the ith index

            number_of_followers = X_temp.loc [i, 'SibSp'] + X_temp.loc [i, 'Parch']

            

            # Split the title from name

            if name.find('.'):

                title = name.split('.')[0].split()[-1]

                if title in self.titles_dict.keys():

                    X_temp.loc[i, 'Title'] = title

                else:

                    X_temp.loc[i, 'Title'] = np.NaN

                # Add title to titles_set to use in transform method

                self.titles_set.add(title)

            

            # Split the surname from name

            if name.find(','):

                surname = name.split(',')[0].split()[-1]

                # Add surname to surname_set to use in transform method

                if number_of_followers > 0:

                    self.surname_set.add(surname)

                    X_temp.loc[i,"Family"]=surname

                

        # Title Encoding

        

        # Drop missing Title rows (Hi rank columns that are mapped to titles_dict keys)

        # so that no 'Title_' columns will appear in transform 

        X_temp.dropna(axis = "index", subset=['Title'], inplace=True)

        

        # Apply one-hot encoding to the Title column.

        self.OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

        # Get column names to use in transform.

        self.OH_encoder = self.OH_encoder.fit(X_temp[['Title']])

        self.title_columns = self.OH_encoder.get_feature_names(['Title'])



        # Family Encoding

        

        # Drop missing Family rows

        # so that no 'Family_' columns will appear in transform 

        X_temp.dropna(axis = "index", subset=['Family'], inplace=True)

        

        # Apply binary encoding to the Family column.

        self.binary_encoder = BinaryEncoder(cols =['Family'])

        self.binary_encoder = self.binary_encoder.fit(X_temp[['Family']])

        

        return self



    def transform(self, X, y=None, **kwargs):

        """

        an abstract method that is used to transform according to what happend in the fit method

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: X: the transformed data - Dataframe

        """

        '''Transforms the titles and family from Names Column'''

        

        # Make a copy to avoid changing original data

        X_temp = X.copy()

        

        # Create Titles column    

        pd.DataFrame.insert(X_temp, len(X_temp.columns),"Title","",False)    

        # Create Family column

        pd.DataFrame.insert(X_temp, len(X_temp.columns),"Family","",False)          

        # Create Rank column

        pd.DataFrame.insert(X_temp, len(X_temp.columns),"Rank","",False)

        # Create Followers column

        pd.DataFrame.insert(X_temp, len(X_temp.columns),"Followers","",False)

        

        # Get the index values

        index_values=X_temp.index.values.astype(int)

        

        for i in index_values:

            # Get the name for the ith index

            name = X_temp.loc[i,'Name']

            # Get the number of followers for the ith index

            number_of_followers = X_temp.loc [i, 'SibSp'] + X_temp.loc [i, 'Parch']

            X_temp.loc[i, 'Followers'] = number_of_followers

            

            # Split the title from name

            if name.find('.'):

                title = name.split('.')[0].split()[-1]

                if title in self.titles_set:

                    for key in self.titles_dict:

                        # Insert title

                        if title in self.titles_dict[key]:

                            X_temp.loc[i, 'Title'] = key 

                        

                        # Insert rank

                        if title in self.normal_titles_list:

                            X_temp.loc[i, 'Rank'] = "Normal"

                        else:

                            X_temp.loc[i, 'Rank'] = "High"

                else:

                    X_temp.loc[i, 'Title'] = "Other"

                    X_temp.loc[i, 'Rank'] = "Normal"

                    

                    

            # Split the surname from name

            if name.find(','):

                surname = name.split(',')[0].split()[-1]

                if surname in self.surname_set and number_of_followers > 0:

                    X_temp.loc[i, 'Family'] = surname                 

                else:

                    X_temp.loc[i, 'Family'] = "NA"                    

        

        # Encoding Title

        encoded = self.OH_encoder.transform(X_temp[['Title']])

        # convert arrays to a dataframe 

        encoded = pd.DataFrame(encoded) 

        # One-hot encoding removed index; put it back

        encoded.index = X_temp.index

        # Insert column names

        encoded.columns = self.title_columns

        encoded = encoded.astype('int64')

        # concating dataframes  

        X_temp = pd.concat([X_temp, encoded], axis = 1)

        

        # Encoding Family

        bin_encoded = self.binary_encoder.transform(X_temp[['Family']])

        # convert arrays to a dataframe 

        bin_encoded = pd.DataFrame(bin_encoded) 

        # One-hot encoding removed index; put it back

        bin_encoded.index = X_temp.index

        bin_encoded = bin_encoded.astype('int64')

        # concating dataframes  

        X_temp = pd.concat([X_temp, bin_encoded], axis = 1)

        # We do not need Family any more

        X_temp.drop("Family", axis = 1, inplace=True) 

        

        # Encoding Rank

        X_temp['Rank'] = X_temp['Rank'].apply(lambda x: 1 if x =='Normal' else (0 if x =='High' else None))

        # We do not need Name any more

        X_temp.drop("Name", axis = 1, inplace=True)



        return X_temp



    def fit_transform(self, X, y=None, **kwargs):

        """

        perform fit and transform over the data

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: X: the transformed data - Dataframe

        """

        self = self.fit(X, y)

        return self.transform(X, y)

print("NameColumnTransformer loaded")
# Custom transformer classes

class AgeColumnTransformer(BaseEstimator, TransformerMixin):

    """

    a general class for transforming age column of Titanic dataset for using in the machine learning pipeline

    """

    def __init__(self):

        """

        constructor

        """

        # Will be used for fitting data

        self.titles_set = set()

        self.titles_dict = {}





    def fit(self, X, y=None, **kwargs):

        """

        an abstract method that is used to fit the step and to learn by examples

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: self: the class object - an instance of the transformer - Transformer

        """

        '''Fits the titles, family and rank from Names Column'''

 

        # Make a copy to avoid changing original data

        X_temp = X.copy()

    

        # Get the index values

        index_values = X_temp.index.values.astype(int)

        

        # Get all the titles from dataset

        for i in index_values:

            title = X_temp.loc [i, 'Title']

            self.titles_set.add(title)

        

        # Calculate mean for all titles

        for title in self.titles_set:

            mean = self.calculate_mean_age(title, X_temp)

            self.titles_dict[title] = mean

            #print("Avarage age for title '{}' is {}".format(title, mean))

       

        return self



    def transform(self, X, y=None, **kwargs):

        """

        an abstract method that is used to transform according to what happend in the fit method

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: X: the transformed data - Dataframe

        """

        '''Transforms the titles and family from Names Column'''

            

        # Make a copy to avoid changing original data

        X_temp = X.copy()        

        

        # Get the index values

        index_values = X_temp.index.values.astype(int)

        

        # If a passangers age is Nan replace it with the avarage value

        # of that title class. e.g. if that passanger is master use the

        # mean value calculated for the masters.

        for i in index_values:

            age = X_temp.at[i, 'Age'].astype(float)

            if np.isnan(age):

                title = X_temp.loc [i, 'Title']

                X_temp.loc[i,'Age'] =  round(self.titles_dict.get(title),2)



        # We do not need Title any more

        X_temp.drop("Title", axis = 1, inplace=True)

        

        return X_temp



    def fit_transform(self, X, y=None, **kwargs):

        """

        perform fit and transform over the data

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: X: the transformed data - Dataframe

        """

        self = self.fit(X, y)

        

        return self.transform(X, y)

                   

    def calculate_mean_age(self, title, X):

        

        # Make a copy to avoid changing original data

        X_temp = X.copy()  

        

        title_X = X_temp[[title in x for x in X_temp['Title']]][X_temp["Age"].notnull()]

        return title_X["Age"].mean()



    print("AgeColumnTransformer loaded")
# Custom transformer classes

class CabinColumnTransformer(BaseEstimator, TransformerMixin):

    """

    a general class for transforming cabin column of Titanic dataset for using in the machine 

    learning pipeline

    """

    def __init__(self):

        """

        constructor

        """

        # Will be used for fitting data

        self.cabin_set = set()



    def fit(self, X, y=None, **kwargs):

        """

        an abstract method that is used to fit the step and to learn by examples

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: self: the class object - an instance of the transformer - Transformer

        """

        '''Fits the titles, family and rank from Names Column'''

         

        # Make a copy to avoid changing original data

        X_temp = X.copy()

        

        # Imputation on X_temp_imputed

        imputer = SimpleImputer(strategy='constant', fill_value="NaN")

        X_temp_imputed = pd.DataFrame(imputer.fit_transform(X_temp[['Cabin']]))

        # Imputation removed column names; put them back

        X_temp_imputed.columns = X_temp[['Cabin']].columns

        X_temp_imputed.index = X_temp[['Cabin']].index

    

        # Get the index values

        index_values = X_temp.index.values.astype(int)

        

        # For each cabin

        for i in index_values:

            cabin = X_temp_imputed.loc[i, 'Cabin']

            X_temp_imputed.loc[i, 'Cabin'] = cabin[0]

            self.cabin_set.add(cabin[0])

        

        # Cabin Encoding

        

        # Apply one-hot encoding to the Cabin column.

        self.OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

        # Get column names to use in transform.

        self.OH_encoder = self.OH_encoder.fit(X_temp_imputed[['Cabin']])

        self.cabin_columns = self.OH_encoder.get_feature_names(['Cabin'])

      

        return self



    def transform(self, X, y=None, **kwargs):

        """

        an abstract method that is used to transform according to what happend in the fit method

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: X: the transformed data - Dataframe

        """

        '''Transforms the titles and family from Names Column'''



        # Make a copy to avoid changing original data

        X_temp = X.copy()        

        

        # Get the index values

        index_values = X_temp.index.values.astype(int)

        

        # Imputation on X_imputed

        imputer = SimpleImputer(strategy='constant', fill_value="NaN")

        X_imputed = pd.DataFrame(imputer.fit_transform(X_temp[['Cabin']]))

        # Imputation removed column names; put them back

        X_imputed.columns = X_temp[['Cabin']].columns

        X_imputed.index = X_temp[['Cabin']].index

        

        for i in index_values:

            cabin = X_imputed.loc[i, 'Cabin']

            if cabin[0] in self.cabin_set:

                X_imputed.loc [i, 'Cabin'] = cabin[0]

            else:

                X_imputed.loc [i, 'Cabin'] = "N"

        X_temp.drop("Cabin",axis = 1, inplace = True)



        # concating dataframes  

        X_temp = pd.concat([X_temp, X_imputed], axis = 1)

               

        # Encoding Cabin

        encoded = self.OH_encoder.transform(X_imputed[['Cabin']])

        # convert arrays to a dataframe 

        encoded = pd.DataFrame(encoded) 

        # One-hot encoding removed index; put it back

        encoded.index = X_imputed.index

        # Insert column names

        encoded.columns = self.cabin_columns

        encoded = encoded.astype('int64')

        # concating dataframes  

        X_temp = pd.concat([X_temp, encoded], axis = 1)



        X_temp.drop("Cabin",axis = 1, inplace = True)

        

        return X_temp



    def fit_transform(self, X, y=None, **kwargs):

        """

        perform fit and transform over the data

        :param X: features - Dataframe

        :param y: target vector - Series

        :param kwargs: free parameters - dictionary

        :return: X: the transformed data - Dataframe

        """

        self = self.fit(X, y)

        return self.transform(X, y)



    print("CabinColumnTransformer loaded")
X.head()
# Apply NameColumnTransformer

name = NameColumnTransformer()

X_name=name.fit_transform(X) 



# Apply AgeColumnTransformer

age=AgeColumnTransformer()

X_age=age.fit_transform(X_name) 



# Apply CabinColumnTransformer

cabin=CabinColumnTransformer()

X_cabin=cabin.fit_transform(X_age) 



X_cabin.head()
# Define the custom transformers for the pipeline

name_transformer = NameColumnTransformer()

age_transformer = AgeColumnTransformer()

cabin_transformer = CabinColumnTransformer()
# Define the columns that will be handled by the column transformer

numerical_cols = ['Pclass', 'Fare']

categorical_cols = ['Sex', 'Embarked']
# Define transformers for numerical columns using a pipeline

numerical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean')),

    ('scaler', StandardScaler())])





# Define transformers for categorical columns using a pipeline

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))

])



# Define column transformer for numerical and categorical data

column_transformer = ColumnTransformer(

    transformers=[

        ('numerical', numerical_transformer, numerical_cols),

        ('categorical', categorical_transformer, categorical_cols)

    ], remainder='passthrough')
# This dataset is way too high-dimensional. Better do PCA:

pca = PCA(n_components=2)



# Maybe some original features where good, too?

selection = SelectKBest(k=12)



# Define union

feature_union = FeatureUnion([('pca', pca), ('select', selection)])
# Define the model

my_model = XGBClassifier(

 learning_rate =0.01,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 seed=42)
# Define preprocessor

preprocessor = Pipeline(steps=[('name', name_transformer),

                              ('age', age_transformer),

                              ('cabin', cabin_transformer),

                              ('column', column_transformer),

                              ('union', feature_union)])



# Make a copy to avoid changing original data 

X_valid_eval=X_valid.copy()



# Preprocessing of validation data

X_valid_eval = preprocessor.fit(X_train, y_train).transform (X_valid_eval)
# Display the number of remaining columns after transformation 

print("We have", X_valid_eval.shape[1], "features left")
# Define XGBoostClassifier fitting parameters for the pipeline

fit_params = {"model__early_stopping_rounds": 50,

              "model__eval_set": [(X_valid_eval, y_valid)],

              "model__verbose": True,

              "model__eval_metric" : "error"}
# Create and Evaluate the Pipeline

# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', my_model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train, **fit_params)



# Get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = accuracy_score(y_valid, preds)



print("Score: {}".format(score))
cv_model = XGBClassifier(learning_rate =0.01,

                         n_estimators=40,

                         max_depth=5,

                         min_child_weight=1,

                         gamma=0,

                         subsample=0.8,

                         colsample_bytree=0.8,

                         seed=42)



cv_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', cv_model)

                             ])



    

scores = cross_val_score(cv_pipeline, X_train, y_train,

                              cv=5,

                              scoring='accuracy')



print("Accuracy of the folds:\n", scores)

print("\nmean:\n", scores.mean())

print("std:\n", scores.std())
# Preprocessing of training data, fit model 

cv_pipeline.fit(X_train, y_train)



# Get predictions

preds = cv_pipeline.predict(X_valid)



# Evaluate the model

score = accuracy_score(y_valid, preds)



print("Score: {}".format(score))
# Preprocessing of training data, fit model 

cv_pipeline.fit(X, y)



# Get predictions

preds = cv_pipeline.predict(X_test)
# Save submission file

save_file(preds)
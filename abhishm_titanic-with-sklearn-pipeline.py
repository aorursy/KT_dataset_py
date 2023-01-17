import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import cross_val_score, train_test_split

from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

from sklearn.svm import SVC, LinearSVC
class OneHotEncoding(BaseEstimator, TransformerMixin):

    """Takes in dataframe and give one hot encoding for categorical features """



    def __init__(self, column_names=[]):

        self.column_names = column_names



    def transform(self, df, y=None):

        """transform a categorical feature into one-hot-encoding"""

        return pd.get_dummies(df, columns=self.column_names)



    def fit(self, df, y=None):

        """Pass"""

        return self
class DropColumns(BaseEstimator, TransformerMixin):

    """Drop the columns in a dataframe """



    def __init__(self, column_names=[]):

        self.column_names = column_names



    def transform(self, df, y=None):

        """drop the columns present in self.columns"""

        return df.drop(self.column_names, axis=1)



    def fit(self, df, y=None):

        """Pass"""

        return self
class ColumnExtractor(BaseEstimator, TransformerMixin):

    """Takes in dataframe, extracts a columns as feture """



    def __init__(self, column_names=[]):

        self.column_names = column_names



    def transform(self, df, y=None):

        """Return the columns"""

        return df.loc[:, self.column_names]



    def fit(self, df, y=None):

        """Pass"""

        return self
class SexBinarizer(BaseEstimator, TransformerMixin):

    """Takes in dataframe, extracts a columns as feture """



    def __init__(self, column_names=[]):

        pass



    def transform(self, df, y=None):

        """female maps to 0 and male maps to 1"""

        df.loc[:, "Sex"] = df.loc[:, "Sex"].map({"male": 0, "female": 1})

        return df



    def fit(self, df, y=None):

        """pass"""

        return self
class FeatureNormalizer(BaseEstimator, TransformerMixin):

    """Takes in dataframe, extracts a columns as feture """



    def __init__(self, column_names=[]):

        self.column_names = column_names

        self.min_max_scalar = MinMaxScaler()



    def transform(self, df, y=None):

        """Min Max Scalar"""

        df.loc[:, self.column_names] = self.min_max_scalar.transform(df[self.column_names].as_matrix())

        return df



    def fit(self, df, y=None):

        """FItting Min Max Scalar"""

        self.min_max_scalar.fit(df[self.column_names].as_matrix())

        return self
class FillNa(BaseEstimator, TransformerMixin):

    """Takes in dataframe, fill NaN values in a given columns """



    def __init__(self, method="mean"):

        self.method = method



    def transform(self, df, y=None):

        """The workhorse of this feature extractor"""

        if self.method == "zeros":

            df.fillna(0)

        elif self.method == "mean":

            df.fillna(df.mean(), inplace=True)

        else:

            raise ValueError("Method should be 'mean' or 'zeros'")

        return df



    def fit(self, df, y=None):

        """Returns `self` unless something different happens in train and test"""

        return self
class AddTwoCategoricalVariables(BaseEstimator, TransformerMixin):

    def __init__(self, column_1, column_2):

        self.column_1 = column_1

        self.column_2 = column_2

    

    def transform(self, df):

        df[self.column_1 + "_" + self.column_2] = (df[self.column_1].astype(float) + 

                                                (len(df[self.column_1].unique()) * 

                                                (df[self.column_2].astype(float)))).astype("category")

        return df

    

    def fit(self, df, y=None):

        return self
class Numerical2Categorical(BaseEstimator, TransformerMixin):

    def __init__(self, column, ranges, labels):

        self.column = column

        self.ranges = ranges

        self.labels = labels

        

    def transform(self, df):

        df.loc[:, self.column + "_" + "cat"] = (pd

                                                .cut(df.loc[:, self.column], 

                                                     self.ranges, labels=self.labels))

        return df

    

    def fit(self, df, y=None):

        return self
def submission(pred):

    submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": pred

    })

    submission.to_csv('titanic.csv', index=False)
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



y_train = df_train.Survived
#Null Accuracy

print("Null Accuracy: {0:0.4f}".format((y_train.value_counts() / len(y_train)).head(1)[0]))
dropped_row_subset = ["Embarked"]

df_train_copy = df_train.dropna(subset=dropped_row_subset)

y_train = df_train_copy.Survived
feature_columns = ["Fare", "Pclass", "Sex", "Age", "SibSp", "Parch"]

normalize_features = ["Fare", "SibSp", "Parch"]
age_range = [0, 15, 35, 50, 80]

age_label = [0, 1, 2, 3]
def cross_val_accuracy():

    pipeline = Pipeline([

            ("column_extractor", ColumnExtractor(feature_columns)),

            ("fill_na", FillNa("mean")),

            ("sex_binarizer", SexBinarizer()),

            ("num2cat", Numerical2Categorical("Age", age_range, age_label)),

            ("add_age_sex", AddTwoCategoricalVariables("Age_cat", "Sex")),

            ("add_sex_class", AddTwoCategoricalVariables("Sex", "Pclass")),

            ("add_age_sex_class", AddTwoCategoricalVariables("Age_cat_Sex", "Pclass")),

            ("one_hot_encoding", OneHotEncoding(["Age_cat_Sex", "Sex_Pclass"])),

            ("drop_columns", DropColumns(["Age_cat"])),

            ("feature_normalizer", FeatureNormalizer(normalize_features)),

            ("clf", LogisticRegression())])

    scores = cross_val_score(pipeline, df_train_copy, y_train, cv=5, scoring="accuracy")

    print("cross-validation score: {0:0.4f}".format(scores.mean()))

    return scores.mean(), pipeline
score, clf = cross_val_accuracy()
clf.fit(df_train_copy, y_train)

submission(clf.predict(df_test))
import numpy as np

import pandas as pd



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import cross_val_score, RepeatedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier



# Kaggle's current docker version throws spurious "FutureWarning: Passing (type, 1) or '1type'..."

# warnings. This disables those messages, while still letting our kernel run properly.

import warnings  

warnings.filterwarnings('ignore')
Xy_train = pd.read_csv("../input/titanic/train.csv", index_col="PassengerId")



# A completion-friendly collection of column names

class CC:

    def __init__(self, dataframe):

        for col in dataframe.columns: setattr(self, col, col)

cc = CC(Xy_train)

 

X_train = Xy_train.drop(columns=[cc.Survived])

y_train = Xy_train[cc.Survived]
class GenericTransformer(BaseEstimator, TransformerMixin):

    """Defines a transformer based on simple lambda functions for fitting and transforming."""

    def __init__(self, transformer, fitter=None):

        self.transformer = transformer

        self.fitter = fitter

    def fit(self, X, y=None):

        self.fit_val = None if self.fitter is None else self.fitter(X)

        return self

    def transform(self, X):

        return self.transformer(X) if self.fit_val is None else self.transformer(X, self.fit_val)
# Creates an imputer which fills missing categorical values with "missing" and missing numerical 

# values with the median value. By using a "fit" step, we ensure that the median values reflect 

# just the train set and not the test set. This will make a small difference during 

# cross-validation.

default_imputer = GenericTransformer(

    fitter = lambda X: {  # create default_values

        col: "missing" if X[col].dtype == "object" else X[col].median() for col in X.columns},

    transformer = lambda X, default_values: (

        X.assign(**{col: X[col].fillna(default_values[col]) for col in X.columns})),

)



assert not default_imputer.fit_transform(X_train).isna().any().any()
class DummiesTransformer(BaseEstimator, TransformerMixin):

    """Transformer which applies the DataFrame 'get_dummies' method in a cv-safe manner."""

    def __init__(self, drop_first=False):

        self.drop_first = drop_first

    def fit_transform(self, x, y=None):

        result = pd.get_dummies(x, drop_first=self.drop_first)

        self.output_cols = result.columns

        return result

    def fit(self, x, y = None):

        self.fit_transform(x, y)

        return self

    def transform(self, x):

        x_dummies = pd.get_dummies(x, drop_first=self.drop_first)

        new_cols = set(x_dummies.columns)

        # Return a new DataSet with exactly the columns found in "fit", zero-filling any that 

        # are missing and dropping extras

        return pd.DataFrame({col: x_dummies[col] if col in new_cols else 0 

                             for col in self.output_cols})

add_dummies = DummiesTransformer()



assert (X_train.dtypes == "object").any()

assert not (add_dummies.fit_transform(X_train).dtypes == "object").any()
# Create a transformer which gets rid of all X_train's original columns, leaving just the new

# ones. This should be invoked before add_dummies.

drop_original = GenericTransformer(lambda X: X.drop(columns=X_train.columns))



assert len(drop_original.fit_transform(X_train).columns) == 0
all_classifiers = [

    RidgeClassifier(random_state=0),

    XGBClassifier(random_state=0),

    DecisionTreeClassifier(random_state=0),

    RandomForestClassifier(n_estimators=10, random_state=0),

    KNeighborsClassifier(),

]

clsf_accumulator = pd.DataFrame()

def test_classifiers(label, transformers):

    for clsf in all_classifiers:

        pipeline = make_pipeline(default_imputer, *transformers, drop_original, add_dummies)

        accuracy = cross_val_score(make_pipeline(pipeline, clsf), X_train, y_train, cv=5).mean()

        clsf_accumulator.loc[label, clsf.__class__.__name__] = accuracy



add_const = GenericTransformer(lambda X: X.assign(CONST=1))

add_random = GenericTransformer(lambda X: X.assign(RAND=np.random.random(X.shape[0])))

add_sex = GenericTransformer(lambda X: X.assign(FEMALE=(X[cc.Sex] == "female").astype("int8")))

add_id = GenericTransformer(lambda X: X.assign(ID=X.index))



test_classifiers("null", [add_const])

test_classifiers("null+rand", [add_const, add_random])

test_classifiers("null+rand+id", [add_const, add_id, add_random])

test_classifiers("sex", [add_sex])

test_classifiers("sex+rand", [add_sex, add_random])

test_classifiers("sex+rand+id", [add_sex, add_random, add_id])



display(clsf_accumulator)
feat_accumulator = pd.DataFrame()

model = RidgeClassifier(random_state=0)



def test_features(label, isolated, combination, prev_label=None, extra_imputers=[]):

    isolated_pipeline = make_pipeline(*extra_imputers, default_imputer, *isolated, 

                                      drop_original, add_dummies, model)

    iso_accuracy = cross_val_score(isolated_pipeline, X_train, y_train, cv=5).mean()

    feat_accumulator.loc[label, "isolated"] = iso_accuracy

    

    combo_pipeline = make_pipeline(*extra_imputers, default_imputer, *combination,

                                   drop_original, add_dummies, model)

    combo_accuracy = cross_val_score(combo_pipeline, X_train, y_train, cv=5).mean()

    feat_accumulator.loc[label, "combination"] = combo_accuracy

    if prev_label is not None:

        old_combo_accuracy = feat_accumulator.loc[prev_label, "combination"]

        feat_accumulator.loc[label, "improvement"] = combo_accuracy - old_combo_accuracy

    display(feat_accumulator)



test_features("null", [add_const], [add_const])    
add_age = GenericTransformer(

    lambda X: X.assign(AGE_BINNED = pd.cut(

        X[cc.Age], [0,7,14,35,60,1000],

        labels=["young child","child", "young adult", "adult", "old"])))



test_features("age", [add_age], [add_age], "null")
add_class = GenericTransformer(lambda X: X.assign(CLASS_BINNED=X[cc.Pclass].astype(str)))



test_features("class", [add_class], [add_age, add_class], "age")
class TicketSurvivalTransformer(BaseEstimator, TransformerMixin):

    """Adds the average survival rate of people on the same ticket.

    

    This is a tricky (and questionable) computation which depends upon having access to the

    target feature, while still accepting that it won't be available when transforming the

    test set. Thus, for each row we pretend that we don't know the survival of the individual,

    and instead give it a fractional value equal to the overall mean survival rate."""

    def __init__(self, xy):

        self.xy = xy

    def fit(self, X, y=None):

        X_with_survival = X.assign(Survived = self.xy.reindex(X.index)[cc.Survived])

        self.mean_survival = X_with_survival[cc.Survived].mean()

        self.group_stats = (X_with_survival.groupby(cc.Ticket)[cc.Survived].agg(["count", "sum"]))

        self.fit_X = X.copy()

        return self

    def transform(self, X):

        X_with_survival = X.assign(Survived = self.xy.reindex(X.index)[cc.Survived])

        group_stats_by_passenger = self.group_stats.reindex(X[cc.Ticket].unique(), fill_value=0)

        X_counts = group_stats_by_passenger.loc[X[cc.Ticket]].set_index(X.index)

        is_overlap = np.array([x in self.fit_X.index for x in X.index])

        other_counts = (X_counts["count"]-1).where(is_overlap, X_counts["count"])

        other_survivor_count = ((X_counts["sum"] - self.xy[cc.Survived].reindex(X.index))

                                .where(is_overlap, X_counts["sum"]))

        survival_fraction = (other_survivor_count + self.mean_survival) / (other_counts + 1)

        return X.assign(SurvivorFraction=survival_fraction)



# This assertion is a weak test, but make sure that average group survival is at least close

# to the overall survival rate.

assert np.isclose((TicketSurvivalTransformer(Xy_train).fit(X_train.loc[:400])

                   .transform(X_train.loc[400:])["SurvivorFraction"].mean()),

                  Xy_train.loc[400:][cc.Survived].mean(), rtol=0.1)
add_tkt = TicketSurvivalTransformer(Xy_train)



test_features("tkt", [add_tkt], [add_age, add_class, add_tkt], "class")
# We use the add_sex transformer defined in "Finding a stable classifier"



test_features("sex", [add_sex], [add_age, add_class, add_tkt, add_sex], "tkt")
class TitleAgeImputer(BaseEstimator, TransformerMixin):

    def __init__(self): pass

    def fit(self, X, y=None):

        self.overall_median = X[cc.Age].median()

        title = X[cc.Name].str.replace(r"[^,]+, *([^.]+)\..*", r"\1")

        self.title_median = X.assign(Title=title).groupby("Title")[cc.Age].agg("median")

        return self

    def transform(self, X):

        title = X[cc.Name].str.replace(r"[^,]+, *([^.]+)\..*", r"\1")

        imputed_ages = self.title_median.reindex(index=title, fill_value=self.overall_median)

        imputed_ages.index = X.index

        result = X.assign(Age=X[cc.Age].fillna(imputed_ages))

        return result



# The average age after imputation should be close to the average supplied age.

assert np.isclose(TitleAgeImputer().fit_transform(X_train)[cc.Age].mean(),

                  X_train[cc.Age].mean(), rtol=2)
impute_age = TitleAgeImputer()



test_features("age_from_title", [add_age], [add_age, add_class, add_tkt, add_sex], "sex",

              extra_imputers=[impute_age])
def feature_weights(transformer, X, y):

    transformed = transformer.fit_transform(X)

    model = RidgeClassifier(random_state=0).fit(transformed, y)

    coef = model.coef_[0]    # RidgeClassifier wraps the coefficients in a 1xN array

    coefficients = pd.DataFrame({"feature": transformed.columns,

                                 "coefficient": coef, 

                                 "abs": np.abs(coef)}) 

    sorted = coefficients.sort_values("abs", ascending=False)

    ranked = sorted.assign(Rank=range(1, len(transformed.columns)+1))

    ranked = ranked.append(pd.Series([" ", "**intercept**", model.intercept_[0], 0], name=0,

                                    index=["Rank", "feature", "coefficient", "abs"]))

    return ranked.set_index("Rank", drop=True).drop(columns=["abs"])



final_pipeline = make_pipeline(impute_age, default_imputer, add_age, add_class, add_tkt, add_sex, 

                               drop_original, add_dummies)

display(feature_weights(final_pipeline, X_train, y_train))
final_model = RidgeClassifier(random_state=0)

final_X_train = final_pipeline.fit_transform(X_train)



cvscore = cross_val_score(final_model, final_X_train, y_train, cv=4)

print(f"cv = {np.mean(cvscore)}: \n  {list(cvscore)}")

final_model.fit(final_X_train, y_train)



X_test = pd.read_csv("../input/titanic/test.csv", index_col="PassengerId")

final_X_test = final_pipeline.transform(X_test)

preds_test = final_model.predict(final_X_test)

output = pd.DataFrame({'PassengerID': X_test.index,

                       'Survived': preds_test})

file = 'submission.csv'

output.to_csv(file, index=False)

print(f"Wrote predictions to '{file}'")
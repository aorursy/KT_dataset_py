import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import xgboost as xgb

from IPython import display
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.python.data import Dataset

titanic_train_dataframe = pd.read_csv("../input/train.csv")
titanic_test_dataframe = pd.read_csv("../input/test.csv")
print("train_dataframe:")
titanic_train_dataframe.info()

print("\ntest_dataframe:")
titanic_test_dataframe.info()
print("missing values: (train_dataframe)")
print(titanic_train_dataframe.isnull().sum().sort_values(ascending=False))

print("\nmissing values: (test_dataframe)")
print(titanic_test_dataframe.isnull().sum().sort_values(ascending=False))
row_indices = list(titanic_train_dataframe.loc[titanic_train_dataframe["Embarked"].isnull()].index)
display.display(titanic_train_dataframe.loc[row_indices])

train_embarked_series = titanic_train_dataframe.loc[titanic_train_dataframe["Embarked"].notnull(), "Embarked"]
test_embarked_series = titanic_test_dataframe["Embarked"]
embarked_mode = pd.concat([train_embarked_series, test_embarked_series]).mode()[0]
print("mode of embarked column: {0}".format(embarked_mode))

titanic_train_dataframe.loc[row_indices, "Embarked"] = embarked_mode
display.display(titanic_train_dataframe.loc[row_indices])
row_indices = list(titanic_test_dataframe.loc[titanic_test_dataframe["Fare"].isnull()].index)
display.display(titanic_test_dataframe.loc[row_indices])

train_fare_series = titanic_train_dataframe.loc[titanic_train_dataframe["Pclass"] == 3, "Fare"]
test_fare_series = titanic_test_dataframe.loc[titanic_test_dataframe["Pclass"] == 3, "Fare"]
fare_median = pd.concat([train_fare_series, test_fare_series]).median()
print("median of fare column: {0:.2f}".format(fare_median))

titanic_test_dataframe.loc[row_indices, "Fare"] = fare_median
display.display(titanic_test_dataframe.loc[row_indices])
for dataframe in [titanic_train_dataframe, titanic_test_dataframe]:
    display.display(dataframe.loc[titanic_train_dataframe["Age"].isnull(), ["Sex", "Pclass", "PassengerId"]].\
            groupby(["Sex", "Pclass"], as_index=True).\
            count().\
            rename(columns={"PassengerId": "PassengerCount"}))
def get_median_age():
    dataframe = pd.concat([
        titanic_train_dataframe.loc[titanic_train_dataframe["Age"].notnull(), ["Sex", "Pclass", "Age"]],
        titanic_test_dataframe.loc[titanic_test_dataframe["Age"].notnull(), ["Sex", "Pclass", "Age"]]
    ])
    
    median_ages = {"male": {}, "female": {}}
    for sex in ["male", "female"]:
        for p_class in [1, 2, 3]:
            median_ages[sex][p_class] = dataframe.loc[(dataframe["Sex"] == sex) & (dataframe["Pclass"] == p_class), "Age"].median()
    
    return median_ages

median_ages = get_median_age()
print("Median Ages:")
for sex in ["male", "female"]:
    for p_class in [1, 2, 3]:
        print("    median_age({0}, {1}) = {2}".format(sex, p_class, median_ages[sex][p_class]))
for dataframe in [titanic_train_dataframe, titanic_test_dataframe]:
    for sex in ["male", "female"]:
        for p_class in [1, 2, 3]:
            dataframe.loc[(dataframe["Sex"] == sex) & (dataframe["Pclass"] == p_class) & (dataframe["Age"].isnull()), "Age"] = median_ages[sex][p_class]
drop_cols = ["Cabin", "Ticket"]
titanic_train_dataframe = titanic_train_dataframe.drop(drop_cols, axis=1)
titanic_test_dataframe = titanic_test_dataframe.drop(drop_cols, axis=1)

display.display(titanic_train_dataframe.head())
display.display(titanic_test_dataframe.head())
titanic_train_dataframe.loc[titanic_train_dataframe["Pclass"] == 3, "Fare"].describe()
grid = sns.FacetGrid(titanic_train_dataframe, row="Sex", col="Survived")
grid.map(plt.hist, "Age", bins=20)
grid.add_legend();
def plot_scatter(dataframe, x_col, y_col, age_threshold=None, fare_threshold=None):
    plt.figure(figsize=(21,6))
    for index, sex in enumerate(["male", "female", None], 1):
        plt.subplot(1, 3, index)
        if sex is not None:
            sex_dataframe = dataframe.loc[dataframe["Sex"] == sex]
        else:
            sex_dataframe = dataframe
        survived_dataframe = sex_dataframe.loc[sex_dataframe["Survived"] == 1, [x_col, y_col]]
        plt.scatter(survived_dataframe[x_col], survived_dataframe[y_col], c="blue", label="survived")
        dead_dataframe = sex_dataframe.loc[sex_dataframe["Survived"] == 0, [x_col, y_col]]
        plt.scatter(dead_dataframe[x_col], dead_dataframe[y_col], c="green", label="dead")
        plt.title(sex if sex is not None else "male + female")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        if fare_threshold is not None:
            plt.axhline(y=fare_threshold, color="red")
        if age_threshold is not None:
            plt.axvline(x=age_threshold, color="red")
        plt.legend(loc="upper left")
plot_scatter(titanic_train_dataframe, "Age", "Fare")
age_threshold = 13.0
fare_threshold = 35.0

plot_scatter(titanic_train_dataframe, "Age", "Fare", age_threshold=age_threshold, fare_threshold=fare_threshold)
def cross_features(dataframe, age_threshold, fare_threshold):
    crossed_feature = "AgeFare"
    dataframe["AgeFare"] = 0
    dataframe.loc[dataframe["Age"] <= age_threshold, crossed_feature] = 1
    dataframe.loc[(dataframe["Sex"] == "male") & (dataframe["Age"] > age_threshold) & (dataframe["Fare"] <= fare_threshold), crossed_feature] = 2
    dataframe.loc[(dataframe["Sex"] == "female") & (dataframe["Age"] > age_threshold) & (dataframe["Fare"] <= fare_threshold), crossed_feature] = 3
    dataframe.loc[dataframe["Fare"] > fare_threshold, crossed_feature] = 4
    
    return dataframe
titanic_train_dataframe = cross_features(titanic_train_dataframe, age_threshold, fare_threshold)
titanic_test_dataframe = cross_features(titanic_test_dataframe, age_threshold, fare_threshold)

display.display(titanic_train_dataframe.head())
display.display(titanic_test_dataframe.head())
def feature_correlation(dataframe, feature_name):
    survived_rate_list = []
    for index, feature_val in enumerate(list(dataframe[feature_name].unique()), 1):
        hist = dataframe.loc[dataframe[feature_name] == feature_val, "Survived"].value_counts()
        count = hist.sum()
        print(("" if index == 1 else "\n") + "  {0} = {1}".format(feature_name, feature_val))
        print("  {0: >3} passengers".format(count))
        survived_cnt = 0 if 1 not in hist.index else hist[1]
        print("  {0: >3} passengers survived. ({1:.2f})".format(survived_cnt, survived_cnt / count * 100))
        dead_cnt = 0 if 0 not in hist.index else hist[0]
        print("  {0: >3} passengers dead.     ({1:.2f})".format(dead_cnt, dead_cnt / count * 100))
        survived_rate_list.append((feature_val, survived_cnt / count * 100))
    survived_rate_list = sorted(survived_rate_list, key=lambda s: s[1], reverse=True)
    
    print("\nSorted by Survived Rate:")
    for feature_val, survived_rate in survived_rate_list:
        print("  {0} = {1}, survived_rate = {2:.2f}".format(feature_name, feature_val, survived_rate))
feature_correlation(titanic_train_dataframe, "Sex")
feature_correlation(titanic_train_dataframe, "Pclass")
feature_correlation(titanic_train_dataframe, "Embarked")
feature_correlation(titanic_train_dataframe, "AgeFare")
for dataframe in [titanic_train_dataframe, titanic_test_dataframe]:
    dataframe["Title"] = dataframe["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

pd.concat([dataframe[["Title", "PassengerId"]] for dataframe in [titanic_train_dataframe, titanic_test_dataframe]]).\
    groupby("Title", as_index=False).\
    count().\
    rename(columns={"PassengerId": "PassengerCount"}).\
    sort_values(by=["PassengerCount"], ascending=False)
for dataframe in [titanic_train_dataframe, titanic_test_dataframe]:
    dataframe["Title"] = dataframe["Title"].replace(["Mlle", "Ms"], "Miss")
    dataframe["Title"] = dataframe["Title"].replace("Mme", "Mrs")
    dataframe["Title"] = dataframe["Title"].replace(["Rev", "Dr", "Col", "Major", "Capt", "Lady", "Jonkheer", "Dona", "Don", "Countess", "Sir"], "Other")

pd.concat([dataframe[["Title", "PassengerId"]] for dataframe in [titanic_train_dataframe, titanic_test_dataframe]]).\
    groupby("Title", as_index=False).\
    count().\
    rename(columns={"PassengerId": "PassengerCount"}).\
    sort_values(by=["PassengerCount"], ascending=False)
feature_correlation(titanic_train_dataframe, "Title")
titanic_train_dataframe = titanic_train_dataframe.drop(["Name"], axis=1)
titanic_test_dataframe = titanic_test_dataframe.drop(["Name"], axis=1)

display.display(titanic_train_dataframe.head())
display.display(titanic_test_dataframe.head())
plt.figure(figsize=(14,12))
plt.subplot(2, 2, 1)
sns.distplot(titanic_train_dataframe["Fare"], fit=norm)
plt.subplot(2, 2, 2)
sns.distplot(np.log1p(titanic_train_dataframe["Fare"]), fit=norm);
plt.subplot(2, 2, 3)
sns.distplot(titanic_test_dataframe["Fare"], fit=norm)
plt.subplot(2, 2, 4)
sns.distplot(np.log1p(titanic_test_dataframe["Fare"]), fit=norm);
titanic_train_dataframe["NormalizedFare"] = np.log1p(titanic_train_dataframe["Fare"])
display.display(titanic_train_dataframe["NormalizedFare"].describe())
titanic_test_dataframe["NormalizedFare"] = np.log1p(titanic_test_dataframe["Fare"])
display.display(titanic_test_dataframe["NormalizedFare"].describe())
plt.figure(figsize=(14,6))
plt.subplot(1, 2, 1)
sns.distplot(titanic_train_dataframe["Age"], fit=norm);
plt.subplot(1, 2, 2)
sns.distplot(titanic_test_dataframe["Age"], fit=norm);
titanic_train_dataframe.head()
std_sc = StandardScaler()

numeric_columns = ["NormalizedAge", "NormalizedFare", "NormalizedPclass", "NormalizedEmbarked"]

titanic_train_dataframe["NormalizedAge"] = titanic_train_dataframe["Age"]
titanic_train_dataframe["NormalizedPclass"] = titanic_train_dataframe["Pclass"].map({1: 0.62, 2: 0.47, 3:0.24})
titanic_train_dataframe["NormalizedEmbarked"] = titanic_train_dataframe["Embarked"].map({"C": 55.36, "Q": 38.96, "S": 33.90})
titanic_train_dataframe.loc[:, numeric_columns] = std_sc.fit_transform(titanic_train_dataframe.loc[:, numeric_columns])

titanic_test_dataframe["NormalizedAge"] = titanic_test_dataframe["Age"]
titanic_test_dataframe["NormalizedPclass"] = titanic_test_dataframe["Pclass"].map({1: 0.62, 2: 0.47, 3:0.24})
titanic_test_dataframe["NormalizedEmbarked"] = titanic_test_dataframe["Embarked"].map({"C": 55.36, "Q": 38.96, "S": 33.90})
titanic_test_dataframe.loc[:, numeric_columns] = std_sc.transform(titanic_test_dataframe.loc[:, numeric_columns])
titanic_train_dataframe["IsAlone"] = np.where((titanic_train_dataframe["SibSp"] + titanic_train_dataframe["Parch"]) == 0,"yes", "no")
titanic_test_dataframe["IsAlone"] = np.where((titanic_test_dataframe["SibSp"] + titanic_test_dataframe["Parch"]) == 0,"yes", "no")

display.display(titanic_train_dataframe["IsAlone"].value_counts() / titanic_train_dataframe.shape[0] * 100)

display.display(titanic_train_dataframe.head())
display.display(titanic_test_dataframe.head())
def print_survival_rate(dataframe):
    row_count = dataframe.shape[0]
    label_hist = dataframe["Survived"].value_counts()
    print("Survived Rate ({0} passengers):".format(row_count))
    print("  {0} passengers dead. ({1:.2f}%)".format(label_hist[0], label_hist[0] / row_count * 100))
    print("  {0} passengers survivied. ({1:.2f}%)".format(label_hist[1], label_hist[1] / row_count * 100))
print_survival_rate(titanic_train_dataframe)
reindex_titanic_train_dataframe = titanic_train_dataframe.reindex(np.random.permutation(titanic_train_dataframe.index))
num_training = int(reindex_titanic_train_dataframe.shape[0] * 0.8)
num_validation = reindex_titanic_train_dataframe.shape[0] - num_training

print("{0} training examples".format(num_training))
print("{0} validiating examples".format(num_validation))
training_dataframe = reindex_titanic_train_dataframe.head(num_training)
validation_dataframe = reindex_titanic_train_dataframe.tail(num_validation)
def get_logistic_features(dataframe):
    processed_dataframe = pd.DataFrame()
    processed_dataframe["IsFemale"] = dataframe["Sex"].map({"female": 1, "male": 0})
    processed_dataframe["IsAlone"] = dataframe["IsAlone"].map({"yes": 1, "no": 0})
    processed_dataframe["NormalizedPclass"] = dataframe["NormalizedPclass"].copy()
    processed_dataframe["NormalizedEmbarked"] = dataframe["NormalizedEmbarked"].copy()
    processed_dataframe["NormalizedAge"] = dataframe["NormalizedAge"].copy()
    processed_dataframe = pd.concat([processed_dataframe, pd.get_dummies(dataframe["Title"], prefix="Title_")], axis=1)
    return processed_dataframe

def get_logistic_labels(dataframe):
    return dataframe["Survived"].copy()
logistic_training_X = get_logistic_features(training_dataframe)
logistic_training_Y = get_logistic_labels(training_dataframe)
logistic_validation_X = get_logistic_features(validation_dataframe)
logistic_validation_Y = get_logistic_labels(validation_dataframe)

logistic_regression = LogisticRegression()
logistic_regression.fit(logistic_training_X, logistic_training_Y)

logistic_training_accuracy = logistic_regression.score(logistic_training_X, logistic_training_Y)
print("training accuracy: {0:.2f}".format(logistic_training_accuracy))
logistic_validation_accuracy = logistic_regression.score(logistic_validation_X, logistic_validation_Y)
print("validation accuracy: {0:.2f}".format(logistic_validation_accuracy))

coefficient_dataframe = pd.DataFrame()
coefficient_dataframe["Features"] = pd.Series(list(logistic_training_X.columns))
coefficient_dataframe["Coefficients"] = logistic_regression.coef_[0]
coefficient_dataframe.sort_values(by="Coefficients", ascending=False)
logistic_test_X = get_logistic_features(titanic_test_dataframe)
logistic_test_Y = logistic_regression.predict(logistic_test_X)

logistic_submit_dataframe = pd.DataFrame()
logistic_submit_dataframe["PassengerId"] = titanic_test_dataframe["PassengerId"].copy()
logistic_submit_dataframe["Survived"] = logistic_test_Y
logistic_submit_dataframe.to_csv("logistic_submission.csv", index=False, header=["PassengerId", "Survived"])

print_survival_rate(logistic_submit_dataframe)
def get_xgb_features(dataframe):
    processed_dataframe = pd.DataFrame()
    processed_dataframe["IsFemale"] = dataframe["Sex"].map({"female": 1, "male": 0})
    processed_dataframe["Pclass"] = dataframe["Pclass"].copy()
    processed_dataframe["Age"] = dataframe["Age"].copy()
    processed_dataframe["Age"] = dataframe["NormalizedAge"].copy()
    processed_dataframe["Fare"] = dataframe["Fare"].copy()
    processed_dataframe["Fare"] = dataframe["NormalizedFare"].copy()
    processed_dataframe["IsAlone"] = dataframe["IsAlone"].map({"yes": 1, "no": 0})
    processed_dataframe = pd.concat([processed_dataframe, pd.get_dummies(dataframe["Title"], prefix="Title_")], axis=1)
    return processed_dataframe

def get_xgb_labels(dataframe):
    return dataframe["Survived"].copy()
xgb_training_X = get_xgb_features(training_dataframe)
xgb_training_Y = get_xgb_labels(training_dataframe)
xgb_validation_X = get_xgb_features(validation_dataframe)
xgb_validation_Y = get_xgb_labels(validation_dataframe)

xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(xgb_training_X, xgb_training_Y)

xgb_training_accuracy = xgb_classifier.score(xgb_training_X, xgb_training_Y)
print("training accuracy: {0:.2f}".format(xgb_training_accuracy))
xgb_validation_accuracy = xgb_classifier.score(xgb_validation_X, xgb_validation_Y)
print("validation accuracy: {0:.2f}".format(xgb_validation_accuracy))
xgb_test_X = get_xgb_features(titanic_test_dataframe)
xgb_test_Y = xgb_classifier.predict(xgb_test_X)

xgb_submit_dataframe = pd.DataFrame()
xgb_submit_dataframe["PassengerId"] = titanic_test_dataframe["PassengerId"].copy()
xgb_submit_dataframe["Survived"] = xgb_test_Y
xgb_submit_dataframe.to_csv("xgb_submission.csv", index=False, header=["PassengerId", "Survived"])

print_survival_rate(xgb_submit_dataframe)
def nn_input_fn(features, labels, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, labels))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
def get_nn_features(dataframe):
    processed_dataframe = pd.DataFrame()
    processed_dataframe["IsFemale"] = dataframe["Sex"].map({"female": 1, "male": 0})
    processed_dataframe["IsAlone"] = dataframe["IsAlone"].map({"yes": 1, "no": 0})
    processed_dataframe["NormalizedAge"] = dataframe["NormalizedAge"].copy()
    processed_dataframe["NormalizedFare"] = dataframe["NormalizedFare"].copy()
    processed_dataframe["NormalizedPclass"] = dataframe["NormalizedPclass"].copy()
    processed_dataframe["Title"] = dataframe["Title"].copy()
    
    return processed_dataframe

def get_nn_labels(dataframe):
    return dataframe["Survived"].copy()

def construct_feature_columns():
    is_female_col = tf.feature_column.numeric_column("IsFemale")
    is_alone_col = tf.feature_column.numeric_column("IsAlone")
    age_col = tf.feature_column.numeric_column("NormalizedAge")
    fare_col = tf.feature_column.numeric_column("NormalizedFare")
    pclass_col = tf.feature_column.numeric_column("NormalizedPclass")
    title_col = tf.feature_column.categorical_column_with_vocabulary_list(key="Title", vocabulary_list=["Mr", "Miss", "Mrs", "Master", "Other"])
    title_col = tf.feature_column.embedding_column(title_col, dimension=2)
    return [is_female_col, is_alone_col, age_col, fare_col, pclass_col, title_col]
def train_nn(steps, 
             batch_size,
             learning_rate,
             hidden_units,
             training_examples, 
             training_labels,
             validation_examples,
             validation_labels):
    periods = 10
    steps_per_period = steps / periods
    
    training_input_fn = lambda: nn_input_fn(training_examples, training_labels, batch_size=batch_size)
    predict_training_input_fn = lambda: nn_input_fn(training_examples, training_labels, num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: nn_input_fn(validation_examples, validation_labels, num_epochs=1, shuffle=False)
    
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    
    classifier = tf.estimator.DNNClassifier(feature_columns=construct_feature_columns(), hidden_units=hidden_units, optimizer=optimizer)
    
    training_losses = []
    validation_losses = []
    for period in range(periods):
        classifier.train(input_fn=training_input_fn, steps=steps_per_period)
        training_metrics = classifier.evaluate(input_fn=predict_training_input_fn)
        validation_metrics = classifier.evaluate(input_fn=predict_validation_input_fn)
        training_loss = training_metrics["loss"]
        training_losses.append(training_loss)
        validation_loss = validation_metrics["loss"]
        validation_losses.append(validation_loss)
        print("period = {0:<2}, training_loss = {1:.2f}, validation_loss = {2:.2f}".format(period, training_loss, validation_loss))
    
    plt.figure(figsize=(6, 6))
    plt.plot(training_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.title("Losses vs. Periods")
    plt.xlabel("Periods")
    plt.ylabel("loss")
    plt.legend()
    
    return classifier
nn_training_X = get_nn_features(training_dataframe)
nn_training_Y = get_nn_labels(training_dataframe)
nn_validation_X = get_nn_features(validation_dataframe)
nn_validation_Y = get_nn_labels(validation_dataframe)

steps = 75
batch_size = 10
learning_rate = 0.1
hidden_units = [14, 7]
nn_classifer = train_nn(steps,
                        batch_size,
                        learning_rate,
                        hidden_units,
                        nn_training_X,
                        nn_training_Y,
                        nn_validation_X,
                        nn_validation_Y)

training_metrics = nn_classifer.evaluate(input_fn=lambda: nn_input_fn(nn_training_X, nn_training_Y, num_epochs=1, shuffle=False))
print("\ntraining accuracy: {0:.2f}".format(training_metrics["accuracy"]))
validation_metrics = nn_classifer.evaluate(input_fn=lambda: nn_input_fn(nn_validation_X, nn_validation_Y, num_epochs=1, shuffle=False))
print("validation accuracy: {0:.2f}".format(validation_metrics["accuracy"]))
nn_test_X = get_nn_features(titanic_test_dataframe)

def prefict_input_fn(features, batch_size=1):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices(features)
    ds = ds.batch(batch_size).repeat(1)
    return ds.make_one_shot_iterator().get_next()

nn_test_Y = nn_classifer.predict(lambda: prefict_input_fn(nn_test_X))
nn_test_Y = np.array([prediction["class_ids"][0] for prediction in nn_test_Y])

nn_submit_dataframe = pd.DataFrame()
nn_submit_dataframe["PassengerId"] = titanic_test_dataframe["PassengerId"].copy()
nn_submit_dataframe["Survived"] = nn_test_Y
nn_submit_dataframe.to_csv("nn_submission.csv", index=False, header=["PassengerId", "Survived"])

print_survival_rate(nn_submit_dataframe)
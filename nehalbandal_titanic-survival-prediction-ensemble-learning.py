import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv("../input/titanic/train.csv")

print(data.shape)

data.head()
data_explore = data.copy()
data_explore.info()
data_explore = data_explore.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

data_explore.shape
max_embarked = data_explore['Embarked'].value_counts().idxmax()

data_explore['Embarked'] = data_explore['Embarked'].fillna(max_embarked)
data_explore['Family'] = data_explore['SibSp'] + data_explore['Parch']

data_explore['Family'].loc[data_explore['Family'] > 0] = "Yes"

data_explore['Family'].loc[data_explore['Family'] == 0] = "No"
data_explore.describe()
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()

data_explore["sex_enc"] = label_encoder.fit_transform(data_explore["Sex"])

print(label_encoder.classes_)

data_explore["embarked_enc"] = label_encoder.fit_transform(data_explore['Embarked'])

print(label_encoder.classes_)

data_explore["Family_enc"] = label_encoder.fit_transform(data_explore["Family"])

print(label_encoder.classes_)
def fill_with_mean(df, num_cols):

    """

    Instead of replacing null values by mean, this function will replace those null values by 

    any random value within range of (mean + std) & (mean - std).

    """

    for col in num_cols:

        total_null = df[col].isna().sum()

        if total_null>0:

            mean = df[col].mean()

            std = df[col].std()

            lower = mean - std

            upper = mean + std

            random_values = np.random.randint(lower, upper, total_null)

            df[col][np.isnan(df[col])] = random_values.copy()

    return df
data_explore = fill_with_mean(data_explore, ['Age'])
columns = data_explore.columns

plt.figure(figsize=(14, 14))

plt.style.use('seaborn')

i=1

for col in columns:

    plt.subplot(4, 3, i)

    i+=1

    ax = plt.gca()

    counts, _, patches = ax.hist(data_explore[col])

    for count, patch in zip(counts, patches):

        if count>0:

            ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()+5))

    plt.title(col)
data_explore_survived = data_explore[data_explore['Survived']==1]

data_explore_not_survived = data_explore[data_explore['Survived']==0]
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)

data_explore["Age"].hist(alpha=0.7, rwidth=0.85)

plt.ylabel("Total Peoples")

plt.title("Overall")

plt.xticks(ticks=list(range(0,95,15)))

plt.ylim(0,250)

plt.subplot(1, 3, 2)

data_explore_not_survived["Age"].hist(alpha=0.7, rwidth=0.85)

plt.title("Non-Survivers")

plt.xlabel("Age Ranges")

plt.xticks(ticks=list(range(0,95,15)))

plt.ylim(0,250)

plt.subplot(1, 3, 3)

data_explore_survived["Age"].hist(alpha=0.7, rwidth=0.85)

plt.title("Survivers")

plt.xticks(ticks=list(range(0,95,15)))

plt.ylim(0,250)

plt.show()
def get_person_type(passanger):

    age, sex = passanger

    return 'child' if age < 16 else sex



data_explore["Person"] = data_explore[["Age", "Sex"]].apply(get_person_type, axis=1)

data_explore_survived = data_explore[data_explore['Survived']==1]

data_explore_not_survived = data_explore[data_explore['Survived']==0]

data_explore["Person_enc"] = label_encoder.fit_transform(data_explore["Person"])

print(label_encoder.classes_)
overall_age = dict(data_explore["Person"].value_counts())

overall_age = sorted(overall_age.items()) 

# reason for sorting is to make sure labels in overall as well as in survived dict remains in same order.

overall_age_values = [item[1] for item in overall_age]

overall_age_label = [item[0] for item in overall_age]

survived_age = dict(data_explore_survived["Person"].value_counts())

survived_age = sorted(survived_age.items())

survived_age_values = [item[1] for item in survived_age]
x_indexes = np.arange(len(overall_age_label))

width=0.25



plt.bar(x_indexes, overall_age_values, color="blue", label="Overall", width=width)

for i in range(3):

    plt.text(x=x_indexes[i]-0.08, y=overall_age_values[i]+1, s=overall_age_values[i])



plt.bar(x_indexes+width, survived_age_values, color="green", label="Survivers", width=width)

for i in range(3):

    plt.text(x=x_indexes[i]+0.15, y=survived_age_values[i]+1, s=survived_age_values[i])



plt.xticks(ticks=x_indexes, labels=overall_age_label)

plt.title("Survived People ")

plt.ylabel("Total Peoples")

plt.legend()

plt.show()
overall_embarked = dict(data_explore["Embarked"].value_counts())

overall_embarked = sorted(overall_embarked.items())

overall_embarked_values = [item[1] for item in overall_embarked]

overall_embarked_labels = [item[0] for item in overall_embarked]

survived_embarked = dict(data_explore_survived["Embarked"].value_counts())

survived_embarked = sorted(survived_embarked.items())

survived_embarked_values = [item[1] for item in survived_embarked]
x_indexes = np.arange(len(overall_embarked_labels))



plt.bar(x_indexes, overall_embarked_values, color="skyblue", label="Overall", width=width)

for i in range(len(x_indexes)):

    plt.text(x=x_indexes[i]-0.05, y=overall_embarked_values[i], s=overall_embarked_values[i])



plt.bar(x_indexes+width, survived_embarked_values, color="green", label="Survived", width=width)

for i in range(len(x_indexes)):

    plt.text(x=x_indexes[i]+width-0.05, y=survived_embarked_values[i], s=survived_embarked_values[i])



plt.xticks(ticks=x_indexes, labels=overall_embarked_labels)

plt.title("Embarked")

plt.ylabel("Total Peoples")

plt.legend()

plt.show()
overall_pclass = dict(data_explore["Pclass"].value_counts())

overall_pclass = sorted(overall_pclass.items())

overall_pclass_labels = ["Upper Class", "Middle Class", "Lower Class"]  #1 ,2, 3

overall_pclass_values = [item[1] for item in overall_pclass]

survived_pclass = dict(data_explore_survived["Pclass"].value_counts())

survived_pclass = sorted(survived_pclass.items())

survived_pclass_values = [item[1] for item in survived_pclass]
x_indexes = np.arange(len(overall_pclass_labels))



plt.bar(x_indexes, overall_pclass_values, color="skyblue", width=width, label="Overall")

for i in range(len(x_indexes)):

    plt.text(x=x_indexes[i]-0.05, y=overall_pclass_values[i], s=overall_pclass_values[i])



plt.bar(x_indexes+width, survived_pclass_values, color="green", width=width, label="Survived")

for i in range(len(x_indexes)):

    plt.text(x=x_indexes[i]+width-0.05, y=survived_pclass_values[i], s=survived_pclass_values[i])



plt.xticks(ticks=x_indexes, labels=overall_pclass_labels)

plt.title("Ticket Class")

plt.ylabel("Total Peoples")

plt.legend()

plt.show()
n_survived, bins, patches = plt.hist(data_explore_survived["Fare"], bins=list(range(0,51,5)))



plt.hist(data_explore["Fare"], bins=list(range(0,51,5)), label="Overall", edgecolor="black", color="skyblue")

plt.plot(bins[:-1]+2.5, n_survived, color="green", label="Survived",marker="o")

plt.title("Fares")

plt.ylabel("Total Peoples")

plt.legend()

plt.show()
overall_family = dict(data_explore["Family"].value_counts())

overall_family = sorted(overall_family.items())

overall_family_labels = ["No", "Yes"]

overall_family_values = [item[1] for item in overall_family]

survived_family = dict(data_explore_survived["Family"].value_counts())

survived_family = sorted(survived_family.items())

survived_family_values = [item[1] for item in survived_family]
x_indexes = np.arange(len(overall_family_labels))



plt.bar(x_indexes, overall_family_values, color="skyblue", width=width, label="Overall")

for i in range(len(x_indexes)):

    plt.text(x=x_indexes[i]-0.05, y=overall_family_values[i], s=overall_family_values[i])



plt.bar(x_indexes+width, survived_family_values, color="green", width=width, label="Survived")

for i in range(len(x_indexes)):

    plt.text(x=x_indexes[i]+width-0.05, y=survived_family_values[i], s=survived_family_values[i])



plt.xticks(ticks=x_indexes, labels=overall_family_labels)

plt.title("Family")

plt.ylabel("Total Peoples")

plt.legend()

plt.show()
corr_matrix = data_explore.corr()



plt.figure(figsize=(12, 6))

sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), annot=True, square=True)

plt.show()
corr_matrix['Survived'].sort_values(ascending=False)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin
X = data.drop(columns=['Survived'], axis=1)

y = data['Survived'].copy()

X.shape, X.columns
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(X, y):

    strat_train_set = data.iloc[train_index]

    strat_test_set = data.iloc[test_index]



X_train = strat_train_set.drop('Survived', axis=1)

y_train = strat_train_set['Survived'].copy()

X_test = strat_test_set.drop('Survived', axis=1)

y_test = strat_test_set['Survived'].copy()

X_train.shape, X_test.shape
def fill_with_mean(df, num_cols):

    for col in num_cols:

        total_null = df[col].isna().sum()

        if total_null>0:

            mean = df[col].mean()

            std = df[col].std()

            lower = mean - std

            upper = mean + std

            random_values = np.random.randint(lower, upper, total_null)

            df[col][np.isnan(df[col])] = random_values.copy()

    return df
class AddCustomAttribute(BaseEstimator, TransformerMixin):

    """

    New attributes:

        1. Person: whether passenger is child, female or male.

        2. Family: does person come with family member or not.

        

        These are categorical attributes hence their encoding is handled in this custom transformer.

    """

    def __init__(self):

        """

        - cat_imp is imputer for categorical attributes.

        - cat_encoder is for encoding newly created attributes.

        - scaler is for scaling Age attribute.

        - All fields are initialized with None to make sure fit_transform() method will get call only on fit() 

        method of pipeline. Only transform() method will get call on predict().

        """

        self.cat_imp=None

        self.cat_encoder=None

        self.scaler=None

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        try:

            # Fill null values

            X = fill_with_mean(X, ['Age', 'SibSp', 'Parch'])

            if self.cat_imp==None:

                self.cat_imp = SimpleImputer(strategy="most_frequent")

                X['Sex'] = self.cat_imp.fit_transform(X[['Sex']])

            else:

                X['Sex'] = self.cat_imp.transform(X[['Sex']])

            

            # Create new attributes

            X['Person'] = X[['Age', 'Sex']].apply(self.get_person_type, axis=1)

            X['Family'] = X['SibSp'] + X['Parch']

            X['Family'].loc[X['Family'] > 0] = 1

            X['Family'].loc[X['Family'] == 0] = 0

            

            # Scale numerical attribute

            if self.scaler==None:

                self.scaler = StandardScaler()

                X['Age'] = self.scaler.fit_transform(X[['Age']])

            else:

                X['Age'] = self.scaler.transform(X[['Age']])

            

            # Encode new categorical attributes

            if self.cat_encoder==None:

                self.cat_encoder = OneHotEncoder(handle_unknown='ignore')

                X_new = self.cat_encoder.fit_transform(X[['Person', 'Family']])

            else:

                X_new = self.cat_encoder.transform(X[['Person', 'Family']])

            

            X_new = pd.DataFrame(X_new.toarray())

            X_new.index = X.index

            X = pd.concat([X, X_new], axis=1)

            X = X.drop(columns=['Sex', 'Person', 'SibSp', 'Parch', 'Family'], axis=1)

        except:

            print("Error generated in AddCustomAttribute!!!")

        return X

    

    @staticmethod

    def get_person_type(passanger):

        sex_enc = dict({'child':0, 'female':1, 'male':2})

        age, sex = passanger

        return 'child' if age < 16 else sex
class FillWithMeanSD(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        try:

            num_cols = X.columns

            for col in num_cols:

                total_null_fare = X[col].isna().sum()

                if total_null_fare>0:

                    mean = X[col].mean()

                    std = X[col].std()

                    lower = mean - std

                    upper = mean + std

                    random_values = np.random.randint(lower, upper, total_null_fare)

                    X[col][np.isnan(X[col])] = random_values.copy()

        except:

            print("Error generated in FillWithSD!!!")

        return X
cat_pipeline = Pipeline([('cat_imputer', SimpleImputer(strategy="most_frequent")),

                        ('cat_encoder', OneHotEncoder())])



num_pipeline = Pipeline([('num_imputer', FillWithMeanSD()),

                        ('scaler', StandardScaler())])





pre_process = ColumnTransformer([('drop_attrs', 'drop', ['PassengerId', 'Name', 'Ticket', 'Cabin']),

                                 ('num_pipeline', num_pipeline, ['Pclass', 'Fare']),

                                 ('cat_pipeline', cat_pipeline, ['Embarked']),

                                 ('custom_attr', AddCustomAttribute(), ['Age', 'Sex', 'SibSp', 'Parch'])], 

                                remainder='passthrough')
X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)

X_train_transformed.shape, X_test_transformed.shape
X_train.iloc[4,:], X_train_transformed[4]
feature_columns = ['Pclass', 'Fare', 'C', 'Q', 'S', 'Age', 'person_child', 'person_female', 'person_male', 

                   'family_no', 'family_yes']
from sklearn.model_selection import GridSearchCV, StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.linear_model import LogisticRegression



lgr_grid_parm=[{'solver':['liblinear', 'lbfgs'], 'C':list(np.linspace(0.01, 1, 15)), 'penalty':['l1', 'l2'], 'class_weight':[None, 'balanced']}]

lgr_grid_search = GridSearchCV(LogisticRegression(random_state=42, n_jobs=-1), 

                               lgr_grid_parm, cv=kf, scoring="accuracy", return_train_score=True, n_jobs=-1)

lgr_grid_search.fit(X_train_transformed, y_train)
lgr_grid_search.best_params_, lgr_grid_search.best_score_
train_models = []

train_models.append(['Logistic Regression', lgr_grid_search.best_params_, lgr_grid_search.best_score_])
best_lgr_clf = lgr_grid_search.best_estimator_

best_lgr_clf
feature_imp = [ col for col in zip(feature_columns, best_lgr_clf.coef_[0])]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
from sklearn.svm import LinearSVC



svc_grid_parm=[{'C':list(np.linspace(0.01, 1, 20)), 'penalty':['l1', 'l2'], 'class_weight':[None, 'balanced']}]

svc_grid_search = GridSearchCV(LinearSVC(loss="hinge", random_state=42), svc_grid_parm, cv=kf, scoring="accuracy", 

                               return_train_score=True, n_jobs=-1)

svc_grid_search.fit(X_train_transformed, y_train)
svc_grid_search.best_params_, svc_grid_search.best_score_
train_models.append(['Linear SVC', svc_grid_search.best_params_, svc_grid_search.best_score_])
best_svc_clf = svc_grid_search.best_estimator_

best_svc_clf
feature_imp = [ col for col in zip(feature_columns, best_svc_clf.coef_[0])]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
from sklearn.ensemble import RandomForestClassifier



rf_grid_parm=[{'n_estimators':[50, 100, 200, 300], 'max_depth':[6, 8, 16], 

               'class_weight':['balanced', 'balanced_subsample', None]}]

rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), rf_grid_parm, cv=kf, 

                              scoring="accuracy", return_train_score=True, n_jobs=-1)

rf_grid_search.fit(X_train_transformed, y_train)
rf_grid_search.best_params_, rf_grid_search.best_score_
train_models.append(['Random Forest', rf_grid_search.best_params_, rf_grid_search.best_score_])
best_rf_clf = rf_grid_search.best_estimator_

best_rf_clf
feature_imp = [ col for col in zip(feature_columns, best_rf_clf.feature_importances_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
from xgboost import XGBClassifier



xgb_grid_parm=[{'n_estimators':[50, 100, 200], 'max_depth':[4, 8, 16, 24], 'subsample':[0.5, 0.75, 1.0], 

                'colsample_bytree':[0.5, 0.75, 1.0], 'gamma':[0, 0.25, 0.5, 0.75, 1.0]}]

xgb_grid_search = GridSearchCV(XGBClassifier(objective='binary:logistic', learning_rate=0.1, random_state=42, 

                                             n_jobs=-1), xgb_grid_parm, cv=kf, scoring="accuracy", 

                               return_train_score=True, n_jobs=-1)

xgb_grid_search.fit(X_train_transformed, y_train)
xgb_grid_search.best_params_, xgb_grid_search.best_score_
train_models.append(['XGBoost', xgb_grid_search.best_params_, xgb_grid_search.best_score_])
best_xgb_clf = xgb_grid_search.best_estimator_

best_xgb_clf
feature_imp = [ col for col in zip(feature_columns, best_xgb_clf.feature_importances_)]

feature_imp.sort(key=lambda x:x[1], reverse=True)

feature_imp
from sklearn.ensemble import StackingClassifier



named_estimators = [('logistic', best_lgr_clf), ('linear_svc', best_svc_clf), 

                    ('forest', best_rf_clf), ('xgb', best_xgb_clf)]
stack_clf = StackingClassifier(estimators=named_estimators, cv=kf, passthrough=False, n_jobs=-1)

stack_clf.fit(X_train_transformed, y_train)
from sklearn.model_selection import cross_val_score



stack_acc = cross_val_score(stack_clf, X_train_transformed, y_train, scoring="accuracy", cv=kf, n_jobs=-1)

stack_acc = np.round(np.median(stack_acc), 4)

train_models.append(['Stacking Classifier', '', stack_acc])
from sklearn.ensemble import VotingClassifier



voting_clf = VotingClassifier(estimators=named_estimators, n_jobs=-1)

voting_clf.fit(X_train_transformed, y_train)
voting_acc = cross_val_score(voting_clf, X_train_transformed, y_train, scoring="accuracy", cv=kf, n_jobs=-1)

voting_acc = np.round(np.mean(voting_acc), 4)

train_models.append(['Voting Classifier', '', voting_acc])
pd.set_option('display.max_colwidth', -1)



train_models_df = pd.DataFrame(train_models, columns=['Model', 'Best Paramas', 'Accuracy'])

train_models_df
def plot_results(model_names, model_accuracy):

        

    plt.figure(figsize=(12, 5))

    x_indexes = np.arange(len(model_names))     

    width = 0.15                            

    

    plt.barh(x_indexes, model_accuracy)

    for i in range(len(x_indexes)):

        plt.text(x=model_accuracy[i], y=x_indexes[i], s=str(model_accuracy[i]), fontsize=12)

    

    plt.xlabel("Accuracy Score", fontsize=14)

    plt.yticks(ticks=x_indexes, labels=model_names, fontsize=14)

    plt.title("Results on Test Dataset")

    plt.show()
results = dict()

best_models = [best_lgr_clf, best_svc_clf, best_rf_clf, best_xgb_clf, stack_clf, voting_clf]

model_names = []

model_accuracy = []



for model in best_models:

    test_accuracy_scores = cross_val_score(model, X_test_transformed, y_test, scoring="accuracy", cv=kf, n_jobs=-1)

    test_accuracy_scores = np.round(test_accuracy_scores,4)

    test_accuracy = np.round(np.mean(test_accuracy_scores),4)

    model_names.append(model.__class__.__name__)

    model_accuracy.append(test_accuracy)
plot_results(model_names, model_accuracy)
best_model = best_models[np.argmax(model_accuracy)]

best_model
final_model = Pipeline([('pre_process', pre_process),

                        ('best_model', best_model)])

final_model.fit(X_train, y_train)
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.info()
predictions = final_model.predict(test_data)
test_predictions = pd.DataFrame(test_data['PassengerId'])

test_predictions['Survived'] = predictions.copy()

test_predictions.head()
test_predictions.shape
test_predictions.to_csv("./submission.csv", index=False)
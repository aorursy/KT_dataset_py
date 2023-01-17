import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# KNN Test

def train_test_KNN(X_train, y_train, X_test, y_test):

    best_neighbors=0

    best_score=0

    current_train_score=0

    for neighbors in range(1,100):

        knn = KNeighborsClassifier(n_neighbors=neighbors)

        knn.fit(X_train, y_train)

        prediction_train = knn.score(X_train, y_train)

        prediction_test = knn.score(X_test, y_test)

        if prediction_test > best_score:

            best_score = prediction_test

            best_neighbors = neighbors

            current_train_score = prediction_train

    print("KNN Score:\t Train: {:.4f} \t Test: {:.4f} \t neighbors: {}".format(current_train_score, best_score, best_neighbors))

    knn = KNeighborsClassifier(n_neighbors=best_neighbors)

    knn.fit(X_train, y_train)

    return knn
#Logistic Regression test



def train_test_logreg(X_train, y_train, X_test, y_test):

    best_score = 0

    current_prediction_train = 0

    best_c = 0

    for c in np.arange(0.1,10,0.1):

        logreg = LogisticRegression(C=c, solver='newton-cg').fit(X_train, y_train)

        prediction_train = logreg.score(X_train, y_train)

        prediction_test = logreg.score(X_test, y_test)

        if prediction_test > best_score:

            best_score = prediction_test

            best_c = c

            current_prediction_train = prediction_train

    print("LogReg Score:\t Train: {:.4f} \t Test: {:.4f} \t C: {}".format(current_prediction_train, best_score, best_c))

    return logreg

#Random Forest test

def train_test_random_forest(X_train, y_train, X_test, y_test):

    best_test_score = 0

    current_train_score = 0

    best_estimator = 0

    best_depth = 0

    

    for estimatorCount in range(1,10):

        for depth in range (1,10):

            forest = RandomForestClassifier(n_estimators=estimatorCount, max_depth=depth, random_state=0)

            forest.fit(X_train, y_train)



            prediction_train = forest.score(X_train, y_train)

            prediction_test = forest.score(X_test, y_test)

            if prediction_test > best_test_score:

                best_test_score = prediction_test

                current_train_score = prediction_train

                best_estimator = estimatorCount

                best_depth = depth

    print("Forest Score:\t Train: {:.4f} \t Test: {:.4f} \t ({} estimators \t {} depth)".format(current_train_score, best_test_score, best_estimator, best_depth))

    

    #Change rf parameters to match the best found, and re-train.

    forest = RandomForestClassifier(n_estimators=best_estimator, max_depth=best_depth, random_state=0)

    forest.fit(X_train, y_train)

    return forest
def train_test_gradient_boost(X_train, y_train, X_test, y_test):

    

    best_score = 0

    current_training_score = 0

    best_depth = 0

    best_learning_rate = 0

    

    for depth in range(1,30):

        print("Depth: \t {}".format(depth))

        for rate in np.arange(0.01,1,0.1):

            gb = GradientBoostingClassifier(random_state=0, max_depth=depth, learning_rate=rate)

            gb.fit(X_train, y_train)

            prediction_train = gnb.score(X_train, y_train)

            prediction_test = gnb.score(X_test, y_test)

            

            if prediction_test > best_score:

                best_score = prediction_test

                current_training_score = prediction_train

                best_depth = depth

                best_learning_rate = rate

            

    print("Grad. Boost \t Train: {:.4f} \t Test: {:.4f} \t Depth: {} \t Rate: {}".format(current_training_score, best_score, best_depth, best_learning_rate))

    return gb
# Gaussian Naive Bayes

def train_test_gnb(X_train, y_train, X_test, y_test):

    gnb = GaussianNB()

    gnb.fit(X_train, y_train)

    prediction_train = gnb.score(X_train, y_train)

    prediction_test = gnb.score(X_test, y_test)

    print("GaussianNB: \t Train: {:.4f} \t Test: {:.4f}".format(prediction_train, prediction_test))

    return gnb
# Decision Tree

def train_test_tree(X_Train, y_train, X_test, y_test):

    best_score = 0

    current_training_score = 0

    best_depth = 0

    for depth in range(1,100):

        tree = DecisionTreeClassifier(random_state=0, max_depth=depth)

        tree.fit(X_train, y_train)

        prediction_train = tree.score(X_train, y_train)

        prediction_test = tree.score(X_test, y_test)

        if(prediction_test > best_score):

            best_score = prediction_test

            best_depth = depth

            current_training_score = prediction_train

    

    print("Tree: \t\t Train: {:.4f} \t Test: {:.4f} \t Depth: {}".format(current_training_score, best_score, best_depth))

    tree = DecisionTreeClassifier(random_state=0, max_depth=best_depth)

    tree.fit(X_train, y_train)

    # print("Feature Names: \t{}".format(feature_names))

    # print("Feature Importances: \t{}".format(tree.feature_importances_))

    

    return tree
def output_predictions(model, data, passengerIds):

    predictions = model.predict(data)

    output = pd.DataFrame({'PassengerId': passengerIds, 'Survived': predictions})

    output.to_csv('submission.csv', index=False)

    print("Predictions Saved.")
# Dataset Cleaning Function

def clean(raw_train_dataset, raw_competition_dataset):

    # Drop columns which are not to be included in model.

    processed_train_data = raw_train_dataset.drop(["Name", "PassengerId", "Ticket", "Embarked"], axis=1)

    processed_competition_data = raw_competition_dataset.drop(["Name", "PassengerId", "Ticket", "Embarked"], axis=1)

    

    # Instead of dropping "Sex", convert to int.

    processed_train_data.Sex = processed_train_data.Sex.eq("male").mul(1)

    processed_competition_data.Sex = processed_competition_data.Sex.eq("male").mul(1)

    

    if 'Survived' in processed_train_data.columns:

        processed_train_data = processed_train_data.drop("Survived", axis=1)

   

    #Add new column - Boolean - Has_Cabin_Number

    processed_train_data['Has_Cabin_Number'] = ~processed_train_data["Cabin"].isnull()

    processed_competition_data['Has_Cabin_Number'] = ~processed_competition_data["Cabin"].isnull()

    

    #Drop Cabin 

    processed_train_data = processed_train_data.drop(["Cabin"], axis=1)

    processed_competition_data = processed_competition_data.drop(["Cabin"], axis=1)



    #Handle Missing Values

    si = SimpleImputer()

    imputed_train_data = pd.DataFrame(si.fit_transform(processed_train_data))

    imputed_competition_data = pd.DataFrame(si.transform(processed_competition_data))

    

    #replace columns

    imputed_train_data.columns = processed_train_data.columns

    imputed_competition_data.columns = processed_competition_data.columns

    

    processed_train_data.head()

    

    return imputed_train_data, imputed_competition_data
data = pd.read_csv("/kaggle/input/titanic/train.csv")

data.head()
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



# import data



train_dataset = pd.read_csv("/kaggle/input/titanic/train.csv")

raw_competition_dataset = pd.read_csv("/kaggle/input/titanic/test.csv")



# get labels and drop from train dataset

labels = train_dataset["Survived"]

train_dataset = train_dataset.drop("Survived", axis=1)





#drop fields wich are not beneficial

train_dataset = train_dataset.drop(["Name", "Ticket", "PassengerId","Embarked"], axis=1)

competition_dataset = raw_competition_dataset.drop(["Name", "Ticket", "PassengerId","Embarked"], axis=1)



#Calculate a boolean variable indicating if record has a named column (reduce cardinality), then drop Cabin field.

train_dataset["Has_Cabin"] = train_dataset.Cabin.notnull()

competition_dataset["Has_Cabin"] = raw_competition_dataset.Cabin.notnull()



train_dataset = train_dataset.drop(["Cabin"], axis=1)

competition_dataset = competition_dataset.drop(["Cabin"], axis=1)



#split into train and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(train_dataset, labels, random_state=0)



#Define pre-processing steps.

# get numerical column labels



numerical_cols = train_dataset.select_dtypes(include=["int"]).columns 

# Numerical - 'PassengerId', 'Pclass', 'SibSp', 'Parch'



# get categorical column labels

categorical_cols = train_dataset.select_dtypes(include=["object"]).columns

# Categorical - 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'



# numerical values pre-processing steps - impute

numerical_transformer = Pipeline(steps = [

                        ("imputer", SimpleImputer()),

                        ("scaler", StandardScaler())

]) 



# categorical values pre-processing steps - imput then One Hot Encode.



#to do - potentially drop high cardinality cateogrical variables.



categorical_transformer = Pipeline(steps = [

                            ('imputer', SimpleImputer(strategy="most_frequent")),

                            ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore'))

])    



# Bundle pre-processing steps into a column transformer

preprocessor = ColumnTransformer(transformers = [

                            ('numerical', numerical_transformer, numerical_cols),

                            ('categorical', categorical_transformer, categorical_cols)

])



#Define model

# model = XGBClassifier(n_estimators=1000, learning_rate=0.01, random_state=0, early_stopping_rounds=5, max_depth=3)

model = DecisionTreeClassifier(max_depth=3)

clf = Pipeline(steps = [

                ('preprocessor', preprocessor),

                ('model', model)

])



X_final = X_train.append(X_valid)

y_final = y_train.append(y_valid)





#clf.fit(X_valid, y_valid)

clf.fit(X_final, y_final)



predictions = clf.predict(X_valid)

accuracy = accuracy_score(y_valid, predictions)



# generate predictions on competiton set

results = clf.predict(competition_dataset)





# output predictions in correct format to csv.

output = pd.DataFrame({'PassengerId' : raw_competition_dataset.PassengerId, 

         'Survived' : results})



output.to_csv("submission.csv", index=False)



accuracy

#train_dataset
''' Old Tests'''



# knn = train_test_KNN(X_train, y_train, X_test, y_test)



# logreg = train_test_logreg(X_train, y_train, X_test, y_test)

# rf = train_test_random_forest(X_train, y_train, X_test, y_test)



# gnb = train_test_gnb(X_train, y_train, X_test, y_test)

# tree = train_test_tree(X_train, y_train, X_test, y_test)



# gb = train_test_gradient_boost(X_train, y_train, X_test, y_test)



# output_predictions(rf, clean_competition_dataset, raw_competition_dataset.PassengerId)

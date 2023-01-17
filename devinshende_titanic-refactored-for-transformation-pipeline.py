import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from termcolor import cprint, colored # colorful printing

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
# I want to have my own test set so I will split the train one they provide into two
from sklearn.model_selection import train_test_split
def load_split_data():
    df = pd.read_csv("../input/train.csv")
    train_labels = pd.Series(df["Survived"])
    test_size = 0.3
    split = round(len(df)*(1-test_size))
    X_train = df[:split].drop("Survived",axis=1) # training data wihout labels
    y_train = df[:split]["Survived"] # labels for training data
    X_test = df[split:].drop("Survived",axis=1) # validation data without labels
    y_test = df[split:]["Survived"] # labels for validation data
    return X_train, y_train, X_test, y_test
X_train, y_train, X_test, y_test = load_split_data()
# functions that are nice just for the me to see missing data (nothing to do with the algorithm)
def how_many_missing(dataset, feature,printing=True,start=""):
    NAs = 0
    index = 0
    isna = list(dataset[feature].isna())
    for age in dataset[feature]:
        if isna[index] == True:
    #         print(train_copy[feature][index], "is na")
            NAs += 1
        index+=1
    if printing: print(start,NAs, "missing values for",feature)
    return NAs

def see_missing(dataset):
    features = list(dataset.columns.values)
    for feature in features:    
        how_many_missing(dataset,feature)

see_missing(X_train)
from sklearn.base import BaseEstimator, TransformerMixin
# get rid of any rows containing NaN for any feature
# don't use it. Runs into issues because it drops rows on training data but doesn't drop those rows in training labels
class DropMissingRows(BaseEstimator, TransformerMixin):
    def __init__(self,printing=False):
        self.p = printing

    def fit(self, X, y=None):
        return self # don't need to fit anything
    
    def transform(self, X, y=None):
        # do the actual stuff
        isna = X.isna()
        if True in np.array(isna): # only do stuff if there is stuff to do it on
            missing = 0
            for column in list(X.columns.values):
                missing += how_many_missing(X,column,printing=False)
            if self.p: print("\tdropping {} rows that contained missing values".format(missing))
            if self.p: print("\tlength of X before dropping rows:",len(X))
            X.dropna(inplace=True) # fill in values with mean
            if self.p: print("\tlength of X after dropping rows:",len(X))
            if self.p:
                for column in list(X.columns.values):
                    how_many_missing(X,column,start="\t")
        else:
            cprint("\tThere were no missing values so nothing was dropped","red")
        return X
# get rid of missing age values by filling them in with the mean. Not the best solution but works
class FillWithMean(BaseEstimator, TransformerMixin):
    def __init__(self, feature="Age",printing=False):
        self.feature = feature
        self.p = printing
        if type(self.feature) == list:
            raise ValueError("feature must be one item, not list. EX: 'Age'")
            
    def fit(self, X, y=None):
        return self # don't need to fit anything
    
    def transform(self, X, y=None):
        # do the actual stuff
#         if self.p: print("In FillWithMean.transform()")
        try:
            mean = round(X[self.feature].mean()) # get the mean if it is numerical
        except:
            raise ValueError(f"you can't run FillWithMean on categorical features. They must be numerical because you can't do the mean on categories. \"{self.feature}\" is not numerical")
            return None
        isna = X[self.feature].isna()
        if True in list(isna): # only do stuff if there is stuff to do it on
            if self.p: print("\tfilling in {} missing values for {} with {}...".format(how_many_missing(X,self.feature,False),colored("'"+str(self.feature)+"'","cyan"), mean))
            X[self.feature].fillna(mean,inplace=True) # fill in values with mean
            if self.p: how_many_missing(X,self.feature,start="\t") # see how many are missing now
        else:
            cprint("\tThere were no missing values for \"{}\" so nothing was filled in".format(self.feature),"red")
        return X
# a lot of the features are useless. Remove them from the dataset
class RemoveFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, featurestoremove, printing=False):
        self.features = featurestoremove
        self.printing = printing
        for item in self.features:
            if type(item) != str:
                raise ValueError("Parameters must all be strings that correspond to features in the dataset. \nEX: RemoveFeatures('Name','PassengerId') or RemoveFeatures('Cabin')")
    def fit(self, X, y=None):
        return self # don't need to fit anything
    
    def transform(self, X, y=None):
        # do the actual stuff
        # drop the features
        p = self.printing
        if p:
#             print("In RemoveFeatures.transform()")
            print("\tBefore removing:",colored(str(list(X.columns.values)),"cyan")) 
            print("\tremoving the features",colored(str(self.features),"cyan"))
        for f in self.features:
            if f in list(X.columns.values): # if it doesn't exist don't drop it. Defensive coding
                X = X.drop(f,axis=1)
            else:
                cprint('\tRemoveFeatures didn\'t remove "{}" because that feature does not exist'.format(f),"red")
        if p: 
            print("\tAfter removing:",colored(str(list(X.columns.values)),"cyan"))
        return X
class FillEmbarked(BaseEstimator, TransformerMixin):
    def __init__(self, value="S", printing=False):
        self.p = printing
        self.valuetofillwith = value

    def fit(self, X, y=None):
        return self # don't need to fit anything
    
    def transform(self, X, y=None):
        missing = how_many_missing(X,"Embarked",False,start="\t")
        if missing == 1:
            if self.p: print(f'\tfilling in 1 missing value for "Embarked" with "S"...')
        elif missing == 0:
            cprint("\tThere were no missing values for 'Embarked' so nothing was filled in","red")
        else:
            if self.p: print(f'\tfilling in {missing} missing values for "Embarked" with "{self.valuetofillwith}"...')
        X["Embarked"].fillna(self.valuetofillwith, inplace=True)
        if self.p: how_many_missing(X,"Embarked",start="\t")
        return X
class ConvertBacktoDF(BaseEstimator, TransformerMixin):
    def __init__(self, printing=False):
        self.p = printing
    
    def fit(self, X, y=None):
        return self # don't need to fit anything
    
    def transform(self, X, y=None):
        if self.p: print("\tDataset type: ",type(X))
        dtype = str(type(X))
        if dtype == "<class 'numpy.ndarray'>":
            if self.p: print("\tX was a numpy array. Converting it to a DataFrame...")
            X = pd.DataFrame(X)
        elif dtype == "<class 'scipy.sparse.csr.csr_matrix'>":
            if self.p: print("\tX was a scipy sparse matrix. Converting it to a numpy array and then into a DataFrame...")
            X = pd.DataFrame(X.toarray())
        elif dtype == "<class 'pandas.core.frame.DataFrame'>":
            cprint("\tX is already a DataFrame. Didn't have to convert it","red")
        return X
class myPipeline(BaseEstimator, TransformerMixin):
    def __init__(self,steps,printing=False):
        self.steps = list(steps)
        self.printing = printing
        self.names = [x[0] for x in self.steps]
        self.transformers = [t[1] for t in self.steps]
        
    def fit(self,X,y=None):
        return self
    
    def set_printing(self,printing):
        self.p = printing
    
    def transform(self,X,y=None):
        p = self.printing
        for i in range(len(self.steps)):
            if p: print("running ",self.names[i])
            X = self.transformers[i].fit_transform(X)
#             if self.printing: print(pd.DataFrame(X).head())
            if p: print("\n")
        return X
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
num_pipeline = myPipeline([
    # my own classes to transform the numerical data
    ('remove_cat_features', RemoveFeatures(["Name","Sex","Ticket","Cabin","Embarked"],printing=False)), 
    # get rid of all numerical features and useless PassengerId one too
    ('remove_useless_features', RemoveFeatures(["PassengerId"],printing=False)),
    # get rid of useless numerical features
    ('fill_missing_ages', FillWithMean("Age",printing=False)), 
    # fills all the empty ages and 1 missing embarked value
    ('stdscaler', StandardScaler()),
    # makes ranges of values the same
    ('convert_back_to_df', ConvertBacktoDF(printing=False))
    # convert it from numpy nd array to pandas DataFrame
],False)

cat_pipeline = myPipeline([
    ('remove_num_features',RemoveFeatures(["PassengerId","Age","Pclass","SibSp","Parch","Fare"])), 
    # get rid of all numerical features
    ('remove_useless_features',RemoveFeatures(["Name","Cabin","Ticket"],False)),
    # get rid of stuff that is categorical but not helpful
    ('fill_missing_embarked_values',FillEmbarked()),
    # fill in one or two missing values for embarked feature
    ('onehot', OneHotEncoder()),
    # sklearn's one hot encoder
    ('convert_back_to_df', ConvertBacktoDF()),
    # convert from scipy sparse matrix (one hot encoding made it that) to pandas DataFrame
],False)

full_pipeline = FeatureUnion([
    ('num_pipeline',num_pipeline), 
    # do all the numerical transformations (see above)
    ('cat_pipeline',cat_pipeline),
    # do all the categorical transformations (see above)
    # then join the two together into one transformed dataset!
])
X_train, y_train, X_test, y_test = load_split_data() # get the unchanged dataset

train_X_prepared = full_pipeline.transform(X_train)
test_X_prepared = full_pipeline.transform(X_test)

print("length of train data: ",len(X_train))
print("length of train labels: ",len(y_train))
model = KNeighborsClassifier()
model.fit(train_X_prepared,y_train)
print("model:",model)

z = model.score(test_X_prepared,y_test)
cprint("\nScore: "+str(z),"yellow") # 80 ain't bad
pd.DataFrame(train_X_prepared).head() # see how the final prepared data looks like
# let's now submit our predictions to the challenge
X_test = pd.read_csv("../input/test.csv")
Ids = X_test["PassengerId"]

X_test_prepared = pd.DataFrame(full_pipeline.transform(X_test))

if False in X_test_prepared.isna():
    print("oh noes! missing value... filling it in with 1")
X_test_prepared.fillna(1,inplace=True)
see_missing(X_test_prepared) # uh oh. Missing value for 4 whatever that is
# lastly... time to predict the actual test set data and submit it to kaggle
predictions = model.predict(X_test_prepared)
print(len(X_test_prepared),"rows of test data")
print(len(predictions),"predictions") # should be 418 but isn't for some stupid reason
print(predictions)
cprint("Predictions:","cyan")
for i in predictions:
    if i == 0:
        print(colored("Died","red"),end=", ") # red means died
    elif i == 1:
        print(colored("Survived","green"),end=", ") # green means survived
import csv
with open("../submission.csv","w") as file:
    writer = csv.writer(file)
    writer.writerow(["PassengerId","Survived"])
    for Id, pred in list(zip(Ids, predictions)):
        writer.writerow([Id,pred])

with open("../submission.csv","r") as file:
    reader = csv.reader(file)
    x = file.read()
    print(x)
from sklearn.gaussian_process import GaussianProcessClassifier
def load_train():
    df = pd.read_csv("../input/train.csv")
    train_labels = pd.Series(df["Survived"])
    test_size = 0.3
    X_train = df.drop("Survived",axis=1)
    y_train = train_labels
    return X_train, y_train

X_train, y_train = load_train() # get the unchanged dataset
print("length of train data: ",len(X_train))
print("length of train labels: ",len(y_train))

train_X_prepared = full_pipeline.transform(X_train)

model = GaussianProcessClassifier()
model.fit(train_X_prepared,y_train)
print("model:",model)
print(model,"is a model")
# z = model.score(test_X_prepared,y_test)
# cprint("\nScore: "+str(z),"yellow") # 80 ain't bad
pd.DataFrame(train_X_prepared).head() # see how the final prepared data looks like
# lastly... time to predict the actual test set data and submit it to kaggle
predictions = model.predict(X_test_prepared)
print(len(X_test_prepared),"rows of test data")
print(len(predictions),"predictions") # should be 418 but isn't for some stupid reason
print(predictions)
cprint("Predictions:","cyan")
for i in predictions:
    if i == 0:
        print(colored("Died","red"),end=", ") # red means died
    elif i == 1:
        print(colored("Survived","green"),end=", ") # green means survived
import csv
with open("../submission.csv","w") as file:
    writer = csv.writer(file)
    writer.writerow(["PassengerId","Survived"])
    for Id, pred in list(zip(Ids, predictions)):
        writer.writerow([Id,pred])

with open("../submission.csv","r") as file:
    reader = csv.reader(file)
    x = file.read()
    print(x)



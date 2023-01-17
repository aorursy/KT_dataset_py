import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer

import copy
import operator
import datetime
import warnings
from itertools import product
warnings.filterwarnings("ignore")

#useful way to visualize directory
!ls ../input
!ls ../input/*
TRAIN_SIZE = 100000
#we'll not be using the full training set, change this parameter if you wish to use more/less
train = pd.read_csv("../input/food-com-recipes-and-user-interactions/interactions_train.csv")
validation = pd.read_csv("../input/food-com-recipes-and-user-interactions/interactions_validation.csv")
recipes_RAW = pd.read_csv("../input/food-com-recipes-and-user-interactions/RAW_recipes.csv")
recipes_RAW = recipes_RAW.rename({"id":"recipe_id"},axis=1)
interactions_RAW = pd.read_csv("../input/food-com-recipes-and-user-interactions/RAW_interactions.csv")

#since we'll be using k-fold validation, there is not need to use a seperate validation set
train = pd.concat([train,validation],axis=0)
test = pd.read_csv("../input/food-com-recipes-and-user-interactions/interactions_test.csv")
train = train.iloc[:TRAIN_SIZE,:]
recipes_RAW.iloc[1,10].split("delimiter")
train = train.merge(interactions_RAW.drop(["rating","date"],axis=1),on=["user_id","recipe_id"])
train = train.merge(recipes_RAW,on="recipe_id")
test = test.merge(interactions_RAW.drop(["rating","date"],axis=1),on=["user_id","recipe_id"])
test = test.merge(recipes_RAW,on="recipe_id")
train.head(2)
train_copy = train.copy()
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
train["submitted"] = pd.to_datetime(train["submitted"])
test["submitted"] = pd.to_datetime(test["submitted"])
train.info()
class FeatureTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, vectorize = True, time=True, drop = True):
        self.vectorize = vectorize
        self.drop = drop
        self.time = time
        if vectorize:
            self.vectorizer1 = CountVectorizer(lowercase=True,max_df=0.3,min_df=0.005,max_features=3000) #review
            self.vectorizer2 = CountVectorizer(lowercase=True,max_df=0.3,min_df=0.005,max_features=3000) #description
            self.vectorizer3 = CountVectorizer(lowercase=True,max_df=0.3,min_df=0.005,max_features=3000) #steps
            self.vectorizer4 = CountVectorizer(lowercase=True,max_df=0.8,min_df=0.01,max_features=200)  #tags
            self.vectorizer5 = CountVectorizer(lowercase=True,max_df=0.8,min_df=0.01,max_features=300)  #ingredients
        
    def fit(self, X, y=None):
        X.loc[X.description.isnull(), "description"] = " "
        if self.vectorize:
            self.vectorizer1.fit(X["review"])
            self.vectorizer2.fit(X["description"])
            self.vectorizer3.fit(X["steps"])
            self.vectorizer4.fit(X["tags"])
            self.vectorizer5.fit(X["ingredients"])
        return self
    
    def transform(self, X, y=None):
        processed = ["nutrition"]
        X = copy.deepcopy(X)
        X = X.drop(columns=["u","i","user_id","recipe_id","name","contributor_id"])
        
        X.loc[X.description.isnull(), "description"] = " "
        X["has_description"] = 1
        X.loc[X.description == " ", "has_description"] = 0
        X["calories"] = X["nutrition"].apply(lambda x: np.array(eval(x)).sum())

        if self.vectorize:
            processed.append("review")
            transformed = pd.DataFrame(self.vectorizer1.transform(X["review"]).toarray()).add_prefix("review_")
            X = pd.concat([X, transformed],axis=1)
            
            processed.append("description")
            transformed = pd.DataFrame(self.vectorizer2.transform(X["description"]).toarray()).add_prefix("description_")
            X = pd.concat([X, transformed],axis=1)
            
            processed.append("steps")
            #works because CountVectorizer ignores puctuations
            transformed = pd.DataFrame(self.vectorizer3.transform(X["steps"]).toarray()).add_prefix("steps_")
            X = pd.concat([X, transformed],axis=1)
            
            processed.append("tags")
            #works because CountVectorizer ignores puctuations
            transformed = pd.DataFrame(self.vectorizer4.transform(X["tags"]).toarray()).add_prefix("tags_")
            X = pd.concat([X, transformed],axis=1)
            
            processed.append("ingredients")
            #works because CountVectorizer ignores puctuations
            transformed = pd.DataFrame(self.vectorizer5.transform(X["ingredients"]).toarray()).add_prefix("ingredients_")
            X = pd.concat([X, transformed],axis=1)
            
        if self.time:
            processed.extend(["date","submitted"])
            X["year"] = [x.year for x in X["date"]]
            X["month"] = [x.month for x in X["date"]]
            X["day_of_week"] = [x.dayofweek for x in X["date"]]
            
            X["upload_year"] = [x.year for x in X["submitted"]]
            
        if self.drop:
            X = X.drop(columns=processed)
        return X
now = datetime.datetime.now()
transformer = FeatureTransformer()
transformer.fit(train)
train = transformer.transform(train)
end = datetime.datetime.now()
print("Train Transformation time: {:5f} minutes".format((end - now).total_seconds() / 60))
train
now = datetime.datetime.now()
test.loc[test.review.isnull(),"review"] = " " 
#one review is missing for who knows what reason
test = transformer.transform(test)
end = datetime.datetime.now()
print("Test Transformation time: {:5f} minutes".format((end - now).total_seconds() / 60))
y_train, X_train = train["rating"], train.iloc[:,1:]
y_test, X_test = test["rating"], test.iloc[:,1:]
print("Train shape: {}\nTest shape:  {}".format(train.shape, test.shape))
class GBDT():
    def __init__(self, args = {}, random_state=42, verbose=0):
        self.random_state = random_state
        self.args = args
        self.verbose = verbose
        self.models = []
        self.func = GradientBoostingClassifier
        
        self.feature_importances = {}
    def fit(self, X_train_, y_train_, cv=1, analyze=False):
        self.models = []
        classification_train(self,X_train_, y_train_, cv,func=self.func,analyze=analyze)
            
    def predict(self, X_test, verbose=0):
        preds = np.zeros(len(X_test))
        if verbose!=0: print("[",end="")
        for model in self.models:
            if verbose!=0:print("-",end="")
            preds += model.predict(X_test)
        if verbose!=0:print("]")
        preds /= len(self.models)
        return [int(round(i)) for i in preds]
    
    def evaluate(self, X_test, y_test, return_score=True):
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print("Accuracy: {:5f}".format(acc))
        if return_score:
            return acc
        
    def copy(self):
        return self.func(random_state=self.random_state, verbose=self.verbose, **self.args)
def classification_train(this, X_train_, y_train_, cv=1, func=None, analyze=False):
    v = 1 if this.verbose == 2 else 0
    if cv == 1:
        this.models.append(func(random_state=this.random_state, verbose=v, **this.args))
        this.models[0].fit(X_train_,y_train_)
        preds_train = [round(i) for i in this.models[0].predict(X_train_)]
        print("Train Accuracy: {:5f}".format(accuracy_score(y_train_, preds_train)))
        if analyze:
            this.feature_importances = dict(zip(this.models[0].feature_importances_, X_train_.columns.values))
            this.feature_importances = sorted(this.feature_importances.items(), key=operator.itemgetter(0),reverse=True)[:20]
            plt.figure(figsize=(30,10))
            plt.bar([x[1] for x in this.feature_importances],[x[0] for x in this.feature_importances],color="red")
            plt.title("Feature Importances")
            plt.show()
    else:
        importances = np.zeros(len(X_train.columns))
        val_accuracies = []
        train_accuracies = []
        size = len(X_train)
        train_size = 1 - 1 / cv
        
        for i in range(cv):
            this.models.append(func(random_state=this.random_state, verbose=v, **this.args))
            if this.verbose != 0: print("{}/{}".format(i+1,cv))
            X_fold, X_val, y_fold, y_val = train_test_split(X_train_, y_train_, train_size=train_size)
            if this.verbose != 0: print("Training on {} instances, Validating on {} instances".format(len(X_fold),len(X_val)))
            this.models[i].fit(X_fold,y_fold)
            val_preds = [round(i) for i in this.models[i].predict(X_val)]
            val_accuracies.append(accuracy_score(y_val,val_preds))
            importances += this.models[i].feature_importances_
            if this.verbose != 0: print("Validation Accuracy: {:5f}\n".format(accuracy_score(y_val,val_preds)))
            
        if analyze:
            this.feature_importances = dict(zip(importances, X_train.columns.values))
            this.feature_importances = sorted(this.feature_importances.items(), key=operator.itemgetter(0),reverse=True)
            print("Mean Validation Accuracy: {:5f}".format(np.mean(val_accuracies)))
            f,axs = plt.subplots(1, 2,figsize=(30,10))
            axs = axs.flatten()
            axs[0].bar([x[1] for x in this.feature_importances],[x[0] for x in this.feature_importances],color="red")
            axs[0].set_title("Feature Importances")
            plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=90)
            
            axs[1].hist(val_preds)
            axs[1].set_title("Prediction Distribution")
            plt.show()
class GridSearch():
    def __init__(self, model, params, verbose=0):
        self.verbose = verbose
        self.model = model
        self.best_params = None
        self.params = params
        self.features = list(params.keys())
        self.combinations = list(product(*list(params.values())))
        
    def fit(self, X, y, train_size=0.8, cv=1, stratify=True, custom=True):
        print("Total Combinations: " + str(len(self.combinations)))
        if cv == 1:
            accuracies = []
            highest, count = 0, 0
            if stratify: X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size,stratify=y,shuffle=True)
            else: X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size,shuffle=True)
            start = datetime.datetime.now()
            for combination in self.combinations:
                count += 1
                model = self.model(args=dict(zip(self.features,combination)))
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test,preds)
                accuracies.append(acc)
                if acc > highest:
                    self.best_params = dict(zip(self.features,combination))
                    highest = acc
                end = datetime.datetime.now()
                duration = (end-start).total_seconds()
                if self.verbose != 0: print("Train Time: {:5f}\tTest Accuracy: {:5f}\t\tETA: {:5f}s\t{}/{}".format(duration,acc,duration/count*(len(self.combinations)-count),count,len(self.combinations)))
        print("Search Complete\nBest params: {}\tHighest accuracy: {:5f}".format(str(self.best_params),highest))
model = GBDT(verbose=2,args={'min_samples_leaf': 2})
model.fit(X_train,y_train,cv=1,analyze=True)
#change cv to > 1 for cross validation
vocab = transformer.vectorizer1.vocabulary_
words = [192, 880, 336, 108, 332, 504, 254, 505, 89, 459, 444, 703, 789, 376, 886, 566, 846, 401, 426, 449]
for idx in words:
    print(list(vocab)[list(vocab.values()).index(idx)],end="   ")
model.evaluate(X_test,y_test)
model.evaluate(X_train,y_train,return_score=False)
X_train.year.mean(), X_test.year.mean()
f,axs = plt.subplots(1,2,figsize=(30,10))
axs = axs.flatten()
axs[0].hist(X_train.year,bins=X_train.year.nunique())
axs[0].set_title("Train Set")
axs[1].hist(X_test.year,bins=X_test.year.nunique())
axs[1].set_title("Test Set")
plt.show()
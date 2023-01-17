# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.tokenize import sent_tokenize, word_tokenize

import re

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
recipes = pd.read_json("../input/full_format_recipes.json")
print('before removing duplicates and missing data:', len(recipes))

recipes.drop_duplicates('title', keep='first', inplace=True)

#pd.options.mode.use_inf_as_na = True

recipes.dropna(subset=['categories', 'directions', 'ingredients', 'title'], inplace=True)

#print(recipes.isna(subset=['ingredients']))

recipes.reset_index(drop = True, inplace = True)

print('after removing duplicates and missing data:', len(recipes))
#pre-process categories

recipes['categories'] = recipes['categories'].apply(lambda x: [i.lower() for i in x])

# recipes['categories'] = recipes['categories'].apply(lambda x: re.sub(r"[\'\[\]]|\bname\b", '', str(x)))

# recipes['categories'] = recipes['categories'].apply(lambda x: re.sub("[^\w]", " ", x).split())
#pre-process ingredients

ingredients = ['almond','amaretto','anchovy','anise','apple juice','apple','apricot','artichoke','arugula','asian pear','asparagus','avocado','bacon','banana','barley','basil','bass','beef rib', 'beef shank','beef tenderloin','beef','beet','bell pepper','blackberry','blue cheese','blueberry','bok choy','bourbon','brandy','bread','breadcrumbs','brie','brisket','broccoli rabe','broccoli','brown rice', 'brussel sprout','buffalo','bulgur','burrito','butter','buttermilk','butternut squash','butterscotch/caramel','cabbage','calvados','campari','cantaloupe','capers','caraway','cardamom','carrot','cashew','cauliflower','caviar','celery','chambord','champagne','chard','chartreuse','cheddar','cherry','chestnut','chicken','chickpea','chile pepper','chili','chive','chocolate','cilantro','cinnamon','citrus','clam','coconut','cod','coffee','cognac/armagnac','collard greens','coriander','corn','cornmeal','cottage cheese','couscous','crab','cranberry sauce','cranberry','cream cheese','créme de cacao','crêpe','cucumber','cumin','currant','curry', 'date', 'dill', 'dried fruit', 'duck', 'eau de vie', 'egg nog', 'egg', 'eggplant', 'endive', 'escarole',  'fennel', 'feta', 'fig', 'fontina', 'frangelico', 'garlic', 'gin', 'ginger', 'goat cheese', 'goose', 'gouda', 'grand marnier', 'granola', 'grape', 'grapefruit', 'grappa', 'green bean', 'green onion/scallion', 'ground beef', 'ground lamb', 'guava', 'halibut', 'ham', 'hamburger', 'hazelnut', 'hominy/cornmeal/masa', 'honey', 'honeydew', 'horseradish', 'hot pepper', 'hummus', 'iced coffee', 'iced tea', 'jalapeño', 'jerusalem artichoke', 'jícama', 'kahlúa', 'kale', 'kirsch', 'kiwi', 'kumquat', 'lamb chop', 'lamb shank', 'lamb', 'leek', 'lemon juice', 'lemon', 'lemongrass', 'lentil', 'lettuce', 'lima bean', 'lime juice', 'lime', 'lingonberry', 'lobster', 'lychee', 'macadamia nut', 'mango', 'maple syrup', 'marsala', 'marscarpone', 'marshmallow', 'mayonnaise', 'mezcal', 'midori', 'mint', 'molasses', 'monterey jack', 'mozzarella', 'mushroom', 'mussel', 'mustard greens', 'nectarine', 'nutmeg', 'oat', 'oatmeal', 'octopus', 'okra', 'olive', 'onion', 'orange juice', 'orange', 'oregano', 'orzo', 'oyster', 'papaya', 'paprika', 'parmesan', 'parsley', 'parsnip', 'passion fruit', 'pea', 'peach', 'peanut butter', 'peanut', 'pear', 'pecan', 'pepper', 'pernod', 'persimmon', 'pickles', 'pine nut', 'pineapple', 'pistachio', 'plantain', 'plum', 'poblano', 'pomegranate juice', 'pomegranate', 'poppy', 'pork chop', 'pork rib', 'pork tenderloin', 'pork', 'port', 'potato', 'prosciutto', 'prune', 'pumpkin', 'quail', 'quince', 'quinoa', 'rabbit', 'rack of lamb', 'radicchio', 'radish', 'raisin', 'raspberry', 'red wine', 'rhubarb', 'rice', 'ricotta', 'rosemary', 'rosé', 'rum', 'rutabaga', 'rye', 'saffron', 'sage', 'sake', 'salmon', 'sardine', 'scallop', 'scotch', 'semolina', 'sesame oil', 'sesame', 'shallot', 'sherry', 'shrimp', 'snapper', 'sorbet', 'sour cream', 'sourdough', 'soy sauce', 'spinach', 'squash', 'squid', 'strawberry', 'sugar', 'snap pea', 'sweet potato/yam', 'swiss cheese', 'swordfish', 'tamarind', 'tangerine', 'tapioca', 'tarragon', 'tea', 'tequila', 'thyme', 'tilapia', 'tofu', 'tomatillo', 'tomato', 'triple sec', 'trout', 'tuna', 'turkey', 'turnip', 'vanilla','veal','venison','vermouth','vodka','walnut','wasabi','watercress','watermelon','whiskey','white wine','wild rice','yellow squash','yogurt','yuca','zucchini','bean','berry','bitters','bran','cheese','fish','fortified wine','fruit juice','fruit', 'game','grains','herb','jam or jelly','leafy green','legume','liqueur','milk/cream','meat','melon','muffin','mustard','noodle','nut','phyllo/puff pastry dough','potato salad','poultry sausage','poultry','root vegetable','sausage','seafood','seed','shellfish','spice','steak','tropical fruit','vegetable','vinegar','whole wheat','wine']



def ingredientTag(strList):

    ingredientList = []

    for string in strList:

        string = re.sub(r"[^a-zA-Z]+", ' ', string)

        string = re.sub(r"(teaspoon|tablespoon|cup|ounce|oz|pound|lb|pin|quart|gallon|inch)(s*)", '', string)

        string = string.lower()

        words = word_tokenize(string)

        tags = []

        prevStem = None

        for word in words:

            stem = WordNetLemmatizer().lemmatize(word)

            for ingredient in ingredients:

                result = re.search('(^|/)'+ stem +'(/|$)',ingredient)

                if result:

                    tags.append(ingredient)

                    break

                elif prevStem is not None:

                    if re.search('(^|\s)'+ prevStem + ' ' + stem +'(\s|$)',ingredient):

                        tags.append(ingredient)

                        break

            prevStem = stem

        for tag in tags:

            if tag not in ingredientList:

                ingredientList.append(tag)

    return ingredientList



recipes['tagged ING'] = recipes['ingredients'].apply(ingredientTag)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



maskString = 'winter'



ing_mask = recipes.categories.apply(lambda x: maskString in x)

recipes[maskString] = ing_mask



features = np.zeros((len(recipes), len(ingredients)))



feature_selector = SelectKBest(chi2, k = 10)

selected_ingredients = feature_selector.fit_transform(features, categoryList)



feature_mask = feature_selector.get_support()

for b in range(len(feature_mask)):

    if(feature_mask[b] == True):

        print(ingredients[b])
def logReg(maskString):

    ing_mask = recipes.categories.apply(lambda x: maskString in x)

    recipes[maskString] = ing_mask



    features = np.zeros((len(recipes), len(ingredients)))



    #print(recipes['tagged ING'][0])

    for index in range(len(recipes)):

        ingList = recipes['tagged ING'][index]

        for ing in ingList:

            ingInd = ingredients.index(ing)

            if ingInd:

                features[index][ingInd] = 1



    categoryList = recipes[maskString].values



    # Split into training and testing sets 70/30

    from sklearn.model_selection import train_test_split

    from sklearn.linear_model import LogisticRegression

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import roc_curve, auc

    from matplotlib import pyplot as plt



    X, Y = features, categoryList

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)



    name = "Logistic Regression"

    line_fmt = "-"

    model =  LogisticRegression(solver=('lbfgs'))

    model.fit(X_train, Y_train.ravel())

    preds = model.predict_proba(X_test)

    pred = pd.Series(preds[:,1])

    fpr, tpr, thresholds = roc_curve(Y_test, pred)

    auc_score = auc(fpr, tpr)

    label='%s: auc=%f' % (name, auc_score)

    plt.plot(fpr, tpr, line_fmt, linewidth=5, label=label)



    lr_coef = model.coef_ # added: get coeff for inspection

    lr_pred = pred



#     name = "Random Forest"

#     line_fmt = ":"

#     model =  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

#     model.fit(X_train, Y_train)

#     preds = model.predict_proba(X_test)

#     pred = pd.Series(preds[:,1])

#     fpr, tpr, thresholds = roc_curve(Y_test, pred)

#     auc_score = auc(fpr, tpr)

#     label='%s: auc=%f' % (name, auc_score)

#     plt.plot(fpr, tpr, line_fmt, linewidth=5, label=label)



#     rf_feat = model.feature_importances_ # added: get coeff for inspection

#     rf_pred = pred



    plt.legend(loc="lower right")

    plt.title(maskString + ' classifier')



    plt.plot([0, 1], [0, 1], 'k--') #x=y line.  Visual aid

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.show()
def kHolds(maskString):

    ing_mask = recipes.categories.apply(lambda x: maskString in x)

    recipes[maskString] = ing_mask



    features = np.zeros((len(recipes), len(ingredients)))



    #print(recipes['tagged ING'][0])

    for index in range(len(recipes)):

        ingList = recipes['tagged ING'][index]

        for ing in ingList:

            ingInd = ingredients.index(ing)

            if ingInd:

                features[index][ingInd] = 1



    categoryList = recipes[maskString].values



    # Split into training and testing sets 70/30

    from sklearn.model_selection import train_test_split

    from sklearn.model_selection import StratifiedKFold

    from sklearn.linear_model import LogisticRegression

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import roc_curve, auc

    from matplotlib import pyplot as plt

    

    X, Y = features, categoryList

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)

    

    skf = StratifiedKFold(n_splits=4)

    skf.get_n_splits(X, Y)

    

    for train_index, test_index in skf.split(X,Y):

        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = X[train_index], X[test_index]

        Y_train, Y_test = Y[train_index], Y[test_index]



        name = "Logistic Regression"

        line_fmt = "-"

        model =  LogisticRegression(solver=('lbfgs'))

        model.fit(X_train, Y_train.ravel())

        preds = model.predict_proba(X_test)

        pred = pd.Series(preds[:,1])

        fpr, tpr, thresholds = roc_curve(Y_test, pred)

        auc_score = auc(fpr, tpr)

        label='%s: auc=%f' % (name, auc_score)

        plt.plot(fpr, tpr, line_fmt, linewidth=5, label=label)



        lr_coef = model.coef_ # added: get coeff for inspection

        lr_pred = pred



    plt.legend(loc="lower right")

    plt.title(maskString + ' classifier')



    plt.plot([0, 1], [0, 1], 'k--') #x=y line.  Visual aid

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.show()
maskList = ['dessert', 'pasta', 'sandwich', 'bake', 'winter', 'dinner', 'healthy' ]



for maskString in maskList:

    logReg(maskString)

    kHolds(maskString)
thresh = pd.DataFrame({'truth':Y_test, 'lr_pred':lr_pred})

thresh.sort_values(by=['lr_pred'], ascending=False)
concat = []

fdist = nltk.FreqDist()

for index1, categories in enumerate(recipes['categories']):

    #print(categories)

    if type(categories) is not list:

        print(index1)

        pass

    else:

        for index2, words in enumerate(categories):

            fdist[words] += 1



common = fdist.most_common()

print(common[0:100])
# Split into training and testing sets 70/30

from sklearn.model_selection import train_test_split

X, Y = features, categoryList

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)



# Classifier

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver='liblinear').fit(X_train, Y_train.ravel())

classifier.predict(X_test)

classifier.score(X_test, Y_test)
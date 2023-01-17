import json
import numpy as np
with open('../input/train.json') as file:
    training_data = json.load(file)
with open('../input/test.json') as file:
    testing_data = json.load(file)
import re
def cleanup_ingredient(ingredient):
    x = re.sub('(\'s)|(\’s)','',ingredient)
    x = re.sub('[0-9%\(\)\/\.™\,®\'\&]|(lb\.)|(oz\.)','',x)
    x = re.sub('\-',' ', x)
    x = re.sub(' +',' ', x)
#     x = re.sub('(low fat)|(full fat)|(reduced fat)|(fat free)|(skimmed)|(fatfree)|(lowfat)|(nonfat)|(non fat)|(low sodium)|(reduced sodium)|(less sodium)|(no salt added)|(homemade)|(gluten free)|(salt free)|(reduc sodium)|(cholesterol free)|(s real)|(light)|(free range)|(shredded)|(low moisture)|(skim)|(part )','',x)
    x = re.sub('(low fat )|(full fat)|(reduced fat )|(fat free )|(skimmed )|(fatfree )|(lowfat )|(nonfat )|(non fat )|(low sodium)|(reduced sodium )|(less sodium )|(no salt added)|(homemade)|(gluten free )|(salt free )|(reduc sodium )|(cholesterol free )|(s real)|(light)|(free range)|(skim)','',x)
    x = re.sub('(small)|(medium)|(large)','', x)
    x = re.sub('(heinz)|(hellmannâ€)|(hellmanns)|(hellmann)|(kikkoman)|(kraft)|(taco bell)|(mccormick)|(mcintosh)|(knorr)|(or best food)|(best food)', '', x)
#     x = re.sub('(mozarella)', 'mozzarella', x)
#     x = re.sub('(cheese)', 'chees', x)
    x = re.sub('[éèê]', 'e', x)
    x = re.sub('[íî]', 'i', x)
    x = re.sub('[â]', 'a', x)
    x = re.sub(' +',' ', x)
    x = x.strip()
#     x = re.sub(' ', '_', x) # force ingredients to be separate words
    return x
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
training_recipes = [" ".join([cleanup_ingredient(ingredient) for ingredient in recipe['ingredients']]) for recipe in training_data]
testing_recipes = [" ".join([cleanup_ingredient(ingredient) for ingredient in recipe['ingredients']]) for recipe in testing_data]
# training_recipes = [" ".join([ingredient for ingredient in recipe['ingredients']]) for recipe in training_data]
# testing_recipes = [" ".join([ingredient for ingredient in recipe['ingredients']]) for recipe in testing_data]
# ingredient_encoder = CountVectorizer(binary=True)
ingredient_encoder = TfidfVectorizer(binary=True)
X_train = ingredient_encoder.fit_transform(training_recipes)
X_test = ingredient_encoder.transform(testing_recipes)
cuisine_encoder = LabelEncoder()
y_train = cuisine_encoder.fit_transform([recipe['cuisine'] for recipe in training_data])
from sklearn.model_selection import train_test_split
X_train, X_devtest, y_train, y_devtest = train_test_split(X_train, y_train, test_size=0.4)
X_devtest, X_testest, y_devtest, y_testest = train_test_split(X_devtest, y_devtest, test_size=0.5)
X_train = X_train.astype('float16')
X_devtest = X_devtest.astype('float16')
X_testest = X_testest.astype('float16') 
from tqdm import tqdm
import matplotlib.pyplot as plt
%matplotlib inline

degree = 3 #unoptimised
gamma = 1.4

C = 13.0 

best_C = C
best_gamma = gamma
best_degree = degree
best_score = -1
best_model = None

hist = {'C':[], 'acc': [], 'gamma':[], 'degree':[]}

svm = SVC(C=C, kernel='rbf', gamma=gamma, degree=degree, max_iter=-1)
svm = OneVsRestClassifier(svm, n_jobs=4)
svm.fit(X_train, y_train)
best_model = svm
best_score = svm.score(X_devtest, y_devtest)
# for gamma in tqdm([1.4, 1.5, 1.6, 1.7, 1.8]):
#     svm = SVC(C=C, kernel='rbf', gamma=gamma, degree=degree, max_iter=-1)
#     svm = OneVsRestClassifier(svm, n_jobs=4)
#     svm.fit(X_train, y_train)
#     acc = svm.score(X_devtest, y_devtest)
#     hist['C'].append(C)
#     hist['gamma'].append(gamma)
#     hist['acc'].append(acc)
#     hist['degree'].append(degree)
#     if acc > best_score:
#         best_score = acc
#         best_gamma = gamma
#         best_degree = degree
#         best_C = C
#         best_model = svm
# plt.scatter(hist['gamma'], hist['acc'])
print("{} {} {}".format(best_C, best_gamma, best_score))
print ("Train accuracy {}".format(best_model.score(X_train, y_train)))
print ("Cross validation accuracy {}".format(best_model.score(X_devtest, y_devtest)))
print ("Holdout test accuracy {}".format(best_model.score(X_testest, y_testest)))
y_test = best_model.predict(X_test)
y_pred = cuisine_encoder.inverse_transform(y_test)
ids = [recipe['id'] for recipe in testing_data]
with open('submission.csv','w') as file:
    file.write('id,cuisine\n')
    for id_,cuisine in zip(ids, y_pred):
        file.write("{},{}\n".format(id_, cuisine))

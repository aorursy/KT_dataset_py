import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



data = pd.read_csv("../input/movie_metadata.csv")
print(data.shape)

clean_data = data.dropna(axis = 0)

print(clean_data.shape)
plt.hist(clean_data['imdb_score'], bins=25)

plt.title("Distribution of IMDB score")

plt.show()
x_list = ['movie_facebook_likes','director_facebook_likes','cast_total_facebook_likes',

          'actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes','duration',

          'num_critic_for_reviews','num_voted_users','num_user_for_reviews','budget','gross']



plt.figure(figsize=(7,10))

for i in range(len(x_list)):

    plt.subplot(6,2,i+1)

    plt.title(x_list[i])

    plt.hist(clean_data[x_list[i]],bins=50)

    plt.grid(True)

plt.tight_layout()

plt.show()
count = pd.DataFrame({'zero count':[0]*len(x_list)},index = x_list)

for element in x_list:

    count.ix[element,'zero count'] = sum(np.array(clean_data[element])==0)

print(count)
if "director_facebook_likes" in x_list: x_list.remove("director_facebook_likes")

if "movie_facebook_likes" in x_list: x_list.remove("movie_facebook_likes")

    

from sklearn.cross_validation import train_test_split



x_train, x_test, y_train, y_test = train_test_split(clean_data.ix[:,x_list], clean_data['imdb_score'], 

                                                    test_size=0.25, random_state=0)



x_corr = np.corrcoef(x_train,rowvar = 0)

eigvl, eigvt = np.linalg.eig(x_corr)

print(eigvl)



plt.imshow(x_corr, interpolation='nearest', cmap=plt.cm.Blues, extent=(0,10,0,10))

plt.colorbar()

plt.show()


min_eigen = pd.DataFrame({'min eigen':[0]*len(x_list)},index = x_list)

for element in x_list:

    x_temp = x_train.drop(element,1)

    x_corr = np.corrcoef(x_temp,rowvar = 0)

    eigvl, eigvt = np.linalg.eig(x_corr)

    min_eigen.ix[element,'min eigen'] = min(eigvl)

print(min_eigen)
from sklearn.decomposition import PCA



pca = PCA(n_components = 10)

pca.fit(x_train)

print(np.cumsum(pca.explained_variance_ratio_))
pca = PCA(n_components = 2)

pca.fit(x_train)



x_train_new = pca.transform(x_train)



from sklearn import linear_model

import pylab

import scipy.stats as stats



def plot_model(x,y,model):

    print("R^2: %f" % mod.score(x,y))

    

    y_fitted = model.predict(x)

    residual = y - y_fitted



    plt.figure(figsize=(7,5))



    plt.subplot(221)

    plt.scatter(y_fitted, y)

    plt.title("fitted value vs actual value")

    plt.xlabel("fitted value")

    plt.ylabel("actual value")



    plt.subplot(222)

    plt.hist(residual, bins=50)

    plt.title("residual histogram")





    plt.subplot(223)

    stats.probplot(residual, dist="norm", plot=pylab)





    plt.subplot(224)

    plt.scatter(y_fitted,residual)

    plt.title("fitted value vs residual")

    plt.xlabel("fitted value")

    plt.ylabel("residual")



    plt.tight_layout()

    plt.show()

    

    

mod = linear_model.LinearRegression()

mod.fit(x_train_new, y_train)

plot_model(x_train_new,y_train,mod)





if "cast_total_facebook_likes" in x_list: x_list.remove("cast_total_facebook_likes")



x_train, x_test, y_train, y_test = train_test_split(clean_data.ix[:,x_list], clean_data['imdb_score'], 

                                                    test_size=0.25, random_state=0)



mod = linear_model.LinearRegression()

mod.fit(x_train, y_train)



plot_model(x_train,y_train,mod)
from sklearn.feature_extraction.text import CountVectorizer



def token(text):

    return(text.split("|"))



cv_kw=CountVectorizer(max_features=50,tokenizer=token )

keywords = cv_kw.fit_transform(clean_data["plot_keywords"])

keywords_list = ["kw_" + i for i in cv_kw.get_feature_names()]



cv_ge=CountVectorizer(tokenizer=token )

genres = cv_ge.fit_transform(clean_data["genres"])

genres_list = ["genres_"+ i for i in cv_ge.get_feature_names()]



new_clean_data = np.hstack([clean_data.ix[:,x_list],keywords.todense(),genres.todense()])

new_coeff_list = x_list+keywords_list+genres_list



x_train, x_test, y_train, y_test = train_test_split(new_clean_data, clean_data['imdb_score'], 

                                                    test_size=0.25, random_state=0)
mod = linear_model.LinearRegression()

mod.fit(x_train, y_train)

plot_model(x_train,y_train,mod)
y_test_fitted = mod.predict(x_test)

plt.scatter(y_test_fitted, y_test)

plt.title("predicted value vs actual value")

plt.xlabel("predicted value")

plt.ylabel("actual value")

plt.show()



print("Score: %f" % mod.score(x_test,y_test))

print("SSE: %f" % sum((y_test_fitted-y_test)**2))
model_aic = linear_model.LassoLarsIC(criterion='aic')

model_aic.fit(x_train, y_train)



def plot_ic_criterion(model, name, color):

    alpha_ = model.alpha_

    alphas_ = model.alphas_

    criterion_ = model.criterion_

    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,

             linewidth=3, label='%s criterion' % name)

    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,

                label='alpha: %s estimate' % name)

    plt.xlabel('-log(alpha)')

    plt.ylabel('criterion')

    plt.show()



plot_ic_criterion(model_aic, 'AIC', 'b')

print(model_aic.alpha_)
mod = linear_model.Lasso(alpha = model_aic.alpha_)

mod.fit(x_train, y_train)

plot_model(x_train,y_train,mod)
y_test_fitted = mod.predict(x_test)



plt.scatter(y_test_fitted, y_test)

plt.title("predicted value vs actual value")

plt.xlabel("predicted value")

plt.ylabel("actual value")

plt.show()



print("Score: %f" % mod.score(x_test,y_test))

print("SSE: %f" % sum((y_test_fitted-y_test)**2))
coeff_value = pd.DataFrame(list(mod.coef_),index = new_coeff_list)

print(sum(coeff_value[0]==0))
imdb_score = np.array(clean_data['imdb_score'])

percent25 = np.percentile(imdb_score,33)

percent75 = np.percentile(imdb_score,67)



clean_list = (imdb_score>percent75) + (imdb_score<percent25)

classifier_clean_data = new_clean_data[clean_list]

classifier_coeff_list = new_coeff_list



imdb_level = list(clean_data['imdb_score'][clean_list]>percent75)

imdb_level = [int(i) for i in imdb_level]



x_train, x_test, y_train, y_test = train_test_split(classifier_clean_data, imdb_level, 

                                                    test_size=0.25, random_state=0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
from sklearn.model_selection import cross_val_score



avg_score_list = []

for i in range(1,20):

    mod = DecisionTreeClassifier(max_depth = i)

    scores = cross_val_score(mod, x_train, y_train, cv=20)

    avg_score_list.append(np.mean(scores))

    

plt.plot(range(1,20),avg_score_list,'--',linewidth=3)

plt.axvline(avg_score_list.index(max(avg_score_list))+1, linewidth=3)

plt.show()



print("max score reached with depth %d" % (avg_score_list.index(max(avg_score_list))+1))
def plot_test(x,y,model):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, [i[1] for i in model.predict_proba(x)])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.plot(false_positive_rate, true_positive_rate, "b", label='AUC %0.2f' % (roc_auc))

    plt.title("AUC Curve")

    plt.show()



    auc_score = roc_auc_score(y, [i[1] for i in mod.predict_proba(x)])

    cm = confusion_matrix(y,mod.predict(x))

    plot_confusion_matrix(cm,classes = ["bad movie","good movie"],normalize=False)

    print("AUC Score: %f" % auc_score)

    print("Accuracy: %f" % (sum(mod.predict(x) == y)/float(len(y))))





mod = DecisionTreeClassifier(max_depth = 7)

mod.fit(x_train,y_train)

plot_test(x_test,y_test,mod)
from sklearn.ensemble import RandomForestClassifier



avg_score_matrix = []

for i in range(1,16):

    temp_score = []

    for j in range(1,11):

        mod = RandomForestClassifier(n_estimators=j*10,max_depth = i)

        scores = cross_val_score(mod, x_train, y_train, cv=10)

        temp_score.append(np.mean(scores))

    avg_score_matrix.append(temp_score)
plt.imshow(np.matrix(avg_score_matrix), interpolation='nearest', cmap=plt.cm.Blues, aspect='auto', extent=(0,100,15,0))

plt.ylabel("depth")

plt.xlabel("tree number")

plt.colorbar()

plt.show()
score_matrix = np.matrix(avg_score_matrix)

i,j = np.unravel_index(score_matrix.argmax(), score_matrix.shape)

print(i,j)
mod = RandomForestClassifier(n_estimators=60,max_depth = 15)

mod.fit(x_train,y_train)

plot_test(x_test,y_test,mod)
from sklearn.ensemble import GradientBoostingClassifier



avg_score_matrix = []

for i in range(1,6):

    temp_score = []

    for j in range(1,11):

        mod = GradientBoostingClassifier(n_estimators=j*20, learning_rate=1.0, max_depth = i, random_state=0)

        scores = cross_val_score(mod, x_train, y_train, cv=10)

        temp_score.append(np.mean(scores))

    avg_score_matrix.append(temp_score)
plt.imshow(np.matrix(avg_score_matrix), interpolation='nearest', cmap=plt.cm.Blues, aspect='auto', extent=(0,200,5,0))

plt.ylabel("depth")

plt.xlabel("tree number")

plt.colorbar()

plt.show()
score_matrix = np.matrix(avg_score_matrix)

i,j = np.unravel_index(score_matrix.argmax(), score_matrix.shape)

print(i,j)
mod = GradientBoostingClassifier(n_estimators=80, learning_rate=1.0, max_depth=1, random_state=0)

mod.fit(x_train,y_train)

plot_test(x_test,y_test,mod)
from sklearn.ensemble import AdaBoostClassifier



avg_score_matrix = []

for i in range(1,2):

    temp_score = []

    for j in range(1,11):

        mod = AdaBoostClassifier(n_estimators=j*20)

        scores = cross_val_score(mod, x_train, y_train, cv=10)

        temp_score.append(np.mean(scores))

    avg_score_matrix.append(temp_score)
avg_score_list = []

for i in range(1,11):

    mod = AdaBoostClassifier(n_estimators=i*20)

    scores = cross_val_score(mod, x_train, y_train, cv=10)

    avg_score_list.append(np.mean(scores))

    

plt.plot(range(1,11),avg_score_list,'--',linewidth=3)

plt.axvline(avg_score_list.index(max(avg_score_list))+1, linewidth=3)

plt.show()



print("max score reached with depth %d" % (avg_score_list.index(max(avg_score_list))+1))
mod = AdaBoostClassifier(n_estimators=120)

mod.fit(x_train,y_train)

plot_test(x_test,y_test,mod)
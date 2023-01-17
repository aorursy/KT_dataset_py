import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# ML models
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
folder = "../input/"
df_train = pd.read_json(folder + "train.json")
df_train.head()
# how many different ingredients there are?
ingredient_set = set([])
for ingredients in df_train['ingredients']:
    for ingredient in ingredients:
        ingredient_set.add(ingredient)
N = len(ingredient_set)
print(N)
ingredient_list = (list(ingredient_set))
def mk_binary(ingredients):
    datapoint = np.zeros(shape=(1,N))
    indices = [ingredient_list.index(ingredient) for ingredient in ingredients]
    datapoint[:, indices] = 1
    return datapoint
# prepare our X dataset
X = np.zeros(shape=(df_train.shape[0], N))
for i in range(df_train.shape[0]):
    X[i,:] = mk_binary(df_train["ingredients"][i])
# which cuisines are in the dataset
different_cuisines = list(df_train["cuisine"].unique())
different_cuisines
# top 3 ingredient of cuisine X
def top_n(cuisine_name, n=3):
    X_part = X[df_train["cuisine"] == cuisine_name]
    sorted_ingredient_indices = np.argsort(X_part.sum(axis=0))
    return [ingredient_list[int(i)] for i in sorted_ingredient_indices[-n:]]
for cuisine in different_cuisines:
    print("{} cuisine mostly uses {}".format(cuisine, top_n(cuisine, n=3)))
def similarity(cuisine_one, cuisine_two):
    most_one = top_n(cuisine_one, n=100)
    most_two = top_n(cuisine_two, n=100)
    return len(set(most_one).intersection(most_two))
k = len(different_cuisines)
similarity_scores = np.zeros(shape=(k,k), dtype=np.int)
for i in range(k):
    for j in range(k):
        similarity_scores[i,j] = similarity(different_cuisines[i], different_cuisines[j])
plt.figure(figsize=(8,8))
plt.imshow(np.array(similarity_scores) / 100, cmap="YlOrRd")
plt.colorbar()
plt.tick_params(bottom=False, labelbottom=False, labeltop=True, left=False)
_ = plt.xticks(list(range(len(different_cuisines))), different_cuisines, rotation="vertical")
_ = plt.yticks(list(range(len(different_cuisines))), different_cuisines, rotation="horizontal")
# let us find most similar for each cuisine
for i in range(k):
    cuisine_i = different_cuisines[i]
    j = np.argsort(similarity_scores[i])[-2]
    cuisine_j = different_cuisines[j]
    print("{} :: {}".format(cuisine_i, cuisine_j))
# prepare labels dataset
y = df_train["cuisine"].apply(lambda c : different_cuisines.index(c)).values
# split data
X_train,X_test, y_train, y_test = train_test_split(X,y)
# man is international fam, check the statistics
num_examples = []
for i in range(len(different_cuisines)):
    num_examples.append((y==i).sum())
plt.style.use("seaborn-white")
plt.rcParams["font.size"] = 15
plt.figure(figsize=(13,15))
plt.box()
bars = plt.barh(y=list(range(k)), 
                width=num_examples, 
                tick_label=different_cuisines,
                color="coral")
for bar, count in zip(bars, num_examples):
    y_coor = bar.get_y()
    plt.text(y=y_coor+0.25, x=count*0.40, 
             s="{0:.0f}%".format(count/y.shape[0]*100), 
             color="lemonchiffon", fontsize=16)
plt.xticks([])
_ = plt.title("Distribution of cuisines in dataset")
p = PCA(n_components=500)
X_train_pca = p.fit_transform(X_train)
# explained variance
exp_var = p.explained_variance_ratio_[:500]
exp_var_cum = np.cumsum(exp_var)
plt.style.use("ggplot")
plt.figure()
plt.plot(list(range(len(exp_var))), exp_var, "r+")
plt.plot(list(range(len(exp_var))), exp_var_cum, "g*")
gnb = GaussianNB().fit(X_train_pca, y_train)
X_test_pca = p.transform(X_test)
gnb.score(X_test_pca, y_test)

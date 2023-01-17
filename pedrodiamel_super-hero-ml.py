# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # visualization tool

import matplotlib.pyplot as plt

%matplotlib inline

import missingno as msno

sns.set()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

data_hero = pd.read_csv('../input/super_hero_powers.csv')

data_hero.head()
target_counts = data_hero.drop(["hero_names"],axis=1).sum(axis=0).sort_values(ascending=False)

plt.figure(figsize=(15,25))

sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)

plt.show()
# correlation mat 

data_hero_corr = data_hero.drop(["hero_names"], axis=1).corr()

plt.figure(figsize=(15,15))

sns.heatmap( data_hero_corr , cmap="RdYlBu", vmin=-1, vmax=1)

plt.show()
# top > 0.65 correlation super powers 

powers = data_hero.drop(['hero_names'], axis=1)

for p in powers.columns:

    top5 = data_hero_corr[p].abs().drop(p).nlargest(5)

    top5 = top5[top5>0.65]

    if top5.empty:

        continue

    for i,v in zip(top5.index, top5.values): 

        c = data_hero_corr[p][i]

        print('{:30}\t| {:30}\t| {:<30}\t| {:<30}'.format(p, i, v, c) )

# powerful (sum_{super powers}) by super heroe

data_hero_plus_power = data_hero

data_hero_plus_power["Powerful"] = data_hero_plus_power.drop(["hero_names"],axis=1).sum(axis=1)

data_hero_sort = data_hero_plus_power.sort_values(by=['Powerful'], ascending=False)

hero_names = data_hero_sort['hero_names'][:100]

powerful   = data_hero_sort['Powerful'][:100]



plt.figure(figsize=(15,20))

sns.barplot(y=hero_names, x=powerful) 

plt.show()

data_info = pd.read_csv('../input/heroes_information.csv', index_col='Unnamed: 0',na_values='-')

data_info.head()
data_info.dtypes
msno.matrix(data_info)

plt.show()
missing_data = data_info.isnull().sum().sort_values(ascending=False)

plt.figure(figsize=(8,8))

sns.barplot(y=missing_data.index.values, x=missing_data.values, order=missing_data.index)

plt.show()
from sklearn.manifold import TSNE

cmap = plt.get_cmap('jet_r')



data_hero = pd.read_csv('../input/super_hero_powers.csv', na_values='-')

Z = TSNE(n_components=2, init='pca', 

    random_state=0, perplexity=30).fit_transform( data_hero.drop(['hero_names'], axis=1))



plt.figure(figsize=(12,12))

plt.scatter(Z[:,0], Z[:,1], s=(20,20), marker='o', color=[0,0,1] );

plt.show()
# Aux function 



import unicodedata

import re



# Turn a Unicode string to plain ASCII, thunicodedata

# http://stackoverflow.com/a/518232/280942unicodedata

def unicodeToAscii(s):

    return ''.join(

        c for c in unicodedata.normalize('NFD', s)

        if unicodedata.category(c) != 'Mn'

    )



# Lowercase, trim, and remove non-letter characters

def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())

    s = re.sub(r"([.!?])", r" \1", s)

    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s



# Filter string features 

def filterFeature( X, strpref='' ):

    for i,x in enumerate(X):

        if X.isnull()[i]: 

            continue

        X[i] = strpref + normalizeString(x)   

    return X



# Sensitive cases errors of the color name   

def filterCaseColor(s):

    if s == 'yellow without irises ': s ='yellow'

    if s == 'no hair': s='none'

    if s == 'strawberry blond': s='blond'

    if s == 'bown': s='brown'

    if s == 'brownn': s='brown'

    return s



# Filter color (union of the colors space)

def filterColor(colors):

    mapcolor={ c:filterCaseColor(normalizeString(c)).split()  for c in colors }

    colors=[]

    for k,v in mapcolor.items():

        colors.append(v)

    colors = np.unique(np.concatenate( colors, axis=0 ))

    colors = { c:i for i,c in enumerate(colors)  }    

    return mapcolor, colors



# Vector to one hot representation 

def v2hot(v,n):

    hot = np.zeros(n); hot[v]=1

    return hot



def color2OneHot( colors, mapcolor, strpref='' ):

    colors_hot = [[] for _ in colors ]

    for i,c in enumerate(colors):

        if colors.isnull()[i]:continue

        colors_hot[i] = mapcolor[c]

        

    colors_uq = np.unique(np.concatenate(colors_hot,axis=0))

    n = len(colors_uq)

    colors_map = { c:i for i,c in enumerate(colors_uq)  } 

    colors_hot = [ v2hot([ colors_map[c] for c in cs ], n) for cs in colors_hot   ]

    colors_hot = np.stack(colors_hot, axis=0)

    colors_uq = np.array([ strpref + c for c in colors_uq ] )

    colors_df = pd.DataFrame( data=colors_hot, columns=colors_uq )

    return colors_df



def norm(x):

    x = (x - np.mean(x))/np.std(x)

    return x



# visualization 

def barplot( df, fsize=(8,8) ):

    df_sum = df.sum( axis=0 ).sort_values(ascending=False)

    plt.figure(figsize=fsize)

    sns.barplot(y=df_sum.index.values, x=df_sum.values, order=df_sum.index)

    plt.show()



info_colors = ['Eye color', 'Hair color', 'Skin color' ]

colors = [ data_info[info_color][~data_info[info_color].isnull()].unique() for info_color in info_colors]

colors = np.concatenate(colors,axis=0)

mapcolor, colors = filterColor( colors )

print(colors)
eye_colors = data_info['Eye color']

eye_colors_df = color2OneHot(eye_colors, mapcolor)

print(eye_colors_df.head())

barplot(eye_colors_df)
hair_colors = data_info['Hair color']

hair_colors_df = color2OneHot(hair_colors, mapcolor, 'hair_')

print(hair_colors_df.head())
barplot(hair_colors_df)
race = data_info['Race'].copy()

race = filterFeature( race, 'race_' ) 

race[race.isnull()] = 'race_no_human'

race_df = pd.get_dummies(race)

race_df.head()
barplot(race_df, (8,14))
publisher = data_info['Publisher'].copy()

publisher = filterFeature( publisher, 'publisher_' ) 

publisher[publisher.isnull()] = 'publisher_none'

publisher_df = pd.get_dummies(publisher)

publisher_df.head()
barplot(publisher_df)
alignment = data_info['Alignment'].copy()

alignment = filterFeature( alignment, 'alig_' ) 

alignment[alignment.isnull()] = 'alig_neutral'

alignment_df = pd.get_dummies(alignment)

alignment_df.head()
barplot(alignment_df)
data_info['Height'].describe()
height = data_info['Height'].copy()

height[data_info['Height'].isnull()] = height[~data_info['Height'].isnull()].mean()

height_df =  pd.DataFrame( data=norm(height))



plt.figure( figsize=(14,6))

plt.subplot(121)

plt.hist( norm(height), bins=50,  density=True, facecolor='g', alpha=0.75)

plt.subplot(122)

plt.boxplot( norm(height))

plt.show()

data_info['Weight'].describe()
weight = data_info['Weight'].copy()

weight[data_info['Weight'].isnull()] = 0

weight_df =  pd.DataFrame( data=norm(weight))



plt.figure( figsize=(14,6))

plt.subplot(121)

plt.hist( norm(weight), bins=50,  density=True, facecolor='g', alpha=0.75)

plt.subplot(122)

plt.boxplot( norm(weight))

plt.show()

# preprocessing pipeline of the data information 

def prepocessing( data_info ):

    data_info_prep = data_info.copy()



    # color prepo

    info_colors = ['Eye color', 'Hair color', 'Skin color' ]

    colors = [ data_info[info_color][~data_info[info_color].isnull()].unique() for info_color in info_colors]

    colors = np.concatenate(colors,axis=0)

    mapcolor, colors = filterColor( colors )



    # Eye color analysis

    eye_colors = data_info['Eye color'].copy()

    eye_colors_df = color2OneHot(eye_colors, mapcolor, 'eye_')



    # Hair color analysis

    hair_colors = data_info['Hair color'].copy()

    hair_colors_df = color2OneHot(hair_colors, mapcolor, 'hair_')



    # Race analysis

    race = data_info['Race'].copy()

    race = filterFeature( race, 'race_' ) 

    race[race.isnull()] = 'race_no_human'

    race_df = pd.get_dummies(race)



    # Publisher anaysis 

    publisher = data_info['Publisher'].copy()

    publisher = filterFeature( publisher, 'pub_' ) 

    publisher[publisher.isnull()] = 'pub_none'

    publisher_df = pd.get_dummies(publisher)



    # Alignmant analysis 

    alignment = data_info['Alignment'].copy()

    alignment = filterFeature( alignment, 'alig_' ) 

    alignment[alignment.isnull()] = 'alig_neutral'

    alignment_df = pd.get_dummies(alignment)



    # Height

    height = data_info['Height'].copy()

    height[data_info['Height'].isnull()] = 0

    height_df =  pd.DataFrame( data=norm(height))



    # Weight

    weight = data_info['Weight'].copy()

    weight[data_info['Weight'].isnull()] = 0

    weight_df =  pd.DataFrame( data=norm(weight))

    

    # Gender

    gender = data_info['Gender'].copy()

    gender_df = (gender=='Male')*1

    gender_df =  pd.DataFrame( data=gender_df )

        

    # Create dataframe 

    # eye_colors_df, hair_colors_df, race_df, publisher_df, alignment_df, height_df, weight_df, gender_df

    return pd.concat([

        eye_colors_df,

        hair_colors_df,

        race_df, 

        publisher_df, 

        alignment_df, 

        height_df, 

        weight_df, 

        gender_df

    ],  axis=1, sort=False)



# load data and preprocessing 

data_info = pd.read_csv('../input/heroes_information.csv', index_col='Unnamed: 0',na_values='-')

new_data_info = prepocessing( data_info )

new_data_info.head()

from sklearn.manifold import TSNE

Z = TSNE(n_components=2, init='pca', 

    random_state=0, perplexity=30).fit_transform( new_data_info )



plt.figure(figsize=(12,12))

plt.scatter(Z[:,0], Z[:,1], s=(20,20), marker='o', color=[1,0,0] );

plt.show()


# load data and preprocessing 

data_info = pd.read_csv('../input/heroes_information.csv', index_col='Unnamed: 0',na_values='-')

data_hero = pd.read_csv('../input/super_hero_powers.csv', na_values='-')



# preprocessiong

new_data_info = prepocessing( data_info )



# union of information 

name_info = filterFeature(data_info['name'].copy())

name_hero = filterFeature(data_hero['hero_names'].copy())



tuplas=[]

for i in range(len(name_hero)):

    for j in range(len(name_info)):

        if name_hero[i] == name_info[j]:

            tuplas.append( pd.concat([new_data_info.iloc[j], data_hero.drop(['hero_names'], axis=1).iloc[i]], axis=0, sort=False) )



# data_processes = pd.DataFrame( data=tuplas )

data_processes = pd.concat(tuplas, axis=1, sort=False).T



data_processes.to_csv( './data_processes.csv' , index=False, encoding='utf-8')

data_processes.head()

Z = TSNE(n_components=2, init='pca', 

    random_state=0, perplexity=30).fit_transform( data_processes )



plt.figure(figsize=(12,12))

plt.scatter(Z[:,0], Z[:,1], s=(20,20), marker='o', color=[1,0,1] );

plt.show()
from scipy.special import logsumexp



class BernoulliMixture:    

    def __init__(self, n_components, max_iter, tol=1e-3):

        self.n_components = n_components

        self.max_iter = max_iter

        self.tol = tol

    

    def fit(self,x):

        self.x = x

        self.init_params()

        log_bernoullis = self.get_log_bernoullis(self.x)

        self.old_logL = self.get_log_likelihood(log_bernoullis)

        for step in range(self.max_iter):

            if step > 0:

                self.old_logL = self.logL            

            # E-Step

            self.gamma = self.get_responsibilities(log_bernoullis)

            self.remember_params()

            # M-Step

            self.get_Neff()

            self.get_mu()

            self.get_pi()

            # Compute new log_likelihood:

            log_bernoullis = self.get_log_bernoullis(self.x)

            self.logL = self.get_log_likelihood(log_bernoullis)            

            if np.isnan(self.logL):

                self.reset_params()

                print(self.logL)

                break



    def reset_params(self):

        self.mu = self.old_mu.copy()

        self.pi = self.old_pi.copy()

        self.gamma = self.old_gamma.copy()

        self.get_Neff()

        log_bernoullis = self.get_log_bernoullis(self.x)

        self.logL = self.get_log_likelihood(log_bernoullis)

        

    def remember_params(self):

        self.old_mu = self.mu.copy()

        self.old_pi = self.pi.copy()

        self.old_gamma = self.gamma.copy()

    

    def init_params(self):

        self.n_samples = self.x.shape[0]

        self.n_features = self.x.shape[1]

        #self.gamma = np.zeros(shape=(self.n_samples, self.n_components))

        self.pi = 1/self.n_components * np.ones(self.n_components)

        self.mu = np.random.RandomState(seed=0).uniform(low=0.25, high=0.75, size=(self.n_components, self.n_features))

        self.normalize_mu()

    

    def normalize_mu(self):

        sum_over_features = np.sum(self.mu, axis=1)

        for k in range(self.n_components):

            self.mu[k,:] /= sum_over_features[k]

            

    def get_responsibilities(self, log_bernoullis):

        gamma = np.zeros(shape=(log_bernoullis.shape[0], self.n_components))

        Z =  logsumexp(np.log(self.pi[None,:]) + log_bernoullis, axis=1)

        for k in range(self.n_components):

            gamma[:, k] = np.exp(np.log(self.pi[k]) + log_bernoullis[:,k] - Z)

        return gamma

        

    def get_log_bernoullis(self, x):

        log_bernoullis = self.get_save_single(x, self.mu)

        log_bernoullis += self.get_save_single(1-x, 1-self.mu)

        return log_bernoullis

    

    def get_save_single(self, x, mu):

        mu_place = np.where(np.max(mu, axis=0) <= 1e-15, 1e-15, mu)

        return np.tensordot(x, np.log(mu_place), (1,1))

        

    def get_Neff(self):

        self.Neff = np.sum(self.gamma, axis=0)

    

    def get_mu(self):

        self.mu = np.einsum('ik,id -> kd', self.gamma, self.x) / self.Neff[:,None] 

        

    def get_pi(self):

        self.pi = self.Neff / self.n_samples

    

    def predict(self, x):

        log_bernoullis = self.get_log_bernoullis(x)

        gamma = self.get_responsibilities(log_bernoullis)

        return np.argmax(gamma, axis=1)

        

    def get_sample_log_likelihood(self, log_bernoullis):

        return logsumexp(np.log( self.pi[None,:] ) + log_bernoullis, axis=1)

    

    def get_log_likelihood(self, log_bernoullis):

        return np.mean(self.get_sample_log_likelihood(log_bernoullis))

        

    def score(self, x):

        log_bernoullis = self.get_log_bernoullis(x)

        return self.get_log_likelihood(log_bernoullis)

    

    def score_samples(self, x):

        log_bernoullis = self.get_log_bernoullis(x)

        return self.get_sample_log_likelihood(log_bernoullis)
from sklearn.model_selection import train_test_split



X = pd.read_csv('../input/super_hero_powers.csv', na_values='-').drop( 'hero_names', axis=1 )*1

X = np.array(X, dtype=np.int)

clusters = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50] 

x_train, x_test = train_test_split(X, shuffle=True, random_state=0)

scores = []

for n in range(len(clusters)):

    model = BernoulliMixture(clusters[n], 200) 

    model.fit(x_train)

    score = model.score(x_test)

    scores.append(score)

    
est_clusters = clusters[np.argmin(scores)]

plt.figure()

plt.plot(clusters, scores)

plt.plot(est_clusters, scores[np.argmin(scores)],'or')

plt.title('Elbow method')

plt.ylabel('J(x)')

plt.xlabel('Number of clusters')

plt.show()
model = BernoulliMixture(est_clusters, 200)

model.fit(X)
results = pd.read_csv('../input/super_hero_powers.csv', na_values='-').drop( 'hero_names', axis=1 )*1

results["cluster"] = np.argmax(model.gamma, axis=1)

G = results.groupby("cluster").sum() / results.drop("cluster", axis=1).sum(axis=0) * 100

G = G.apply(np.round).astype(np.int32)



plt.figure(figsize=(40,5))

sns.heatmap(G, cmap="Oranges", annot=True, fmt="g", cbar=False, annot_kws={"size": 6});

plt.title("How are specific super power over clusters in percent?");
for g in np.array(G):

    gi = np.argsort(g)[::-1]

    #top = np.where(np.array(g)>75)[0]

    name_list = np.array(results.columns[ gi ][:3] )

    print('{}/{}/{}'.format(name_list[0],name_list[1],name_list[2] ))
from sklearn.manifold import TSNE

cmap = plt.get_cmap('jet_r')



X = pd.read_csv('../input/super_hero_powers.csv', na_values='-').drop( 'hero_names', axis=1 )*1

Y = np.argmax(model.gamma, axis=1)

nameG = [

    'Qwardian Power Ring/Speed Force/Power Cosmic',

    'Spatial Awareness/Anti-Gravity/Hyperkinesis',

    'Echolocation/Wallcrawling/Web Creation',

    'Intuitive aptitude/Hair Manipulation/Illumination',

    'Vision - Cryo/Vision - Microscopic/Vision - Heat',

    'Terrakinesis/Weather Control/Water Control',

    'Omniscient/Banish/Astral Travel',

    'The Force/Cloaking/Mind Control Resistance',

    'Toxin and Disease Resistance/Elasticity/Magic Resistance',

    'Changing Armor/Photographic Reflexes/Peak Human Condition'

        ]



Z = TSNE(n_components=2, init='pca', 

    random_state=0, perplexity=50).fit_transform(X)



#show

plt.figure( figsize=(12,12) )

#plt.scatter(Xt[:,0], Xt[:,1], s=(10,10), marker='o', c=Yo);

n = len(np.unique(Y))

for i in range( n ):

    index = Y==i

    color = cmap(float(i)/n) 

    plt.scatter(Z[index,0], Z[index,1], s=(20,20), marker='o', color=color, label='{}'.format( nameG[i] ) ); #dataloader.dataset.data.classes[i]



plt.legend()

plt.show()

#load prep dataset

data_prep = pd.read_csv('./data_processes.csv')

data_prep.head()


from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import sklearn.metrics as metrics



X = np.array(data_prep.drop( ['alig_bad', 'alig_good', 'alig_neutral'], axis=1 ).copy())

Y = np.array(data_prep[ ['alig_bad', 'alig_good', 'alig_neutral'] ].copy())



X = X[Y[:,2]==0,:]              #delete neutral in x

Y = Y[Y[:,2]==0,:2]             #delete neutral in y

Y = (Y[:,0] + Y[:,1]*2) - 1     #hot2val



# Leave-One-Out cross-validator

# X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, shuffle=True, random_state=0)



# 10Fold Cross Validation

result=[]

k=10

kf = StratifiedKFold(n_splits=k)

kf.get_n_splits(X, Y)



clf = GaussianNB()

# clf = BernoulliNB()

# clf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=123456)

print('KFold\t|Acc\t|Prec\t|Rec\t|F1\t|')

for i,(train_index, test_index) in enumerate(kf.split(X,Y)):

    

    X_train = X[train_index,:]; X_test = X[test_index,:]

    y_train = Y[train_index  ]; y_test = Y[test_index  ]

        

    # Estimate

    clf.fit(X_train,y_train)

    # Predict

    y_test_hat = clf.predict(X_test)



    # Evaluate

    acc = metrics.accuracy_score(y_test, y_test_hat)

    precision = metrics.precision_score(y_test, y_test_hat, average='macro')

    recall = metrics.recall_score(y_test, y_test_hat, average='macro')

    f1_score = 2*precision*recall/(precision+recall)



    result.append([acc,precision,recall,f1_score])  

    print( 'K({}):\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|'.format(i,acc,precision,recall,f1_score).replace('.',',')  )

result_mat = np.stack(result, axis=1).T

meanNB = result_mat.mean(axis=0)

stdNB  = result_mat.std(axis=0)



plt.figure(figsize=(12,5))

ind = np.arange(4) 

plt.bar(ind, meanNB, 0.35, yerr=stdNB, color='red')

plt.ylabel('Scores')

plt.title('Scores by group and gender')

plt.xticks(ind, ('Acc', 'Prec', 'Rec', 'F1'))

plt.show()
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import sklearn.metrics as metrics



X = np.array(data_prep.drop( ['alig_bad', 'alig_good', 'alig_neutral'], axis=1 ).copy())

Y = np.array(data_prep[ ['alig_bad', 'alig_good', 'alig_neutral'] ].copy())



X = X[Y[:,2]==0,:]              #delete neutral in x

Y = Y[Y[:,2]==0,:2]             #delete neutral in y

Y = (Y[:,0] + Y[:,1]*2) - 1     #hot2val



# Leave-One-Out cross-validator

# X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, shuffle=True, random_state=0)



# 10Fold Cross Validation



k=10

kf = StratifiedKFold(n_splits=k)

kf.get_n_splits(X, Y)



all_result = []

names_methods = ['GaussianNB', 'BernoulliNB', 'RandomForestClassifier']

# clf = GaussianNB()

# clf = BernoulliNB()

# clf = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=123456)



for i,clf in enumerate([GaussianNB(), BernoulliNB(), RandomForestClassifier(n_estimators=300, oob_score=True, random_state=123456)]):



    result=[]

    print(names_methods[i])

    print('---'*20)

    print('KFold\t|Acc\t|Prec\t|Rec\t|F1\t|')

    for i,(train_index, test_index) in enumerate(kf.split(X,Y)):



        X_train = X[train_index,:]; X_test = X[test_index,:]

        y_train = Y[train_index  ]; y_test = Y[test_index  ]



        # Estimate

        clf.fit(X_train,y_train)

        # Predict

        y_test_hat = clf.predict(X_test)



        # Evaluate

        acc = metrics.accuracy_score(y_test, y_test_hat)

        precision = metrics.precision_score(y_test, y_test_hat, average='macro')

        recall = metrics.recall_score(y_test, y_test_hat, average='macro')

        f1_score = 2*precision*recall/(precision+recall)



        result.append([acc,precision,recall,f1_score])  

        print( 'K({}):\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|{:0.3f}\t|'.format(i,acc,precision,recall,f1_score).replace('.',',')  )

    print('---'*20)

    

    all_result.append( np.stack(result, axis=1).T )

    print(' ')

    

    



Means=[]

Stds=[]

for mat in all_result:

    Means.append( mat.mean(axis=0) )

    Stds.append( mat.std(axis=0) )



    

plt.figure(figsize=(12,5))

ind = np.arange(4) 

width = 0.35 



p1 = plt.bar(ind - width/3, Means[0], 0.35, yerr=Stds[0], color='SkyBlue')

p2 = plt.bar(ind, Means[1], 0.35, yerr=Stds[1], color='IndianRed')

p3 = plt.bar(ind + width/3, Means[2], 0.35, yerr=Stds[2], color='green')



plt.ylabel('Scores')

plt.title('Scores by group and gender')

plt.xticks(ind, ('Acc', 'Prec', 'Rec', 'F1'))

plt.legend((p1[0], p2[0], p3[0]), names_methods)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

import sklearn.metrics as metrics



# X = np.array(data_prep.drop( ['Weight'], axis=1 ).copy())

X = np.array(data_prep[ ['Height', 'Gender'] ].copy())

Y = np.array(data_prep[ ['Weight'] ].copy())



# Leave-One-Out cross-validator

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=0)



# Import the model we are using

from sklearn.preprocessing import PolynomialFeatures

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model.stochastic_gradient import SGDRegressor



# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

sg = SGDRegressor(penalty='elasticnet', alpha=0.01, l1_ratio=0.25, fit_intercept=True, tol=1e-4)



# Train the model on training data

poly = PolynomialFeatures(degree=2)

rf.fit(poly.fit_transform(X_train), y_train);

sg.fit(poly.fit_transform(X_train), y_train);

# Use the forest's predict method on the test data

predictions_rf = rf.predict(poly.fit_transform(X_test))

predictions_sg = sg.predict(poly.fit_transform(X_test))



# Calculate the absolute errors

errors_rf = abs(predictions_rf - y_test)

errors_sg = abs(predictions_sg - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error of RandomForestRegressor:', round(np.mean(errors_rf), 2), 'kg')

print('Mean Absolute Error of SGDRegressor:', round(np.mean(errors_sg), 2), 'kg')



    
colors = ['teal', 'yellowgreen', 'gold']

x_plot = np.arange(len(y_test))

lw = 2



plt.figure(figsize=(22,5))

plt.plot(x_plot, y_test, color='cornflowerblue', linewidth=lw, label="Actual")

plt.plot(x_plot, predictions_rf[:,np.newaxis], color='gold', linewidth=lw, label="Random Forest")

plt.plot(x_plot, predictions_sg[:,np.newaxis], color='teal', linewidth=lw, label="SGD")

plt.scatter(x_plot, y_test, color='navy', s=30, marker='o', label="Points")

plt.legend()

plt.show()
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import subplots, show

df = pd.read_csv('../input/add.csv',low_memory=False)
df.head(10)
df = df.applymap(lambda val: np.nan if str(val).strip() == '?' else val)
df = df.dropna()
from sklearn.preprocessing import StandardScaler

# Define a standard scaler
sc = StandardScaler()

# Remove the first column, it's useless
data = df.iloc[:,1:].reset_index(drop=True)

# Factorization 
data.loc[data['1558'] == 'ad.', '1558'] = 1
data.loc[data['1558'] == 'nonad.', '1558'] = 0

# Scale features and extract targets
x = data.iloc[:,:-1]
x = pd.DataFrame(sc.fit_transform(x), index=x.index, columns=x.columns)
y = data.iloc[:, -1]

n_components = 250;
pca = PCA(n_components=n_components)
pca.fit(x)
PCA(copy=True, iterated_power='auto', n_components=n_components, random_state=None, svd_solver='auto', tol=0.0, whiten=False)

# variance explained 
fig, ax = subplots()
plt.plot( pca.explained_variance_ratio_*100)
ax.set_xlabel("#Component")
ax.set_ylabel("Explained variance ratio")
show()

# cumulative variance explained
fig, ax = subplots()
plt.plot( pca.explained_variance_ratio_.cumsum()*100)
ax.set_xlabel("#Component")
ax.set_ylabel("Cumulative explained variance ratio")

show()
idx_ad = y[y==1].index # ad indexes
idx_nonad = y[y==0].index # non ad indexes

xs_ad = pca.transform(x)[idx_ad,0] # scores 1st component (ads)
ys_ad = pca.transform(x)[idx_ad,1] # scores 2nd component (ads)
xs_nad = pca.transform(x)[idx_nonad,0] # scores 1st component (non ads)
ys_nad = pca.transform(x)[idx_nonad,1] # scores 2nd component (non ads)

d = pd.DataFrame({'x':[],'y':[],'type':[]})
d=d.append(pd.DataFrame({'x':xs_ad, 'y':ys_ad, 'type' : 'ad.'}))
d=d.append(pd.DataFrame({'x':xs_nad, 'y':ys_nad, 'type' : 'nonad.'}))
d = d.reset_index(drop=True);

# scatterplot 
g = sns.lmplot('x', 'y', data=d, hue='type', fit_reg=False)
g.set(xlabel='1st component', ylabel='2st component')

plt.show()
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# Split predictors and targets
y = df.iloc[:, -1]
x = df.iloc[:,1:-1]
from sklearn import linear_model

pres_rec = pd.DataFrame( columns=['precision','recall','c','out']);
acc = pd.DataFrame( columns=['accuracy','c']);
kf = KFold(n_splits=5, random_state=True, shuffle=True)


C = [0.1,1,100,300,500,700];
for c in C:
    for train_index, test_index in kf.split(x):

        xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
        ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
        logreg = linear_model.LogisticRegression(C=c)
        logreg.fit(xtrain, ytrain)
        predicted = logreg.predict(xtest)
        cm = confusion_matrix(ytest,predicted)

        # precision
        ad_precision = cm[0][0] / ( cm[0][0] + cm[1][0] )
        nonad_precision = cm[1][1] / ( cm[0][1] + cm[1][1] )

        #recall
        ad_recall = cm[0][0] / ( cm[0][0] + cm[0][1] )
        nonad_recall = cm[1][1] / ( cm[1][0] + cm[1][1] )
        
        #accuracy
        accuracy = (cm[0][0] + cm[1][1]) / ( cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]);

        pres_rec = pres_rec.append([{'precision': ad_precision, 'recall' : ad_recall, 'c' : c , 'out' : 'ad'}]);
        pres_rec = pres_rec.append([{'precision': nonad_precision, 'recall' : nonad_recall, 'c' : c , 'out' : 'nonad'}]);
        acc = acc.append([{'accuracy' : accuracy, 'c' : c}])
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(11,5))

g = sns.factorplot(x="c", y="accuracy", data=acc,capsize=.2, ax = ax1)
plt.close(g.fig)

g = sns.factorplot(x="c", y="precision", hue="out", data=pres_rec,capsize=.2, palette="YlGnBu_d", ax = ax2)
plt.close(g.fig)

g = sns.factorplot(x="c", y="recall", hue="out", data=pres_rec,capsize=.2, palette="YlGnBu_d", ax = ax3)
plt.close(g.fig)

fig.tight_layout()
plt.show()
from sklearn.svm import SVC

pres_rec = pd.DataFrame( columns=['precision','recall','c','kernel','out']);
acc = pd.DataFrame( columns=['accuracy','c', 'kernel']);
kf = KFold(n_splits=3, random_state=True, shuffle=True)

C = [0.1,1,50,100]
kernels = ['linear','poly','rbf','sigmoid']

for kernel in kernels:
    for c in C:
        for train_index, test_index in kf.split(x):

            xtrain, xtest = x.iloc[train_index], x.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            clf = SVC(kernel=kernel, C=c)
            clf.fit(xtrain, ytrain)
            predicted = clf.predict(xtest)
            cm = confusion_matrix(ytest,predicted)

            # precision
            ad_precision = cm[0][0] / ( cm[0][0] + cm[1][0] )
            nonad_precision = cm[1][1] / ( cm[0][1] + cm[1][1] )

            #recall
            ad_recall = cm[0][0] / ( cm[0][0] + cm[0][1] )
            nonad_recall = cm[1][1] / ( cm[1][0] + cm[1][1] )

            #accuracy
            accuracy = (cm[0][0] + cm[1][1]) / ( cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0]);

            pres_rec = pres_rec.append([{'precision': ad_precision, 'recall' : ad_recall, 'c' : c , 'kernel' : kernel, 'out' : 'ad'}]);
            pres_rec = pres_rec.append([{'precision': nonad_precision, 'recall' : nonad_recall, 'c' : c , 'kernel' : kernel, 'out' : 'nonad'}]);
            acc = acc.append([{'accuracy' : accuracy, 'c' : c, 'kernel' : kernel}])
g = sns.factorplot(x="c", y="accuracy", col="kernel", data=acc, capsize=.2, size=4)
plt.show()
g = sns.factorplot(x="c", y="precision", hue="out", col="kernel", data=pres_rec, capsize=.2, palette="YlGnBu_d", size=4)
g.despine(left=True)
plt.show()

g = sns.factorplot(x="c", y="recall", hue="out", col="kernel", data=pres_rec, capsize=.2, palette="YlGnBu_d", size=4)
g.despine(left=True)
plt.show()
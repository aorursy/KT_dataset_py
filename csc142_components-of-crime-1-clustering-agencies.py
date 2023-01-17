# must for data analysis

% matplotlib inline

import numpy as np

import pandas as pd



# plots

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import *



# useful for data wrangling

import io, os, re, subprocess



# for sanity

from pprint import pprint
# learn you some machines

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
data_files = os.listdir('../input/')

pprint(data_files)
def ca_law_enforcement_by_agency(data_directory):

    filename = 'ca_law_enforcement_by_agency.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        content = f.read()



    content = re.sub('\r',' ',content)

    [header,data] = content.split("civilians\"")

    header += "civilians\""

    

    data = data.strip()

    agencies = re.findall('\w+ Agencies', data)

    all_but_agencies = re.split('\w+ Agencies',data)

    del all_but_agencies[0]

    

    newlines = []

    for (a,aba) in zip(agencies,all_but_agencies):

        newlines.append(''.join([a,aba]))

    

    # Combine into one long string, and do more processing

    one_string = '\n'.join(newlines)

    sio = io.StringIO(one_string)

    

    # Process column names

    columnstr = header.strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    columns = [s.strip() for s in columns]



    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',names=columns,thousands=',')



    return df



def ca_offenses_by_agency(data_directory):

    filename = 'ca_offenses_by_agency.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        lines = f.readlines()

    

    one_line = '\n'.join(lines[1:])

    sio = io.StringIO(one_line)

    

    # Process column names

    columnstr = lines[0].strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    columns = [s.strip() for s in columns]

    

    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',names=columns,thousands=',')



    return df



df1 = ca_law_enforcement_by_agency('../input/')

df1.head()



df2 = ca_offenses_by_agency('../input/')

df2.head()



df = pd.merge(df1,df2)
print(df.shape)

print(df.head(2))
# We should note that the columns

# "violent crime" and "property crime" 

# are sums of other columns.



col1 = df['Violent crime']

col2 = (df['Murder and nonnegligent manslaughter']+df['Rape (revised definition)']+df['Robbery']+df['Aggravated assault'])



print("Columns col1 (violent crime) and col2 (sum of violent types of crime) are identical.")

print((col2-col1)[:10])
# This column does not have data

try:

    del df['Rape (legacy definition)']

except KeyError:

    pass



df = df.replace(np.nan,0.0)



for col in df.columns.tolist():

    print("Number of NaNs in column %s is %d"%(col, df[col].isnull().sum() ))
pca_cols = df.columns.tolist()[3:]



X_orig = df[pca_cols].values
def get_normed_mean_cov(X):

    X_std = StandardScaler().fit_transform(X)

    X_mean = np.mean(X_std, axis=0)

    

    ## Automatic:

    #X_cov = np.cov(X_std.T)

    

    # Manual:

    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0]-1)

    

    return X_std, X_mean, X_cov



X_std, X_mean, X_cov = get_normed_mean_cov(X_orig)
xlabels = pca_cols

xlabels = [re.sub("Murder and nonnegligent manslaughter","Murder, Manslaughter",j) for j in xlabels]

xlabels = [re.sub("Total law enforcement employees","Tot law enf empl",j) for j in xlabels]
fig = plt.figure(figsize=(6,6))

sns.heatmap(pd.DataFrame(X_cov), 

            xticklabels=xlabels, yticklabels=xlabels,

            vmin=-1,vmax=1,

            annot=False, square=True, cmap='BrBG')

plt.title('Heatmap of Covariance Matrix Magnitude: Law Enforcement Agency Data', size=14)



plt.show()
eigenvals, eigenvecs = np.linalg.eig(X_cov)



eigenvals = np.abs(eigenvals)

eigenvecs = np.abs(eigenvecs)



# Eigenvalues are not necessarily sorted, but eigenval[i] *does* correspond to eigenvec[i]

#print "Eigenvals shape: "+str(eigenvals.shape)

#print "Eigenvecs shape: "+str(eigenvecs.shape)



# Create a tuple of (eigenvalues, eigenvectors)

unsrt_eigenvalvec = [(eigenvals[i], eigenvecs[:,i]) for i in range(len(eigenvals))]



# Sort tuple by eigenvalues

eigenvalvec = sorted(unsrt_eigenvalvec, reverse=True, key=lambda x:x[0])



## This is noisy, but interesting:

#pprint([pair for pair in eigenvalvec])

## We will visualize this below.


fig = plt.figure(figsize=(6,3))

sns.heatmap(pd.DataFrame([pair[1] for pair in eigenvalvec]), 

            annot=False, cmap='coolwarm',

            xticklabels=xlabels, yticklabels=range(len(eigenvalvec)),

            vmin=-1,vmax=1)



plt.ylabel("Ranked Eigenvalue")

plt.xlabel("Eigenvector Components")

plt.title('Eigenvalue Analysis: Law Enforcement Agency Data', size=14)

plt.show()
lam_sum = sum([j[0] for j in eigenvalvec])

explained_variance = [(lam_k/lam_sum) for lam_k in sorted(eigenvals, reverse=True)]
plt.figure(figsize=(6, 4))



plt.bar(range(len(explained_variance)), explained_variance, 

        alpha=0.5, align='center',

        label='Individual Explained Variance $\lambda_{k}$')



plt.ylabel('Explained variance ratio')

plt.xlabel('Ranked Eigenvalues')

plt.title("Scree Graph: Law Enforcement Agency Data", size=14)



plt.legend(loc='best')

plt.ylim([0,np.max(explained_variance)+0.1])

plt.tight_layout()
fig = plt.figure(figsize=(6,4))

ax1 = fig.add_subplot(111)



ax1.plot(np.cumsum(explained_variance),'o')



ax1.set_ylim([0,1.01])



ax1.set_xlabel('Number of Principal Components')

ax1.set_ylabel('Cumulative explained variance')

ax1.set_title('Explained Variance: Law Enforcement Agency Data', size=14)



plt.show()
print(np.cumsum(explained_variance)[:4])
N_PCA = 4



# 4 components should explain about 90% of the variance.

sklearn_pca = PCA(n_components = N_PCA).fit(X_std)

print(sklearn_pca.components_.shape)
print("Principal Components:")

print(sklearn_pca.components_)
# This requires a weird bar chart label offset.

# 

# xticks() controls where the tick marks are located,

# and parameters like rotation angle of text.

# this is available through pyplot (plt).

#

# xticklabels controls the x tick labels.

# of course, this is NOT available through pyplot.

# you have to have a handle to the axis itself.

# that's why I use gca().set_xticklabels()

# 

# the more sane way would be plt.xticklabels()



colors = [sns.xkcd_rgb[z] for z in ['dusty purple','dusty green','dusty blue','orange']]

for i in range(4):

    fig = plt.figure(figsize=(6,4))

    xstuff = list(range(len(sklearn_pca.components_[i])))

    sns.barplot(xstuff,

                sklearn_pca.components_[i], color=colors[i])

    

    gca().set_xticklabels(xlabels)

    

    plt.xticks(np.arange(len(sklearn_pca.components_[i]))-0.1,rotation=90,size=14)

    plt.ylabel('Principal Component '+str(i+1)+' Value',size=12)

    plt.title('Principal Component '+str(i+1),size=12)

    plt.show()
Z = sklearn_pca.fit_transform(X_std)

print(Z.shape)
fig = plt.figure(figsize=(14,6))

ax1, ax2 = [fig.add_subplot(120 + i + 1) for i in range(2)]







ax1.scatter( Z[:,0], Z[:,1], s=80 )



ax1.set_title('Principal Components 0 and 1\nSubspace Projection', size=14)

ax1.set_xlabel('Principal Component 0')

ax1.set_ylabel('Principal Component 1')







ax2.scatter( Z[:,2], Z[:,3], s=80 )



ax2.set_title('Principal Components 2 and 3\nSubspace Projection', size=14)

ax1.set_xlabel('Principal Component 0')

ax1.set_ylabel('Principal Component 1')



plt.show()
for i in range(4):

    print("Explained Variance, Principal Component %d: %0.4f"%(i,sklearn_pca.explained_variance_[i]/np.sum(sklearn_pca.explained_variance_.sum())))
km = KMeans(n_clusters=6, n_init=4, random_state=False)

km.fit(Z)

print(km.n_clusters)

print(km.predict(Z))
# To color each point by the digit it represents,

# create a color map with N elements (N rgb values).

# One for each cluster.

#

# Then, use the system response (y_training), which conveniently

# is a digit from 0 to 9.

def get_cmap(n):

    colorz = plt.get_cmap('Set1')

    return[ colorz(float(i)/n) for i in range(n)]



colorz = get_cmap( km.n_clusters )

colors = [colorz[j] for j in km.predict(Z)]



fig = plt.figure(figsize=(12,4))

ax1, ax2 = [fig.add_subplot(120 + i + 1) for i in range(2)]



s1 = ax1.scatter( Z[:,0], Z[:,1] , c=colors, s=80 )

ax1.set_title('Principal Components 0 and 1\nSubspace Projection')



s2 = ax2.scatter( Z[:,2], Z[:,3] , c=colors, s=80 )

ax2.set_title('Principal Components 2 and 3\nSubspace Projection')





# ------------

# thanks to matplotlib for legend stupid-ness.

# guess i'll just draw the legend myself.

labels = ["Cluster "+str(j) for j in range(km.n_clusters)]

rs = []

for i in range(len(colorz)):

    p = Rectangle((0,0), 1, 1, fc = colorz[i])

    rs.append(p)

ax1.legend(rs, labels, loc='best')

ax2.legend(rs, labels, loc='best')

# ------------



ax1.set_ylim([-3,7])

ax2.set_ylim([-3,7])



ax1.set_xlabel("Principal Component 0")

ax1.set_ylabel("Principal Component 1")



ax2.set_xlabel("Principal Component 2")

ax2.set_ylabel("Principal Component 3")



plt.show()
# Store the cluster number in a new DataFrame column 

cluster_col = km.predict(Z)

df['Cluster'] = cluster_col
for k in range(km.n_clusters):

    if k!=0:

        print("-"*20)

    print("Cluster %d:"%(k))

    pprint(df['Agency'][df['Cluster']==k].tolist())
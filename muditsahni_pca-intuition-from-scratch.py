#Importing required libraries
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
cmap = sns.color_palette("Blues")
import pandas as pd
import numpy as np
sns.set_style('whitegrid')
#importing in the data
sky = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")

#taking a peek at the data
sky.head()
#Looking at some summary statistics
sky.describe()
#Let's check for any NULL values
sky.isnull().sum()
drop_columns = ['objid','specobjid','camcol','rerun','run','field']
sky.drop(drop_columns, axis=1, inplace=True)

sky.head()
sky['class'].value_counts()
values = sky['class'].value_counts().values
proportion = values/np.sum(values)
classes = ['Galaxies','Stars','Quasars']
for i in range(3):
    print(f"Proportion of {classes[i]}: {round(proportion[i]*100,2)}%")
sns.pairplot(sky)
plt.show()
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(sky)


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(sky[sky['class']=='STAR'][['u', 'g', 'r', 'i', 'z']].corr(), linewidths=0.1, ax = axes[0], \
                 cmap='coolwarm', linecolor='white', annot=True)
ax.set_title('Star')
ax = sns.heatmap(sky[sky['class']=='GALAXY'][['u', 'g', 'r', 'i', 'z']].corr(), linewidths=0.1, ax = axes[1], \
                 cmap='coolwarm', linecolor = 'white', annot=True)
ax.set_title('Galaxy')
ax = sns.heatmap(sky[sky['class']=='QSO'][['u', 'g', 'r', 'i', 'z']].corr(), linewidths=0.1, ax = axes[2], \
                 cmap='coolwarm', linecolor = 'white', annot=True)
ax = ax.set_title('QSO')
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(sky[sky['class']=='STAR'][['mjd','plate']].corr(), linewidths=0.1, ax = axes[0], \
                 cmap='coolwarm', linecolor='white', annot=True)
ax.set_title('Star')
ax = sns.heatmap(sky[sky['class']=='GALAXY'][['mjd','plate']].corr(), linewidths=0.1, ax = axes[1], \
                 cmap='coolwarm', linecolor = 'white', annot=True)
ax.set_title('Galaxy')
ax = sns.heatmap(sky[sky['class']=='QSO'][['mjd','plate']].corr(), linewidths=0.1, ax = axes[2], \
                 cmap='coolwarm', linecolor = 'white', annot=True)
ax = ax.set_title('QSO')
# plt.figure(figsize=(15,7))
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (15,7))
ax1.boxplot(sky[sky['class']=='QSO']['redshift'])
ax1.set_title("Quasars")
ax2.boxplot(sky[sky['class']=='STAR']['redshift'])
ax2.set_title("Stars")
ax3.boxplot(sky[sky['class']=='GALAXY']['redshift'])
ax3.set_title("Galaxy")
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
ax = sns.distplot(sky[sky['class']=='STAR']['redshift'], bins = 30, ax = axes[0], kde = True)
ax.set_title('Star')
ax = sns.distplot(sky[sky['class']=='GALAXY']['redshift'], bins = 30, ax = axes[1], kde = True)
ax.set_title('Galaxy')
ax = sns.distplot(sky[sky['class']=='QSO']['redshift'], bins = 30, ax = axes[2], kde = True)
ax = ax.set_title('Quasar')
plt.show()
#Separating out the dataframe
stars = sky[sky['class'] == "STAR"]
galaxies = sky[sky['class'] == "GALAXY"]
quasars = sky[sky['class'] == "QSO"]

#Extracting the features
all_samples = sky[['u','g','r','i','z']].values

print(f'Shape of Stars: {all_samples.shape}')
means = np.zeros((5,1))
for i in range(5):
    means[i,0] = np.mean(all_samples[:,i], axis=0)

print(f"Shape of means: {means.shape}")
all_samples -= means.T
cov_mat = np.cov([all_samples[:,i] for i in range(5)])
print('Covariance Matrix:\n', cov_mat)
eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
eigen_values
eigen_values_p = eigen_values/np.sum(eigen_values)
print(eigen_values_p)
eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
eigen_pairs.sort(key = lambda x: x[0], reverse=True)
eigen_pairs
eigen_vectors_final = np.hstack((eigen_pairs[0][1].reshape(-1,1), eigen_pairs[1][1].reshape(-1,1), \
                                 eigen_pairs[2][1].reshape(-1,1)))
eigen_vectors_final
ugriz = eigen_vectors_final.T.dot(all_samples.T)
ugriz.T
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
ugriz = pca.fit_transform(sky[['u', 'g', 'r', 'i', 'z']])
ugriz
pca = PCA(n_components=1)
mjdplate = pca.fit_transform(sky[['mjd','plate']])
mjdplate
# update dataframe 
sky_pca = pd.concat((sky, pd.DataFrame(ugriz)), axis=1)
sky_pca.rename({0: 'bands_1', 1: 'bands_2', 2: 'bands_3'}, axis=1, inplace = True)
sky_pca = pd.concat((sky_pca, pd.DataFrame(mjdplate)), axis =1)
sky_pca.rename({0: 'mjdplate_1'}, axis=1, inplace = True)

sky_pca.drop(['u','g','r','i','z'], axis = 1, inplace=True)

#Encoding the class variables to quantitative variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(sky_pca['class'])
sky_pca['class'] = y_encoded

sky_pca.head()
X = sky_pca[['bands_1','bands_2','bands_3','redshift','ra','dec','fiberid','mjdplate_1']].values
y = sky_pca[['class']].values
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scaled = ss.fit_transform(X)
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 420)
from sklearn.linear_model import LogisticRegression

LG = LogisticRegression(penalty="l2")
LG.fit(X_train, y_train)
y_preds = LG.predict(X_test)
accuracy = LG.score(X_test,y_test)

print(f"Accuracy of Logistic Model: {accuracy*100}%")
#Plotting the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
ss, sg, sq, gs, gg, gq, qs, qg, qq = confusion_matrix(y_test,y_preds).ravel()
cm = confusion_matrix(y_test, y_preds)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt="g", cmap="Blues_r"); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Stars', 'Galaxies','Quasars']); 
ax.yaxis.set_ticklabels(['Stars', 'Galaxies','Quasars']);
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test,y_preds, average="micro")
recall = recall_score(y_test, y_preds, average="micro")
f1 = f1_score(y_test, y_preds, average="micro")

print(f"Precision Score: {precision}\n\
Recall Score: {recall}\n\
F1 Score: {f1}")
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LG, X_train, y_train.squeeze(), cv=10, scoring = "accuracy")

print(f"Mean Cross Validation Score: {scores.mean()}\n\
Cross Validation Score Standard Deviation: {scores.std()}")

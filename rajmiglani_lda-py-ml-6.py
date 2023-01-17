import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
df_wine = pd.read_csv('../input/Wine.csv');
df_wine.head()
df_wine.columns = [  'name'
                 ,'alcohol'
             	,'malicAcid'
             	,'ash'
            	,'ashalcalinity'
             	,'magnesium'
            	,'totalPhenols'
             	,'flavanoids'
             	,'nonFlavanoidPhenols'
             	,'proanthocyanins'
            	,'colorIntensity'
             	,'hue'
             	,'od280_od315'
             	,'proline'
                ]
df_wine.head()
#make train-test sets
from sklearn.model_selection import train_test_split;
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values;
#print(np.unique(y))
#split with stratify on y for equal proportion of classes in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, stratify = y,random_state = 0);

#standardize the features with same model on train and test sets
from sklearn.preprocessing import StandardScaler;
sc = StandardScaler();
X_train_std = sc.fit_transform(X_train);
X_test_sd = sc.transform(X_test);
#set precision of the vectors
np.set_printoptions(precision = 4);
mean_vecs = [];

#for each of the label compute the mean vector 
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train == label],axis = 0));
    print('Mean Vector %s: %s\n' %(label, mean_vecs[label - 1]));
#define number of features
d  = 13;
#define the within class scatter matrix of dimension d x d
S_W = np.zeros((d,d));

# run through each class label and keep track of the corresponding mean vector
for label , mv in zip(range(1,4),mean_vecs):
    #define class scatter matrix for each label of dimension d x d 
    class_scatter = np.zeros((d,d));
    
    #run through each row corresponding to a class label and compute the class scatter matrix
    for row in X_train_std[y_train == label]:
        #reshape to vectors of dimension d x 1
        row, mv  = row.reshape(d,1), mv.reshape(d,1);
        #sum for each row d x d dimensional class matrices
        class_scatter += (row - mv).dot((row - mv).T);
    S_W += class_scatter;
# within class scatter matrix of dimension d x d
print("Within Class Scatter Matrix: %s x %s" % (S_W.shape[0], S_W.shape[1]));
print('Class label distribution: %s' % np.bincount(y_train)[1:])
S_W = np.zeros((d,d));
for label, mv in zip(range(1,4),mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T);
    S_W += class_scatter;
print('Scaled Within Class Scatter Matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]));
#calculate the overall mean vector
mean_overall = np.mean(X_train_std,axis = 0);
#define Between Class Scatter Matrix of dimension d x d
S_B = np.zeros((d,d));
for i, mean_vec in enumerate(mean_vecs):
    #find number of samples for each class
    n = X_train[y_train == i + 1].shape[0];
    mean_vec = mean_vec.reshape(d,1);
    mean_overall = mean_overall.reshape(d,1);
    #find the scatter matrix using the above equation
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T);
print('Between Class Scatter Matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]));
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B));
eigen_pairs  = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))];
eigen_pairs = sorted(eigen_pairs, key = lambda k: k[0], reverse = True);
print('Eigenvalues in descending order: \n');
for eigen_val in eigen_pairs:
    print(eigen_val[0]);
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real,reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center',label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()
#transformation matrix of dimension d x k i.e. 13 x 2 here
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)
# Xnew = Xorig.W
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers): plt.scatter(X_train_lda[y_train==l, 0],X_train_lda[y_train==l, 1] * (-1),c=c, label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.show()
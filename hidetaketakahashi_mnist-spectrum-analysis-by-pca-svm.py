import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", dtype = "int16")

test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", dtype = "int16")

sample_submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv", dtype = "int16")

train_y = train_df.iloc[:,0]

all_df = np.concatenate((train_df.values[:,1:], test_df.values))

n_train = train_y.shape[0]

n_test = test_df.shape[0]
n_train, n_test
Cov_mat = np.cov(all_df, rowvar = False, bias = True)
variances = np.sort(np.diagonal(Cov_mat))[::-1]

variances_pca = np.sort(np.linalg.eigvalsh(Cov_mat))[::-1]
cumsum_var = np.cumsum(variances)

cumsum_var_pca =  np.cumsum(variances_pca)
fig, ax = plt.subplots(figsize = (8,5))

ax = plt.plot(cumsum_var/cumsum_var[-1], label = "original", lw = 2)

ax = plt.plot(cumsum_var_pca/cumsum_var_pca[-1], label = "pca", lw = 2)

plt.legend(fontsize=14)

plt.ylabel("Fraction of total variance",fontsize=14)

plt.xlabel("Dimensions",fontsize=14)

plt.show()
ori_90_percent = np.where(cumsum_var/cumsum_var[-1] > 0.9)[0][0]

pca_90_percent = np.where(cumsum_var_pca/cumsum_var_pca[-1] > 0.9)[0][0]

ori_95_percent = np.where(cumsum_var/cumsum_var[-1] > 0.95)[0][0]

pca_95_percent = np.where(cumsum_var_pca/cumsum_var_pca[-1] > 0.95)[0][0]

print("Before PCA, 90% of total variances is in", ori_90_percent, "dimensions")

print(" After PCA, 90% of total variances is in", pca_90_percent, "dimensions")

print("Before PCA, 95% of total variances is in", ori_95_percent, "dimensions")

print(" After PCA, 95% of total variances is in", pca_95_percent, "dimensions")
eigenvalues, eigenvectors = np.linalg.eigh(Cov_mat)
fix1, ax1 = plt.subplots(3,8, figsize = (16,7))

for i in range(0,24):

    y = i % 8

    x = int(i / 8)

    ax1[x, y].imshow(np.reshape(eigenvectors[:,-(i+1)], (28,28)), cmap = "gray")

    ax1[x, y].set_title("mode" + str((i+1)))  

    ax1[x, y].set_xticks([], [])

    ax1[x, y].set_yticks([], [])

plt.show()

def spectrum_plot_multiple(value):

    sample = np.random.choice(np.where(train_y == value)[0],3)

    range_x = 20

    fig3, ax3 = plt.subplots(3,2, figsize = (15,16))

    for i in range(3):

        

        ax3[i,0].imshow(np.reshape(all_df[sample[i],:],(28,28)), cmap = "gray")

        ax3[i,0].set_title("Original Image", fontsize=14)

        ax3[i,1].bar(range(1,range_x+1),np.flip(np.dot(all_df[sample[i],:], eigenvectors)[-range_x:]))

        ax3[i,1].set_xlabel("Modes", fontsize=14)

        ax3[i,1].set_ylabel("Amplitude", fontsize=14)

    

    plt.show()    
spectrum_plot_multiple(value = 0)
spectrum_plot_multiple(value = 1)
spectrum_plot_multiple(value = 2)
spectrum_plot_multiple(value = 4)
spectrum_plot_multiple(value = 9)
def pca_projection(mode, sample):

    

    U = eigenvectors[:,-mode:]

    U2 = np.dot(all_df[sample,:], U)

    U.shape, U2.shape, (U2*U).shape

    P = np.sum(U2*U, axis = 1)

    return np.reshape(P,(28,28))



def plot_projections(sample):

    PR1 = pca_projection(mode=86, sample = sample)

    PR2 = pca_projection(mode=153, sample = sample)

    fig2, ax2 = plt.subplots(1,3, figsize=(10,4))

    ax2[0].imshow(PR1, cmap = "gray")

    ax2[0].set_title("PCA 86 dimensions")  

    ax2[1].imshow(PR1, cmap = "gray")

    ax2[1].set_title("PCA 153 dimensions")  

    ax2[2].imshow(np.reshape(all_df[sample,:], (28,28)), cmap = "gray")

    ax2[2].set_title("Original Image")  

    

    for i in range(3):

        ax2[i].set_xticks([], [])

        ax2[i].set_yticks([], [])

    

    plt.show()
plot_projections(sample = 55)
plot_projections(sample = 1001)
plot_projections(sample = 9000)
all_pca = np.dot(all_df, eigenvectors[:,-86:])
pca_df = pd.DataFrame(np.flip(all_pca[:n_train], axis = 1))

pca_df["label"] = train_y
plt.figure(figsize=(7,7))

sns.scatterplot(data = pca_df.iloc[4000:7000,:], x=0, y=1, hue = "label", palette = 'Set1', alpha = 0.8)
sns.relplot(data = pca_df.iloc[4000:15000,:], x=0, y=1, col = "label", hue = "label", palette = 'Set1', alpha = 0.4, col_wrap = 4)
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
X_train, X_val, y_train, y_val = train_test_split(all_pca[:n_train], train_y.values, test_size = 0.03)
X_train.shape, y_train.shape, X_val.shape, y_val.shape
model_svc = SVC(kernel = "poly", C = 1.0, coef0 = 1.0, degree = 2)
time1 = time.time()

model_svc.fit(X_train, y_train)

time2 = time.time()

print("Training by SVM (quadratic kernel) took only", int(time2 - time1), "sec" )
pred_train = model_svc.predict(X_train)

print("train error = ", np.sum(pred_train != y_train)/y_train.shape[0])
pred_val = model_svc.predict(X_val)

print("validation error = ", np.sum(pred_val != y_val)/y_val.shape[0])
confusion_matrix(y_val, pred_val)
pred_test = model_svc.predict(all_pca[n_train:])
test_submission = sample_submission

test_submission["Label"] = pred_test
test_submission.to_csv('submission.csv',index=False)
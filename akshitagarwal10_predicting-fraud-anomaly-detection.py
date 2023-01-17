#Loading necessary libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import multivariate_normal


#Loading the dataset
df = pd.read_csv("../input/creditcard.csv")
#Exploring the dataset
print("Dataset is of shape: {}".format(df.shape))
print("Fraud cases: {}".format(len(df[df.Class==1])))
print("Normal cases: {}".format(len(df[df.Class==0])))
print("Contamination: {}".format((float(len(df[df.Class==1]))/len(df))*100))
df.describe()
#Sampling the dataset for tsne algorithm
tsne_data_fraud = df[df.Class==1]
tsne_data_normal = df[df.Class==0].sample(frac=0.05, random_state=1)
print(tsne_data_fraud.shape)
print(tsne_data_normal.shape)
#Data engineering for tsne
tsne_data = tsne_data_fraud.append(tsne_data_normal, ignore_index=True)
tsne_data = shuffle(tsne_data)
label = tsne_data.iloc[:, -1]
tsne_data = tsne_data.iloc[:, :30]
tsne_data = tsne_data.astype(np.float64)

standard_scaler = StandardScaler()
tsne_data = standard_scaler.fit_transform(tsne_data)

print(tsne_data.shape)
print(label.shape) 
#Performing dimension reduction (tsne)
tsne = TSNE(n_components=2, random_state=0)
tsne_data = tsne.fit_transform(tsne_data)
#Making final changes to the resulted data from tsne
print(tsne_data.shape)
tsne_plot = np.vstack((tsne_data.T, label))
tsne_plot = tsne_plot.T
print(tsne_plot.shape)
#Plotting the tsne results
tsne_plot = pd.DataFrame(data=tsne_plot, columns=("V1", "V2", "Class"))
sb.FacetGrid(tsne_plot, size=6, hue="Class").map(plt.scatter, "V1", "V2").add_legend()
#Visualizing each feature separately
df.hist(figsize=(20,20), bins=50, color="green", alpha=0.5)
plt.show()
#Creating train, cross-validation and test set
df_fraud = shuffle(df[df.Class==1])
df_normal = shuffle(df[df.Class==0].sample(n=280000))
print(df_fraud.shape)
print(df_normal.shape)
df_train = df_normal.iloc[:240000, :].drop(labels = ["Class", "Time"], axis = 1)
df_cross = shuffle(df_normal.iloc[240000:260000, :].append(df_fraud.iloc[:246, :]))
Y_cross = df_cross.loc[:, "Class"]
df_cross = df_cross.drop(labels = ["Class", "Time"], axis = 1)
df_test = shuffle(df_normal.iloc[260000:, :].append(df_fraud.iloc[246:, :]))
Y_test = df_test.loc[:, "Class"]
df_test = df_test.drop(labels = ["Class", "Time"], axis = 1)
print(df_train.shape)
print(df_cross.shape)
print(Y_cross.shape)
print(df_test.shape)
print(Y_test.shape)

#Defining fuctions to calculate mean, cov and gaussian probablities
def mean_variance(data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    return mean, cov

def gaussian_dist(data, mean, cov):
    prob = multivariate_normal.pdf(data, mean=mean, cov=cov)
    return prob
#Fitting the model for train, cross and test set using mean and cov from train_set
mean, cov = mean_variance(df_train)
print(mean.shape)
print(cov.shape)
prob_train = gaussian_dist(df_train, mean, cov)
prob_cross = gaussian_dist(df_cross, mean, cov)
prob_test = gaussian_dist(df_test, mean, cov)

print(prob_train.shape)
print(prob_cross.shape)
print(prob_test.shape)
#Using cross-validation set to find the optimum epsilon
def optimize_for_epsilon(prob_train, prob_cross, Y_cross):
    best_f1 = 0
    max_e = 2.062044871798754e-79
    min_e = prob_train.min()
    step = (max_e - min_e) / 1000
    
    for e in np.arange(prob_cross.min(), max_e, step):
        Y_cross_pred = prob_cross < e
        precision, recall, f1_score, support = prfs(Y_cross, Y_cross_pred, average="binary")
        print("for epsilon: {}".format(e))
        print("f1_score: {}".format(f1_score))
        print("recall: {}".format(recall))
        print("precision: {}".format(precision))
        print("support: {}".format(support))
        print()
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_epsilon = e
            recall = recall
        
    return best_f1, best_epsilon, recall

best_f1, best_epsilon, recall = optimize_for_epsilon(prob_train, prob_cross, Y_cross)
print(best_f1, best_epsilon, recall)
    
#Predicting the anomalies on test_set using the optimal epsilon from above results
Y_test_pred = prob_test < best_epsilon
precision, recall, f1_score, ignore = prfs(Y_test, Y_test_pred, average="binary")
print("epsilon: {}".format(best_epsilon))
print("f1_score: {}".format(f1_score))
print("recall: {}".format(recall))
print("precision: {}".format(precision))
# print("support: {}".format(support))

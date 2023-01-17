# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb 

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import fbeta_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc   # area under precision-recall-curve





%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
train_data.shape
train_data.head()
#check datatypes

train_data.dtypes.value_counts()
#split the dataset into features and target

features = train_data.iloc[:,2:]

target = train_data.iloc[:,1]
#distribution of labels

sb.countplot(target)



target.value_counts()
# check for missing values

train_data.isnull().sum().sum()

train_data.hist(figsize = (20,20))



plt.tight_layout
#Standard scaling



scaler = StandardScaler()

features_scaled = pd.DataFrame(scaler.fit_transform(features), columns= features.columns)
#PCA 

pca = PCA()

pca.fit(features_scaled)
plt.plot(pca.explained_variance_ratio_)
def show_pca_variance(pca_obj, desired_ratio = None):

    

    var = pca_obj.explained_variance_ratio_

    cumu = var.cumsum()

    

    f, ax1 = plt.subplots()

    ax1.set_xlabel('n_components')

    ax1.set_ylabel('ratio of explained variance')

    ax1.plot(var)

    

    ax2 = ax1.twinx()

    

    

    ax2.set_ylabel('cumulative variance')

    ax2.plot(cumu, color = 'tab:red')

    f.tight_layout()

    

    #get x such that x number of components retain the desired ratio of variance.

    if desired_ratio != None:

        x = np.where(cumu > desired_ratio)[0][0] + 1

        plt.axvline(x = x, color = 'tab:green')

        print(f' {desired_ratio*100}% of variability can be achieved by minimum of {x} components, or {x * 100/pca_obj.n_components_:.2f}% of total features')

    

    plt.show()

     

    return cumu



#sort weights

def sorted_weights_pca(pca_obj, i, features, k ):

    

    # first create a dictionary that contain map components weights to feature names.

    dict_feature_weight = {}

    

    for j in range(len(pca_obj.components_[i])):

        dict_feature_weight[features[j]] = [pca_obj.components_[i][j]]

        

    

    # dataframe

    df = pd.DataFrame( data = dict_feature_weight)

    df = df.T

    df.columns = ['weights']

    

    df = df.sort_values(by = 'weights' , ascending=False)

    

    #plot the top 10 features

    fig = plt.figure(1)

    plt.figure(figsize = (10,5))

    

        

    plt.subplot(121)

    plt.barh(np.arange(k), df['weights'][-k:])

    plt.yticks(np.arange(k), tuple(df.index)[-k::])

    

    plt.subplot(122)

    plt.barh(np.arange(k), df['weights'][:k])

    plt.yticks(np.arange(k), tuple(df.index)[:k:])

    

    plt.tight_layout()

    

    #print(tuple(df.index))

    return df
show_pca_variance(pca, desired_ratio= 0.99);

show_pca_variance(pca, desired_ratio= 0.95);

show_pca_variance(pca, desired_ratio= 0.90);

show_pca_variance(pca, desired_ratio= 0.85);

show_pca_variance(pca, desired_ratio= 0.80);
#refit pca

pca = PCA(n_components= 179)

pca.fit(features_scaled)
features_scaled_pca = pca.transform(features_scaled)
features_scaled_pca.shape
#fit a logistic model. this serves as our baseline model.



model = LogisticRegression(random_state=42, solver= "liblinear")



model.fit(features_scaled_pca,target)



#model.score(features_scaled_pca, target)



#alternaive, we can first use predict(), then accuracy_score().

train_predict = model.predict(features_scaled_pca)

accuracy_score(target, train_predict)

#precision

# tp/(tp + fp)

# out of all the datapoints we identified as positive, how many of them are right?

# measures ability to 

# 0.69 means every 100 positives, 69 of them are correct. 



precision_score(target, train_predict)
#recalls

#tp/(tp + fn)

#out of all the datapoints that are actually positive, how many of them get identified as positive?

# measures the ability to 

# 0.27 means every 100 datapoints that are actually positive, 27 of them would get identified correctly.



recall_score(target, train_predict)
#given equal weights on precision and recall



fbeta_score(target, train_predict, beta = 1)
#or alternative, this function output precision, recall, f score and support altogether. 

precision_recall_fscore_support(target, train_predict, beta=1)
#confusion matrix

#C_ij: number of observations known to be in group i but predicted to be in group j.



sb.heatmap(confusion_matrix(target, train_predict), cmap="YlGnBu")

confusion_matrix(target, train_predict)
train_predict_prob = model.predict_proba(features_scaled_pca)
#ROC curve



fpr, tpr, thresholds = roc_curve(target, train_predict_prob[:,1])





# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')



auc = roc_auc_score(target, train_predict_prob[:,1])



print(f'auc: {auc}')
#If positive class is rare or if we care more about false positive than false negative, use Precision/Recall curve instead.



pre, rec, thres = precision_recall_curve(target, train_predict_prob[:,1])



auc_pr = auc(rec, pre)



print(f'auc precision recall curve: {auc_pr}')



plt.plot(rec, pre)
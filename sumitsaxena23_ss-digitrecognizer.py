import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
# Importing train and test data

digit = pd.read_csv('../input/train.csv')



# Test data is being called 'final' (i.e. one for final submission) to avoid confusion with the test data for validating the model

digit_final = pd.read_csv('../input/test.csv')
digit.shape
digit.describe()
digit.columns
y = digit.iloc[:,0]
X = digit.iloc[:,1:]

X_final = digit_final.iloc[:,1:]
X[X!=0]=1

X_final[X_final!=0]=1
# All data is not between 0 and 1. No scaling is required

# train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,train_size=0.7, test_size=0.3)
# running PCA to reduce dimensionality



from sklearn.decomposition import PCA
pca = PCA(svd_solver= 'randomized', random_state=42)
pca.fit(X_train)
pca.components_
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.ylabel('Cummulative Variance Explained')

plt.xlabel('Number of PCs')

plt.show()
list(zip(range(X_train.shape[1]),np.cumsum(pca.explained_variance_ratio_)))
### First 167 PCs explain ~90% variance
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=167)
df_train_pca = pca_final.fit_transform(X_train)
df_train_pca.shape
df_test_pca = pca_final.transform(X_test)
# Logistics Regression

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



learner_pca = LogisticRegression()

model_pca = learner_pca.fit(df_train_pca,y_train)
#Making prediction on the test data

pred_probs_test = model_pca.predict_proba(df_test_pca)

#metrics.roc_auc_score(y_test, pred_probs_test)
y_test_pred = pd.DataFrame(pred_probs_test).idxmax(axis=1)
# Confusion matrix 

confusion = metrics.confusion_matrix( y_test, y_test_pred)

confusion
#Let's check the overall accuracy.

metrics.accuracy_score( y_test, y_test_pred)
# 90% accuracy
# precision

list(zip(range(10),metrics.precision_score( y_test, y_test_pred,average=None)))
# recall

list(zip(range(10),metrics.precision_score( y_test, y_test_pred,average=None)))
# Creating list for submission

# Applying PCA

df_final_pca = pca_final.transform(digit_final)

# Applying model

pred_probs_final = model_pca.predict_proba(df_final_pca)

# generating labels

y_final_pred = pd.DataFrame(pred_probs_final).idxmax(axis=1)
ImageId= pd.DataFrame(np.arange(1,digit_final.shape[0]+1))

y_final_pred=pd.concat([ImageId,y_final_pred],axis=1)

y_final_pred.columns=['ImageId','Label']
y_final_pred.to_csv('submission.csv',index=False)
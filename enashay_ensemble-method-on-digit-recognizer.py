# 1. import libraries
# Random Forest method is used from sklearn library
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 2. read data
# reading the data into data dataframe and split into X and y
data = pd.read_csv('../input/train.csv')
X, y = data.iloc[:,1:], data.iloc[:,0]

# 3. train kNN classified
# Not changing any default parameters. We want to make it as simple as possible
# So just training the model and doing prediction on test dataset
clf = clf = RandomForestClassifier(n_estimators=10).fit(X,y)
pred = clf.predict(pd.read_csv('../input/test.csv'))

# 4. save and predict
# saving results in required format
df_pred = pd.DataFrame(pred, index = range(1,len(pred)+1)).reset_index().rename(columns={'index' : 'ImageId',0: "Label"})
df_pred.to_csv('../knn_submission.csv', index = False)

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



# Read Data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head



print(train.shape)



# save the labels to a Pandas series target

target = train['label']

# Drop the label feature

train = train.drop("label",axis=1)



# Standardize Data

X = train.values

X_std = StandardScaler().fit_transform(X)

Y = target.values



print(test.shape)

x_std = StandardScaler().fit_transform(test)



# Multinomial Logistic Regression

lgRg = LogisticRegression(multi_class='multinomial',

                         penalty='l1', solver='saga', tol=0.0001)

lgRg.fit(X_std, Y)

y = lgRg.predict(x_std)



# Submission

submission = pd.DataFrame({"ImageId": range(1,y.shape[0]+1),

    "label": y})



submission.to_csv("submission_Logistic.csv", index=False)
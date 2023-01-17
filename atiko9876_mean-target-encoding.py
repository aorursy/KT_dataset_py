!pip install target_encoding

# https://github.com/KirillTushin/target_encoding
import pandas as pd



from target_encoding import TargetEncoderClassifier, TargetEncoder

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression, LogisticRegression
train=pd.read_csv("../input/cat-in-the-dat/train.csv")

test=pd.read_csv("../input/cat-in-the-dat/test.csv")

sample_submission = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')



cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
len_uniques = []

for c in train.columns.drop(['id', 'target']):

    le = LabelEncoder()

    le.fit(pd.concat([train[c], test[c]])) 

    train[c] = le.transform(train[c])

    test[c] = le.transform(test[c])

    print(c, len(le.classes_))

    len_uniques.append(len(le.classes_))

    

X = train.drop(['target', 'id'], axis=1)

y = train['target']
ALPHA = 75

MAX_UNIQUE = max(len_uniques)

FEATURES_COUNT = X.shape[1]
'''

alpha: float or int, smoothing for generalization.



max_unique: int, maximum number of unique values in a feature. 

            If there are more unique values inside the feature,

            then the algorithm will split this feature into bins, 

            the number of max_unique.



used_features: int, this is a number of used features for prediction

               The algorithm encodes all features with the average value of the target, 

               then the std is considered inside each feature,

               and "used_features" features with the highest std are selected to use only informative features. 

'''



enc = TargetEncoderClassifier(alpha=ALPHA, max_unique=MAX_UNIQUE, used_features=FEATURES_COUNT)

score = cross_val_score(enc, X, y, scoring='roc_auc', cv=cv)

print(f'score: {score.mean():.4}, std: {score.std():.4}')



enc.fit(X, y)

pred_enc = enc.predict_proba(test.drop('id', axis=1))[:,1]
'''

split: list of int or cross-validator class,

            if split is [], then algorithm will encode features without cross-validation

            This situation features will overfit on target



            if split len is 1 for example [5], algorithm will encode features by using cross-validation on 5 folds

            This situation you will not overfit on tests, but when you will validate, your score will overfit



            if split len is 2 for example [5, 3], algorithm will separate data on 5 folds, afterwords

            will encode features by using cross-validation on 3 folds

            This situation is the best way to avoid overfit, but algorithm will use small data for encode.

'''





enc = TargetEncoder(alpha=ALPHA, max_unique=MAX_UNIQUE, split=[cv])

X_train = enc.transform_train(X=X, y=y)

X_test = enc.transform_test(test.drop('id', axis=1))
lin = LogisticRegression()

score = cross_val_score(lin, X_train, y, scoring='roc_auc', cv=cv)

print(f'score: {score.mean():.4}, std: {score.std():.4}')





lin.fit(X_train, y)

pred_lin = lin.predict_proba(X_test)[:,1]
sample_submission['target'] = pred_enc + pred_lin

sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()
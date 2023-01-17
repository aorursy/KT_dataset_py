import pandas as pd
train_df = pd.read_csv('../input/digit-recognizer/train.csv')
test_df = pd.read_csv('../input/digit-recognizer/test.csv')
sub_df = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
train_df
train_df.isna().sum()
X_train = train_df.drop(columns=['label'],axis=1)
y_train = train_df['label']
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
test_df
X_test = scalar.fit_transform(test_df)
sub_df
from sklearn.svm import SVC
model_svc = SVC()
model_svc.fit(X_train, y_train)
model_svc.score(X_train,y_train)
y_predict = model_svc.predict(X_test)
y_predict
submission  = pd.DataFrame({'ImageId': sub_df.ImageId, 'label': y_predict})
submission
submission.to_csv("digit_recog_svm.csv",index=False)
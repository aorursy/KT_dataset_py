import pandas as pd
from sklearn import ensemble
# The competition datafiles are in the directory ../input
file_train = "../input/train.csv"
file_test = "../input/test.csv"
df_train = pd.read_csv(file_train)
df_test = pd.read_csv(file_test)
df_train.head()
feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]
y = df_train['Cover_Type'] # target
test_ids = df_test['Id'] # for submission
clf = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
clf.fit(X_train, y)
file_submission = "rf200.submission.csv"
with open(file_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict(X_test))):
        outfile.write("%s,%s\n"%(test_ids[e],val))

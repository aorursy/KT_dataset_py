import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

def extract_data(train):
    feature_names = train.columns.values

    if feature_names[0] == 'label':
        labels = train['label']
        feature_names = feature_names[1:]
        feature_matrix = train[feature_names]
        return (labels, feature_names, feature_matrix)
    else:
        feature_matrix = train[feature_names]
        return (feature_names, feature_matrix)

labels, feature_names, feature_matrix = extract_data(train)

forest = RandomForestClassifier(n_estimators=1000, n_jobs = 5)
forest.fit(feature_matrix, labels)

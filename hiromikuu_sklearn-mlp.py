data_path = "data/"

train = pd.read_csv(data_path+'act_train.csv')

test = pd.read_csv(data_path+'act_test.csv')

people = pd.read_csv(data_path+'people.csv')
data = pd.concat([train,test])

data = pd.merge(data,people,how='left',on='people_id').fillna('missing')

train = data[:train.shape[0]]

test = data[train.shape[0]:]
columns = train.columns.tolist()

columns.remove('activity_id')

columns.remove('outcome')

data = pd.concat([train,test])

for c in columns:

    data[c] = LabelEncoder().fit_transform(data[c].values)

train = data[:train.shape[0]]

test = data[train.shape[0]:]
train_labels = train['outcome']
del train['activity_id']
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)
columns = train.columns.tolist()

X_t = test[columns].values
activity_id = test['activity_id']
y_preds = clf.predict_proba(X_t)
def load():

    clf = joblib.load('mlp_100.pkl') 

    return clf
clf2 = load()
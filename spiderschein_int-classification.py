import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.io import arff
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

input_path = "../input/kiwhs-comp-1-complete/"

content, _meta = arff.loadarff(input_path + "train.arff")

df = pd.DataFrame(content)
inputs = df.iloc[:,:2].values
labels = df.iloc[:,2].astype('int')

test_df = pd.read_csv(input_path + "test.csv")
test = test_df[['X', 'Y']].values
plt.scatter(inputs[:,0], inputs[:,1], c=labels.apply(lambda x: 'red' if x == 1 else 'blue'))
df.head()
df.describe()
train, eval, train_labels, eval_labels = train_test_split(inputs, labels, test_size=0.1)
scaler = StandardScaler()
#scaler.fit(train)

#train = scaler.transform(train)
#eval = scaler.transform(eval)
#test = scaler.transform(test)
#pd.DataFrame(train).describe()

# Train the model
#model = KNeighborsClassifier(n_neighbors=13)
#model = SVC()
model = MLPClassifier(
    hidden_layer_sizes=(50,100),
    solver='sgd',
    random_state=100,
    verbose=True,
    tol=0.0001,
    max_iter=500)
model.fit(train, train_labels)

# Evaluate the model
eval_pred = model.predict(eval)
score = accuracy_score(eval_pred, eval_labels)
print("Score: {}".format(score))

test_pred = model.predict(test)
result_df = pd.DataFrame({
    'Id (String)': pd.RangeIndex(400),
    'Category (String)': test_pred
})
result_df.to_csv('out.csv', index=False)
result_df.head()

result_df.describe()
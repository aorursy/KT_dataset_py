import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/kaggle/input/epitope-prediction/input_bcell.csv')

dataset.head()
dataset.columns
dataset.isnull().sum()
for col in dataset.columns:
    x = dataset[col].unique()
    if len(x) < 20:
        print(f"{col}: {x}")
X = dataset.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq', 'target'])
X.head()
y = dataset['target']
y.head()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

X
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 10))
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(0.2))
classifier.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(0.2))
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 300)
# Save the model
classifier.save('model.h5')

# Save a dictionary into a pickle file.
import pickle
pickle.dump(sc, open( "scaler.p", "wb" ))
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm
import seaborn as sn
sn.heatmap(cm, cmap="Blues", annot=True,annot_kws={"size": 16})
from sklearn.metrics import accuracy_score

print('VALIDATION ACCURACY', accuracy_score(y_test, y_pred))
sars = pd.read_csv('/kaggle/input/epitope-prediction/input_sars.csv')

sars.head()
sars.columns
sars.isnull().sum()
X_sars = sars.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq', 'target'])
X_sars.head()
y_sars = sars['target']
y_sars.head()
X_sars = sc.fit_transform(X_sars)

X_sars
# Predicting the Test set results
y_sars_pred = classifier.predict(X_sars)
y_sars_pred = (y_sars_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_sars, y_sars_pred)

cm
from sklearn.metrics import accuracy_score

print('VALIDATION ACCURACY', accuracy_score(y_sars, y_sars_pred))

import sys
!{sys.executable} -m pip install --upgrade pip
import sys
!{sys.executable} -m pip install skorch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder 
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import EarlyStopping
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch
#predefinovanie hodnout X_valid a Y_valid, ktoré používám nižšie
def predefined_array_split(X_valid, Y_valid):
    return predefined_split(
        TensorDataset(
            torch.as_tensor(X_valid),
            torch.as_tensor(Y_valid)
        )
    )
#zobrazenie ciest ku potrebným súborom
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#načítanie datovej množiny s údajmi na trening modelu
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
#zobrazenie informacií o dátovej množine
train.info()
#rozdenie datovej množina na treningovú : validačnú : testovaciu a to 70 : 5 : 25
train_valid, test1 = train_test_split(train, test_size=0.25,
                                     stratify=train['Survived'],
                                     random_state=4)
train, valid = train_test_split(train_valid, test_size=0.05/0.75,
                                     stratify=train_valid['Survived'],
                                     random_state=4)
#vyber stlpcou pomocou, ktorých budem rozhodovať a rozndelenie na textové a numerické
categorical_inputs = ["Sex", "Embarked",]
numeric_inputs = ["Pclass","Age", "SibSp", 'Parch', 'Fare']

output = ["Survived"]
#použitie pipeline
input_preproc = make_column_transformer(
    (make_pipeline(
        SimpleImputer(strategy='constant', fill_value='MISSING'),
        OneHotEncoder()),
     categorical_inputs),
    
    (make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()),
     numeric_inputs)
)
output_preproc = OrdinalEncoder()
#transformacia dát
X_train = input_preproc.fit_transform(train[categorical_inputs+numeric_inputs])
Y_train = output_preproc.fit_transform(train[output]).reshape(-1)
X_valid = input_preproc.transform(valid[categorical_inputs+numeric_inputs])
Y_valid = output_preproc.transform(valid[output]).reshape(-1)
X_test1 = input_preproc.transform(test1[categorical_inputs+numeric_inputs])
Y_test1 = output_preproc.transform(test1[output]).reshape(-1)
#zmena dátoveho typu
X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.int64)
X_valid = X_valid.astype(np.float32)
Y_valid = Y_valid.astype(np.int64)
X_test1 = X_test1.astype(np.float32)
Y_test1 = Y_test1.astype(np.int64)
num_inputs = X_train.shape[1]
num_outputs = len(np.unique(Y_train))

#vytvorenie vrstiev
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(num_inputs, 60)
        self.fc2 = nn.Linear(60, 70)
        self.fc3 = nn.Linear(70, 80)
        self.fc4 = nn.Linear(80, num_outputs)

    def forward(self, x):
        y = self.fc1(x)
        y = torch.relu(y)
        y = self.dropout1(y)
        
        y = self.fc2(y)
        y = torch.relu(y)
        y = self.dropout2(y)

        y = self.fc3(y)
        y = torch.relu(y)
        y = self.dropout3(y)
        
        y = self.fc4(y)
        y = torch.softmax(y, dim=1)
        
        return y
#definovanie parametrov neuronovej siete
net = NeuralNetClassifier(
    Net,
    max_epochs=50,
    batch_size=-1,
    optimizer=torch.optim.Adam,
    train_split=predefined_array_split(X_valid, Y_valid),
    device="cuda" if torch.cuda.is_available() else "cpu",
    callbacks=[
        EarlyStopping(patience=10)
    ]
)

#trening neuronovaj siete na treningových dátach
net.fit(X_train, Y_train)
y_valid = net.predict(X_valid)

cm = pd.crosstab(
    output_preproc.inverse_transform(
        Y_valid.reshape(-1, 1)).reshape(-1),
    output_preproc.inverse_transform(
        y_valid.reshape(-1, 1)).reshape(-1),
    rownames=['actual'],
    colnames=['predicted']
)
print(cm)

acc = accuracy_score(Y_valid, y_valid)
print("Accuracy on valid = {}".format(acc))
y_test1 = net.predict(X_test1)

cm = pd.crosstab(
    output_preproc.inverse_transform(
        Y_test1.reshape(-1, 1)).reshape(-1),
    output_preproc.inverse_transform(
        y_test1.reshape(-1, 1)).reshape(-1),
    rownames=['actual'],
    colnames=['predicted']
)
print(cm)

acc = accuracy_score(Y_test1, y_test1)
print("Accuracy on valid = {}".format(acc))
X_test = input_preproc.transform(test[categorical_inputs+numeric_inputs])
Y_test = test.values.reshape(-1)

X_test = X_test.astype(np.float32)
y_test = net.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('v3gender_submission.csv', index=False)
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
pd.set_option('display.max_rows', None)
dataset = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
dataset.head()
dataset.shape
dataset.info()
dataset.isnull().sum()
dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
dataset['LoanAmount'] = dataset['LoanAmount'].astype(float)
dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].median())
dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].astype(float)
dataset['Credit_History'] = dataset['Credit_History'].fillna(dataset['Credit_History'].median())
dataset['Credit_History'] = dataset['Credit_History'].astype(float)
dataset.dropna(inplace = True)
X = dataset.drop(columns=['Loan_ID','Loan_Status'])

dep = {'0':'0','1':'1','2':'2','3+':'3'}
X['Dependents'] = X['Dependents'].map(dep)

X.info()
X.isnull().sum()
y = dataset['Loan_Status']
y.head()
from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
y = lb_make.fit_transform(y)

y
##Encode Categorical Values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
categorical_features = ['Gender', 'Married','Dependents','Education','Self_Employed','Property_Area']
#categorical_features = ['Dependents', 'Education','Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",one_hot,categorical_features)], remainder = 'passthrough')
X = transformer.fit_transform(X)
### Splitting data into test set and training set

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
### Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train.shape, X_test.shape, y_train.shape
import keras
from keras.models import Sequential 
from keras.layers import Dense,Dropout
classifier = Sequential()
classifier.add(Dense(units = 12 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 20))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 12 , kernel_initializer = 'uniform' , activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid'))
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
classifier.summary()
classifier.fit(X_train , y_train , batch_size = 10 , epochs = 100)
y_pred = classifier.predict(X_test)  ### will give the probability as the output

y_pred = (y_pred > 0.5) ### to see the true and false results
y_pred
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier ():
    classifier = Sequential()
    classifier.add(Dense(units = 12 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 20))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 12 , kernel_initializer = 'uniform' , activation = 'relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10 , epochs = 100)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train , cv = 10, n_jobs=1,verbose=1)
mean = accuracies.mean()
variance = accuracies.std()
print(mean , variance)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier (optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 11 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 20))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(units = 11 , kernel_initializer = 'uniform' , activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(units = 11 , kernel_initializer = 'uniform' , activation = 'relu'))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameter = {'batch_size' : [32,35,38],
             'epochs' : [50,100,500],
            'optimizer' : ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameter,scoring = 'accuracy',cv=10)
grid_search = grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters,
     best_accuracy)


















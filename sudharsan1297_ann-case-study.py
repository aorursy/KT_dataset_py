import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
tf.__version__
df = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
X = df.iloc[:, 3:-1]
y = df.iloc[:, -1]
X.head(2)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Gender'] = le.fit_transform(X["Gender"])
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

dumm = pd.get_dummies(X['Geography'])


X = pd.concat([X,dumm],axis = 1)
X = X.drop("Geography",1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
n = int(input("Enter number of layers: "))
neuron = int(input("Enter the number of neurons per layer: "))

ann = tf.keras.models.Sequential()
for i in range(1,n+1):
  ann.add(tf.keras.layers.Dense(units = neuron, activation='relu'))

ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epoch = int(input("Enter the number of epochs: "))
ann.fit(X_train,y_train, batch_size=32, epochs = epoch)
y_pred = ann.predict(X_test)
y_test = np.array(y_test).reshape(-1,1)
from sklearn.preprocessing import binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
for i in range(1,11):
    y_pred2=binarize(y_pred,i/10)[:]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    print('The accuracy score is: ',accuracy_score(y_test,y_pred2))
    print('The f1 score is: ',f1_score(y_test,y_pred2))
    print('\n')
y_pred2=binarize(y_pred,0.5)[:]
df.head(1)
X_test[0]
result = ann.predict(sc.transform([[600,0,40,3,60000,2,1,1,50000,1,0,0]]))
result_=binarize(result,0.5)[:]
result_
print("As the value of the prediction is ",int(result_[0])," we can retain the customer as he is not likely to switch")

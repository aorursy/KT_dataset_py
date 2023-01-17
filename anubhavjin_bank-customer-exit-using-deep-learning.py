df=pd.read_csv('/kaggle/input/bank-customers/Churn Modeling.csv')
df.head()
df.info()
df.describe()
X=df.iloc[:,3:13]
y=df.iloc[:,13]
X.head()
states=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X["Gender"],drop_first=True)
X=pd.concat([X,states,gender],axis=1)
X.head()
X=X.drop(['Geography','Gender'],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(activation='relu',input_dim=11,units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))
classifier.summary()
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=50)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
y_pred
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)

acc
cm
classifier=Sequential()
classifier.add(Dense(activation='relu',input_dim=11,units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=9,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='relu',units=6,kernel_initializer='uniform'))
classifier.add(Dense(activation='sigmoid',units=1,kernel_initializer='uniform'))
classifier.summary()
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=30)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
acc=accuracy_score(y_test,y_pred)
acc

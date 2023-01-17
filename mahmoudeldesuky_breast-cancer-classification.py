#importing the libraries
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
#calling the data 
cancer=load_breast_cancer()
x=cancer.data
y=cancer.target
print('the feature names are',cancer.feature_names)
print('the x ata ',x[:10])
print('the x shape is ',x.shape)

#spilliting the data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,shuffle=True)
print('the x_train ',x_train.shape)

#training the model
classification=LogisticRegression(max_iter=100000,solver='liblinear',C=112,random_state=101, )
classification.fit(x_train, y_train)
#testing the model
y_pred=classification.predict(x_test)
y_pred_prob=classification.predict_proba(x_test)

#the accurecy
cm=confusion_matrix(y_test,y_pred)
print('the model score is',classification.score(x_train, y_train))
print('the number of iterations is ',classification.n_iter_)
print('the cofusion matrix \n', cm)
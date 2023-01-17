import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
digits=datasets.load_digits()
data=digits.data
df=pd.DataFrame(digits.data)
df['Numbers']=digits.target
df.head(10)
#Count of different numbers
sns.countplot(x=df['Numbers'])
for i in range(10):
    plt.gray()
    plt.matshow(digits.images[i])
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#training the model without tuning hyperparameters
from sklearn.svm import SVC
classifier=SVC(gamma='auto')
classifier.fit(X_train,y_train)
classifier.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV

grid_classifier=GridSearchCV(SVC(),{'C':[2,5,10,20,50],
                'kernel':['rbf','linear'],'gamma':[0.1,0.01,0.001]
               },cv=5)
grid_classifier.fit(X,y)
result=pd.DataFrame(grid_classifier.cv_results_)
result[['param_gamma','param_C','param_kernel','mean_test_score']]
#Best Estimator
grid_classifier.best_params_
model=SVC(C=2,gamma=0.001,kernel='rbf')
model.fit(X_train,y_train)
model.score(X_test,y_test)
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test,model.predict(X_test),figsize=(8,8))
from sklearn.metrics import classification_report
print(classification_report(y_test,model.predict(X_test)))
from sklearn.model_selection import cross_val_score
accuracy=cross_val_score(model,X_train,y_train,cv=10)
accuracy.mean()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/Iris.csv',index_col=['Id'])
df.head(2)
import numpy as np
df['Species'].unique()
mc=df.sample(frac=1)
sns.scatterplot(x=mc['PetalLengthCm'],y=mc['SepalLengthCm'],hue = mc['Species'])
def Change_species(x):
    if x == 'Iris-setosa':
        return 0
    elif x == 'Iris-versicolor':
        return 1
    else:
        return 2
    
    
mc['Species']=mc['Species'].apply(Change_species)
mc.head(10)
#iris-setosa = 1
#iris-versicolor=2
#iris-verginica = 3


mc.columns = ['sepal_length','sepal_width','petal_length','petal_width','target']
mc.head(2)

mc.tail(5)


y=mc['target']
X=mc.drop('target',axis=1)
X.head(2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#import tensorflow as tf
#feature columns for tensorflow models
X.columns
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))




#feat_cols = []

#for col in X.columns:
    #feat_cols.append(tf.feature_column.numeric_column(col))
#tf.feature_column.numeric_column() creates numeric entries into tf format
feat_cols = X.columns
#now to create input function
#input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=5,shuffle=True) #because we have pandas dataframe(shuffle is the targets are sorted, epoch is going through data one time)
#classifier = tf.estimator.DNNClassifier(hidden_units=[10],n_classes=3,feature_columns=feat_cols) #(hidden_units = [] defines the layer, and number of neurons in it,n_classes defines the number of species of flower(here),)
#classifier.train(input_fn=input_func,steps=5)
#pred_fn=tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False) #batch size= all of X_test because our trainig model is already trained and we are now going to jjust test
#predictions=list(classifier.predict(input_fn=pred_fn))
#predictions
#list(predictions) or predictions=list(classifier.predict(input_fn=pred_fn))
#final_preds=[]
#for pred in predictions:
    #final_preds.append(pred['class_ids'][0])
#final_preds[:10]

#final_preds
#print(classification_report(y_test,final_preds))
y_test.head()

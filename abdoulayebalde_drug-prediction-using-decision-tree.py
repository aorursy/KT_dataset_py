df = pd.read_csv('/kaggle/input/drugsets/drug200.csv')
df.head()
# let see the shape of the data
df.shape
# now we can see that we are dealing with categorical data so we have to transform them because machine learning algorithms work only with numerical data for now let import the important package
# we are going to use 
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['BP'] = encoder.fit_transform(df['BP'])
df['Cholesterol'] = encoder.fit_transform(df['Cholesterol'])
df['Na_to_K'] = encoder.fit_transform(df['Na_to_K'])

X[:5]
X = df.iloc[:,[1,2,3,4]]
y = df.iloc[:,5]
X[:5] ,y[:5]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =.3,random_state =3)
# now let see the shape of train and test set
X_train.shape, y_train.shape
X_test.shape , y_test.shape
from sklearn.tree import DecisionTreeClassifier
classifer = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
classifer.fit(X,y)
# y pred variable 
y_pred = classifer.predict(X_test)
y_pred[:5]
#You can print out y_pred and y_test if you want to visually compare the prediction to the actual values.
print(y_pred[:5])
print(y_test[:5])
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_pred))
from sklearn.externals.six import StringIO
# import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 
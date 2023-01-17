import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
df = pd.read_csv('../input/Iris.csv')
df.head()
#Checking the data. Iris-Setosa seems to  be the most separable
sns.pairplot(df,hue = 'Species')
sns.jointplot('SepalWidthCm','SepalLengthCm',df[df['Species'] == 'Iris-setosa'],kind='kde')
def encode_species(Species):
    if Species == 'Iris-setosa':
        return 0
    elif Species == 'Iris-versicolor':
        return 1
    else:
        return 2
#Creating test and train data
df['Type'] = df['Species'].apply(encode_species)
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Species','Type'],axis=1), df['Type'], test_size=0.30)
#SVM Classifier
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

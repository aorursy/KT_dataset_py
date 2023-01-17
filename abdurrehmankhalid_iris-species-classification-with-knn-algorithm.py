import numpy as numpyInstance
import pandas as pandasInstance
import seaborn as seabornInstance
import matplotlib.pyplot as matplotlibInstance
%matplotlib inline
speciesData = pandasInstance.read_csv('../input/Iris.csv')
speciesData.head()
speciesData.info()
seabornInstance.pairplot(speciesData,hue='Species')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
speciesData.columns
scaler.fit(speciesData[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
transformedFeatures = scaler.transform(speciesData[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformedFeatures, speciesData['Species'], test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
kModel = KNeighborsClassifier(n_neighbors=1)
kModel.fit(X_train,y_train)
predictions = kModel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(predictions,y_test))
errors = []
for interger in range(1,40):
    newModel = KNeighborsClassifier(n_neighbors=interger)
    newModel.fit(X_train,y_train)
    newPredictions = newModel.predict(X_test)
    errors.append(numpyInstance.mean(newPredictions!=y_test))
matplotlibInstance.figure(figsize=(10,6))
matplotlibInstance.plot(range(1,40),errors,color='green', linestyle='dashed', marker='o')
matplotlibInstance.title('Error Rate vs. K Value')
matplotlibInstance.xlabel('K')
matplotlibInstance.ylabel('Error Rate')
kModel = KNeighborsClassifier(n_neighbors=5)
kModel.fit(X_train,y_train)
predictions = kModel.predict(X_test)
print(classification_report(predictions,y_test))

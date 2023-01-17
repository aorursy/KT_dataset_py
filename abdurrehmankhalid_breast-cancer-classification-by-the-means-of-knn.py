import numpy as numpyInstance
import pandas as pandasInstance
import matplotlib.pyplot as matplotlibInstance
import seaborn as seabornInstance
%matplotlib inline
breastCancerData = pandasInstance.read_csv('../input/data.csv')
breastCancerData.head()
breastCancerData.info()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
breastCancerData.columns
scaler.fit(breastCancerData[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']])
transformed = scaler.transform(breastCancerData[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']])
toMakeNewDataFrame = breastCancerData[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
newDataFrameWithFeatures = pandasInstance.DataFrame(transformed,columns=toMakeNewDataFrame.columns)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(newDataFrameWithFeatures, breastCancerData['diagnosis'], test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knModel = KNeighborsClassifier(n_neighbors=1)
knModel.fit(X_train,y_train)
predictions = knModel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(predictions,y_test))
#Here we are Calculating predictions for each Value of K from 1 to 40 and Calculating the Average Error Value and Storing it in ERROR.
errors = []
for number in range(1,50):
    anOtherModel = KNeighborsClassifier(n_neighbors=number)
    anOtherModel.fit(X_train,y_train)
    anOtherpredictions = anOtherModel.predict(X_test)
    errors.append(numpyInstance.mean(predictions!=y_test))
matplotlibInstance.figure(figsize=(10,6))
matplotlibInstance.plot(range(1,50),errors,color='green', linestyle='dashed', marker='o')
matplotlibInstance.title('Error Rate vs. K Value')
matplotlibInstance.xlabel('K')
matplotlibInstance.ylabel('Error Rate')

#Import all necessary modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse
#Importing the data set

df=pd.read_csv("/kaggle/input/red_wine_quality.csv")
df.head(10)
#check for null values

df.info()
#gives number of null values in each column.

df.isnull().sum()
#drawing histograms

#generate gaps between subplots
fig, axes = plt.subplots(nrows=2, ncols=3)
fig.tight_layout(pad=3)

#histogram plots
fig.suptitle("Histogram plots")

plt.subplot(2,3,1)
plt.hist(df['chlorides'],bins=10,color='lightblue',ec='black')
plt.xlabel("chlorides")
plt.ylabel("Frequency")

plt.subplot(2,3,2)
plt.hist(df['density'],bins=10,color='lightblue',ec='black')
plt.xlabel("density")
plt.ylabel("Frequency")

plt.subplot(2,3,3)
plt.hist(df['pH'],bins=10,color='lightblue',ec='black')
plt.xlabel("pH")
plt.ylabel("Frequency")

plt.subplot(2,3,4)
plt.hist(df['sulphates'],bins=10,color='lightblue',ec='black')
plt.xlabel("sulphates")
plt.ylabel("Frequency")

plt.subplot(2,3,5)
plt.hist(df['alcohol'],bins=10,color='lightblue',ec='black')
plt.xlabel("alcohol")
plt.ylabel("Frequency")

plt.subplot(2,3,6)
plt.hist(df['citric acid'],color='lightblue',ec='black',bins=10)
plt.xlabel("citric acid")
plt.ylabel("Frequency")

plt.show()
#box plots for all numerical columns

for cl in df:
    plt.figure()
    plt.ylabel("frequency")
    df.boxplot([cl],color='red')
#scatter plots

plt.scatter(x=df['fixed acidity'],y=df['alcohol'],marker='D',color='black')
plt.xlabel("fixed acidity")
plt.ylabel("alcohol")
plt.show()

plt.scatter(x=df['volatile acidity'],y=df['sulphates'],marker='v',color='violet')
plt.xlabel("volatile acidity")
plt.ylabel("sulphates")
plt.show()

plt.scatter(x=df['citric acid'],y=df['pH'],marker='*',color='green')
plt.xlabel("citric acid")
plt.ylabel("pH")
plt.show()

plt.scatter(x=df['residual sugar'],y=df['density'],marker='X',color='red')
plt.xlabel("residual sugar")
plt.ylabel("density")
plt.show()

plt.scatter(x=df['chlorides'],y=df['total sulfur dioxide'],marker='p',color='blue')
plt.xlabel("chlorides")
plt.ylabel("total SO2")
plt.show()
#splitting the data into train and test
#fixed acidity, free sulfur dioxide, total sulfur dioxide, sulphates, alcohol are trained

X = df.drop(['quality'],axis=1).values
y = df["quality"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42,stratify=y)
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)

#accuracy score

acc=knn.score(X_test,y_test)
print('percentage accuracy :{:f} %'.format(100*acc))
# accuracy score

k_range=range(1,40)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
print(scores)
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
SEED=1
# Instantiate rf
rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=0.12,random_state=SEED)
            
# Fit rf to the training set    
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
rmse_test=mse(y_test,y_pred)**(1/2)
print('test set RMSE of RF: {:.3f}'.format(rmse_test))
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightblue')
plt.title('Features Importances')
plt.show()
correlation_df = df.corr()
print(correlation_df)









#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
%matplotlib inline
dig = load_digits() #initialization
dir(dig) #returns the properties of an object without it's values
plt.gray() #display images in grayscale
#Loop through the top 5 rows of the images and display each of the images
for i in range(5): 
    plt.matshow(dig.images[i])
data_df = pd.DataFrame(dig.data) #convert data to a DataFrame 
data_df["Target"] = dig.target #add target from initial data to see the target number for each line of data in the data frame
data_df
#import scikit learn libraries for machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#assign independent variables to x
x = data_df.drop(["Target"], axis = "columns")
#assign dependent variable to y
y = dig.target
#split your data into train and test samples
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
#initialize randomforestclassifier and fit model
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
classifier.score(x_test,y_test)
y_pred = classifier.predict(x_test)
c_matrix = confusion_matrix(y_test,y_pred)
import seaborn as sb
#plot actual and predicted values in a confusion matrix to visualize how accurate the model predictions are
plt.subplots(figsize=(8,8))
ax = sb.heatmap(c_matrix, annot = True)
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Confusion Matrix")
accuracy_score(y_test,y_pred)
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

% matplotlib inline
df = pd.read_csv("../input/diabetes.csv")

df.shape
df.head()
df.tail()
# Check for null value

df.isnull().values.any()
def plot_corr(df,size=10):

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot

        Display : matrix of correlation b/w collumns

         Blue-Cyan-Yellow-Red-Darked  -> less to more correlated

         0--------->1

         expect a darked line running from top left to bottom right'''



    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns);

    plt.yticks(range(len(corr.columns)), corr.columns);
plot_corr(df)
num_1 = len(df.loc[df['Outcome'] ==1])

num_0 = len(df.loc[df['Outcome'] == 0])

print("Number of true case: {}".format(num_1))

print("Number of False case: {}".format(num_0))
from sklearn.model_selection import train_test_split

feature_col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',

       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

      

predicted_class_name = ['Outcome']

x = df[feature_col_names].values

y = df[predicted_class_name].values

split_test_size = 0.30

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=split_test_size,random_state=42)

# Skin Thickness can't be 0

df.head()
from sklearn.preprocessing import Imputer

fill_0 = Imputer (missing_values=0,strategy="mean",axis=0)

x_train = fill_0.fit_transform(x_train)

x_test = fill_0.fit_transform(x_test)
from sklearn.naive_bayes import GaussianNB

# Create Gaussian naive bayes model and trains it with data

nb_model = GaussianNB()

nb_model.fit(x_train,y_train.ravel())

# performance on training data

nb_predict_train = nb_model.predict(x_train)

# performance on test data

nb_predict_test = nb_model.predict(x_test)

# Import the performance metrics Library

from sklearn import metrics

print("Accuracy of Training data : {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train)))

print("Accuracy of Testing data: {0:.4f}".format(metrics.accuracy_score(y_test,nb_predict_test)))

print("Confusion matrix")

# Note the use for label for set 1 = True to upper left and 0=False for lower right

print("{}".format(metrics.confusion_matrix(y_test,nb_predict_test,labels=[1,0])))
print("Classification Report")

print(metrics.classification_report(y_test,nb_predict_test,labels=[1,0]))
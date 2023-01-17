import pandas as pd

pd.options.mode.chained_assignment = None



import plotly.express as px



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
# Importing the data

data = pd.read_csv('/kaggle/input/iris/Iris.csv')



# Showing the 5 first values from our dataset

data.head(5)
# Verifying the description of the dataset

data.describe()
# Creating a box plot 

fig_sepal_length = px.box(data, x='Species', y='SepalLengthCm', color='Species')



# Showing the box plot

fig_sepal_length.show()
# Creating a box plot

fig_sepal_width = px.box(data, x='Species', y='SepalWidthCm', color='Species')



# Showing the box plot

fig_sepal_width.show()
# Creating a scatter plot - Length x Width

fig_sepal_features = px.scatter(data, x='SepalLengthCm', y='SepalWidthCm', 

                                color='Species')



# Showing the scatter plot

fig_sepal_features.show()
# Creating a box plot

fig_petal_length = px.box(data, x='Species', y='PetalLengthCm', color='Species')



# Showing the box plot

fig_petal_length.show()
# Creating a box plot

fig_width_petal = px.box(data, x='Species', y='PetalWidthCm', color='Species')



# Showing the box plot

fig_width_petal.show()
# Creating a scatter plot - Length x Width

fig_petal_features = px.scatter(data, x='PetalLengthCm', y='PetalWidthCm', 

                                color='Species')



# Showing the scatter plot

fig_petal_features.show()
# Creating a DataFrame with the median of all features per species

df_median = data.groupby(['Species'], as_index=False)[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].median()



# Showing the DataFrame

df_median
# Defining the features

X = data.drop(['Species', 'Id'], axis=1)



# Defining the target

y = data['Species']
# Separating data in training and testing

#30% of the data will be for testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
# Creating a Naive Bayes model

nb_model = GaussianNB()



# Training the model

nb_model.fit(X_train, y_train)
# Making predictions

prediction = nb_model.predict(X_test)
# Defining model accuracy

accuracy =  accuracy_score(y_test, prediction)



# Showing the model accuracy

print("Model accuracy: %.2f%%" %(accuracy*100))
# Defining number of hits

num_hits = accuracy_score(y_test, prediction, normalize=False)



# Showing the number of predictions

print('Number of predictions:', prediction.size)



# Showing the number of hits

print('Number of hits:', num_hits)
# Creating a DataFram to show the comparasion

df_comparasion = X_test



# Creating column for the actual values

df_comparasion['Actual values'] = y_test



# Creating column for the predicted values

df_comparasion['Predicted values'] = prediction



# Showing the 10 first values from our comparasion

df_comparasion.head(10)
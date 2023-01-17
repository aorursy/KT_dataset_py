# Assigning features and label variables

# First Feature

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',

'Rainy','Sunny','Overcast','Overcast','Rainy']

# Second Feature

temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']



# Label or target varible

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
# Import LabelEncoder

from sklearn import preprocessing

#creating labelEncoder

le = preprocessing.LabelEncoder()

# Converting string labels into numbers.

weather_encoded=le.fit_transform(weather)

print(weather_encoded)
# converting string labels into numbers

temp_encoded=le.fit_transform(temp)

print(temp_encoded)

label=le.fit_transform(play)

print(label)
#combinig weather and temp into single listof tuples

features=list(zip(weather_encoded,temp_encoded))

print(features)
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=3)



# Train the model using the training sets

model.fit(features,label)



#Predict Output

predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild

print(predicted)
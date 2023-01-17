model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
model.summary()

import pandas as pd

#Make predictions using the features from the test data set
predictions = model.predict(test[features])

#Display our predictions - they are either 0 or 1 for each training instance 
#depending on whether our algorithm believes the person survived or not.
predictions

#Create a  DataFrame with the tree Ids and there Forest Cover Type
submission = pd.DataFrame({'Id':test['Cover_Type'],'Cover_Type':predictions})

#Visualize the first 5 rows
submission.head()


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Forest_Cover_Type_Prediction.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

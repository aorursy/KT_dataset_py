import pandas as pd 



# Read the dataset

df = pd.read_csv('../input/data.csv')

# Drop the 'id' column from the dataset as it is irrelevent

df.drop(['id'],axis=1,inplace=True)

# Drop the column contatining 'NaN' elements (Here the last column in the dataset will be dropped)

df.dropna(axis=1, inplace=True)

df.head()
from sklearn.preprocessing import LabelEncoder





# parameters 

X = df.drop(['diagnosis'],1)

#label

Y = df['diagnosis']

le = LabelEncoder()

Y = le.fit_transform(Y)



from sklearn import neighbors,model_selection



# Splitting the dataset into test_data and train_data

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y, test_size=0.2)



# Applying KNN_Classifier

model = neighbors.KNeighborsClassifier(n_neighbors=3)

model.fit(X_train,Y_train)



# Checking the accuracy on the Test set

accuracy = model.score(X_test,Y_test)

print('Accuracy on the test set:',accuracy)
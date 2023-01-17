import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



from scipy.stats import zscore



# To enable plotting graphs in Jupyter notebook

%matplotlib inline 

import seaborn as sns
NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )
import os

print(os.listdir('../input/anly-53053b/Lab3'))
bc_df = pd.read_csv("../input/anly-53053b/Lab3/wisc_bc_data.csv")

bc_df.head()
bc_df.shape
bc_df.dtypes
bc_df['diagnosis'] = bc_df.diagnosis.astype('category')

bc_df.dtypes
bc_df.describe().transpose()
bc_df.groupby(["diagnosis"]).count()



# Class distribution among B and M is almost 2:1. The model will better predict B and M
# The first column is id column which is patient id and nothing to do with the model attriibutes. So drop it.



bc_df = bc_df.drop(labels = "id", axis = 1)

bc_df.shape
# Create a separate dataframe consisting only of the features i.e independent attributes



bc_feature_df = bc_df.drop(labels= "diagnosis" , axis = 1)

bc_feature_df.head()
# convert the features into z scores as we do not know what units / scales were used and store them in new dataframe

# It is always adviced to scale numeric attributes in models that calculate distances.



bc_feature_df_z = bc_feature_df.apply(zscore)  # convert all attributes to Z scale 



bc_feature_df_z.describe()
# Capture the class values from the 'diagnosis' column into a pandas series akin to array 



bc_labels = bc_df["diagnosis"]
# store the normalized features data into np array 



X = np.array(bc_feature_df_z)

X.shape
# store the bc_labels data into a separate np array



y = np.array(bc_labels)

y.shape
# Split X and y into training and test set in 75:25 ratio



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
# Call Nearest Neighbour algorithm



NNH.fit(X_train, y_train)
# For every test data point, predict it's label based on 5 nearest neighbours in this model. The majority class will 

# be assigned to the test data point



predicted_labels = NNH.predict(X_test)

NNH.score(X_test, y_test)
# calculate accuracy measures and confusion matrix

from sklearn import metrics



print(metrics.confusion_matrix(y_test, predicted_labels))
# To improve performance ------------------------- Iteration 2 -----------------------------------

# Let us analyze the different attributes for distribution and the correlation by using scatter matrix



sns.pairplot(bc_df)
# As is evident from the scatter matrix, many dimensions have strong correlation and that is not surprising

# Area and Perimeter are function of radius, so they will have strong correlation. Why take multiple dimensions 

# when they convey the same information to the model?
# To to drop dependent columns from bc_df

#Since radius,perimeter,area are of same type/group

bc_features_pruned_df_z =  bc_feature_df_z.drop(['radius_mean'], axis=1)
X = np.array(bc_features_pruned_df_z)







X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
# Call Nearest Neighbour algorithm



NNH.fit(X_train, y_train)
# For every test data point, predict it's label based on 5 nearest neighbours in this model. The majority class will 

# be assigned to the test data point



predicted_labels = NNH.predict(X_test)
# get the accuracy score which is how many test cases were correctly predicted as a ratio of total number of test cases



NNH.score(X_test, y_test)
# calculate accuracy measures and confusion matrix

from sklearn import metrics



print(metrics.confusion_matrix(y_test, predicted_labels))
# peformance has dropped! So, be careful about the dimensions you drop. 

#Domain expertise is a must to know whether dropping radius or dropping area

#will be better. The area may be a stronger predictor than radius and the

#way they are calculated under a electron microscope may be effecting the outcome
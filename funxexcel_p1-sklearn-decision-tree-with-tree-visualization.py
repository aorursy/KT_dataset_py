import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

data.head()
#Get Target data 

y = data['target']



#Load X Variables into a Pandas Dataframe with columns 

X = data.drop(['target'], axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
DT_Model = DecisionTreeClassifier()
DT_Model.fit(X_train,y_train)
print (f'Train Accuracy - : {DT_Model.score(X_train,y_train):.3f}')

print (f'Test Accuracy - : {DT_Model.score(X_test,y_test):.3f}')
dot_data = export_graphviz(DT_Model, max_depth = 3,  #Limit to a Depth of 3 only

                      out_file=None, 

                      feature_names=X.columns,       #Provide X Variables Column Names 

                      class_names=['Yes','No'],          # Provide Target Variable Column Name

                      filled=True, rounded=True,     # Controls the look of the nodes and colours it

                      special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
dot_data = export_graphviz(DT_Model, out_file=None, 

                      feature_names=X.columns,  

                      class_names=['Yes','No'], 

                      filled=True, rounded=True,  

                      special_characters=True)  

graph = graphviz.Source(dot_data) 

graph.render("Heart_Diesease") 
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics



dataset = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

dataset.head()
# divido i dati relativi agli features (attributi) dalle classi (Outcome)

feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']

x = dataset[feature_cols] # features

y = dataset.Outcome # classi

# divido il dataset in training set e test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test
# creo il classificatore DecisionTreeClassifier

dtc = DecisionTreeClassifier()

# training del classificatore

dtc = dtc.fit(x_train,y_train)

# eseguo la predizione dei dati del test set

y_pred = dtc.predict(x_test)

# calcolo l'accuracy

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image 

!pip install pydotplus

import pydotplus



dot_data = StringIO()

export_graphviz(dtc, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True,feature_names = feature_cols,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('diabetes.png')

Image(graph.create_png())
# creo il classificatore DecisionTreeClassifier

dtc_optimized = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# training del classificatore

dtc_optimized = dtc_optimized.fit(x_train,y_train)

# eseguo la predizione dei dati del test set

y_pred = dtc_optimized.predict(x_test)

# calcolo l'accuracy per il classificatore ottimizzato

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


dot_data = StringIO()

export_graphviz(dtc_optimized, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True, feature_names = feature_cols,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('diabetes.png')

Image(graph.create_png())
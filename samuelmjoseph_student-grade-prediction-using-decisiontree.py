import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
data = pd.read_csv('../input/student-mat.csv')
data.shape
data.isnull().values.any()
data.columns
data.head()
data.nunique()
data['GAvg'] = (data['G1'] + data['G2'] + data['G3']) / 3
def define_grade(df):
    # Create a list to store the data
    grades = []

    # For each row in the column,
    for row in df['GAvg']:
        # if more than a value,
        if row >= (0.9 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('A')
        # else, if more than a value,
        elif row >= (0.7 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('B')
        # else, if more than a value,
        elif row < (0.7 * df['GAvg'].max()):
            # Append a letter grade
            grades.append('C')   
    # Create a column from the list
    df['grades'] = grades
    return df
data = define_grade(data)
data.head()
data.drop(["school","age"], axis=1, inplace=True)
data.describe()
# for yes / no values:
d = {'yes': 1, 'no': 0}
data['schoolsup'] = data['schoolsup'].map(d)
data['famsup'] = data['famsup'].map(d)
data['paid'] = data['paid'].map(d)
data['activities'] = data['activities'].map(d)
data['nursery'] = data['nursery'].map(d)
data['higher'] = data['higher'].map(d)
data['internet'] = data['internet'].map(d)
data['romantic'] = data['romantic'].map(d)
# map the sex data
d = {'F': 1, 'M': 0}
data['sex'] = data['sex'].map(d)

# map the address data
d = {'U': 1, 'R': 0}
data['address'] = data['address'].map(d)

# map the famili size data
d = {'LE3': 1, 'GT3': 0}
data['famsize'] = data['famsize'].map(d)

# map the parent's status
d = {'T': 1, 'A': 0}
data['Pstatus'] = data['Pstatus'].map(d)

# map the parent's job
d = {'teacher': 0, 'health': 1, 'services': 2,'at_home': 3,'other': 4}
data['Mjob'] = data['Mjob'].map(d)
data['Fjob'] = data['Fjob'].map(d)

# map the reason data
d = {'home': 0, 'reputation': 1, 'course': 2,'other': 3}
data['reason'] = data['reason'].map(d)

# map the guardian data
d = {'mother': 0, 'father': 1, 'other': 2}
data['guardian'] = data['guardian'].map(d)

# map the grades data
d = {'C': 0, 'B': 1, 'A': 2}
data['grades'] = data['grades'].map(d)
data.nunique()

student_features = data.columns.tolist()
student_features.remove('grades') 
student_features.remove('GAvg') 
student_features.remove('G1') 
student_features.remove('G2') 
student_features.remove('G3') 
student_features
X = data[student_features].copy()
X.columns
y=data[['grades']].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)
grade_classifier = tree.DecisionTreeClassifier(max_leaf_nodes=len(X.columns), random_state=0)
grade_classifier.fit(X_train, y_train)
dot_data = StringIO()  
tree.export_graphviz(grade_classifier, out_file=dot_data,  
                         feature_names=student_features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())  
predictions = grade_classifier.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)

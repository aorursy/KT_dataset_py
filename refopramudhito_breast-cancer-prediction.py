import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns  # visualization tool
import itertools
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore') #ignore warning messages 


plt.style.use('ggplot')
df = pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')
X_train, X_test, y_train, y_test = train_test_split(df.drop(['diagnosis'],axis=1),df['diagnosis'], test_size=0.10, random_state=101)


# Visualization
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
#correlation
correlation = df.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)

#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   xgap = 2,
                   ygap = 2,
                   colorscale='Viridis',
                   colorbar   = dict() ,
                  )
layout = go.Layout(dict(title = 'Correlation Matrix for variables',
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                     ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9)),
                       )
                  )
fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)

# Decision Tree Classification
dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
print("Confussion Matrix : ",confusion_matrix(y_test,dt_pred))
print ("")
print(classification_report(y_test,dt_pred))
print ("")
print (dt_model)

# Random Forest Classification
rf= RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)
print("Confussion Matrix : ", confusion_matrix(y_test,rf_pre))
print("")
print(classification_report(y_test,rf_pre))
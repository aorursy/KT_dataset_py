import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
# data processing
df['average_score']=0

for index,row in df.iterrows():
    avg_score= row[-4:-1].mean()
    if avg_score >=60:
        avg_score='Pass'
    elif avg_score <60 :
        avg_score='fail'

    df.iloc[index:,-1]=avg_score

df_n_score_word = df.drop(['math score','reading score','writing score','race/ethnicity'],axis=1)  
df_n_score_word.head()
# Combine the parental education (optional)
count=0
for col in df_n_score_word.iloc[:,1]:
    if(col == "bachelor's degree" or  col== "associate's degree" ):
        df_n_score_word.iloc[count,1]= 'Higher education'
#     elif (col =='high school' or col =='some high school'):
#         df_n_score_word.iloc[count,1] = 'Secondary education'

    count+=1  
df_n_score_word.head()
df_n_score=pd.get_dummies(df_n_score_word,drop_first=True)
df_n_score.head()
X=df_n_score.iloc[:,:-1]
y=df_n_score.iloc[:,-1:]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

param_dist = {'criterion':('gini', 'entropy'), 
#               "max_features": [2,3,4,5],
              'min_samples_split':[2,3,4,5], 
              'max_depth':[8,9,10,11],
              'class_weight':('balanced', None),
              'presort':(False,True),
             }


tree_d  = DecisionTreeClassifier()

tree_cv = GridSearchCV(tree_d,param_dist,cv=10)
# tree_cv = RandomizedSearchCV(tree_d, param_dist, cv=10)

tree_cv.fit(X_train, y_train)

print('R square : ',tree_cv.score(X_test,y_test))
cv_scores=cross_val_score(tree_cv,X,y,cv=5)
print('Cross validation score : ',cv_scores)
y_pred =tree_cv.predict(X_test)
print(classification_report(y_test,y_pred))
# from sklearn.externals.six import StringIO  
# # import pydotplus
# from IPython.display import Image
# dot_data = StringIO()

# export_graphviz(tree_cv.best_estimator_, out_file=dot_data, 
#                 feature_names=df_n_score.columns[:-1],
#                 class_names=df_n_score.columns,
#                 filled=True, rounded=True,
#                 special_characters=True)

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

# Image(graph.create_png())
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test,y_pred)
print("AUC: {}".format(auc))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr,label='AUC = %0.2f' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
# Random forest
from sklearn.ensemble import RandomForestClassifier
param_grid = {
                 'n_estimators': [5, 10, 15, 20],
                 'max_depth': [2, 5, 7, 9],
#                  'class_weight':"balanced"
             }
model = RandomForestClassifier()
random_model = GridSearchCV(model,param_grid, cv=10)
random_model.fit(X_train,y_train)
ypred = random_model.predict(X_test)

print('R square : ',random_model.score(X_test,y_test))

print(classification_report(ypred, y_test))
cv_R_scores=cross_val_score(random_model,X,y,cv=5)
print('Cross validation score : ',cv_R_scores.mean())
fpr, tpr, thresholds = roc_curve(y_test, ypred)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr,label='AUC = %0.2f' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
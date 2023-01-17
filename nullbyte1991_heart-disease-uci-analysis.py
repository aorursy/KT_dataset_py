# Common Packages
import pandas as pd
import numpy as np

# plot and visulization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.offline import iplot

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

print("Number of rows in data : {}".format(df.shape[0]))
print("Number of columns in data :{}".format(df.shape[1]))
df.head()
df.info()
# Statistical properties
df.describe().round(3)
# Change the names of the columns for better understanding
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

df.columns
df.head()
plt.rcParams['figure.figsize'] = (10, 10)
plt.style.use('ggplot')

sns.heatmap(df.corr(), annot = True, cmap = 'rocket_r')
plt.title('Heatmap for the Dataset', fontsize = 20)
plt.show()
print(f"Minimum Age : {min(df.age)} years")
print(f"Maximum Age : {max(df.age)} years")

hist_data = [df['age']]
group_labels = ['age'] 

colors = ['#835AF1']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels=group_labels, colors=colors,
                         bin_size=10, show_rug=False)

# Add title
fig.update_layout(width=700, title_text='Age Distribution')
fig.show()
categorical_cols = ['sex','chest_pain_type','fasting_blood_sugar','rest_ecg','exercise_induced_angina','st_slope','num_major_vessels','thalassemia']
numeric_cols = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']
multi_label_cols = [i for i in categorical_cols if df[i].nunique()>2]
multi_label_cols
# Lets normalize the numerical_cols
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
df[numeric_cols] = std.fit_transform(df[numeric_cols])
df.head()
# Catgorical Encoding
df = pd.get_dummies(data = df,columns = multi_label_cols)
df.head()
x = df.drop(['target'],axis=1) 
y = df['target']
from sklearn.model_selection import train_test_split

# Train = 70 % Test= 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
print("Shape of x_train : {}".format(x_train.shape))
print("Shape of x_test : {}".format(x_test.shape))
print("Shape of y_train :{}".format(y_train.shape))
print("Shape of y_test :{}".format(y_test.shape))
lr = LogisticRegression()

# Model fit
lr.fit(x_train, y_train)

y_pred_prob = lr.predict_proba(x_test)[:, 1]
y_pred = lr.predict(x_test)


# AUC Score
auc = roc_auc_score(y_test, y_pred_prob)

# evaluating the model
print("Training Accuracy :{}".format(lr.score(x_train, y_train)))
print("Testing Accuracy :{}".format(lr.score(x_test, y_test)))
print("AUC Score :{}".format(auc))
# cofusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')

# classification report
cr = classification_report(y_test, y_pred)
print(cr)
total=sum(sum(cm))

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,1]+cm[0,1])
print('Specificity : ', specificity)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.rcParams['figure.figsize'] = (15, 5)
plt.title('ROC curve for diabetes classifier', fontweight = 30)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(x_train, y_train)
y_pred_prob = rf.predict_proba(x_test)[:, 1]
y_pred = rf.predict(x_test)

# AUC Score
auc = roc_auc_score(y_test, y_pred_prob)

# evaluating the model
print("Training Accuracy :{}".format(rf.score(x_train, y_train)))
print("Testing Accuracy :{}".format(rf.score(x_test, y_test)))
print("AUC Score :{}".format(auc))
# cofusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')

# classification report
cr = classification_report(y_test, y_pred)
print(cr)
total=sum(sum(cm))

sensitivity = cm[0,0]/(cm[0,0]+cm[1,0])
print('Sensitivity : ', sensitivity )

specificity = cm[1,1]/(cm[1,1]+cm[0,1])
print('Specificity : ', specificity)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.rcParams['figure.figsize'] = (15, 5)
plt.title('ROC curve for diabetes classifier', fontweight = 30)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
# Random forest model
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf, random_state=1).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = x_test.columns.tolist())
import pickle

# Save a model to  pickle file
Pkl_Filename = "Pickle_lr_Model.pkl"
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(lr, file)
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model
# Use the Reloaded Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = Pickled_LR_Model.score(x_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
Ypredict = Pickled_LR_Model.predict(x_test)  

Ypredict

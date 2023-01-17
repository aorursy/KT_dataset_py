# Importing the needed libraries
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

# Importing libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# Importing the machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# This is for sampling unbalanced data
from imblearn.under_sampling import RandomUnderSampler

# Importing the counter
from collections import Counter
data = pd.read_csv('../input/Accidents_2016.csv', low_memory=False)
data.head()
data.info()
data.describe()
names = ["Accident_Index","Location_Easting","Location_Northing",
         "Longitude","Latitude","Police_Force","Accident_Severity","No_of_Vehicles",
         "No_of_Casualties","Date","Day_of_Week","Time","Local_Authority_District",
         "Local_Authority_Highway","First_Road_Class","First_Road_No","Road_Type","Speed_Limit",
         "Junction_Detail","Junction_Control","Second_Road_Class","Second_Road_Number",
         "Pedestrian_Crossing_Control","Pedestrian_Crossing_Facilities",
         "Light_Conditions","Weather_Conditions","Road_Surface_Conditions","Special_Conditions_at_Site",
         "Carriageway_Hazards","Urban_or_Rural_Area","Police_Attendance","Accident_Location"]
data.columns = names
fig, ax = plt.subplots()
fig.set_size_inches(14, 10)
ax =sns.heatmap(data.corr())
data_no_na = data.dropna()
data_no_na['Month']=data_no_na['Date'].apply(lambda x: x.split("/")[1])
data_no_na['Hour']=data_no_na['Time'].apply(lambda x: int(x.split(":")[0]))
data_no_na['Accident_Location'] = data_no_na['Accident_Location'].astype('category')
data_no_na['Accident_Location_Cat'] = data_no_na['Accident_Location'].cat.codes

data_no_na['Local_Authority_Highway'] = data_no_na['Local_Authority_Highway'].astype('category')
data_no_na['Local_Authority_Highway_Cat'] = data_no_na['Local_Authority_Highway'].cat.codes

data_no_na['Police_Attendance']= data_no_na['Police_Attendance'].apply(lambda x: 1 if x==1 else 0)
features_minimal = data_no_na[[ 'Location_Easting', 'Location_Northing', 'Police_Force', 'Accident_Severity', 'No_of_Vehicles',
       'No_of_Casualties', 'Day_of_Week','Local_Authority_District',
       'First_Road_Class', 'First_Road_No', 'Road_Type', 'Speed_Limit',
       'Junction_Detail', 'Junction_Control', 'Second_Road_Class',
       'Second_Road_Number','Weather_Conditions', 'Road_Surface_Conditions',
       'Special_Conditions_at_Site', 'Carriageway_Hazards',
       'Urban_or_Rural_Area','Month', 'Hour','Accident_Location_Cat','Local_Authority_Highway_Cat']]
target = data_no_na['Police_Attendance']
def plot_confusion_matrix(y_test, y_pred,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
   

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_side_by_side_confusion_matrix(y_test, y_pred):
    """
    Plots the confusion matrix
    """
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plot_confusion_matrix(y_test, y_pred, [0,1], normalize=False, title='Confusion Matrix')
    plt.subplot(122)
    plot_confusion_matrix(y_test, y_pred, [0,1], normalize=True, title='Normalized Confusion Matrix')
    
def plot_roc_curve(y_test, probs):
    """
    Plots the ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, lw=1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

def plot_precision_recall_curve(y_test, probs):
    """
    Plots the Precision-Recall Curve
    """
    precision, recall, _ = precision_recall_curve(y_test, probs)
    average_precision = average_precision_score(y_test, probs)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
lin_svc_X_train, lin_svc_X_test, lin_svc_y_train, lin_svc_y_test = train_test_split(features_minimal, target, test_size=0.4, random_state=101)
lin_svc_model = svm.LinearSVC()
lin_svc_model.fit(lin_svc_X_train,lin_svc_y_train)
lin_svc_predictions = lin_svc_model.predict(lin_svc_X_test)
print(confusion_matrix(lin_svc_y_test,lin_svc_predictions))
print(classification_report(lin_svc_y_test,lin_svc_predictions))
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(features_minimal, target)
print(sorted(Counter(y_resampled).items()))
balanced_X_train, balanced_X_test, balanced_y_train, balanced_y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=101)
lin_svc_balanced_model = svm.LinearSVC()
lin_svc_balanced_model.fit(balanced_X_train,balanced_y_train)
lin_svc_balanced_predictions = lin_svc_balanced_model.predict(balanced_X_test)
print(confusion_matrix(balanced_y_test,lin_svc_balanced_predictions))
print(classification_report(balanced_y_test,lin_svc_balanced_predictions))
param_grid = {'C':[1,10,100,1000]}
grid = GridSearchCV(svm.LinearSVC(),param_grid,verbose=3)
grid.fit(lin_svc_X_train,lin_svc_y_train)
grid.best_params_
grid_predictions = grid.predict(lin_svc_X_test)
print(confusion_matrix(lin_svc_y_test, grid_predictions))
print(classification_report(lin_svc_y_test, grid_predictions))
scaler = StandardScaler()
scaler.fit(features_minimal)
scaled_features  = scaler.transform(features_minimal)
knn_features = pd.DataFrame(scaled_features, columns=features_minimal.columns)
knn_features.head()
knn_X_train, knn_X_test, knn_y_train, knn_y_test = train_test_split(knn_features, target, test_size=0.4, random_state=101)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(knn_X_train,knn_y_train)
knn_predictions = knn.predict(knn_X_test)
print(confusion_matrix(knn_y_test,knn_predictions))
print(classification_report(knn_y_test,knn_predictions))
test_error_rate = list()

for i in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(knn_X_train,knn_y_train)
    knn_pred_i = knn.predict(knn_X_test)
    test_error_rate.append(np.mean(knn_pred_i != knn_y_test))
plt.figure(figsize=(7,4))
plt.plot(range(1,20),test_error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=5)
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs. K Value')
knn9 = KNeighborsClassifier(n_neighbors=9)
knn9.fit(knn_X_train,knn_y_train)
knn9_predictions = knn9.predict(knn_X_test)
print(confusion_matrix(knn_y_test,knn9_predictions))
print(classification_report(knn_y_test,knn9_predictions))
knn9_predictions_probs = knn9.predict_proba(knn_X_test)
plot_side_by_side_confusion_matrix(knn_y_test,knn9_predictions)
plot_roc_curve(knn_y_test, knn9_predictions_probs[:,1])
plot_precision_recall_curve(knn_y_test, knn9_predictions_probs[:,1])
#Import packages
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, log_loss, jaccard_similarity_score
from  sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.font_manager as font_manager
from matplotlib import rcParams
%matplotlib inline
warnings.filterwarnings("ignore")
print(os.listdir("../input"))
data = pd.read_csv("../input/loan_payments_data.csv")
data.head()
data.isnull().sum()
print('Percent of missing "paid_off_time" records is %.2f%%' %((data['paid_off_time'].isnull().sum()/data.shape[0])*100))
print('Percent of missing "past_due_days" records is %.2f%%' %((data['past_due_days'].isnull().sum()/data.shape[0])*100))
data['Principal'].unique()
data['loan_status'].unique()
data.info()
data_preview = data.copy()
label_unique = data['loan_status'].unique()
label_occurance_count = data_preview.groupby('loan_status').size()
plt.pie(label_occurance_count, labels = label_occurance_count,counterclock=False, shadow=True, radius = 2, autopct='%1.1f%%', labeldistance = 1.1)
plt.title('loan status types as percentage in a graph view', y=1.5, bbox={'facecolor':'#EBF1DE', 'pad':18})
plt.legend(label_unique,loc="top right", bbox_to_anchor=(1.36,1.26))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75)
plt.show()
new_data = data.copy()
le=preprocessing.LabelEncoder()
data['loan_status']=le.fit_transform(data['loan_status'])
data['Gender']=le.fit_transform(data['Gender'])
data['education']=le.fit_transform(data['education'])
data['past_due_days']=le.fit_transform(data['past_due_days'])
data.head()
data.info()
new_data.groupby('loan_status')['Gender'].agg(['count'])
sns.barplot(x="Gender", y="loan_status", hue="education", data=data);
data['past_due_days'].unique()
x=data.groupby('Gender')['past_due_days'].agg(['sum'])
x=pd.DataFrame(x)
x
sns.barplot(x='Gender',y='loan_status',data=data)
sns.barplot(x='age',y='loan_status',data=data)
sns.factorplot(x='age',y='loan_status',data=data)
sns.barplot(x='education',y='loan_status',data=data)
sns.countplot(x='Gender',data=data)
processed_analysis_data=data
label = processed_analysis_data.pop('loan_status')
processed_analysis_data.drop('Loan_ID', axis=1, inplace=True)
processed_analysis_data.drop('effective_date', axis=1, inplace=True)
processed_analysis_data.drop('due_date', axis=1, inplace=True)
processed_analysis_data.drop('paid_off_time', axis=1, inplace=True)
processed_analysis_data.head(5)
data_train, data_test, label_train, label_test = train_test_split(processed_analysis_data, label, test_size = 0.2, random_state = 42)
classifiers={
    "K_Neighbors_Classifier": KNeighborsClassifier(), 
    "Decision_Tree_Classifier": DecisionTreeClassifier(random_state = 42),
    "support_vector_machine": svm.LinearSVC(random_state = 42), 
    "Logistic_Regression": LogisticRegression(random_state = 42)
            }
metrics = pd.DataFrame(index = 
      ['jaccard_index', 'f1_score', 'log_loss', 'accuracy', 'cross_validation_score'], 
                       columns = 
     ["K_Neighbors_Classifier", "Decision_Tree_Classifier", "support_vector_machine", "Logistic_Regression"])

for i, (clf_name, clf) in enumerate(classifiers.items()):

    if clf_name == "K_Neighbors_Classifier":
        clf.fit(data_train, label_train)
        y_pred = clf.predict(data_test)
        
    elif clf_name == "Decision_Tree_Classifier":
        clf.fit(data_train, label_train)
        y_pred = clf.predict(data_test)
        
    elif clf_name == "support_vector_machine":
        clf.fit(data_train, label_train)
        y_pred = clf.predict(data_test)
        
    else: 
        clf.fit(data_train, label_train)
        y_pred = clf.predict(data_test)
    
    n_errors = (y_pred != label_test).sum()

    print("\n\n\n")
    
    # error in prediction
    print('{} {} {}: {}'.format("error in " ,clf_name, "prediction", n_errors))
    
    print("\n\n")
    
    # accuracy score in prediction
    print("accuracy score in prediction: ")
    print(accuracy_score(label_test, y_pred))
    
    
    # model accuracy
    print("model accuracy: ")
    print(clf.score(data_test, label_test))
    
    
    # cross validation score
    print("cross validation score: ")
    cross_val = cross_val_score(clf, processed_analysis_data, label, scoring='accuracy', cv=10)
    print(cross_val)
    print("cross_validation_score.mean(): ")
    print(cross_val.mean())
    
    print("\n\n")
    
    # confusion_matrix
    print("confusion matrix: ")
    print(confusion_matrix(label_test, y_pred))
    
    print("\n\n")
    
    print("f1_score by average as macro : {}".format(f1_score(label_test, y_pred, average='macro')))

    print("f1_score by average as micro : {}".format(f1_score(label_test, y_pred, average='micro')))

    print("f1_score by average as weighted : {}".format(f1_score(label_test, y_pred, average='weighted')))

    print("f1_score by average as None : {}".format(f1_score(label_test, y_pred, average=None)))

    print("\n\n")
    
    print("jaccard similarity score 1 : {}".format(jaccard_similarity_score(label_test, y_pred)))

    print("jaccard similarity score 2 : {}".format(jaccard_similarity_score(label_test, y_pred, normalize=False)))
    
    print("\n\n")
        
    if(clf_name == "support_vector_machine"):
        pred = clf.decision_function(data_test)
        
    else: 
        pred = clf.predict_proba(data_test)
        
    print("log loss : {}".format(log_loss(label_test, pred)))
    
    print("\n\n")
    
    print("precision score with average value as micro: {}".format(precision_score(label_test, y_pred, average="micro")))
    print("precision score with average value as macro: {}".format(precision_score(label_test, y_pred, average="macro")))
    print("precision score with average value as weighted: {}".format(precision_score(label_test, y_pred, average="weighted")))
    
    print("\n\n")
    
    print("recall score with average value as micro: {}".format(recall_score(label_test, y_pred, average="micro")))
    print("recall score with average value as macro: {}".format(recall_score(label_test, y_pred, average="macro")))
    print("recall score with average value as weighted: {}".format(recall_score(label_test, y_pred, average="weighted")))
    
    print("\n\n")
    
    # classification report
    print("classification report: ")
    print(classification_report(label_test, y_pred)) 
    
    print("\n\n\n")
    
    print(clf_name + " classifier section end. ")
    print("\n\n--------------------------------------------------------- \n\n")
    print("\n \n")
      
    metrics.loc['jaccard_index', clf_name] = jaccard_similarity_score(label_test, y_pred)
    metrics.loc['f1_score', clf_name] = f1_score(label_test, y_pred, average=None).mean()
    metrics.loc['log_loss', clf_name] = log_loss(label_test, pred)
    metrics.loc['accuracy', clf_name] = accuracy_score(label_test, y_pred)
    metrics.loc['cross_validation_score', clf_name] = cross_val_score(clf, processed_analysis_data, label, scoring='accuracy', cv=10).mean()
    
   
metrics_in_percentage = 100*metrics 
metrics_in_percentage
metrics = metrics.convert_objects(convert_numeric=True)   

metrics_in_percentage = (100*metrics)
font = font_manager.FontProperties(family='Lucida Fax', size=46)
rcParams['font.family'] = 'Britannic Bold'
fig, ax = plt.subplots(figsize = (35, 25)) 
metrics_in_percentage.plot(kind = 'barh', ax = ax, fontsize = 80) 
#plt.rcParams["font.family"] = "Monotype Corsiva"

title_font = {'fontname':'Monotype Corsiva'}
#legend_font = {'fontname':'Impact'}

plt.title(s = "classification applied on loan application dataset", color = "green", fontsize = 120, loc = "center", fontweight = "bold", **title_font)
legend = ax.legend()


legend = ax.legend(loc = "upper right", labelspacing=2, borderpad=0.45, prop=font)
legend
frame = legend.get_frame()
frame.set_facecolor("#EBF1DE")
frame.set_edgecolor('chartreuse')
frame.set_linewidth(10)
ax.margins(0.6)
ax.grid()
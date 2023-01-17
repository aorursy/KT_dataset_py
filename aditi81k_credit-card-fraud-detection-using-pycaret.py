import numpy as np # linear algebra
import pandas as pd 
!pip3 install pycaret
train = pd.read_csv("../input/creditcardfraud/creditcard.csv")
train.head()
# to print the full summary
train.info()
from pycaret.classification import *
classification_setup = setup(data=train,target='Class', normalize=True, session_id=42)
compare_models(sort='Recall')
qda_cls = create_model('qda')
tune_qda = tune_model(qda_cls, optimize = 'Recall', choose_better = True)
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(train['Class'])
plt.show()
print('Percent of fraud transaction: ',len(train[train['Class']==1])/len(train['Class'])*100,"%")
print('Percent of normal transaction: ',len(train[train['Class']==0])/len(train['Class'])*100,"%")
classification_setup1 = setup(data=train,target='Class', normalize=True, session_id=42, fix_imbalance = True)
compare_models(sort='Recall', blacklist = ['gbc'])
svm_cls = create_model('svm')
tune_svm = tune_model(svm_cls, optimize = 'Recall', choose_better = True)
# plotting a model
plot_model(tune_svm,plot = 'confusion_matrix')
# error Curve
plot_model(tune_svm, plot = 'error')
# Precision Recall Curve
plot_model(tune_svm, plot = 'pr')
# Classification Report Curve
plot_model(tune_svm, plot = 'class_report')
y_pred = predict_model(tune_svm)
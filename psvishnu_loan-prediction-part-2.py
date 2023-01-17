import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

from warnings import filterwarnings

filterwarnings('ignore')



from sklearn.utils import resample

from sklearn.metrics import classification_report,confusion_matrix,f1_score

from sklearn import tree as t

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

import graphviz
!head '/kaggle/input/loan-prediction-part-1/test_clean.csv'
%%time

data = pd.read_csv('/kaggle/input/loan-prediction-part-1/train_clean.csv')

submission = pd.read_csv('/kaggle/input/loan-prediction-part-1/test_clean.csv')
"{:.2f} MB on disk".format(data.memory_usage().sum().sum() / 1024**2)
data.info()
# Credit: https://www.kaggle.com/wkirgsn/fail-safe-parallel-memory-reduction

from fail_safe_parallel_memory_reduction import Reducer
%%time

data = Reducer().reduce(data)
data.info()
plt.figure(figsize=(12,4))

plt.subplot(121)

sns.barplot(data.Loan_Status.value_counts(),y=['Approved (1)','Rejected (0)'])

plt.subplot(122)

data.Loan_Status.value_counts(normalize=True).plot.pie(autopct="%.2f%%")

plt.show()
#drop ID columns

data.drop(columns=['Loan_ID'],inplace=True)
X,y = data.drop(columns=['Loan_Status']), data['Loan_Status']

print(X.shape,y.shape,sep='\n') 
# Use a stratified train-test split

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,test_size=0.2)



print("Train distribution",y_train.value_counts(normalize=True),sep="\n")



print("Test distribution",y_test.value_counts(normalize=True),sep="\n")
y_train.value_counts()
# Use Oversampling to upscale the train data

sm = SMOTE()

X_s,y_s =sm.fit_sample(X_train,y_train)
pd.Series(y_s).value_counts()
print("In binary classification, as per sklearn confusion_matrix function","the count of true negatives is C00","false negatives is C10",

      "true positives is C11","and false positives is C01.",sep="\n")

def classification_metrics(y_actual,y_predict):

    # get performance metrics

    print(classification_report(y_actual,y_predict))



    # confusion matrix

    sns.heatmap(confusion_matrix(y_test,y_predict),cmap='coolwarm',annot=True)

    plt.show()

    

# https://www.kaggle.com/paultimothymooney/decision-trees-for-binary-classification-0-99 

def plot_decision_tree(a,b):

    dot_data = t.export_graphviz(a, out_file=None, 

                             feature_names=b,  

                            # 0,1

                             class_names=['Ineligible','Eligible'],  

                             filled=False, rounded=True,  

                             special_characters=False)  

    graph = graphviz.Source(dot_data)  

    return graph 
# fit model

dt = DecisionTreeClassifier(random_state=5)

tree = dt.fit(X_s,y_s)



classification_metrics(y_test,tree.predict(X_test))
f1_score(y_test,tree.predict(X_test))
# Visualize the tree classifier

plot_decision_tree(tree,X.columns)
# fit model

tree = DecisionTreeClassifier(class_weight='balanced',random_state=5).fit(X_train,y_train)



# get performance metrics

print(classification_report(y_test,tree.predict(X_test)))



# confusion matrix

sns.heatmap(confusion_matrix(y_test,tree.predict(X_test)),cmap='coolwarm',annot=True)

plt.show()
# It shows the f1-score for Loan Approvals

f1_score(y_test,tree.predict(X_test))
#random forest with original features

rfc = RandomForestClassifier(random_state=5,class_weight="balanced").fit(X_train,y_train)



classification_metrics(y_test,rfc.predict(X_test))

pd.DataFrame(rfc.feature_importances_,index=X_test.columns,columns=['importance']).sort_values(by='importance').plot.barh()

plt.show()
X_train1,X_test1 = X_train[::],X_test[::]



X_train1['Total Income'] = X_train['ApplicantIncome'] + X_train['CoapplicantIncome']

# X_train1['Credibility'] = X_train['LoanAmount'] / X_train1['Total Income']

X_test1['Total Income'] = X_test['ApplicantIncome'] + X_test['CoapplicantIncome']

# X_test1['Credibility'] = X_test['LoanAmount'] / X_test1['Total Income']



X_train1.drop(columns=['ApplicantIncome','CoapplicantIncome'],inplace=True)

X_test1.drop(columns=['ApplicantIncome','CoapplicantIncome'],inplace=True)



#random forest with new feature

rfc = RandomForestClassifier(random_state=5).fit(X_train1,y_train)



classification_metrics(y_test,rfc.predict(X_test1))



pd.DataFrame(rfc.feature_importances_,index=X_test1.columns,columns=['importance']).sort_values(by='importance').plot.barh()

plt.show()
# Credit: https://www.kaggle.com/mlwhiz/how-to-use-hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
def objective(space):

    

    rfc = RandomForestClassifier(

        random_state=5,

        class_weight="balanced",

        max_depth = space['max_depth']

    ).fit(X_train,y_train)



    f1 = f1_score(y_test,rfc.predict(X_test))

    

    # return needs to be in this below format. 

    # We use negative of accuracy since we want to maximize it.

    return {'loss': -f1, 'status': STATUS_OK }
# Using single parameter to show its use

space = {

    'max_depth': hp.quniform("x_max_depth", 4, 16, 1),

    }
%%timw

trials = Trials()

best = fmin(fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=100,

            trials=trials)

print(best)
rfc = RandomForestClassifier(

    random_state=5,

    class_weight="balanced",

    max_depth = 9

).fit(X_train,y_train)



prediction = rfc.predict(X_test)
submission.drop(columns = ['Unnamed: 0','Loan_ID'], inplace=True)

predictions = rfc.predict(submission)
submission['predictions'] = predictions
submission[["predictions"]].sample(10).style.background_gradient(cmap='Accent_r')
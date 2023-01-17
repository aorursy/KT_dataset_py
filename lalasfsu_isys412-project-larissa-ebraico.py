#load essential libraries
#essential Data Science
import numpy as np
import pandas as pd 
import math #for ln transformation
from sklearn import preprocessing # for feature engineering + selection
from sklearn import model_selection, linear_model, metrics #for analytics
import itertools #for confusion matrix
import matplotlib.pyplot as plt #visualization
%pylab inline
#load the train.csv file into Python
titanic = pd.read_csv("../input/train.csv")
#look at the datatable, the first 5
titanic.head()
#Review the actual metadata in Python
titanic.info()
#provide summary statistics for numerical variables
titanic.describe().T
#provide summary statistics for categorical variables
titanic.describe(include=["O"]).T
#save a copy of the original
titanic_org = titanic.copy()
titanic["Age"].mean()
titanic["AgeFilled"] = titanic["Age"]
titanic["AgeFilled"] = titanic["AgeFilled"].fillna(titanic["Age"].mean())

#titanic["AgeFilled"] = titanic["Age"].fillna(titanic["Age"].mean())

titanic["Fare"]= titanic["Fare"]
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic['Married'] = titanic['Name'].str.contains("\(")
titanic = titanic.loc[:,['Survived', 'Sex', 'SibSp',
       'Parch', 'Embarked', 'AgeFilled', 'Pclass', 'Fare', 'Married'
       ]]
#check to see the columns
titanic.columns

titanicDummy = pd.get_dummies(titanic,columns=['Sex',"Embarked", 'Pclass'],drop_first=False) #k-1
titanicDummy.head()
#split train and test inside the Train dataset to see how well we do inside train
#because Kaggle called the submission set "test", let's use validatedata as a variable to avoid confusion
traindata, validatedata = model_selection.train_test_split(titanicDummy,train_size=0.8,random_state = 42, shuffle=True)


#now we need to split between DV and IV
y_train = traindata["Survived"]
X_train = traindata.drop("Survived",axis = 1)
#now we need to split between DV and IV
y_validate = validatedata["Survived"]
X_validate = validatedata.drop("Survived",axis = 1)
logregression = linear_model.LogisticRegression(random_state=25) #first define the regression model
logreg = logregression.fit(X_train,y_train)
y_pred = logreg.predict(X_validate)
logreg.coef_
for i in range(len(X_validate.columns)):
    print("Variable: ", X_validate.columns[i],"\t", " coefficients: ", logreg.coef_[0][i])
    
print(metrics.classification_report(y_validate, y_pred))
AUCscore = metrics.roc_auc_score(y_validate, y_pred)
fpr,tpr, threshold = metrics.roc_curve(y_validate, y_pred) #three arrays: false positive rate, true positive rate, and thresholds
#graph ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % AUCscore)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



# Compute confusion matrix
cnf_matrix = metrics.confusion_matrix(y_validate, y_pred)
np.set_printoptions(precision=2)
class_names = y_validate.unique()
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
#X_submit will be the test.csv dataset. You can call test data set as well, but I call this subnit to avoid confusion
X_submit = pd.read_csv("../input/test.csv")
X_submit.head()
#look at null in dataset, Age has null, now Fare has null, and Cabin has null, not embarked, so we need to deal with the Fare

X_submit.isna().sum()
X_submit["Embarked"].value_counts() #check to see if there is any surprises on Embarked
X_submit["Sex"].value_counts() #check to see if there is any surprises on Sex
#prior to manipulation, always good to keep a copy
X_submit_org = X_submit.copy()
#first manipulation, replace Age with mean (your pick, of test or of train)
#I choose mean Age in train (original dataset s)
X_submit["AgeFilled"] = X_submit["Age"]
X_submit["AgeFilled"] = X_submit["AgeFilled"].fillna(titanic_org["Age"].mean())
#second manipulation, replace Fare with the mean. 
#Note that Embarked as no null, so we changed our plan a little bit
X_submit["Fare"] = X_submit["Fare"].fillna(X_submit["Fare"].mean())
X_submit['Married'] = X_submit['Name'].str.contains("\(")
### third manipulation, select only Sex, SipSp, Parch, Embarked, AgeFilled, Pclass and Married  as X_train data(added Pclass), y_train data is survived.
X_submit= X_submit.loc[:,[ 'Sex', 'SibSp',
       'Parch', 'Embarked', 'AgeFilled', 'Pclass', 'Fare', 'Married'
       ]]
X_submit.columns

### fourth manipulation, create dummy variable
X_submit_dummy =  pd.get_dummies(X_submit,columns=['Sex',"Embarked", 'Pclass'],drop_first=False) #k-1
X_submit_dummy.head()
#then, we use the logistic regression above to create y_submit. This will be our submission
y_submit = logreg.predict(X_submit_dummy)
#here is our prediction
y_submit
#let's create a blank Dataframe with X_submit index for our submission
submission = pd.DataFrame()

#now put passengerid into the DF
submission["PassengerId"]  =  X_submit_org["PassengerId"]

#...and put the prediction into the DF
submission["Survived"] = y_submit
submission.head() #check
#final step is to output the submission as csv so that we can submit
submission.to_csv("titanic_submission.csv", index = False) #do not write the index out

#then we will submit this to see how well we do!
#can you perform other manipulation and (possibly) get a high score?



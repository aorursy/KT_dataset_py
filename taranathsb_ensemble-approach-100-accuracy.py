import warnings

warnings.filterwarnings('ignore')



import pandas                        as pd

import numpy                         as np

import matplotlib.pyplot             as plt

%matplotlib inline

import seaborn                       as sns

from sklearn import preprocessing





from sklearn.preprocessing           import normalize,StandardScaler,label

from sklearn.model_selection         import train_test_split,GridSearchCV,RandomizedSearchCV

from sklearn.metrics                 import *



# ML Libraries



from sklearn.ensemble                import RandomForestClassifier

from sklearn.decomposition           import PCA

from xgboost                         import XGBClassifier

from sklearn.linear_model            import LogisticRegression

from lightgbm                        import LGBMClassifier

from sklearn.tree                    import DecisionTreeClassifier

from sklearn.neighbors               import KNeighborsClassifier
df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
df.head()
# Visualize the Data and check for class balance

for i,col in enumerate(df):

    plt.figure(i)

    sns.countplot(x=df[col])
# Convert to Numberical Values

labelEncoder = preprocessing.LabelEncoder()

for col in df.columns:

    df[col] = labelEncoder.fit_transform(df[col])
# Check if any Null Values

df.isnull().any()
#Create Data for training

Y = df['class']

X = df.iloc[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=7,shuffle=True)
#listing out the different ML Algorithms

models = []

models.append(('Random Forest with Entropy', RandomForestClassifier(criterion= "entropy", random_state= 111)))

models.append(('Random Forest with gini', RandomForestClassifier(criterion= "gini", random_state= 111)))

models.append(('XGBoost', XGBClassifier()))

models.append(('LGBM', LGBMClassifier()))

models.append(('DecisionTree with entropy', DecisionTreeClassifier(criterion= "entropy", random_state= 101)))

models.append(('DecisionTree with gini', DecisionTreeClassifier(criterion= "gini", random_state= 101)))

models.append(('Logistic Regression', LogisticRegression(random_state= 7)))

models.append(('KNN', KNeighborsClassifier(n_neighbors=10)))
#Predefined ROC Function

def ROCcurve(fpr, tpr):

    plt.figure()

    lw = 2

    plt.plot(fpr, tpr, color='darkorange',

             lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    return (plt.show())
# Checking with Multiple accuracy metrics and check for Overfitting

def allmodels():

    model_list = pd.DataFrame(columns=("Model","Accuracy","F1Score","AUC"))

    rownumber = 0

    for name, model in models:

        classifier = model

        classifier.fit(X_train, y_train)

        # prediction

        Y_predict = classifier.predict(X_test)

        #ROCcurve(fpr, tpr)

        model_list.loc[rownumber,"Model"]= name

        model_list.loc[rownumber,"Accuracy"] = round(((accuracy_score(y_test,Y_predict))*100))

        model_list.loc[rownumber,"F1Score"]= round((f1_score(y_test,Y_predict)),2)

        model_list.loc[rownumber,"AUC"]= round((roc_auc_score(y_test,Y_predict)),2)

        Y_pt = classifier.predict(X_train)

        model_list.loc[rownumber,"Accuracy_Train"] = round(((accuracy_score(y_train,Y_pt))*100))

        model_list.loc[rownumber,"F1Score_Train"]= round((f1_score(y_train,Y_pt)),2)

        model_list.loc[rownumber,"AUC_Train"]= round((roc_auc_score(y_train,Y_pt)),2)

        rownumber += 1

    return (model_list.sort_values(by="AUC",ascending=False))
#Check for any overfitting

print (allmodels())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../input/train.csv')
y = df['Survived']

X = df.drop(['Survived','PassengerId','Ticket','Name'],axis = 1)

X['Age'].fillna(X['Age'].mean(), inplace = True)

X['Cabin'] = X['Cabin'].isnull().astype('int')

enc = LabelEncoder()

X['Embarked'].fillna(method = 'pad',inplace = True)
X['Sex'] = enc.fit_transform(X['Sex'])
X['Embarked'] = enc.fit_transform(X['Embarked'])

print(X.head())

def plot_curve(clf,title):
    
    train_sizes,train_scores,test_scores = learning_curve(clf,X,y,random_state = 42,cv = 5)

    plt.figure()
    plt.title(title)
    
    ylim = (0.7, 1.01)
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1,
                color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
        label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
        label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

plot_curve(LogisticRegression(),'Learning Curve of Logistic Regression')
plot_curve(RandomForestClassifier(),'Learning Curve of Random Forest')
plot_curve(DecisionTreeClassifier(),'Learning Curve of Decision Tree')

# In this scenario random forest algorithm is doing quit good.But after doing some hyperparameter 
# tuning or some other statistical operations may be some other algorithm may also perform well or 
# simply outperform random forest. 
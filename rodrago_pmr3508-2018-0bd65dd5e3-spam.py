import pandas as pd
import matplotlib.pyplot as plt 
train_data = pd.read_csv('../input/spamdata/train_data.csv', 
                         engine='python', 
                         sep=r'\s*,\s*')
train_data.head()
train_data.describe()
correlation = train_data.corr()
corr_ham = correlation["ham"].abs()
corr_ham = corr_ham.drop("Id")
corr_ham.sort_values().plot()
print('Correlation mean value: ' + str(corr_ham.mean()))
print('Correlation median value: ' + str(corr_ham.median()))
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score
import numpy as np
# Instantianting cases
corr_thrs = [0,0.2,0.3]
cases = ['First Case','Second Case','Third Case']

# Instantiating Naive Bayes classifiers
classifiers = [BernoulliNB(), GaussianNB(), MultinomialNB()]
classifiers_name = ['BernoulliNB', 'GaussianNB', 'MultinomialNB']


for i in range(3):
    print(cases[i])
    # Feature selection
    valid_features = corr_ham[abs(corr_ham) > corr_thrs[i]]
    valid_features_list = list(valid_features.index.drop("ham"))
    
     # Instantiating X and Y
    X_train = train_data[valid_features_list]
    Y_train = train_data.ham    
    
    for j in range (3):
        # Evaluate precision and f3 score for each case
        classifiers[j].fit(X_train, Y_train)
        score = cross_val_score(classifiers[j], X_train, Y_train, cv=10)
        precision = np.mean(score)
        print('    '+classifiers_name[j])
        print('    Precision: ', precision)
        prediction = classifiers[j].predict(X_train)
        f3 = fbeta_score(Y_train, prediction, 3)
        print('    F3: ', f3, '\n')
    
   

# Feature selection
features = corr_ham[abs(corr_ham) > 0]
features_list = list(features.index.drop("ham"))

 # Instantiating X and Y
X_train = train_data[features_list]
Y_train = train_data.ham    
    
nb = BernoulliNB().fit(X_train, Y_train)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train, Y_train)

scores_knn = cross_val_score(knn, X_train, Y_train, cv=10)
scores_nb = cross_val_score(nb, X_train, Y_train, cv=10)
print('Naive Bayes Mean Score: ' + str(scores_nb.mean()))
print('KNN Mean Score: ' + str(scores_knn.mean()))
test_data = pd.read_csv('../input/spamdata/test_features.csv', 
                         engine='python', 
                         sep=r'\s*,\s*')
X_test = test_data[features_list]
prediction = nb.predict(X_test)
submission = pd.DataFrame({"id":test_data.Id, "ham":prediction})
submission.to_csv("submission.csv", index=False)
import pandas as pd
import numpy as np

# sklearn classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
df        = pd.read_csv("../input/StudentsPerformance.csv", header=0)
df_course = pd.read_csv("../input/StudentsPerformance.csv", header=0, usecols=['lunch'])
df_scores = pd.read_csv("../input/StudentsPerformance.csv", header=0, usecols=['math score', 'reading score', 'writing score'])
df.head(5)
df.describe()
# Transform labels to binary (i.e. 1s and 0s)
LEncoder = LabelEncoder()
LEncoder.fit(df_course)
T_Labels = LEncoder.transform(df_course)

# What we are trying to guess
math_train    = df_scores.iloc[0:800, 0].values.reshape(-1,1)
reading_train = df_scores.iloc[0:800, 1].values.reshape(-1,1)
writing_train = df_scores.iloc[0:800, 2].values.reshape(-1,1)

math_test    = df_scores.iloc[800:1000, 0].values.reshape(-1,1)
reading_test = df_scores.iloc[800:1000, 1].values.reshape(-1,1)
writing_test = df_scores.iloc[800:1000, 2].values.reshape(-1,1)

X_Train = T_Labels[0:800].reshape(-1,1)
X_Test  = T_Labels[800:1000].reshape(-1,1)
Gaussian_Classifier = GaussianNB()
Gaussian_Classifier.fit(math_train, X_Train)
GaussianNB(priors=None, var_smoothing=1e-09)

RandomForest = RandomForestClassifier(n_estimators=10, random_state=3)
RandomForest = RandomForest.fit(math_train, X_Train)

Logistic_Regression = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=123)
Logistic_Regression.fit(math_test, X_Test)

MLP = MLPClassifier(hidden_layer_sizes=(300, 200), random_state=123)
MLP.fit(math_train, X_Train)

r_list = RandomForest.predict_proba(math_test)
g_list = Gaussian_Classifier.predict_proba(math_test)
l_list = Logistic_Regression.predict_proba(math_test)
m_list = MLP.predict_proba(math_test)

print("Random Forest prediction accuracy:", end='')
print(RandomForest.score(math_test, X_Test))
print("Gaussian prediction accuracy:", end='')
print(Gaussian_Classifier.score(math_test, X_Test))
print("Logistic Regression prediction accuracy:", end='')
print(Logistic_Regression.score(math_test, X_Test))
print("Multilayer Perceptron prediction accuracy:", end='')
print(MLP.score(math_test, X_Test))
predictions = []
for x in range(0, len(g_list)):
    if ((r_list[x][0] + g_list[x][0] + l_list[x][0] + m_list[x][0]) / 4) > 0.5:
        predictions.append(0)
    else:
        predictions.append(1)
        
num_right = 0

for x in range(0, len(predictions)):
    if predictions[x] == X_Test[x][0]:
        num_right += 1

print("Ensemble prediction accuracy:", end='')
print(float(num_right) / 200)
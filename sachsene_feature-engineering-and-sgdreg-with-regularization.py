import pandas as pd

pd.options.display.max_colwidth = 80



import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVC # SVM model with kernels

from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error





import warnings

warnings.filterwarnings('ignore')
student_por = pd.read_csv('/kaggle/input/student-performance-data-set/student-por.csv')

student_por.head()
student_por.describe()
# check missing values in variables



student_por.isnull().sum()
student_por.isnull().any()
student_por.info()
copied = student_por.copy()



mean = 5.7

max_min = 75



def mean_normalization(x):

    return((x-mean)/max_min)



copied['absences'] = copied['absences'].apply(mean_normalization)

copied['health'] = copied['health'].apply(mean_normalization)



corr_matrix = copied.corr()



corr_matrix["absences"].sort_values(ascending=False)
corr_matrix = student_por.corr()



corr_matrix["absences"].sort_values(ascending=False)
corr_matrix["G3"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



# I don't take G2 and G1 into account, because they are an obvious choice

attributes = ["G3", "studytime", "Fedu", "failures", "Dalc", "Walc"] 



scatter_matrix(student_por[attributes], figsize=(16, 12))
import seaborn as sns



corr_matrix = student_por.corr()



plt.figure(figsize=(20,20))

sns.heatmap(corr_matrix, annot=True, cmap="Blues")

plt.title('Correlation Heatmap', fontsize=20)
#comparing sex with G3

sns.boxplot(x="sex", y="G3", data=student_por)
#comparing school with G3

sns.boxplot(x="school", y="G3", data=student_por)
#comparing adress with G3

sns.boxplot(x="address", y="G3", data=student_por)
#comparing parent's jobs with G3

sns.boxplot(x="Mjob", y="G3", data=student_por)

sns.boxplot(x="Fjob", y="G3", data=student_por)
#comparing famsize with G3

sns.boxplot(x="famsize", y="G3", data=student_por)
#comparing Pstatus with G3

sns.boxplot(x="Pstatus", y="G3", data=student_por)
#comparing reason with G3

sns.boxplot(x="reason", y="G3", data=student_por)
#comparing guardian with G3

sns.boxplot(x="guardian", y="G3", data=student_por)
#comparing schoolsup with G3

sns.boxplot(x="schoolsup", y="G3", data=student_por)
#comparing famsup with G3

sns.boxplot(x="famsup", y="G3", data=student_por)
#comparing paid with G3

sns.boxplot(x="paid", y="G3", data=student_por)
#comparing activities with G3

sns.boxplot(x="activities", y="G3", data=student_por)
#comparing nursery with G3

sns.boxplot(x="nursery", y="G3", data=student_por)
#comparing higher with G3

sns.boxplot(x="higher", y="G3", data=student_por)
#comparing internet with G3

sns.boxplot(x="internet", y="G3", data=student_por)
#comparing romantic with G3

sns.boxplot(x="romantic", y="G3", data=student_por)
# making dataframe I'm gonna work with + target G3



features_chosen = ['studytime', 'failures', 'Dalc', 'Walc', 'traveltime', 'freetime',  'Medu', 'Fedu', 

                   'sex', 'school', 'address', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 

                   'higher', 'internet', 'G1', 'G2', 'G3']



student_reduced = student_por[features_chosen].copy()



student_reduced
student_reduced.hist(bins=20, figsize=(20,15))

plt.show()
features_cat = ['sex','school','address','Mjob','Fjob','reason','schoolsup','guardian','higher','internet']



student_reduced_cat = pd.get_dummies(student_reduced, columns = features_cat)

student_reduced_cat
student_reduced_cat.columns
X = np.array(student_reduced_cat.drop(['G3'],1))

y = np.array(student_reduced_cat['G3'])  
scaler = StandardScaler()



X = scaler.fit_transform(X)
X.shape
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=42)
X_train.shape, X_test.shape
from sklearn.linear_model import SGDRegressor



sgd_reg = SGDRegressor(penalty="l2") # specifying Ridge Regression



sgd_reg.fit(X_train, y_train)
accuracy=sgd_reg.score(X_test,y_test)  

accuracy
def plot_learning_curves(model, X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) 

    train_errors, val_errors = [], []

    

    for m in range(1, len(X_train)):

        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])

        y_val_predict = model.predict(X_val) 

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict)) 

        val_errors.append(mean_squared_error(y_val, y_val_predict))

        

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train") 

    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
sgd_reg_curves = SGDRegressor(penalty='l2') 



plot_learning_curves(sgd_reg_curves, X, y)
scores = cross_val_score(sgd_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10) 



sgd_reg_scores = np.sqrt(-scores)
sgd_reg_scores
def display_scores(scores):

    print('Scores:', scores)

    print('Std.  :', scores.std())

    print('Mean  :', scores.mean())

    

display_scores(sgd_reg_scores)
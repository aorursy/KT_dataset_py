
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import itertools
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
#from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
#import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
# في كل مرة   print لاستعراض كل مخرجات التشغيل في نفس الخلية من غير الحاجه الى استخدام أمر  
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# قراءة البیانات 
data=pd.read_csv("../input/wa-fnusec-telcocustomerchurn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# عرض اول 10 صفوف
data.head(10)


# عرض آخر 10 صفوف
data.tail(10)
# عرض ابعاد قاعدة البیانات ( عدد الصفوف و الأعمدة )
data.shape
data.describe()
data.info()
# lowercase تحويل جميع الحروف الى
data.columns = map(str.lower, data.columns)
data.columns
# customerid غير مفيد لذا من الأفضل حذفه
data.drop(['customerid'], axis=1, inplace=True)
#استعراض القيم الفريده لهذا العمود
data['totalcharges'].unique()

#total charges في عمود  nan "" استبدال القيم الفارغه 
data['totalcharges'] = data["totalcharges"].replace(" ",np.nan)
data.info()
# فحص البیانات للتأكد هل فیها اي قیم nulls
data.isna().sum()

data['totalcharges'].isna().sum()
print((data['totalcharges'].isna().sum()/len(data))*100) # حساب النسبة المئويه للقيم المفقودة

#حذف القيم المفقوده وهي فقط 0.15% 
data.dropna(inplace=True)
# فحص البیانات مرة أخرى للتأكد هل فیها اي قیم nulls
data.isna().sum()
#تحويل العمود الى نوع رقمي 
data["totalcharges"] = data["totalcharges"].astype(float)
data.info()
# استعراض محتوى الأعمده
data['onlinesecurity'].value_counts()
data['onlinebackup'].value_counts()
data['deviceprotection'].value_counts()
data['techsupport'].value_counts()
data['streamingtv'].value_counts()
data['streamingmovies'].value_counts()
#استبدال 'No internet service' ب 'No' للأعمدة التالية
# استبدال 
replace_cols = [ 'onlinesecurity', 'onlinebackup', 'deviceprotection',
                'techsupport','streamingtv', 'streamingmovies']
for i in replace_cols : 
    data[i]  = data[i].replace({'No internet service' : 'No'})
# فحص محتوى الأعمدة من جديد
data['onlinesecurity'].unique()
data['onlinesecurity'].value_counts()
data['onlinebackup'].value_counts()
data['deviceprotection'].value_counts()
data['techsupport'].value_counts()
data['streamingtv'].value_counts()
data['streamingmovies'].value_counts()
# استبدال ارقام صفر و واحد ب yes , no من اجل أن يكون مثل باقي الاعمده الفئويه
data['seniorcitizen']=data['seniorcitizen'].replace({1:'Yes',0:"No"})
data['seniorcitizen'].value_counts()
#   'zeo' و 'one'ب  yes ,no معالجه العمود الهدف وهو المخرج الذي يصنف اما صفر أو واحد ولذا سيتم استبدال 
data['churn'] = data['churn'].apply(lambda x: x.strip().replace("Yes", "1").replace("No", "0"))
data['churn'] = data['churn'].astype('int')
#### عمل labelencoder للفیتشرز ال categorical
df=pd.get_dummies(data, drop_first=True)
df.info()
df.head()
from sklearn.model_selection import train_test_split

# تعريف y
y=df['churn'].copy()

# تعريف X

X = df.drop(columns=['churn'], axis=1)



#تقسيم  البيانات الى تدريب واختبار
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,
                                                 random_state=42)

# بناء   logistic regression كموديل صفري باستخدام statsmodels

import statsmodels.api as sm
# تعريف y
y=df['churn'].copy()

# تعريف X

X = df.drop(columns=['churn'], axis=1)


# تعريف العنصر الثابت وهو الحد القاطع على محور واي عندما اكس تساوي صفر
X = sm.add_constant(X)

# Fit model
logit_model = sm.Logit(y, X)

# نتائج النمذجة
result = logit_model.fit()
result.summary()
#حذف الاأعمده ذات قيمة  p-value >0.05 
# لكونها فيتشر غير مفيدة بسبب قيمة ال المرتفعه higher p-values

df.drop(['monthlycharges','gender_Male','partner_Yes','dependents_Yes','onlinesecurity_Yes','onlinebackup_Yes',
 'deviceprotection_Yes','techsupport_Yes','streamingtv_Yes','streamingmovies_Yes','paymentmethod_Credit card (automatic)',
 'paymentmethod_Mailed check'],axis=1, inplace=True)
display(df.columns)
X=df.drop(columns=['churn'], axis=1) ### اعادة تعريف X بعد حذف الفيشر غير  المهمة 
X
df.info()
#scale البيانات الرقمية 
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
# Scale بيانات التدريب والاختبار
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
import itertools
from sklearn.metrics import confusion_matrix 

### تعريف رسم دالة الكونفيوجين ماتريكس
def plot_confusion_matrix(y_true, y_preds):
    # Print confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_preds)
    # Create the basic matrix
    plt.imshow(cnf_matrix,  cmap=plt.cm.Blues)
    # Add title and axis labels
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Add appropriate axis scales
    class_names = set(y) # Get class labels to add to matrix
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)
    # Add labels to each cell
    thresh = cnf_matrix.max() / 2. # Used for text coloring below
    # Here we iterate through the confusion matrix and append labels to our visualization
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],
                     horizontalalignment='center',
                     color='white' if cnf_matrix[i, j] > thresh else 'black')
    # Add a legend
    plt.colorbar();
    plt.show();
def metrics(model_name, y_train, y_test, y_hat_train, y_hat_test):
    '''Print out the evaluation metrics for a given models predictions'''
    print(f'Model: {model_name}', )
    print('-'*60)
    plot_confusion_matrix(y_test,y_hat_test)
    print(f'test accuracy: {accuracy_score(y_test, y_hat_test)}')
    print(f'train accuracy: {accuracy_score(y_train, y_hat_train)}')
    print('-'*60)
    print('-'*60)
    print('Confusion Matrix:\n', pd.crosstab(y_test, y_hat_test, rownames=['Actual'], colnames=['Predicted'],margins = True))
    print('\ntest report:\n' + classification_report(y_test, y_hat_test))
    print('~'*60)
    print('\ntrain report:\n' + classification_report(y_train, y_hat_train))
    print('-'*60)
## Define X, y and split data into training and testing
from sklearn.model_selection import train_test_split

# Split data into X and y
y=df['churn'].copy()

# Define X

X = df.drop(columns=['churn'], axis=1)



#importing train_test_split
from sklearn.model_selection import train_test_split
# Split the data into a training and a test set and set stratify=y to help with imbalance data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,
                                                 random_state=42)



## logistic regression with sickit learn before smote
logreg = LogisticRegression()
base_log = logreg.fit(X_train, y_train)
base_log
#predictions
y_hat_train=base_log.predict(X_train)
y_hat_test = base_log.predict(X_test)

# model results

metrics(base_log, y_train, y_test, y_hat_train, y_hat_test)
# To get the  coffients of all the variables of logistic Regression
base_log_cof = pd.Series(base_log.coef_[0], index=X.columns.values)
print(base_log_cof)

base_log_cof.sort_values(inplace=True)
plt.figure(figsize=(15, 6))
plt.xticks(rotation=90)
features=plt.bar(base_log_cof.index,base_log_cof.values)
# نموذج RandomForestClassifier with n_estimators=10
forest_1 = RandomForestClassifier(n_estimators=10)

forest_1.fit(X_train, y_train)

#predictions
y_hat_train=forest_1.predict(X_train)
y_hat_test = forest_1.predict(X_test)

# model results

metrics(forest_1, y_train, y_test, y_hat_train, y_hat_test)
rf_param = RandomForestClassifier()
param_grid = {
     'criterion':['gini','entropy'],
    'max_depth':[2,3,4,5,20],
    'min_samples_split':[5,20,50],
    'min_samples_leaf':[15,20,30],
    'n_estimators': [1,5,10]
}
gs = GridSearchCV(forest_1, param_grid, cv=3, n_jobs=-1)
gs.fit(X_train, y_train)
gs.best_params_
# Instantiate and fit a RandomForestClassifier with n_estimators=100
forest_2 = RandomForestClassifier(n_estimators=10,
                                criterion= 'gini',
                                max_depth= 20,
                                min_samples_leaf= 15,
                                min_samples_split= 50)

forest_2.fit(X_train, y_train)

#predictions
y_hat_train=forest_2.predict(X_train)
y_hat_test = forest_2.predict(X_test)

# model results

metrics(forest_2, y_train, y_test, y_hat_train, y_hat_test)
# To get the feature importance
feature_important=forest_2.feature_importances_
# Plot features importances
imp = pd.Series(data=forest_2.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.style.use('dark_background')
plt.figure(figsize=(10,12))
plt.title("Feature importance of Random Forest model")
ax = sns.barplot(y=imp.index, x=imp.values, palette="RdBu")
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
clf_xgb = XGBClassifier()
param_grid = {
    "learning_rate": [0.1,0.2,0.5,0.9],
    'max_depth': [3, 9, 12],
    'min_child_weight': [10, 18],
    'subsample': [0.3, 0.9],
    'n_estimators': [5, 30, 100, 150],
    'nthread' : [-1],
}
grid_clf = GridSearchCV(clf_xgb, param_grid, scoring='recall', cv=5, n_jobs=1)
grid_clf.fit(X_train, y_train)

best_parameters = grid_clf.best_params_

print("Grid Search found the following optimal parameters: ")
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

y_hat_train = grid_clf.predict(X_train)
y_hat_test= grid_clf.predict(X_test)
metrics(grid_clf, y_train, y_test, y_hat_train, y_hat_test)
train_accuracy=recall_score(y_train, y_hat_train)
test_accuracy = recall_score(y_test, y_hat_test)
print("")
print("Training recall: {:.4}%".format(train_accuracy * 100))
print("Test recall: {:.4}%".format(test_accuracy * 100))
